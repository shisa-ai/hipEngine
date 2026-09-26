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
template <int kDimsPerLane, int kPrefetch, typename scalar_t>
__device__ __forceinline__ void gemma4_decode_pass3(
    const float* __restrict__ logits_s,
    const scalar_t* __restrict__ v_base,
    scalar_t* __restrict__ out_row,
    int64_t v_row_stride,
    int64_t head_dim,
    int64_t keys,
    float denominator,
    unsigned int threads,
    unsigned int tid) {
  const int64_t step = static_cast<int64_t>(threads) * kDimsPerLane;
  for (int64_t d0 = static_cast<int64_t>(tid) * kDimsPerLane; d0 < head_dim; d0 += step) {
    gemma4_pair2<scalar_t> pipe[kPrefetch];
#pragma unroll
    for (int n = 0; n < kPrefetch; ++n) {
      const int64_t jn = n;
      pipe[n] = jn < keys
          ? gemma4_load_v<kDimsPerLane>(v_base + jn * v_row_stride + d0)
          : gemma4_pair2<scalar_t>{static_cast<scalar_t>(0), static_cast<scalar_t>(0)};
    }
    float acc_lo = 0.0f;
    float acc_hi = 0.0f;
    for (int64_t j = 0; j < keys; ++j) {
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
    // Split phase 1 (see gemma4_attention_decode_slice_kernel): when
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
    // Split phase 1: hand pass 3 to gemma4_attention_decode_slice_kernel. The
    // weights written here are the ones pass 3 would have read from LDS, and
    // the header carries the row max the slice weights were built against, so
    // every slice's summands are bit-identical to the incumbent path's and the
    // combine is a plain sum rather than a per-slice rescale.
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
  if (head_dim >= 2 * static_cast<int64_t>(threads)) {
    gemma4_decode_pass3<2, 4>(logits_s, v_base, out_row, v_row_stride, head_dim, keys,
                              denominator, threads, tid);
  } else {
    gemma4_decode_pass3<1, 4>(logits_s, v_base, out_row, v_row_stride, head_dim, keys,
                              denominator, threads, tid);
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

// Split phase 2: one block per (token, head, slice) accumulates the weighted V
// sum over its slice of the key range, reading the weights phase 1 wrote.
//
// This is the only changed arithmetic in the two-phase split: the incumbent
// accumulates one ascending-j chain over every key, and this accumulates
// ascending j within a slice and then sums the slices in ascending order. Every
// summand's weight is the same f32 value the incumbent path uses - there is no
// per-slice max and no second expf - so the perturbation is the association
// order of an f32 sum, not a rescale.
//
// The slicing exists because the kernel's parallelism is structurally low:
// grid = tokens * num_heads puts 16 blocks on a 96-CU GPU, and
// scripts/gemma4_attention_scale_probe.py measures per-head time improving 6.9x
// from 16 to 64 blocks.
template <typename scalar_t, int kDimsPerLane, int kPrefetch>
__global__ void gemma4_attention_decode_slice_kernel(
    const float* __restrict__ weights,
    const scalar_t* __restrict__ value,
    float* __restrict__ slice_out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t keys,
    int64_t slices) {
  const unsigned int threads = blockDim.x;
  const unsigned int tid = threadIdx.x;
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t token = linear / (num_heads * slices);
  const int64_t head = (linear / slices) % num_heads;
  const int64_t slice = linear % slices;
  const int64_t kv_head = head / (num_heads / num_kv_heads);
  const int64_t head_index = token * num_heads + head;

  // Ceil-divided slices, with the last one clamped: the same range phase 1 and
  // the combine must agree on.
  const int64_t slice_len = (keys + slices - 1) / slices;
  const int64_t j_begin = slice * slice_len;
  int64_t j_end = j_begin + slice_len;
  if (j_end > keys) {
    j_end = keys;
  }
  // Empty slices still write zeros below: combine reads every slice, and
  // scratch may contain values from a previous request.

  const float* w_row = weights + head_index * keys;
  const scalar_t* v_base = value + kv_head * head_dim;
  const int64_t v_row_stride = num_kv_heads * head_dim;
  float* out_row = slice_out + (head_index * slices + slice) * head_dim;

  const int64_t step = static_cast<int64_t>(threads) * kDimsPerLane;
  for (int64_t d0 = static_cast<int64_t>(tid) * kDimsPerLane; d0 < head_dim; d0 += step) {
    gemma4_pair2<scalar_t> pipe[kPrefetch];
#pragma unroll
    for (int n = 0; n < kPrefetch; ++n) {
      const int64_t jn = j_begin + n;
      pipe[n] = jn < j_end
          ? gemma4_load_v<kDimsPerLane>(v_base + jn * v_row_stride + d0)
          : gemma4_pair2<scalar_t>{static_cast<scalar_t>(0), static_cast<scalar_t>(0)};
    }
    float acc_lo = 0.0f;
    float acc_hi = 0.0f;
    for (int64_t j = j_begin; j < j_end; ++j) {
      const gemma4_pair2<scalar_t> v_cur = pipe[0];
#pragma unroll
      for (int n = 0; n < kPrefetch - 1; ++n) {
        pipe[n] = pipe[n + 1];
      }
      const int64_t jn = j + kPrefetch;
      pipe[kPrefetch - 1] = jn < j_end
          ? gemma4_load_v<kDimsPerLane>(v_base + jn * v_row_stride + d0)
          : gemma4_pair2<scalar_t>{static_cast<scalar_t>(0), static_cast<scalar_t>(0)};
      const float weight = w_row[j];
      if (weight == 0.0f) {
        continue;
      }
      acc_lo += weight * gemma4_attn_to_float(v_cur.lo);
      if (kDimsPerLane == 2) {
        acc_hi += weight * gemma4_attn_to_float(v_cur.hi);
      }
    }
    out_row[d0] = acc_lo;
    if (kDimsPerLane == 2 && d0 + 1 < head_dim) {
      out_row[d0 + 1] = acc_hi;
    }
  }
}

// Split phase 3: sum the slice partials in ascending slice order and divide by
// phase 1's denominator. One block per (token, head); each lane owns its own d
// stride and reads the same few slice values, so no synchronisation is needed.
template <typename scalar_t>
__global__ void gemma4_attention_decode_combine_kernel(
    const float* __restrict__ slice_out,
    const float* __restrict__ header,
    scalar_t* __restrict__ out,
    int64_t num_heads,
    int64_t head_dim,
    int64_t slices) {
  const int64_t head_index = static_cast<int64_t>(blockIdx.x);
  const float denominator = header[head_index * 2];
  const float* slices_base = slice_out + head_index * slices * head_dim;
  scalar_t* out_row = out + head_index * head_dim;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t s = 0; s < slices; ++s) {
      acc += slices_base[s * head_dim + d];
    }
    out_row[d] = gemma4_attn_to_scalar<scalar_t>(acc / denominator);
  }
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

// Which decode kernel the launcher last selected: 1 = key-class (head_dim 256
// or 512), 0 = block kernel. Introspection only - it changes no behaviour, and
// it is what the routing test asserts instead of inferring the path from
// timings.
inline int& gemma4_decode_selection() {
  static int selection = 0;
  return selection;
}

// Bytes of workspace the split needs for one launch: the weights phase 1 hands
// to phase 2 (tokens * num_heads * keys floats), the (denominator, row max)
// header per (token, head), and the f32 slice partials
// (tokens * num_heads * slices * head_dim floats).
size_t gemma4_decode_split_workspace_bytes(
    int64_t tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t keys,
    int slices) {
  const size_t weights = static_cast<size_t>(tokens) * num_heads * keys;
  const size_t header = static_cast<size_t>(tokens) * num_heads * 2;
  const size_t partials =
      static_cast<size_t>(tokens) * num_heads * slices * head_dim;
  return (weights + header + partials) * sizeof(float);
}

template <typename scalar_t, int kHalves, int kDimsPerLane>
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
  constexpr int kPrefetch = 4;
  const size_t shared_bytes =
      (static_cast<size_t>(keys) + 256 * (kThreads / kGemma4ReduceThreads) +
       kThreads / 32) * sizeof(float);
  if (shared_bytes > 64 * 1024 || workspace == nullptr || slices < 2) {
    return -1;
  }
  float* weights = workspace;
  float* header = weights + static_cast<size_t>(tokens) * num_heads * keys;
  float* partials = header + static_cast<size_t>(tokens) * num_heads * 2;

  const unsigned int head_grid = static_cast<unsigned int>(tokens * num_heads);
  gemma4_attention_decode_class_kernel<scalar_t, kHalves, kKeysPerTile>
      <<<dim3(head_grid), dim3(kThreads), shared_bytes, stream>>>(
          query, key, value, keep_mask, out, num_heads, num_kv_heads, head_dim, scale, tokens,
          keys, weights, header);
  int rc = static_cast<int>(hipGetLastError());
  if (rc != 0) {
    return rc;
  }

  const unsigned int slice_grid =
      static_cast<unsigned int>(tokens * num_heads * slices);
  if (head_dim >= 2 * static_cast<int64_t>(kThreads)) {
    gemma4_attention_decode_slice_kernel<scalar_t, 2, kPrefetch>
        <<<dim3(slice_grid), dim3(kThreads), 0, stream>>>(
            weights, value, partials, num_heads, num_kv_heads, head_dim, keys, slices);
  } else {
    gemma4_attention_decode_slice_kernel<scalar_t, 1, kPrefetch>
        <<<dim3(slice_grid), dim3(kThreads), 0, stream>>>(
            weights, value, partials, num_heads, num_kv_heads, head_dim, keys, slices);
  }
  rc = static_cast<int>(hipGetLastError());
  if (rc != 0) {
    return rc;
  }

  gemma4_attention_decode_combine_kernel<scalar_t>
      <<<dim3(head_grid), dim3(kThreads), 0, stream>>>(
          partials, header, out, num_heads, head_dim, slices);
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
    if (split_slices > 1 && split_workspace != nullptr) {
      const int split_rc = head_dim == 256
          ? gemma4_decode_split_launch<scalar_t, 1, 1>(query, key, value, keep_mask, out, num_heads,
                num_kv_heads, head_dim, scale, tokens, keys, stream, split_workspace,
                split_slices)
          : gemma4_decode_split_launch<scalar_t, 2, 2>(query, key, value, keep_mask, out, num_heads,
                num_kv_heads, head_dim, scale, tokens, keys, stream, split_workspace,
                split_slices);
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

// Returns 1 when the last decode launch used the key-class kernel, 0 when it
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
//
// ``split_workspace`` is the two-phase split's scratch (see
// gemma4_decode_split_workspace_bytes) and ``split_slices`` its slice count;
// passing ``split_slices <= 1`` or a null workspace selects the single-kernel
// path. The caller owns the allocation because it is reused across steps and
// must outlive the in-flight stream.
// Which decode kernel the launcher last selected, exported so a caller can
// confirm the intended path actually ran rather than inferring it from timings:
// 0 = block kernel, 1 = key-class single kernel, 2 = key-class two-phase split.
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
