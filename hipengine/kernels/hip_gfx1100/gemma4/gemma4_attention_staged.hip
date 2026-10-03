// Gemma 4 staged attention (head_dim 256/512) on gfx11.
//
// The strict family (gemma4_attention.hip) keeps one FP32 logit per live key in
// shared memory, so its shared footprint grows with the context and its launch
// refuses above 15,616 keys at head_dim 512 (`gemma4_attention_shared_bytes`).
// This candidate computes the same arithmetic in three kernels over a
// caller-owned global workspace, so its shared footprint is a constant 8 KB
// that does not mention `keys` at all:
//
//   1. `gemma4_attention_staged_score_kernel` writes FP32 logits -- one warp per
//      key, the strict warp-key tree -- and one running maximum per
//      (row, key chunk).
//   2. `gemma4_attention_staged_softmax_kernel` reduces those maxima to the row
//      maximum and runs the strict pass 2: the 256-lane `j mod 256` partition,
//      the 128/64/32 shared tree and the 16..1 shuffle stages, every weight
//      written back over its logit, and the denominator published.
//   3. `gemma4_attention_staged_pv_kernel` runs the strict pass 3 with the
//      weights staged in a bounded shared tile for up to eight query rows that
//      share one KV head.
//
// Nothing here reassociates: the score tree, the denominator partition and the
// numerator chain are the strict kernel's own, so for the same inputs the output
// is bit-identical to `gemma4_attention.hip`'s. That equality is the candidate's
// contract and is asserted against the shipped wrapper on identical device
// buffers in tests/test_gpu_gemma4_attention_staged.py.
//
// The workspace is `tokens * num_heads * keys` FP32 logits (later overwritten
// with the weights), `tokens * num_heads * ceil(keys / 1024)` chunk maxima and
// `tokens * num_heads` denominators. The caller owns it on the launch stream and
// device; `hipengine_gemma4_attention_staged_workspace_bytes` is the one
// definition of its size and the Python launcher sizes the allocation from that
// export.
//
// The launch shape, for whoever measures it: three kernels, three traversals of
// the workspace, a score grid of `tokens * num_heads * ceil(keys / 1024)` CTAs
// and a PV grid of `tokens * num_kv_heads * ceil(GQA / 8)`. Exactness pins each
// output dimension to one ascending-key chain, so the key range cannot be split
// across CTAs -- a property of the contract, not a statement about how the shape
// performs at any given context.
//
// Every one of those grids lives inside a single gfx11 grid dimension (65535
// wide), and the workspace is indexed with int64: `gemma4_staged_validate_launch`
// bounds each raw dimension before any product of it is formed, and a shape past
// a bound is refused with `hipErrorInvalidValue` before a pointer is taken. The
// export below applies the same predicate and returns its 0 sentinel instead.
//
// Two properties this file deliberately does *not* have:
//
//   * It does not implement head_dim values other than 256 and 512, because the
//     warp-key tree is specialised for those two widths (the strict launcher
//     falls back to a block-sum kernel for other widths; this one refuses).
//   * It does not re-derive the mask. `window > 0` is the caller's promise that
//     the mask is zero outside `[row_offset + token - window + 1,
//     row_offset + token]`, exactly as in the strict family; the trim it
//     licenses only drops columns whose weight is zero. The two bounds are
//     computed in saturating int64 and clamped to `[0, keys]`, so an extreme
//     `window` or `row_offset` narrows the walk instead of indexing outside it,
//     and a window that does not intersect the key range leaves the row NaN --
//     the same result an all-masked row has.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

// Every stage runs this block width. The score stage needs 256 so its
// warp-per-key mapping matches the strict kernel's; the softmax stage needs 256
// because its reduction is the strict 256-lane one; the PV stage uses it for
// the dimension walk (head_dim 256 -> one dimension per thread, 512 -> two).
constexpr int kStagedThreads = 256;
// Keys per score CTA. Sets the workspace's chunk-maximum array: one float per
// (row, chunk). A power of two so the partial-max reduction is a short loop.
constexpr int kStagedScoreChunkKeys = 1024;
// Query rows per PV CTA. Rows in one CTA share a KV head, so this is the V
// reuse factor. The accumulator array is `kStagedMaxRows * kDimsPerThread`
// registers, which is what bounds it: 8 rows x 2 dimensions = 16 accumulators.
constexpr int kStagedMaxRows = 8;
// Keys per PV weight tile, resident in shared memory. The PV footprint is
// `kStagedMaxRows * kStagedPvTileKeys` floats and does not depend on `keys`.
constexpr int kStagedPvTileKeys = 256;
constexpr size_t kStagedLdsBudgetBytes = 64 * 1024;
// gfx11 permits 2**31-1 CTAs in X and 65535 in Y/Z. Score uses (rows,
// chunks), PV uses (tokens, num_kv_heads, row_groups). Keep these distinct so
// long-query grids are not rejected by a Y/Z limit they do not use.
constexpr int64_t kStagedMaxGridX = INT32_MAX;
constexpr int64_t kStagedMaxGridDim = 65535;
constexpr int64_t kStagedMaxKeys = kStagedMaxGridDim * kStagedScoreChunkKeys;
// The largest GQA ratio whose PV row groups still fit one grid dimension.
constexpr int64_t kStagedMaxRowsPerHead = kStagedMaxRows * kStagedMaxGridDim;

// A clamped key bound is a column index, so it fits an `int`; and the launch
// limits keep every workspace product inside the int64 index arithmetic the
// stages use and the size_t byte count the host allocates from.
static_assert(kStagedMaxKeys <= INT32_MAX, "a key index must fit in an int");
static_assert(kStagedMaxGridX * kStagedMaxKeys <= INT64_MAX / 8,
              "the workspace layout must stay representable at the launch limits");

// ---------------------------------------------------------------------------
// Arithmetic helpers mirrored from gemma4_attention.hip.
//
// They are copied rather than shared through an include so this candidate does
// not pull the strict file's own `extern "C"` entry points and decode-selection
// state into a second shared object. The copy is only safe because it is
// checked: tests/test_gpu_gemma4_attention_staged.py launches this kernel and
// the shipped strict wrapper on identical buffers and compares the outputs
// bitwise, so a change to either tree that breaks the equality fails there.
// ---------------------------------------------------------------------------

__device__ inline float gemma4_staged_bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t gemma4_staged_float_to_bf16_bits(float value) {
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
__device__ inline scalar_t gemma4_staged_to_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ inline uint16_t gemma4_staged_to_scalar<uint16_t>(float value) {
  return gemma4_staged_float_to_bf16_bits(value);
}

template <typename scalar_t>
__device__ inline float gemma4_staged_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float gemma4_staged_to_float<uint16_t>(uint16_t value) {
  return gemma4_staged_bf16_bits_to_float(value);
}

// Warp-per-key logit for one key, byte for byte the strict kernel's
// `gemma4_warp_key_logit` (see that file for why every stage is ordered the way
// it is: `kLeaves` is head_dim / 32, the first half is uncontracted so a leaf
// rounds once, the second half at 512 is an fmaf into it, and the intra-lane
// stages plus the shuffles reproduce `gemma4_attn_block_sum`'s pair order).
template <typename scalar_t, int kLeaves>
__device__ __forceinline__ float gemma4_staged_warp_key_logit(
    const float* q_s, const scalar_t* k_row, int lane) {
  float p[kLeaves];
  {
#pragma clang fp contract(off)
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int d = lane + 32 * k;
      p[k] = q_s[d] * gemma4_staged_to_float(k_row[d]);
    }
  }
  if (kLeaves == 16) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int d = lane + 32 * (k + 8);
      p[k] = fmaf(q_s[d], gemma4_staged_to_float(k_row[d]), p[k]);
    }
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    p[k] += p[k + 4];
  }
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    p[k] += p[k + 2];
  }
  p[0] += p[1];
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    p[0] += __shfl_down(p[0], offset);
  }
  return __shfl_sync(0xffffffffull, p[0], 0);
}

// Saturating int64 addition and subtraction for the window bounds below.
// `window`, `row_offset` and the row's own position are caller-supplied int64
// values, so a sum or a difference of them can leave the representable range
// before it is clamped -- and a wrapped bound is what turns into a negative
// column index rather than a clamped one.
__device__ __forceinline__ int64_t gemma4_staged_sat_add(int64_t a, int64_t b) {
  int64_t sum = 0;
  if (__builtin_add_overflow(a, b, &sum)) {
    return b > 0 ? INT64_MAX : INT64_MIN;
  }
  return sum;
}

__device__ __forceinline__ int64_t gemma4_staged_sat_sub(int64_t a, int64_t b) {
  int64_t difference = 0;
  if (__builtin_sub_overflow(a, b, &difference)) {
    return b > 0 ? INT64_MIN : INT64_MAX;
  }
  return difference;
}

// The first column this row can keep and the last one it can, both intersected
// with the columns that exist. They mirror the strict `gemma4_attn_first_kept`
// and the strict block kernel's `walk_end`, with two differences that only
// matter at the edges of int64: the arithmetic saturates instead of wrapping,
// and the result is clamped to [0, keys] instead of being narrowed to an int.
// A bound is a column index, so a value outside [0, keys] is meaningless rather
// than merely large, and `keys <= kStagedMaxKeys` is what makes the final cast
// exact. `window <= 0` is the caller's "no promise about the mask" case and
// keeps the full walk.
//
// The two bounds are equal when the window and the key range do not intersect
// (both 0, or both `keys`). Every stage treats that as an empty walk: the score
// stage writes -INFINITY, the softmax stage has no column to exponentiate, the
// denominator is 0, and the PV stage returns 0/0 -- the same NaN an all-masked
// row produces, and no column outside [0, keys) is ever read.
__device__ __forceinline__ int gemma4_staged_first_kept(
    int64_t window, int64_t row_offset, int64_t token, int64_t keys) {
  if (window <= 0) {
    return 0;
  }
  // Subtract the positive window before adding token: own may exceed int64
  // even when own-window is representable (e.g. offset=window=INT64_MAX).
  // Underflow here stays below zero after any launchable token is added.
  const int64_t first = gemma4_staged_sat_add(
      gemma4_staged_sat_add(gemma4_staged_sat_sub(row_offset, window), token), 1);
  if (first <= 0) {
    return 0;
  }
  return first >= keys ? static_cast<int>(keys) : static_cast<int>(first);
}

// The last column this row can keep. `window > 0` promises the mask is zero
// above the row's own position as well as below its window, so every column
// past it has weight zero and dropping it cannot change a result.
__device__ __forceinline__ int gemma4_staged_walk_end(
    int64_t window, int64_t row_offset, int64_t token, int64_t keys) {
  if (window <= 0) {
    return static_cast<int>(keys);
  }
  const int64_t own = gemma4_staged_sat_add(row_offset, token);
  const int64_t last = gemma4_staged_sat_add(own, 1);
  if (last <= 0) {
    return 0;
  }
  return last >= keys ? static_cast<int>(keys) : static_cast<int>(last);
}

// The first column at or after `from` that lane `lane` owns when the block
// strides over columns by `stride`; mirrors `gemma4_attn_stride_first`. Callers
// guard the call with `lane < from`, and `from` is a clamped key bound, so the
// result is a column in [from, from + stride) and cannot overflow.
__device__ __forceinline__ int64_t gemma4_staged_stride_first(
    int64_t from, int64_t lane, int64_t stride) {
  return lane + ((from - lane + stride - 1) / stride) * stride;
}

// The strict 256-lane denominator reduction: shared partials for the strides
// 128/64/32, shuffles for 16..1, element 0 returned to every lane. Mirrors
// `gemma4_decode_tree1`, which is the class kernel's pass 2 and adds the same
// pairs in the same descending order as the block kernel's `block_sum`.
__device__ inline float gemma4_staged_tree1(
    float* partial, float value, unsigned int threads, unsigned int tid) {
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

// ---------------------------------------------------------------------------
// Stage 1: one warp per key.
//
// Grid (tokens * num_heads, key chunks); 256 threads = 8 warps; warp w owns the
// columns w, w + 8, ... of this chunk, and lane 0 writes each logit to the
// workspace. A column outside `[first_kept, walk_end)` or masked out is
// -INFINITY, which is exactly what the strict kernel stores for it, so the
// softmax stage needs no mask of its own: expf(-INFINITY - row_max) is 0.
//
// The chunk maximum is an fmaxf over the eight warps' running maxima. That
// reduction is order-independent for these values (there are no NaNs in the
// logits), so it is the strict row maximum bit for bit.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void gemma4_attention_staged_score_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const uint8_t* __restrict__ keep_mask,
    float* __restrict__ scores,
    float* __restrict__ chunk_max,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t keys,
    int64_t window,
    int64_t row_offset,
    int64_t chunks) {
  extern __shared__ float shared[];
  float* q_s = shared;
  float* max_s = q_s + head_dim;

  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t chunk = static_cast<int64_t>(blockIdx.y);
  const int64_t token = row / num_heads;
  const int64_t head = row - token * num_heads;
  const int64_t kv_head = head / (num_heads / num_kv_heads);

  const int64_t j0 = chunk * kStagedScoreChunkKeys;
  const int64_t j1 =
      (j0 + kStagedScoreChunkKeys < keys) ? j0 + kStagedScoreChunkKeys : keys;
  const int first_kept = gemma4_staged_first_kept(window, row_offset, token, keys);
  const int walk_end = gemma4_staged_walk_end(window, row_offset, token, keys);

  // A chunk with no column in [first_kept, walk_end) contributes nothing: every
  // column it would write is -INFINITY, so its logits and its maximum are both
  // -INFINITY, and no later stage reads a column outside that range. Skipping
  // it keeps a sliding layer from computing the columns behind its window. The
  // branch is uniform across the CTA, so it is safe before any barrier.
  if (j1 <= first_kept || j0 >= walk_end) {
    if (threadIdx.x == 0) {
      chunk_max[row * chunks + chunk] = -INFINITY;
    }
    return;
  }

  const scalar_t* q_row = query + row * head_dim;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    q_s[d] = gemma4_staged_to_float(q_row[d]);
  }
  __syncthreads();

  const uint8_t* mask_row = keep_mask + token * keys;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int64_t k_stride = num_kv_heads * head_dim;
  float* scores_row = scores + row * keys;

  int64_t j = j0 + warp;
  const int64_t j_load = j < keys ? j : 0;
  const scalar_t* k_row = key + (j_load * num_kv_heads + kv_head) * head_dim;
  float running_max = -INFINITY;
  for (; j < j1; j += 8) {
    float logit = -INFINITY;
    if (j >= first_kept && j < walk_end && mask_row[j] != 0) {
      logit = (head_dim == 512)
                  ? gemma4_staged_warp_key_logit<scalar_t, 16>(q_s, k_row, lane)
                  : gemma4_staged_warp_key_logit<scalar_t, 8>(q_s, k_row, lane);
      logit *= scale;
    }
    if (lane == 0) {
      scores_row[j] = logit;
    }
    running_max = fmaxf(running_max, logit);
    k_row += 8 * k_stride;
  }
  if (lane == 0) {
    max_s[warp] = running_max;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float chunk_running_max = -INFINITY;
    for (int w = 0; w < (kStagedThreads >> 5); ++w) {
      chunk_running_max = fmaxf(chunk_running_max, max_s[w]);
    }
    chunk_max[row * chunks + chunk] = chunk_running_max;
  }
}

// ---------------------------------------------------------------------------
// Stage 2: the strict softmax, one CTA per (token, head) row, 256 threads.
//
// No mask and no geometry: every column in [first_kept, walk_end) is read from
// the workspace, and a masked column is -INFINITY there, so this is the strict
// pass 2 with its input in global memory instead of shared memory.
// ---------------------------------------------------------------------------
__global__ void gemma4_attention_staged_softmax_kernel(
    float* __restrict__ scores,
    const float* __restrict__ chunk_max,
    float* __restrict__ denominator,
    int64_t num_heads,
    int64_t keys,
    int64_t window,
    int64_t row_offset,
    int64_t chunks) {
  extern __shared__ float shared[];
  float* partial = shared;
  float* row_max_s = partial + kStagedThreads;

  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t token = row / num_heads;
  const unsigned int tid = threadIdx.x;

  if (tid == 0) {
    float row_max = -INFINITY;
    for (int64_t c = 0; c < chunks; ++c) {
      row_max = fmaxf(row_max, chunk_max[row * chunks + c]);
    }
    row_max_s[0] = row_max;
  }
  __syncthreads();
  const float row_max = row_max_s[0];

  const int first_kept = gemma4_staged_first_kept(window, row_offset, token, keys);
  const int walk_end = gemma4_staged_walk_end(window, row_offset, token, keys);
  float* scores_row = scores + row * keys;

  float local_sum = 0.0f;
  int64_t j = tid;
  if (j < first_kept) {
    j = gemma4_staged_stride_first(first_kept, tid, kStagedThreads);
  }
  for (; j < walk_end; j += kStagedThreads) {
    const float weight = expf(scores_row[j] - row_max);
    local_sum += weight;
    scores_row[j] = weight;
  }
  const float row_denominator =
      gemma4_staged_tree1(partial, local_sum, kStagedThreads, tid);
  if (tid == 0) {
    denominator[row] = row_denominator;
  }
}

// ---------------------------------------------------------------------------
// Stage 3: the strict weighted sum over a bounded weight tile.
//
// Grid (tokens, num_kv_heads, row groups); 256 threads; thread `tid` owns the
// output dimensions `tid` and `tid + 256` (head_dim 256 leaves the second slot
// idle). The rows in one group share a KV head -- row `r` of the group is query
// head `kv_head * GQA + r` -- so the V value a thread loads for a key is the
// value all `rows` of the group multiply, and the global V traffic is divided by
// the group size compared with one row per CTA.
//
// Each (row, dimension) accumulator is one ascending-key FP32 FMA chain, and the
// tile loop adds tiles in ascending key order, so the chain is the strict
// kernel's chain element for element. Zero weights are skipped, which is exact:
// the strict kernel skips them too, and adding 0.0f to a sum of
// non-negative-magnitude terms cannot change it.
// ---------------------------------------------------------------------------
template <typename scalar_t, int kDimsPerThread>
__global__ void gemma4_attention_staged_pv_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ denominator,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t keys,
    int64_t window,
    int64_t row_offset) {
  extern __shared__ float shared[];
  float* w_s = shared;

  const int64_t token = static_cast<int64_t>(blockIdx.x);
  const int64_t kv_head = static_cast<int64_t>(blockIdx.y);
  const int64_t group = static_cast<int64_t>(blockIdx.z);
  const int64_t rows_per_head = num_heads / num_kv_heads;
  // The workspace is indexed by (token, head) row, not by head alone: row `r`
  // of this group is query head `kv_head * rows_per_head + r` of this token.
  const int64_t row_base = token * num_heads + kv_head * rows_per_head +
                           group * kStagedMaxRows;
  const int64_t rows = (rows_per_head - group * kStagedMaxRows < kStagedMaxRows)
                           ? rows_per_head - group * kStagedMaxRows
                           : kStagedMaxRows;

  float acc[kStagedMaxRows][kDimsPerThread];
#pragma unroll
  for (int r = 0; r < kStagedMaxRows; ++r) {
#pragma unroll
    for (int t = 0; t < kDimsPerThread; ++t) {
      acc[r][t] = 0.0f;
    }
  }

  const int first_kept = gemma4_staged_first_kept(window, row_offset, token, keys);
  const int walk_end = gemma4_staged_walk_end(window, row_offset, token, keys);
  const int64_t v_stride = num_kv_heads * head_dim;
  const scalar_t* v_base = value + kv_head * head_dim;
  const unsigned int tid = threadIdx.x;

  for (int64_t j0 = first_kept; j0 < walk_end; j0 += kStagedPvTileKeys) {
    const int64_t j1 =
        (j0 + kStagedPvTileKeys < walk_end) ? j0 + kStagedPvTileKeys : walk_end;
    const int tile = static_cast<int>(j1 - j0);
    for (int64_t r = 0; r < rows; ++r) {
      const float* src = weights + (row_base + r) * keys + j0;
      float* dst = w_s + r * kStagedPvTileKeys;
      for (int k = static_cast<int>(tid); k < tile; k += kStagedThreads) {
        dst[k] = src[k];
      }
    }
    __syncthreads();

    for (int k = 0; k < tile; ++k) {
      const int64_t j = j0 + k;
      float v[kDimsPerThread];
#pragma unroll
      for (int t = 0; t < kDimsPerThread; ++t) {
        const int64_t d = static_cast<int64_t>(tid) + t * kStagedThreads;
        v[t] = (d < head_dim) ? gemma4_staged_to_float(
                                    v_base[j * v_stride + d])
                              : 0.0f;
      }
#pragma unroll
      for (int r = 0; r < kStagedMaxRows; ++r) {
        if (r >= rows) {
          continue;
        }
        const float weight = w_s[r * kStagedPvTileKeys + k];
        if (weight == 0.0f) {
          continue;
        }
#pragma unroll
        for (int t = 0; t < kDimsPerThread; ++t) {
          acc[r][t] += weight * v[t];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < kStagedMaxRows; ++r) {
    if (r >= rows) {
      continue;
    }
    const float row_denominator = denominator[row_base + r];
#pragma unroll
    for (int t = 0; t < kDimsPerThread; ++t) {
      const int64_t d = static_cast<int64_t>(tid) + t * kStagedThreads;
      if (d < head_dim) {
        out[(row_base + r) * head_dim + d] =
            gemma4_staged_to_scalar<scalar_t>(acc[r][t] / row_denominator);
      }
    }
  }
}

// Shared footprint of each stage, in bytes, for a given head_dim. Constant in
// `keys`: the score stage holds the query row and eight warp maxima, the
// softmax stage holds the 256-lane tree plus the broadcast maximum, and the PV
// stage holds one weight tile per row.
size_t gemma4_staged_score_shared_bytes(int64_t head_dim) {
  return static_cast<size_t>(head_dim + (kStagedThreads >> 5)) * sizeof(float);
}

size_t gemma4_staged_softmax_shared_bytes() {
  return static_cast<size_t>(kStagedThreads + 1) * sizeof(float);
}

size_t gemma4_staged_pv_shared_bytes() {
  return static_cast<size_t>(kStagedMaxRows) * kStagedPvTileKeys * sizeof(float);
}

int64_t gemma4_staged_chunks(int64_t keys) {
  // A quotient and a remainder rather than `keys + chunk - 1`: the latter
  // overflows for a key count near INT64_MAX, and the export is a public entry
  // point that has to answer for any argument.
  return keys / kStagedScoreChunkKeys +
         (keys % kStagedScoreChunkKeys != 0 ? 1 : 0);
}

// Workspace sizing knows rows and score chunks, not PV Y/Z dimensions.
// Bound raw factors before multiplication; then rows fit X and every score,
// mask, Q/output and KV offset and workspace byte count fits signed int64.
// The full launcher validates PV Y/Z separately.
bool gemma4_staged_workspace_shape_ok(int64_t tokens, int64_t num_heads,
                                      int64_t keys) {
  if (tokens <= 0 || num_heads <= 0 || keys <= 0) {
    return false;
  }
  if (tokens > kStagedMaxGridX || num_heads > kStagedMaxGridX) {
    return false;
  }
  if (gemma4_staged_chunks(keys) > kStagedMaxGridDim) {
    return false;
  }
  return tokens * num_heads <= kStagedMaxGridX;
}

// 0 when this shape can be launched, or the HIP error code naming why it cannot.
// The raw dimensions are checked before any product of them is formed, and the
// score grid's row count is checked before the PV grid is built from it.
int gemma4_staged_validate_launch(int64_t tokens, int64_t keys, int64_t num_heads,
                                  int64_t num_kv_heads, int64_t head_dim) {
  if (tokens <= 0 || keys <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
      head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (num_heads % num_kv_heads != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  // The warp-key tree is specialised for these two widths.
  if (head_dim != 256 && head_dim != 512) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (num_kv_heads > kStagedMaxGridDim) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (!gemma4_staged_workspace_shape_ok(tokens, num_heads, keys)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (num_heads / num_kv_heads > kStagedMaxRowsPerHead) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return 0;
}

template <typename scalar_t, int kDimsPerThread>
int launch_gemma4_attention_staged_pv(
    const float* weights,
    const float* denominator,
    const scalar_t* value,
    scalar_t* out,
    int64_t tokens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t keys,
    int64_t window,
    int64_t row_offset,
    hipStream_t stream) {
  const int64_t rows_per_head = num_heads / num_kv_heads;
  const int64_t row_groups = (rows_per_head + kStagedMaxRows - 1) / kStagedMaxRows;
  const size_t shared_bytes = gemma4_staged_pv_shared_bytes();
  const dim3 grid(static_cast<unsigned int>(tokens),
                  static_cast<unsigned int>(num_kv_heads),
                  static_cast<unsigned int>(row_groups));
  gemma4_attention_staged_pv_kernel<scalar_t, kDimsPerThread>
      <<<grid, dim3(kStagedThreads), shared_bytes, stream>>>(
          weights, denominator, value, out, num_heads, num_kv_heads, head_dim,
          keys, window, row_offset);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t>
int launch_gemma4_attention_staged(
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
    int64_t window,
    int64_t row_offset,
    float* workspace) {
  const int shape_error =
      gemma4_staged_validate_launch(tokens, keys, num_heads, num_kv_heads, head_dim);
  if (shape_error != 0) {
    return shape_error;
  }
  if (workspace == nullptr) {
    // Distinguish missing device storage from an invalid shape. Besides giving
    // callers an accurate diagnostic, this permits safe raw-validator tests
    // with null pointers that cannot pass solely on the workspace rejection.
    return static_cast<int>(hipErrorInvalidDevicePointer);
  }
  const int64_t chunks = gemma4_staged_chunks(keys);
  const int64_t rows = tokens * num_heads;
  const size_t score_shared = gemma4_staged_score_shared_bytes(head_dim);
  const size_t softmax_shared = gemma4_staged_softmax_shared_bytes();
  const size_t pv_shared = gemma4_staged_pv_shared_bytes();
  if (score_shared > kStagedLdsBudgetBytes || softmax_shared > kStagedLdsBudgetBytes ||
      pv_shared > kStagedLdsBudgetBytes) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  float* scores = workspace;
  float* chunk_max = scores + rows * keys;
  float* denominator = chunk_max + rows * chunks;

  gemma4_attention_staged_score_kernel<scalar_t>
      <<<dim3(static_cast<unsigned int>(rows), static_cast<unsigned int>(chunks)),
         dim3(kStagedThreads), score_shared, stream>>>(
          query, key, keep_mask, scores, chunk_max, num_heads, num_kv_heads,
          head_dim, scale, keys, window, row_offset, chunks);
  int rc = static_cast<int>(hipGetLastError());
  if (rc != 0) {
    return rc;
  }

  gemma4_attention_staged_softmax_kernel
      <<<dim3(static_cast<unsigned int>(rows)), dim3(kStagedThreads),
         softmax_shared, stream>>>(
          scores, chunk_max, denominator, num_heads, keys, window, row_offset,
          chunks);
  rc = static_cast<int>(hipGetLastError());
  if (rc != 0) {
    return rc;
  }

  return (head_dim == 512)
             ? launch_gemma4_attention_staged_pv<scalar_t, 2>(
                   scores, denominator, value, out, tokens, num_heads,
                   num_kv_heads, head_dim, keys, window, row_offset, stream)
             : launch_gemma4_attention_staged_pv<scalar_t, 1>(
                   scores, denominator, value, out, tokens, num_heads,
                   num_kv_heads, head_dim, keys, window, row_offset, stream);
}

}  // namespace

// Workspace layout, exported so the Python launcher and the kernel share one
// definition of it: logits (later overwritten with the weights), one chunk
// maximum per (row, chunk), one denominator per row.
//
// Returns 0 -- a size no launchable shape has, because a launchable shape has
// rows >= 1 and keys >= 1 -- for a shape this launch cannot serve: a
// non-positive dimension, a dimension past one of the grids, or a layout the
// kernel's arithmetic could not represent. The Python planner raises a named
// reason for the same shapes, so "a size exists" means the same thing on both
// sides of the ABI and neither one can wrap where the other refuses.
extern "C" size_t hipengine_gemma4_attention_staged_workspace_bytes(
    int64_t tokens,
    int64_t num_heads,
    int64_t keys) {
  if (!gemma4_staged_workspace_shape_ok(tokens, num_heads, keys)) {
    return 0;
  }
  const int64_t rows = tokens * num_heads;
  const int64_t chunks = gemma4_staged_chunks(keys);
  return static_cast<size_t>(rows * keys + rows * chunks + rows) * sizeof(float);
}

// Largest shared-memory allocation any of the three stages asks for, so a
// caller can confirm the bounded-layout property (it does not mention `keys`)
// without a device.
extern "C" size_t hipengine_gemma4_attention_staged_lds_bytes(int64_t head_dim) {
  const size_t score = gemma4_staged_score_shared_bytes(head_dim);
  const size_t softmax = gemma4_staged_softmax_shared_bytes();
  const size_t pv = gemma4_staged_pv_shared_bytes();
  size_t largest = score > softmax ? score : softmax;
  return pv > largest ? pv : largest;
}

extern "C" int hipengine_gemma4_attention_staged_bf16(
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
    int64_t window,
    int64_t row_offset,
    float* workspace) {
  return launch_gemma4_attention_staged<uint16_t>(
      query, key, value, keep_mask, out, tokens, num_heads, num_kv_heads,
      head_dim, scale, stream, keys, window, row_offset, workspace);
}

extern "C" int hipengine_gemma4_attention_staged_f32(
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
    int64_t window,
    int64_t row_offset,
    float* workspace) {
  return launch_gemma4_attention_staged<float>(
      query, key, value, keep_mask, out, tokens, num_heads, num_kv_heads,
      head_dim, scale, stream, keys, window, row_offset, workspace);
}
