// Gemma 4 BF16 sliding-window prefill attention on gfx11, on the matrix cores.
//
// The strict `gemma4_attention_prefill_bf16` kernel is one block of 256 threads
// per (token, head) with a warp-level reduction per key. At head_dim 256 each
// lane contributes one multiply-add to a key's logit and the other 31 lanes'
// work is the shuffle tree that combines them, so the kernel's cost is its
// per-key loop machinery rather than its arithmetic: the census that scoped
// this unit reports 276-450 GFLOP/s across its points, and this worktree's
// harness measures 462 GFLOP/s at 4096 tokens, about 1 percent of the part's
// BF16 rate, with a measured 41x scaling against 8x tokens. This kernel removes
// the per-key reduction instead of rescheduling it: a 16x16x16 F16 WMMA tile
// consumes sixteen keys at once, so no per-key cross-lane step exists to pay
// for. It measures 6344.7 GFLOP/s against the strict kernel's 462.0 at 4096
// tokens, 13.73x on the same row; the worklog entry for this unit carries the
// comparison, the tolerance, and what is not claimed.
//
// This is NOT the refuted `gemma4_attention_prefill_tiled_kernel` route. That
// variant kept the same execution unit (one warp per key, a scalar reduction)
// and only changed when logits were materialised, and it measured slower than
// the strict kernel at every size. The difference here is the execution unit.
//
// Arithmetic lineage: the tile structure, the online-softmax K-batches, and the
// WMMA fragment conventions follow `laguna_flash_attention_prefill.hip` (itself
// a port of llama.cpp fattn-mma-f16.cuh) and the INT8 WMMA flash prefill in
// `paged_attn_decode.hip`, which is the in-tree precedent for head_dim 256.
//
// Geometry (Gemma 4 26B-A4B sliding layers: 25 of 30 layers):
//   head_dim 256, 16 query heads, 8 KV heads, GQA ratio 2, window 1024, causal.
//
//   QUERY_ROWS   16   one WMMA tile row count; the strict kernel has no query
//                     tiling at all, which is why M=16 is the whole point.
//   GQA_HEADS     2   the model's GQA ratio, so a tile covers both query heads
//                     that share a KV head and stages each K/V row once.
//   COLUMNS      32   QUERY_ROWS * GQA_HEADS, the WMMA column count.
//   K_BATCH      32   keys per online-softmax batch (two 16-key WMMA tiles).
//   WAVES         2   THREADS 64. One wave owns 16 columns.
//
// Register pressure, not shared memory, is the binding constraint at head_dim
// 256: the accumulator is HEAD_DIM * COLUMNS / THREADS = 128 floats per lane,
// which is the same operating point the INT8 WMMA prefill above chose. The Q
// operand is therefore re-read from LDS per dimension tile rather than held in
// registers -- holding it would cost another 128 registers per lane.
//
// Masking is the caller's keep-mask, not a re-derived causal test. Two
// block-level bounds trim the walk: `first_kept` from the tile's first row and
// `walk_end` from its last. Both are only used when `window > 0`, which is the
// caller's promise that the mask is sliding-causal and therefore zero outside
// that range -- the same promise, and the same bounds, the strict kernel spends
// on its per-row walk. `window == 0` means no promise, so the walk covers
// every column and only the mask decides visibility.
//
// This is a changed-arithmetic candidate, not a strict variant: WMMA
// accumulates each 16-wide dot in the matrix unit's own order, so the logits
// are reassociated against the strict kernel's. The operands themselves are
// exact -- every BF16 value in FP16's normal range converts losslessly, and a
// product of two FP16 values is exact in FP32 -- so the only new error is
// summation order in the score dot and FP16 rounding of the softmax weights
// feeding the P*V dot. See the worklog entry for the measured tolerance.

#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
using half2_t = half_t __attribute__((ext_vector_type(2)));
using half8_t = half_t __attribute__((ext_vector_type(8)));
using half16_t = half_t __attribute__((ext_vector_type(16)));
using float8_t = float __attribute__((ext_vector_type(8)));

constexpr int kHeadDim = 256;
constexpr int kQueryRows = 16;
constexpr int kGqaHeads = 2;
constexpr int kColumns = kQueryRows * kGqaHeads;
constexpr int kDimTiles = kHeadDim / 16;
constexpr int kKBatch = 32;
constexpr int kKeyTiles = kKBatch / 16;
constexpr int kWaves = 2;
constexpr int kThreads = 32 * kWaves;
// Eight halves of row padding. llama.cpp's RDNA3 tiles use the same pad; it
// keeps every row start 16-byte aligned for the staging stores, which is what
// the staging loop vectorises on.
constexpr int kPad = 8;
constexpr int kKvStride = kHeadDim + kPad;
constexpr int kQStride = kHeadDim + kPad;

__device__ __forceinline__ float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

// Round-to-nearest-even BF16, the same conversion the strict kernel writes its
// output through, so the two kernels' outputs differ only where their
// arithmetic does.
__device__ __forceinline__ uint16_t float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

// One 16-byte load of eight BF16 values as eight FP16 values.
//
// BF16 -> FP16 is exact over FP16's normal range: FP16 carries three more
// mantissa bits than BF16 and the exponent rebias is lossless, so no value this
// kernel stages is rounded by the conversion. Splitting each 32-bit word into
// two FP32 bit patterns is a mask and a shift; the pair pack is the compiler's.
__device__ __forceinline__ half8_t bf16x8_to_half8(uint4 raw) {
  half8_t out;
  const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float lo = __uint_as_float((words[i] & 0xFFFFU) << 16);
    const float hi = __uint_as_float(words[i] & 0xFFFF0000U);
    out[2 * i] = static_cast<half_t>(lo);
    out[2 * i + 1] = static_cast<half_t>(hi);
  }
  return out;
}

__device__ __forceinline__ half8_t zero_half8() {
  return half8_t{static_cast<half_t>(0.0f), static_cast<half_t>(0.0f),
                 static_cast<half_t>(0.0f), static_cast<half_t>(0.0f),
                 static_cast<half_t>(0.0f), static_cast<half_t>(0.0f),
                 static_cast<half_t>(0.0f), static_cast<half_t>(0.0f)};
}

__device__ __forceinline__ float8_t zero_float8() {
  return float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}

// Stage one K or V tile (kKBatch rows of kHeadDim BF16) as FP16 into LDS.
//
// The chunk index advances by kThreads and a row is kHeadDim/8 chunks wide, so
// the key index advances by kThreads / (kHeadDim/8) every iteration while the
// dimension offset stays put. The source address is therefore one
// loop-invariant stride from the previous one, and hoisting it out of the loop
// removes a 64-bit multiply chain per chunk -- the same rewrite the strict
// kernel's pass 1 and pass 3 both needed.
__device__ __forceinline__ void stage_tile(
    half_t* __restrict__ tile,
    const uint16_t* __restrict__ source,
    int64_t k_base,
    int batch_keys,
    int kv_head,
    int64_t num_kv_heads,
    int linear_tid) {
  constexpr int kChunksPerRow = kHeadDim / 8;
  constexpr int kRowStep = kThreads / kChunksPerRow;
  const int first_key = linear_tid / kChunksPerRow;
  const int dim = (linear_tid - first_key * kChunksPerRow) * 8;
  const int64_t key_stride = static_cast<int64_t>(kRowStep) * num_kv_heads * kHeadDim;
  const uint16_t* src =
      source + ((k_base + first_key) * num_kv_heads + kv_head) * kHeadDim + dim;
  for (int key_local = first_key; key_local < batch_keys; key_local += kRowStep) {
    *reinterpret_cast<half8_t*>(&tile[key_local * kKvStride + dim]) =
        bf16x8_to_half8(*reinterpret_cast<const uint4*>(src));
    src += key_stride;
  }
}

__device__ __forceinline__ float lane_xor16(float value) {
  return __shfl_xor(value, 16, 32);
}

// One block per (16-query-row tile, KV head, GQA tile).
//
// Fragment conventions are the ones the two in-tree WMMA kernels already use,
// and they are what makes the P*V step below need no transpose:
//
//   A fragment (16 rows x 16 cols): lane l holds A[l & 15][k] for k in 0..15.
//   B fragment (16 rows x 16 cols): lane l holds B[k][l & 15] for k in 0..15.
//   D fragment (16 x 16, FP32):     lane l's element e is D[2e + (l >> 4)][l & 15].
//
// So the score tile C = A*B with A = K (rows are keys, columns are dims) and
// B = Q (rows are dims, columns are (query row, query head) pairs) lands as
// C[key][column] with the key index split across `lane >> 4`. The softmax
// reduction runs over keys, which is exactly that split, so one shuffle pairs a
// lane with its partner and the reduction costs one instruction per tile rather
// than one per key.
__global__ void __launch_bounds__(kThreads, 2)
gemma4_attention_prefill_wmma_kernel(
    const uint16_t* __restrict__ query,
    const uint16_t* __restrict__ key,
    const uint16_t* __restrict__ value,
    const uint8_t* __restrict__ keep_mask,
    uint16_t* __restrict__ out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t tokens,
    int64_t keys,
    int64_t window,
    int64_t row_offset) {
  extern __shared__ half_t shared[];
  half_t* q_tile = shared;                         // [kColumns][kQStride]
  half_t* kv_tile = q_tile + kColumns * kQStride;  // [kKBatch][kKvStride]

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int wave = static_cast<int>(threadIdx.x) >> 5;
  const int linear_tid = static_cast<int>(threadIdx.x);
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  const int query_tile = static_cast<int>(blockIdx.x);
  const int kv_head = static_cast<int>(blockIdx.y);
  const int gqa_tile = static_cast<int>(blockIdx.z);
  const int gqa_ratio = static_cast<int>(num_heads / num_kv_heads);

  // This lane's column: a (query row within the tile, query head within the
  // KV head's GQA group) pair. `jc` is the WMMA column index, so the low four
  // lane bits select the column and the high bit selects which half of the
  // 16-key reduction this lane owns.
  const int jc = wave * 16 + lane_lo;
  const int query_local = jc / kGqaHeads;
  const int gqa_local = jc - query_local * kGqaHeads;
  const int query_row = query_tile * kQueryRows + query_local;
  const int gqa_index = gqa_tile * kGqaHeads + gqa_local;
  const bool valid_column = (query_row < tokens) && (gqa_index < gqa_ratio);
  const int query_head = kv_head * gqa_ratio + gqa_index;
  // Rows past the token count must not be read, but the mask pointer for them
  // is still formed; clamping the index keeps that pointer in bounds.
  const uint8_t* mask_row =
      keep_mask + static_cast<int64_t>(valid_column ? query_row : 0) * keys;

  // Stage this tile's Q columns once: every K-batch re-reads them from LDS, and
  // they are the same across the whole walk. Masked-out columns are staged as
  // zero and their results are never written.
  for (int chunk = linear_tid; chunk < kColumns * (kHeadDim / 8); chunk += kThreads) {
    const int tile_jc = chunk / (kHeadDim / 8);
    const int dim = (chunk - tile_jc * (kHeadDim / 8)) * 8;
    const int tile_query_local = tile_jc / kGqaHeads;
    const int tile_gqa_local = tile_jc - tile_query_local * kGqaHeads;
    const int tile_row = query_tile * kQueryRows + tile_query_local;
    const int tile_gqa_index = gqa_tile * kGqaHeads + tile_gqa_local;
    half8_t staged = zero_half8();
    if (tile_row < tokens && tile_gqa_index < gqa_ratio) {
      const int tile_head = kv_head * gqa_ratio + tile_gqa_index;
      const uint16_t* src =
          query + (static_cast<int64_t>(tile_row) * num_heads + tile_head) * kHeadDim + dim;
      staged = bf16x8_to_half8(*reinterpret_cast<const uint4*>(src));
    }
    *reinterpret_cast<half8_t*>(&q_tile[tile_jc * kQStride + dim]) = staged;
  }
  // The K/V tile is zeroed once here rather than per batch. A batch's staging
  // loop then only has to write the rows it has, and the tail rows of a short
  // final batch keep the previous batch's values -- finite, and masked out of
  // the softmax before any of them can reach a result.
  for (int chunk = linear_tid; chunk < kKBatch * kKvStride / 8; chunk += kThreads) {
    *reinterpret_cast<half8_t*>(&kv_tile[chunk * 8]) = zero_half8();
  }
  __syncthreads();

  // Block-level walk bounds. A sliding layer keeps key k for a query at
  // position q only while q - k < window, so the tile's first row gives the
  // earliest column any of its rows can keep and its last row's own position
  // bounds the latest. Both are exact for the block and neither is used unless
  // `window > 0` promises the mask is zero outside them.
  int64_t first_kept = 0;
  int64_t walk_end = keys;
  if (window > 0) {
    const int64_t tile_first_position =
        row_offset + static_cast<int64_t>(query_tile) * kQueryRows;
    const int64_t candidate = tile_first_position - window + 1;
    if (candidate > 0) {
      first_kept = candidate;
    }
    const int64_t last = tile_first_position + kQueryRows;
    if (last < keys) {
      walk_end = last;
    }
  }
  if (first_kept > walk_end) {
    first_kept = walk_end;
  }

  float8_t accumulator[kDimTiles];
#pragma unroll
  for (int tile = 0; tile < kDimTiles; ++tile) {
    accumulator[tile] = zero_float8();
  }
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  for (int64_t k_base = first_kept; k_base < walk_end; k_base += kKBatch) {
    const int64_t batch_end =
        (walk_end - k_base < kKBatch) ? walk_end : k_base + kKBatch;
    const int batch_keys = static_cast<int>(batch_end - k_base);

    // --- stage K -----------------------------------------------------------
    // One 16-byte load and one 16-byte store per eight dimensions, so the whole
    // 32x256 tile is 16 loads and 16 stores per lane.
    stage_tile(kv_tile, key, k_base, batch_keys, kv_head, num_kv_heads, linear_tid);
    __syncthreads();

    // --- scores: C[key][column] = dot(K[key], Q[column]) -------------------
    float8_t scores[kKeyTiles];
#pragma unroll
    for (int key_tile = 0; key_tile < kKeyTiles; ++key_tile) {
      scores[key_tile] = zero_float8();
#pragma unroll
      for (int dim_tile = 0; dim_tile < kDimTiles; ++dim_tile) {
        half16_t key_fragment;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          key_fragment[k] =
              kv_tile[(key_tile * 16 + lane_lo) * kKvStride + dim_tile * 16 + k];
        }
        half16_t query_fragment;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          query_fragment[k] = q_tile[jc * kQStride + dim_tile * 16 + k];
        }
        scores[key_tile] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            key_fragment, query_fragment, scores[key_tile]);
      }
    }
    // Every wave is done reading the K tile before V overwrites it.
    __syncthreads();

    // --- mask, batch max, and the online-softmax rescale --------------------
    // `scale` is applied here rather than folded into the staged Q, so it is a
    // single FP32 multiply of the dot exactly where the strict kernel applies
    // its own. Gemma 4 passes 1.0 (it folds the softmax scale into the query
    // norm weight), which makes this an exact no-op, but the kernel multiplies
    // by whatever it is handed rather than assuming the reciprocal square root.
    float local_max = -INFINITY;
#pragma unroll
    for (int key_tile = 0; key_tile < kKeyTiles; ++key_tile) {
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const int key_local = key_tile * 16 + 2 * element + lane_hi;
        bool visible = valid_column && key_local < batch_keys;
        if (visible) {
          visible = mask_row[k_base + key_local] != 0;
        }
        scores[key_tile][element] =
            visible ? scores[key_tile][element] * scale : -INFINITY;
        local_max = fmaxf(local_max, scores[key_tile][element]);
      }
    }
    // The reduction over keys is the lane-high-bit split, so one shuffle
    // combines the two halves and every key of the batch is covered.
    const float batch_max = fmaxf(local_max, lane_xor16(local_max));
    const bool has_data = isfinite(batch_max);
    const float next_max = has_data ? fmaxf(running_max, batch_max) : 0.0f;
    // A batch with no visible key must preserve the running state rather than
    // rescale it toward an empty max, and expf(-inf - -inf) is NaN, so the
    // first batch is guarded rather than computed.
    const float previous_scale = !has_data
        ? 1.0f
        : (isfinite(running_max) ? expf(running_max - next_max) : 0.0f);
    // Once the running max stops moving this is a multiply by exactly 1.0, so
    // skipping it is exact and not an approximation.
    if (previous_scale != 1.0f) {
#pragma unroll
      for (int dim_tile = 0; dim_tile < kDimTiles; ++dim_tile) {
#pragma unroll
        for (int element = 0; element < 8; ++element) {
          accumulator[dim_tile][element] *= previous_scale;
        }
      }
    }

    // Weights, and the P operand for the P*V tile built in its B layout: lane l
    // must hold P[k][l & 15] for every k, and it holds only the keys whose
    // parity matches its own high bit, so its partner's eight values come back
    // over one shuffle.
    half2_t probability_pairs[kKeyTiles][8];
    float local_sum = 0.0f;
#pragma unroll
    for (int key_tile = 0; key_tile < kKeyTiles; ++key_tile) {
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const float score = scores[key_tile][element];
        const float probability =
            (has_data && isfinite(score)) ? expf(score - next_max) : 0.0f;
        scores[key_tile][element] = probability;
        local_sum += probability;
        const float partner = lane_xor16(probability);
        const half2_t ordered = half2_t{static_cast<half_t>(probability),
                                        static_cast<half_t>(partner)};
        probability_pairs[key_tile][element] =
            (lane_hi == 0) ? ordered
                           : half2_t{ordered[1], ordered[0]};
      }
    }
    const float batch_sum = local_sum + lane_xor16(local_sum);
    running_sum = previous_scale * running_sum + batch_sum;
    running_max = has_data ? next_max : running_max;

    // --- stage V into the tile K just vacated -------------------------------
    stage_tile(kv_tile, value, k_base, batch_keys, kv_head, num_kv_heads, linear_tid);
    __syncthreads();

    // --- accumulate: D[dim][column] += sum_key P[key][column] * V[key][dim] --
    // A is V transposed (rows are dims, columns are keys), which is a strided
    // LDS read rather than a contiguous one; storing V transposed would cost
    // more than it saves, since the staging write is coalesced and this read is
    // 16 halves against 16 keys of tensor-core work.
#pragma unroll
    for (int key_tile = 0; key_tile < kKeyTiles; ++key_tile) {
      const half16_t probability_fragment =
          *reinterpret_cast<const half16_t*>(probability_pairs[key_tile]);
#pragma unroll
      for (int dim_tile = 0; dim_tile < kDimTiles; ++dim_tile) {
        half16_t value_fragment;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          value_fragment[k] =
              kv_tile[(key_tile * 16 + k) * kKvStride + dim_tile * 16 + lane_lo];
        }
        accumulator[dim_tile] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            value_fragment, probability_fragment, accumulator[dim_tile]);
      }
    }
    __syncthreads();
  }

  if (!valid_column) {
    return;
  }
  // The strict kernel divides by its denominator and writes the quotient, so a
  // fully masked row is 0/0 there; this divides the same way rather than
  // special-casing it.
  const int64_t out_base =
      (static_cast<int64_t>(query_row) * num_heads + query_head) * kHeadDim;
#pragma unroll
  for (int dim_tile = 0; dim_tile < kDimTiles; ++dim_tile) {
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const int dim = dim_tile * 16 + 2 * element + lane_hi;
      out[out_base + dim] = float_to_bf16_bits(accumulator[dim_tile][element] / running_sum);
    }
  }
}

}  // namespace

extern "C" int hipengine_gemma4_attention_prefill_wmma_bf16(
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
    int64_t row_offset) {
  if (query == nullptr || key == nullptr || value == nullptr ||
      keep_mask == nullptr || out == nullptr || tokens <= 0 || keys <= 0 ||
      num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  // This kernel is the sliding-window geometry: head_dim 256, GQA ratio 2. The
  // head_dim-512 full layers are a separate unit; naming the miss is the
  // contract, silently running a different geometry is not.
  if (head_dim != kHeadDim || num_heads / num_kv_heads != kGqaHeads) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int query_tiles = static_cast<int>((tokens + kQueryRows - 1) / kQueryRows);
  const int gqa_tiles = static_cast<int>((num_heads / num_kv_heads + kGqaHeads - 1) / kGqaHeads);
  const size_t shared_bytes =
      static_cast<size_t>(kColumns * kQStride + kKBatch * kKvStride) * sizeof(half_t);
  hipLaunchKernelGGL(
      gemma4_attention_prefill_wmma_kernel,
      dim3(static_cast<unsigned int>(query_tiles), static_cast<unsigned int>(num_kv_heads),
           static_cast<unsigned int>(gqa_tiles)),
      dim3(kThreads, 1, 1),
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
      keys,
      window,
      row_offset);
  return static_cast<int>(hipGetLastError());
}
