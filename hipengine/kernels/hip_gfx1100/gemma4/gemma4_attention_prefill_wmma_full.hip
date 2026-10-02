// Gemma 4 BF16 full-attention (head_dim 512) prefill on gfx11, on the matrix cores.
//
// The strict `gemma4_attention_prefill_bf16` kernel serves every Gemma 4
// geometry at 276-462 GFLOP/s, about one percent of this part's BF16 rate: its
// pass 1 is one block of 256 threads per (token, head) with a warp-level
// reduction per key, so its cost is the per-key loop machinery rather than its
// arithmetic. The sliding layers are covered by
// `gemma4_attention_prefill_wmma.hip`, which removed that reduction for
// head_dim 256. This is the other half of the family: the five head_dim-512
// full layers, which the sliding kernel refuses because it does not implement
// their geometry.
//
// Geometry (Gemma 4 26B-A4B full layers: 5 of 30, indices 5, 11, 17, 23, 29):
//   head_dim 512, 16 query heads, 2 KV heads, GQA ratio 8, causal, no window.
//
// Arithmetic lineage: the tile structure, the online-softmax K-batches, the
// fragment conventions and the mask contract follow
// `gemma4_attention_prefill_wmma.hip` (itself a port of llama.cpp
// fattn-mma-f16.cuh through `laguna_flash_attention_prefill.hip`). The one
// change of substance is the operand width: this kernel feeds the matrix unit
// BF16 directly instead of converting each staged value to FP16 first, which is
// both exact (the artifact is BF16, so no value is rounded by staging) and
// cheaper (the staging loop is a 16-byte copy rather than four masks, two
// shifts and eight converts per eight values). A standalone probe on this part
// measures BF16 and FP16 WMMA at the same rate -- 54.6 TFLOP/s f16 against
// 54.6-55.3 TFLOP/s bf16 across three runs, 5120 blocks x 4 chains x 4096
// iterations -- so the wider operand costs nothing.
//
// Why the block is shaped the way it is:
//
//   QUERY_ROWS  16   one WMMA tile row count. The strict kernel has no query
//                    tiling at all, which is why M=16 is the whole point.
//   GQA_HEADS    2   query heads per block. Eight query heads share one KV
//                    head, so this is a free tiling axis: a wider block
//                    re-reads the same K/V, a narrower one re-stages it. The
//                    staged bytes per useful flop are exactly 1 / COLUMNS, so
//                    this is the only lever on arithmetic intensity.
//   COLUMNS     32   QUERY_ROWS * GQA_HEADS, the WMMA column count.
//   K_BATCH     16   keys per online-softmax batch (one 16-key WMMA tile).
//   WAVES        4   THREADS 128. Two column groups of sixteen columns and two
//                    output-dimension groups of 256 dimensions.
//
// Register pressure is the binding constraint, exactly as at head_dim 256. A
// wave's accumulator is (columns it owns) * (dimensions it owns) / 32 lanes
// floats, and the FP32 score/max/sum state is separate. At head_dim 512 the
// natural shape -- one wave per sixteen columns owning all 512 output
// dimensions -- needs 256 accumulator floats per lane, which is the wave32
// architectural ceiling before any other state and cannot compile without
// spilling. Splitting the output dimensions across two waves per column group
// halves it to the same 128 floats per lane the in-tree INT8 WMMA prefill and
// the sliding candidate both run at.
//
// That split is not free: the two waves that share a column group each need the
// full head_dim score for those columns, so the score dot is issued twice. The
// block issues 1.5x the ideal WMMA count. It does not change the staged bytes,
// because both dimension groups are served by the same staged V tile, and the
// tensor unit is not the binding cost here.
//
// Shared memory is the second constraint and it is why K_BATCH is 16 rather
// than 32. The Q tile must hold COLUMNS * head_dim BF16 for the whole walk --
// each wave re-reads its columns' Q for every K-batch -- and at head_dim 512
// that is 32 KB of the 64 KB a gfx11 workgroup can allocate. A 32-key K/V tile
// would need another 32 KB and the two would not fit; a 16-key tile leaves
// 49,920 bytes. Both tiles are padded by eight elements so that every row start
// lands on a distinct 16-byte LDS phase: an unpadded 1024-byte row stride puts
// every lane of a 16-byte fragment load on the same bank group.
//
// Masking is the caller's keep-mask, not a re-derived causal test, and the
// walk bounds are the sliding candidate's. Two block-level bounds trim the
// walk: `first_kept` from the tile's first row and `walk_end` from its last.
// Both are used only when `window > 0`, which is the caller's promise that the
// mask is zero outside them -- for a sliding layer that is the sliding-causal
// form, and for a full layer the runtime passes `window = start + rows -
// key_begin`, i.e. the whole context, which makes the leading bound inactive
// and leaves `walk_end` as the causal trim. `window == 0` means no promise, so
// the walk covers every column and only the mask decides visibility.
//
// This is a changed-arithmetic candidate, not a strict variant. WMMA
// accumulates each 16-wide dot in the matrix unit's own order, so the logits
// are reassociated against the strict kernel's, and the softmax weights reach
// the P*V dot as BF16 rather than FP32. Both operands are exact -- the staged
// K, Q and V values are the artifact's own BF16 values, unrounded -- so the
// only new error is summation order in the score dot, the BF16 rounding of the
// weights feeding the P*V dot, and the FP32 accumulation order of the P*V sum.
// The worklog entry for this unit carries the measured tolerance.

#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace {

using bf16_t = __bf16;
using bf16x2_t = bf16_t __attribute__((ext_vector_type(2)));
using bf16x16_t = bf16_t __attribute__((ext_vector_type(16)));
using float8_t = float __attribute__((ext_vector_type(8)));

// The query-block shape is a build-time parameter, not a runtime one.
//
// `kColumns` is `kQueryRows * kGqaHeads` and must be a whole number of WMMA M
// tiles, and `gqa_tiles` is the GQA ratio over `kGqaHeads` -- so the pair sets
// both the shared-memory the Q tile costs and how many times the KV band is
// read. Prefill and decode want opposite ends of that trade, so they are built
// as two objects from this one source rather than reconciled in one shape:
//
//   prefill  kQueryRows 16, kGqaHeads 2 -> kColumns 32, gqa_tiles 4, LDS 48.8 KB
//   decode   kQueryRows  2, kGqaHeads 8 -> kColumns 16, gqa_tiles 1, LDS 32.5 KB
//
// A 512-row prefill block is 32 query tiles and already presents 256 blocks, so
// its 4x KV re-read is paid to keep the Q tile at 32.5 KB and is not what it is
// waiting on. A decode block is one query tile against the whole context: it
// has blocks to spare and nothing to spend them on but the walk, and at
// `kGqaHeads` 8 each band is read exactly once. Sixteen rows is what makes the
// decode shape not fit -- at `kQueryRows` 16 and `kGqaHeads` 8 the Q tile alone
// is 130 KB against a 64 KB budget -- and two rows is all a decode needs, with
// the second row padding the WMMA M tile to sixteen columns.
#ifndef GEMMA4_WMMA_FULL_QUERY_ROWS
#define GEMMA4_WMMA_FULL_QUERY_ROWS 16
#endif
#ifndef GEMMA4_WMMA_FULL_GQA_HEADS
#define GEMMA4_WMMA_FULL_GQA_HEADS 2
#endif
#ifndef GEMMA4_WMMA_FULL_SYMBOL
#define GEMMA4_WMMA_FULL_SYMBOL hipengine_gemma4_attention_prefill_wmma_full_bf16
#endif
// The device kernel is in an anonymous namespace, so its name is what a
// `rocprofv3` trace reports and it is the only thing that distinguishes the two
// shapes there -- the launcher symbol does not appear in a kernel trace. It
// carries the query-block shape for that reason: "confirm the intended path
// ran" is not answerable if both objects dispatch a kernel of the same name.
#ifndef GEMMA4_WMMA_FULL_KERNEL_NAME
#define GEMMA4_WMMA_FULL_KERNEL_NAME gemma4_attention_prefill_wmma_full_kernel
#endif

constexpr int kHeadDim = 512;
constexpr int kQueryRows = GEMMA4_WMMA_FULL_QUERY_ROWS;
constexpr int kGqaHeads = GEMMA4_WMMA_FULL_GQA_HEADS;
constexpr int kGqaRatio = 8;
constexpr int kColumns = kQueryRows * kGqaHeads;
constexpr int kColumnGroups = kColumns / 16;
constexpr int kDimTiles = kHeadDim / 16;
// How the head dimension is split across waves. `kWaves` is
// `kColumnGroups * kDimGroups` and each wave owns `kDimPerWave` output columns,
// so this is what sets `kThreads` -- and `kThreads` is what sets how many loads
// the staging loop has in flight. The prefill shape's `kColumnGroups` is 2 and
// the decode shape's is 1, so at a shared 2 the decode build runs 64 threads
// where the prefill runs 128. A pure-read probe over the walk's own address
// sequence says that grid is the walk's residual: 2,048 threads reached 158.8
// GB/s at the walk's 32x64 grid while 1024x256 reached 231.0, and a flat
// contiguous read at the walk's grid was *slower* than the strided one (101.1),
// so it is the thread count and not the KV layout.
#ifndef GEMMA4_WMMA_FULL_DIM_GROUPS
#define GEMMA4_WMMA_FULL_DIM_GROUPS 2
#endif
constexpr int kDimGroups = GEMMA4_WMMA_FULL_DIM_GROUPS;
constexpr int kDimTilesPerWave = kDimTiles / kDimGroups;
constexpr int kDimPerWave = kHeadDim / kDimGroups;
constexpr int kKBatch = 16;
constexpr int kKeyTiles = kKBatch / 16;
constexpr int kWaves = kColumnGroups * kDimGroups;
constexpr int kThreads = 32 * kWaves;
// Eight halves of row padding. llama.cpp's RDNA3 tiles use the same pad; it
// keeps every row start 16-byte aligned for the staging stores and puts
// consecutive rows on consecutive LDS 16-byte phases, which is what keeps a
// sixteen-lane fragment load off a sixteen-way bank conflict.
constexpr int kPad = 8;
constexpr int kKvStride = kHeadDim + kPad;
constexpr int kQStride = kHeadDim + kPad;
// Staging geometry: a row is 64 chunks of eight BF16 values, so 128 threads
// cover two rows per iteration and each thread walks its own key stride.
constexpr int kChunksPerRow = kHeadDim / 8;
constexpr int kRowStep = kThreads / kChunksPerRow;

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

__device__ __forceinline__ float8_t zero_float8() {
  return float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ bf16x16_t zero_bf16x16() {
  bf16x16_t values;
#pragma unroll
  for (int item = 0; item < 16; ++item) {
    values[item] = static_cast<bf16_t>(0.0f);
  }
  return values;
}

// A 32-byte LDS fragment load. Both tiles are padded so this is 16-byte
// aligned; the compiler splits it into two 128-bit LDS reads.
__device__ __forceinline__ bf16x16_t load_bf16x16(const bf16_t* __restrict__ source) {
  return *reinterpret_cast<const bf16x16_t*>(source);
}

// Stage one K or V tile (kKBatch rows of kHeadDim BF16) into LDS.
//
// A pure copy: the artifact is BF16 and the matrix unit consumes BF16, so no
// value is converted or rounded here. The chunk index advances by kThreads and
// a row is kHeadDim/8 chunks wide, so the key index advances by kRowStep every
// iteration while the dimension offset stays put. The source address is
// therefore one loop-invariant stride from the previous one, and hoisting it
// out of the loop removes a 64-bit multiply chain per chunk -- the same rewrite
// the strict kernel's pass 1 and pass 3 both needed.
__device__ __forceinline__ void stage_tile(
    bf16_t* __restrict__ tile,
    const uint16_t* __restrict__ source,
    int64_t k_base,
    int batch_keys,
    int kv_head,
    int64_t num_kv_heads,
    int linear_tid) {
  constexpr int kMaxIters = kKBatch / kRowStep;
  const int first_key = linear_tid / kChunksPerRow;
  const int dim = (linear_tid - first_key * kChunksPerRow) * 8;
  const int64_t key_stride = static_cast<int64_t>(kRowStep) * num_kv_heads * kHeadDim;
  const uint16_t* src =
      source + ((k_base + first_key) * num_kv_heads + kv_head) * kHeadDim + dim;
  // All eight reads are issued before the first store consumes one, so their
  // latency overlaps instead of serialising behind a store-and-advance loop.
  uint4 staged[kMaxIters];
#pragma unroll
  for (int i = 0; i < kMaxIters; ++i) {
    const int key_local = first_key + i * kRowStep;
    staged[i] = (key_local < batch_keys)
        ? *reinterpret_cast<const uint4*>(src + static_cast<int64_t>(i) * key_stride)
        : make_uint4(0u, 0u, 0u, 0u);
  }
#pragma unroll
  for (int i = 0; i < kMaxIters; ++i) {
    const int key_local = first_key + i * kRowStep;
    if (key_local < batch_keys) {
      *reinterpret_cast<uint4*>(&tile[key_local * kKvStride + dim]) = staged[i];
    }
  }
}

__device__ __forceinline__ float lane_xor16(float value) {
  return __shfl_xor(value, 16, 32);
}

// One block per (16-query-row tile, KV head, GQA tile).
//
// Fragment conventions are the ones the in-tree WMMA kernels use, and they are
// what makes the P*V step below need no transpose:
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
//
// The wave's (column group, dimension group) pair is fixed for the whole
// launch. Every wave computes the complete score for its own sixteen columns --
// both dimension groups need it -- and accumulates only its own 256 output
// dimensions, which is what keeps the accumulator at 128 floats per lane.
__global__ void __launch_bounds__(kThreads, 1)
GEMMA4_WMMA_FULL_KERNEL_NAME(
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
    int64_t row_offset,
    float* __restrict__ partial,
    float* __restrict__ partial_stats,
    int64_t slices) {
  extern __shared__ bf16_t shared[];
  bf16_t* q_tile = shared;                         // [kColumns][kQStride]
  bf16_t* kv_tile = q_tile + kColumns * kQStride;  // [kKBatch][kKvStride]

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int wave = static_cast<int>(threadIdx.x) >> 5;
  const int linear_tid = static_cast<int>(threadIdx.x);
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  // Axis order is a performance decision, not a correctness one: the same work
  // happens in any order. Each KV band is read once per (query_tile, kv_head,
  // gqa_tile), so `gqa_tiles` blocks read the *same* band, and the only chance
  // they have of sharing it is to run together. The LDS a block holds (49.9 KB
  // against 64 KB per CU) means about one block per CU is resident, so
  // co-residency is decided entirely by launch order. With query_tile on x the
  // four readers sit 32 blocks apart and never meet; with gqa_tile on x they
  // are adjacent and walk the same key tile in lockstep. The key slice rides on
  // x beside it for the split walk, where consecutive slices read disjoint
  // ranges and the order is immaterial.
  //
  // Measured: 2.7 percent at 262144 keys, 4.2 at 131072, 3.2 at 65536, 8.4 at
  // 16384 -- small, free (no arithmetic changes, and the parity suite is
  // bit-identical), and worth keeping. The mechanism is *not* the L2 reuse this
  // was written expecting. That was tested directly: at 2048 keys the whole band
  // is 4.2 MB and fits in L2, and the walk still reaches only 122 percent of the
  // 238.5 GB/s DRAM roofline, so at most ~18 percent of the redundant traffic is
  // L2-served even when the entire working set is cache-resident. Co-scheduling
  // the readers is not sufficient to recover the redundancy; the smaller win
  // here comes from something else and is not attributed.
  const int gqa_ratio = static_cast<int>(num_heads / num_kv_heads);
  const int gqa_tiles = (gqa_ratio + kGqaHeads - 1) / kGqaHeads;
  const int gqa_tile = static_cast<int>(blockIdx.x) % gqa_tiles;
  const int slice = static_cast<int>(blockIdx.x) / gqa_tiles;
  const int query_tile = static_cast<int>(blockIdx.y);
  const int kv_head = static_cast<int>(blockIdx.z);

  // This wave's column group and output-dimension group.
  const int column_group = wave / kDimGroups;
  const int dim_group = wave - column_group * kDimGroups;
  const int dim_base = dim_group * kDimPerWave;

  // This lane's column: a (query row within the tile, query head within the
  // KV head's GQA group) pair. `jc` is the WMMA column index, so the low four
  // lane bits select the column and the high bit selects which half of the
  // 16-key reduction this lane owns.
  const int jc = column_group * 16 + lane_lo;
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
  for (int chunk = linear_tid; chunk < kColumns * kChunksPerRow; chunk += kThreads) {
    const int tile_jc = chunk / kChunksPerRow;
    const int dim = (chunk - tile_jc * kChunksPerRow) * 8;
    const int tile_query_local = tile_jc / kGqaHeads;
    const int tile_gqa_local = tile_jc - tile_query_local * kGqaHeads;
    const int tile_row = query_tile * kQueryRows + tile_query_local;
    const int tile_gqa_index = gqa_tile * kGqaHeads + tile_gqa_local;
    uint4 staged = make_uint4(0u, 0u, 0u, 0u);
    if (tile_row < tokens && tile_gqa_index < gqa_ratio) {
      const int tile_head = kv_head * gqa_ratio + tile_gqa_index;
      const uint16_t* src =
          query + (static_cast<int64_t>(tile_row) * num_heads + tile_head) * kHeadDim + dim;
      staged = *reinterpret_cast<const uint4*>(src);
    }
    *reinterpret_cast<uint4*>(&q_tile[tile_jc * kQStride + dim]) = staged;
  }
  // The K/V tile is zeroed once here rather than per batch. A batch's staging
  // loop then only has to write the rows it has, and the tail rows of a short
  // final batch keep the previous batch's values -- finite, and masked out of
  // the softmax before any of them can reach a result.
  for (int chunk = linear_tid; chunk < kKBatch * kKvStride / 8; chunk += kThreads) {
    *reinterpret_cast<uint4*>(&kv_tile[chunk * 8]) = make_uint4(0u, 0u, 0u, 0u);
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
  // Key split. A decode step presents one query tile against the whole context,
  // so the grid is `num_kv_heads * gqa_tiles` blocks -- 8 at this geometry --
  // and each of them walks every key. Measured on gfx1151 at 262,144 keys, 8
  // blocks reach 75 GB/s where 64 reach 225 and 256 reach 252, so the kernel is
  // waiting on itself rather than on memory. Dividing the walk gives the
  // hardware something to overlap; the per-slice softmax state is combined
  // afterwards. The slices partition `[first_kept, walk_end)` so their union is
  // the unsplit walk exactly, and each is rounded up to the K batch so no slice
  // is left with a partial tile.
  if (slices > 1) {
    const int64_t span = walk_end - first_kept;
    const int64_t per_slice =
        ((span + slices - 1) / slices + kKBatch - 1) / kKBatch * kKBatch;
    const int64_t begin = first_kept + static_cast<int64_t>(slice) * per_slice;
    const int64_t end = begin + per_slice;
    first_kept = (begin < walk_end) ? begin : walk_end;
    walk_end = (end < walk_end) ? end : walk_end;
  }

  float8_t accumulator[kDimTilesPerWave];
#pragma unroll
  for (int tile = 0; tile < kDimTilesPerWave; ++tile) {
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
    // 16x512 tile is 8 loads and 8 stores per lane. No conversion: the staged
    // value is the artifact's own BF16 value.
    stage_tile(kv_tile, key, k_base, batch_keys, kv_head, num_kv_heads, linear_tid);
    __syncthreads();

    // --- scores: C[key][column] = dot(K[key], Q[column]) -------------------
    // The dimension loop is the outer one so the Q fragment is loaded once per
    // dimension tile rather than once per (dimension tile, key tile). At
    // K_BATCH 16 there is one key tile, so this is the same order the sliding
    // candidate runs; it matters only if K_BATCH grows.
    float8_t scores[kKeyTiles];
#pragma unroll
    for (int key_tile = 0; key_tile < kKeyTiles; ++key_tile) {
      scores[key_tile] = zero_float8();
    }
#pragma unroll
    for (int dim_tile = 0; dim_tile < kDimTiles; ++dim_tile) {
      const bf16x16_t query_fragment = load_bf16x16(&q_tile[jc * kQStride + dim_tile * 16]);
#pragma unroll
      for (int key_tile = 0; key_tile < kKeyTiles; ++key_tile) {
        const bf16x16_t key_fragment =
            load_bf16x16(&kv_tile[(key_tile * 16 + lane_lo) * kKvStride + dim_tile * 16]);
        scores[key_tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
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
    // The 16-key batch is a contiguous 16-byte window of this lane's mask row,
    // so one vector read covers the whole batch. Eight scattered byte reads per
    // lane per batch were the same eight sectors eight times over, and the
    // dependent compare made them the longest link in the batch's critical
    // path. The read starts at a clamped offset so a short final batch cannot
    // run past the end of the row.
    //
    // A row shorter than one batch cannot be read 16 bytes at a time at all, so
    // those four words are assembled from byte reads here, once, rather than
    // left as a per-element branch inside the loop below: the branch cost more
    // than the read it guarded. A row that short has exactly one batch in its
    // walk, so this happens once per row.
    const bool mask_wide = keys >= kKBatch;
    const int64_t mask_base =
        mask_wide ? ((k_base + kKBatch <= keys) ? k_base : keys - kKBatch) : 0;
    const int mask_shift = static_cast<int>(k_base - mask_base);
    uint32_t mask_word[4] = {0u, 0u, 0u, 0u};
    if (mask_wide) {
      const uint4 raw = *reinterpret_cast<const uint4*>(mask_row + mask_base);
      mask_word[0] = raw.x;
      mask_word[1] = raw.y;
      mask_word[2] = raw.z;
      mask_word[3] = raw.w;
    } else {
#pragma unroll
      for (int index = 0; index < kKBatch; ++index) {
        if (index < keys) {
          mask_word[index >> 2] |= static_cast<uint32_t>(mask_row[index] != 0)
              << (8 * (index & 3));
        }
      }
    }
#pragma unroll
    for (int key_tile = 0; key_tile < kKeyTiles; ++key_tile) {
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const int key_local = key_tile * 16 + 2 * element + lane_hi;
        bool visible = valid_column && key_local < batch_keys;
        if (visible) {
          const int byte_index = key_local + mask_shift;
          visible = (mask_word[byte_index >> 2] >> (8 * (byte_index & 3))) & 0xFF;
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
    // skipping it is exact and not an approximation. At 128 accumulator floats
    // per lane this is the most expensive thing in the batch after the walk
    // itself, so the skip is worth having.
    if (previous_scale != 1.0f) {
#pragma unroll
      for (int dim_tile = 0; dim_tile < kDimTilesPerWave; ++dim_tile) {
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
    bf16x2_t probability_pairs[kKeyTiles][8];
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
        const bf16x2_t ordered = bf16x2_t{static_cast<bf16_t>(probability),
                                          static_cast<bf16_t>(partner)};
        probability_pairs[key_tile][element] =
            (lane_hi == 0) ? ordered : bf16x2_t{ordered[1], ordered[0]};
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
    // 16 halves against 16 keys of tensor-core work. Only this wave's own 256
    // output dimensions are accumulated.
#pragma unroll
    for (int key_tile = 0; key_tile < kKeyTiles; ++key_tile) {
      const bf16x16_t probability_fragment =
          *reinterpret_cast<const bf16x16_t*>(probability_pairs[key_tile]);
#pragma unroll
      for (int dim_tile = 0; dim_tile < kDimTilesPerWave; ++dim_tile) {
        const int dim = dim_base + dim_tile * 16 + lane_lo;
        bf16x16_t value_fragment;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          value_fragment[k] = kv_tile[(key_tile * 16 + k) * kKvStride + dim];
        }
        accumulator[dim_tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            value_fragment, probability_fragment, accumulator[dim_tile]);
      }
    }
    __syncthreads();
  }

  if (!valid_column) {
    return;
  }
  // A split walk exports its own softmax state and the combine pass finishes
  // the reduction. The accumulator is this lane's `kDimTilesPerWave` x 8
  // elements at the dims the epilogue below would have written, so the export
  // and the combine share one mapping and neither needs a layout conversion.
  if (slices > 1) {
    const int64_t plane =
        static_cast<int64_t>(num_kv_heads) * gqa_tiles * kColumns;
    const int64_t base =
        (static_cast<int64_t>(kv_head) * gqa_tiles + gqa_tile) * kColumns + jc;
    float* dst = partial + (static_cast<int64_t>(slice) * plane + base) * kHeadDim;
#pragma unroll
    for (int dim_tile = 0; dim_tile < kDimTilesPerWave; ++dim_tile) {
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const int dim = dim_base + dim_tile * 16 + 2 * element + lane_hi;
        dst[dim] = accumulator[dim_tile][element];
      }
    }
    // The max and the denominator depend only on this column's scores, not on
    // the output dimensions, so the two waves that share a `jc` compute the
    // same pair and both store it.
    float* stats =
        partial_stats + static_cast<int64_t>(slice) * plane * 2 + base * 2;
    stats[0] = running_max;
    stats[1] = running_sum;
    return;
  }
  // The strict kernel divides by its denominator and writes the quotient, so a
  // fully masked row is 0/0 there; this divides the same way rather than
  // special-casing it. The division is once per (column, block), not once per
  // key, so it is not worth replacing with a reciprocal multiply.
  const int64_t out_base =
      (static_cast<int64_t>(query_row) * num_heads + query_head) * kHeadDim;
#pragma unroll
  for (int dim_tile = 0; dim_tile < kDimTilesPerWave; ++dim_tile) {
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const int dim = dim_base + dim_tile * 16 + 2 * element + lane_hi;
      out[out_base + dim] = float_to_bf16_bits(accumulator[dim_tile][element] / running_sum);
    }
  }
}

// Combine the slices a split walk produced.
//
// Each slice's block ran its own online softmax, so slice `s` holds
// `acc_s = sum_k exp(score_k - m_s) V_k` and `sum_s = sum_k exp(score_k - m_s)`
// over its own key range. Rescaling every slice to the largest of the `m_s` and
// adding gives exactly the unsplit result; the arithmetic differs only in which
// maximum the exponentials were taken against, which is a reassociation and not
// an approximation.
//
// The thread mapping is the main kernel's, so each thread reads back the same
// `kDimTilesPerWave` x 8 elements it wrote and no shuffle or shared-memory
// staging is needed. A slice with no visible key left `m_s = -infinity` and
// `sum_s = 0`; `isfinite` sends it to a zero weight rather than an `expf` of
// NaN. If every slice is empty the denominator is 0 and the quotient is 0/0,
// which is what the strict kernel produces for a fully masked row.
__global__ void gemma4_attention_wmma_combine_kernel(
    const float* __restrict__ partial,
    const float* __restrict__ partial_stats,
    uint16_t* __restrict__ out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t tokens,
    int64_t slices) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int wave = static_cast<int>(threadIdx.x) >> 5;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  const int query_tile = static_cast<int>(blockIdx.y);
  const int kv_head = static_cast<int>(blockIdx.z);
  const int gqa_tile = static_cast<int>(blockIdx.x);
  const int gqa_ratio = static_cast<int>(num_heads / num_kv_heads);
  const int gqa_tiles = (gqa_ratio + kGqaHeads - 1) / kGqaHeads;

  const int column_group = wave / kDimGroups;
  const int dim_group = wave - column_group * kDimGroups;
  const int dim_base = dim_group * kDimPerWave;
  const int jc = column_group * 16 + lane_lo;
  const int query_local = jc / kGqaHeads;
  const int gqa_local = jc - query_local * kGqaHeads;
  const int query_row = query_tile * kQueryRows + query_local;
  const int gqa_index = gqa_tile * kGqaHeads + gqa_local;
  if (query_row >= tokens || gqa_index >= gqa_ratio) {
    return;
  }
  const int query_head = kv_head * gqa_ratio + gqa_index;

  const int64_t plane = static_cast<int64_t>(num_kv_heads) * gqa_tiles * kColumns;
  const int64_t base =
      (static_cast<int64_t>(kv_head) * gqa_tiles + gqa_tile) * kColumns + jc;
  const int64_t stat_stride = plane * 2;

  float best = -INFINITY;
  for (int64_t s = 0; s < slices; ++s) {
    best = fmaxf(best, partial_stats[s * stat_stride + base * 2]);
  }
  float accumulator[kDimTilesPerWave][8];
#pragma unroll
  for (int dim_tile = 0; dim_tile < kDimTilesPerWave; ++dim_tile) {
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      accumulator[dim_tile][element] = 0.0f;
    }
  }
  float denominator = 0.0f;
  for (int64_t s = 0; s < slices; ++s) {
    const float slice_max = partial_stats[s * stat_stride + base * 2];
    const float weight = isfinite(slice_max) ? expf(slice_max - best) : 0.0f;
    denominator += weight * partial_stats[s * stat_stride + base * 2 + 1];
    const float* src = partial + (s * plane + base) * kHeadDim;
#pragma unroll
    for (int dim_tile = 0; dim_tile < kDimTilesPerWave; ++dim_tile) {
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        accumulator[dim_tile][element] +=
            weight * src[dim_base + dim_tile * 16 + 2 * element + lane_hi];
      }
    }
  }

  const int64_t out_base =
      (static_cast<int64_t>(query_row) * num_heads + query_head) * kHeadDim;
#pragma unroll
  for (int dim_tile = 0; dim_tile < kDimTilesPerWave; ++dim_tile) {
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const int dim = dim_base + dim_tile * 16 + 2 * element + lane_hi;
      out[out_base + dim] =
          float_to_bf16_bits(accumulator[dim_tile][element] / denominator);
    }
  }
}

}  // namespace

extern "C" int GEMMA4_WMMA_FULL_SYMBOL(
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
    void* scratch,
    int64_t slices) {
  if (query == nullptr || key == nullptr || value == nullptr ||
      keep_mask == nullptr || out == nullptr || tokens <= 0 || keys <= 0 ||
      num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  // This kernel is the full-layer geometry: head_dim 512, GQA ratio 8. The
  // sliding layers are head_dim 256 with a ratio of 2 and are a different unit;
  // naming the miss is the contract, silently running a different geometry is
  // not.
  if (head_dim != kHeadDim || num_heads / num_kv_heads != kGqaRatio) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int query_tiles = static_cast<int>((tokens + kQueryRows - 1) / kQueryRows);
  const int gqa_tiles = static_cast<int>((num_heads / num_kv_heads + kGqaHeads - 1) / kGqaHeads);
  const size_t shared_bytes =
      static_cast<size_t>(kColumns * kQStride + kKBatch * kKvStride) * sizeof(bf16_t);

  // The key split. The caller plans it -- see
  // `plan_gemma4_attention_wmma_full_slices`, which owns the rule and the
  // measurement behind it -- and this launcher only has to lay the workspace
  // out and size the grid. `slices == 1` is the unsplit walk: no workspace, no
  // combine launch, and the epilogue writes the output directly, so every
  // prefill shape runs exactly the arithmetic it always has.
  if (slices < 1) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (slices > 1 && scratch == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t planes = static_cast<int64_t>(num_kv_heads) * gqa_tiles;
  float* partial = nullptr;
  float* partial_stats = nullptr;
  if (slices > 1) {
    partial = static_cast<float*>(scratch);
    partial_stats = partial + slices * planes * kColumns * kHeadDim;
  }

  hipLaunchKernelGGL(
      GEMMA4_WMMA_FULL_KERNEL_NAME,
      // See the kernel's axis comment: gqa_tile on x is what puts the readers
      // of one KV band next to each other. Worth 2.7 percent at 262144 keys.
      dim3(static_cast<unsigned int>(gqa_tiles * slices), static_cast<unsigned int>(query_tiles),
           static_cast<unsigned int>(num_kv_heads)),
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
      row_offset,
      partial,
      partial_stats,
      slices);
  if (slices > 1) {
    hipLaunchKernelGGL(
        gemma4_attention_wmma_combine_kernel,
        dim3(static_cast<unsigned int>(gqa_tiles), static_cast<unsigned int>(query_tiles),
             static_cast<unsigned int>(num_kv_heads)),
        dim3(kThreads, 1, 1),
        0,
        stream,
        partial,
        partial_stats,
        out,
        num_heads,
        num_kv_heads,
        tokens,
        slices);
  }
  return static_cast<int>(hipGetLastError());
}
