// WMMA masked prefill attention for Qwen4Exp QSA layers (24 query heads of
// 256 dims over two KV heads, four-token selection blocks).
//
// WmmaCausalAttentionKernel and its lane helpers are ported unchanged from
// gufo v0.12.0 (commit 88090afb, MIT License, Copyright (c) 2026 gufo
// contributors), src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp.
// The two kernels after it are hipEngine's: gufo keeps an F16 K/V cache and
// a per-query block bitmask, while hipEngine keeps a paged BF16 K/V cache and
// per-query position lists, so the cache is gathered to contiguous F16 rows
// and the lists are folded into bitmasks before the attention launch.
//
// Queries, keys and values are F16 operands on the matrix units with F32
// score and output accumulation and an online softmax, so this changes
// arithmetic relative to the scalar FP32 attention owners.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr unsigned kThreads = 256;

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

/// Value of the lane whose index differs from this one in the `offset` bit,
/// across a wave32.
__device__ __forceinline__ int XorLane(int v, int offset) {
#if defined(__gfx1151__)
  switch (offset) {
    case 1:
      return __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, false);
    case 2:
      return __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, false);
    case 4:
      return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, false);
    case 8:
      return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, false);
    case 16:
      return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false,
                                          false);
  }
#endif
  return __shfl_xor(v, offset);
}

__device__ __forceinline__ unsigned XorLane(unsigned v, int offset) {
  return static_cast<unsigned>(XorLane(static_cast<int>(v), offset));
}

__device__ __forceinline__ float XorLane(float v, int offset) {
  return __builtin_bit_cast(float, XorLane(__builtin_bit_cast(int, v), offset));
}

// F16 WMMA fragments (wave32): sixteen halves per lane, eight F32
// accumulators per lane.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

// A barrier for kernels whose waves exchange data through LDS only.
// __syncthreads also waits for every outstanding global load and drops the
// L0 cache, which serializes register prefetches behind the barrier. The
// global inputs are read-only here, so completing LDS traffic is enough.
__device__ __forceinline__ __attribute__((convergent)) void SyncLds() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
  asm volatile("" ::: "memory");
}

// Masked prefill attention on the WMMA matrix cores. Wave32 fragment layout:
// A holds row L%16 and 16 contiguous k, B holds column L%16 and 16 contiguous
// k, and C element i is row 2i + L/16, column L%16. Every block owns its
// query rows against one 16-key tile at a time; eight waves split as S tiles
// x two halves of the head dimension, and the next tile's K/V is prefetched
// into registers behind the current tile's S, softmax and PV phases. V
// reaches LDS transposed with one key per lane so the writes stay
// conflict-free. The optional block mask follows the model's sparse
// selection: keys at or past the query's incomplete tail block are always
// visible, earlier blocks only when their bit is set.
constexpr std::uint32_t kWmmaHeadDim = 256;
constexpr std::uint32_t kWmmaQueryHeads = 24;
constexpr std::uint32_t kWmmaKvHeads = 2;
constexpr std::uint32_t kWmmaGqa = kWmmaQueryHeads / kWmmaKvHeads;
constexpr std::uint32_t kWmmaAttnWidth = kWmmaQueryHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaKvWidth = kWmmaKvHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaHeads = 2;  // query heads per block, divides GQA
// Sixteen queries keep the causal-tail accumulator in registers.
constexpr std::uint32_t kWmmaQueryRows = 16;
constexpr std::uint32_t kWmmaKeys = 16;
constexpr std::uint32_t kWmmaRatio = 4;
constexpr std::uint32_t kWmmaKSteps = kWmmaHeadDim / 16;
// Row padding keeps a 16-byte-per-lane fragment read off a single bank group.
constexpr std::uint32_t kWmmaKStride = kWmmaHeadDim + 8;
// Union-mask capacity: one bit per 4-token block, 262,144 tokens of context.
constexpr std::uint32_t kWmmaMaxMaskWords = 2048;

/// Two row layouts share the kernel. The dense window packs 16 queries x 2
/// heads (a row block is 16 queries of one head, grid.y over head pairs).
/// The sparse window packs four queries x twelve heads into three row
/// blocks without padding, with adjacent blocks covering both KV heads: the key
/// tiles are gathered from the union of the block's query selections and
/// four selections overlap far less than 32 (measured at 16k depth: 846
/// versus 2,478 selected blocks against 512 per query).
template<std::uint32_t kQueryRows, std::uint32_t kKeys, bool kPackHeads,
         bool kLateV = false>
__launch_bounds__(256, 2) __global__ void WmmaCausalAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    const std::uint32_t* __restrict__ mask, std::uint32_t mask_words,
    float* __restrict__ out, std::uint32_t start_pos, std::uint32_t n_tokens,
    std::uint32_t first_query_group = 0) {
  // The launcher accepts only four-token selection blocks. Keep that
  // geometry constant throughout the key sweep.
  constexpr std::uint32_t ratio = kWmmaRatio;
  constexpr std::uint32_t kHeadDim = kWmmaHeadDim;
  constexpr std::uint32_t kRowBlocks = kPackHeads
                                           ? (kQueryRows * kWmmaGqa + 15) / 16
                                           : (kQueryRows / 16) * kWmmaHeads;
  static_assert(!kPackHeads || kQueryRows * kWmmaGqa == 48,
                "four queries pack twelve heads into three tiles");
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / 2;
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kOTilesPerWave = (kRowBlocks * kWmmaKSteps) / 8;
  constexpr std::uint32_t kSoftmaxLanes = 4;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles == (kPackHeads ? 3 : 2), "two waves per S tile");
  static_assert(kOTilesPerWave == 2 * kRowBlocks, "O tiles per wave");
  static_assert(kKeys % 16 == 0 && (kPackHeads || kQueryRows % 16 == 0),
                "16-row WMMA tiles");
  static_assert(kKStepsPerWave * 2 == kWmmaKSteps, "k split");
  static_assert(kWmmaKStride % 8 == 0 && kVtStride % 8 == 0,
                "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  // Keep the two KV heads of nearby queries together in the launch order.
  // This improves cache reuse without changing any query's key sweep.
  const std::uint32_t linear = blockIdx.y * gridDim.x + blockIdx.x;
  const std::uint32_t query_group =
      first_query_group + (kPackHeads ? linear / kWmmaKvHeads : blockIdx.x);
  const std::uint32_t query_start = query_group * kQueryRows;
  const std::uint32_t kv_head =
      kPackHeads ? linear % kWmmaKvHeads : blockIdx.y / (kWmmaGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      kPackHeads ? kv_head * kWmmaGqa
                 : (kv_head * kWmmaGqa) +
                       ((blockIdx.y % (kWmmaGqa / kWmmaHeads)) * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  // Row (row block rb, row r) -> (local query, query head, live).
  const auto row_query = [&](std::uint32_t rb, std::uint32_t r) {
    return kPackHeads ? query_start + (rb * 16 + r) / kWmmaGqa
                      : query_start + ((rb / kWmmaHeads) * 16) + r;
  };
  const auto row_head = [&](std::uint32_t rb, std::uint32_t r) {
    return kPackHeads ? first_query_head + (rb * 16 + r) % kWmmaGqa
                      : first_query_head + (rb % kWmmaHeads);
  };
  const auto row_live = [&](std::uint32_t rb, std::uint32_t r) {
    return row_query(rb, r) < n_tokens &&
           (!kPackHeads || rb * 16 + r < kQueryRows * kWmmaGqa);
  };

  constexpr std::uint32_t kKvLdsHalves =
      (kKeys * kWmmaKStride > kHeadDim * kVtStride) ? (kKeys * kWmmaKStride)
                                                    : (kHeadDim * kVtStride);
  __shared__ __half kv_lds[kKvLdsHalves];
  // Pad score rows for the four-lane softmax reads and probability rows
  // for the WMMA fragments. Both transposes otherwise repeat LDS banks.
  __shared__ float s_lds[2][kSTiles][16][17];
  __shared__ __half p_lds[kRows][kKeys + 8];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  if (wave < 2 * kSTiles) {
    const std::uint32_t local_query = row_query(s_rb, sub);
    const bool live = row_live(s_rb, sub);
    const float* q_row =
        q +
        (static_cast<std::size_t>(live ? local_query : 0) * kWmmaAttnWidth) +
        (static_cast<std::size_t>(live ? row_head(s_rb, sub) : 0) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  // This wave owns dim tiles `wave` and `wave + 8` for every row block.
  v8f o_acc[kRowBlocks][2] = {};
  // The same four threads own each softmax row throughout the key sweep.
  // Keep its running statistics in registers; only the final denominator
  // and each tile's rescale factor need to cross waves.
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t context_end = start_pos + n_tokens;
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  // Key tiles are gathered from `kBlocksPerTile` selection blocks (ratio
  // keys each). In the sparse window the blocks come from the union of this
  // block's query masks (every row still applies its own mask below), then
  // every block from the earliest row's tail on; the dense window walks the
  // blocks in order. The sweep therefore costs the union of the selected
  // windows, not the context.
  constexpr std::uint32_t kBlocksPerTile = kKeys / 4;
  static_assert(kBlocksPerTile == 4, "a tile is four selection blocks");
  const std::uint32_t n_blocks = (max_visible + ratio - 1) / ratio;
  const std::uint32_t tail_block =
      ((start_pos + query_start + 1) / ratio);  // first always-visible block
  constexpr bool sparse = kPackHeads;
  // Four 512-block selections plus their incomplete tail. The same LDS
  // stores a mask for arbitrary wider selections used by operator callers.
  constexpr unsigned kListCapacity = 4 * 512 + 4;
  // A compact entry also carries which of the packed queries selected the
  // block, so the softmax reads its mask bits from the tile, not global.
  constexpr unsigned kMaskRows = kPackHeads ? kQueryRows : 1;
  constexpr unsigned kMemberShift = 28;
  static_assert(kMaskRows <= 32 - kMemberShift, "membership bits fit");
  __shared__ unsigned union_words[kListCapacity];
  __shared__ unsigned wave_counts[8];
  bool compact = false;
  unsigned selected = 0;
  if (sparse) {
    // Consecutive word ranges per thread make the prefix scan preserve
    // increasing block order, including ties and the always-visible tail.
    const unsigned live_rows = min(kQueryRows, n_tokens - query_start);
    const unsigned word_count = (n_blocks + 31) / 32;
    const unsigned words_per_thread = (word_count + 255) / 256;
    unsigned local_words[8];
    unsigned row_words[8][kMaskRows];
    unsigned count = 0;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      const unsigned w = tid * words_per_thread + j;
      unsigned bits = 0;
#pragma unroll
      for (unsigned r = 0; r < kMaskRows; ++r)
        row_words[j][r] = 0;
      if (j < words_per_thread && w < word_count) {
#pragma unroll
        for (unsigned r = 0; r < kMaskRows; ++r) {
          if (r < live_rows) {
            row_words[j][r] = mask[size_t(query_start + r) * mask_words + w];
            bits |= row_words[j][r];
          }
        }
        if (w * 32 >= tail_block)
          bits = ~0u;
        else if ((w + 1) * 32 > tail_block)
          bits |= ~0u << (tail_block % 32);
        if ((w + 1) * 32 > n_blocks)
          bits &= (1u << (n_blocks % 32)) - 1u;
      }
      local_words[j] = bits;
      count += __popc(bits);
    }
    unsigned scan = count;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset *= 2) {
      unsigned before = __shfl_up(scan, offset, 32);
      if (lane >= offset)
        scan += before;
    }
    if (lane == 31)
      wave_counts[wave] = scan;
    __syncthreads();
    unsigned prefix = scan - count;
#pragma unroll
    for (unsigned w = 0; w < 8; ++w) {
      if (w < wave)
        prefix += wave_counts[w];
      selected += wave_counts[w];
    }
    compact = selected <= kListCapacity;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      const unsigned word = tid * words_per_thread + j;
      unsigned bits = local_words[j];
      if (compact) {
        while (bits) {
          unsigned bit = __builtin_ctz(bits);
          unsigned members = 0;
#pragma unroll
          for (unsigned r = 0; r < kMaskRows; ++r)
            members |= ((row_words[j][r] >> bit) & 1u) << r;
          union_words[prefix++] = (word * 32 + bit) | (members << kMemberShift);
          bits &= bits - 1;
        }
      } else if (j < words_per_thread && word < word_count) {
        union_words[word] = bits;
      }
    }
    __syncthreads();
  }
  // First visible block at or after `b` (n_blocks when none). Uniform across
  // the block: every thread walks the same words.
  const auto next_block = [&](std::uint32_t b) -> std::uint32_t {
    if (!sparse) {
      return b;
    }
    while (b < tail_block && b < n_blocks) {
      const std::uint32_t bits = union_words[b / 32] >> (b % 32);
      // Never jump past the tail: from tail_block on every block is visible.
      if (bits != 0) {
        return min(b + static_cast<std::uint32_t>(__builtin_ctz(bits)),
                   tail_block);
      }
      b = min(((b / 32) + 1) * 32u, tail_block);
    }
    return b;
  };
  struct Tile {
    std::uint32_t block[kBlocksPerTile];
    std::uint32_t count;
    std::uint32_t members;  // kMaskRows bits per block, compact lists only
  };
  // Gathers the next tile starting the search at block `b`; returns the
  // block to continue from.
  const auto gather_tile = [&](std::uint32_t b, Tile& tile) {
    if (compact) {
      tile.count = min(kBlocksPerTile, selected - min(b, selected));
      tile.members = 0;
#pragma unroll
      for (unsigned i = 0; i < kBlocksPerTile; ++i) {
        const unsigned entry = b + i < selected ? union_words[b + i] : 0u;
        tile.block[i] =
            b + i < selected ? entry & ((1u << kMemberShift) - 1u) : n_blocks;
        tile.members |= (entry >> kMemberShift) << (i * kMaskRows);
      }
      return b + tile.count;
    }
    tile.count = 0;
    tile.members = 0;
#pragma unroll
    for (std::uint32_t i = 0; i < kBlocksPerTile; ++i) {
      b = next_block(b);
      const bool have = b < n_blocks;
      tile.block[i] = have ? b : n_blocks;
      tile.count += have ? 1u : 0u;
      b = have ? b + 1 : b;
    }
    return b;
  };
  // Key position of tile row `r` (0..15), or context_end past the tile.
  const auto tile_key = [&](const Tile& tile, std::uint32_t r) {
    const std::uint32_t i = r / ratio;
    std::uint32_t blk = n_blocks;
#pragma unroll
    for (std::uint32_t j = 0; j < kBlocksPerTile; ++j) {
      if (i == j) {
        blk = tile.block[j];
      }
    }
    return blk < n_blocks ? (blk * ratio) + (r % ratio) : context_end;
  };

  // One key per lane for V, a 16-dim slice per thread: a strided global read
  // in exchange for conflict-free transpose writes.
  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  static_assert(kKRegs * 256 == kKeys * (kHeadDim / 8), "K stages evenly");
  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  const auto* v_base =
      v_cache + (static_cast<std::size_t>(kv_head) * kHeadDim) + v_slice;
  const auto* k_base = k_cache + (static_cast<std::size_t>(kv_head) * kHeadDim);

  const auto load_v = [&](const Tile& tile, uint4* dst) {
    const std::uint32_t key_position = tile_key(tile, v_key);
    const bool live = key_position < context_end;
    const auto* src =
        v_base +
        (static_cast<std::size_t>(live ? key_position : 0) * kWmmaKvWidth);
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      dst[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                    : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  // Coalesced: a wave reads one key row's 512 contiguous bytes.
  const auto load_k = [&](const Tile& tile, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position = tile_key(tile, idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      const bool live = key_position < context_end;
      dst[n] =
          live ? *reinterpret_cast<const uint4*>(
                     k_base +
                     (static_cast<std::size_t>(key_position) * kWmmaKvWidth) +
                     d8)
               : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  uint4 k_cur[kKRegs];
  uint4 v_cur[kVRegs];
  uint4 k_pre[kKRegs];
  uint4 v_pre[kVRegs];
  Tile cur;
  Tile pre;
  std::uint32_t cursor = gather_tile(0, cur);
  if (cur.count != 0) {
    load_k(cur, k_cur);
    load_v(cur, v_cur);
  }

  while (cur.count != 0) {
    // --- stage K from the registers the previous iteration prefetched
    SyncLds();
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kWmmaKStride) + d8]) =
          k_cur[n];
    }
    SyncLds();

    // --- prefetch the next tile. Everything below covers its latency.
    cursor = gather_tile(cursor, pre);
    if (pre.count != 0) {
      load_k(pre, k_pre);
      if constexpr (!kLateV)
        load_v(pre, v_pre);
    }

    // --- S = Q K^T ---
    if (wave < 2 * kSTiles) {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&kv_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = Wmma(q_frag[ks], k_frag, s_acc);
      }
      // Each half writes its own slot; the reader sums them.
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    SyncLds();

    // QK has finished reading K. V and softmax P use separate LDS, so
    // their writes can share the barrier at the end of softmax.
    {
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        const auto* packed = reinterpret_cast<const __half*>(&v_cur[j]);
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          kv_lds[((v_slice + (j * 8) + i) * kVtStride) + v_key] = packed[i];
        }
      }
    }
    if constexpr (kLateV) {
      if (pre.count != 0)
        load_v(pre, v_pre);
    }

    // --- online softmax: kSoftmaxLanes threads per query row ---
    {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      if (rg < kRows) {
        const std::uint32_t seg = tid % kSoftmaxLanes;
        constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
        const std::uint32_t rb = rg / 16;
        const std::uint32_t row = rg % 16;
        const std::uint32_t local_query = row_query(rb, row);
        const bool live_row = row_live(rb, row);
        const std::uint32_t absolute_query = start_pos + local_query;
        // Keys in the query's own incomplete block are always visible; earlier
        // blocks follow the selection mask.
        const std::uint32_t tail_start = ((absolute_query + 1) / ratio) * ratio;
        const std::uint32_t* words =
            mask == nullptr || !live_row
                ? nullptr
                : mask + static_cast<std::size_t>(local_query) * mask_words;
        float part_max = -INFINITY;
        float vals[kPerLane];
#pragma unroll
        for (std::uint32_t m = 0; m < kPerLane; ++m) {
          const std::uint32_t col = (seg * kPerLane) + m;
          const std::uint32_t key_position = tile_key(cur, col);
          bool valid = live_row && key_position <= absolute_query &&
                       key_position < context_end;
          if (valid && words != nullptr && key_position < tail_start) {
            if (compact) {
              const std::uint32_t slot =
                  ((col / ratio) * kMaskRows) + (local_query - query_start);
              valid = ((cur.members >> slot) & 1u) != 0u;
            } else {
              const std::uint32_t b = key_position / ratio;
              valid = ((words[b / 32] >> (b % 32)) & 1u) != 0u;
            }
          }
          const std::uint32_t tile = ((col / 16) * kRowBlocks) + rb;
          vals[m] = valid ? (s_lds[0][tile][row][col % 16] +
                             s_lds[1][tile][row][col % 16])
                          : -INFINITY;
          part_max = fmaxf(part_max, vals[m]);
        }
#pragma unroll
        for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
          part_max = fmaxf(part_max, XorLane(part_max, off));
        }
        const float prev_max = running_max;
        const float next_max = fmaxf(prev_max, part_max);
        const float prior_scale =
            isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
        float part_sum = 0.0F;
#pragma unroll
        for (std::uint32_t m = 0; m < kPerLane; ++m) {
          const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
          part_sum += w;
          p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
        }
#pragma unroll
        for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
          part_sum += XorLane(part_sum, off);
        }
        running_max = next_max;
        running_sum = (running_sum * prior_scale) + part_sum;
        if (seg == 0) {
          row_scale[rg] = prior_scale;
        }
      }
    }
    SyncLds();

    // --- rescale the running O by the new maximum. A lane touches only rows
    // 2i + half_id, so the factors are read once per row block.
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }

    // --- O += P V ---
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
      v16h v_frag[kKeyBlocks];
#pragma unroll
      for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
        v_frag[kb] = LoadFrag(
            &kv_lds[(((dim_tile * 16) + sub) * kVtStride) + (kb * 16)]);
      }
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          v8f next = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
          if constexpr (!kPackHeads) {
            const std::uint32_t first_key = cur.block[0] * ratio;
            // WMMA's dot accumulation can change its last bits when zero
            // probabilities multiply populated future values instead of the
            // zero padding at a chunk boundary. Accumulate each query's final
            // partial tile over its visible keys only. Earlier complete tiles
            // retain the matrix-core path.
            if (first_key + kKeys - 1 > start_pos + query_start) {
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                const std::uint32_t row = (2 * i) + half_id;
                const std::uint32_t absolute_query =
                    start_pos + row_query(rb, row);
                if (absolute_query < first_key + kKeys - 1) {
                  float acc = o_acc[rb][t][i];
                  for (std::uint32_t key = 0;
                       key < kKeys && first_key + key <= absolute_query;
                       ++key) {
                    acc = fmaf(
                        __half2float(p_lds[(rb * 16) + row][key]),
                        __half2float(
                            kv_lds[((dim_tile * 16) + sub) * kVtStride + key]),
                        acc);
                  }
                  next[i] = acc;
                }
              }
            }
          }
          o_acc[rb][t] = next;
        }
      }
    }

#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      k_cur[n] = k_pre[n];
    }
#pragma unroll
    for (std::uint32_t n = 0; n < kVRegs; ++n) {
      v_cur[n] = v_pre[n];
    }
    cur = pre;
  }
  if (tid < kRows * kSoftmaxLanes && tid % kSoftmaxLanes == 0)
    row_sum[tid / kSoftmaxLanes] = running_sum;
  SyncLds();

  // --- epilogue: normalize and apply the sigmoid output gate ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        if (!row_live(rb, row)) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(row_query(rb, row)) * kWmmaAttnWidth) +
            (static_cast<std::size_t>(row_head(rb, row)) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr) {
          value *= SigmoidF(gate[offset]);
        }
        out[offset] = value;
      }
    }
  }
}

/// Gathers `context_len` positions of the paged BF16 K/V caches into
/// contiguous F16 rows [position][kv_head][head_dim]. One thread converts
/// eight consecutive elements of both caches.
__global__ void PagedKvToF16Kernel(const std::uint16_t* __restrict__ key_cache,
                                   const std::uint16_t* __restrict__ value_cache,
                                   const std::int32_t* __restrict__ block_table,
                                   std::int64_t block_size,
                                   std::int64_t first_position,
                                   std::int64_t context_len,
                                   __half* __restrict__ k_out,
                                   __half* __restrict__ v_out) {
  constexpr std::int64_t kRow = kWmmaKvWidth;
  const std::int64_t chunk =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t position = first_position + chunk / (kRow / 8);
  if (position >= context_len) {
    return;
  }
  const std::int64_t d8 = (chunk % (kRow / 8)) * 8;
  const std::int64_t physical = block_table[position / block_size];
  const std::int64_t src =
      ((physical * block_size) + (position % block_size)) * kRow + d8;
  const std::int64_t dst = position * kRow + d8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    k_out[dst + i] = __float2half(__builtin_bit_cast(
        float, static_cast<std::uint32_t>(key_cache[src + i]) << 16U));
    v_out[dst + i] = __float2half(__builtin_bit_cast(
        float, static_cast<std::uint32_t>(value_cache[src + i]) << 16U));
  }
}

/// One block per query row: folds its selected key positions into a bitmask
/// over four-token selection blocks (bit b of the row is block b).
__global__ void SelectionMaskKernel(
    const std::int64_t* __restrict__ selected_positions,
    const std::int32_t* __restrict__ selected_counts,
    std::int64_t selected_stride, std::uint32_t* __restrict__ mask,
    std::uint32_t mask_words) {
  __shared__ unsigned words[kWmmaMaxMaskWords];
  const std::uint32_t row = blockIdx.x;
  for (std::uint32_t w = threadIdx.x; w < mask_words; w += blockDim.x) {
    words[w] = 0u;
  }
  __syncthreads();
  const std::int64_t count = selected_counts[row];
  const std::int64_t* positions =
      selected_positions + static_cast<std::int64_t>(row) * selected_stride;
  for (std::int64_t i = threadIdx.x; i < count; i += blockDim.x) {
    const std::int64_t block = positions[i] / kWmmaRatio;
    if (block >= 0 && block < static_cast<std::int64_t>(mask_words) * 32) {
      atomicOr(&words[block / 32], 1u << (block % 32));
    }
  }
  __syncthreads();
  std::uint32_t* out = mask + static_cast<std::size_t>(row) * mask_words;
  for (std::uint32_t w = threadIdx.x; w < mask_words; w += blockDim.x) {
    out[w] = words[w];
  }
}

}  // namespace

extern "C" int hipengine_qwen4_exp_qsa_wmma_gather_kv_f16(
    const void* key_cache, const void* value_cache, const void* block_table,
    std::int64_t block_size, std::int64_t first_position,
    std::int64_t context_len, void* k_out, void* v_out, hipStream_t stream) {
  if (!key_cache || !value_cache || !block_table || !k_out || !v_out ||
      block_size <= 0 || first_position < 0 || context_len <= first_position) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const std::int64_t chunks =
      (context_len - first_position) * (kWmmaKvWidth / 8);
  hipLaunchKernelGGL(
      PagedKvToF16Kernel,
      dim3(static_cast<unsigned>((chunks + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, stream,
      static_cast<const std::uint16_t*>(key_cache),
      static_cast<const std::uint16_t*>(value_cache),
      static_cast<const std::int32_t*>(block_table), block_size,
      first_position, context_len, static_cast<__half*>(k_out),
      static_cast<__half*>(v_out));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_wmma_selection_mask(
    const void* selected_positions, const void* selected_counts,
    std::int64_t selected_stride, void* mask, std::int64_t rows,
    std::int64_t mask_words, hipStream_t stream) {
  if (!selected_positions || !selected_counts || !mask || rows <= 0 ||
      selected_stride <= 0 || mask_words <= 0 ||
      mask_words > static_cast<std::int64_t>(kWmmaMaxMaskWords)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(SelectionMaskKernel, dim3(static_cast<unsigned>(rows)),
                     dim3(kThreads), 0, stream,
                     static_cast<const std::int64_t*>(selected_positions),
                     static_cast<const std::int32_t*>(selected_counts),
                     selected_stride, static_cast<std::uint32_t*>(mask),
                     static_cast<std::uint32_t>(mask_words));
  return static_cast<int>(hipGetLastError());
}

/// out[row][24 heads][256] for `rows` queries at absolute positions
/// start_position.., FP32 queries and output, contiguous F16 K/V rows.
/// `mask` null runs the dense causal window; otherwise each query sees its
/// masked blocks plus its own incomplete tail block.
extern "C" int hipengine_qwen4_exp_qsa_wmma_attention_f32(
    const float* query, const void* k_f16, const void* v_f16, const void* mask,
    std::int64_t mask_words, float* out, std::int64_t rows,
    std::int64_t start_position, hipStream_t stream) {
  if (!query || !k_f16 || !v_f16 || !out || rows <= 0 || start_position < 0 ||
      (mask != nullptr &&
       (mask_words <= 0 ||
        mask_words > static_cast<std::int64_t>(kWmmaMaxMaskWords)))) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const auto n_tokens = static_cast<std::uint32_t>(rows);
  const auto start = static_cast<std::uint32_t>(start_position);
  const auto* k = static_cast<const __half*>(k_f16);
  const auto* v = static_cast<const __half*>(v_f16);
  if (mask != nullptr) {
    constexpr std::uint32_t kPackedQueries = 4;
    const dim3 grid((n_tokens + kPackedQueries - 1) / kPackedQueries,
                    kWmmaKvHeads);
    hipLaunchKernelGGL(
        (WmmaCausalAttentionKernel<kPackedQueries, kWmmaKeys, true>), grid,
        dim3(kThreads), 0, stream, query, nullptr, k, v,
        static_cast<const std::uint32_t*>(mask),
        static_cast<std::uint32_t>(mask_words), out, start, n_tokens, 0u);
  } else {
    const dim3 grid((n_tokens + kWmmaQueryRows - 1) / kWmmaQueryRows,
                    kWmmaKvHeads * (kWmmaGqa / kWmmaHeads));
    hipLaunchKernelGGL(
        (WmmaCausalAttentionKernel<kWmmaQueryRows, kWmmaKeys, false>), grid,
        dim3(kThreads), 0, stream, query, nullptr, k, v, nullptr, 0u, out,
        start, n_tokens, 0u);
  }
  return static_cast<int>(hipGetLastError());
}
