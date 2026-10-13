// Routed-expert F16 WMMA GEMMs over raw GGUF Q4_K / Q5_K / Q5_1 / Q8_0 weights.
//
// The kernel body (RoutedF16GEMMKernel and its decode helpers) is ported from
// gufo v0.12.0 (commit 88090afb, MIT License, Copyright (c) 2026 gufo
// contributors), src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp.
// hipEngine changes: activations arrive as BF16 rows already grouped by
// expert and are converted to F16 while they are staged in LDS; outputs are
// written as BF16 in the same grouped row order (no row indirection); the
// expert bounds are the int64 grouped prefix; and the tile grid is sized to
// a static bound with the live tile count published by a device map kernel.
//
// The weights are dequantized to F16 right after the LDS read and the whole
// K extent accumulates in F32, so there is no integer risk tracking and no
// repair pass. This changes arithmetic relative to the exact grouped owners.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

constexpr unsigned kThreads = 256;

// GGUF type ids.
enum class WeightType : std::uint32_t {
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
};

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// F16 WMMA fragments (wave32): sixteen halves per lane, eight F32
// accumulators per lane.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5_1Block {
  __half d;
  __half m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q5_1Block) == 24, "block_q5_1 must be 24 bytes");

/// Byte `i` of the 16-byte block header (d, dmin, scales[12]).
__device__ __forceinline__ std::uint32_t HeaderByte(const uint4& h,
                                                    std::uint32_t i) {
  const std::uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
  return (word >> (8U * (i & 3U))) & 0xFFU;
}

/// Round-to-nearest-even F32 to BF16 bits.
__device__ __forceinline__ std::uint16_t F32ToBf16(float value) {
  std::uint32_t bits = __builtin_bit_cast(std::uint32_t, value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

/// Two BF16 values packed in one word, as two F16 values in the same slots.
__device__ __forceinline__ std::uint32_t Bf16x2ToHalves(std::uint32_t pair) {
  const float lo = __builtin_bit_cast(float, pair << 16U);
  const float hi = __builtin_bit_cast(float, pair & 0xFFFF0000U);
  return __builtin_bit_cast(std::uint32_t, __floats2half2_rn(lo, hi));
}

/// Eight BF16 activations as the eight F16 values the WMMA fragment reads.
__device__ __forceinline__ uint4 Bf16x8ToHalves(const uint4& v) {
  return make_uint4(Bf16x2ToHalves(v.x), Bf16x2ToHalves(v.y),
                    Bf16x2ToHalves(v.z), Bf16x2ToHalves(v.w));
}

// A code becomes a half through a byte permute into the mantissa of 1024.0
// (0x6400 | q = 1024 + q exactly for q < 32, the half's unit being 1 there),
// a packed subtract of 1024 (exact), then one packed FMA:
//
//     w = q * scale + bias
//
// with (scale, bias) = (d * sc, -dmin * mn) for Q4_K and (d, m) for Q5_1,
// staged per (row, K block) as a half2.
//
// grid (m / BM, tiles): `tiles[y]` packs the expert in the low 16 bits and
// the token macro tile index in the high 16. The row blocks of one tile are
// consecutive in dispatch order so they share the tile's activations through
// L2. Block (x, y) computes rows x*BM.. of the expert against its grouped
// rows [pad_bounds[e] + j*BN, +BN) and writes them to out_bf16[row] (BF16;
// the paired gate/up instantiation applies SwiGLU and writes the down
// projection's input).
constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

/// block_q5_K: the Q4_K header, 32 high-bit bytes (bit s of byte j is the
/// fifth bit of element j of K block s), then the Q4_K nibble layout.
constexpr std::size_t kQ5KBlockBytes = 176;

template<WeightType kType>
__device__ __forceinline__ std::size_t RoutedF16RowBytes(std::size_t k) {
  return kType == WeightType::kQ4_K   ? (k / 256) * sizeof(Q4KBlock)
         : kType == WeightType::kQ5_K ? (k / 256) * kQ5KBlockBytes
         : kType == WeightType::kQ5_1 ? (k / 32) * sizeof(Q5_1Block)
                                      : (k / 32) * sizeof(Q8_0Block);
}

/// Bit `s` of each of the four bytes of `w`, packed into bits 0-3.
__device__ __forceinline__ std::uint32_t GatherBit(std::uint32_t w, int s) {
  // 0x01020408 moves byte b's bit to bit 24 + b.
  return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}

/// Four packed 5-bit codes: `nib` holds the 4-bit parts one per byte, `bits`
/// bits j..j+3 of the Q5_1 high-bit word, spread to bit 4 of each byte.
__device__ __forceinline__ std::uint32_t SpreadHighBits(std::uint32_t bits) {
  // 0x00204081 = 1 + 2^7 + 2^14 + 2^21: bit b of `bits` lands at 8b, every
  // cross term falls off the 0x01010101 mask.
  return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}

/// Four halves from four code bytes: 1024 + q as F16, minus `magic` (1024,
/// or 1152 for a signed byte carried as q + 128), then the affine.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 magic, __half2 scale2,
                                              __half2 bias2, __half2& lo,
                                              __half2& hi) {
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}

template<WeightType kType, int BM, int BN, int BK, bool kPair = false>
__launch_bounds__(256) __global__
    void RoutedF16GEMMKernel(const void* __restrict__ w,
                             const std::uint16_t* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ n_tiles,
                             const std::int64_t* __restrict__ pad_bounds,
                             std::uint16_t* __restrict__ out_bf16,
                             std::size_t m, std::size_t k,
                             const void* __restrict__ w_up) {
  // The grid is sized to a static tile bound; the map kernel publishes the
  // live count.
  if (static_cast<std::int32_t>(blockIdx.y) >= n_tiles[0]) {
    return;
  }
  static_assert(BM == 128 || BM == 256, "eight waves, 16-row tiles");
  static_assert(!kPair || BM == 128);
  static_assert(BN % 16 == 0 && BN / 16 <= 8);
  static_assert(BK == 2, "one stage is one 32-byte Q4_K nibble group");
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = BM / 128;  // 16-row tiles per wave
  constexpr bool kQ5 = kType == WeightType::kQ5_1;
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  constexpr bool kQ8 = kType == WeightType::kQ8_0;
  constexpr bool kKQuant = kType == WeightType::kQ4_K || kQ5K;
  // 16-byte code chunks per row and stage: Q4_K's nibble pair and Q5_1's
  // two nibble blocks are two, Q8_0's two byte blocks are four.
  constexpr int kChunks = kQ8 ? 2 * BK : BK;

  // LDS plan (bytes): the code plane holds BM rows x kChunks 16-byte chunks
  // with the chunks of nearby rows permuted so a fragment read (one row per
  // lane) covers all bank groups; the activation plane is
  // [kb][16-element quarter][token][16 B] so a fragment read is 256
  // contiguous bytes; the epilogue reuses it all.
  constexpr int kCodeBytes = BM * kChunks * 16;
  constexpr int kHighBytes = (kQ5 || kQ5K) ? BK * BM * 4 : 0;
  constexpr int kScaleBytes = BK * BM * 4;
  // One slot of padding per activation quarter plane: the eight chunks of
  // a token then land on eight bank groups when they are written.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // The epilogue transposes one 16x16 tile per wave through the same
  // bytes (8 KB), which the narrow tile's stages do not reach.
  constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes + kActBytes;
  constexpr int kLdsBytes = kStageBytes > 8 * 1024 ? kStageBytes : 8 * 1024;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_high = reinterpret_cast<std::uint32_t*>(lds + kCodeBytes);
  auto* s_scale =
      reinterpret_cast<std::uint32_t*>(lds + kCodeBytes + kHighBytes);
  auto* s_act =
      reinterpret_cast<uint4*>(lds + kCodeBytes + kHighBytes + kScaleBytes);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = static_cast<int>(pad_bounds[expert]);
  const int bucket_rows =
      static_cast<int>(pad_bounds[expert + 1]) - bucket_begin;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = RoutedF16RowBytes<kType>(k);
  const auto* w_expert = static_cast<const std::uint8_t*>(w) +
                         static_cast<std::size_t>(expert) * m * row_bytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  constexpr int kRows = kPair ? BM / 2 : BM;
  const int r_block = static_cast<int>(blockIdx.x) * kRows;

  // Weight fetch: unit u of a thread is (row = tid / 2 + 128 u, chunk c =
  // tid % 2). Q4_K: the two 16-byte halves of one 32-byte nibble group (two
  // K blocks, low and high nibbles); Q5_1: one 24-byte K block each.
  const int f_c = tid & 1;
  const std::uint8_t* f_ptr[kWaveRowTiles];
  bool f_live[kWaveRowTiles];
  uint4 f_header[kWaveRowTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r =
        r_block + (kPair ? (tid >> 1) % kRows : (tid >> 1) + (u * 128));
    f_live[u] = r < m_i;
    const std::uint8_t* weights = w_expert;
    if constexpr (kPair) {
      if ((tid >> 1) >= kRows) {
        weights = static_cast<const std::uint8_t*>(w_up) +
                  static_cast<std::size_t>(expert) * m * row_bytes;
      }
    }
    f_ptr[u] = weights +
               static_cast<std::size_t>(f_live[u] ? r : (m_i - 1)) * row_bytes;
    f_header[u] = make_uint4(0u, 0u, 0u, 0u);
  }
  // The next stage's weights and activations, fetched one stage ahead. The
  // (scale, bias) pair is derived from the raw header word only when the
  // stage is committed, so nothing waits on the loads before the compute.
  uint4 f_codes[kWaveRowTiles];
  // Paired gate/up tiles reuse the full quantized block over four stages.
  // Keep its remaining codes in registers alongside the cached header.
  uint4 code_cache[kWaveRowTiles][4];
  uint4 f_codes_hi[kWaveRowTiles];  ///< Q8_0: the block's second 16 codes
  uint4 f_qh[kWaveRowTiles][2];     ///< Q5_K: the superblock's high bits
  std::uint32_t f_high[kWaveRowTiles];
  std::uint32_t f_dm[kWaveRowTiles];  ///< Q5_1: d | m; Q8_0: d
  int f_sb32[kWaveRowTiles];          ///< Q4_K: the K block in its superblock
  constexpr int kActFetch = BN <= 64 ? 2 : 4;
  uint4 a_data[kActFetch];

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; each thread fetches consecutive 256-chunk
  // strides, up to four for a 128-token tile.
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256);
  const std::uint16_t* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    // Rows are already grouped by expert: compact row c is input row c.
    const std::int32_t src = (chunk < kActChunks && c_row < bucket_rows)
                                 ? bucket_begin + c_row
                                 : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    // s_act[(kb * 4 + quarter) * kActStride + t]
    a_slot[i] = chunk < kActChunks ? (sub * kActStride) + t : -1;
  }

  const auto swizzle = [](int row, int c) {
    return (row * kChunks) +
           (c ^ (kChunks == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1)));
  };

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      if constexpr (kQ5) {
        const int kb = kb0 + f_c;
        const auto* words = reinterpret_cast<const uint2*>(f_ptr[u]) + (kb * 3);
        const uint2 w0 = words[0];
        const uint2 w1 = words[1];
        const uint2 w2 = words[2];
        f_codes[u] = make_uint4(w1.x, w1.y, w2.x, w2.y);
        f_high[u] = w0.y;
        f_dm[u] = w0.x;
      } else if constexpr (kQ8) {
        // block_q8_0 is 34 bytes, so the code loads are 2-byte aligned.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 34);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        __builtin_memcpy(&f_codes[u], blk + 2, 16);
        __builtin_memcpy(&f_codes_hi[u], blk + 18, 16);
      } else {
        constexpr int kBlockChunks = kQ5K ? 11 : 9;
        constexpr int kCodeChunk = kQ5K ? 3 : 1;
        const int block = kb0 / 8;
        const auto* blk =
            reinterpret_cast<const uint4*>(f_ptr[u]) + (block * kBlockChunks);
        // The K sweep enters a new superblock every eight Q8-sized blocks.
        if (kb0 % 8 == 0) {
          f_header[u] = blk[0];
          if constexpr (kPair) {
#pragma unroll
            for (int group = 0; group < 4; ++group)
              code_cache[u][group] = blk[kCodeChunk + group * 2 + f_c];
          }
          if constexpr (kQ5K) {
            f_qh[u][0] = blk[1];
            f_qh[u][1] = blk[2];
          }
        }
        const int sb32 = (kb0 % 8) + f_c;
        if constexpr (kPair) {
          const int group = sb32 / 2;
          // Select register values explicitly: a runtime array index or C++
          // conditional can become an address select and spill the cache.
          // Three cndmask instructions per word also avoid four ANDs and
          // three ORs. These are wave32 predicates; group is uniform.
          const std::uint32_t low = 0u - (group & 1u);
          const std::uint32_t high = 0u - ((group >> 1u) & 1u);
          const auto select = [](std::uint32_t a, std::uint32_t b,
                                 std::uint32_t mask) {
            std::uint32_t value;
            asm("v_cndmask_b32_e64 %0, %1, %2, %3"
                : "=v"(value)
                : "v"(a), "v"(b), "s"(mask));
            return value;
          };
          const uint4& c0 = code_cache[u][0];
          const uint4& c1 = code_cache[u][1];
          const uint4& c2 = code_cache[u][2];
          const uint4& c3 = code_cache[u][3];
          f_codes[u] = make_uint4(
              select(select(c0.x, c1.x, low), select(c2.x, c3.x, low), high),
              select(select(c0.y, c1.y, low), select(c2.y, c3.y, low), high),
              select(select(c0.z, c1.z, low), select(c2.z, c3.z, low), high),
              select(select(c0.w, c1.w, low), select(c2.w, c3.w, low), high));
        } else {
          f_codes[u] = blk[kCodeChunk + (sb32 / 2) * 2 + f_c];
        }
        f_sb32[u] = sb32;
        if constexpr (kQ5K) {
          // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
          const std::uint32_t qh[8] = {f_qh[u][0].x, f_qh[u][0].y, f_qh[u][0].z,
                                       f_qh[u][0].w, f_qh[u][1].x, f_qh[u][1].y,
                                       f_qh[u][1].z, f_qh[u][1].w};
          std::uint32_t high = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            high |= GatherBit(qh[i], sb32) << (4 * i);
          }
          f_high[u] = high;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      a_data[i] = a_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32))
                      : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      const int row = (tid >> 1) + (u * 128);
      std::uint32_t scale_bias = 0;
      if constexpr (kQ8) {
        s_codes[swizzle(row, 2 * f_c)] = f_codes[u];
        s_codes[swizzle(row, (2 * f_c) + 1)] = f_codes_hi[u];
        scale_bias = f_live[u] ? f_dm[u] : 0U;  // half2 (d, 0)
      } else {
        s_codes[swizzle(row, f_c)] = f_codes[u];
      }
      if constexpr (kQ5K) {
        s_high[(f_c * BM) + row] = f_high[u];
      }
      if constexpr (kQ8) {
      } else if constexpr (kQ5) {
        s_high[(f_c * BM) + row] = f_high[u];
        const __half2 dm = __builtin_bit_cast(__half2, f_dm[u]);
        const float d = f_live[u] ? __low2float(dm) : 0.0F;
        const float mn = f_live[u] ? __high2float(dm) : 0.0F;
        scale_bias =
            __builtin_bit_cast(std::uint32_t, __floats2half2_rn(d, mn));
      } else {
        const int sb32 = f_sb32[u];
        std::uint32_t sc = 0;
        std::uint32_t mn = 0;
        if (sb32 < 4) {
          sc = HeaderByte(f_header[u], 4 + sb32) & 0x3FU;
          mn = HeaderByte(f_header[u], 8 + sb32) & 0x3FU;
        } else {
          sc = (HeaderByte(f_header[u], 8 + sb32) & 0x0FU) |
               ((HeaderByte(f_header[u], sb32) >> 6U) << 4U);
          mn = (HeaderByte(f_header[u], 8 + sb32) >> 4U) |
               ((HeaderByte(f_header[u], 4 + sb32) >> 6U) << 4U);
        }
        const __half2 dm = __builtin_bit_cast(__half2, f_header[u].x);
        const float scale =
            f_live[u] ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
        const float offset =
            f_live[u] ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
        scale_bias = __builtin_bit_cast(std::uint32_t,
                                        __floats2half2_rn(scale, -offset));
      }
      s_scale[(f_c * BM) + row] = scale_bias;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_slot[i] >= 0) {
        s_act[a_slot[i]] = Bf16x8ToHalves(a_data[i]);
      }
    }
  };

  v8f acc[kWaveRowTiles][kTokTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      acc[u][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  const __half2 magic =
      __floats2half2_rn(kQ8 ? -1152.0F : -1024.0F, kQ8 ? -1152.0F : -1024.0F);
  const auto compute_stage = [&]() {
    uint4 raw[kWaveRowTiles][BK];
    if constexpr (!kQ8) {
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
#pragma unroll
        for (int c = 0; c < BK; ++c) {
          raw[u][c] = s_codes[swizzle(row, c)];
        }
      }
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
        const __half2 sb =
            __builtin_bit_cast(__half2, s_scale[(kb * BM) + row]);
        const __half2 scale2 = __low2half2(sb);
        const __half2 bias2 = __high2half2(sb);
        std::uint32_t nib[8];
        if constexpr (kQ8) {
          // Q8_0: the block's 32 signed bytes are chunks 2 kb and 2 kb + 1;
          // flipping the sign bit carries q + 128, which the 1152 magic
          // takes back out.
          const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
          const uint4 c1 = s_codes[swizzle(row, (2 * kb) + 1)];
          const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w,
                                          c1.x, c1.y, c1.z, c1.w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = words[i] ^ 0x80808080U;
          }
        } else if constexpr (kQ5) {
          // Q5_1: K block kb's 16 bytes are chunk kb; elements 0-15 take
          // the low nibbles, 16-31 the high, plus bit j of the high-bit
          // word.
          const uint4 r = raw[u][kb];
          const std::uint32_t high = s_high[(kb * BM) + row];
          const std::uint32_t words[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            nib[i] = (words[i] & 0x0F0F0F0FU) |
                     SpreadHighBits((high >> (4 * i)) & 0xFU);
            nib[4 + i] = ((words[i] >> 4U) & 0x0F0F0F0FU) |
                         SpreadHighBits((high >> (16 + 4 * i)) & 0xFU);
          }
        } else {
          // Q4_K / Q5_K: elements 0-15 of K block kb0 + kb are the low
          // (kb = 0) or high (kb = 1) nibbles of chunk 0, elements 16-31
          // of chunk 1; Q5_K adds bit j of the staged high-bit word.
          const unsigned shift = 4U * static_cast<unsigned>(kb);
          const std::uint32_t words[8] = {raw[u][0].x, raw[u][0].y, raw[u][0].z,
                                          raw[u][0].w, raw[u][1].x, raw[u][1].y,
                                          raw[u][1].z, raw[u][1].w};
          const std::uint32_t high = kQ5K ? s_high[(kb * BM) + row] : 0U;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = (words[i] >> shift) & 0x0F0F0F0FU;
            if constexpr (kQ5K) {
              nib[i] |= SpreadHighBits((high >> (4 * i)) & 0xFU);
            }
          }
        }
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          CodesToHalves(nib[i], magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
        }
        __builtin_memcpy(&a_lo[u], &h[0], 32);
        __builtin_memcpy(&a_hi[u], &h[8], 32);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if constexpr (kPair || ((kQ5 || kQ8) && BN >= 48)) {
          // Keep one token tile's LDS fragments live at a time. Hoisting
          // all eight tiles spills registers and defeats the wider tile's
          // reuse of each weight decode. This is a compiler barrier only.
          asm volatile("" ::: "memory");
        }
        // A short expert bucket has no output in the remaining token
        // tiles, so omit their WMMA work.
        if constexpr ((kPair || ((kQ5 || kQ8) && BN >= 48)) && kTokTiles > 1) {
          if (j >= live_tok_tiles)
            continue;
        }
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          b[q] = frag[q * kActStride];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          acc[u][j] = Wmma(a_lo[u], b_lo, acc[u][j]);
          acc[u][j] = Wmma(a_hi[u], b_hi, acc[u][j]);
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    compute_stage();
    __syncthreads();
  }

  if constexpr (kPair) {
    // Four waves compute gate rows and four compute the matching up rows.
    // Pair them in the existing LDS allocation, keeping the K accumulation
    // order and avoiding the gate's F32 write/read between projections.
    // Two padding floats keep the accumulator scatter off repeated banks.
    constexpr unsigned stride = 18;
    constexpr unsigned plane = 16 * stride;
    static_assert(8 * plane * sizeof(float) <= kLdsBytes);
    float* base = reinterpret_cast<float*>(lds);
    float* scratch = base + wave_id * plane;
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        scratch[sub_lane * stride + 2 * l + half_id] = acc[0][j][l];
      }
      __syncthreads();
#pragma unroll
      for (int unit = 0; unit < 2; ++unit) {
        const int flat = (unit * 256 + tid) * 2;
        const int t = t_local + j * 16 + flat / kRows;
        const int r = flat % kRows;
        if (t < bucket_rows && r_block + r < m_i) {
          const std::int32_t dst = bucket_begin + t;
          {
            const int idx = (r / 16) * plane + (flat / kRows) * stride + r % 16;
            // Preserve the separate projection epilogue's F32 evaluation
            // order before narrowing. Fast-math can otherwise regroup the
            // products and change an F16 rounding tie.
            std::uint16_t values[2];
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              float product = base[idx + v + 4 * plane] * base[idx + v];
              asm volatile("" : "+v"(product));
              float value = product * SigmoidF(base[idx + v]);
              asm volatile("" : "+v"(value));
              values[v] = F32ToBf16(value);
            }
            const auto offset = static_cast<std::size_t>(dst) * m + r_block + r;
            if (m % 2 == 0 && r_block + r + 1 < m_i) {
              *reinterpret_cast<std::uint32_t*>(out_bf16 + offset) =
                  static_cast<std::uint32_t>(values[0]) |
                  (static_cast<std::uint32_t>(values[1]) << 16U);
            } else {
              out_bf16[offset] = values[0];
              if (r_block + r + 1 < m_i)
                out_bf16[offset + 1] = values[1];
            }
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  // One wave writes a complete 128-byte line of F16 output. Padding the
  // shared row by two floats also makes the accumulator scatter conflict-free.
  // Narrow buckets keep the lighter wave-local epilogue below.
  if constexpr (BN >= 48) {
    static_assert(kLdsBytes >= 16 * (BM + 2) * sizeof(float));
    {
      constexpr unsigned stride = BM + 2;
      float* scratch = reinterpret_cast<float*>(lds);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
          for (int l = 0; l < 8; ++l)
            scratch[sub_lane * stride + wave_id * 16 + u * 128 + 2 * l +
                    half_id] = acc[u][j][l];
        }
        __syncthreads();
#pragma unroll
        for (int round = 0; round < 16 * BM / (256 * 2); ++round) {
          const unsigned flat = (round * 256 + tid) * 2;
          const unsigned tr = flat / BM, row = flat % BM;
          const unsigned t = t_local + j * 16 + tr, r = r_block + row;
          if (t < unsigned(bucket_rows) && r < m) {
            const int dst = bucket_begin + static_cast<int>(t);
            {
              float2 v =
                  *reinterpret_cast<const float2*>(scratch + tr * stride + row);
              const size_t o = size_t(dst) * m + r;
              if (m % 2 == 0 && r + 1 < m)
                *reinterpret_cast<std::uint32_t*>(out_bf16 + o) =
                    static_cast<std::uint32_t>(F32ToBf16(v.x)) |
                    (static_cast<std::uint32_t>(F32ToBf16(v.y)) << 16U);
              else {
                out_bf16[o] = F32ToBf16(v.x);
                if (r + 1 < m)
                  out_bf16[o + 1] = F32ToBf16(v.y);
              }
            }
          }
        }
        __syncthreads();
      }
      return;
    }
  }
  // Transpose each 16x16 tile through LDS, then scatter the 16 rows of each
  // token to its output row.
  float* tile_scratch = reinterpret_cast<float*>(lds) + (wave_id * 256);
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r0 = r_block + (wave_id * 16) + (u * 128);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[u][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const int t0 = t_local + (j * 16);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = (s * 32) + lane_id;
        const int t = t0 + (flat >> 4);
        const int r = r0 + (flat & 15);
        if (t < bucket_rows && r < m_i) {
          const std::int32_t dst = bucket_begin + t;
          {
            const std::size_t o = (static_cast<std::size_t>(dst) * m) +
                                  static_cast<std::size_t>(r);
            out_bf16[o] = F32ToBf16(tile_scratch[flat]);
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

/// One block: expands the grouped prefix into (expert, macro tile) entries
/// of `tile_rows` grouped rows each and publishes the live tile count.
__global__ void RoutedTileMapKernel(const std::int64_t* __restrict__ bounds,
                                    std::int32_t* __restrict__ tiles,
                                    std::int32_t* __restrict__ n_tiles,
                                    std::int32_t n_experts,
                                    std::int32_t tile_rows,
                                    std::int32_t capacity) {
  __shared__ std::int32_t first[1025];
  for (std::int32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    const std::int32_t rows =
        static_cast<std::int32_t>(bounds[e + 1] - bounds[e]);
    first[e + 1] = (rows + tile_rows - 1) / tile_rows;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::int32_t e = 0; e < n_experts; ++e) {
      const std::int32_t count = first[e + 1];
      first[e] = running;
      running += count;
    }
    first[n_experts] = running;
    // A count past the capacity launches no tile: the caller sizes the
    // capacity from the static bound, so this only guards a broken prefix.
    n_tiles[0] = running <= capacity ? running : 0;
  }
  __syncthreads();
  if (first[n_experts] > capacity) {
    return;
  }
  for (std::int32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    const std::int32_t begin = first[e];
    const std::int32_t end = first[e + 1];
    for (std::int32_t j = begin; j < end; ++j) {
      tiles[j] = e | ((j - begin) << 16);
    }
  }
}

template<WeightType kType, int BN>
int LaunchPair(const void* gate, const void* up, const std::uint16_t* x,
               const std::int32_t* tiles, const std::int32_t* n_tiles,
               const std::int64_t* bounds, std::uint16_t* out, std::size_t m,
               std::size_t k, std::uint32_t capacity, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned int>((m + 63) / 64), capacity);
  hipLaunchKernelGGL((RoutedF16GEMMKernel<kType, 128, BN, 2, true>), grid,
                     dim3(kThreads), 0, stream, gate, x, tiles, n_tiles,
                     bounds, out, m, k, up);
  return static_cast<int>(hipGetLastError());
}

template<WeightType kType, int BN>
int LaunchSingle(const void* w, const std::uint16_t* x,
                 const std::int32_t* tiles, const std::int32_t* n_tiles,
                 const std::int64_t* bounds, std::uint16_t* out, std::size_t m,
                 std::size_t k, std::uint32_t capacity, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned int>((m + 127) / 128), capacity);
  hipLaunchKernelGGL((RoutedF16GEMMKernel<kType, 128, BN, 2>), grid,
                     dim3(kThreads), 0, stream, w, x, tiles, n_tiles, bounds,
                     out, m, k, nullptr);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_qwen4_exp_routed_f16_tile_map(
    const std::int64_t* bounds, std::int32_t* tiles, std::int32_t* n_tiles,
    std::int64_t n_experts, std::int64_t tile_rows, std::int64_t capacity,
    hipStream_t stream) {
  if (!bounds || !tiles || !n_tiles || n_experts <= 0 || n_experts > 1024 ||
      tile_rows < 16 || tile_rows % 16 != 0 || capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(RoutedTileMapKernel, dim3(1), dim3(1024), 0, stream,
                     bounds, tiles, n_tiles,
                     static_cast<std::int32_t>(n_experts),
                     static_cast<std::int32_t>(tile_rows),
                     static_cast<std::int32_t>(capacity));
  return static_cast<int>(hipGetLastError());
}

/// out[row][m] = silu(gate x) * (up x) for grouped rows, BF16 in and out.
extern "C" int hipengine_qwen4_exp_routed_f16_gate_up_silu_bf16(
    const void* gate, const void* up, std::int64_t ggml_type,
    const std::uint16_t* x, const std::int32_t* tiles,
    const std::int32_t* n_tiles, const std::int64_t* bounds,
    std::uint16_t* out, std::int64_t m, std::int64_t k,
    std::int64_t tile_rows, std::int64_t capacity, hipStream_t stream) {
  if (!gate || !up || !x || !tiles || !n_tiles || !bounds || !out || m <= 0 ||
      k <= 0 || k % 256 != 0 || capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const auto type = static_cast<WeightType>(ggml_type);
  const auto cap = static_cast<std::uint32_t>(capacity);
  const auto um = static_cast<std::size_t>(m);
  const auto uk = static_cast<std::size_t>(k);
  if (type == WeightType::kQ4_K && tile_rows == 64) {
    return LaunchPair<WeightType::kQ4_K, 64>(gate, up, x, tiles, n_tiles,
                                             bounds, out, um, uk, cap, stream);
  }
  if (type == WeightType::kQ4_K && tile_rows == 128) {
    return LaunchPair<WeightType::kQ4_K, 128>(gate, up, x, tiles, n_tiles,
                                              bounds, out, um, uk, cap, stream);
  }
  if (type == WeightType::kQ5_K && tile_rows == 64) {
    return LaunchPair<WeightType::kQ5_K, 64>(gate, up, x, tiles, n_tiles,
                                             bounds, out, um, uk, cap, stream);
  }
  if (type == WeightType::kQ5_K && tile_rows == 128) {
    return LaunchPair<WeightType::kQ5_K, 128>(gate, up, x, tiles, n_tiles,
                                              bounds, out, um, uk, cap, stream);
  }
  return static_cast<int>(hipErrorInvalidValue);
}

/// out[row][m] = w x for grouped rows, BF16 in and out.
extern "C" int hipengine_qwen4_exp_routed_f16_linear_bf16(
    const void* w, std::int64_t ggml_type, const std::uint16_t* x,
    const std::int32_t* tiles, const std::int32_t* n_tiles,
    const std::int64_t* bounds, std::uint16_t* out, std::int64_t m,
    std::int64_t k, std::int64_t tile_rows, std::int64_t capacity,
    hipStream_t stream) {
  if (!w || !x || !tiles || !n_tiles || !bounds || !out || m <= 0 || k <= 0 ||
      capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const auto type = static_cast<WeightType>(ggml_type);
  const auto cap = static_cast<std::uint32_t>(capacity);
  const auto um = static_cast<std::size_t>(m);
  const auto uk = static_cast<std::size_t>(k);
  if (type == WeightType::kQ5_1 && k % 64 == 0) {
    if (tile_rows == 16)
      return LaunchSingle<WeightType::kQ5_1, 16>(w, x, tiles, n_tiles, bounds,
                                                 out, um, uk, cap, stream);
    if (tile_rows == 48)
      return LaunchSingle<WeightType::kQ5_1, 48>(w, x, tiles, n_tiles, bounds,
                                                 out, um, uk, cap, stream);
    if (tile_rows == 64)
      return LaunchSingle<WeightType::kQ5_1, 64>(w, x, tiles, n_tiles, bounds,
                                                 out, um, uk, cap, stream);
  }
  if (type == WeightType::kQ8_0 && k % 64 == 0) {
    if (tile_rows == 16)
      return LaunchSingle<WeightType::kQ8_0, 16>(w, x, tiles, n_tiles, bounds,
                                                 out, um, uk, cap, stream);
    if (tile_rows == 48)
      return LaunchSingle<WeightType::kQ8_0, 48>(w, x, tiles, n_tiles, bounds,
                                                 out, um, uk, cap, stream);
    if (tile_rows == 64)
      return LaunchSingle<WeightType::kQ8_0, 64>(w, x, tiles, n_tiles, bounds,
                                                 out, um, uk, cap, stream);
  }
  if ((type == WeightType::kQ4_K || type == WeightType::kQ5_K) &&
      k % 256 == 0) {
    const bool q4 = type == WeightType::kQ4_K;
    if (tile_rows == 16)
      return q4 ? LaunchSingle<WeightType::kQ4_K, 16>(w, x, tiles, n_tiles,
                                                      bounds, out, um, uk, cap,
                                                      stream)
                : LaunchSingle<WeightType::kQ5_K, 16>(w, x, tiles, n_tiles,
                                                      bounds, out, um, uk, cap,
                                                      stream);
    if (tile_rows == 48)
      return q4 ? LaunchSingle<WeightType::kQ4_K, 48>(w, x, tiles, n_tiles,
                                                      bounds, out, um, uk, cap,
                                                      stream)
                : LaunchSingle<WeightType::kQ5_K, 48>(w, x, tiles, n_tiles,
                                                      bounds, out, um, uk, cap,
                                                      stream);
  }
  return static_cast<int>(hipErrorInvalidValue);
}
