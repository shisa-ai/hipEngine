// Wide-row Q8_0 prefill GEMM for gfx1100: 128 output columns x 256 rows per
// 256-thread block, with both operands staged in LDS so every weight byte is
// read from global memory once per 256 rows instead of once per 4-32 rows.
//
// Why this kernel exists
// ---------------------
// The Q8_0 prefill projections on the GGUF path spend their time re-reading
// weights. ``gguf_k_gemv.hip``'s coltile family covers 4-8 rows per block, so
// its weight traffic is (rows / row_batch) x (M x K) bytes; at rows=1024 that
// is 7.13 GB for a 27.85 MB weight. The dequantized weight tile here is staged
// once per K-tile in LDS and reused by every one of the 256 rows in the block,
// which drops the same projection to 0.111 GB.
//
// Structure ported from the pinned comparator's dense MMB kernel:
//   ggml/src/ggml-cuda/mmb.cu, ``mmb_dense_kernel<128, 256, 64, 64, 1>``
//   (the WTYPE==1 Q8_0 instantiation selected for this geometry)
//   commit c4aa302294fcd5121af2039fd4d3dee0d472ec03
//
// Deliberate differences from the source:
//   * The activation arrives as f32 and is converted to f16 during LDS
//     staging, instead of a separate pre-pass materializing a bf16 activation
//     tensor. This keeps the registered variant self-contained with an f32
//     ABI and no hidden scratch state. The ``*_f16in_*`` exports take the
//     pre-converted activation instead, for a caller that hoists the
//     conversion out of the K loop; the staging path above is unchanged and
//     remains the registered default.
//   * Operands are f16 (10-bit mantissa) rather than bf16 (7-bit). RDNA3 runs
//     both at the same WMMA rate, so this is strictly better precision for
//     free, and it matches the in-tree f16 WMMA lineage in
//     ``gguf_q8_0_prefill.hip``.
//   * f32 output only; the source's optional bf16 output (Dh) is not ported.
//   * The epilogue stages each wave's 16x16 result through a per-wave LDS
//     region with a wave barrier, rather than the source's block-wide
//     ``__syncthreads()`` pair per output tile.
//
// Tiling: BM output columns, BN rows, BK = 64 K per stage, 256 threads = 8
// waves. LDS holds a dequantized weight tile (BM x 72 halves) and a converted
// activation tile (BN x 72 halves), so the layout's LDS cost is
// (BM + BN) * 144 bytes. wave wm = wave % (BM/WTM) covers a WTM-column half,
// wn = wave / (BM/WTM) a WTN-row half.
//
// The K loop is double-buffered through registers: the next K-tile is loaded
// from global into registers while the current one is consumed from LDS.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef half_t __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

constexpr int Q8_0_BLOCK = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;  // 2-byte fp16 d + 32-byte int8 qs

constexpr int WIDE_THREADS = 256;
constexpr int WIDE_BK = 64;                   // K elements staged per LDS tile
constexpr int WIDE_LDS_STRIDE = WIDE_BK + 8;  // 72 halves; +8 pads bank conflicts
// The tile family (BM, BN, WTM, WTN) is a launch-side choice; see the exports at
// the bottom of this file for the instantiated layouts and their tradeoff.

// LDS byte offsets in this kernel are always a multiple of 16 but not always of
// 32: a row of the staged tiles spans WIDE_LDS_STRIDE * 2 == 144 bytes, and
// 144 mod 32 == 16. Operand fragments are therefore moved as 16-byte half8_t
// pieces, never as a single 32-byte half16_t.

__device__ inline float wide_half_bits_to_float(uint16_t bits) {
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

// One K-tile of one weight row arrives as 68 consecutive Q8_0 bytes held in 17
// little-endian words. Byte i lives in word (i >> 2), byte (i & 3) of that
// word.
//
// The words are passed by value and the array is built inside the dequant so
// that SROA can promote it to registers: an earlier version passed a pointer to
// a caller-built array, which forced 92 bytes/lane of scratch and a global
// round trip per K-tile. The index argument is a constant after the caller's
// loop unrolls, which is what makes ``words[i >> 2]`` a constant index.
__device__ __forceinline__ int wide_stage_byte(const uint32_t (&words)[17], int i) {
  return static_cast<int>((words[i >> 2] >> ((i & 3) * 8)) & 0xFFu);
}

// Dequantize the 68-byte K-tile into 64 f16 values. Layout per Q8_0 block:
// [0..1] fp16 d, [2..33] int8 qs[32]; block 1 starts at byte 34.
//
// Values are packed two-at-a-time into 32-bit LDS writes: 64 individual 16-bit
// stores would be twice the store traffic, and the paired f32->f16 conversion
// is a single instruction on RDNA3.
__device__ __forceinline__ void wide_dq_q8_0_tile(const uint4 w0,
                                                  const uint4 w1,
                                                  const uint4 w3,
                                                  const uint4 w4,
                                                  const uint32_t w2,
                                                  half_t* __restrict__ dst) {
  const uint32_t words[17] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w,
                              w3.x, w3.y, w3.z, w3.w, w4.x, w4.y, w4.z, w4.w, w2};
  const uint16_t d0_bits = static_cast<uint16_t>(wide_stage_byte(words, 0) |
                                                (wide_stage_byte(words, 1) << 8));
  const uint16_t d1_bits = static_cast<uint16_t>(wide_stage_byte(words, 34) |
                                                (wide_stage_byte(words, 35) << 8));
  const float d0 = wide_half_bits_to_float(d0_bits);
  const float d1 = wide_half_bits_to_float(d1_bits);
  uint32_t* out = reinterpret_cast<uint32_t*>(dst);
#pragma unroll
  for (int m = 0; m < Q8_0_BLOCK / 2; ++m) {
    union {
      half_t h[2];
      uint32_t u;
    } p0, p1;
    p0.h[0] = static_cast<half_t>(
        d0 * static_cast<float>(static_cast<int8_t>(wide_stage_byte(words, 2 + 2 * m))));
    p0.h[1] = static_cast<half_t>(
        d0 * static_cast<float>(static_cast<int8_t>(wide_stage_byte(words, 2 + 2 * m + 1))));
    p1.h[0] = static_cast<half_t>(
        d1 * static_cast<float>(static_cast<int8_t>(wide_stage_byte(words, 36 + 2 * m))));
    p1.h[1] = static_cast<half_t>(
        d1 * static_cast<float>(static_cast<int8_t>(wide_stage_byte(words, 36 + 2 * m + 1))));
    out[m] = p0.u;
    out[Q8_0_BLOCK / 2 + m] = p1.u;
  }
}

// Precision-recovering split: scale the residual before the F16 cast to keep
// small residuals out of the F16 subnormal/flush range. Cross products are
// accumulated separately from the large high-high sum, then unscaled once.
constexpr float SPLIT_SCALE = 2048.0f;
constexpr float SPLIT_INV_SCALE = 1.0f / SPLIT_SCALE;

__device__ __forceinline__ half_t wide_split_low(float v, half_t hi) {
  return static_cast<half_t>((v - static_cast<float>(hi)) * SPLIT_SCALE);
}

__device__ __forceinline__ void wide_dq_q8_0_split_tile(
    const uint4 w0, const uint4 w1, const uint4 w3, const uint4 w4,
    const uint32_t w2, half_t* dst_hi, half_t* dst_lo) {
  const uint32_t words[17] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w,
                             w3.x, w3.y, w3.z, w3.w, w4.x, w4.y, w4.z, w4.w, w2};
  const float d0 = wide_half_bits_to_float(static_cast<uint16_t>(
      wide_stage_byte(words, 0) | (wide_stage_byte(words, 1) << 8)));
  const float d1 = wide_half_bits_to_float(static_cast<uint16_t>(
      wide_stage_byte(words, 34) | (wide_stage_byte(words, 35) << 8)));
#pragma unroll
  for (int block = 0; block < 2; ++block) {
#pragma unroll
    for (int m = 0; m < 16; ++m) {
      union { half_t h[2]; uint32_t u; } hi, lo;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float v = (block == 0 ? d0 : d1) * static_cast<float>(
            static_cast<int8_t>(wide_stage_byte(words, block * 34 + 2 + m * 2 + e)));
        hi.h[e] = static_cast<half_t>(v);
        lo.h[e] = wide_split_low(v, hi.h[e]);
      }
      reinterpret_cast<uint32_t*>(dst_hi)[block * 16 + m] = hi.u;
      reinterpret_cast<uint32_t*>(dst_lo)[block * 16 + m] = lo.u;
    }
  }
}

// One 16x16 accumulator fragment to global memory.
//
// The WMMA C fragment puts element e of lane L at (row = 2e + (L >> 4),
// col = L & 15) inside the tile. That is transposed relative to a coalesced
// store, so the tile is bounced through the wave's own 256-float LDS region:
// the fragment is written in its natural layout, then re-read as 16 rows x 16
// columns so lane L stores output row (L >> 1) and the 8 columns starting at
// m_base + (L & 1) * 8.
//
// ``stg`` is this wave's private region, so a wave barrier is sufficient and
// the block-wide barrier the source kernel used is unnecessary.
__device__ inline void wide_store_tile(const float8_t& acc,
                                       float* __restrict__ stg,
                                       float* __restrict__ out,
                                       const int out_features,
                                       const int rows,
                                       const int t0,
                                       const int n_base,
                                       const int m_base,
                                       const int lane) {
  const int cm = lane & 15;
  const int cn = lane >> 4;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    stg[(2 * e + cn) * 16 + cm] = acc[e];
  }
  __syncwarp();

  const int n = lane >> 1;
  const int half = lane & 1;
  const int row = t0 + n_base + n;
  if (row < rows) {
    const float4 v0 = *reinterpret_cast<const float4*>(stg + n * 16 + half * 8);
    const float4 v1 = *reinterpret_cast<const float4*>(stg + n * 16 + half * 8 + 4);
    const size_t base = static_cast<size_t>(row) * out_features + m_base + half * 8;
    if (m_base + 16 <= out_features && (out_features & 3) == 0) {
      *reinterpret_cast<float4*>(out + base) = v0;
      *reinterpret_cast<float4*>(out + base + 4) = v1;
    } else {
      const float values[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        if (m_base + half * 8 + k < out_features) {
          out[base + k] = values[k];
        }
      }
    }
  }
  __syncwarp();
}

// Eight activation values as f16. The f32 form is the ported load path: two
// float4 reads plus eight conversions, staged in registers because eight f32
// values cost 8 VGPRs where eight f16 cost 4. The f16 form is one 16-byte load
// and no conversion, which is what hoisting the conversion out of this loop
// buys; see the ``*_f16in_*`` exports.
//
// The f16 form requires 16-byte alignment. ``off`` is a multiple of 8 halves
// and the row stride ``K`` is a multiple of WIDE_BK, so the address is 16-byte
// aligned whenever the activation base is.
__device__ __forceinline__ half8_t wide_load_act8(const float* p) {
  const float4 lo = *reinterpret_cast<const float4*>(p);
  const float4 hi = *reinterpret_cast<const float4*>(p + 4);
  half8_t h;
  h[0] = static_cast<half_t>(lo.x);
  h[1] = static_cast<half_t>(lo.y);
  h[2] = static_cast<half_t>(lo.z);
  h[3] = static_cast<half_t>(lo.w);
  h[4] = static_cast<half_t>(hi.x);
  h[5] = static_cast<half_t>(hi.y);
  h[6] = static_cast<half_t>(hi.z);
  h[7] = static_cast<half_t>(hi.w);
  return h;
}

__device__ __forceinline__ half8_t wide_load_act8(const half_t* p) {
  return *reinterpret_cast<const half8_t*>(p);
}

// ``x_t`` is the activation element type: ``float`` converts during LDS
// staging, ``half_t`` is an activation already converted by
// ``hipengine_gguf_q8_0_f32_to_f16``. Both paths put the same f16 bytes in LDS
// for the same values, so the matmul and its result are unchanged.
// ``WMMA_REP`` is the DP-3 split-operand probe. A split-f16-operand variant
// represents each f32 operand as an f16 hi/lo pair and needs three WMMA
// products per tile where this kernel issues one. Building the real split is
// "scope work"; before paying for it the question is only whether issuing
// three products per tile is affordable at all. ``WMMA_REP = 3`` issues the
// same product three times on the same operands, which isolates the **WMMA
// issue cost** and nothing else: it does not pay for the extra LDS bytes or
// the splitting arithmetic a real split needs, so its timing is a lower bound
// on the split's cost. A negative here kills the scheme cheaply; a positive
// says the split is worth scoping.
template <typename x_t, int BM, int BN, int WTM, int WTN, bool TAIL,
          int WMMA_REP = 1, bool SPLIT = false>
__device__ __forceinline__ void wide_tile_gemm(
    const uint8_t* __restrict__ w_base,  // first weight row of this block
    const size_t wrow_bytes,
    const int a_rows,                    // valid weight rows remaining
    const x_t* __restrict__ x,           // [rows, K] activation
    const int K,
    float* __restrict__ out,             // [rows, M] f32
    const int M,
    const int rows,
    const int t0,
    const int m0,
    half_t* As,
    half_t* Bs,
    half_t* As_lo = nullptr,
    half_t* Bs_lo = nullptr) {
  constexpr int WAVES_M = BM / WTM;
  constexpr int TM = WTM / 16;
  constexpr int TN = WTN / 16;
  constexpr int A_ITEMS = (BM + WIDE_THREADS - 1) / WIDE_THREADS;
  constexpr int B_ITEMS = (BN * 8) / WIDE_THREADS;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int wm = wave % WAVES_M;
  const int wn = wave / WAVES_M;

  // ---- register staging for the software pipeline ------------------------
  // One weight row's 68 bytes per thread (A_ITEMS == 1 for BM == 128), and
  // B_ITEMS chunks of 8 activation elements.
  //
  // The activation is converted to f16 here rather than in store_lds: staging
  // eight f32 values costs 8 VGPRs where eight f16 values cost 4, and the
  // accumulators already claim 128 VGPRs (BM * BN / WIDE_THREADS). The earlier
  // f32 staging pushed the kernel to 256 VGPRs and spilled 22 of them.
  uint4 a0[A_ITEMS], a1[A_ITEMS], a3[A_ITEMS], a4[A_ITEMS];
  uint32_t a2[A_ITEMS];
  half8_t bst[B_ITEMS];
  half8_t bst_lo[SPLIT ? B_ITEMS : 1];

  int weight_ks = 0;
  auto load_regs = [&](const int ks) {
    weight_ks = ks;
#pragma unroll
    for (int i = 0; i < A_ITEMS; ++i) {
      const int row = tid + i * WIDE_THREADS;
      if (row < BM && row < a_rows) {
        const uint8_t* p = w_base + static_cast<size_t>(row) * wrow_bytes +
                           static_cast<size_t>(ks) * 2 * Q8_0_BLOCK_BYTES;
        a0[i] = *reinterpret_cast<const uint4*>(p);
        a1[i] = *reinterpret_cast<const uint4*>(p + 16);
        a3[i] = *reinterpret_cast<const uint4*>(p + 32);
        a4[i] = *reinterpret_cast<const uint4*>(p + 48);
        a2[i] = *reinterpret_cast<const uint32_t*>(p + 64);
      } else {
        a0[i] = make_uint4(0, 0, 0, 0);
        a1[i] = a0[i];
        a3[i] = a0[i];
        a4[i] = a0[i];
        a2[i] = 0;
      }
    }
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) {
      const int c = tid + i * WIDE_THREADS;
      const int off = (c & 7) * 8;
      const int brow = (t0 + (c >> 3) < rows) ? (t0 + (c >> 3)) : -1;
      half8_t h;
      if (brow >= 0) {
        const x_t* p = x + static_cast<size_t>(brow) * K +
                       static_cast<size_t>(ks) * WIDE_BK + off;
        h = wide_load_act8(p);
        if constexpr (SPLIT) {
#pragma unroll
          for (int e = 0; e < 8; ++e) bst_lo[i][e] = wide_split_low(p[e], h[e]);
        }
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          h[e] = static_cast<half_t>(0.0f);
          if constexpr (SPLIT) bst_lo[i][e] = static_cast<half_t>(0.0f);
        }
      }
      bst[i] = h;
    }
  };

  auto store_lds = [&]() {
    // Weight tile: dequantize the register-resident Q8_0 bytes into As. The
    // 17 words are the same bytes load_regs fetched, so this reads no global
    // memory and the pipeline stays overlapped.
#pragma unroll
    for (int i = 0; i < A_ITEMS; ++i) {
      const int row = tid + i * WIDE_THREADS;
      if (row < BM) {
        if constexpr (SPLIT) {
          wide_dq_q8_0_split_tile(a0[i], a1[i], a3[i], a4[i], a2[i],
                                As + row * WIDE_LDS_STRIDE,
                                As_lo + row * WIDE_LDS_STRIDE);
        } else {
          wide_dq_q8_0_tile(a0[i], a1[i], a3[i], a4[i], a2[i],
                            As + row * WIDE_LDS_STRIDE);
        }
      }
    }
    // Activation tile: the f16 values are already in registers; this is a
    // pure LDS store of eight values per item at a 16-byte-aligned offset.
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) {
      const int c = tid + i * WIDE_THREADS;
      const int off = (c & 7) * 8;
      *reinterpret_cast<half8_t*>(Bs + (c >> 3) * WIDE_LDS_STRIDE + off) = bst[i];
      if constexpr (SPLIT) {
        *reinterpret_cast<half8_t*>(Bs_lo + (c >> 3) * WIDE_LDS_STRIDE + off) = bst_lo[i];
      }
    }
  };

  // ---- accumulators ------------------------------------------------------
  float8_t acc[TM][TN];
  float8_t cross[SPLIT ? TM : 1][SPLIT ? TN : 1];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        acc[i][j][e] = 0.0f;
        if constexpr (SPLIT) cross[i][j][e] = 0.0f;
      }
    }
  }

  bool jact[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    jact[j] = !TAIL || (t0 + wn * WTN + j * 16) < rows;
  }

  const int nks = K / WIDE_BK;
  load_regs(0);
  store_lds();
  __syncthreads();

  for (int ks = 0; ks < nks; ++ks) {
    if (ks + 1 < nks) load_regs(ks + 1);

#pragma unroll
    for (int kk = 0; kk < WIDE_BK; kk += 16) {
      half16_t a[TM], b[TN];
      half16_t a_lo[SPLIT ? TM : 1], b_lo[SPLIT ? TN : 1];
      const int r = lane & 15;
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const half_t* p = As + (wm * WTM + i * 16 + r) * WIDE_LDS_STRIDE + kk;
        const half8_t lo = *reinterpret_cast<const half8_t*>(p);
        const half8_t hi = *reinterpret_cast<const half8_t*>(p + 8);
        a[i] = __builtin_bit_cast(half16_t, (half8_t[2]){lo, hi});
        if constexpr (SPLIT) {
          const half_t* pl = As_lo + (wm * WTM + i * 16 + r) * WIDE_LDS_STRIDE + kk;
          const half8_t l0 = *reinterpret_cast<const half8_t*>(pl);
          const half8_t l1 = *reinterpret_cast<const half8_t*>(pl + 8);
          a_lo[i] = __builtin_bit_cast(half16_t, (half8_t[2]){l0, l1});
        }
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        if (TAIL && !jact[j]) continue;
        const half_t* p = Bs + (wn * WTN + j * 16 + r) * WIDE_LDS_STRIDE + kk;
        const half8_t lo = *reinterpret_cast<const half8_t*>(p);
        const half8_t hi = *reinterpret_cast<const half8_t*>(p + 8);
        b[j] = __builtin_bit_cast(half16_t, (half8_t[2]){lo, hi});
        if constexpr (SPLIT) {
          const half_t* pl = Bs_lo + (wn * WTN + j * 16 + r) * WIDE_LDS_STRIDE + kk;
          const half8_t l0 = *reinterpret_cast<const half8_t*>(pl);
          const half8_t l1 = *reinterpret_cast<const half8_t*>(pl + 8);
          b_lo[j] = __builtin_bit_cast(half16_t, (half8_t[2]){l0, l1});
        }
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          if (TAIL && !jact[j]) continue;
          if constexpr (SPLIT) {
            // gfx11 WMMA's long high-high accumulation loses precision as K
            // grows. Round one 16-term dot independently, then sum the dots
            // with ordinary F32 addition instead of feeding the growing sum
            // back through the matrix accumulator.
            const float8_t zero = {};
            const float8_t dot = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b[j], a[i], zero);
            acc[i][j] += dot;
            cross[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b[j], a_lo[i], cross[i][j]);
            cross[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b_lo[j], a[i], cross[i][j]);
          } else {
#pragma unroll
            for (int rep = 0; rep < WMMA_REP; ++rep) {
              acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b[j], a[i], acc[i][j]);
            }
          }
        }
      }
    }

    __syncthreads();
    if (ks + 1 < nks) store_lds();
    __syncthreads();
  }

  // ---- epilogue ----------------------------------------------------------
  // Bs is 36,864 bytes, so the eight per-wave 1 KB staging regions live there.
  // The loop above ends on a block-wide barrier, so no wave is still reading
  // Bs. ``ok`` and ``jact[j]`` are wave-uniform, so every wave executes the
  // same number of wave barriers.
  float* stg = reinterpret_cast<float*>(Bs) + wave * 256;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int ml = wm * WTM + i * 16;
    const bool ok = !TAIL || (m0 + ml < M);
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      if (ok && (!TAIL || jact[j])) {
        if constexpr (SPLIT) {
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            acc[i][j][e] = __builtin_fmaf(cross[i][j][e], SPLIT_INV_SCALE, acc[i][j][e]);
          }
        }
        wide_store_tile(acc[i][j], stg, out, M, rows, t0,
                        wn * WTN + j * 16, m0 + ml, lane);
      }
    }
  }
}

template <typename x_t, int BM, int BN, int WTM, int WTN, bool TAIL,
          int WMMA_REP = 1>
__global__ void __launch_bounds__(WIDE_THREADS, 1)
q8_0_dense_wide_kernel(const x_t* __restrict__ x,
                       const uint8_t* __restrict__ qweight,
                       float* __restrict__ out,
                       const int rows,
                       const int in_features,
                       const int out_features) {
  __shared__ __align__(16) half_t As[BM * WIDE_LDS_STRIDE];
  __shared__ __align__(16) half_t Bs[BN * WIDE_LDS_STRIDE];

  const int m0 = blockIdx.x * BM;
  const int t0 = blockIdx.y * BN;
  const size_t wrow_bytes =
      static_cast<size_t>(in_features / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;

  wide_tile_gemm<x_t, BM, BN, WTM, WTN, TAIL, WMMA_REP>(
      qweight + static_cast<size_t>(m0) * wrow_bytes, wrow_bytes,
      out_features - m0, x, in_features, out, out_features, rows, t0, m0, As, Bs);
}

// Measurement-only true split sibling. BM64/BN128 keeps doubled operand LDS
// at 55,296 bytes. Existing kernel instantiations retain their original arrays.
template <bool TAIL>
__global__ void __launch_bounds__(WIDE_THREADS, 1)
q8_0_dense_split64x128_kernel(const float* __restrict__ x,
                             const uint8_t* __restrict__ qweight,
                             float* __restrict__ out,
                             const int rows, const int K, const int M) {
  __shared__ __align__(16) half_t As[64 * WIDE_LDS_STRIDE];
  __shared__ __align__(16) half_t Bs[128 * WIDE_LDS_STRIDE];
  __shared__ __align__(16) half_t As_lo[64 * WIDE_LDS_STRIDE];
  __shared__ __align__(16) half_t Bs_lo[128 * WIDE_LDS_STRIDE];
  const int m0 = blockIdx.x * 64;
  const int t0 = blockIdx.y * 128;
  const size_t wb = static_cast<size_t>(K / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
  wide_tile_gemm<float, 64, 128, 32, 32, TAIL, 1, true>(
      qweight + static_cast<size_t>(m0) * wb, wb, M - m0, x, K,
      out, M, rows, t0, m0, As, Bs, As_lo, Bs_lo);
}

// f32 -> f16 activation conversion: the hoisted form of the staging conversion
// above. ``static_cast<half_t>`` rounds the same way in both places, so an
// f16in route consumes exactly the bytes the f32 route would have produced.
__global__ void q8_0_f32_to_f16_kernel(const float* __restrict__ x,
                                       half_t* __restrict__ out,
                                       int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = static_cast<half_t>(x[i]);
  }
}

}  // namespace

template <typename x_t, int BM, int BN, int WTM, int WTN, bool TAIL,
          int WMMA_REP = 1>
int wide_launch_tail(const void* x,
                     const void* qweight,
                     void* out,
                     int64_t rows,
                     int64_t in_features,
                     int64_t out_features,
                     hipStream_t stream) {
  const dim3 grid(static_cast<unsigned int>((out_features + BM - 1) / BM),
                  static_cast<unsigned int>((rows + BN - 1) / BN));
  hipLaunchKernelGGL((q8_0_dense_wide_kernel<x_t, BM, BN, WTM, WTN, TAIL, WMMA_REP>),
                     grid, dim3(WIDE_THREADS), 0, stream,
                     static_cast<const x_t*>(x),
                     static_cast<const uint8_t*>(qweight),
                     static_cast<float*>(out),
                     static_cast<int>(rows),
                     static_cast<int>(in_features),
                     static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

// The tail-free instantiation has no per-fragment predication at all; the tail
// one bounds-checks every load and store. Picking between them at launch keeps
// the common (exactly-divisible) case free of the checks.
template <typename x_t, int BM, int BN, int WTM, int WTN, int WMMA_REP = 1>
int wide_launch(const void* x,
                const void* qweight,
                void* out,
                int64_t rows,
                int64_t in_features,
                int64_t out_features,
                hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || (in_features % WIDE_BK) != 0 || out_features <= 0 ||
      rows > INT32_MAX || in_features > INT32_MAX || out_features > INT32_MAX) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const bool tail = (rows % BN) != 0 || (out_features % BM) != 0;
  return tail ? wide_launch_tail<x_t, BM, BN, WTM, WTN, true, WMMA_REP>(
                    x, qweight, out, rows, in_features, out_features, stream)
              : wide_launch_tail<x_t, BM, BN, WTM, WTN, false, WMMA_REP>(
                    x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q8_0_f32_to_f16(const void* x,
                                              void* out,
                                              int64_t n,
                                              hipStream_t stream) {
  if (x == nullptr || out == nullptr || n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (n + 255) / 256;
  if (blocks > INT32_MAX) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(q8_0_f32_to_f16_kernel,
                     dim3(static_cast<unsigned int>(blocks)), dim3(256), 0,
                     stream, static_cast<const float*>(x),
                     static_cast<half_t*>(out), n);
  return static_cast<int>(hipGetLastError());
}

// The tile family. ``threads`` is accepted for ABI compatibility with the other
// raw-pointer linear wrappers and is ignored: every instantiation runs the fixed
// 256-thread block.
//
// The 128x256 layout is the comparator's own tiling and needs 128 accumulator
// registers per thread (BM * BN / 256), which pushes the kernel to the 256-VGPR
// ceiling and spills. The narrower layouts give up some weight reuse
// (traffic is M * K * rows / BN) for headroom; which one wins is a measurement,
// not an argument.
#define HIPENGINE_DENSE_WIDE_EXPORT(name, x_t, BM, BN, WTM, WTN)               \
  extern "C" int name(const void* x, const void* qweight, void* out,           \
                      int64_t rows, int64_t in_features, int64_t out_features, \
                      int64_t threads, hipStream_t stream) {                   \
    (void)threads;                                                             \
    return wide_launch<x_t, BM, BN, WTM, WTN>(x, qweight, out, rows,           \
                                              in_features, out_features,       \
                                              stream);                         \
  }

// DP-3 probe: the same tile with three WMMA products issued per accumulator
// instead of one. See the ``WMMA_REP`` comment on ``wide_tile_gemm``.
#define HIPENGINE_DENSE_WIDE_EXPORT_REP(name, x_t, BM, BN, WTM, WTN, REP)      \
  extern "C" int name(const void* x, const void* qweight, void* out,           \
                      int64_t rows, int64_t in_features, int64_t out_features, \
                      int64_t threads, hipStream_t stream) {                   \
    (void)threads;                                                             \
    return wide_launch<x_t, BM, BN, WTM, WTN, REP>(x, qweight, out, rows,      \
                                                   in_features, out_features,  \
                                                   stream);                    \
  }

// Use the predicated specialization even on full tiles: the tail-free LLVM
// specialization spills 204 bytes/lane at 256 VGPR, whereas the predicated
// body is scratch-free at 241 static VGPR. This is a resource dispatch choice,
// not a numerical change; full tiles simply pass all of its predicates.
#define HIPENGINE_DENSE_WIDE_EXPORT_SPLIT(name, x_t, BM, BN)                   \
  extern "C" int name(const void* x, const void* qweight, void* out,           \
                      int64_t rows, int64_t in_features, int64_t out_features, \
                      int64_t threads, hipStream_t stream) {                   \
    (void)threads;                                                             \
    if (!x || !qweight || !out || rows <= 0 || in_features <= 0 ||              \
        in_features % WIDE_BK || out_features <= 0 || rows > INT32_MAX ||      \
        in_features > INT32_MAX || out_features > INT32_MAX)                   \
      return static_cast<int>(hipErrorInvalidValue);                           \
    const dim3 grid((out_features + BM - 1) / BM, (rows + BN - 1) / BN);       \
    hipLaunchKernelGGL((q8_0_dense_split64x128_kernel<true>), grid,              \
        dim3(WIDE_THREADS), 0, stream, static_cast<const x_t*>(x),              \
        static_cast<const uint8_t*>(qweight), static_cast<float*>(out),         \
        static_cast<int>(rows), static_cast<int>(in_features),                 \
        static_cast<int>(out_features));                                      \
    return static_cast<int>(hipGetLastError());                                \
  }

HIPENGINE_DENSE_WIDE_EXPORT_SPLIT(
    hipengine_gguf_q8_0_dense_split64x128_f32_f32_out, float, 64, 128)

// ``x_t`` selects the activation ABI. The f32 exports convert during staging;
// the f16in exports consume an activation already converted by
// ``hipengine_gguf_q8_0_f32_to_f16``, which must hold ``rows * in_features``
// f16 values in a 16-byte-aligned buffer.
HIPENGINE_DENSE_WIDE_EXPORT(hipengine_gguf_q8_0_dense_wide256_f32_f32_out,
                            float, 128, 256, 64, 64)
HIPENGINE_DENSE_WIDE_EXPORT(hipengine_gguf_q8_0_dense_wide64x256_f32_f32_out,
                            float, 64, 256, 32, 64)
HIPENGINE_DENSE_WIDE_EXPORT(hipengine_gguf_q8_0_dense_wide128x128_f32_f32_out,
                            float, 128, 128, 64, 32)
HIPENGINE_DENSE_WIDE_EXPORT(hipengine_gguf_q8_0_dense_wide64x128_f32_f32_out,
                            float, 64, 128, 32, 32)
HIPENGINE_DENSE_WIDE_EXPORT(hipengine_gguf_q8_0_dense_wide256_f16in_f32_f32_out,
                            half_t, 128, 256, 64, 64)
HIPENGINE_DENSE_WIDE_EXPORT(hipengine_gguf_q8_0_dense_wide128x128_f16in_f32_f32_out,
                            half_t, 128, 128, 64, 32)
HIPENGINE_DENSE_WIDE_EXPORT_REP(
    hipengine_gguf_q8_0_dense_wide256_split3_f32_f32_out, float, 128, 256, 64,
    64, 3)
