// E6b-1: exact rows==1 (IQ4_XS gate, Q4_K up) pair + SiLU owner
// (2026-09-25, campaign UD-GFX1151-OPTIMIZE2 E6b). The ordered
// (23, 12) mixed-quant family - 7 gate/up layers in the UD artifact.
//
// Exactness contract: bit-identical to the production unfused chain
// IQ4_XS local32 single + dense Q4T16 single + silu_mul_separate_out.
//   - Side A is the local32 dual's decode verbatim: split-K across the
//     block's waves, lane-grouped contiguous k, wave32 shuffle tree,
//     fixed-order cross-wave sum - the same contract as the session
//     local32 single at the same wave count (shared rule in
//     _local32_waves), so its association matches the single's.
//   - Side B is the dense Q4T16 single's FULL single-wave chain run in
//     every wave, with only wave 0's partial published: the single
//     always accumulates all superblocks in one wave, so taking wave 0's
//     chain keeps side B bit-identical for any WAVES value instead of
//     re-associating K across waves.
//   - Both accumulators bf16-round exactly where silu_mul_separate_out
//     would read the two bf16 buffers, then that kernel's SiLU
//     arithmetic verbatim: ga * (1/(1+expf(-ga))) * ub in f32.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int T16_COLS = 16;
constexpr int Q4_SUBBLOCK = 32;
constexpr int Q4_SUBBLOCKS = 8;
constexpr int Q4_T16_D_OFFSET = 0;
constexpr int Q4_T16_DMIN_OFFSET = Q4_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_SCALE_OFFSET = Q4_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_MIN_OFFSET = Q4_T16_SCALE_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_Q_OFFSET = Q4_T16_MIN_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_BLOCK_BYTES =
    Q4_T16_Q_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);

// Q5_T16 tile layout (E6b-3): same header/scales/mins shape as Q4_T16
// with the 4-bit coeff split into a low nibble plane and a 1-bit high
// plane - copied from gguf_t16_selected_gemv.hip so both owners agree.
constexpr int Q5_T16_D_OFFSET = 0;
constexpr int Q5_T16_DMIN_OFFSET = Q5_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q5_T16_SCALE_OFFSET = Q5_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q5_T16_MIN_OFFSET = Q5_T16_SCALE_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q5_T16_QL_OFFSET = Q5_T16_MIN_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q5_T16_QH_OFFSET = Q5_T16_QL_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);
constexpr int Q5_T16_BLOCK_BYTES = Q5_T16_QH_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 8);

// The tile8 single's exact wave reduction (verbatim shfl sequence).
#define IQ_PAIR_T16_WAVE_REDUCE(acc) \
  do {                               \
    acc += __shfl_down(acc, 16);     \
    acc += __shfl_down(acc, 8);      \
    acc += __shfl_down(acc, 4);      \
    acc += __shfl_down(acc, 2);      \
    acc += __shfl_down(acc, 1);      \
  } while (0)

// Both owners serve eight output columns per block: the local32 dual's
// GGUF_IQ_LOCAL32_COLS and the dense Q4T16 single's kTileCols.
constexpr int COLS = 8;

__device__ __constant__ int8_t iq4_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ float half_load(const uint8_t* p) {
  union {
    uint16_t u;
    half_t h;
  } v;
  v.u = uint16_t(p[0]) | (uint16_t(p[1]) << 8);
  return float(v.h);
}

__device__ inline float fp16_bits_to_float(uint16_t bits) {
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

__device__ inline float q5_fp16_bytes_to_float(const uint8_t* p) {
  return fp16_bits_to_float(
      static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8));
}

// t16_q5_quant verbatim from gguf_t16_selected_gemv.hip: the 5-bit
// coefficient is the low nibble plane plus the high bit plane.
__device__ inline uint8_t iq4_q5_quant(const uint8_t* tile, int sb,
                                       int lane32, int col) {
  const uint8_t low_packed = tile[Q5_T16_QL_OFFSET +
                                  (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
                                  (col >> 1)];
  const uint8_t low = (col & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed = tile[Q5_T16_QH_OFFSET +
                                   (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 8) +
                                   (col >> 3)];
  const uint8_t high = (high_packed >> (col & 7)) & 0x01u;
  return low | (high << 4);
}

__device__ float bf16_load(uint16_t x) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = uint32_t(x) << 16;
  return v.f;
}

__device__ uint16_t bf16_store(float x) {
  union {
    uint32_t u;
    float f;
  } v;
  v.f = x;
  return uint16_t((v.u + 0x7fff + ((v.u >> 16) & 1)) >> 16);
}

// Four template axes: GATE_IS_Q4 picks which side's rounded accumulator
// is the gate - "gate comes from side B", whatever chain side B runs
// (pointer roles never swap; E6b-2's mirror and E6b-5's Q3 pair both
// instantiate it true), B_KIND selects side B's chain (0 = dense Q4T16
// single chain, 1 = E6b-3's wave-distributed tile8 emulation, 2 =
// E6b-5's strict Q3_K per-row GEMV emulation), and A_IS_CHAIN (E6b-4)
// runs side A as the Q4_K T16 tile chain instead of the IQ4_XS local32
// split-K - for (Q4_K gate, Q5_K up) pairs neither side is IQ4, and
// the Q4 gate chain is per-wave-redundant like the other side-B chains
// (the epilogue takes wave 0's published value, not a cross-wave sum).
template <
    int WAVES,
    bool GATE_IS_Q4 = false,
    int B_KIND = 0,
    bool A_IS_CHAIN = false>
__global__ __launch_bounds__(32 * WAVES, 16 / WAVES > 0 ? 16 / WAVES : 1)
void iq4_q4_pair_silu_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ wa,
    const uint8_t* __restrict__ wb,
    uint16_t* __restrict__ out,
    int64_t K,
    int64_t N) {
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int64_t n0 = int64_t(blockIdx.x) * COLS;
  const int64_t blocks_per_row = K / QK_K;

  // Side A geometry (IQ4_XS local32, verbatim from the local32 dual).
  const int64_t col_bytes = blocks_per_row * 136;
  const int g = lane >> 2;
  const int nib = (lane >> 1) & 1;
  const int byteoff = 8 + g * 16 + (lane & 1) * 8;

  // Side B geometry (dense Q4T16 single): this block owns the same eight
  // columns; lane owns k [lane*8, lane*8+8) of every superblock.
  const int64_t out_group = int64_t(blockIdx.x);
  const int64_t out_tile = out_group >> 1;
  const int col_base = static_cast<int>(out_group & 1) * 8;
  const int k_lane_base = lane * 8;
  const int sb = k_lane_base >> 5;

  __shared__ float s_cb[512];
  for (int i = tid; i < 512; i += 32 * WAVES)
    s_cb[i] = float(iq4_values[i >= 256 ? ((i & 255) >> 4) : (i & 15)]);
  __syncthreads();

  float acc_a[COLS];
  float acc_b[COLS];
#pragma unroll
  for (int t = 0; t < COLS; ++t) {
    acc_a[t] = 0.0f;
    acc_b[t] = 0.0f;
  }

  // Side A: split-K across waves - the local32 dual's loop verbatim.
  // A_IS_CHAIN (E6b-4) replaces it with the dense Q4T16 single's full
  // chain over wa's tiles: every wave recomputes the whole row (all
  // waves publish identical wave-0-redundant values, shfl-reduced per
  // wave like side B), and the epilogue takes xwave_a[0].
  __shared__ float xwave_a[WAVES > 1 ? WAVES : 1][COLS];
  __shared__ float xwave_b[WAVES > 1 ? WAVES : 1][COLS];
  if constexpr (A_IS_CHAIN) {
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* tile =
        wa + (out_tile * blocks_per_row + blk) * Q4_T16_BLOCK_BYTES;
    float scale[COLS];
    float min_value[COLS];
#pragma unroll
    for (int local_pair = 0; local_pair < COLS / 2; ++local_pair) {
      const int local_col0 = local_pair * 2;
      const int local_col1 = local_col0 + 1;
      const int col0 = col_base + local_col0;
      const uint32_t d =
          *reinterpret_cast<const uint32_t*>(tile + Q4_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_DMIN_OFFSET + col0 * 2);
      const uint16_t coeff_scale = *reinterpret_cast<const uint16_t*>(
          tile + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
      const uint16_t coeff_min = *reinterpret_cast<const uint16_t*>(
          tile + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
      scale[local_col0] = fp16_bits_to_float(static_cast<uint16_t>(d)) *
                          static_cast<float>(coeff_scale & 0xFFu);
      scale[local_col1] = fp16_bits_to_float(static_cast<uint16_t>(d >> 16)) *
                          static_cast<float>(coeff_scale >> 8);
      min_value[local_col0] = fp16_bits_to_float(static_cast<uint16_t>(dmin)) *
                              static_cast<float>(coeff_min & 0xFFu);
      min_value[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin >> 16)) *
          static_cast<float>(coeff_min >> 8);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int k_in_block = k_lane_base + j;
      const int lane32 = k_in_block & 31;
      const float xv = bf16_load(x[blk * QK_K + k_in_block]);
#pragma unroll
      for (int local_pair = 0; local_pair < COLS / 2; ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const int q_word = Q4_T16_Q_OFFSET +
            (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) + (col_base >> 1);
        const uint32_t packed_word =
            *reinterpret_cast<const uint32_t*>(tile + q_word);
        const uint8_t packed =
            static_cast<uint8_t>((packed_word >> (8 * local_pair)) & 0xFFu);
        acc_a[local_col0] = fmaf(
            xv,
            static_cast<float>(packed & 0x0Fu) * scale[local_col0] -
                min_value[local_col0],
            acc_a[local_col0]);
        acc_a[local_col1] = fmaf(
            xv,
            static_cast<float>(packed >> 4) * scale[local_col1] -
                min_value[local_col1],
            acc_a[local_col1]);
      }
    }
  }
  } else {
  for (int64_t blk = wave; blk < blocks_per_row; blk += WAVES) {
    const uint16_t* xp = x + blk * 256 + lane * 8;
    float xv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) xv[j] = bf16_load(xp[j]);
#pragma unroll
    for (int t = 0; t < COLS; ++t) {
      if (n0 + t >= N) break;
      const uint8_t* ba = wa + (n0 + t) * col_bytes + blk * 136;
      const uint32_t hdr_a = *reinterpret_cast<const uint32_t*>(ba);
      const uint32_t scl_a = *reinterpret_cast<const uint32_t*>(ba + 4);
      const int scale_a =
          (((scl_a >> (8 * (g >> 1) + 4 * (g & 1))) & 15) |
           (((hdr_a >> (16 + 2 * g)) & 3) << 4)) -
          32;
      const float sd_a = half_load(ba) * float(scale_a);
      const uint32_t pa[2] = {
          *reinterpret_cast<const uint32_t*>(ba + byteoff),
          *reinterpret_cast<const uint32_t*>(ba + byteoff + 4)};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int code_a = int((pa[j >> 2] >> (8 * (j & 3))) & 0xffu);
        acc_a[t] += xv[j] * (sd_a * s_cb[code_a + (nib << 8)]);
      }
    }
  }
  }  // end if constexpr (A_IS_CHAIN) else (IQ4 local32 split-K)

  // Side B: B_KIND=0 keeps the dense Q4T16 single's full
  // single-wave chain recomputed in every wave (only wave 0's partial is
  // published below). B_KIND=1 (E6b-3) instead emulates the dense
  // Q5T16 tile8 single's exact 128-thread chain inside wave 0: that
  // single runs 4 wave-groups over strided k (thread (g,l) owns
  // k = g*32+lane and +128 per superblock), wave-reduces each group with
  // the same shfl tree, then sums the four lane-0 results serially in
  // group order. Lane 0 of wave 0 reproduces all of that (group
  // accumulators in the single's fmaf order), so side B stays
  // bit-identical to the tile8 owner for any WAVES value.
  if constexpr (B_KIND == 1) {
    // Wave w emulates single-waves g = wave, wave+WAVES, ... (identity
    // lane mapping within each group: single thread (g, lane) owns
    // k = g*32+lane and +128 per superblock). Each group keeps the
    // single's fmaf order, its own shfl-tree reduce, and lane-0 publish;
    // the four results are summed serially in group order afterwards.
    __shared__ float s_q5r[4][COLS];
    for (int g = wave; g < 4; g += WAVES) {
      float acc[COLS];
#pragma unroll
      for (int t = 0; t < COLS; ++t) acc[t] = 0.0f;
      for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
        const uint8_t* tile =
            wb + (out_tile * blocks_per_row + blk) * Q5_T16_BLOCK_BYTES;
#pragma unroll
        for (int half_k = 0; half_k < 2; ++half_k) {
          const int k_in_block = g * 32 + lane + half_k * 128;
          const int sb5 = k_in_block >> 5;
          const int lane32 = k_in_block & 31;
          const float xv = bf16_load(x[blk * QK_K + k_in_block]);
#pragma unroll
          for (int local_col = 0; local_col < COLS; ++local_col) {
            const int col = col_base + local_col;
            const float d = q5_fp16_bytes_to_float(
                tile + Q5_T16_D_OFFSET + col * 2);
            const float dmin = q5_fp16_bytes_to_float(
                tile + Q5_T16_DMIN_OFFSET + col * 2);
            const float scale = d * static_cast<float>(
                tile[Q5_T16_SCALE_OFFSET + sb5 * T16_COLS + col]);
            const float minv = dmin * static_cast<float>(
                tile[Q5_T16_MIN_OFFSET + sb5 * T16_COLS + col]);
            const float q = static_cast<float>(
                iq4_q5_quant(tile, sb5, lane32, col));
            acc[local_col] += xv * (scale * q - minv);
          }
        }
      }
#pragma unroll
      for (int local_col = 0; local_col < COLS; ++local_col)
        IQ_PAIR_T16_WAVE_REDUCE(acc[local_col]);
      if (lane == 0)
#pragma unroll
        for (int t = 0; t < COLS; ++t) s_q5r[g][t] = acc[t];
    }
    __syncthreads();
    // The single sums the four lane-0 group results serially in g order;
    // thread 0 reproduces that exact association from shared memory.
    if (tid == 0) {
      float total[COLS];
#pragma unroll
      for (int t = 0; t < COLS; ++t) {
        float v = s_q5r[0][t] + s_q5r[1][t];
        v += s_q5r[2][t];
        v += s_q5r[3][t];
        total[t] = v;
        xwave_b[0][t] = v;
      }
    }
  } else if constexpr (B_KIND == 2) {
    // E6b-5: side B = the strict per-row Q3_K GEMV's exact 128-thread
    // tile (gguf_iq_dense_strict_kernel<3, true, 8, 1>), emulated across
    // this pair block's waves. Each hardware wave w serially owns
    // strict-waves sw = w, w+WAVES, ... in lane position (lane l holds
    // strict-tid sw*32+l), so the strict kernel's k-ascending per-thread
    // accumulation, its phase-A (d*(scale-32)) hoist, its wave32
    // shuffle tree, and thread 0's serial wave-0..3 total all reproduce
    // bit-exactly. Only wave 0 publishes (the per-wave-redundant
    // side-B convention shared with the Q4 chain).
    constexpr int Q3_S = 110;          // hmask32 + qs64 + scales12 + d16
    constexpr int Q3_GPB = 16;         // 32-element groups per 256-block
    constexpr int Q3_HDR = COLS * 16;  // phase-A header slots per block
    constexpr int Q3_RB = 4;           // quant blocks staged per window
    const int64_t nblk = blocks_per_row;
    const int64_t col_bytes_q3 = blocks_per_row * Q3_S;
    __shared__ float s_d[Q3_RB][COLS * 16];
    __shared__ float s_q3r[4][COLS];
    float acc3[4][COLS];
#pragma unroll
    for (int sww = 0; sww < 4; ++sww)
#pragma unroll
      for (int t = 0; t < COLS; ++t) acc3[sww][t] = 0.0f;
    for (int64_t rbase = 0; rbase < nblk; rbase += Q3_RB) {
      const int nb = int(nblk - rbase < Q3_RB ? nblk - rbase : Q3_RB);
      // Phase A (verbatim): hoist half_load(b+108) * float(scale-32) per
      // (column, group); the nb*128 slots preload cooperatively.
      for (int slot = tid; slot < nb * Q3_HDR; slot += 32 * WAVES) {
        const int bi = slot / Q3_HDR, hslot = slot % Q3_HDR;
        const int t = hslot / Q3_GPB, g = hslot % Q3_GPB;
        const uint8_t* b =
            wb + (n0 + t) * col_bytes_q3 + (rbase + bi) * Q3_S;
        const int scale =
            ((b[96 + (g & 7)] >> (4 * (g >> 3))) & 15) |
            (((b[104 + (g & 3)] >> (2 * (g >> 2))) & 3) << 4);
        s_d[bi][t * 16 + g] = half_load(b + 108) * float(scale - 32);
      }
      __syncthreads();
      // Phase B (verbatim arithmetic): each lane runs the strict thread
      // sw*32+lane's two contract elements per quant block, k ascending;
      // per-accumulator order is rbase -> bi -> h, matching the strict
      // kernel exactly for every sw it owns.
      for (int sw = wave; sw < 4; sw += WAVES) {
        const int tid3 = sw * 32 + lane;
        for (int bi = 0; bi < nb; ++bi) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int k = int(rbase + bi) * 256 + h * 128 + tid3;
            const float xv = bf16_load(x[k]);
            const int kk = k & 255;
#pragma unroll
            for (int t = 0; t < COLS; ++t) {
              const uint8_t* b =
                  wb + (n0 + t) * col_bytes_q3 + (rbase + bi) * Q3_S;
              int q = (b[32 + (kk >> 7) * 32 + (kk & 31)] >>
                       (2 * ((kk & 127) >> 5))) & 3;
              q -= (b[kk & 31] & (1 << (kk >> 5))) ? 0 : 4;
              const float wv = s_d[bi][t * 16 + (kk >> 4)] * float(q);
              acc3[sw][t] += xv * wv;
            }
          }
        }
      }
      __syncthreads();
    }
    // Strict tail: wave32 shuffle tree per owned strict-wave, lane-0
    // publish, then thread 0's serial wave-0..3 total in index order
    // (total = 0, then total += sums[i] for i = 0..3).
    for (int sw = wave; sw < 4; sw += WAVES) {
#pragma unroll
      for (int t = 0; t < COLS; ++t)
        for (int delta = 16; delta; delta >>= 1)
          acc3[sw][t] += __shfl_down(acc3[sw][t], delta);
      if (lane == 0)
#pragma unroll
        for (int t = 0; t < COLS; ++t) s_q3r[sw][t] = acc3[sw][t];
    }
    __syncthreads();
    if (tid == 0) {
#pragma unroll
      for (int t = 0; t < COLS; ++t) {
        float total = 0.0f;
        for (int i = 0; i < 4; ++i) total += s_q3r[i][t];
        xwave_b[0][t] = total;
      }
    }
  } else {
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* tile =
        wb + (out_tile * blocks_per_row + blk) * Q4_T16_BLOCK_BYTES;
    float scale[COLS];
    float min_value[COLS];
#pragma unroll
    for (int local_pair = 0; local_pair < COLS / 2; ++local_pair) {
      const int local_col0 = local_pair * 2;
      const int local_col1 = local_col0 + 1;
      const int col0 = col_base + local_col0;
      const uint32_t d =
          *reinterpret_cast<const uint32_t*>(tile + Q4_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_DMIN_OFFSET + col0 * 2);
      const uint16_t coeff_scale = *reinterpret_cast<const uint16_t*>(
          tile + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
      const uint16_t coeff_min = *reinterpret_cast<const uint16_t*>(
          tile + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
      scale[local_col0] = fp16_bits_to_float(static_cast<uint16_t>(d)) *
                          static_cast<float>(coeff_scale & 0xFFu);
      scale[local_col1] = fp16_bits_to_float(static_cast<uint16_t>(d >> 16)) *
                          static_cast<float>(coeff_scale >> 8);
      min_value[local_col0] = fp16_bits_to_float(static_cast<uint16_t>(dmin)) *
                              static_cast<float>(coeff_min & 0xFFu);
      min_value[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin >> 16)) *
          static_cast<float>(coeff_min >> 8);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int k_in_block = k_lane_base + j;
      const int lane32 = k_in_block & 31;
      const float xv = bf16_load(x[blk * QK_K + k_in_block]);
#pragma unroll
      for (int local_pair = 0; local_pair < COLS / 2; ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const int q_word = Q4_T16_Q_OFFSET +
            (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) + (col_base >> 1);
        const uint32_t packed_word =
            *reinterpret_cast<const uint32_t*>(tile + q_word);
        const uint8_t packed =
            static_cast<uint8_t>((packed_word >> (8 * local_pair)) & 0xFFu);
        acc_b[local_col0] = fmaf(
            xv,
            static_cast<float>(packed & 0x0Fu) * scale[local_col0] -
                min_value[local_col0],
            acc_b[local_col0]);
        acc_b[local_col1] = fmaf(
            xv,
            static_cast<float>(packed >> 4) * scale[local_col1] -
                min_value[local_col1],
            acc_b[local_col1]);
      }
    }
  }
  }  // end if constexpr (B_KIND) side-B chain

#pragma unroll
  for (int t = 0; t < COLS; ++t)
    for (int delta = 16; delta; delta >>= 1) {
      acc_a[t] += __shfl_down(acc_a[t], delta);
      if constexpr (B_KIND == 0)
        acc_b[t] += __shfl_down(acc_b[t], delta);
    }

  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < COLS; ++t) {
      xwave_a[wave][t] = acc_a[t];
      // The Q5 path already published its wave-0 lane-0 total directly
      // into xwave_b[0]; only the Q4 chain publishes per wave here.
      if constexpr (B_KIND == 0) xwave_b[wave][t] = acc_b[t];
    }
  }
  __syncthreads();
  if (tid < COLS) {
    const int t = tid;
    if (n0 + t < N) {
      float ta;
      if constexpr (A_IS_CHAIN) {
        // Side A's chain is per-wave redundant: wave 0's published value
        // is the single's own result (same convention as side B).
        ta = xwave_a[0][t];
      } else {
        ta = 0.0f;
#pragma unroll
        for (int wv = 0; wv < WAVES; ++wv) ta += xwave_a[wv][t];
      }
      // Side B publishes wave 0 only: that chain is the single's own.
      const float tb = xwave_b[0][t];
      // bf16-round both accumulators exactly like the unfused path,
      // then the silu_mul kernel's arithmetic verbatim. The gate/up
      // roles follow the family: GATE_IS_Q4=false (E6b-1, IQ4 gate)
      // takes the gate from side A's local32 chain; GATE_IS_Q4=true
      // (E6b-2 mirror, Q4 gate) takes it from side B's Q4 chain. The
      // pointer roles never swap - each side keeps its own decode
      // geometry - only which rounded accumulator is the gate changes.
      float ga, ub;
      if constexpr (GATE_IS_Q4) {
        ga = bf16_load(bf16_store(tb));
        ub = bf16_load(bf16_store(ta));
      } else {
        ga = bf16_load(bf16_store(ta));
        ub = bf16_load(bf16_store(tb));
      }
      const float sig = 1.0f / (1.0f + expf(-ga));
      out[n0 + t] = bf16_store(ga * sig * ub);
    }
  }
}

}  // namespace

extern "C" int hipengine_gguf_iq4_q4_pair_silu(const void* x, const void* wa,
    const void* wb, void* out, int64_t rows, int64_t K, int64_t N, int waves,
    void* stream) {
    if (!x || !wa || !wb || !out || rows != 1 || K <= 0 || K % 256 || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + COLS - 1) / COLS));
    const hipStream_t s = (hipStream_t)stream;
    int err;
    switch (waves) {
        case 1:
            err = (iq4_q4_pair_silu_kernel<1, false><<<grid, 32, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 2:
            err = (iq4_q4_pair_silu_kernel<2, false><<<grid, 64, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 4:
            err = (iq4_q4_pair_silu_kernel<4, false><<<grid, 128, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        default:
            return int(hipErrorInvalidValue);
    }
}

// E6b-2 mirror family: (Q4_K gate, IQ4_XS up). Pointer roles keep their
// decode geometry exactly as above - wa is still the IQ4_XS local32
// weight, wb still the Q4_K T16 tiles - only the epilogue takes the gate
// from side B's Q4 chain (GATE_IS_Q4=true). Callers pass (iq4, q4) here;
// the Python wrapper swaps the route's (gate=q4, up=iq4) order to match.
extern "C" int hipengine_gguf_q4_iq4_pair_silu(const void* x, const void* wa,
    const void* wb, void* out, int64_t rows, int64_t K, int64_t N, int waves,
    void* stream) {
    if (!x || !wa || !wb || !out || rows != 1 || K <= 0 || K % 256 || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + COLS - 1) / COLS));
    const hipStream_t s = (hipStream_t)stream;
    int err;
    switch (waves) {
        case 1:
            err = (iq4_q4_pair_silu_kernel<1, true><<<grid, 32, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 2:
            err = (iq4_q4_pair_silu_kernel<2, true><<<grid, 64, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 4:
            err = (iq4_q4_pair_silu_kernel<4, true><<<grid, 128, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        default:
            return int(hipErrorInvalidValue);
    }
}
// E6b-3 family: (IQ4_XS gate, Q5_K up). wa is the IQ4_XS local32 raw
// weight (side A geometry unchanged), wb the Q5_K T16 tiles for the
// tile8-exact side-B chain; the gate stays side A (GATE_IS_Q4=false).
// N must be a multiple of 16: Q5 tiles are resident 16-column tiles.
extern "C" int hipengine_gguf_iq4_q5_pair_silu(const void* x, const void* wa,
    const void* wb, void* out, int64_t rows, int64_t K, int64_t N, int waves,
    void* stream) {
  if (!x || !wa || !wb || !out || rows != 1 || K <= 0 || K % 256 || N <= 0 ||
      N % 16)
    return int(hipErrorInvalidValue);
  const dim3 grid(unsigned((N + COLS - 1) / COLS));
  const hipStream_t s = (hipStream_t)stream;
  int err;
  switch (waves) {
    case 1:
      err = (iq4_q4_pair_silu_kernel<1, false, true><<<grid, 32, 0, s>>>(
                 (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                 (uint16_t*)out, K, N),
             int(hipGetLastError()));
      return err;
    case 2:
      err = (iq4_q4_pair_silu_kernel<2, false, true><<<grid, 64, 0, s>>>(
                 (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                 (uint16_t*)out, K, N),
             int(hipGetLastError()));
      return err;
    case 4:
      err = (iq4_q4_pair_silu_kernel<4, false, true><<<grid, 128, 0, s>>>(
                 (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                 (uint16_t*)out, K, N),
             int(hipGetLastError()));
      return err;
    default:
      return int(hipErrorInvalidValue);
  }
}
// E6b-4 family: (Q4_K gate, Q5_K up). Neither side is IQ4: wa is the
// Q4_K T16 tiles (side A's chain, A_IS_CHAIN=true), wb the Q5_K T16
// tiles (side B's wave-distributed tile8 emulation). Gate-first
// argument order already matches (Q4 gate first), so no reorder.
extern "C" int hipengine_gguf_q4_q5_pair_silu(const void* x, const void* wa,
    const void* wb, void* out, int64_t rows, int64_t K, int64_t N, int waves,
    void* stream) {
    if (!x || !wa || !wb || !out || rows != 1 || K <= 0 || K % 256 || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + COLS - 1) / COLS));
    const hipStream_t s = (hipStream_t)stream;
    int err;
    switch (waves) {
        case 1:
            err = (iq4_q4_pair_silu_kernel<1, false, true, true><<<grid, 32, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 2:
            err = (iq4_q4_pair_silu_kernel<2, false, true, true><<<grid, 64, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 4:
            err = (iq4_q4_pair_silu_kernel<4, false, true, true><<<grid, 128, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        default:
            return int(hipErrorInvalidValue);
    }
}
// E6b-5 family: (Q3_K gate, IQ4_XS up). Route order is gate-first
// (Q3, IQ4), but the C geometry keeps IQ4 in side A's slot: wa is the
// IQ4_XS local32 raw weight (split-K), wb the Q3_K raw blocks (side B's
// strict-GEMV emulation, B_KIND=2); the Python wrapper reorders the
// route's arguments to match, and GATE_IS_Q4=true takes the gate from
// side B (the Q3 chain).
extern "C" int hipengine_gguf_q3_iq4_pair_silu(const void* x, const void* wa,
    const void* wb, void* out, int64_t rows, int64_t K, int64_t N, int waves,
    void* stream) {
    if (!x || !wa || !wb || !out || rows != 1 || K <= 0 || K % 256 || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + COLS - 1) / COLS));
    const hipStream_t s = (hipStream_t)stream;
    int err;
    switch (waves) {
        case 1:
            err = (iq4_q4_pair_silu_kernel<1, true, 2, false><<<grid, 32, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 2:
            err = (iq4_q4_pair_silu_kernel<2, true, 2, false><<<grid, 64, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 4:
            err = (iq4_q4_pair_silu_kernel<4, true, 2, false><<<grid, 128, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        default:
            return int(hipErrorInvalidValue);
    }
}
// E6b-6 family: (Q5_K gate, Q4_K up) - exact role-swap of the E6b-4
// pair, no new arithmetic. Route order is gate-first (Q5, Q4), but the
// C geometry keeps the Q4 chain in side A's slot: wa is the Q4_K T16
// tiles (A_IS_CHAIN, the dense single's chain = the UP), wb the Q5_K
// T16 tiles (B_KIND=1, the tile8 single's exact emulation = the GATE);
// the Python wrapper reorders the route's arguments to match, and
// GATE_IS_Q4=true takes the gate from side B (the tile8 chain). All
// three axes are independent - flags proven exact in E6b-2/E6b-3/E6b-4.
extern "C" int hipengine_gguf_q5_q4_pair_silu(const void* x, const void* wa,
    const void* wb, void* out, int64_t rows, int64_t K, int64_t N, int waves,
    void* stream) {
    if (!x || !wa || !wb || !out || rows != 1 || K <= 0 || K % 256 || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + COLS - 1) / COLS));
    const hipStream_t s = (hipStream_t)stream;
    int err;
    switch (waves) {
        case 1:
            err = (iq4_q4_pair_silu_kernel<1, true, true, true><<<grid, 32, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 2:
            err = (iq4_q4_pair_silu_kernel<2, true, true, true><<<grid, 64, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 4:
            err = (iq4_q4_pair_silu_kernel<4, true, true, true><<<grid, 128, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        default:
            return int(hipErrorInvalidValue);
    }
}
// E6b-7 family: (IQ4_XS gate, Q3_K up) - the original E6b-5 geometry
// (IQ4 in side A's slot, Q3 in side B's) under the OPPOSITE epilogue
// role: route order is gate-first (IQ4, Q3), which already matches the
// C geometry, so no wrapper reorder happens. wa is the IQ4_XS raw
// blocks (local32 split-K = the GATE), wb the Q3_K raw blocks
// (B_KIND=2 strict-GEMV emulation = the UP); GATE_IS_Q4=false takes
// the gate from side A (the IQ4 chain). Arithmetic identical to the
// dormant E6b-5 instance - only the rounded accumulator feeding the
// silu differs, same proven bit-exact contract.
extern "C" int hipengine_gguf_iq4_q3_pair_silu(const void* x, const void* wa,
    const void* wb, void* out, int64_t rows, int64_t K, int64_t N, int waves,
    void* stream) {
    if (!x || !wa || !wb || !out || rows != 1 || K <= 0 || K % 256 || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + COLS - 1) / COLS));
    const hipStream_t s = (hipStream_t)stream;
    int err;
    switch (waves) {
        case 1:
            err = (iq4_q4_pair_silu_kernel<1, false, 2, false><<<grid, 32, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 2:
            err = (iq4_q4_pair_silu_kernel<2, false, 2, false><<<grid, 64, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 4:
            err = (iq4_q4_pair_silu_kernel<4, false, 2, false><<<grid, 128, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        default:
            return int(hipErrorInvalidValue);
    }
}
