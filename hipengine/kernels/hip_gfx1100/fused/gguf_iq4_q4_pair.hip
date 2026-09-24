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

template <int WAVES>
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

  // Side B: the dense Q4T16 single's full single-wave chain, recomputed
  // in every wave; only wave 0's partial is published below.
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

#pragma unroll
  for (int t = 0; t < COLS; ++t)
    for (int delta = 16; delta; delta >>= 1) {
      acc_a[t] += __shfl_down(acc_a[t], delta);
      acc_b[t] += __shfl_down(acc_b[t], delta);
    }

  __shared__ float xwave_a[WAVES > 1 ? WAVES : 1][COLS];
  __shared__ float xwave_b[WAVES > 1 ? WAVES : 1][COLS];
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < COLS; ++t) {
      xwave_a[wave][t] = acc_a[t];
      xwave_b[wave][t] = acc_b[t];
    }
  }
  __syncthreads();
  if (tid < COLS) {
    const int t = tid;
    if (n0 + t < N) {
      float ta = 0.0f;
#pragma unroll
      for (int wv = 0; wv < WAVES; ++wv) ta += xwave_a[wv][t];
      // Side B publishes wave 0 only: that chain is the single's own.
      const float tb = xwave_b[0][t];
      // bf16-round both accumulators exactly like the unfused path,
      // then the silu_mul kernel's arithmetic verbatim.
      const float ga = bf16_load(bf16_store(ta));
      const float ub = bf16_load(bf16_store(tb));
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
            err = (iq4_q4_pair_silu_kernel<1><<<grid, 32, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 2:
            err = (iq4_q4_pair_silu_kernel<2><<<grid, 64, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 4:
            err = (iq4_q4_pair_silu_kernel<4><<<grid, 128, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        default:
            return int(hipErrorInvalidValue);
    }
}