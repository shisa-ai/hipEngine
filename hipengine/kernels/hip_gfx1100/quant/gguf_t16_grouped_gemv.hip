// Exact rows-1 T16 decode GEMVs with fewer launches and wider loads.
//
// q4_k_t16_tile16_grouped_gemv: one launch computes up to four Q4T16
// projections that share one BF16 input vector and in_features (for example
// attention Q/K/V/gate, or FFN gate/up). One wave owns a full 16-column T16
// tile. Lane t keeps the dense_single_local32 owner's K ownership (eight
// contiguous K per 256-wide superblock), its per-column FMA expression and
// order, and its wave32 reduction tree, so every output is bit-identical to
// q4_k_t16_dense_single_local32_gemv_kernel. The lane's 64 nibble bytes are
// contiguous in the T16 layout and load as four uint4 instead of eight
// strided dwords split across two workgroups.
//
// q6_k_t16_qmicro_planar_hoisted_gemv: the q6_k_t16_qmicro_planar_gemv owner
// (local128 block, K = tid and tid + 128 per superblock, identical per-column
// expression, wave reduction then ordered four-wave sum) with the per-tile d
// values and per-group scales hoisted into registers and each 12-byte qmicro
// record read as three dwords. Bit-identical to the planar owner.
//
// Kernel bodies take raw device pointers; the extern "C" launchers validate
// shapes and pick the wave count.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int T16_COLS = 16;
constexpr int Q4_SUBBLOCKS = 8;
constexpr int Q4_SUBBLOCK = 32;
constexpr int Q6_GROUPS16 = 16;

constexpr int Q4_T16_D_OFFSET = 0;
constexpr int Q4_T16_DMIN_OFFSET = Q4_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_SCALE_OFFSET = Q4_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_MIN_OFFSET = Q4_T16_SCALE_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_Q_OFFSET = Q4_T16_MIN_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_BLOCK_BYTES =
    Q4_T16_Q_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);
static_assert(Q4_T16_Q_OFFSET % 16 == 0, "Q4T16 nibbles must be uint4 aligned");
static_assert(Q4_T16_BLOCK_BYTES % 16 == 0, "Q4T16 tiles must be uint4 aligned");

constexpr int Q6_T16_D_OFFSET = 0;
constexpr int Q6_T16_SCALE_OFFSET = Q6_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q6_T16_QL_OFFSET = Q6_T16_SCALE_OFFSET + Q6_GROUPS16 * T16_COLS;
constexpr int Q6_T16_QH_OFFSET = Q6_T16_QL_OFFSET + QK_K * (T16_COLS / 2);
constexpr int Q6_T16_BLOCK_BYTES = Q6_T16_QH_OFFSET + QK_K * (T16_COLS / 4);
constexpr int Q6_T16_QMICRO_OFFSET = Q6_T16_QL_OFFSET;
constexpr int Q6_T16_QMICRO_RECORD_BYTES = 12;
static_assert(Q6_T16_BLOCK_BYTES % 16 == 0, "Q6T16 tiles must be uint4 aligned");
static_assert(Q6_T16_QMICRO_OFFSET % 4 == 0, "qmicro records must be dword aligned");

constexpr int kMaxSegments = 4;

#define GGUF_T16_WAVE_REDUCE(acc) \
  do {                            \
    acc += __shfl_down(acc, 16);  \
    acc += __shfl_down(acc, 8);   \
    acc += __shfl_down(acc, 4);   \
    acc += __shfl_down(acc, 2);   \
    acc += __shfl_down(acc, 1);   \
  } while (0)

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ inline float fp16_bits_to_float(uint16_t bits) {
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

struct GroupSegment {
  const uint8_t* tiles;
  uint16_t* out;
  int64_t out_tiles;
};

struct GroupArgs {
  GroupSegment seg[kMaxSegments];
  int n_segments;
};

template <int WAVES>
__global__ __launch_bounds__(32 * WAVES)
void q4_k_t16_tile16_grouped_gemv_kernel(
    const uint16_t* __restrict__ x,
    GroupArgs args,
    int64_t in_features) {
  int64_t out_tile = static_cast<int64_t>(blockIdx.x) * WAVES + (threadIdx.x >> 5);
  int seg = 0;
#pragma unroll
  for (int i = 0; i < kMaxSegments - 1; ++i) {
    if (seg < args.n_segments - 1 && out_tile >= args.seg[seg].out_tiles) {
      out_tile -= args.seg[seg].out_tiles;
      ++seg;
    }
  }
  if (out_tile >= args.seg[seg].out_tiles) return;
  const uint8_t* __restrict__ tiles = args.seg[seg].tiles;
  uint16_t* __restrict__ out = args.seg[seg].out;

  const int tid = threadIdx.x & 31;
  const int64_t blocks_per_row = in_features / QK_K;
  const int sb = tid >> 2;
  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q4_T16_BLOCK_BYTES;
    const uint4* __restrict__ q =
        reinterpret_cast<const uint4*>(tile + Q4_T16_Q_OFFSET + tid * 64);
    uint4 qv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = q[i];
    const uint4 xr = *reinterpret_cast<const uint4*>(x + blk * QK_K + tid * 8);

    float scale[T16_COLS];
    float min_value[T16_COLS];
#pragma unroll
    for (int pair = 0; pair < T16_COLS / 2; ++pair) {
      const int col0 = 2 * pair;
      const uint32_t d =
          *reinterpret_cast<const uint32_t*>(tile + Q4_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin =
          *reinterpret_cast<const uint32_t*>(tile + Q4_T16_DMIN_OFFSET + col0 * 2);
      const uint16_t coeff_scale = *reinterpret_cast<const uint16_t*>(
          tile + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
      const uint16_t coeff_min = *reinterpret_cast<const uint16_t*>(
          tile + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
      scale[col0] = fp16_bits_to_float(static_cast<uint16_t>(d)) *
                    static_cast<float>(coeff_scale & 0xFFu);
      scale[col0 + 1] = fp16_bits_to_float(static_cast<uint16_t>(d >> 16)) *
                        static_cast<float>(coeff_scale >> 8);
      min_value[col0] = fp16_bits_to_float(static_cast<uint16_t>(dmin)) *
                        static_cast<float>(coeff_min & 0xFFu);
      min_value[col0 + 1] = fp16_bits_to_float(static_cast<uint16_t>(dmin >> 16)) *
                            static_cast<float>(coeff_min >> 8);
    }

    const uint32_t* qw = reinterpret_cast<const uint32_t*>(qv);
    const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xr);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xv = bf16_bits_to_float(
          static_cast<uint16_t>(xw[j >> 1] >> (16 * (j & 1))));
#pragma unroll
      for (int pair = 0; pair < T16_COLS / 2; ++pair) {
        const uint8_t packed = static_cast<uint8_t>(
            (qw[j * 2 + (pair >> 2)] >> (8 * (pair & 3))) & 0xFFu);
        const int col0 = 2 * pair;
        acc[col0] = fmaf(
            xv,
            static_cast<float>(packed & 0x0Fu) * scale[col0] - min_value[col0],
            acc[col0]);
        acc[col0 + 1] = fmaf(
            xv,
            static_cast<float>(packed >> 4) * scale[col0 + 1] - min_value[col0 + 1],
            acc[col0 + 1]);
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc[col]);
  if (tid == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      out[out_tile * T16_COLS + col] = float_to_bf16_bits(acc[col]);
    }
  }
}

__global__ __launch_bounds__(128, 8)
void q6_k_t16_qmicro_planar_hoisted_gemv_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const uint16_t* __restrict__ x_row = x + row * in_features;
  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    float d[T16_COLS];
    {
      const uint4 d0 = *reinterpret_cast<const uint4*>(tile + Q6_T16_D_OFFSET);
      const uint4 d1 = *reinterpret_cast<const uint4*>(tile + Q6_T16_D_OFFSET + 16);
      const uint32_t dw[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        d[col] = fp16_bits_to_float(static_cast<uint16_t>(dw[col >> 1] >> (16 * (col & 1))));
      }
    }
#pragma unroll
    for (int it = 0; it < QK_K / 128; ++it) {
      const int k_in_block = tid + it * 128;
      const int g16 = k_in_block >> 4;
      const int subblock = k_in_block >> 5;
      const int pack = (k_in_block & 31) >> 2;
      const int k4 = k_in_block & 3;
      const float xv = bf16_bits_to_float(x_row[blk * QK_K + k_in_block]);
      const uint4 scale_vec = *reinterpret_cast<const uint4*>(
          tile + Q6_T16_SCALE_OFFSET + g16 * T16_COLS);
      const int8_t* scales = reinterpret_cast<const int8_t*>(&scale_vec);
#pragma unroll
      for (int col_quartet = 0; col_quartet < T16_COLS / 4; ++col_quartet) {
        const uint32_t* record = reinterpret_cast<const uint32_t*>(
            tile + Q6_T16_QMICRO_OFFSET +
            ((subblock * (T16_COLS / 4) + col_quartet) * 8 + pack) *
                Q6_T16_QMICRO_RECORD_BYTES);
        const uint32_t w0 = record[0];
        const uint32_t w1 = record[1];
        const uint32_t w2 = record[2];
        const uint16_t low_pair = static_cast<uint16_t>(
            ((w0 >> (8 * k4)) & 0xFFu) | (((w1 >> (8 * k4)) & 0xFFu) << 8));
        const uint8_t high_packed = static_cast<uint8_t>(w2 >> (8 * k4));
#pragma unroll
        for (int col4 = 0; col4 < 4; ++col4) {
          const int col = col_quartet * 4 + col4;
          const uint8_t low_byte = static_cast<uint8_t>(low_pair >> (8 * (col4 >> 1)));
          const uint8_t low = (col4 & 1) ? (low_byte >> 4) : (low_byte & 0x0Fu);
          const uint8_t high = (high_packed >> (2 * col4)) & 0x03u;
          const int q_i = static_cast<int>(low | (high << 4)) - 32;
          acc[col] += xv * (d[col] * static_cast<float>(static_cast<int>(scales[col]) * q_i));
        }
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc[col]);
  __shared__ float xchg[4 * T16_COLS];
  if ((tid & 31) == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) xchg[(tid >> 5) * T16_COLS + col] = acc[col];
  }
  __syncthreads();
  if (tid == 0) {
    uint16_t* __restrict__ out_row = out + row * out_features + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) total += xchg[g * T16_COLS + col];
      out_row[col] = float_to_bf16_bits(total);
    }
  }
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_t16_tile16_grouped_gemv_bf16_out(
    const void* x,
    const void* const* tiles,
    void* const* outs,
    const int64_t* out_features,
    int n_segments,
    int64_t in_features,
    int waves,
    hipStream_t stream) {
  if (!x || !tiles || !outs || !out_features || n_segments < 1 ||
      n_segments > kMaxSegments || in_features <= 0 || (in_features % QK_K) != 0) {
    return hipErrorInvalidValue;
  }
  GroupArgs args{};
  args.n_segments = n_segments;
  int64_t total_tiles = 0;
  for (int i = 0; i < n_segments; ++i) {
    if (!tiles[i] || !outs[i] || out_features[i] <= 0 || (out_features[i] % T16_COLS) != 0) {
      return hipErrorInvalidValue;
    }
    args.seg[i].tiles = reinterpret_cast<const uint8_t*>(tiles[i]);
    args.seg[i].out = reinterpret_cast<uint16_t*>(outs[i]);
    args.seg[i].out_tiles = out_features[i] / T16_COLS;
    total_tiles += args.seg[i].out_tiles;
  }
  for (int i = n_segments; i < kMaxSegments; ++i) args.seg[i] = args.seg[n_segments - 1];
  const auto* xp = reinterpret_cast<const uint16_t*>(x);
  const auto blocks = [&](int w) {
    return dim3(static_cast<unsigned int>((total_tiles + w - 1) / w));
  };
  switch (waves) {
    case 1:
      hipLaunchKernelGGL((q4_k_t16_tile16_grouped_gemv_kernel<1>), blocks(1), dim3(32), 0,
                         stream, xp, args, in_features);
      break;
    case 2:
      hipLaunchKernelGGL((q4_k_t16_tile16_grouped_gemv_kernel<2>), blocks(2), dim3(64), 0,
                         stream, xp, args, in_features);
      break;
    case 4:
      hipLaunchKernelGGL((q4_k_t16_tile16_grouped_gemv_kernel<4>), blocks(4), dim3(128), 0,
                         stream, xp, args, in_features);
      break;
    case 8:
      hipLaunchKernelGGL((q4_k_t16_tile16_grouped_gemv_kernel<8>), blocks(8), dim3(256), 0,
                         stream, xp, args, in_features);
      break;
    default:
      return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

extern "C" int hipengine_gguf_q6_k_t16_qmicro_planar_hoisted_gemv_bf16_out(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows <= 0 || in_features <= 0 ||
      (in_features % QK_K) != 0 || out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS),
                  static_cast<unsigned int>(rows));
  hipLaunchKernelGGL(q6_k_t16_qmicro_planar_hoisted_gemv_kernel, grid, dim3(128), 0, stream,
                     reinterpret_cast<const uint16_t*>(x),
                     reinterpret_cast<const uint8_t*>(tiles),
                     reinterpret_cast<uint16_t*>(out), rows, in_features, out_features);
  return hipGetLastError();
}
