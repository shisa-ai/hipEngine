// LDS-staged f32 tiled GEMM prototype for the hyper-connection GR up
// projection (Q8_0 weights, f32 activation).
//
// Why this exists
// ---------------
// `qwen4exp_gr_projection_screen.py` measured the production fused
// up+sigmoid+mean kernel at 6.538 ms for rows=1024, K=320, N=10240 and the
// `dense_wide256` WMMA tile at 1.419-1.493 ms, a 4.4-4.6x gap, and attributed
// the whole of it to weight traffic: the fused kernel re-reads a 3.48 MB
// weight tensor once per 4-row batch (3.565 GB per launch) because each block
// holds only 8 weight rows, while the wide tile stages a dequantized weight
// tile in LDS once per K-tile and reuses it across every row in the block.
//
// The wide tile is 4.6x faster and fails the production gate (mean KL
// 1.554e-3) because RDNA3 matrix cores take only f16/bf16 operands, so its
// multiply rounds both operands to 2^-11. The traffic mechanism itself does not
// require WMMA: this prototype keeps the wide tile's staging and tiling but
// does the multiply-accumulate in **f32 FMA on f32 operands**, so the only
// arithmetic change against the exact chain is the summation order.
//
// Tiling: BM=128 rows x BN=64 outputs x BK=32 K elements (exactly one Q8_0
// block), 256 threads, 8x4 accumulators per thread. LDS holds an f32 weight
// tile (64x32) and an f32 activation tile (128x32), 25 KB with padding, which
// fits two blocks per CU.
//
// BN=128 with 8x8 blocking was measured and is worse (3.038 ms vs 2.737): the
// 33 KB tile drops residency to one block per CU and the occupancy loss costs
// more than the traffic and FMA-per-LDS-read gains return.
//
// Prototype: no production registration, no default change.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace {

constexpr int Q8_0_BLOCK = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;

__device__ __forceinline__ float grf32_q8_0_scale(const uint8_t* row, int block) {
  uint16_t bits;
  const uint8_t* p = row + block * Q8_0_BLOCK_BYTES;
  bits = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
  return __half2float(*reinterpret_cast<const __half*>(&bits));
}

__device__ __forceinline__ float grf32_q8_0_value(
    const uint8_t* row, int k) {
  const int block = k / Q8_0_BLOCK;
  const int within = k - block * Q8_0_BLOCK;
  const float scale = grf32_q8_0_scale(row, block);
  const int8_t q = static_cast<int8_t>(row[block * Q8_0_BLOCK_BYTES + 2 + within]);
  return scale * static_cast<float>(q);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256)
gr_staged_f32_up_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    float* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  constexpr int BM = 128;
  constexpr int BN = 64;
  constexpr int BK = 32;
  constexpr int ROW_GROUPS = 16;
  constexpr int COL_GROUPS = 16;
  constexpr int ROWS_PER_GROUP = BM / ROW_GROUPS;  // 8
  constexpr int COLS_PER_GROUP = BN / COL_GROUPS;  // 4

  // Padded to BK+1 so the row stride is coprime with the 32 LDS banks: with a
  // stride of BK the sixteen row groups of a wave all land on bank `kk` and
  // serialize 16 deep. Padding makes the x-tile reads conflict-free and leaves
  // the weight-tile reads at two-way.
  __shared__ float w_tile[BN][BK + 1];
  __shared__ float x_tile[BM][BK + 1];

  const int row_base = static_cast<int>(blockIdx.y) * BM;
  const int col_base = static_cast<int>(blockIdx.x) * BN;
  const int tid = static_cast<int>(threadIdx.x);
  const int row_bytes = (in_features / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;

  float acc[ROWS_PER_GROUP][COLS_PER_GROUP];
#pragma unroll
  for (int i = 0; i < ROWS_PER_GROUP; ++i) {
#pragma unroll
    for (int j = 0; j < COLS_PER_GROUP; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  const int n_blocks = in_features / Q8_0_BLOCK;
  for (int block = 0; block < n_blocks; ++block) {
    const int k_base = block * BK;

    // Stage the weight tile: BN rows x BK columns, one Q8_0 block each.
    {
      const int row = tid / BK;
      const int within = tid - row * BK;
#pragma unroll
      for (int r = row; r < BN; r += 256 / BK) {
        const int out_row = col_base + r;
        float value = 0.0f;
        if (out_row < out_features) {
          value = grf32_q8_0_value(qweight + static_cast<int64_t>(out_row) * row_bytes,
                                   k_base + within);
        }
        w_tile[r][within] = value;
      }
    }

    // Stage the activation tile: BM rows x BK columns.
    {
      const int row = tid / BK;
      const int within = tid - row * BK;
#pragma unroll
      for (int r = row; r < BM; r += 256 / BK) {
        const int in_row = row_base + r;
        x_tile[r][within] = in_row < rows
            ? x[static_cast<int64_t>(in_row) * in_features + k_base + within]
            : 0.0f;
      }
    }
    __syncthreads();

    const int row_group = tid / COL_GROUPS;
    const int col_group = tid - row_group * COL_GROUPS;
#pragma unroll 4
    for (int kk = 0; kk < BK; ++kk) {
      float xv[ROWS_PER_GROUP];
#pragma unroll
      for (int i = 0; i < ROWS_PER_GROUP; ++i) {
        xv[i] = x_tile[row_group * ROWS_PER_GROUP + i][kk];
      }
      float wv[COLS_PER_GROUP];
#pragma unroll
      for (int j = 0; j < COLS_PER_GROUP; ++j) {
        wv[j] = w_tile[col_group * COLS_PER_GROUP + j][kk];
      }
#pragma unroll
      for (int i = 0; i < ROWS_PER_GROUP; ++i) {
#pragma unroll
        for (int j = 0; j < COLS_PER_GROUP; ++j) {
          acc[i][j] = fmaf(xv[i], wv[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  const int row_group = tid / COL_GROUPS;
  const int col_group = tid - row_group * COL_GROUPS;
#pragma unroll
  for (int i = 0; i < ROWS_PER_GROUP; ++i) {
    const int row = row_base + row_group * ROWS_PER_GROUP + i;
    if (row >= rows) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < COLS_PER_GROUP; ++j) {
      const int col = col_base + col_group * COLS_PER_GROUP + j;
      if (col < out_features) {
        out[static_cast<int64_t>(row) * out_features + col] = acc[i][j];
      }
    }
  }
}

extern "C" int hipengine_gr_staged_f32_up(
    const float* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || in_features % 32 != 0 || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int BM = 128;
  constexpr int BN = 64;
  dim3 grid(
      static_cast<unsigned int>((out_features + BN - 1) / BN),
      static_cast<unsigned int>((rows + BM - 1) / BM));
  hipLaunchKernelGGL(
      gr_staged_f32_up_kernel, grid, dim3(256), 0, stream, x, qweight, out,
      static_cast<int>(rows), static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}
