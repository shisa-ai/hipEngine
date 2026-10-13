// F16-operand WMMA GEMM for small dense F32/BF16 weight matrices.
//
// out[row][col] = sum_k x[row][k] * weight[col][k], FP32 activations and
// FP32 or BF16 weights rounded to F16 while staged in LDS, F16 WMMA with F32
// accumulation, unrounded FP32 output. 256-thread workgroups compute a
// 128-column x 64-row output tile from BK=32 shared F16 tiles (4x2 waves,
// 32x32 wave tiles). The staging and fragment layout follow
// dense_prefill_wmma_out_kernel in dense_gemv.hip; this kernel differs in its
// operand types, its FP32 output, and in accepting any output width.
//
// This changes arithmetic relative to the scalar FP32 GEMV owners: operands
// are rounded to F16 and the accumulation order is the WMMA order.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

typedef _Float16 half_t;
typedef half_t halfx16_t __attribute__((ext_vector_type(16)));
typedef float float8_t __attribute__((ext_vector_type(8)));

__device__ __forceinline__ half_t to_half(float value) {
  return static_cast<half_t>(value);
}

__device__ __forceinline__ half_t to_half(uint16_t bf16_bits) {
  return static_cast<half_t>(
      __builtin_bit_cast(float, static_cast<uint32_t>(bf16_bits) << 16U));
}

template <typename WeightT, int BM, int BN, int BK>
__launch_bounds__(256, 2)
__global__ void dense_f16_gemm_f32_out_kernel(
    const float* __restrict__ x,
    const WeightT* __restrict__ weight,
    float* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  constexpr int S_STRIDE = BK + 8;
  constexpr int WARPS_M = 4;
  constexpr int WARP_TILE_M = BM / WARPS_M;
  constexpr int WARP_TILE_N = BN / 2;
  constexpr int WM = WARP_TILE_M / 16;
  constexpr int WN = WARP_TILE_N / 16;
  constexpr int A_ELEMS = BM * BK;
  constexpr int B_ELEMS = BN * BK;
  __shared__ half_t sA[BM][S_STRIDE];  // weights: output columns x k
  __shared__ half_t sB[BN][S_STRIDE];  // activations: rows x k

  const int tid = threadIdx.x;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BM;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * BN;
  if (col0 >= out_features || row0 >= rows) return;

  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warp_m = warp & 3;
  const int warp_n = warp >> 2;
  // Waves whose 32 output columns are all past the matrix skip the WMMA
  // work; narrow matrices (a handful of columns) then cost one wave pair.
  const bool wave_live = col0 + warp_m * WARP_TILE_M < out_features;

  float8_t acc[WM][WN];
  #pragma unroll
  for (int fm = 0; fm < WM; ++fm) {
    #pragma unroll
    for (int fn = 0; fn < WN; ++fn) {
      acc[fm][fn] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }

  for (int64_t k0 = 0; k0 < in_features; k0 += BK) {
    __syncthreads();
    for (int idx = tid; idx < A_ELEMS; idx += 256) {
      const int c = idx / BK;
      const int k = idx - c * BK;
      const int64_t col = col0 + c;
      sA[c][k] = col < out_features
                     ? to_half(weight[col * in_features + k0 + k])
                     : static_cast<half_t>(0.0f);
    }
    for (int idx = tid; idx < B_ELEMS; idx += 256) {
      const int r = idx / BK;
      const int k = idx - r * BK;
      const int64_t row = row0 + r;
      sB[r][k] = row < rows ? to_half(x[row * in_features + k0 + k])
                            : static_cast<half_t>(0.0f);
    }
    __syncthreads();
    if (!wave_live) continue;

    #pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      halfx16_t b_frag[WN];
      #pragma unroll
      for (int fn = 0; fn < WN; ++fn) {
        const int r = warp_n * WARP_TILE_N + fn * 16 + (lane & 15);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
          b_frag[fn][i] = sB[r][kk * 16 + i];
        }
      }
      #pragma unroll
      for (int fm = 0; fm < WM; ++fm) {
        const int c = warp_m * WARP_TILE_M + fm * 16 + (lane & 15);
        halfx16_t a_frag;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
          a_frag[i] = sA[c][kk * 16 + i];
        }
        #pragma unroll
        for (int fn = 0; fn < WN; ++fn) {
          acc[fm][fn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a_frag, b_frag[fn], acc[fm][fn]);
        }
      }
    }
  }

  #pragma unroll
  for (int fn = 0; fn < WN; ++fn) {
    const int64_t out_row = row0 + warp_n * WARP_TILE_N + fn * 16 + (lane & 15);
    if (out_row >= rows) continue;
    #pragma unroll
    for (int fm = 0; fm < WM; ++fm) {
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t col =
            col0 + warp_m * WARP_TILE_M + fm * 16 + 2 * j + (lane >> 4);
        if (col < out_features) {
          out[out_row * out_features + col] = acc[fm][fn][j];
        }
      }
    }
  }
}

template <typename WeightT>
int launch(const float* x, const WeightT* weight, float* out, int64_t rows,
           int64_t in_features, int64_t out_features, hipStream_t stream) {
  constexpr int BM = 128;
  constexpr int BN = 64;
  constexpr int BK = 32;
  if (!x || !weight || !out || rows <= 0 || in_features <= 0 ||
      out_features <= 0 || (in_features % BK) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (dense_f16_gemm_f32_out_kernel<WeightT, BM, BN, BK>),
      dim3(static_cast<unsigned int>((out_features + BM - 1) / BM),
           static_cast<unsigned int>((rows + BN - 1) / BN)),
      dim3(256), 0, stream, x, weight, out, rows, in_features, out_features);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_dense_f16_gemm_f32_f32w_f32_out(
    const float* x, const float* weight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch<float>(x, weight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_dense_f16_gemm_f32_bf16w_f32_out(
    const float* x, const uint16_t* weight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch<uint16_t>(x, weight, out, rows, in_features, out_features,
                          stream);
}
