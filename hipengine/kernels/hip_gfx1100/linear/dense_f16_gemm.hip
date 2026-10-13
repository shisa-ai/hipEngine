// F16-operand WMMA GEMM for small dense F32/BF16 weight matrices.
//
// out[row][col] = sum_k x[row][k] * weight[col][k], FP32 activations and
// FP32 or BF16 weights rounded to F16 while staged in LDS, F16 WMMA with F32
// accumulation, unrounded FP32 output.
//
// Two kernels compute the same products in the same order. The staged one
// (dense_f16_gemm_staged_kernel) is the launch default: it moves operands as
// four-element vectors, converts them in registers, fetches the next K tile
// while the current one multiplies, and stores each output fragment as
// coalesced rows. The scalar one (dense_f16_gemm_f32_out_kernel) is the
// original and serves K extents that are not a multiple of 64.
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


typedef half_t halfx8_t __attribute__((ext_vector_type(8)));
typedef half_t halfx4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ halfx4_t stage4(const float* p) {
  const float4 v = *reinterpret_cast<const float4*>(p);
  halfx4_t h;
  h[0] = static_cast<half_t>(v.x);
  h[1] = static_cast<half_t>(v.y);
  h[2] = static_cast<half_t>(v.z);
  h[3] = static_cast<half_t>(v.w);
  return h;
}

__device__ __forceinline__ halfx4_t stage4(const uint16_t* p) {
  const ushort4 v = *reinterpret_cast<const ushort4*>(p);
  halfx4_t h;
  h[0] = to_half(v.x);
  h[1] = to_half(v.y);
  h[2] = to_half(v.z);
  h[3] = to_half(v.w);
  return h;
}

__device__ __forceinline__ halfx4_t stage4(const half_t* p) {
  return *reinterpret_cast<const halfx4_t*>(p);
}

// BM output columns x BN rows per 256-thread workgroup, BK = 64. WAVES_M of
// the eight waves split the columns; the rest split the rows. Operand roles in
// the WMMA call match the scalar kernel (A = weights, B = activations), so each
// output is the same sequence of 16-element F16 products accumulated in FP32.
template <typename WeightT, int BM, int BN, int WAVES_M, typename ActT = float>
__launch_bounds__(256, 1)
__global__ void dense_f16_gemm_staged_kernel(
    const ActT* __restrict__ x,
    const WeightT* __restrict__ weight,
    float* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  constexpr int BK = 64;
  constexpr int STRIDE = BK + 8;
  constexpr int WAVES_N = 8 / WAVES_M;
  constexpr int WTM = BM / WAVES_M;
  constexpr int WTN = BN / WAVES_N;
  constexpr int TM = WTM / 16;
  constexpr int TN = WTN / 16;
  constexpr int A_ITEMS = BM * (BK / 4) / 256;
  constexpr int B_ITEMS = BN * (BK / 4) / 256;
  __shared__ __align__(16) half_t sA[BM * STRIDE];
  __shared__ __align__(16) half_t sB[BN * STRIDE];
  __shared__ __align__(16) float stage[8 * 256];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int wm = wave % WAVES_M;
  const int wn = wave / WAVES_M;
  const int col0 = blockIdx.x * BM;
  const int row0 = blockIdx.y * BN;

  halfx4_t areg[A_ITEMS], breg[B_ITEMS];
  auto load_regs = [&](const int k0) {
#pragma unroll
    for (int i = 0; i < A_ITEMS; ++i) {
      const int idx = tid + i * 256;
      const int col = col0 + idx / (BK / 4);
      const int k = k0 + (idx % (BK / 4)) * 4;
      if (col < out_features) {
        areg[i] = stage4(weight + static_cast<size_t>(col) * in_features + k);
      } else {
        areg[i] = halfx4_t{0, 0, 0, 0};
      }
    }
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) {
      const int idx = tid + i * 256;
      const int row = row0 + idx / (BK / 4);
      const int k = k0 + (idx % (BK / 4)) * 4;
      if (row < rows) {
        breg[i] = stage4(x + static_cast<size_t>(row) * in_features + k);
      } else {
        breg[i] = halfx4_t{0, 0, 0, 0};
      }
    }
  };
  auto store_lds = [&]() {
#pragma unroll
    for (int i = 0; i < A_ITEMS; ++i) {
      const int idx = tid + i * 256;
      *reinterpret_cast<halfx4_t*>(
          sA + (idx / (BK / 4)) * STRIDE + (idx % (BK / 4)) * 4) = areg[i];
    }
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) {
      const int idx = tid + i * 256;
      *reinterpret_cast<halfx4_t*>(
          sB + (idx / (BK / 4)) * STRIDE + (idx % (BK / 4)) * 4) = breg[i];
    }
  };

  float8_t acc[TM][TN];
  bool mlive[TM], nlive[TN];
#pragma unroll
  for (int fm = 0; fm < TM; ++fm) {
    mlive[fm] = col0 + wm * WTM + fm * 16 < out_features;
#pragma unroll
    for (int fn = 0; fn < TN; ++fn) {
      acc[fm][fn] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }
#pragma unroll
  for (int fn = 0; fn < TN; ++fn) {
    nlive[fn] = row0 + wn * WTN + fn * 16 < rows;
  }

  const int nks = in_features / BK;
  load_regs(0);
  store_lds();
  __syncthreads();
  for (int ks = 0; ks < nks; ++ks) {
    if (ks + 1 < nks) load_regs((ks + 1) * BK);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      halfx16_t a[TM], b[TN];
      const int r = lane & 15;
#pragma unroll
      for (int fm = 0; fm < TM; ++fm) {
        const half_t* p = sA + (wm * WTM + fm * 16 + r) * STRIDE + kk;
        const halfx8_t lo = *reinterpret_cast<const halfx8_t*>(p);
        const halfx8_t hi = *reinterpret_cast<const halfx8_t*>(p + 8);
        a[fm] = __builtin_bit_cast(halfx16_t, (halfx8_t[2]){lo, hi});
      }
#pragma unroll
      for (int fn = 0; fn < TN; ++fn) {
        const half_t* p = sB + (wn * WTN + fn * 16 + r) * STRIDE + kk;
        const halfx8_t lo = *reinterpret_cast<const halfx8_t*>(p);
        const halfx8_t hi = *reinterpret_cast<const halfx8_t*>(p + 8);
        b[fn] = __builtin_bit_cast(halfx16_t, (halfx8_t[2]){lo, hi});
      }
#pragma unroll
      for (int fm = 0; fm < TM; ++fm) {
        if (!mlive[fm]) continue;
#pragma unroll
        for (int fn = 0; fn < TN; ++fn) {
          if (!nlive[fn]) continue;
          acc[fm][fn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a[fm], b[fn], acc[fm][fn]);
        }
      }
    }
    __syncthreads();
    if (ks + 1 < nks) store_lds();
    __syncthreads();
  }

  // Lane L of a fragment holds row (L & 15), columns 2e + (L >> 4). Bounce the
  // fragment through this wave's LDS region so lane L can store row (L >> 1),
  // eight consecutive columns, as two coalesced four-element writes.
  float* stg = stage + wave * 256;
#pragma unroll
  for (int fm = 0; fm < TM; ++fm) {
    if (!mlive[fm]) continue;
#pragma unroll
    for (int fn = 0; fn < TN; ++fn) {
      if (!nlive[fn]) continue;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        stg[(lane & 15) * 16 + 2 * e + (lane >> 4)] = acc[fm][fn][e];
      }
      __syncwarp();
      const int row = row0 + wn * WTN + fn * 16 + (lane >> 1);
      const int col = col0 + wm * WTM + fm * 16 + (lane & 1) * 8;
      if (row < rows) {
        const float* src = stg + (lane >> 1) * 16 + (lane & 1) * 8;
        const float4 v0 = *reinterpret_cast<const float4*>(src);
        const float4 v1 = *reinterpret_cast<const float4*>(src + 4);
        float* dst = out + static_cast<size_t>(row) * out_features + col;
        if (col + 8 <= out_features) {
          *reinterpret_cast<float4*>(dst) = v0;
          *reinterpret_cast<float4*>(dst + 4) = v1;
        } else {
          const float values[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            if (col + c < out_features) dst[c] = values[c];
          }
        }
      }
      __syncwarp();
    }
  }
}

template <typename WeightT, int BM, int BN, int WAVES_M>
int launch_staged(const float* x, const WeightT* weight, float* out,
                  int64_t rows, int64_t in_features, int64_t out_features,
                  hipStream_t stream) {
  hipLaunchKernelGGL(
      (dense_f16_gemm_staged_kernel<WeightT, BM, BN, WAVES_M>),
      dim3(static_cast<unsigned int>((out_features + BM - 1) / BM),
           static_cast<unsigned int>((rows + BN - 1) / BN)),
      dim3(256), 0, stream, x, weight, out, static_cast<int>(rows),
      static_cast<int>(in_features), static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

// tile: 0 picks by shape; 1 = 128x128, 2 = 128x256, 3 = 64x256 (columns x
// rows); -1 = the scalar kernel.
template <typename WeightT>
int launch(const float* x, const WeightT* weight, float* out, int64_t rows,
           int64_t in_features, int64_t out_features, hipStream_t stream,
           int tile = 0) {
  constexpr int BM = 128;
  constexpr int BN = 64;
  constexpr int BK = 32;
  if (!x || !weight || !out || rows <= 0 || in_features <= 0 ||
      out_features <= 0 || (in_features % BK) != 0 || rows > INT32_MAX ||
      in_features > INT32_MAX || out_features > INT32_MAX) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  // 128x128 measured fastest at every shape tried (4096 rows: 512 columns
  // 1.07 ms against 1.90 for 128x256 and 1.46 for 64x256; 48 columns 0.41 ms).
  if (tile == 0) tile = 1;
  if (tile > 0 && (in_features % 64) == 0) {
    if (tile == 1) {
      return launch_staged<WeightT, 128, 128, 2>(
          x, weight, out, rows, in_features, out_features, stream);
    }
    if (tile == 2) {
      return launch_staged<WeightT, 128, 256, 2>(
          x, weight, out, rows, in_features, out_features, stream);
    }
    return launch_staged<WeightT, 64, 256, 1>(
        x, weight, out, rows, in_features, out_features, stream);
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

// F16 activations and F16 weights: the staged kernel with no conversion.
// tile: 1 = 128x128, 2 = 128x256, 3 = 64x256 (columns x rows).
extern "C" int hipengine_dense_f16_gemm_f16_f16w_f32_out(
    const void* x, const void* weight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream, int tile) {
  if (!x || !weight || !out || rows <= 0 || in_features <= 0 ||
      out_features <= 0 || (in_features % 64) != 0 || rows > INT32_MAX ||
      in_features > INT32_MAX || out_features > INT32_MAX) {
    return static_cast<int>(hipErrorInvalidValue);
  }
#define HIPENGINE_F16_F16_LAUNCH(BM, BN, WAVES_M)                              \
  hipLaunchKernelGGL(                                                          \
      (dense_f16_gemm_staged_kernel<half_t, BM, BN, WAVES_M, half_t>),         \
      dim3(static_cast<unsigned int>((out_features + BM - 1) / BM),            \
           static_cast<unsigned int>((rows + BN - 1) / BN)),                   \
      dim3(256), 0, stream, static_cast<const half_t*>(x),                     \
      static_cast<const half_t*>(weight), out, static_cast<int>(rows),         \
      static_cast<int>(in_features), static_cast<int>(out_features))
  if (tile == 2) {
    HIPENGINE_F16_F16_LAUNCH(128, 256, 2);
  } else if (tile == 3) {
    HIPENGINE_F16_F16_LAUNCH(64, 256, 1);
  } else {
    HIPENGINE_F16_F16_LAUNCH(128, 128, 2);
  }
#undef HIPENGINE_F16_F16_LAUNCH
  return static_cast<int>(hipGetLastError());
}

// F16 activations (an image of the FP32 buffer, half(value) per element) with
// FP32 or BF16 weights: the staged 128x128 kernel without the activation
// conversion, so the products are those of the FP32-activation entry points.
template <typename WeightT>
int launch_f16_act(const void* x, const WeightT* weight, float* out,
                   int64_t rows, int64_t in_features, int64_t out_features,
                   hipStream_t stream) {
  if (!x || !weight || !out || rows <= 0 || in_features <= 0 ||
      out_features <= 0 || (in_features % 64) != 0 || rows > INT32_MAX ||
      in_features > INT32_MAX || out_features > INT32_MAX) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (dense_f16_gemm_staged_kernel<WeightT, 128, 128, 2, half_t>),
      dim3(static_cast<unsigned int>((out_features + 127) / 128),
           static_cast<unsigned int>((rows + 127) / 128)),
      dim3(256), 0, stream, static_cast<const half_t*>(x), weight, out,
      static_cast<int>(rows), static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dense_f16_gemm_f16_f32w_f32_out(
    const void* x, const float* weight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_f16_act<float>(x, weight, out, rows, in_features, out_features,
                               stream);
}

extern "C" int hipengine_dense_f16_gemm_f16_bf16w_f32_out(
    const void* x, const uint16_t* weight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_f16_act<uint16_t>(x, weight, out, rows, in_features,
                                  out_features, stream);
}

// Measurement and parity entry points: the same GEMM on a named tile.
extern "C" int hipengine_dense_f16_gemm_f32_f32w_f32_out_tile(
    const float* x, const float* weight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream, int tile) {
  return launch<float>(x, weight, out, rows, in_features, out_features, stream,
                       tile == 0 ? -1 : tile);
}

extern "C" int hipengine_dense_f16_gemm_f32_bf16w_f32_out_tile(
    const float* x, const uint16_t* weight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream, int tile) {
  return launch<uint16_t>(x, weight, out, rows, in_features, out_features,
                          stream, tile == 0 ? -1 : tile);
}
