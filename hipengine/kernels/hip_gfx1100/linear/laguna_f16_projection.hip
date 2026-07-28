// Laguna source-F16 dense projections with BF16/F32 activations and outputs.

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

using half_t = _Float16;

constexpr int kExactSmallTileM = 8;
constexpr int kExactLargeTileM = 16;
constexpr int kExactTileN = 4;
constexpr int kWmmaWave = 32;
constexpr int kWmmaTile = 16;

using half16_t = half_t __attribute__((ext_vector_type(16)));
using float8_t = float __attribute__((ext_vector_type(8)));

__device__ inline float bf16_to_f32(uint16_t value) {
  union { uint32_t u; float f; } out;
  out.u = static_cast<uint32_t>(value) << 16;
  return out.f;
}

__device__ inline uint16_t f32_to_bf16(float value) {
  union { float f; uint32_t u; } in;
  in.f = value;
  in.u += 0x7FFFu + ((in.u >> 16) & 1u);
  return static_cast<uint16_t>(in.u >> 16);
}

template <typename T>
__device__ inline float load_scalar(T value) { return static_cast<float>(value); }

template <>
__device__ inline float load_scalar<uint16_t>(uint16_t value) { return bf16_to_f32(value); }

template <typename T>
__device__ inline void store_scalar(T* out, int64_t idx, float value) { out[idx] = static_cast<T>(value); }

template <>
__device__ inline void store_scalar<uint16_t>(uint16_t* out, int64_t idx, float value) {
  out[idx] = f32_to_bf16(value);
}

__device__ inline float block_sum(float value) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down(value, offset);
  __shared__ float partial[8];
  if (lane == 0) partial[wave] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    const int waves = (blockDim.x + 31) >> 5;
    for (int i = 0; i < waves; ++i) total += partial[i];
    partial[0] = total;
  }
  __syncthreads();
  return partial[0];
}

__device__ inline float block_sum_thread0(float value) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  __shared__ float partial[8];
  if (lane == 0) partial[wave] = value;
  __syncthreads();
  if (threadIdx.x != 0) return 0.0f;
  float total = 0.0f;
  for (int i = 0; i < 8; ++i) total += partial[i];
  return total;
}

template <typename x_t, typename out_t>
__global__ void laguna_f16w_gemv_kernel(
    const x_t* __restrict__ x,
    const half_t* __restrict__ weight,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t col = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || col >= out_features) return;
  float acc = 0.0f;
  for (int64_t k = threadIdx.x; k < in_features; k += blockDim.x) {
    acc += load_scalar(x[row * in_features + k]) * static_cast<float>(weight[col * in_features + k]);
  }
  const float total = block_sum(acc);
  if (threadIdx.x == 0) store_scalar(out, row * out_features + col, total);
}

template <typename x_t, typename out_t>
__global__ void laguna_f16w_onebarrier_gemv_kernel(
    const x_t* __restrict__ x,
    const half_t* __restrict__ weight,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t col = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || col >= out_features) return;
  float acc = 0.0f;
  for (int64_t k = threadIdx.x; k < in_features; k += blockDim.x) {
    acc += load_scalar(x[row * in_features + k]) *
           static_cast<float>(weight[col * in_features + k]);
  }
  const float total = block_sum_thread0(acc);
  if (threadIdx.x == 0) {
    store_scalar(out, row * out_features + col, total);
  }
}

template <typename x_t>
__global__ void laguna_f16w_dual_gemv_kernel(
    const x_t* __restrict__ x,
    const half_t* __restrict__ weight_a,
    const half_t* __restrict__ weight_b,
    float* __restrict__ out_a,
    float* __restrict__ out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_a_features,
    int64_t out_b_features) {
  const int64_t col = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int64_t total_cols = out_a_features + out_b_features;
  if (row >= rows || col >= total_cols) return;
  const bool a = col < out_a_features;
  const int64_t local_col = a ? col : col - out_a_features;
  const half_t* weight = a ? weight_a : weight_b;
  float acc = 0.0f;
  for (int64_t k = threadIdx.x; k < in_features; k += blockDim.x) {
    acc += load_scalar(x[row * in_features + k]) * static_cast<float>(weight[local_col * in_features + k]);
  }
  const float total = block_sum(acc);
  if (threadIdx.x == 0) {
    if (a) out_a[row * out_a_features + local_col] = total;
    else out_b[row * out_b_features + local_col] = total;
  }
}

template <typename x_t>
__global__ void laguna_f16w_triple_gemv_kernel(
    const x_t* __restrict__ x,
    const half_t* __restrict__ weight_a,
    const half_t* __restrict__ weight_b,
    const half_t* __restrict__ weight_c,
    float* __restrict__ out_a,
    float* __restrict__ out_b,
    float* __restrict__ out_c,
    int64_t rows,
    int64_t in_features,
    int64_t out_a_features,
    int64_t out_b_features,
    int64_t out_c_features) {
  const int64_t col = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || col >= out_a_features + out_b_features + out_c_features) return;
  const half_t* weight;
  float* out;
  int64_t local_col;
  int64_t width;
  if (col < out_a_features) {
    weight = weight_a; out = out_a; local_col = col; width = out_a_features;
  } else if (col < out_a_features + out_b_features) {
    weight = weight_b; out = out_b; local_col = col - out_a_features; width = out_b_features;
  } else {
    weight = weight_c; out = out_c; local_col = col - out_a_features - out_b_features; width = out_c_features;
  }
  float acc = 0.0f;
  for (int64_t k = threadIdx.x; k < in_features; k += blockDim.x) {
    acc += load_scalar(x[row * in_features + k]) * static_cast<float>(weight[local_col * in_features + k]);
  }
  const float total = block_sum(acc);
  if (threadIdx.x == 0) out[row * width + local_col] = total;
}

template <typename x_t>
__global__ void laguna_f16w_triple_onebarrier_gemv_kernel(
    const x_t* __restrict__ x,
    const half_t* __restrict__ weight_a,
    const half_t* __restrict__ weight_b,
    const half_t* __restrict__ weight_c,
    float* __restrict__ out_a,
    float* __restrict__ out_b,
    float* __restrict__ out_c,
    int64_t rows,
    int64_t in_features,
    int64_t out_a_features,
    int64_t out_b_features,
    int64_t out_c_features) {
  const int64_t col = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows ||
      col >= out_a_features + out_b_features + out_c_features) {
    return;
  }
  const half_t* weight;
  float* out;
  int64_t local_col;
  int64_t width;
  if (col < out_a_features) {
    weight = weight_a;
    out = out_a;
    local_col = col;
    width = out_a_features;
  } else if (col < out_a_features + out_b_features) {
    weight = weight_b;
    out = out_b;
    local_col = col - out_a_features;
    width = out_b_features;
  } else {
    weight = weight_c;
    out = out_c;
    local_col = col - out_a_features - out_b_features;
    width = out_c_features;
  }
  float acc = 0.0f;
  for (int64_t k = threadIdx.x; k < in_features; k += blockDim.x) {
    acc += load_scalar(x[row * in_features + k]) *
           static_cast<float>(weight[local_col * in_features + k]);
  }
  const float total = block_sum_thread0(acc);
  if (threadIdx.x == 0) out[row * width + local_col] = total;
}

template <typename out_t, int TILE_M>
__global__ void laguna_f16w_tiled_exact_kernel(
    const uint16_t* __restrict__ x,
    const half_t* __restrict__ weight,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t row_base = static_cast<int64_t>(blockIdx.y) * TILE_M;
  const int64_t col_base = static_cast<int64_t>(blockIdx.x) * kExactTileN;
  float acc[TILE_M][kExactTileN] = {};
  for (int64_t k = threadIdx.x; k < in_features; k += blockDim.x) {
    float x_values[TILE_M];
    float weight_values[kExactTileN];
#pragma unroll
    for (int row = 0; row < TILE_M; ++row) {
      const int64_t global_row = row_base + row;
      x_values[row] = global_row < rows
          ? bf16_to_f32(x[global_row * in_features + k])
          : 0.0f;
    }
#pragma unroll
    for (int col = 0; col < kExactTileN; ++col) {
      const int64_t global_col = col_base + col;
      weight_values[col] = global_col < out_features
          ? static_cast<float>(weight[global_col * in_features + k])
          : 0.0f;
    }
#pragma unroll
    for (int row = 0; row < TILE_M; ++row) {
#pragma unroll
      for (int col = 0; col < kExactTileN; ++col) {
        // Preserve the GEMV thread-local K sequence exactly; the tile changes
        // only when each activation/weight load is reused.
        acc[row][col] += x_values[row] * weight_values[col];
      }
    }
  }
#pragma unroll
  for (int row = 0; row < TILE_M; ++row) {
#pragma unroll
    for (int col = 0; col < kExactTileN; ++col) {
      const float total = block_sum(acc[row][col]);
      if (threadIdx.x == 0) {
        const int64_t global_row = row_base + row;
        const int64_t global_col = col_base + col;
        if (global_row < rows && global_col < out_features) {
          store_scalar(out, global_row * out_features + global_col, total);
        }
      }
    }
  }
}

template <typename out_t, int TILE_M>
int launch_tiled_exact_shape(
    const uint16_t* x,
    const half_t* weight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0 || threads != 256) {
    return hipErrorInvalidValue;
  }
  laguna_f16w_tiled_exact_kernel<out_t, TILE_M><<<
      dim3((out_features + kExactTileN - 1) / kExactTileN,
           (rows + TILE_M - 1) / TILE_M),
      dim3(threads),
      0,
      stream>>>(x, weight, out, rows, in_features, out_features);
  return hipGetLastError();
}

template <typename out_t>
int launch_tiled_exact(
    const uint16_t* x,
    const half_t* weight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (rows < kExactLargeTileM) {
    return launch_tiled_exact_shape<out_t, kExactSmallTileM>(
        x, weight, out, rows, in_features, out_features, threads, stream);
  }
  return launch_tiled_exact_shape<out_t, kExactLargeTileM>(
      x, weight, out, rows, in_features, out_features, threads, stream);
}

__device__ inline half16_t zero_half16() {
  half16_t values;
#pragma unroll
  for (int i = 0; i < kWmmaTile; ++i) values[i] = static_cast<half_t>(0.0f);
  return values;
}

__device__ inline float8_t zero_float8() {
  float8_t values;
#pragma unroll
  for (int i = 0; i < 8; ++i) values[i] = 0.0f;
  return values;
}

__device__ inline half16_t load_bf16_half16(
    const uint16_t* __restrict__ x,
    int64_t row,
    int64_t k_base,
    int64_t in_features,
    bool valid) {
  if (!valid) return zero_half16();
  half16_t values;
#pragma unroll
  for (int i = 0; i < kWmmaTile; ++i) {
    values[i] = static_cast<half_t>(
        bf16_to_f32(x[row * in_features + k_base + i]));
  }
  return values;
}

__device__ inline void compensated_wmma_accumulate(
    const half16_t a,
    const half16_t b,
    float8_t* sum,
    float8_t* compensation) {
  const float8_t partial = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
      a, b, zero_float8());
  const float8_t corrected = partial - *compensation;
  const float8_t next_sum = *sum + corrected;
  *compensation = (next_sum - *sum) - corrected;
  *sum = next_sum;
}

// One wave computes a 16-row x 16-column tile directly from the resident
// row-major F16 source weight. BF16 activations are converted to F16 in
// registers; no repack, sidecar, or inference-time staging is introduced.
constexpr int kWmmaWavesPerBlock = 4;
constexpr int kWmmaBlockThreads = kWmmaWave * kWmmaWavesPerBlock;
constexpr int kWmmaBlockN = kWmmaTile * kWmmaWavesPerBlock;

template <typename out_t, int ROW_TILES, bool COMPENSATED>
__launch_bounds__(kWmmaBlockThreads, 2)
__global__ void laguna_f16w_wmma_kernel(
    const uint16_t* __restrict__ x,
    const half_t* __restrict__ weight,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  constexpr int accumulator_states = COMPENSATED ? 2 : 1;
  constexpr int rows_per_block = ROW_TILES * kWmmaTile;
  __shared__ half_t x_tile[rows_per_block * kWmmaTile];
  const int wave = threadIdx.x / kWmmaWave;
  const int wave_lane = threadIdx.x & (kWmmaWave - 1);
  const int lane16 = wave_lane & 15;
  const int half_id = wave_lane >> 4;
  const int64_t row_base =
      static_cast<int64_t>(blockIdx.y) * rows_per_block;
  const int64_t col_base = static_cast<int64_t>(blockIdx.x) * kWmmaBlockN;
  const int64_t load_col = col_base + wave * kWmmaTile + lane16;
  const bool valid_col = load_col < out_features;
  float8_t acc[ROW_TILES][accumulator_states];
#pragma unroll
  for (int tile = 0; tile < ROW_TILES; ++tile) {
#pragma unroll
    for (int group = 0; group < accumulator_states; ++group) {
      acc[tile][group] = zero_float8();
    }
  }

  for (int64_t k_base = 0; k_base < in_features; k_base += kWmmaTile) {
    if (threadIdx.x < rows_per_block) {
      const int64_t load_row = row_base + threadIdx.x;
      const half16_t values = load_bf16_half16(
          x, load_row, k_base, in_features, load_row < rows);
      *reinterpret_cast<half16_t*>(
          x_tile + threadIdx.x * kWmmaTile) = values;
    }
    __syncthreads();
    const half16_t b = valid_col
        ? *reinterpret_cast<const half16_t*>(
              weight + load_col * in_features + k_base)
        : zero_half16();
#pragma unroll
    for (int tile = 0; tile < ROW_TILES; ++tile) {
      const half16_t a = *reinterpret_cast<const half16_t*>(
          x_tile + (tile * kWmmaTile + lane16) * kWmmaTile);
      if constexpr (COMPENSATED) {
        compensated_wmma_accumulate(
            a, b, &acc[tile][0], &acc[tile][1]);
      } else {
        acc[tile][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            a, b, acc[tile][0]);
      }
    }
    __syncthreads();
  }

  if (!valid_col) return;
#pragma unroll
  for (int tile = 0; tile < ROW_TILES; ++tile) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float value = acc[tile][0][j];
      const int64_t store_row =
          row_base + tile * kWmmaTile + half_id + 2 * j;
      if (store_row < rows) {
        store_scalar(out, store_row * out_features + load_col, value);
      }
    }
  }
}

template <typename out_t, int M_GROUPS, int N_GROUPS, int ROW_TILES,
          bool COMPENSATED>
__launch_bounds__(kWmmaWave * M_GROUPS * N_GROUPS, 1)
__global__ void laguna_f16w_wmma_large_kernel(
    const uint16_t* __restrict__ x,
    const half_t* __restrict__ weight,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  constexpr int accumulator_states = COMPENSATED ? 2 : 1;
  constexpr int rows_per_wave = ROW_TILES * kWmmaTile;
  constexpr int rows_per_block = M_GROUPS * rows_per_wave;
  constexpr int cols_per_block = N_GROUPS * kWmmaTile;
  constexpr int block_threads = kWmmaWave * M_GROUPS * N_GROUPS;
  __shared__ half_t x_tile[rows_per_block * kWmmaTile];
  __shared__ half_t w_tile[N_GROUPS * kWmmaTile * kWmmaTile];
  const int wave = threadIdx.x / kWmmaWave;
  const int wave_lane = threadIdx.x & (kWmmaWave - 1);
  const int m_group = wave / N_GROUPS;
  const int n_group = wave % N_GROUPS;
  const int lane16 = wave_lane & 15;
  const int half_id = wave_lane >> 4;
  const int64_t row_base =
      static_cast<int64_t>(blockIdx.y) * rows_per_block;
  const int64_t col_base =
      static_cast<int64_t>(blockIdx.x) * cols_per_block;
  const int64_t store_col = col_base + n_group * kWmmaTile + lane16;
  const bool valid_col = store_col < out_features;
  float8_t acc[ROW_TILES][accumulator_states];
#pragma unroll
  for (int tile = 0; tile < ROW_TILES; ++tile) {
#pragma unroll
    for (int group = 0; group < accumulator_states; ++group) {
      acc[tile][group] = zero_float8();
    }
  }

  for (int64_t k_base = 0; k_base < in_features; k_base += kWmmaTile) {
    for (int row = threadIdx.x; row < rows_per_block;
         row += block_threads) {
      const int64_t load_row = row_base + row;
      const half16_t values = load_bf16_half16(
          x, load_row, k_base, in_features, load_row < rows);
      *reinterpret_cast<half16_t*>(x_tile + row * kWmmaTile) = values;
    }
    constexpr int weight_fragments = N_GROUPS * kWmmaTile;
    for (int fragment = threadIdx.x; fragment < weight_fragments;
         fragment += block_threads) {
      const int64_t load_col = col_base + fragment;
      const half16_t values = load_col < out_features
          ? *reinterpret_cast<const half16_t*>(
                weight + load_col * in_features + k_base)
          : zero_half16();
      *reinterpret_cast<half16_t*>(
          w_tile + fragment * kWmmaTile) = values;
    }
    __syncthreads();
    const half16_t b = *reinterpret_cast<const half16_t*>(
        w_tile + (n_group * kWmmaTile + lane16) * kWmmaTile);
#pragma unroll
    for (int tile = 0; tile < ROW_TILES; ++tile) {
      const int row_in_block =
          m_group * rows_per_wave + tile * kWmmaTile + lane16;
      const half16_t a = *reinterpret_cast<const half16_t*>(
          x_tile + row_in_block * kWmmaTile);
      if constexpr (COMPENSATED) {
        compensated_wmma_accumulate(
            a, b, &acc[tile][0], &acc[tile][1]);
      } else {
        acc[tile][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            a, b, acc[tile][0]);
      }
    }
    __syncthreads();
  }

  if (!valid_col) return;
#pragma unroll
  for (int tile = 0; tile < ROW_TILES; ++tile) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float value = acc[tile][0][j];
      const int64_t store_row = row_base + m_group * rows_per_wave +
          tile * kWmmaTile + half_id + 2 * j;
      if (store_row < rows) {
        store_scalar(out, store_row * out_features + store_col, value);
      }
    }
  }
}

template <typename out_t, int M_GROUPS, int N_GROUPS, int ROW_TILES,
          bool COMPENSATED>
int launch_wmma_large_shape(
    const uint16_t* x,
    const half_t* weight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  constexpr int rows_per_block = M_GROUPS * ROW_TILES * kWmmaTile;
  constexpr int cols_per_block = N_GROUPS * kWmmaTile;
  constexpr int block_threads = kWmmaWave * M_GROUPS * N_GROUPS;
  laguna_f16w_wmma_large_kernel<
      out_t, M_GROUPS, N_GROUPS, ROW_TILES, COMPENSATED><<<
      dim3((out_features + cols_per_block - 1) / cols_per_block,
           (rows + rows_per_block - 1) / rows_per_block),
      dim3(block_threads),
      0,
      stream>>>(x, weight, out, rows, in_features, out_features);
  return hipGetLastError();
}

template <typename out_t, int ROW_TILES, bool COMPENSATED>
int launch_wmma_shape(
    const uint16_t* x,
    const half_t* weight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  constexpr int rows_per_block = ROW_TILES * kWmmaTile;
  laguna_f16w_wmma_kernel<out_t, ROW_TILES, COMPENSATED><<<
      dim3((out_features + kWmmaBlockN - 1) / kWmmaBlockN,
           (rows + rows_per_block - 1) / rows_per_block),
      dim3(kWmmaBlockThreads),
      0,
      stream>>>(x, weight, out, rows, in_features, out_features);
  return hipGetLastError();
}

template <bool COMPENSATED, typename out_t>
int launch_wmma(
    const uint16_t* x,
    const half_t* weight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (in_features % kWmmaTile) != 0 || threads != 256) {
    return hipErrorInvalidValue;
  }
  if constexpr (COMPENSATED) {
    if (rows >= 512) {
      return launch_wmma_large_shape<out_t, 4, 2, 4, true>(
          x, weight, out, rows, in_features, out_features, stream);
    }
    if (rows >= 256) {
      return launch_wmma_large_shape<out_t, 2, 4, 4, true>(
          x, weight, out, rows, in_features, out_features, stream);
    }
    if (rows >= 128) {
      return launch_wmma_shape<out_t, 4, true>(
          x, weight, out, rows, in_features, out_features, stream);
    }
  }
  if (rows >= 512) {
    return launch_wmma_large_shape<out_t, 4, 2, 8, COMPENSATED>(
        x, weight, out, rows, in_features, out_features, stream);
  }
  if (rows >= 256) {
    return launch_wmma_large_shape<out_t, 2, 4, 8, COMPENSATED>(
        x, weight, out, rows, in_features, out_features, stream);
  }
  if (rows >= 128) {
    return launch_wmma_shape<out_t, 8, COMPENSATED>(
        x, weight, out, rows, in_features, out_features, stream);
  }
  if (rows >= 64) {
    return launch_wmma_shape<out_t, 4, COMPENSATED>(
        x, weight, out, rows, in_features, out_features, stream);
  }
  if (rows >= 32) {
    return launch_wmma_shape<out_t, 2, COMPENSATED>(
        x, weight, out, rows, in_features, out_features, stream);
  }
  return launch_wmma_shape<out_t, 1, COMPENSATED>(
      x, weight, out, rows, in_features, out_features, stream);
}

template <bool COMPENSATED>
int launch_triple_wmma(
    const uint16_t* x,
    const half_t* wa,
    const half_t* wb,
    const half_t* wc,
    float* oa,
    float* ob,
    float* oc,
    int64_t rows,
    int64_t in_features,
    int64_t out_a,
    int64_t out_b,
    int64_t out_c,
    int64_t threads,
    hipStream_t stream) {
  hipError_t error = static_cast<hipError_t>(
      launch_wmma<COMPENSATED>(
          x, wa, oa, rows, in_features, out_a, threads, stream));
  if (error != hipSuccess) return error;
  error = static_cast<hipError_t>(
      launch_wmma<COMPENSATED>(
          x, wb, ob, rows, in_features, out_b, threads, stream));
  if (error != hipSuccess) return error;
  return launch_wmma<COMPENSATED>(
      x, wc, oc, rows, in_features, out_c, threads, stream);
}

int launch_triple_tiled_exact(
    const uint16_t* x,
    const half_t* wa,
    const half_t* wb,
    const half_t* wc,
    float* oa,
    float* ob,
    float* oc,
    int64_t rows,
    int64_t in_features,
    int64_t out_a,
    int64_t out_b,
    int64_t out_c,
    int64_t threads,
    hipStream_t stream) {
  hipError_t error = static_cast<hipError_t>(
      launch_tiled_exact(x, wa, oa, rows, in_features, out_a, threads, stream));
  if (error != hipSuccess) return error;
  error = static_cast<hipError_t>(
      launch_tiled_exact(x, wb, ob, rows, in_features, out_b, threads, stream));
  if (error != hipSuccess) return error;
  return launch_tiled_exact(x, wc, oc, rows, in_features, out_c, threads, stream);
}

template <typename x_t, typename out_t>
int launch_single(const x_t* x, const half_t* weight, out_t* out, int64_t rows,
                  int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0 || threads <= 0 || threads > 256 || threads % 32) return hipErrorInvalidValue;
  laguna_f16w_gemv_kernel<<<dim3(out_features, rows), dim3(threads), 0, stream>>>(x, weight, out, rows, in_features, out_features);
  return hipGetLastError();
}

template <typename x_t, typename out_t>
int launch_onebarrier_single(
    const x_t* x,
    const half_t* weight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (rows != 1 || in_features <= 0 || out_features <= 0 || threads != 256) {
    return hipErrorInvalidValue;
  }
  laguna_f16w_onebarrier_gemv_kernel<<<
      dim3(out_features, rows), dim3(threads), 0, stream>>>(
      x, weight, out, rows, in_features, out_features);
  return hipGetLastError();
}

template <typename x_t>
int launch_dual(const x_t* x, const half_t* wa, const half_t* wb, float* oa, float* ob,
                int64_t rows, int64_t in_features, int64_t out_a, int64_t out_b,
                int64_t threads, hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_a <= 0 || out_b <= 0 || threads <= 0 || threads > 256 || threads % 32) return hipErrorInvalidValue;
  laguna_f16w_dual_gemv_kernel<<<dim3(out_a + out_b, rows), dim3(threads), 0, stream>>>(x, wa, wb, oa, ob, rows, in_features, out_a, out_b);
  return hipGetLastError();
}

template <typename x_t>
int launch_triple(const x_t* x, const half_t* wa, const half_t* wb, const half_t* wc,
                  float* oa, float* ob, float* oc, int64_t rows, int64_t in_features,
                  int64_t out_a, int64_t out_b, int64_t out_c, int64_t threads, hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_a <= 0 || out_b <= 0 || out_c <= 0 || threads <= 0 || threads > 256 || threads % 32) return hipErrorInvalidValue;
  laguna_f16w_triple_gemv_kernel<<<dim3(out_a + out_b + out_c, rows), dim3(threads), 0, stream>>>(x, wa, wb, wc, oa, ob, oc, rows, in_features, out_a, out_b, out_c);
  return hipGetLastError();
}

template <typename x_t>
int launch_triple_onebarrier(
    const x_t* x,
    const half_t* wa,
    const half_t* wb,
    const half_t* wc,
    float* oa,
    float* ob,
    float* oc,
    int64_t rows,
    int64_t in_features,
    int64_t out_a,
    int64_t out_b,
    int64_t out_c,
    int64_t threads,
    hipStream_t stream) {
  if (rows != 1 || in_features <= 0 || out_a <= 0 || out_b <= 0 ||
      out_c <= 0 || threads != 256) {
    return hipErrorInvalidValue;
  }
  laguna_f16w_triple_onebarrier_gemv_kernel<<<
      dim3(out_a + out_b + out_c, rows), dim3(threads), 0, stream>>>(
      x, wa, wb, wc, oa, ob, oc, rows, in_features, out_a, out_b, out_c);
  return hipGetLastError();
}

}  // namespace

extern "C" int hipengine_laguna_f16w_gemv_bf16_f32_out(const uint16_t* x, const half_t* w, float* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_single(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_gemv_bf16_bf16_out(const uint16_t* x, const half_t* w, uint16_t* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_single(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_onebarrier_gemv_bf16_f32_out(const uint16_t* x, const half_t* w, float* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_onebarrier_single(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_onebarrier_gemv_bf16_bf16_out(const uint16_t* x, const half_t* w, uint16_t* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_onebarrier_single(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_gemv_f32_f32_out(const float* x, const half_t* w, float* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_single(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_gemv_f32_bf16_out(const float* x, const half_t* w, uint16_t* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_single(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_dual_gemv_bf16_f32_out(const uint16_t* x, const half_t* wa, const half_t* wb, float* oa, float* ob, int64_t r, int64_t k, int64_t na, int64_t nb, int64_t t, hipStream_t s) { return launch_dual(x,wa,wb,oa,ob,r,k,na,nb,t,s); }
extern "C" int hipengine_laguna_f16w_triple_gemv_bf16_f32_out(const uint16_t* x, const half_t* wa, const half_t* wb, const half_t* wc, float* oa, float* ob, float* oc, int64_t r, int64_t k, int64_t na, int64_t nb, int64_t nc, int64_t t, hipStream_t s) { return launch_triple(x,wa,wb,wc,oa,ob,oc,r,k,na,nb,nc,t,s); }
extern "C" int hipengine_laguna_f16w_triple_onebarrier_gemv_bf16_f32_out(const uint16_t* x, const half_t* wa, const half_t* wb, const half_t* wc, float* oa, float* ob, float* oc, int64_t r, int64_t k, int64_t na, int64_t nb, int64_t nc, int64_t t, hipStream_t s) { return launch_triple_onebarrier(x,wa,wb,wc,oa,ob,oc,r,k,na,nb,nc,t,s); }
extern "C" int hipengine_laguna_f16w_tiled_bf16_f32_out(const uint16_t* x, const half_t* w, float* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_tiled_exact(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_tiled_bf16_bf16_out(const uint16_t* x, const half_t* w, uint16_t* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_tiled_exact(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_triple_tiled_bf16_f32_out(const uint16_t* x, const half_t* wa, const half_t* wb, const half_t* wc, float* oa, float* ob, float* oc, int64_t r, int64_t k, int64_t na, int64_t nb, int64_t nc, int64_t t, hipStream_t s) { return launch_triple_tiled_exact(x,wa,wb,wc,oa,ob,oc,r,k,na,nb,nc,t,s); }
extern "C" int hipengine_laguna_f16w_wmma_bf16_f32_out(const uint16_t* x, const half_t* w, float* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_wmma<false>(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_wmma_bf16_bf16_out(const uint16_t* x, const half_t* w, uint16_t* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_wmma<false>(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_triple_wmma_bf16_f32_out(const uint16_t* x, const half_t* wa, const half_t* wb, const half_t* wc, float* oa, float* ob, float* oc, int64_t r, int64_t k, int64_t na, int64_t nb, int64_t nc, int64_t t, hipStream_t s) { return launch_triple_wmma<false>(x,wa,wb,wc,oa,ob,oc,r,k,na,nb,nc,t,s); }
extern "C" int hipengine_laguna_f16w_wmma_comp_bf16_f32_out(const uint16_t* x, const half_t* w, float* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_wmma<true>(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_wmma_comp_bf16_bf16_out(const uint16_t* x, const half_t* w, uint16_t* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_wmma<true>(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_triple_wmma_comp_bf16_f32_out(const uint16_t* x, const half_t* wa, const half_t* wb, const half_t* wc, float* oa, float* ob, float* oc, int64_t r, int64_t k, int64_t na, int64_t nb, int64_t nc, int64_t t, hipStream_t s) { return launch_triple_wmma<true>(x,wa,wb,wc,oa,ob,oc,r,k,na,nb,nc,t,s); }
