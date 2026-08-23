#include <hip/hip_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kHadamardBlock = 1024;
constexpr int kPack = 8;
constexpr uint32_t kGateSeed = 0xA511E9B3U;
constexpr uint32_t kDownSeed = 0x63D83595U;

using i32x2_t = int __attribute__((__vector_size__(2 * sizeof(int))));
using i32x4_t = int __attribute__((__vector_size__(4 * sizeof(int))));
using i32x8_t = int __attribute__((__vector_size__(8 * sizeof(int))));

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

__host__ __device__ inline float hadamard_sign(int index, uint32_t seed) {
  uint32_t value = static_cast<uint32_t>(index) + seed;
  value ^= value >> 16;
  value *= 0x7FEB352DU;
  value ^= value >> 15;
  value *= 0x846CA68BU;
  value ^= value >> 16;
  return (value & 1U) ? -1.0f : 1.0f;
}

__device__ inline uint32_t insert_u4(uint32_t word, int value, int index) {
  return word | (static_cast<uint32_t>(value) << (4 * index));
}

template <int K, uint32_t Seed, bool SwiGLU>
__launch_bounds__(256, 1)
__global__ void iu4_pfs_pack_hadamard_bf16_kernel(
    const uint16_t* __restrict__ input,
    uint32_t* __restrict__ packed,
    float* __restrict__ scales,
    int32_t* __restrict__ zero_points,
    int64_t rows) {
  static_assert(K % kHadamardBlock == 0, "Hadamard K must be block aligned");
  constexpr int kBlocks = K / kHadamardBlock;
  constexpr int kItemsPerThread = kHadamardBlock / 256;
  constexpr int kWordsPerRow = K / kPack;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (row >= rows) return;

  __shared__ float block[kHadamardBlock];
  float transformed[kBlocks][kItemsPerThread];
  float lo = FLT_MAX;
  float hi = -FLT_MAX;
#pragma unroll
  for (int block_index = 0; block_index < kBlocks; ++block_index) {
    const int block0 = block_index * kHadamardBlock;
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      const int local = tid + item * 256;
      const int col = block0 + local;
      float value;
      if constexpr (SwiGLU) {
        const float gate = bf16_bits_to_float(
            input[row * (2 * static_cast<int64_t>(K)) + col]);
        const float up = bf16_bits_to_float(
            input[row * (2 * static_cast<int64_t>(K)) + K + col]);
        value = (gate / (1.0f + expf(-gate))) * up;
      } else {
        value = bf16_bits_to_float(input[row * K + col]);
      }
      block[local] = value * hadamard_sign(col, Seed);
    }
    __syncthreads();
    for (int stride = 1; stride < kHadamardBlock; stride <<= 1) {
      for (int pair = tid; pair < kHadamardBlock / 2; pair += 256) {
        const int group = pair / stride;
        const int offset = pair - group * stride;
        const int i = group * (2 * stride) + offset;
        const int j = i + stride;
        const float a = block[i];
        const float b = block[j];
        block[i] = a + b;
        block[j] = a - b;
      }
      __syncthreads();
    }
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      const int local = tid + item * 256;
      const float value = block[local] * (1.0f / 32.0f);
      transformed[block_index][item] = value;
      lo = fminf(lo, value);
      hi = fmaxf(hi, value);
    }
    __syncthreads();
  }

  block[tid] = lo;
  block[256 + tid] = hi;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      block[tid] = fminf(block[tid], block[tid + stride]);
      block[256 + tid] = fmaxf(block[256 + tid], block[256 + tid + stride]);
    }
    __syncthreads();
  }
  float scale = (block[256] - block[0]) * (1.0f / 15.0f);
  if (!(scale > 0.0f) || !isfinite(scale)) scale = 1.0f;
  int zero = __float2int_rn(-block[0] / scale);
  zero = zero < 0 ? 0 : (zero > 15 ? 15 : zero);
  if (tid == 0) {
    scales[row] = scale;
    zero_points[row] = zero;
  }

#pragma unroll
  for (int block_index = 0; block_index < kBlocks; ++block_index) {
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      int code = __float2int_rn(transformed[block_index][item] / scale) + zero;
      code = code < 0 ? 0 : (code > 15 ? 15 : code);
      uint32_t word = 0;
#pragma unroll
      for (int nibble = 0; nibble < kPack; ++nibble) {
        word = insert_u4(word, __shfl(code, nibble, kPack), nibble);
      }
      if ((tid & (kPack - 1)) == 0) {
        const int word_index =
            block_index * (kHadamardBlock / kPack) +
            item * (256 / kPack) + tid / kPack;
        packed[row * kWordsPerRow + word_index] = word;
      }
    }
  }
}

template <int OUT_TILES_PER_BLOCK, int ROW_TILES_PER_WAVE, int K_PAIRS_PER_STAGE>
__launch_bounds__(128, 1)
__global__ void iu4_pfs_linear_bf16_bulk_kernel(
    const uint32_t* __restrict__ activations,
    const float* __restrict__ activation_scales,
    const int32_t* __restrict__ activation_zero_points,
    const uint32_t* __restrict__ weights,
    const float* __restrict__ weight_scales,
    const int32_t* __restrict__ weight_sums,
    uint16_t* __restrict__ output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  constexpr int WAVES_PER_BLOCK = 4;
  constexpr int ROWS_PER_WAVE = ROW_TILES_PER_WAVE * 16;
  constexpr int ROWS_PER_BLOCK = WAVES_PER_BLOCK * ROWS_PER_WAVE;
  constexpr int WORDS_PER_PAIR = 4;
  constexpr int COLS_PER_TILE = 16;
  constexpr int OUT_COLS_PER_BLOCK = OUT_TILES_PER_BLOCK * COLS_PER_TILE;
  __shared__ uint32_t
      staged_weights[OUT_TILES_PER_BLOCK][K_PAIRS_PER_STAGE][COLS_PER_TILE][WORDS_PER_PAIR];

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int lane16 = lane & 15;
  const int half = lane >> 4;
  const int64_t words_per_row = in_features / 8;
  const int64_t k_pairs = in_features / 32;
  const int64_t block_col =
      static_cast<int64_t>(blockIdx.x) * OUT_COLS_PER_BLOCK;
  const int64_t block_row = static_cast<int64_t>(blockIdx.y) * ROWS_PER_BLOCK;
  const int64_t wave_row = block_row + wave * ROWS_PER_WAVE;

  i32x8_t acc[OUT_TILES_PER_BLOCK][ROW_TILES_PER_WAVE];
#pragma unroll
  for (int tile = 0; tile < OUT_TILES_PER_BLOCK; ++tile) {
#pragma unroll
    for (int row_tile = 0; row_tile < ROW_TILES_PER_WAVE; ++row_tile) {
      acc[tile][row_tile] = i32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }

  for (int64_t stage_k_pair = 0; stage_k_pair < k_pairs;
       stage_k_pair += K_PAIRS_PER_STAGE) {
    constexpr int STAGE_WORDS = K_PAIRS_PER_STAGE * WORDS_PER_PAIR;
    constexpr int TOTAL_VALUES = STAGE_WORDS * OUT_COLS_PER_BLOCK;
    for (int index = tid; index < TOTAL_VALUES; index += blockDim.x) {
      const int stage_word = index / OUT_COLS_PER_BLOCK;
      const int local_col = index - stage_word * OUT_COLS_PER_BLOCK;
      const int tile = local_col / COLS_PER_TILE;
      const int col_in_tile = local_col - tile * COLS_PER_TILE;
      const int local_k_pair = stage_word / WORDS_PER_PAIR;
      const int word_in_pair = stage_word - local_k_pair * WORDS_PER_PAIR;
      const int64_t output_col = block_col + local_col;
      const int64_t out_tile =
          static_cast<int64_t>(blockIdx.x) * OUT_TILES_PER_BLOCK + tile;
      const int64_t k_pair = stage_k_pair + local_k_pair;
      uint32_t value = 0;
      if (output_col < out_features && k_pair < k_pairs) {
        const int64_t source_index =
            ((out_tile * k_pairs + k_pair) * COLS_PER_TILE + col_in_tile) *
                WORDS_PER_PAIR +
            word_in_pair;
        value = weights[source_index];
      }
      staged_weights[tile][local_k_pair][col_in_tile][word_in_pair] = value;
    }
    __syncthreads();

#pragma unroll
    for (int local_k_pair = 0; local_k_pair < K_PAIRS_PER_STAGE;
         ++local_k_pair) {
      const int64_t k_pair = stage_k_pair + local_k_pair;
      if (k_pair >= k_pairs) break;
      i32x4_t activation_pair[ROW_TILES_PER_WAVE];
#pragma unroll
      for (int row_tile = 0; row_tile < ROW_TILES_PER_WAVE; ++row_tile) {
        activation_pair[row_tile] = i32x4_t{0, 0, 0, 0};
        const int64_t activation_row = wave_row + row_tile * 16 + lane16;
        if (activation_row < rows) {
          activation_pair[row_tile] = *reinterpret_cast<const i32x4_t*>(
              activations + activation_row * words_per_row + k_pair * WORDS_PER_PAIR);
        }
      }
      i32x4_t weight_fragment[OUT_TILES_PER_BLOCK];
#pragma unroll
      for (int tile = 0; tile < OUT_TILES_PER_BLOCK; ++tile) {
        weight_fragment[tile] = *reinterpret_cast<const i32x4_t*>(
            &staged_weights[tile][local_k_pair][lane16][0]);
      }
#pragma unroll
      for (int tile = 0; tile < OUT_TILES_PER_BLOCK; ++tile) {
        const i32x2_t weight0 = {
            weight_fragment[tile][0], weight_fragment[tile][1]};
#pragma unroll
        for (int row_tile = 0; row_tile < ROW_TILES_PER_WAVE; ++row_tile) {
          const i32x2_t activation0 = {
              activation_pair[row_tile][0], activation_pair[row_tile][1]};
          acc[tile][row_tile] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              false, activation0, true, weight0, acc[tile][row_tile], false);
        }
      }
#pragma unroll
      for (int tile = 0; tile < OUT_TILES_PER_BLOCK; ++tile) {
        const i32x2_t weight1 = {
            weight_fragment[tile][2], weight_fragment[tile][3]};
#pragma unroll
        for (int row_tile = 0; row_tile < ROW_TILES_PER_WAVE; ++row_tile) {
          const i32x2_t activation1 = {
              activation_pair[row_tile][2], activation_pair[row_tile][3]};
          acc[tile][row_tile] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              false, activation1, true, weight1, acc[tile][row_tile], false);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int tile = 0; tile < OUT_TILES_PER_BLOCK; ++tile) {
    const int64_t output_col = block_col + tile * COLS_PER_TILE + lane16;
#pragma unroll
    for (int row_tile = 0; row_tile < ROW_TILES_PER_WAVE; ++row_tile) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t row = wave_row + row_tile * 16 + half + 2 * j;
        if (row < rows && output_col < out_features) {
          const int32_t corrected =
              acc[tile][row_tile][j] -
              activation_zero_points[row] * weight_sums[output_col];
          output[row * out_features + output_col] = float_to_bf16_bits(
              static_cast<float>(corrected) * activation_scales[row] *
              weight_scales[output_col]);
        }
      }
    }
  }
}

}  // namespace

extern "C" int hipengine_iu4_pfs_pack_gate_bf16(
    const void* input,
    void* packed,
    void* scales,
    void* zero_points,
    int64_t rows,
    int64_t hidden,
    void* stream) {
  if (input == nullptr || packed == nullptr || scales == nullptr ||
      zero_points == nullptr || rows <= 0 || rows > 2048) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
  if (hidden == 1024) {
    hipLaunchKernelGGL(
        (iu4_pfs_pack_hadamard_bf16_kernel<1024, kGateSeed, false>),
        dim3(static_cast<uint32_t>(rows)), dim3(256), 0, hip_stream,
        static_cast<const uint16_t*>(input), static_cast<uint32_t*>(packed),
        static_cast<float*>(scales), static_cast<int32_t*>(zero_points), rows);
  } else if (hidden == 5120) {
    hipLaunchKernelGGL(
        (iu4_pfs_pack_hadamard_bf16_kernel<5120, kGateSeed, false>),
        dim3(static_cast<uint32_t>(rows)), dim3(256), 0, hip_stream,
        static_cast<const uint16_t*>(input), static_cast<uint32_t*>(packed),
        static_cast<float*>(scales), static_cast<int32_t*>(zero_points), rows);
  } else {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_iu4_pfs_pack_swiglu_down_bf16(
    const void* gate_up,
    void* packed,
    void* scales,
    void* zero_points,
    int64_t rows,
    int64_t width,
    void* stream) {
  if (gate_up == nullptr || packed == nullptr || scales == nullptr ||
      zero_points == nullptr || rows <= 0 || rows > 2048 || width != 17408) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (iu4_pfs_pack_hadamard_bf16_kernel<17408, kDownSeed, true>),
      dim3(static_cast<uint32_t>(rows)), dim3(256), 0,
      reinterpret_cast<hipStream_t>(stream),
      static_cast<const uint16_t*>(gate_up), static_cast<uint32_t*>(packed),
      static_cast<float*>(scales), static_cast<int32_t*>(zero_points), rows);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_iu4_pfs_linear_bf16_out(
    const void* activations,
    const void* activation_scales,
    const void* activation_zero_points,
    const void* weights,
    const void* weight_scales,
    const void* weight_sums,
    void* output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    void* stream) {
  if (activations == nullptr || activation_scales == nullptr ||
      activation_zero_points == nullptr || weights == nullptr ||
      weight_scales == nullptr || weight_sums == nullptr || output == nullptr ||
      rows <= 0 || rows > 2048 || in_features <= 0 || in_features % 256 != 0 ||
      out_features <= 0 || out_features % 64 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int OUT_TILES = 4;
  constexpr int ROW_TILES = 4;
  constexpr int K_PAIRS = 8;
  hipLaunchKernelGGL(
      (iu4_pfs_linear_bf16_bulk_kernel<OUT_TILES, ROW_TILES, K_PAIRS>),
      dim3(static_cast<uint32_t>((out_features + 63) / 64),
           static_cast<uint32_t>((rows + 255) / 256)),
      dim3(128), 0, reinterpret_cast<hipStream_t>(stream),
      static_cast<const uint32_t*>(activations),
      static_cast<const float*>(activation_scales),
      static_cast<const int32_t*>(activation_zero_points),
      static_cast<const uint32_t*>(weights),
      static_cast<const float*>(weight_scales),
      static_cast<const int32_t*>(weight_sums),
      static_cast<uint16_t*>(output), rows, in_features, out_features);
  return static_cast<int>(hipGetLastError());
}
