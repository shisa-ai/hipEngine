#include <hip/hip_runtime.h>

#include <cstdint>
#include <cmath>

namespace {

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

__device__ inline uint8_t quantize_u4(float value, float scale, int zero) {
  int q = __float2int_rn(value / scale) + zero;
  q = q < 0 ? 0 : (q > 15 ? 15 : q);
  return static_cast<uint8_t>(q);
}

__launch_bounds__(256, 1)
__global__ void iu4_u4_quantize_bf16_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ packed,
    float* __restrict__ scales,
    int32_t* __restrict__ zero_points,
    int64_t rows,
    int64_t hidden) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (row >= rows) return;

  __shared__ float reductions[512];
  __shared__ float row_scale;
  __shared__ int32_t row_zero;
  float local_min = INFINITY;
  float local_max = -INFINITY;
  const uint16_t* row_input = input + row * hidden;
  for (int64_t k = tid; k < hidden; k += blockDim.x) {
    const float value = bf16_bits_to_float(row_input[k]);
    local_min = fminf(local_min, value);
    local_max = fmaxf(local_max, value);
  }
  reductions[tid] = local_min;
  reductions[256 + tid] = local_max;
  __syncthreads();

  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] = fminf(reductions[tid], reductions[tid + stride]);
      reductions[256 + tid] =
          fmaxf(reductions[256 + tid], reductions[256 + tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    const float lo = fminf(reductions[0], 0.0f);
    const float hi = fmaxf(reductions[256], 0.0f);
    float scale = (hi - lo) / 15.0f;
    if (!(scale > 0.0f) || !isfinite(scale)) scale = 1.0f;
    int zero = __float2int_rn(-lo / scale);
    zero = zero < 0 ? 0 : (zero > 15 ? 15 : zero);
    row_scale = scale;
    row_zero = zero;
    scales[row] = scale;
    zero_points[row] = zero;
  }
  __syncthreads();

  const int64_t row_tile = row / 16;
  const int64_t row_in_tile = row % 16;
  const int64_t k_pairs = hidden / 32;
  for (int64_t byte = tid; byte < hidden / 2; byte += blockDim.x) {
    const uint8_t low = quantize_u4(
        bf16_bits_to_float(row_input[2 * byte]), row_scale, row_zero);
    const uint8_t high = quantize_u4(
        bf16_bits_to_float(row_input[2 * byte + 1]), row_scale, row_zero);
    const int64_t k_pair = byte / 16;
    const int64_t byte_in_pair = byte % 16;
    const int64_t packed_index =
        ((row_tile * k_pairs + k_pair) * 16 + row_in_tile) * 16 + byte_in_pair;
    packed[packed_index] = static_cast<uint8_t>(low | (high << 4));
  }
}

__launch_bounds__(32, 1)
__global__ void iu4_s4_matmul_i32_probe_kernel(
    const uint8_t* __restrict__ activations,
    const uint8_t* __restrict__ weights,
    int32_t* __restrict__ output,
    int64_t rows,
    int64_t hidden,
    int64_t out_features) {
  const int lane = static_cast<int>(threadIdx.x);
  const int lane16 = lane & 15;
  const int half = lane >> 4;
  const int64_t row_base = static_cast<int64_t>(blockIdx.y) * 16;
  const int64_t activation_row = row_base + lane16;
  const int64_t output_col = static_cast<int64_t>(blockIdx.x) * 16 + lane16;
  i32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

  const int64_t k_pairs = hidden / 32;
  for (int64_t k_pair = 0; k_pair < k_pairs; ++k_pair) {
    i32x4_t a_pair = {0, 0, 0, 0};
    i32x4_t b_pair = {0, 0, 0, 0};
    if (activation_row < rows) {
      a_pair = *reinterpret_cast<const i32x4_t*>(
          activations + ((blockIdx.y * k_pairs + k_pair) * 16 + lane16) * 16);
    }
    if (output_col < out_features) {
      b_pair = *reinterpret_cast<const i32x4_t*>(
          weights + ((blockIdx.x * k_pairs + k_pair) * 16 + lane16) * 16);
    }
    const i32x2_t a0 = {a_pair[0], a_pair[1]};
    const i32x2_t a1 = {a_pair[2], a_pair[3]};
    const i32x2_t b0 = {b_pair[0], b_pair[1]};
    const i32x2_t b1 = {b_pair[2], b_pair[3]};
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
        false, a0, true, b0, acc, false);
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
        false, a1, true, b1, acc, false);
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t row = row_base + half + 2 * j;
    if (row < rows && output_col < out_features) {
      output[row * out_features + output_col] = acc[j];
    }
  }
}

// Tiny-M DOT8 owner. Four waves split gate/up and the two four-column halves
// of one output octet. Each packed S4 word is loaded once and reused across
// every active row; independent row/column accumulators hide DOT8 latency.
// This is the bandwidth-shaped B4 leaf, not evidence for reopening tiny-M IU4.
template <int ROW_CAP, int COLS_PER_WAVE>
__launch_bounds__(128, 1)
__global__ void iu4_s4_dual_silu_bf16_dot8_kernel(
    const uint8_t* __restrict__ activations,
    const float* __restrict__ activation_scales,
    const int32_t* __restrict__ activation_zero_points,
    const uint8_t* __restrict__ gate_weights,
    const float* __restrict__ gate_scales,
    const int32_t* __restrict__ gate_sums,
    const uint8_t* __restrict__ up_weights,
    const float* __restrict__ up_scales,
    const int32_t* __restrict__ up_sums,
    uint16_t* __restrict__ output,
    int64_t rows,
    int64_t hidden,
    int64_t out_features) {
  constexpr int COLS_PER_BLOCK = 8;
  constexpr int WAVES_PER_PLANE = COLS_PER_BLOCK / COLS_PER_WAVE;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const bool up_plane = wave >= WAVES_PER_PLANE;
  const int col_base = (wave % WAVES_PER_PLANE) * COLS_PER_WAVE;
  const int64_t output_base = static_cast<int64_t>(blockIdx.x) * COLS_PER_BLOCK;
  const int64_t k_pairs = hidden / 32;
  const int64_t k_packs = hidden / 8;
  const uint8_t* weights = up_plane ? up_weights : gate_weights;

  int32_t acc[ROW_CAP][COLS_PER_WAVE];
#pragma unroll
  for (int row = 0; row < ROW_CAP; ++row) {
#pragma unroll
    for (int col = 0; col < COLS_PER_WAVE; ++col) acc[row][col] = 0;
  }

  for (int64_t k_pack = lane; k_pack < k_packs; k_pack += 32) {
    const int64_t k_pair = k_pack >> 2;
    const int pack_in_pair = static_cast<int>(k_pack & 3);
    uint32_t activation_pack[ROW_CAP];
#pragma unroll
    for (int row = 0; row < ROW_CAP; ++row) {
      activation_pack[row] = 0;
      if (row < rows) {
        const int64_t activation_offset =
            (k_pair * 16 + row) * 16 + pack_in_pair * 4;
        activation_pack[row] = *reinterpret_cast<const uint32_t*>(
            activations + activation_offset);
      }
    }
#pragma unroll
    for (int col = 0; col < COLS_PER_WAVE; ++col) {
      const int64_t output_col = output_base + col_base + col;
      uint32_t weight_pack = 0;
      if (output_col < out_features) {
        const int64_t out_tile = output_col >> 4;
        const int col_in_tile = static_cast<int>(output_col & 15);
        const int64_t weight_offset =
            (out_tile * k_pairs + k_pair) * 16 * 16 +
            col_in_tile * 16 + pack_in_pair * 4;
        weight_pack = *reinterpret_cast<const uint32_t*>(weights + weight_offset);
      }
#pragma unroll
      for (int row = 0; row < ROW_CAP; ++row) {
        acc[row][col] = __builtin_amdgcn_sudot8(
            false,
            static_cast<int32_t>(activation_pack[row]),
            true,
            static_cast<int32_t>(weight_pack),
            acc[row][col],
            false);
      }
    }
  }

#pragma unroll
  for (int row = 0; row < ROW_CAP; ++row) {
#pragma unroll
    for (int col = 0; col < COLS_PER_WAVE; ++col) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[row][col] += __shfl_down(acc[row][col], offset, 32);
      }
    }
  }

  __shared__ uint16_t projection_bits[2][ROW_CAP][COLS_PER_BLOCK];
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < ROW_CAP; ++row) {
      if (row >= rows) continue;
#pragma unroll
      for (int col = 0; col < COLS_PER_WAVE; ++col) {
        const int local_col = col_base + col;
        const int64_t output_col = output_base + local_col;
        if (output_col < out_features) {
          const int32_t* sums = up_plane ? up_sums : gate_sums;
          const float* scales = up_plane ? up_scales : gate_scales;
          const int32_t corrected =
              acc[row][col] - activation_zero_points[row] * sums[output_col];
          projection_bits[up_plane ? 1 : 0][row][local_col] =
              float_to_bf16_bits(
                  static_cast<float>(corrected) * activation_scales[row] *
                  scales[output_col]);
        }
      }
    }
  }
  __syncthreads();

  if (tid < rows * COLS_PER_BLOCK) {
    const int row = tid / COLS_PER_BLOCK;
    const int local_col = tid - row * COLS_PER_BLOCK;
    const int64_t output_col = output_base + local_col;
    if (output_col < out_features) {
      const float gate = bf16_bits_to_float(projection_bits[0][row][local_col]);
      const float up = bf16_bits_to_float(projection_bits[1][row][local_col]);
      const float sigmoid = 1.0f / (1.0f + expf(-gate));
      output[static_cast<int64_t>(row) * out_features + output_col] =
          float_to_bf16_bits(gate * sigmoid * up);
    }
  }
}

// Wider-M owner: every wave owns four M16 tiles and two N16 tiles for gate
// and up (16 independent chains). A paired-K stage loads the complete 8-KiB
// S4 slab once for all four row waves, amortizing one weight sweep over 256
// physical rows like the tuned F16 prefill control.
template <
    int OUT_TILES_PER_BLOCK,
    int ROW_TILES_PER_WAVE,
    int K_PAIRS_PER_STAGE>
__launch_bounds__(128, 1)
__global__ void iu4_s4_dual_silu_bf16_bulk_kernel(
    const uint8_t* __restrict__ activations,
    const float* __restrict__ activation_scales,
    const int32_t* __restrict__ activation_zero_points,
    const uint8_t* __restrict__ gate_weights,
    const float* __restrict__ gate_scales,
    const int32_t* __restrict__ gate_sums,
    const uint8_t* __restrict__ up_weights,
    const float* __restrict__ up_scales,
    const int32_t* __restrict__ up_sums,
    uint16_t* __restrict__ output,
    int64_t rows,
    int64_t hidden,
    int64_t out_features) {
  constexpr int WAVES_PER_BLOCK = 4;
  constexpr int ROWS_PER_WAVE = ROW_TILES_PER_WAVE * 16;
  constexpr int ROWS_PER_BLOCK = WAVES_PER_BLOCK * ROWS_PER_WAVE;
  constexpr int FRAGMENT_BYTES = 16;
  constexpr int SEGMENT_BYTES = K_PAIRS_PER_STAGE * 16 * FRAGMENT_BYTES;
  constexpr int VECTORS_PER_SEGMENT = SEGMENT_BYTES / 16;
  constexpr int SEGMENTS = 2 * OUT_TILES_PER_BLOCK;
  __shared__ __align__(16)
      uint8_t staged_weights[2][OUT_TILES_PER_BLOCK][K_PAIRS_PER_STAGE][16][FRAGMENT_BYTES];

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int lane16 = lane & 15;
  const int half = lane >> 4;
  const int64_t k_pairs = hidden / 32;
  const int64_t block_out_tile =
      static_cast<int64_t>(blockIdx.x) * OUT_TILES_PER_BLOCK;
  const int64_t block_row = static_cast<int64_t>(blockIdx.y) * ROWS_PER_BLOCK;
  const int64_t wave_row = block_row + wave * ROWS_PER_WAVE;

  i32x8_t gate_acc[OUT_TILES_PER_BLOCK][ROW_TILES_PER_WAVE];
  i32x8_t up_acc[OUT_TILES_PER_BLOCK][ROW_TILES_PER_WAVE];
#pragma unroll
  for (int tile = 0; tile < OUT_TILES_PER_BLOCK; ++tile) {
#pragma unroll
    for (int row_tile = 0; row_tile < ROW_TILES_PER_WAVE; ++row_tile) {
      gate_acc[tile][row_tile] = i32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
      up_acc[tile][row_tile] = i32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }

  for (int64_t stage_k_pair = 0; stage_k_pair < k_pairs;
       stage_k_pair += K_PAIRS_PER_STAGE) {
    const bool full_stage = stage_k_pair + K_PAIRS_PER_STAGE <= k_pairs;
    if (full_stage) {
      constexpr int TOTAL_VECTORS = SEGMENTS * VECTORS_PER_SEGMENT;
      for (int index = tid; index < TOTAL_VECTORS; index += blockDim.x) {
        const int segment = index / VECTORS_PER_SEGMENT;
        const int vector_in_segment = index - segment * VECTORS_PER_SEGMENT;
        const int matrix = segment / OUT_TILES_PER_BLOCK;
        const int tile = segment - matrix * OUT_TILES_PER_BLOCK;
        const int64_t out_tile = block_out_tile + tile;
        uint4 value = uint4{0, 0, 0, 0};
        if (out_tile * 16 < out_features) {
          const uint8_t* weights = matrix == 0 ? gate_weights : up_weights;
          const int64_t offset =
              (out_tile * k_pairs + stage_k_pair) * 16 * FRAGMENT_BYTES +
              static_cast<int64_t>(vector_in_segment) * 16;
          value = *reinterpret_cast<const uint4*>(weights + offset);
        }
        reinterpret_cast<uint4*>(staged_weights)[index] = value;
      }
    } else {
      constexpr int TOTAL_FRAGMENTS = SEGMENTS * K_PAIRS_PER_STAGE * 16;
      for (int index = tid; index < TOTAL_FRAGMENTS; index += blockDim.x) {
        const int fragments_per_segment = K_PAIRS_PER_STAGE * 16;
        const int segment = index / fragments_per_segment;
        const int fragment_in_segment = index - segment * fragments_per_segment;
        const int matrix = segment / OUT_TILES_PER_BLOCK;
        const int tile = segment - matrix * OUT_TILES_PER_BLOCK;
        const int local_k_pair = fragment_in_segment / 16;
        const int column = fragment_in_segment - local_k_pair * 16;
        const int64_t out_tile = block_out_tile + tile;
        i32x4_t value = i32x4_t{0, 0, 0, 0};
        if (stage_k_pair + local_k_pair < k_pairs &&
            out_tile * 16 + column < out_features) {
          const uint8_t* weights = matrix == 0 ? gate_weights : up_weights;
          const int64_t offset =
              ((out_tile * k_pairs + stage_k_pair + local_k_pair) * 16 + column) *
              FRAGMENT_BYTES;
          value = *reinterpret_cast<const i32x4_t*>(weights + offset);
        }
        *reinterpret_cast<i32x4_t*>(
            &staged_weights[matrix][tile][local_k_pair][column][0]) = value;
      }
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
          const int64_t activation_tile =
              static_cast<int64_t>(blockIdx.y) *
                  WAVES_PER_BLOCK * ROW_TILES_PER_WAVE +
              wave * ROW_TILES_PER_WAVE + row_tile;
          activation_pair[row_tile] = *reinterpret_cast<const i32x4_t*>(
              activations +
              (activation_tile * k_pairs + k_pair) * 16 * FRAGMENT_BYTES +
              lane16 * FRAGMENT_BYTES);
        }
      }
      i32x4_t gate_fragment[OUT_TILES_PER_BLOCK];
      i32x4_t up_fragment[OUT_TILES_PER_BLOCK];
#pragma unroll
      for (int tile = 0; tile < OUT_TILES_PER_BLOCK; ++tile) {
        gate_fragment[tile] = *reinterpret_cast<const i32x4_t*>(
            &staged_weights[0][tile][local_k_pair][lane16][0]);
        up_fragment[tile] = *reinterpret_cast<const i32x4_t*>(
            &staged_weights[1][tile][local_k_pair][lane16][0]);
      }
#pragma unroll
      for (int tile = 0; tile < OUT_TILES_PER_BLOCK; ++tile) {
        const i32x2_t gate0 = {gate_fragment[tile][0], gate_fragment[tile][1]};
        const i32x2_t up0 = {up_fragment[tile][0], up_fragment[tile][1]};
#pragma unroll
        for (int row_tile = 0; row_tile < ROW_TILES_PER_WAVE; ++row_tile) {
          const i32x2_t activation0 = {
              activation_pair[row_tile][0], activation_pair[row_tile][1]};
          gate_acc[tile][row_tile] =
              __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                  false, activation0, true, gate0,
                  gate_acc[tile][row_tile], false);
          up_acc[tile][row_tile] =
              __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                  false, activation0, true, up0,
                  up_acc[tile][row_tile], false);
        }
      }
#pragma unroll
      for (int tile = 0; tile < OUT_TILES_PER_BLOCK; ++tile) {
        const i32x2_t gate1 = {gate_fragment[tile][2], gate_fragment[tile][3]};
        const i32x2_t up1 = {up_fragment[tile][2], up_fragment[tile][3]};
#pragma unroll
        for (int row_tile = 0; row_tile < ROW_TILES_PER_WAVE; ++row_tile) {
          const i32x2_t activation1 = {
              activation_pair[row_tile][2], activation_pair[row_tile][3]};
          gate_acc[tile][row_tile] =
              __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                  false, activation1, true, gate1,
                  gate_acc[tile][row_tile], false);
          up_acc[tile][row_tile] =
              __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                  false, activation1, true, up1,
                  up_acc[tile][row_tile], false);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int tile = 0; tile < OUT_TILES_PER_BLOCK; ++tile) {
    const int64_t output_col = (block_out_tile + tile) * 16 + lane16;
#pragma unroll
    for (int row_tile = 0; row_tile < ROW_TILES_PER_WAVE; ++row_tile) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t row = wave_row + row_tile * 16 + half + 2 * j;
        if (row < rows && output_col < out_features) {
          const int32_t gate_corrected =
              gate_acc[tile][row_tile][j] -
              activation_zero_points[row] * gate_sums[output_col];
          const int32_t up_corrected =
              up_acc[tile][row_tile][j] -
              activation_zero_points[row] * up_sums[output_col];
          const uint16_t gate_bits = float_to_bf16_bits(
              static_cast<float>(gate_corrected) * activation_scales[row] *
              gate_scales[output_col]);
          const uint16_t up_bits = float_to_bf16_bits(
              static_cast<float>(up_corrected) * activation_scales[row] *
              up_scales[output_col]);
          const float gate = bf16_bits_to_float(gate_bits);
          const float up = bf16_bits_to_float(up_bits);
          const float sigmoid = 1.0f / (1.0f + expf(-gate));
          output[row * out_features + output_col] =
              float_to_bf16_bits(gate * sigmoid * up);
        }
      }
    }
  }
}

}  // namespace

extern "C" int hipengine_iu4_u4_quantize_bf16(
    const void* input,
    void* packed,
    void* scales,
    void* zero_points,
    int64_t rows,
    int64_t hidden,
    void* stream) {
  if (input == nullptr || packed == nullptr || scales == nullptr ||
      zero_points == nullptr || rows <= 0 || rows > 1024 || hidden <= 0 ||
      hidden % 32 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      iu4_u4_quantize_bf16_kernel,
      dim3(static_cast<uint32_t>(rows)),
      dim3(256),
      0,
      reinterpret_cast<hipStream_t>(stream),
      static_cast<const uint16_t*>(input),
      static_cast<uint8_t*>(packed),
      static_cast<float*>(scales),
      static_cast<int32_t*>(zero_points),
      rows,
      hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_iu4_s4_matmul_i32_probe(
    const void* activations,
    const void* weights,
    void* output,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    void* stream) {
  if (activations == nullptr || weights == nullptr || output == nullptr ||
      rows <= 0 || rows > 1024 || hidden <= 0 || hidden % 32 != 0 ||
      out_features <= 0 || out_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      iu4_s4_matmul_i32_probe_kernel,
      dim3(static_cast<uint32_t>(out_features / 16),
           static_cast<uint32_t>((rows + 15) / 16)),
      dim3(32),
      0,
      reinterpret_cast<hipStream_t>(stream),
      static_cast<const uint8_t*>(activations),
      static_cast<const uint8_t*>(weights),
      static_cast<int32_t*>(output),
      rows,
      hidden,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_iu4_s4_dual_silu_bf16_out(
    const void* activations,
    const void* activation_scales,
    const void* activation_zero_points,
    const void* gate_weights,
    const void* gate_scales,
    const void* gate_sums,
    const void* up_weights,
    const void* up_scales,
    const void* up_sums,
    void* output,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    void* stream) {
  if (activations == nullptr || activation_scales == nullptr ||
      activation_zero_points == nullptr || gate_weights == nullptr ||
      gate_scales == nullptr || gate_sums == nullptr || up_weights == nullptr ||
      up_scales == nullptr || up_sums == nullptr || output == nullptr ||
      rows < 2 || rows > 1024 || hidden <= 0 || hidden % 32 != 0 ||
      out_features <= 0 || out_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int OUT_TILES = 2;
  constexpr int ROW_TILES_PER_WAVE = 4;
  constexpr int WAVES = 4;
  const hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
  if (rows <= 16) {
#define HIPENGINE_LAUNCH_IU4_DOT8(ROW_CAP, COLS_PER_WAVE)                    \
    hipLaunchKernelGGL(                                                       \
        (iu4_s4_dual_silu_bf16_dot8_kernel<ROW_CAP, COLS_PER_WAVE>),         \
        dim3(static_cast<uint32_t>(out_features / 8)),                        \
        dim3(2 * (8 / COLS_PER_WAVE) * 32),                                  \
        0,                                                                    \
        hip_stream,                                                           \
        static_cast<const uint8_t*>(activations),                             \
        static_cast<const float*>(activation_scales),                         \
        static_cast<const int32_t*>(activation_zero_points),                  \
        static_cast<const uint8_t*>(gate_weights),                            \
        static_cast<const float*>(gate_scales),                               \
        static_cast<const int32_t*>(gate_sums),                               \
        static_cast<const uint8_t*>(up_weights),                              \
        static_cast<const float*>(up_scales),                                 \
        static_cast<const int32_t*>(up_sums),                                 \
        static_cast<uint16_t*>(output),                                       \
        rows,                                                                 \
        hidden,                                                               \
        out_features)
    if (rows <= 4) {
      HIPENGINE_LAUNCH_IU4_DOT8(4, 4);
    } else if (rows <= 8) {
      HIPENGINE_LAUNCH_IU4_DOT8(8, 4);
    } else {
      HIPENGINE_LAUNCH_IU4_DOT8(16, 4);
    }
#undef HIPENGINE_LAUNCH_IU4_DOT8
  } else {
    constexpr int K_PAIRS_PER_STAGE = 8;
    hipLaunchKernelGGL(
        (iu4_s4_dual_silu_bf16_bulk_kernel<
            OUT_TILES, ROW_TILES_PER_WAVE, K_PAIRS_PER_STAGE>),
        dim3(static_cast<uint32_t>((out_features + OUT_TILES * 16 - 1) /
                                  (OUT_TILES * 16)),
             static_cast<uint32_t>(
                 (rows + WAVES * ROW_TILES_PER_WAVE * 16 - 1) /
                 (WAVES * ROW_TILES_PER_WAVE * 16))),
        dim3(128),
        0,
        hip_stream,
        static_cast<const uint8_t*>(activations),
        static_cast<const float*>(activation_scales),
        static_cast<const int32_t*>(activation_zero_points),
        static_cast<const uint8_t*>(gate_weights),
        static_cast<const float*>(gate_scales),
        static_cast<const int32_t*>(gate_sums),
        static_cast<const uint8_t*>(up_weights),
        static_cast<const float*>(up_scales),
        static_cast<const int32_t*>(up_sums),
        static_cast<uint16_t*>(output),
        rows,
        hidden,
        out_features);
  }
  return static_cast<int>(hipGetLastError());
}
