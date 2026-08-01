// Source-faithful selected-down IQ3_XXS/IQ4_XS integer-MMQ for gfx1100.
//
// Source lineage: llama.cpp HIP c0bc8591e8815c63cb01dd3f051a8b0df02501c9,
// ggml/src/ggml-cuda/{mmq.cuh,mmq-load-tiles.cuh,mmq-vec-dot.cuh,
// mma.cuh}. The port keeps the selected I128/J128/K256, local (32,8),
// Q8_0-weight x Q8_1-activation RDNA3 integer-WMMA dataflow. hipEngine's
// compact expert metadata replaces llama.cpp's expert_bounds/ids_dst ABI, and
// the caller owns K-major 144-byte DS4 activation records. Raw expert-major
// GGUF weights remain resident; no weight sidecar is introduced.

#include "gguf_iq_gemv.hip"

namespace {

using source_i32x4_t = int __attribute__((ext_vector_type(4)));
using source_i32x8_t = int __attribute__((ext_vector_type(8)));

constexpr int IQ_SOURCE_MMQ_I = 128;
constexpr int IQ_SOURCE_MMQ_J = 128;
constexpr int IQ_SOURCE_MMQ_K = 256;
constexpr int IQ_SOURCE_MMQ_THREADS = 256;
constexpr int IQ_SOURCE_WEIGHT_STRIDE_WORDS = 76;
constexpr int IQ_SOURCE_MMQ_SHARED_BYTES = 57856;
constexpr int IQ_SOURCE_Q8_BLOCK = 128;

struct iq_source_q8_1_ds4_kmajor_block {
  uint16_t ds4[8];
  int8_t qs[IQ_SOURCE_Q8_BLOCK];
};
static_assert(sizeof(iq_source_q8_1_ds4_kmajor_block) == 144,
              "unexpected source DS4 Q8_1 block size");

struct iq_source_q8_1_d4x2_kmajor_block {
  float d4[4];
  int8_t qs[IQ_SOURCE_Q8_BLOCK];
};
static_assert(sizeof(iq_source_q8_1_d4x2_kmajor_block) == 144,
              "unexpected residual D4x2 Q8_1 block size");

__device__ inline uint32_t source_pack_i8x4(const int8_t* ptr) {
  return static_cast<uint32_t>(static_cast<uint8_t>(ptr[0])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(ptr[1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(ptr[2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(ptr[3])) << 24);
}

__device__ inline uint32_t iq3_xxs_pack_signed_grid4(
    uint32_t grid,
    uint32_t signs,
    int sign_base) {
  uint32_t packed = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int magnitude = static_cast<int>((grid >> (8 * j)) & 255U);
    const int value = ((signs >> (sign_base + j)) & 1U) != 0U
        ? -magnitude
        : magnitude;
    packed |= static_cast<uint32_t>(static_cast<uint8_t>(value)) << (8 * j);
  }
  return packed;
}

__device__ inline void iq3_xxs_expand_group32(
    const uint8_t* __restrict__ block,
    int group32,
    uint32_t* __restrict__ words,
    float& scale) {
  const uint32_t aux = load_u32_le(block + 66 + 4 * group32);
#pragma unroll
  for (int local8 = 0; local8 < 4; ++local8) {
    const uint32_t signs =
        iq3_sign_byte((aux >> (7 * local8)) & 127U);
    const uint32_t grid0 =
        IQ3_XXS_GRID[block[2 + group32 * 8 + 2 * local8 + 0]];
    const uint32_t grid1 =
        IQ3_XXS_GRID[block[2 + group32 * 8 + 2 * local8 + 1]];
    words[2 * local8 + 0] =
        iq3_xxs_pack_signed_grid4(grid0, signs, 0);
    words[2 * local8 + 1] =
        iq3_xxs_pack_signed_grid4(grid1, signs, 4);
  }
  const float d = fp16_bytes_to_float(block);
  const int ls = static_cast<int>(aux >> 28);
  scale = (static_cast<float>(ls) * d + 0.5f * d) * 0.5f;
}

__device__ inline uint32_t iq4_xs_pack_table4(
    const uint8_t* __restrict__ packed,
    int value_base,
    bool high) {
  uint32_t word = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint8_t q = packed[value_base + j];
    const uint8_t index = high ? (q >> 4) : (q & 0x0FU);
    word |= static_cast<uint32_t>(
                static_cast<uint8_t>(IQ4_NL_VALUES[index]))
            << (8 * j);
  }
  return word;
}

__device__ inline void iq4_xs_expand_group32(
    const uint8_t* __restrict__ block,
    int group32,
    uint32_t* __restrict__ words,
    float& scale) {
  const uint8_t* __restrict__ qs = block + 8 + group32 * 16;
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    words[word] = iq4_xs_pack_table4(qs, 4 * word, false);
    words[word + 4] = iq4_xs_pack_table4(qs, 4 * word, true);
  }
  const uint16_t scales_h = load_u16_le(block + 2);
  const uint8_t scale_l_byte = block[4 + (group32 >> 1)];
  const uint32_t scale_l =
      (static_cast<uint32_t>(scale_l_byte) >> (4 * (group32 & 1))) &
      0x0FU;
  const uint32_t scale_h =
      (static_cast<uint32_t>(scales_h) >> (2 * group32)) & 0x03U;
  const int quant_scale = static_cast<int>(scale_l | (scale_h << 4)) - 32;
  scale = fp16_bytes_to_float(block) * static_cast<float>(quant_scale);
}

template <int QTYPE>
__device__ inline void iq_source_expand_group32(
    const uint8_t* __restrict__ block,
    int group32,
    uint32_t* __restrict__ words,
    float& scale) {
  static_assert(QTYPE == 3 || QTYPE == 4,
                "source IQ MMQ supports IQ3_XXS/IQ4_XS");
  if constexpr (QTYPE == 3) {
    iq3_xxs_expand_group32(block, group32, words, scale);
  } else {
    iq4_xs_expand_group32(block, group32, words, scale);
  }
}

// Eight wave32s own the 128 output columns of one expert-local I128/J128 tile.
// Each K256 raw IQ block is expanded once into signed Q8_0 bytes plus eight
// K32 scales. Each K128 DS4 activation slab is staged once, and each wave
// executes two signed-int8 WMMAs per K32 group across eight J16 row minitiles.
template <int QTYPE>
__launch_bounds__(256, 2)
__global__ void gguf_iq_selected_mmq_i128_j128_k256_q8_1_ds4_kernel(
    const iq_source_q8_1_ds4_kmajor_block* __restrict__ xq,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_mmq,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  extern __shared__ uint8_t shared_bytes[];
  uint32_t* weight_tile = reinterpret_cast<uint32_t*>(shared_bytes);
  iq_source_q8_1_ds4_kmajor_block* activation =
      reinterpret_cast<iq_source_q8_1_ds4_kmajor_block*>(
          weight_tile +
          IQ_SOURCE_MMQ_I * IQ_SOURCE_WEIGHT_STRIDE_WORDS);

  const int wave = static_cast<int>(threadIdx.y);
  const int wave_tid = static_cast<int>(threadIdx.x);
  const int tid = wave * 32 + wave_tid;
  const int lane16 = wave_tid & 15;
  const int half_id = wave_tid >> 4;
  const int64_t tile = static_cast<int64_t>(blockIdx.y);
  const int64_t expert = tile_expert[tile];
  if (expert < 0 || expert >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert];
  const int64_t expert_rows =
      expert_start_compact[expert + 1] - compact_base;
  const int64_t local_row_base =
      tile * IQ_SOURCE_MMQ_J - expert_start_mmq[expert];
  const int64_t output_base =
      static_cast<int64_t>(blockIdx.x) * IQ_SOURCE_MMQ_I;
  const int weight_input_local = wave * 16 + lane16;
  const int64_t blocks_per_weight_row = in_features / QK_K;
  const int block_bytes =
      QTYPE == 3 ? IQ3_XXS_BLOCK_BYTES : IQ4_XS_BLOCK_BYTES;
  const int64_t weight_row_bytes = blocks_per_weight_row * block_bytes;
  const int64_t expert_bytes = out_features * weight_row_bytes;

  float acc[8][8];
#pragma unroll
  for (int rt = 0; rt < 8; ++rt) {
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[rt][j] = 0.0f;
  }

  for (int64_t weight_block = 0;
       weight_block < blocks_per_weight_row;
       ++weight_block) {
    // Four lane8 subgroups per wave expand four output rows concurrently.
    // Four rounds cover all I128 rows while each lane8 owns one K32 group.
    const int load_row_in_wave = wave_tid >> 3;
    const int load_group32 = wave_tid & 7;
#pragma unroll
    for (int load_round = 0; load_round < 4; ++load_round) {
      const int output_round =
          load_round * 32 + wave * 4 + load_row_in_wave;
      const int64_t global_output = output_base + output_round;
      const uint8_t* __restrict__ block =
          qweight + expert * expert_bytes +
          global_output * weight_row_bytes + weight_block * block_bytes;
      uint32_t words[8];
      float scale = 0.0f;
      iq_source_expand_group32<QTYPE>(
          block, load_group32, words, scale);
#pragma unroll
      for (int word = 0; word < 8; ++word) {
        weight_tile[
            output_round * IQ_SOURCE_WEIGHT_STRIDE_WORDS +
            load_group32 * 8 + word] = words[word];
      }
      reinterpret_cast<float*>(
          weight_tile +
          output_round * IQ_SOURCE_WEIGHT_STRIDE_WORDS + 64)[load_group32] =
          scale;
    }
    __syncthreads();

#pragma unroll
    for (int khalf = 0; khalf < 2; ++khalf) {
      const int64_t activation_block = weight_block * 2 + khalf;
      uint32_t* activation_words = reinterpret_cast<uint32_t*>(activation);
      constexpr int words_per_record =
          sizeof(iq_source_q8_1_ds4_kmajor_block) / sizeof(uint32_t);
      for (int index = tid;
           index < IQ_SOURCE_MMQ_J * words_per_record;
           index += IQ_SOURCE_MMQ_THREADS) {
        const int local_row = index / words_per_record;
        const int word = index - local_row * words_per_record;
        const int64_t expert_local_row = local_row_base + local_row;
        uint32_t value = 0;
        if (expert_local_row < expert_rows) {
          const int64_t compact_row = compact_base + expert_local_row;
          const iq_source_q8_1_ds4_kmajor_block* source =
              xq + activation_block * compact_rows + compact_row;
          value = reinterpret_cast<const uint32_t*>(source)[word];
        }
        activation_words[index] = value;
      }
      __syncthreads();

#pragma unroll
      for (int local_group = 0; local_group < 4; ++local_group) {
        const int weight_group = khalf * 4 + local_group;
        const int8_t* weight_values = reinterpret_cast<const int8_t*>(
            weight_tile +
            weight_input_local * IQ_SOURCE_WEIGHT_STRIDE_WORDS) +
            weight_group * 32;
        const source_i32x4_t weight0 = source_i32x4_t{
            static_cast<int>(source_pack_i8x4(weight_values + 0)),
            static_cast<int>(source_pack_i8x4(weight_values + 4)),
            static_cast<int>(source_pack_i8x4(weight_values + 8)),
            static_cast<int>(source_pack_i8x4(weight_values + 12)),
        };
        const source_i32x4_t weight1 = source_i32x4_t{
            static_cast<int>(source_pack_i8x4(weight_values + 16)),
            static_cast<int>(source_pack_i8x4(weight_values + 20)),
            static_cast<int>(source_pack_i8x4(weight_values + 24)),
            static_cast<int>(source_pack_i8x4(weight_values + 28)),
        };

#pragma unroll
        for (int rt = 0; rt < 8; ++rt) {
          const int activation_row = rt * 16 + lane16;
          const int64_t expert_local_row =
              local_row_base + activation_row;
          if (expert_local_row >= expert_rows) continue;
          const iq_source_q8_1_ds4_kmajor_block& xb =
              activation[activation_row];
          const int8_t* activation_values =
              xb.qs + local_group * 32;
          const source_i32x4_t activation0 = source_i32x4_t{
              static_cast<int>(source_pack_i8x4(activation_values + 0)),
              static_cast<int>(source_pack_i8x4(activation_values + 4)),
              static_cast<int>(source_pack_i8x4(activation_values + 8)),
              static_cast<int>(source_pack_i8x4(activation_values + 12)),
          };
          const source_i32x4_t activation1 = source_i32x4_t{
              static_cast<int>(source_pack_i8x4(activation_values + 16)),
              static_cast<int>(source_pack_i8x4(activation_values + 20)),
              static_cast<int>(source_pack_i8x4(activation_values + 24)),
              static_cast<int>(source_pack_i8x4(activation_values + 28)),
          };
          source_i32x8_t dot = source_i32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
          dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, weight0, true, activation0, dot, true);
          dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, weight1, true, activation1, dot, true);
          const float activation_scale =
              fp16_bytes_to_float(
                  reinterpret_cast<const uint8_t*>(
                      &xb.ds4[2 * local_group]));
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int output_local = wave * 16 + half_id + j * 2;
            const float weight_scale = reinterpret_cast<const float*>(
                weight_tile +
                output_local * IQ_SOURCE_WEIGHT_STRIDE_WORDS + 64)[
                    weight_group];
            acc[rt][j] += weight_scale * activation_scale *
                static_cast<float>(dot[j]);
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int rt = 0; rt < 8; ++rt) {
    const int local_row = rt * 16 + lane16;
    const int64_t expert_local_row = local_row_base + local_row;
    if (expert_local_row >= expert_rows) continue;
    const int64_t compact_row = compact_base + expert_local_row;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int output_local = wave * 16 + half_id + j * 2;
      const int64_t output_col = output_base + output_local;
      out[compact_row * out_features + output_col] =
          float_to_bf16_bits(acc[rt][j]);
    }
  }
}

// Physical symbol:
// gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_kernel
#define H7E_JOIN_IMPL(left, right) left##right
#define H7E_JOIN(left, right) H7E_JOIN_IMPL(left, right)
#define H7E_IQ3_D4X2_KERNEL \
  H7E_JOIN( \
      gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_, kernel)

// H7E preserves the source-IQ3 weight tile and WMMA ownership while replaying
// one primary and one residual D4 activation plane against each staged tile.
__launch_bounds__(256, 2)
__global__ void H7E_IQ3_D4X2_KERNEL(
    const iq_source_q8_1_d4x2_kmajor_block* __restrict__ xq,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_mmq,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  extern __shared__ uint8_t shared_bytes[];
  uint32_t* weight_tile = reinterpret_cast<uint32_t*>(shared_bytes);
  iq_source_q8_1_d4x2_kmajor_block* activation =
      reinterpret_cast<iq_source_q8_1_d4x2_kmajor_block*>(
          weight_tile +
          IQ_SOURCE_MMQ_I * IQ_SOURCE_WEIGHT_STRIDE_WORDS);

  const int wave = static_cast<int>(threadIdx.y);
  const int wave_tid = static_cast<int>(threadIdx.x);
  const int tid = wave * 32 + wave_tid;
  const int lane16 = wave_tid & 15;
  const int half_id = wave_tid >> 4;
  const int64_t tile = static_cast<int64_t>(blockIdx.y);
  const int64_t expert = tile_expert[tile];
  if (expert < 0 || expert >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert];
  const int64_t expert_rows =
      expert_start_compact[expert + 1] - compact_base;
  const int64_t local_row_base =
      tile * IQ_SOURCE_MMQ_J - expert_start_mmq[expert];
  const int64_t output_base =
      static_cast<int64_t>(blockIdx.x) * IQ_SOURCE_MMQ_I;
  const int weight_input_local = wave * 16 + lane16;
  const int64_t blocks_per_weight_row = in_features / QK_K;
  const int64_t weight_row_bytes =
      blocks_per_weight_row * IQ3_XXS_BLOCK_BYTES;
  const int64_t expert_bytes = out_features * weight_row_bytes;

  float acc[8][8];
#pragma unroll
  for (int rt = 0; rt < 8; ++rt) {
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[rt][j] = 0.0f;
  }

  for (int64_t weight_block = 0;
       weight_block < blocks_per_weight_row;
       ++weight_block) {
    const int load_row_in_wave = wave_tid >> 3;
    const int load_group32 = wave_tid & 7;
#pragma unroll
    for (int load_round = 0; load_round < 4; ++load_round) {
      const int output_round =
          load_round * 32 + wave * 4 + load_row_in_wave;
      const int64_t global_output = output_base + output_round;
      const uint8_t* __restrict__ block =
          qweight + expert * expert_bytes +
          global_output * weight_row_bytes +
          weight_block * IQ3_XXS_BLOCK_BYTES;
      uint32_t words[8];
      float scale = 0.0f;
      iq3_xxs_expand_group32(block, load_group32, words, scale);
#pragma unroll
      for (int word = 0; word < 8; ++word) {
        weight_tile[
            output_round * IQ_SOURCE_WEIGHT_STRIDE_WORDS +
            load_group32 * 8 + word] = words[word];
      }
      reinterpret_cast<float*>(
          weight_tile +
          output_round * IQ_SOURCE_WEIGHT_STRIDE_WORDS + 64)[load_group32] =
          scale;
    }
    __syncthreads();

#pragma unroll
    for (int khalf = 0; khalf < 2; ++khalf) {
#pragma unroll 1
      for (int activation_pass = 0; activation_pass < 2; ++activation_pass) {
        const int64_t activation_block = weight_block * 2 + khalf;
        const int64_t plane_blocks =
            (in_features / IQ_SOURCE_Q8_BLOCK) * compact_rows;
        uint32_t* activation_words =
            reinterpret_cast<uint32_t*>(activation);
        constexpr int words_per_record =
            sizeof(iq_source_q8_1_d4x2_kmajor_block) / sizeof(uint32_t);
        for (int index = tid;
             index < IQ_SOURCE_MMQ_J * words_per_record;
             index += IQ_SOURCE_MMQ_THREADS) {
          const int local_row = index / words_per_record;
          const int word = index - local_row * words_per_record;
          const int64_t expert_local_row = local_row_base + local_row;
          uint32_t value = 0;
          if (expert_local_row < expert_rows) {
            const int64_t compact_row = compact_base + expert_local_row;
            const iq_source_q8_1_d4x2_kmajor_block* source =
                xq + static_cast<int64_t>(activation_pass) * plane_blocks +
                activation_block * compact_rows + compact_row;
            value = reinterpret_cast<const uint32_t*>(source)[word];
          }
          activation_words[index] = value;
        }
        __syncthreads();

#pragma unroll
        for (int local_group = 0; local_group < 4; ++local_group) {
          const int weight_group = khalf * 4 + local_group;
          const int8_t* weight_values = reinterpret_cast<const int8_t*>(
              weight_tile +
              weight_input_local * IQ_SOURCE_WEIGHT_STRIDE_WORDS) +
              weight_group * 32;
          const source_i32x4_t weight0 = source_i32x4_t{
              static_cast<int>(source_pack_i8x4(weight_values + 0)),
              static_cast<int>(source_pack_i8x4(weight_values + 4)),
              static_cast<int>(source_pack_i8x4(weight_values + 8)),
              static_cast<int>(source_pack_i8x4(weight_values + 12)),
          };
          const source_i32x4_t weight1 = source_i32x4_t{
              static_cast<int>(source_pack_i8x4(weight_values + 16)),
              static_cast<int>(source_pack_i8x4(weight_values + 20)),
              static_cast<int>(source_pack_i8x4(weight_values + 24)),
              static_cast<int>(source_pack_i8x4(weight_values + 28)),
          };

#pragma unroll
          for (int rt = 0; rt < 8; ++rt) {
            const int activation_row = rt * 16 + lane16;
            const int64_t expert_local_row =
                local_row_base + activation_row;
            if (expert_local_row >= expert_rows) continue;
            const iq_source_q8_1_d4x2_kmajor_block& xb =
                activation[activation_row];
            const int8_t* activation_values =
                xb.qs + local_group * 32;
            const source_i32x4_t activation0 = source_i32x4_t{
                static_cast<int>(source_pack_i8x4(activation_values + 0)),
                static_cast<int>(source_pack_i8x4(activation_values + 4)),
                static_cast<int>(source_pack_i8x4(activation_values + 8)),
                static_cast<int>(source_pack_i8x4(activation_values + 12)),
            };
            const source_i32x4_t activation1 = source_i32x4_t{
                static_cast<int>(source_pack_i8x4(activation_values + 16)),
                static_cast<int>(source_pack_i8x4(activation_values + 20)),
                static_cast<int>(source_pack_i8x4(activation_values + 24)),
                static_cast<int>(source_pack_i8x4(activation_values + 28)),
            };
            source_i32x8_t dot =
                source_i32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
            dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, weight0, true, activation0, dot, true);
            dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, weight1, true, activation1, dot, true);
            const float activation_scale = xb.d4[local_group];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const int output_local = wave * 16 + half_id + j * 2;
              const float weight_scale = reinterpret_cast<const float*>(
                  weight_tile +
                  output_local * IQ_SOURCE_WEIGHT_STRIDE_WORDS + 64)[
                      weight_group];
              acc[rt][j] += weight_scale * activation_scale *
                  static_cast<float>(dot[j]);
            }
          }
        }
        __syncthreads();
      }
    }
  }

#pragma unroll
  for (int rt = 0; rt < 8; ++rt) {
    const int local_row = rt * 16 + lane16;
    const int64_t expert_local_row = local_row_base + local_row;
    if (expert_local_row >= expert_rows) continue;
    const int64_t compact_row = compact_base + expert_local_row;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int output_local = wave * 16 + half_id + j * 2;
      const int64_t output_col = output_base + output_local;
      out[compact_row * out_features + output_col] =
          float_to_bf16_bits(acc[rt][j]);
    }
  }
}

template <int QTYPE>
int launch_iq_source_mmq(
    const void* xq,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  if (xq == nullptr || expert_start_compact == nullptr ||
      expert_start_mmq == nullptr || tile_expert == nullptr ||
      qweight == nullptr || out == nullptr || compact_rows <= 0 ||
      in_features <= 0 || (in_features % IQ_SOURCE_MMQ_K) != 0 ||
      out_features <= 0 || (out_features % IQ_SOURCE_MMQ_I) != 0 ||
      num_experts <= 0 || mmq_total_rows <= 0 ||
      (mmq_total_rows % IQ_SOURCE_MMQ_J) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(32, 8);
  const dim3 grid(
      static_cast<unsigned int>(out_features / IQ_SOURCE_MMQ_I),
      static_cast<unsigned int>(mmq_total_rows / IQ_SOURCE_MMQ_J));
  hipLaunchKernelGGL(
      (gguf_iq_selected_mmq_i128_j128_k256_q8_1_ds4_kernel<QTYPE>),
      grid,
      block,
      IQ_SOURCE_MMQ_SHARED_BYTES,
      stream,
      reinterpret_cast<const iq_source_q8_1_ds4_kmajor_block*>(xq),
      expert_start_compact,
      expert_start_mmq,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

int launch_iq3_source_mmq_d4x2(
    const void* xq,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  if (xq == nullptr || expert_start_compact == nullptr ||
      expert_start_mmq == nullptr || tile_expert == nullptr ||
      qweight == nullptr || out == nullptr || compact_rows <= 0 ||
      in_features <= 0 || (in_features % IQ_SOURCE_MMQ_K) != 0 ||
      out_features <= 0 || (out_features % IQ_SOURCE_MMQ_I) != 0 ||
      num_experts <= 0 || mmq_total_rows <= 0 ||
      (mmq_total_rows % IQ_SOURCE_MMQ_J) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(32, 8);
  const dim3 grid(
      static_cast<unsigned int>(out_features / IQ_SOURCE_MMQ_I),
      static_cast<unsigned int>(mmq_total_rows / IQ_SOURCE_MMQ_J));
  hipLaunchKernelGGL(
      H7E_IQ3_D4X2_KERNEL,
      grid,
      block,
      IQ_SOURCE_MMQ_SHARED_BYTES,
      stream,
      reinterpret_cast<const iq_source_q8_1_d4x2_kmajor_block*>(xq),
      expert_start_compact,
      expert_start_mmq,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

#undef H7E_IQ3_D4X2_KERNEL
#undef H7E_JOIN
#undef H7E_JOIN_IMPL

}  // namespace

extern "C" int
hipengine_gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out(
    const void* xq,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_iq_source_mmq<3>(
      xq,
      expert_start_compact,
      expert_start_mmq,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out(
    const void* xq,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_iq3_source_mmq_d4x2(
      xq,
      expert_start_compact,
      expert_start_mmq,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out(
    const void* xq,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_iq_source_mmq<4>(
      xq,
      expert_start_compact,
      expert_start_mmq,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}
