// Raw IQ2_XS x D4-Q8_1 selected-MoE MMQ32 prefill for gfx1100.
//
// Source lineage: llama.cpp HIP 1ebf790cda38, ggml-cuda mmq.cuh/mma.cuh.
// Each 128-thread workgroup covers one expert-local 32-row tile and 32
// concatenated gate/up output columns. Four wave32 waves compute independent
// 16x16 integer-WMMA minitiles. Raw IQ2_XS weights are expanded directly into
// signed-byte fragments; caller-owned activations use llama.cpp's K-major D4
// block_q8_1_mmq layout. The grouped scalar primitive remains the fallback.

#include "gguf_iq_gemv.hip"

namespace {

using i32x4_t = int __attribute__((ext_vector_type(4)));
using i32x8_t = int __attribute__((ext_vector_type(8)));

constexpr int IQ2_MMQ_ROWS = 32;
constexpr int IQ2_MMQ_COLS = 32;
constexpr int IQ2_MMQ_K = 16;
constexpr int Q8_MMQ_BLOCK = 128;

struct block_q8_1_mmq_d4 {
  float d4[4];
  int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq_d4) == 144, "unexpected D4 block size");

__device__ inline uint32_t pack_iq2_xs_signed4(
    uint16_t grid,
    uint32_t signs,
    int value_start) {
  uint32_t packed = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int value_index = value_start + j;
    const uint32_t code =
        (static_cast<uint32_t>(grid) >> (2 * value_index)) & 3U;
    const int magnitude = static_cast<int>(8U + 17U * code + (code >> 1));
    const int value = ((signs >> value_index) & 1U) != 0U
        ? -magnitude
        : magnitude;
    packed |= static_cast<uint32_t>(static_cast<uint8_t>(value)) << (8 * j);
  }
  return packed;
}

__device__ inline i32x4_t load_iq2_xs_pair16_i8(
    const uint8_t* __restrict__ block,
    int group8_even) {
  const int group32 = group8_even >> 2;
  const int local8 = group8_even & 3;
  const uint32_t packed_q =
      load_u32_le(block + 2 + 2 * (4 * group32 + local8));
  const uint16_t q0 = static_cast<uint16_t>(packed_q);
  const uint16_t q1 = static_cast<uint16_t>(packed_q >> 16);
  const uint16_t grid0 = IQ2_XS_GRID_PACKED[q0 & 511U];
  const uint16_t grid1 = IQ2_XS_GRID_PACKED[q1 & 511U];
  const uint32_t signs0 = iq3_sign_byte(static_cast<uint32_t>(q0 >> 9));
  const uint32_t signs1 = iq3_sign_byte(static_cast<uint32_t>(q1 >> 9));
  return i32x4_t{
      static_cast<int>(pack_iq2_xs_signed4(grid0, signs0, 0)),
      static_cast<int>(pack_iq2_xs_signed4(grid0, signs0, 4)),
      static_cast<int>(pack_iq2_xs_signed4(grid1, signs1, 0)),
      static_cast<int>(pack_iq2_xs_signed4(grid1, signs1, 4)),
  };
}

__device__ inline float iq2_xs_pair16_scale(
    const uint8_t* __restrict__ block,
    int group8_even) {
  const int group32 = group8_even >> 2;
  const int local8 = group8_even & 3;
  const uint8_t scale_byte = block[66 + group32];
  const uint32_t scale =
      (static_cast<uint32_t>(scale_byte) >> (4 * (local8 >> 1))) & 0x0FU;
  return fp16_bytes_to_float(block) *
         (0.5f + static_cast<float>(scale)) * 0.25f;
}

__device__ inline i32x4_t load_q8_d4_i8x16(
    const block_q8_1_mmq_d4* __restrict__ x_d4,
    int64_t compact_row,
    int64_t compact_rows,
    int block128,
    int within128,
    bool valid) {
  if (!valid) return i32x4_t{0, 0, 0, 0};
  const block_q8_1_mmq_d4& xb =
      x_d4[static_cast<int64_t>(block128) * compact_rows + compact_row];
  const int8_t* __restrict__ q = xb.qs + within128;
  return *reinterpret_cast<const i32x4_t*>(q);
}

__launch_bounds__(128, 2)
__global__ void gguf_iq2_xs_selected_dual_mmq32_prefill_q8_1_d4_kernel(
    const block_q8_1_mmq_d4* __restrict__ x_d4,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_mmq,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int wave = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int lane16 = lane & 15;
  const int half_wave = lane >> 4;
  const int row_half = wave >> 1;
  const int col_half = wave & 1;
  const int64_t tile = static_cast<int64_t>(blockIdx.y);
  const int64_t expert = tile_expert[tile];
  if (expert < 0 || expert >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert];
  const int64_t expert_rows =
      expert_start_compact[expert + 1] - compact_base;
  const int64_t local_tile_base =
      tile * IQ2_MMQ_ROWS - expert_start_mmq[expert];
  const int64_t local_input_row =
      local_tile_base + row_half * 16 + lane16;
  const bool valid_input_row = local_input_row < expert_rows;
  const int64_t compact_input_row =
      valid_input_row ? compact_base + local_input_row : 0;

  const int64_t output_total = 2 * out_features;
  const int64_t global_col =
      static_cast<int64_t>(blockIdx.x) * IQ2_MMQ_COLS +
      col_half * 16 + lane16;
  const bool valid_global_col = global_col < output_total;
  const bool use_up = global_col >= out_features;
  const int64_t local_col = use_up ? global_col - out_features : global_col;
  const bool valid_col = valid_global_col && local_col < out_features;
  const uint8_t* __restrict__ qweight = use_up ? up_weight : gate_weight;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * IQ2_XS_BLOCK_BYTES;
  const int64_t expert_bytes = out_features * row_bytes;
  const uint8_t* __restrict__ weight_row = valid_col
      ? qweight + expert * expert_bytes + local_col * row_bytes
      : qweight;

  __shared__ int weight_tile[IQ2_MMQ_COLS][QK_K / 4];
  __shared__ float weight_scales[IQ2_MMQ_COLS][QK_K / IQ2_MMQ_K];
  const int tile_col = col_half * 16 + lane16;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int64_t block256 = 0; block256 < blocks_per_row; ++block256) {
    // One physical lane per output column expands the raw IQ2 block once.
    // Both row halves and both WMMA half-waves then reuse the packed bytes.
    if (row_half == 0 && half_wave == 0) {
      const uint8_t* __restrict__ iq2_block = valid_col
          ? weight_row + block256 * IQ2_XS_BLOCK_BYTES
          : weight_row;
#pragma unroll
      for (int segment = 0; segment < QK_K / IQ2_MMQ_K; ++segment) {
        const int group8_even = 2 * segment;
        const i32x4_t packed = valid_col
            ? load_iq2_xs_pair16_i8(iq2_block, group8_even)
            : i32x4_t{0, 0, 0, 0};
        *reinterpret_cast<i32x4_t*>(&weight_tile[tile_col][4 * segment]) =
            packed;
        weight_scales[tile_col][segment] = valid_col
            ? iq2_xs_pair16_scale(iq2_block, group8_even)
            : 0.0f;
      }
    }
    __syncthreads();

#pragma unroll
    for (int segment = 0; segment < QK_K / IQ2_MMQ_K; ++segment) {
      const i32x4_t weights = *reinterpret_cast<const i32x4_t*>(
          &weight_tile[tile_col][4 * segment]);
      const float weight_scale = weight_scales[tile_col][segment];
      const int within256 = segment * IQ2_MMQ_K;
      const int64_t k_base = block256 * QK_K + within256;
      const int block128 = static_cast<int>(k_base / Q8_MMQ_BLOCK);
      const int within128 = static_cast<int>(k_base & (Q8_MMQ_BLOCK - 1));
      const i32x4_t activation = load_q8_d4_i8x16(
          x_d4,
          compact_input_row,
          compact_rows,
          block128,
          within128,
          valid_input_row);
      i32x8_t dot = i32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
      dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
          true, activation, true, weights, dot, true);

#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t local_output_row =
            local_tile_base + row_half * 16 + half_wave + 2 * j;
        if (valid_col && local_output_row < expert_rows) {
          const int64_t compact_output_row = compact_base + local_output_row;
          const block_q8_1_mmq_d4& xb =
              x_d4[static_cast<int64_t>(block128) * compact_rows +
                   compact_output_row];
          const float activation_scale = xb.d4[within128 / 32];
          acc[j] += static_cast<float>(dot[j]) *
                    weight_scale * activation_scale;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t local_output_row =
        local_tile_base + row_half * 16 + half_wave + 2 * j;
    if (valid_col && local_output_row < expert_rows) {
      const int64_t compact_output_row = compact_base + local_output_row;
      out[compact_output_row * output_total + global_col] =
          float_to_bf16_bits(acc[j]);
    }
  }
}

int launch_iq2_xs_selected_dual_mmq32(
    const void* x_d4,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq,
    const int64_t* tile_expert,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  if (x_d4 == nullptr || expert_start_compact == nullptr ||
      expert_start_mmq == nullptr || tile_expert == nullptr ||
      gate_weight == nullptr || up_weight == nullptr || out == nullptr ||
      compact_rows <= 0 || in_features <= 0 ||
      (in_features % QK_K) != 0 || out_features <= 0 ||
      num_experts <= 0 || mmq_total_rows <= 0 ||
      (mmq_total_rows % IQ2_MMQ_ROWS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>((2 * out_features + IQ2_MMQ_COLS - 1) /
                                IQ2_MMQ_COLS),
      static_cast<unsigned int>(mmq_total_rows / IQ2_MMQ_ROWS));
  hipLaunchKernelGGL(
      gguf_iq2_xs_selected_dual_mmq32_prefill_q8_1_d4_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const block_q8_1_mmq_d4*>(x_d4),
      expert_start_compact,
      expert_start_mmq,
      tile_expert,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_iq2_xs_selected_dual_mmq32_prefill_q8_1_d4_bf16_bf16_out(
    const void* x_d4,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq,
    const int64_t* tile_expert,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_iq2_xs_selected_dual_mmq32(
      x_d4,
      expert_start_compact,
      expert_start_mmq,
      tile_expert,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}
