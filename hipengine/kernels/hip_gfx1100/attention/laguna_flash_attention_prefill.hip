// Source-shaped D128/V128 F16-WMMA FlashAttention for Laguna on gfx1100.
//
// Arithmetic/geometry lineage: llama.cpp HIP c0bc8591e8815c63cb01dd3f051a8b0df02501c9,
// ggml/src/ggml-cuda/fattn-mma-f16.cuh (SHA-256 08977d8e...a4038).
// This specialization preserves its eight-query x eight-GQA-head ownership,
// local (32,4), K64 online-softmax batches, F16 WMMA operands, and F32
// score/max/rowsum state. Unlike llama.cpp's contiguous tensor ABI, physical
// K/V addresses and visibility are resolved through every KVLiveSpans field.

#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
using half2_t = half_t __attribute__((ext_vector_type(2)));
using half8_t = half_t __attribute__((ext_vector_type(8)));
using half16_t = half_t __attribute__((ext_vector_type(16)));
using ushort8_t = uint16_t __attribute__((ext_vector_type(8)));
using float2_t = float __attribute__((ext_vector_type(2)));
using float8_t = float __attribute__((ext_vector_type(8)));

constexpr int HEAD_DIM = 128;
constexpr int KV_HEADS = 8;
constexpr int QUERY_ROWS = 8;
constexpr int GQA_HEADS = 8;
constexpr int K_BATCH = 64;
// llama.cpp measures its shared strides in half2 values: +4 means eight
// scalar halves, yielding a 272-byte row aligned for 16-byte vector moves.
constexpr int K_PAD = 8;
constexpr int WAVES = 4;
constexpr int THREADS = 128;
constexpr int GLOBAL_BLOCK_SIZE = 256;
constexpr int TILE_COLUMNS = QUERY_ROWS * GQA_HEADS;
constexpr int SPAN_PAGED_DENSE = 0;
constexpr int SPAN_SLIDING_RING = 1;

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline float8_t zero_float8() {
  return float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}

__device__ inline half2_t zero_half2() {
  return half2_t{0.0f, 0.0f};
}

__device__ inline int physical_slot(
    const int32_t* __restrict__ base_offsets,
    int logical_slot,
    int span_mode,
    int block_size) {
  if (span_mode == SPAN_PAGED_DENSE) {
    const int block = logical_slot / block_size;
    const int within = logical_slot - block * block_size;
    return base_offsets[block] * block_size + within;
  }
  return base_offsets[logical_slot];
}

__device__ inline float lane_xor16(float value) {
  return __shfl_xor(value, 16, 32);
}

__device__ __forceinline__ void process_attention_tile(
    const float* __restrict__ query,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    float* __restrict__ out,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    int rows,
    int capacity,
    int num_q_heads,
    int span_mode,
    int block_size,
    int sliding_window,
    float scale,
    int query_tile,
    int kv_head,
    int gqa_tile,
    half_t (*__restrict__ shared_tile)[HEAD_DIM + K_PAD],
    int* __restrict__ physical_slots,
    int64_t* __restrict__ staged_positions,
    uint8_t* __restrict__ staged_evict) {
  const int lane = static_cast<int>(threadIdx.x);
  const int wave = static_cast<int>(threadIdx.y);
  const int gqa_ratio = num_q_heads / KV_HEADS;
  const int jc = wave * 16 + (lane & 15);
  const int query_local = jc / GQA_HEADS;
  const int gqa_local = jc - query_local * GQA_HEADS;
  const int query_row = query_tile * QUERY_ROWS + query_local;
  const int gqa_index = gqa_tile * GQA_HEADS + gqa_local;
  const int query_head = kv_head * gqa_ratio + gqa_index;
  const bool valid_query = query_row < rows && gqa_index < gqa_ratio;
  const int64_t query_position =
      valid_query ? row_positions[0] + query_row : INT64_MIN;

  const int linear_tid = wave * 32 + lane;
  constexpr int QUERY_PACK_WIDTH = 2;
  constexpr int QUERY_PACKS_PER_ROW = HEAD_DIM / QUERY_PACK_WIDTH;
  const half_t scale_f16 = static_cast<half_t>(scale);
  for (int pack = linear_tid;
       pack < TILE_COLUMNS * QUERY_PACKS_PER_ROW;
       pack += THREADS) {
    const int tile_jc = pack / QUERY_PACKS_PER_ROW;
    const int dim =
        (pack - tile_jc * QUERY_PACKS_PER_ROW) * QUERY_PACK_WIDTH;
    const int tile_query_local = tile_jc / GQA_HEADS;
    const int tile_gqa_local = tile_jc - tile_query_local * GQA_HEADS;
    const int tile_query_row = query_tile * QUERY_ROWS + tile_query_local;
    const int tile_gqa_index = gqa_tile * GQA_HEADS + tile_gqa_local;
    const bool tile_valid =
        tile_query_row < rows && tile_gqa_index < gqa_ratio;
    const int tile_query_head = kv_head * gqa_ratio + tile_gqa_index;
    half2_t values = half2_t{0.0f, 0.0f};
    if (tile_valid) {
      const int64_t index =
          (static_cast<int64_t>(tile_query_row) * num_q_heads +
           tile_query_head) *
              HEAD_DIM +
          dim;
      const float2_t inputs =
          *reinterpret_cast<const float2_t*>(query + index);
      values = half2_t{
          static_cast<half_t>(inputs[0]) * scale_f16,
          static_cast<half_t>(inputs[1]) * scale_f16};
    }
    *reinterpret_cast<half2_t*>(&shared_tile[tile_jc][dim]) = values;
  }
  __syncthreads();

  half16_t query_fragments[HEAD_DIM / 16];
#pragma unroll
  for (int tile = 0; tile < HEAD_DIM / 16; ++tile) {
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      query_fragments[tile][k] = shared_tile[jc][tile * 16 + k];
    }
  }

  // Match llama.cpp's RDNA3 ``tile<16,16,half2,I_MAJOR>`` contract.
  // Each accumulator owns eight half2 values (32 output dimensions across
  // the two lane halves); retaining the half2 storage is important because
  // the intrinsic's opsel calls update opposite halves of the same C tile.
  half2_t output_acc[HEAD_DIM / 32][8];
#pragma unroll
  for (int tile = 0; tile < HEAD_DIM / 32; ++tile) {
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      output_acc[tile][element] = zero_half2();
    }
  }
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  int live = static_cast<int>(live_counts[0]);
  live = max(0, min(live, capacity));
  const int scan_slots = live;
  const int k_batches = (scan_slots + K_BATCH - 1) / K_BATCH;

  for (int batch = 0; batch < k_batches; ++batch) {
    const int k_base = batch * K_BATCH;
    if (linear_tid < K_BATCH) {
      const int logical_slot = k_base + linear_tid;
      if (logical_slot < scan_slots) {
        physical_slots[linear_tid] = physical_slot(
            base_offsets, logical_slot, span_mode, block_size);
        staged_positions[linear_tid] = token_positions[logical_slot];
        staged_evict[linear_tid] = evict_mask[logical_slot];
      } else {
        physical_slots[linear_tid] = 0;
        staged_positions[linear_tid] = -1;
        staged_evict[linear_tid] = 1;
      }
    }
    __syncthreads();
    constexpr int PACK_WIDTH = 8;
    constexpr int PACKS_PER_ROW = HEAD_DIM / PACK_WIDTH;
    for (int pack = linear_tid; pack < K_BATCH * PACKS_PER_ROW;
         pack += THREADS) {
      const int key_local = pack / PACKS_PER_ROW;
      const int dim = (pack - key_local * PACKS_PER_ROW) * PACK_WIDTH;
      const int logical_slot = k_base + key_local;
      half8_t values = half8_t{0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f};
      if (logical_slot < scan_slots) {
        const int64_t index =
            (static_cast<int64_t>(physical_slots[key_local]) * KV_HEADS +
             kv_head) *
                HEAD_DIM +
            dim;
        const ushort8_t bits =
            *reinterpret_cast<const ushort8_t*>(key_cache + index);
#pragma unroll
        for (int element = 0; element < PACK_WIDTH; ++element) {
          values[element] =
              static_cast<half_t>(bf16_bits_to_float(bits[element]));
        }
      }
      *reinterpret_cast<half8_t*>(&shared_tile[key_local][dim]) = values;
    }
    __syncthreads();

    float8_t scores[K_BATCH / 16];
#pragma unroll
    for (int key_tile = 0; key_tile < K_BATCH / 16; ++key_tile) {
      scores[key_tile] = zero_float8();
#pragma unroll
      for (int dim_tile = 0; dim_tile < HEAD_DIM / 16; ++dim_tile) {
        half16_t key_fragment;
        const int key_local = key_tile * 16 + (lane & 15);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          key_fragment[k] = shared_tile[key_local][dim_tile * 16 + k];
        }
        scores[key_tile] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                key_fragment,
                query_fragments[dim_tile],
                scores[key_tile]);
      }
    }
    __syncthreads();

    float local_max = -INFINITY;
#pragma unroll
    for (int key_tile = 0; key_tile < K_BATCH / 16; ++key_tile) {
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const int logical_slot =
            k_base + key_tile * 16 + 2 * element + (lane >> 4);
        bool visible = valid_query && logical_slot < scan_slots;
        int64_t key_position = -1;
        if (visible) {
          const int key_local = logical_slot - k_base;
          key_position = staged_positions[key_local];
          visible = key_position >= 0 && key_position <= query_position &&
                    staged_evict[key_local] == 0;
          if (span_mode == SPAN_SLIDING_RING) {
            visible = visible && sliding_window > 0 &&
                      key_position > query_position - sliding_window;
          }
        }
        if (!visible) {
          scores[key_tile][element] = -INFINITY;
        }
        local_max = fmaxf(local_max, scores[key_tile][element]);
      }
    }
    const float batch_max = fmaxf(local_max, lane_xor16(local_max));
    const bool has_data = isfinite(batch_max);
    const float next_max = has_data ? fmaxf(running_max, batch_max) : 0.0f;
    // A causal query tile can exhaust its visible keys before the shared
    // K64 scan. Empty future-only batches must preserve, not erase, the
    // online-softmax state accumulated by preceding batches.
    const float previous_scale = !has_data
        ? 1.0f
        : isfinite(running_max) ? expf(running_max - next_max) : 0.0f;
#pragma unroll
    for (int tile = 0; tile < HEAD_DIM / 32; ++tile) {
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        output_acc[tile][element][0] = static_cast<half_t>(
            static_cast<float>(output_acc[tile][element][0]) * previous_scale);
        output_acc[tile][element][1] = static_cast<half_t>(
            static_cast<float>(output_acc[tile][element][1]) * previous_scale);
      }
    }

    float local_sum = 0.0f;
#pragma unroll
    for (int key_tile = 0; key_tile < K_BATCH / 16; ++key_tile) {
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const float score = scores[key_tile][element];
        const float probability =
            has_data && isfinite(score) ? expf(score - next_max) : 0.0f;
        scores[key_tile][element] = probability;
        local_sum += probability;
      }
    }
    const float batch_sum = local_sum + lane_xor16(local_sum);
    running_sum = previous_scale * running_sum + batch_sum;
    running_max = has_data ? next_max : running_max;

    for (int pack = linear_tid; pack < K_BATCH * PACKS_PER_ROW;
         pack += THREADS) {
      const int key_local = pack / PACKS_PER_ROW;
      const int dim = (pack - key_local * PACKS_PER_ROW) * PACK_WIDTH;
      const int logical_slot = k_base + key_local;
      half8_t values = half8_t{0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f};
      if (logical_slot < scan_slots) {
        const int64_t index =
            (static_cast<int64_t>(physical_slots[key_local]) * KV_HEADS +
             kv_head) *
                HEAD_DIM +
            dim;
        const ushort8_t bits =
            *reinterpret_cast<const ushort8_t*>(value_cache + index);
#pragma unroll
        for (int element = 0; element < PACK_WIDTH; ++element) {
          values[element] =
              static_cast<half_t>(bf16_bits_to_float(bits[element]));
        }
      }
      *reinterpret_cast<half8_t*>(&shared_tile[key_local][dim]) = values;
    }
    __syncthreads();

#pragma unroll
    for (int key_tile = 0; key_tile < K_BATCH / 16; ++key_tile) {
      // Source ``get_half2`` pairs the score owned by lane_hi=0 with its
      // lane_hi=1 partner before reinterpreting eight half2s as WMMA B.
      half2_t probability_pairs[8];
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const float own = scores[key_tile][element];
        const float partner = lane_xor16(own);
        probability_pairs[element][lane >> 4] = static_cast<half_t>(own);
        probability_pairs[element][(lane >> 4) ^ 1] =
            static_cast<half_t>(partner);
      }
      const half16_t probability_fragment =
          *reinterpret_cast<const half16_t*>(probability_pairs);
#pragma unroll
      for (int dim_tile = 0; dim_tile < HEAD_DIM / 32; ++dim_tile) {
        // Reproduce ``load_ldmatrix_trans(tile<32,8,half2>)`` verbatim:
        // x[0:8] carries the low halves of adjacent V pairs and x[8:16]
        // carries the highs. The two contiguous half16s feed opsel 0/1.
        half2_t value_pairs[16];
        const int dim0 = dim_tile * 32 + 2 * (lane & 15);
        const int dim1 = dim0 + 1;
#pragma unroll
        for (int element = 0; element < 8; ++element) {
          value_pairs[element] = half2_t{
              shared_tile[key_tile * 16 + 2 * element][dim0],
              shared_tile[key_tile * 16 + 2 * element + 1][dim0]};
          value_pairs[element + 8] = half2_t{
              shared_tile[key_tile * 16 + 2 * element][dim1],
              shared_tile[key_tile * 16 + 2 * element + 1][dim1]};
        }
        const half16_t* value_fragments =
            reinterpret_cast<const half16_t*>(value_pairs);
        half16_t accumulator =
            *reinterpret_cast<half16_t*>(output_acc[dim_tile]);
        accumulator = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
            value_fragments[0], probability_fragment, accumulator, 0);
        accumulator = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
            value_fragments[1], probability_fragment, accumulator, 1);
        *reinterpret_cast<half16_t*>(output_acc[dim_tile]) = accumulator;
      }
    }
    __syncthreads();
  }

  if (valid_query && running_sum != 0.0f) {
    const int64_t out_base =
        (static_cast<int64_t>(query_row) * num_q_heads + query_head) * HEAD_DIM;
    const float inverse_sum = 1.0f / running_sum;
#pragma unroll
    for (int dim_tile = 0; dim_tile < HEAD_DIM / 32; ++dim_tile) {
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        // C.get_j(l)=2*l+lane_hi in physical half2 units.
        const int dim = dim_tile * 32 + 4 * element + 2 * (lane >> 4);
        out[out_base + dim] =
            static_cast<float>(output_acc[dim_tile][element][0]) * inverse_sum;
        out[out_base + dim + 1] =
            static_cast<float>(output_acc[dim_tile][element][1]) * inverse_sum;
      }
    }
  }
  __syncthreads();
}

__launch_bounds__(THREADS, 2)
__global__ void laguna_flash_attention_prefill_f16_wmma_whole_kernel(
    const float* __restrict__ query,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    float* __restrict__ out,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    int rows,
    int capacity,
    int num_q_heads,
    int span_mode,
    int block_size,
    int sliding_window,
    float scale) {
  __shared__ half_t shared_tile[K_BATCH][HEAD_DIM + K_PAD];
  __shared__ int physical_slots[K_BATCH];
  __shared__ int64_t staged_positions[K_BATCH];
  __shared__ uint8_t staged_evict[K_BATCH];
  process_attention_tile(
      query,
      key_cache,
      value_cache,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      rows,
      capacity,
      num_q_heads,
      span_mode,
      block_size,
      sliding_window,
      scale,
      static_cast<int>(blockIdx.x),
      static_cast<int>(blockIdx.y),
      static_cast<int>(blockIdx.z),
      shared_tile,
      physical_slots,
      staged_positions,
      staged_evict);
}

}  // namespace

extern "C" int hipengine_laguna_flash_attention_prefill_f16_wmma_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int rows,
    int capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int span_mode,
    int block_size,
    int sliding_window,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      out == nullptr || base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || rows <= 0 || capacity <= 0 ||
      (num_q_heads != 48 && num_q_heads != 72) ||
      num_kv_heads != KV_HEADS || head_dim != HEAD_DIM ||
      (span_mode != SPAN_PAGED_DENSE && span_mode != SPAN_SLIDING_RING) ||
      (span_mode == SPAN_PAGED_DENSE && block_size != GLOBAL_BLOCK_SIZE) ||
      (span_mode == SPAN_SLIDING_RING && sliding_window <= 0)) {
    return hipErrorInvalidValue;
  }

  const int gqa_ratio = num_q_heads / KV_HEADS;
  const int query_tiles = (rows + QUERY_ROWS - 1) / QUERY_ROWS;
  const int gqa_tiles = (gqa_ratio + GQA_HEADS - 1) / GQA_HEADS;
  hipLaunchKernelGGL(
      laguna_flash_attention_prefill_f16_wmma_whole_kernel,
      dim3(query_tiles, KV_HEADS, gqa_tiles),
      dim3(32, WAVES, 1),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      rows,
      capacity,
      num_q_heads,
      span_mode,
      block_size,
      sliding_window,
      scale);
  return static_cast<int>(hipGetLastError());
}
