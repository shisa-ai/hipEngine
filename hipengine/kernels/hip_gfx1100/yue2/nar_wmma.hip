// YuE2 NAR (acoustic flow matching) attention on BF16/f16 WMMA.
//
// The scalar NAR attention (nar.hip) runs at ~1.3 TFLOP/s: it is scalar FP32
// FMA, and after the softmax reductions, the weight exchange and the loads only
// about a quarter of its instructions are useful arithmetic. The pinned
// upstream's own attention kernel does this workload at 12.4 TFLOP/s on the same
// host with tensor cores, so the remaining distance is arithmetic class rather
// than loop structure. This kernel is the tensor-core path.
//
// Geometry and fragment contracts follow the in-tree Laguna flash attention
// (hipengine/kernels/hip_gfx1100/attention/laguna_flash_attention_prefill.hip),
// which in turn preserves llama.cpp's HIP fattn-mma-f16 design:
//   * a lane owns one (query row, query head) column of the score tile,
//   * the C fragment of a 16x16x16 WMMA holds the lane's eight columns at
//     ``2 * element + (lane >> 4)``, so a row's sixteen keys are split across the
//     two half-wavefronts and ``lane_xor16`` reduces them,
//   * K and V share one staged buffer, 64 keys at a time, and the softmax is
//     online (running max, running sum, rescaled output accumulator),
//   * the output accumulator is f16, as in llama.cpp's RDNA3 path.
//
// Unlike Laguna this attention is bidirectional over the whole key range with no
// causal mask, no eviction and no sliding window: the keys are the visible AR
// conditioning positions followed by the NAR rows themselves, and every query
// sees all of them.
//
// The arithmetic differs from the scalar kernel by design (f16 WMMA operands,
// f16 output accumulator), so this is a production-profile variant: it is held
// to the M4 solver gates, the AR replay floor and the session gate rather than
// to the parent-bits fixture. The scalar kernel stays registered as the strict
// fallback.

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
constexpr int Q_HEADS = 16;
// Query heads that share one key/value head. The NAR model uses 16 query heads
// over 8 key/value heads, so a 16-column WMMA row block covers eight query rows
// for both heads of one key/value head.
constexpr int GQA_HEADS = Q_HEADS / KV_HEADS;
constexpr int ROWS_PER_WAVE = 8;
constexpr int WAVES = 4;
constexpr int THREADS = WAVES * 32;
constexpr int ROWS_PER_BLOCK = WAVES * ROWS_PER_WAVE;
constexpr int TILE_COLUMNS = ROWS_PER_BLOCK * GQA_HEADS;
constexpr int K_BATCH = 64;
// llama.cpp measures its shared strides in half2 values: +4 means eight scalar
// halves, yielding a 272-byte row aligned for 16-byte vector moves.
constexpr int K_PAD = 8;
constexpr int PACK_WIDTH = 8;
constexpr int PACKS_PER_ROW = HEAD_DIM / PACK_WIDTH;

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

__device__ inline float lane_xor16(float value) {
  return __shfl_xor(value, 16, 32);
}

// Stage one 64-key batch of a key/value tensor into the shared tile as f16.
// ``key_base`` is the batch's first key in the concatenated AR + NAR key order;
// keys at or beyond ``ar_rows`` come from the NAR tensor. Rows past ``keys`` are
// zeroed and masked out of the scores below.
__device__ inline void stage_keys(
    half_t (*__restrict__ tile)[HEAD_DIM + K_PAD],
    const uint16_t* __restrict__ ar_key,
    const uint16_t* __restrict__ nar_key,
    int key_base,
    int keys,
    int ar_rows,
    int num_kv_heads,
    int kv_head,
    int linear_tid) {
  for (int pack = linear_tid; pack < K_BATCH * PACKS_PER_ROW;
       pack += THREADS) {
    const int key_local = pack / PACKS_PER_ROW;
    const int dim = (pack - key_local * PACKS_PER_ROW) * PACK_WIDTH;
    const int key = key_base + key_local;
    half8_t values =
        half8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (key < keys) {
      const uint16_t* row = key < ar_rows
          ? ar_key + (static_cast<int64_t>(key) * num_kv_heads + kv_head) * HEAD_DIM
          : nar_key +
                (static_cast<int64_t>(key - ar_rows) * num_kv_heads + kv_head) *
                    HEAD_DIM;
      const ushort8_t bits = *reinterpret_cast<const ushort8_t*>(row + dim);
#pragma unroll
      for (int element = 0; element < PACK_WIDTH; ++element) {
        values[element] = static_cast<half_t>(bf16_bits_to_float(bits[element]));
      }
    }
    *reinterpret_cast<half8_t*>(&tile[key_local][dim]) = values;
  }
}

__global__ void __launch_bounds__(THREADS, 2)
yue2_nar_attention_wmma_kernel(
    const float* __restrict__ query,
    const uint16_t* __restrict__ nar_key,
    const uint16_t* __restrict__ nar_value,
    const uint16_t* __restrict__ ar_key,
    const uint16_t* __restrict__ ar_value,
    float* __restrict__ out,
    int rows,
    int ar_rows,
    int num_q_heads,
    int num_kv_heads,
    float scale) {
  __shared__ half_t shared_tile[K_BATCH][HEAD_DIM + K_PAD];
  __shared__ half_t shared_query[TILE_COLUMNS][HEAD_DIM + K_PAD];

  const int lane = static_cast<int>(threadIdx.x);
  const int wave = static_cast<int>(threadIdx.y);
  const int linear_tid = wave * 32 + lane;
  const int kv_head = static_cast<int>(blockIdx.y);
  const int row_base = static_cast<int>(blockIdx.x) * ROWS_PER_BLOCK;
  const int keys = ar_rows + rows;

  // One column per lane: eight query rows for both query heads of this
  // key/value head. The column index also selects the query row the lane owns.
  const int jc = wave * 16 + (lane & 15);
  const int query_local = jc / GQA_HEADS;
  const int gqa_local = jc - query_local * GQA_HEADS;
  const int query_row = row_base + query_local;
  const int query_head = kv_head * GQA_HEADS + gqa_local;
  const bool valid_query = query_row < rows;

  const half_t scale_f16 = static_cast<half_t>(scale);
  for (int pack = linear_tid; pack < TILE_COLUMNS * (HEAD_DIM / 2);
       pack += THREADS) {
    const int tile_jc = pack / (HEAD_DIM / 2);
    const int dim = (pack - tile_jc * (HEAD_DIM / 2)) * 2;
    const int tile_query_local = tile_jc / GQA_HEADS;
    const int tile_gqa_local = tile_jc - tile_query_local * GQA_HEADS;
    const int tile_query_row = row_base + tile_query_local;
    const int tile_query_head = kv_head * GQA_HEADS + tile_gqa_local;
    half2_t values = zero_half2();
    if (tile_query_row < rows) {
      const int64_t index =
          (static_cast<int64_t>(tile_query_row) * num_q_heads +
           tile_query_head) *
              HEAD_DIM +
          dim;
      const float2_t inputs = *reinterpret_cast<const float2_t*>(query + index);
      values = half2_t{static_cast<half_t>(inputs[0]) * scale_f16,
                       static_cast<half_t>(inputs[1]) * scale_f16};
    }
    *reinterpret_cast<half2_t*>(&shared_query[tile_jc][dim]) = values;
  }
  __syncthreads();

  half16_t query_fragments[HEAD_DIM / 16];
#pragma unroll
  for (int tile = 0; tile < HEAD_DIM / 16; ++tile) {
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      query_fragments[tile][k] = shared_query[jc][tile * 16 + k];
    }
  }

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

  const int k_batches = (keys + K_BATCH - 1) / K_BATCH;
  for (int batch = 0; batch < k_batches; ++batch) {
    const int key_base = batch * K_BATCH;
    stage_keys(
        shared_tile, ar_key, nar_key, key_base, keys, ar_rows, num_kv_heads,
        kv_head, linear_tid);
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
        scores[key_tile] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            key_fragment, query_fragments[dim_tile], scores[key_tile]);
      }
    }
    __syncthreads();

    // Every key in range is visible; the tile is only masked where the key
    // count runs past the batch.
    float local_max = -INFINITY;
#pragma unroll
    for (int key_tile = 0; key_tile < K_BATCH / 16; ++key_tile) {
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const int key_local = key_tile * 16 + 2 * element + (lane >> 4);
        if (key_base + key_local >= keys) {
          scores[key_tile][element] = -INFINITY;
        }
        local_max = fmaxf(local_max, scores[key_tile][element]);
      }
    }
    const float batch_max = fmaxf(local_max, lane_xor16(local_max));
    const bool has_data = isfinite(batch_max);
    const float next_max = has_data ? fmaxf(running_max, batch_max) : 0.0f;
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

    // The value tile reuses the key tile's buffer, so the scores must be fully
    // consumed before this write.
    __syncthreads();
    stage_keys(
        shared_tile, ar_value, nar_value, key_base, keys, ar_rows, num_kv_heads,
        kv_head, linear_tid);
    __syncthreads();

#pragma unroll
    for (int key_tile = 0; key_tile < K_BATCH / 16; ++key_tile) {
      // The score owned by lane_hi=0 is paired with its lane_hi=1 partner
      // before reinterpreting eight half2s as the WMMA B operand.
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
}

}  // namespace

extern "C" int hipengine_yue2_nar_attention_wmma(
    const float* query,
    const uint16_t* nar_key,
    const uint16_t* nar_value,
    const uint16_t* ar_key,
    const uint16_t* ar_value,
    float* out,
    int rows,
    int ar_rows,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || nar_key == nullptr || nar_value == nullptr ||
      out == nullptr || rows <= 0 || ar_rows < 0 || head_dim != HEAD_DIM ||
      num_q_heads != Q_HEADS || num_kv_heads != KV_HEADS ||
      (ar_rows > 0 && (ar_key == nullptr || ar_value == nullptr))) {
    return hipErrorInvalidValue;
  }
  const int row_blocks = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  hipLaunchKernelGGL(
      yue2_nar_attention_wmma_kernel,
      dim3(row_blocks, KV_HEADS, 1),
      dim3(32, WAVES, 1),
      0,
      stream,
      query,
      nar_key,
      nar_value,
      ar_key,
      ar_value,
      out,
      rows,
      ar_rows,
      num_q_heads,
      num_kv_heads,
      scale);
  return static_cast<int>(hipGetLastError());
}
