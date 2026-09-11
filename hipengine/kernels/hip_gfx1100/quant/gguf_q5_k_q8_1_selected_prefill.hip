// Q5_K selected dual gate/up prefill: iu8-WMMA risk route + sparse exact
// repair (llama.cpp mul_mat_q arithmetic class; lineage
// ggml-org/llama.cpp@17252c769 ggml/src/ggml-cuda/mmq.cuh, mma.cuh RDNA3
// int path). Mirrors gguf_q4_k_selected_prefill.hip's production route with
// the Q5_K weight decode (5-bit q = ql | (qh_bit << 4), 176-byte blocks)
// swapped in. The activation side (3-plane residual int8 quantization,
// Kahan-bounded risk collection) is unchanged; weights are the raw Q5_K q
// values [0, 31] exactly, so the weight path has no requantization error.
// The sparse exact repair reproduces the row4 parent
// (gguf_q5_k_selected_grouped_row4_bf16_kernel, 128 threads, strided k
// ownership, shuffle tree + serial wave sum) bit-exactly.

#include <hip/hip_runtime.h>
#include <cfloat>
#include <stdint.h>

#ifndef HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS
#define HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS 2
#endif

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q5_K_BLOCK_BYTES = 176;
constexpr int Q5_K_SCALE_BYTES = 12;
constexpr int Q5_K_QH_OFFSET = 2 + 2 + Q5_K_SCALE_BYTES;  // == 16
constexpr int Q5_K_QS_OFFSET = Q5_K_QH_OFFSET + QK_K / 8;  // == 48
constexpr int Q5_K_SUBBLOCK = 32;
constexpr int Q5_K_SUBBLOCKS = QK_K / Q5_K_SUBBLOCK;      // == 8
constexpr int K_TILE = 16;
constexpr int K_TILES_PER_SUBBLOCK = Q5_K_SUBBLOCK / K_TILE;

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

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  const uint16_t bits = static_cast<uint16_t>(ptr[0]) |
                        (static_cast<uint16_t>(ptr[1]) << 8);
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

__device__ inline uint8_t gguf_q5_k_scale(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] & 0x0Fu) | ((scales[idx] >> 2) & 0x30u);
}

__device__ inline uint8_t gguf_q5_k_min(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[4 + subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30u);
}

// q in [0, 31]: low nibble from qs, 5th bit from the qh plane.
__device__ inline uint8_t gguf_q5_k_quant(
    const uint8_t* block_ptr,
    int subblock,
    int lane32) {
  const uint8_t* qh = block_ptr + Q5_K_QH_OFFSET;
  const uint8_t* qs = block_ptr + Q5_K_QS_OFFSET;
  const uint8_t packed = qs[(subblock >> 1) * 32 + lane32];
  const uint8_t ql = (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
  const uint8_t high = (qh[lane32] >> subblock) & 0x01u;
  return ql | (high << 4);
}

// bf16 rounding boundary distance of the publication value (weight-agnostic
// output-space risk criterion; identical contract to the Q4_K route).
__device__ inline float q5_pair_bf16_rounding_boundary_distance(float value) {
  if (!isfinite(value)) return INFINITY;
  const uint16_t center_bits = float_to_bf16_bits(value);
  uint16_t lower_bits;
  uint16_t upper_bits;
  if ((center_bits & 0x7FFFU) == 0) {
    lower_bits = 0x8001U;
    upper_bits = 0x0001U;
  } else if ((center_bits & 0x8000U) != 0) {
    lower_bits = static_cast<uint16_t>(center_bits + 1U);
    upper_bits = static_cast<uint16_t>(center_bits - 1U);
  } else {
    lower_bits = static_cast<uint16_t>(center_bits - 1U);
    upper_bits = static_cast<uint16_t>(center_bits + 1U);
  }
  const float center = bf16_bits_to_float(center_bits);
  const float lower_boundary =
      0.5f * (bf16_bits_to_float(lower_bits) + center);
  const float upper_boundary =
      0.5f * (center + bf16_bits_to_float(upper_bits));
  return fmaxf(0.0f, fminf(value - lower_boundary, upper_boundary - value));
}

typedef int __attribute__((vector_size(4 * sizeof(int)))) iu8_int32x4;
typedef int __attribute__((vector_size(8 * sizeof(int)))) iu8_int32x8;

constexpr int IU8_LDS_ROW_STRIDE = QK_K + 16;  // bank-conflict padding

bool valid_selected_dual_common(
    const void* x,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* qweight_a,
    const void* qweight_b,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t tile_m,
    int64_t tile_n);

template <typename scalar_t>
__launch_bounds__(128, 2)
__global__ void gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    scalar_t* __restrict__ out,
    int* __restrict__ risk_count,
    int* __restrict__ risk_indices,
    int64_t max_risks,
    float risk_multiplier,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  constexpr int IU8_ACT_PLANES = 3;  // residual quantization planes (ds4 class)
  constexpr float IU8_ATRISK_AMAX_FLOOR = 0x1p-80f;
  __shared__ int lds_row_atrisk[16];
  __shared__ int8_t lds_q8a[IU8_ACT_PLANES * 16 * IU8_LDS_ROW_STRIDE];
  __shared__ float lds_da[IU8_ACT_PLANES * 16 * Q5_K_SUBBLOCKS];
  __shared__ int lds_sumq8a[IU8_ACT_PLANES * 16 * Q5_K_SUBBLOCKS];
  __shared__ int64_t meta[4];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t col_base = static_cast<int64_t>(blockIdx.x) * 128;
  const int64_t row_tile = blockIdx.y;

  if (tid == 0) {
    const int64_t expert = tile_expert[row_tile];
    const bool valid = expert >= 0 && expert < num_experts;
    meta[0] = valid ? expert : -1;
    meta[1] = valid ? expert_start_compact[expert] : 0;
    meta[2] = valid ? (expert_start_compact[expert + 1] - meta[1]) : 0;
    meta[3] = valid ? (row_tile * 16 - expert_start_wmma[expert]) : 0;
  }
  if (tid < 16) {
    lds_row_atrisk[tid] = 0;
  }
  __syncthreads();
  const int64_t expert_id = meta[0];
  if (expert_id < 0) return;
  const int64_t compact_base = meta[1];
  const int64_t expert_count = meta[2];
  const int64_t tile_row_in_expert = meta[3];

  const int64_t global_col = col_base + warp * 32 + lane;
  const bool col_valid = global_col < out_features_total;
  const bool use_b = col_valid && global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const uint8_t* qweight = use_b ? qweight_b : qweight_a;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
  const bool wcol_valid = col_valid && local_col < local_out_features;
  const uint8_t* block_base = wcol_valid
      ? (qweight + expert_id * (local_out_features * row_bytes) +
         local_col * row_bytes)
      : nullptr;

  float sum[16];
  float kahan_err[16];
  float abs_terms[16];
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    sum[r] = 0.0f;
    kahan_err[r] = 0.0f;
    abs_terms[r] = 0.0f;
  }

  // A-operand lane -> activation row (hardware permutation).
  int a_row = -1;
  if ((lane & 1) == 0 && lane < 16) {
    a_row = lane >> 1;
  } else if ((lane & 1) == 1 && lane >= 17) {
    a_row = (lane - 17) >> 1;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    // Cooperative activation staging: thread -> (actrow = tid/8, sub = tid%8).
    {
      const int actrow = tid >> 3;
      const int sub = tid & 7;
      const int64_t local_row = tile_row_in_expert + actrow;
      const bool row_valid = local_row < expert_count;
      const int64_t compact_row = row_valid ? (compact_base + local_row) : 0;
      const scalar_t* row_ptr =
          x + compact_row * in_features + blk * QK_K + sub * Q5_K_SUBBLOCK;
      float vals[32];
      float amax = 0.0f;
#pragma unroll
      for (int l = 0; l < 32; ++l) {
        const float v = row_valid ? scalar_to_float(row_ptr[l]) : 0.0f;
        vals[l] = v;
        amax = fmaxf(amax, fabsf(v));
        if (!isfinite(v)) {
          lds_row_atrisk[actrow] = 1;
        }
      }
      if (row_valid && amax > 0.0f && amax < IU8_ATRISK_AMAX_FLOOR) {
        lds_row_atrisk[actrow] = 1;
      }
      float work[32];
#pragma unroll
      for (int l = 0; l < 32; ++l) work[l] = vals[l];
#pragma unroll
      for (int plane = 0; plane < IU8_ACT_PLANES; ++plane) {
        float plane_max = 0.0f;
#pragma unroll
        for (int l = 0; l < 32; ++l) plane_max = fmaxf(plane_max, fabsf(work[l]));
        const float da = plane_max > 0.0f ? plane_max / 127.0f : 0.0f;
        lds_da[plane * 16 * Q5_K_SUBBLOCKS + actrow * Q5_K_SUBBLOCKS + sub] = da;
        int8_t* dst = lds_q8a + plane * 16 * IU8_LDS_ROW_STRIDE +
                      actrow * IU8_LDS_ROW_STRIDE + sub * Q5_K_SUBBLOCK;
        int qsum = 0;
        if (da > 0.0f) {
          const float inv = 1.0f / da;
#pragma unroll
          for (int l = 0; l < 32; ++l) {
            int q = __float2int_rn(work[l] * inv);
            q = max(-127, min(127, q));
            dst[l] = static_cast<int8_t>(q);
            qsum += q;
            work[l] = fmaf(-static_cast<float>(q), da, work[l]);
          }
        } else {
#pragma unroll
          for (int l = 0; l < 32; ++l) dst[l] = 0;
        }
        lds_sumq8a[plane * 16 * Q5_K_SUBBLOCKS + actrow * Q5_K_SUBBLOCKS + sub] = qsum;
      }
    }
    __syncthreads();

    const uint8_t* block_ptr = block_base + blk * Q5_K_BLOCK_BYTES;
    // Hoist the per-block metadata decode out of the sub loop.
    float d_f = 0.0f, dmin_f = 0.0f;
    float sub_d[Q5_K_SUBBLOCKS];
    float sub_m[Q5_K_SUBBLOCKS];
    {
      if (wcol_valid) {
        d_f = fp16_bytes_to_float(block_ptr);
        dmin_f = fp16_bytes_to_float(block_ptr + 2);
        const uint8_t* scales = block_ptr + 4;
#pragma unroll
        for (int sub = 0; sub < Q5_K_SUBBLOCKS; ++sub) {
          sub_d[sub] = d_f * static_cast<float>(gguf_q5_k_scale(scales, sub));
          sub_m[sub] = dmin_f * static_cast<float>(gguf_q5_k_min(scales, sub));
        }
      } else {
#pragma unroll
        for (int sub = 0; sub < Q5_K_SUBBLOCKS; ++sub) {
          sub_d[sub] = 0.0f;
          sub_m[sub] = 0.0f;
        }
      }
    }
#pragma unroll 1
    for (int sub = 0; sub < Q5_K_SUBBLOCKS; ++sub) {
      // Weight bytes are the raw Q5_K q values (0..31) exactly; the
      // min-offset term is reconstructed per subblock from the staged
      // activation q8 sums (llama mul_mat_q ds-trick), so the weight path has
      // no requantization error at all.
      int8_t wq8[32];
      if (wcol_valid) {
#pragma unroll
        for (int l = 0; l < 32; ++l) {
          wq8[l] = static_cast<int8_t>(gguf_q5_k_quant(block_ptr, sub, l));
        }
      } else {
#pragma unroll
        for (int l = 0; l < 32; ++l) wq8[l] = 0;
      }
      const float dsub = sub_d[sub];
      const float msub = sub_m[sub];

#pragma unroll
      for (int g = 0; g < 2; ++g) {
        iu8_int32x8 acc[IU8_ACT_PLANES];
#pragma unroll
        for (int plane = 0; plane < IU8_ACT_PLANES; ++plane) {
          acc[plane] = {0, 0, 0, 0, 0, 0, 0, 0};
        }
#pragma unroll
        for (int khalf = 0; khalf < 2; ++khalf) {
          signed char bbytes[16];
#pragma unroll
          for (int j = 0; j < 16; ++j) bbytes[j] = wq8[khalf * 16 + j];
          iu8_int32x4 b_frag;
          {
            int32_t words[4];
            memcpy(words, bbytes, 16);
            b_frag = (iu8_int32x4){words[0], words[1], words[2], words[3]};
          }
#pragma unroll
          for (int plane = 0; plane < IU8_ACT_PLANES; ++plane) {
            signed char abytes[16];
            if (a_row >= 0) {
              const int8_t* src = lds_q8a + plane * 16 * IU8_LDS_ROW_STRIDE +
                                  (g * 8 + a_row) * IU8_LDS_ROW_STRIDE +
                                  sub * Q5_K_SUBBLOCK + khalf * 16;
#pragma unroll
              for (int j = 0; j < 16; ++j) abytes[j] = src[j];
            } else {
#pragma unroll
              for (int j = 0; j < 16; ++j) abytes[j] = 0;
            }
            iu8_int32x4 a_frag;
            {
              int32_t words[4];
              memcpy(words, abytes, 16);
              a_frag = (iu8_int32x4){words[0], words[1], words[2], words[3]};
            }
            acc[plane] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, a_frag, true, b_frag, acc[plane], true);
          }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int r = g * 8 + i;
          float main_sum = 0.0f;
          float act_sum = 0.0f;
#pragma unroll
          for (int plane = 0; plane < IU8_ACT_PLANES; ++plane) {
            const float da = lds_da[plane * 16 * Q5_K_SUBBLOCKS + r * Q5_K_SUBBLOCKS + sub];
            const int asumb = lds_sumq8a[plane * 16 * Q5_K_SUBBLOCKS + r * Q5_K_SUBBLOCKS + sub];
            const int* cr = reinterpret_cast<const int*>(&acc[plane]);
            main_sum += da * static_cast<float>(cr[i]);
            act_sum += da * static_cast<float>(asumb);
          }
          const float term = dsub * main_sum - msub * act_sum;
          const float updated = sum[r] + term;
          kahan_err[r] += (sum[r] - updated) + term;
          sum[r] = updated;
          abs_terms[r] += fabsf(term);
        }
      }
    }
    __syncthreads();
  }

  if (!col_valid) return;
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    const int64_t local_row = tile_row_in_expert + r;
    if (local_row < expert_count) {
      const int64_t compact_row = compact_base + local_row;
      if (compact_row < compact_rows) {
        const float value = sum[r];
        out[compact_row * out_features_total + global_col] =
            float_to_scalar<scalar_t>(value);
        // Nonfinite values are always queued: the iu8 activation planes
        // discard NaN/Inf (fmaxf ignores them), so the published value can
        // differ from the exact parent; the sparse repair restores the
        // parent's propagated nonfinite result.
        const float bound = risk_multiplier *
            (fabsf(kahan_err[r]) + FLT_EPSILON * abs_terms[r]);
        const bool at_risk = lds_row_atrisk[r] != 0 ||
            !isfinite(value) ||
            q5_pair_bf16_rounding_boundary_distance(value) < bound;
        if (at_risk) {
          const int slot = atomicAdd(risk_count, 1);
          if (slot < max_risks) {
            risk_indices[slot] = static_cast<int>(
                compact_row * out_features_total + global_col);
          }
        }
      }
    }
  }
}

// Sparse exact repair for the Q5_K iu8-WMMA selected dual gate/up prefill.
//
// For each queued at-risk output index (compact_row * out_features_total +
// global_col), recompute the value with the exact arithmetic of
// gguf_q5_k_selected_grouped_row4_bf16_kernel (the strict row4 parent):
// 128 threads, per-thread strided k ownership (k = tid, tid + 128, ...),
// per-element fp32 dequant product x[k] * q5_k_weight(k), then the
// reduce_block_sum order (shuffle tree per warp, serial wave-0..3 sum by
// thread 0) and the bf16 publication.
__launch_bounds__(128, 2)
__global__ void gguf_q5_k_selected_dual_sparse_exact_repair_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ expert_start,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    const int* __restrict__ risk_count,
    const int* __restrict__ risk_indices,
    int64_t max_risks,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts) {
  __shared__ float wave_sums[4];
  const int64_t out_features_total = out_features_a + out_features_b;
  const int count = static_cast<int>(
      max(static_cast<int64_t>(*risk_count), static_cast<int64_t>(0)));
  const int bounded = static_cast<int>(
      min(static_cast<int64_t>(count), max_risks));
  const int blocks_per_row = static_cast<int>(in_features / QK_K);
  const int64_t row_bytes =
      static_cast<int64_t>(blocks_per_row) * Q5_K_BLOCK_BYTES;
  for (int slot = static_cast<int>(blockIdx.x); slot < bounded;
       slot += static_cast<int>(gridDim.x)) {
    const int64_t idx = risk_indices[slot];
    if (idx < 0 || idx >= compact_rows * out_features_total) continue;
    const int64_t compact_row = idx / out_features_total;
    const int64_t global_col = idx - compact_row * out_features_total;
    if (global_col >= out_features_total) continue;
    // Locate the owning expert with a binary search over group starts.
    int64_t lo = 0;
    int64_t hi = num_experts;  // first start greater than compact_row
    while (lo < hi) {
      const int64_t mid = (lo + hi) >> 1;
      if (expert_start[mid] <= compact_row) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    const int64_t expert = lo - 1;
    if (expert < 0 || expert >= num_experts) continue;
    if (compact_row < expert_start[expert] ||
        compact_row >= expert_start[expert + 1]) {
      continue;
    }
    const bool use_b = global_col >= out_features_a;
    const int64_t local_col =
        use_b ? (global_col - out_features_a) : global_col;
    const int64_t local_out_features = use_b ? out_features_b : out_features_a;
    if (local_col >= local_out_features) continue;
    const uint8_t* __restrict__ weight_row =
        (use_b ? qweight_b : qweight_a) +
        expert * (local_out_features * row_bytes) + local_col * row_bytes;
    const uint16_t* x_row = input + compact_row * in_features;

    // Parent arithmetic: strided per-thread k, fp32 element product.
    float acc = 0.0f;
    for (int k = static_cast<int>(threadIdx.x); k < static_cast<int>(in_features);
         k += static_cast<int>(blockDim.x)) {
      const int block_idx = k / QK_K;
      const int within = k - block_idx * QK_K;
      const int subblock = within >> 5;
      const int lane = within & 31;
      const uint8_t* block =
          weight_row + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
      const float d = fp16_bytes_to_float(block);
      const float dmin = fp16_bytes_to_float(block + 2);
      const uint8_t* scales = block + 4;
      const uint8_t q = gguf_q5_k_quant(block, subblock, lane);
      const float wv = d * static_cast<float>(gguf_q5_k_scale(scales, subblock)) *
              static_cast<float>(q) -
          dmin * static_cast<float>(gguf_q5_k_min(scales, subblock));
      const float xv = bf16_bits_to_float(x_row[k]);
      acc += xv * wv;
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      acc += __shfl_down(acc, delta);
    }
    if ((threadIdx.x & 31) == 0) {
      wave_sums[threadIdx.x >> 5] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
#pragma unroll
      for (int wave = 0; wave < 4; ++wave) {
        total += wave_sums[wave];
      }
      out[compact_row * out_features_total + global_col] =
          float_to_bf16_bits(total);
    }
    __syncthreads();
  }
}

bool valid_selected_dual_common(
    const void* x,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* qweight_a,
    const void* qweight_b,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t tile_m,
    int64_t tile_n) {
  (void)tile_m;
  (void)tile_n;
  return x != nullptr && expert_start_compact != nullptr &&
      expert_start_wmma != nullptr && tile_expert != nullptr &&
      qweight_a != nullptr && qweight_b != nullptr && out != nullptr &&
      compact_rows > 0 && in_features > 0 && (in_features % QK_K) == 0 &&
      out_features_a > 0 && out_features_b > 0 && num_experts > 0 &&
      wmma_total_rows >= 0 && (wmma_total_rows % 16) == 0;
}

template <typename scalar_t>
static int launch_selected_dual_wmma_iu8_risk(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    scalar_t* out,
    int* risk_count,
    int* risk_indices,
    int64_t max_risks,
    float risk_multiplier,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_selected_dual_common(
          x, expert_start_compact, expert_start_wmma, tile_expert, qweight_a,
          qweight_b, out, compact_rows, in_features, out_features_a,
          out_features_b, num_experts, wmma_total_rows, 64, 16)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if ((out_features_a % 128) != 0 || (out_features_b % 128) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (risk_count == nullptr || risk_indices == nullptr || max_risks < 0 ||
      max_risks > INT32_MAX || risk_multiplier < 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 127) / 128),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      (gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_kernel<scalar_t>),
      grid, block, 0, stream,
      x, expert_start_compact, expert_start_wmma, tile_expert, qweight_a,
      qweight_b, out, risk_count, risk_indices, max_risks, risk_multiplier,
      compact_rows, in_features, out_features_a, out_features_b, num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

static int launch_selected_dual_sparse_exact_repair_bf16(
    const uint16_t* input,
    const int64_t* expert_start,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    const int* risk_count,
    const int* risk_indices,
    int64_t max_risks,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t grid_blocks,
    hipStream_t stream) {
  if (input == nullptr || expert_start == nullptr || qweight_a == nullptr ||
      qweight_b == nullptr || out == nullptr || risk_count == nullptr ||
      risk_indices == nullptr || max_risks < 0 || compact_rows <= 0 ||
      num_experts <= 0 || in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features_a <= 0 || out_features_b <= 0 || grid_blocks <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gguf_q5_k_selected_dual_sparse_exact_repair_bf16_kernel),
      dim3(static_cast<unsigned int>(grid_blocks)), dim3(128), 0, stream,
      input, expert_start, qweight_a, qweight_b, out, risk_count,
      risk_indices, max_risks, compact_rows, in_features, out_features_a,
      out_features_b, num_experts);
  return static_cast<int>(hipGetLastError());
}
}  // namespace

extern "C" int hipengine_gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out(
    const void* input,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* qweight_a,
    const void* qweight_b,
    void* output,
    void* risk_count,
    void* risk_indices,
    int64_t max_risks,
    float risk_multiplier,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_wmma_iu8_risk<uint16_t>(
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start_compact),
      static_cast<const int64_t*>(expert_start_wmma),
      static_cast<const int64_t*>(tile_expert),
      static_cast<const uint8_t*>(qweight_a),
      static_cast<const uint8_t*>(qweight_b),
      static_cast<uint16_t*>(output),
      static_cast<int*>(risk_count),
      static_cast<int*>(risk_indices),
      max_risks,
      risk_multiplier,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q5_k_selected_dual_sparse_exact_repair_bf16(
    const void* input,
    const void* expert_start,
    const void* qweight_a,
    const void* qweight_b,
    void* output,
    const void* risk_count,
    const void* risk_indices,
    int64_t max_risks,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t grid_blocks,
    hipStream_t stream) {
  return launch_selected_dual_sparse_exact_repair_bf16(
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start),
      static_cast<const uint8_t*>(qweight_a),
      static_cast<const uint8_t*>(qweight_b),
      static_cast<uint16_t*>(output),
      static_cast<const int*>(risk_count),
      static_cast<const int*>(risk_indices),
      max_risks,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      grid_blocks,
      stream);
}
