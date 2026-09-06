// Raw Q8_0 x D4-Q8_1 MMQ128 prefill for gfx1100.
//
// Source lineage: llama.cpp HIP 1ebf790cda38, ggml-cuda mmq.cuh/mma.cuh.
// One 256-thread workgroup computes a 128-output x 128-token tile over K256.
// Raw output-major block_q8_0 rows are staged into the same padded logical
// shared tile as llama.cpp; activations use its K-major block_q8_1_mmq D4
// layout. The exact raw-Q8 tiled family remains the unfused fallback.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
typedef int __attribute__((ext_vector_type(4))) i32x4_t;
typedef int __attribute__((ext_vector_type(8))) i32x8_t;

constexpr int Q8_QK = 32;
constexpr int Q8_BLOCK_BYTES = 34;
constexpr int MMQ_M = 128;
constexpr int MMQ_N = 128;
constexpr int MMQ_K = 256;
constexpr int MMQ_Q8_BLOCKS = MMQ_K / Q8_QK;
constexpr int SPARSE_EXACT_MAX_BLOCKS = 4096;
constexpr int MMQ_QS_I32 = MMQ_K / 4;
constexpr int MMQ_WEIGHT_STRIDE_I32 = 76;  // 64 qs + 8 float d + 4 pad
constexpr int MMQ_D4_STRIDE_I32 = 36;      // 4 float d + 32 qs
constexpr int MMQ_IDS_I32 = MMQ_N;
constexpr int MMQ_ACTIVATION_I32 = MMQ_N * MMQ_D4_STRIDE_I32;
constexpr int MMQ_WEIGHT_I32 = MMQ_M * MMQ_WEIGHT_STRIDE_I32;
constexpr int MMQ_SHARED_BYTES =
    (MMQ_IDS_I32 + MMQ_ACTIVATION_I32 + MMQ_WEIGHT_I32) * sizeof(int);
static_assert(MMQ_SHARED_BYTES == 57856, "unexpected MMQ shared-memory budget");

struct block_q8_1_mmq_d4 {
  float d4[4];
  int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq_d4) == 144, "unexpected D4 block size");

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
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = static_cast<uint16_t>(ptr[0]) |
           (static_cast<uint16_t>(ptr[1]) << 8);
  return static_cast<float>(in.f16);
}

__device__ inline float q8_0_weight(const uint8_t* row, int k) {
  const int block_idx = k / Q8_QK;
  const int lane = k & (Q8_QK - 1);
  const uint8_t* block =
      row + static_cast<int64_t>(block_idx) * Q8_BLOCK_BYTES;
  const float d = fp16_bytes_to_float(block);
  const int8_t q = reinterpret_cast<const int8_t*>(block + 2)[lane];
  return d * static_cast<float>(q);
}

__device__ inline float bf16_rounding_boundary_distance(float value) {
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

__launch_bounds__(256, 2)
__global__ void q8_0_mmq128_quantize_bf16_d4_kernel(
    const uint16_t* __restrict__ x,
    block_q8_1_mmq_d4* __restrict__ out,
    int rows,
    int hidden) {
  const int row = static_cast<int>(blockIdx.x);
  const int i0 = (static_cast<int>(blockIdx.y) * 256 + static_cast<int>(threadIdx.x)) * 4;
  if (row >= rows || i0 >= hidden) return;

  float values[4];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] = bf16_bits_to_float(x[static_cast<int64_t>(row) * hidden + i0 + i]);
    amax = fmaxf(amax, fabsf(values[i]));
  }
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
  }

  const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
  const int block_idx = i0 / 128;
  const int iqs = i0 & 127;
  block_q8_1_mmq_d4* dst = out + static_cast<int64_t>(block_idx) * rows + row;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int q = d == 0.0f ? 0 : static_cast<int>(roundf(values[i] / d));
    q = max(-128, min(127, q));
    dst->qs[iqs + i] = static_cast<int8_t>(q);
  }
  if ((iqs & 31) == 0) dst->d4[iqs >> 5] = d;
}

template <int residual_passes>
__launch_bounds__(256, 2)
__global__ void q8_0_mmq128_quantize_bf16_residual_d4_kernel(
    const uint16_t* __restrict__ x,
    block_q8_1_mmq_d4* __restrict__ out,
    int rows,
    int hidden) {
  const int row = static_cast<int>(blockIdx.x);
  const int i0 = (static_cast<int>(blockIdx.y) * 256 + static_cast<int>(threadIdx.x)) * 4;
  if (row >= rows || i0 >= hidden) return;

  float residual[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    residual[i] = bf16_bits_to_float(x[static_cast<int64_t>(row) * hidden + i0 + i]);
  }

  const int block_idx = i0 / 128;
  const int iqs = i0 & 127;
  const int64_t plane_blocks = static_cast<int64_t>(hidden / 128) * rows;
#pragma unroll
  for (int pass = 0; pass < residual_passes; ++pass) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      amax = fmaxf(amax, fabsf(residual[i]));
    }
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
    block_q8_1_mmq_d4* dst =
        out + static_cast<int64_t>(pass) * plane_blocks +
        static_cast<int64_t>(block_idx) * rows + row;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int q = d == 0.0f ? 0 : static_cast<int>(roundf(residual[i] / d));
      q = max(-128, min(127, q));
      dst->qs[iqs + i] = static_cast<int8_t>(q);
      residual[i] = fmaf(-static_cast<float>(q), d, residual[i]);
    }
    if ((iqs & 31) == 0) dst->d4[iqs >> 5] = d;
  }
}

template <int residual_passes>
__launch_bounds__(256, 2)
__global__ void q8_0_mmq128_quantize_f32_residual_d4_kernel(
    const float* __restrict__ x,
    block_q8_1_mmq_d4* __restrict__ out,
    int rows,
    int hidden) {
  const int row = static_cast<int>(blockIdx.x);
  const int i0 = (static_cast<int>(blockIdx.y) * 256 + static_cast<int>(threadIdx.x)) * 4;
  if (row >= rows || i0 >= hidden) return;

  float residual[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    residual[i] = x[static_cast<int64_t>(row) * hidden + i0 + i];
  }

  const int block_idx = i0 / 128;
  const int iqs = i0 & 127;
  const int64_t plane_blocks = static_cast<int64_t>(hidden / 128) * rows;
#pragma unroll
  for (int pass = 0; pass < residual_passes; ++pass) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      amax = fmaxf(amax, fabsf(residual[i]));
    }
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
    block_q8_1_mmq_d4* dst =
        out + static_cast<int64_t>(pass) * plane_blocks +
        static_cast<int64_t>(block_idx) * rows + row;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int q = d == 0.0f ? 0 : static_cast<int>(roundf(residual[i] / d));
      q = max(-128, min(127, q));
      dst->qs[iqs + i] = static_cast<int8_t>(q);
      residual[i] = fmaf(-static_cast<float>(q), d, residual[i]);
    }
    if ((iqs & 31) == 0) dst->d4[iqs >> 5] = d;
  }
}

__global__ void q8_0_mmq_pack_weights_kernel(
    const uint8_t* __restrict__ raw, int* __restrict__ packed,
    int64_t hidden, int64_t outputs) {
  const int64_t padded = (outputs + 127) / 128 * 128;
  const int64_t words = hidden / 256 * padded * 76;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < words; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int word = index % 76;
    const int64_t row = index / 76;
    const int64_t output = min(row % padded, outputs - 1);
    const int64_t kblock = row / padded;
    if (word < 64) {
      const int64_t raw_block = (output * (hidden / 32) + kblock * 8 + word / 8) * 34;
      packed[index] = *reinterpret_cast<const int*>(raw + raw_block + 2 + (word % 8) * 4);
    } else if (word < 72) {
      const int64_t raw_block = (output * (hidden / 32) + kblock * 8 + word - 64) * 34;
      reinterpret_cast<float*>(packed)[index] = fp16_bytes_to_float(raw + raw_block);
    } else {
      packed[index] = 0;
    }
  }
}

template <int MMQ_M_T, int MMQ_N_T, bool need_output_check, int activation_passes, typename output_t, bool collect_risks, bool PREPACKED = false>
__launch_bounds__(256, 2)
__global__ void q8_0_raw_mmq_tile_q8_1_d4_kernel(
    const block_q8_1_mmq_d4* __restrict__ x_d4,
    const uint8_t* __restrict__ qweight,
    output_t* __restrict__ out,
    int* __restrict__ risk_count,
    int* __restrict__ risk_indices,
    int max_risks,
    float risk_threshold,
    int rows,
    int hidden,
    int out_features) {
  // Tile-generalized variant of the retained 128x128 MMQ kernel. The
  // per-element arithmetic is unchanged: identical integer WMMA fragments in
  // identical (k_iter, k_half, activation_pass, subblock) order, so any
  // (MMQ_M_T, MMQ_N_T) instantiation publishes bit-identical outputs. Only
  // the cooperative load mapping and block/thread geometry adapt: WAVES
  // output rows per wave in the load phase generalize the 8-wave layout, and
  // the compute-phase ownership formulas ((wave/2)*32 rows, (wave&1)*16
  // token offset) cover both 128x128/256-thread and 64x64/128-thread tiles.
  // Occupancy direction informed by halo-box PR11 a7ad7b7f
  // mmq-config-rdna3-5.cuh (gfx1151 tile shrink for residency).
  constexpr int WAVES = MMQ_M_T / 16;
  constexpr int THREADS = WAVES * 32;
  constexpr int IDS_I32 = MMQ_N_T;
  constexpr int ACTIVATION_I32 = MMQ_N_T * MMQ_D4_STRIDE_I32;
  constexpr int WEIGHT_I32 = MMQ_M_T * MMQ_WEIGHT_STRIDE_I32;
  constexpr int SHARED_BYTES = (IDS_I32 + ACTIVATION_I32 + WEIGHT_I32) * sizeof(int);
  (void)SHARED_BYTES;
  const int lane = static_cast<int>(threadIdx.x);
  const int wave = static_cast<int>(threadIdx.y);
  const int tid = wave * 32 + lane;
  const int lane16 = lane & 15;
  const int half_wave = lane >> 4;
  const int out_start = static_cast<int>(blockIdx.x) * MMQ_M_T;
  const int token_start = static_cast<int>(blockIdx.y) * MMQ_N_T;
  const int blocks_per_row = hidden / Q8_QK;
  const int row_bytes = blocks_per_row * Q8_BLOCK_BYTES;
  const int k_iters = hidden / MMQ_K;

  extern __shared__ int shared[];
  int* ids = shared;
  int* activation_tile = ids + IDS_I32;
  int* weight_tile = activation_tile + ACTIVATION_I32;
  if (tid < MMQ_N_T) ids[tid] = tid;
  __syncthreads();

  float acc[MMQ_N_T / 2];
#pragma unroll
  for (int i = 0; i < MMQ_N_T / 2; ++i) acc[i] = 0.0f;

  for (int k_iter = 0; k_iter < k_iters; ++k_iter) {
    if constexpr (PREPACKED) {
      // Packed rows and tail padding match LDS exactly, so tile copies are vectorized.
      const int padded_outputs = (out_features + 127) / 128 * 128;
      const int* src = reinterpret_cast<const int*>(qweight) +
          (static_cast<int64_t>(k_iter) * padded_outputs + out_start) * MMQ_WEIGHT_STRIDE_I32;
      for (int linear = tid * 4; linear < WEIGHT_I32; linear += THREADS * 4) {
        *reinterpret_cast<i32x4_t*>(weight_tile + linear) =
            *reinterpret_cast<const i32x4_t*>(src + linear);
      }
    } else {
    // Exact llama.cpp load_tiles_q8_0 mapping for RDNA3 wave32 generalized to
    // WAVES waves: each wave owns rows i0 + rep*WAVES + wave, and each lane
    // loads two packed int32 fragments from the first/second K128 halves.
#pragma unroll
    for (int i0 = 0; i0 < MMQ_M_T; i0 += 8) {
#pragma unroll
      for (int rep = 0; rep < 8 / WAVES; ++rep) {
        const int row_local = i0 + rep * WAVES + wave;
        const int output_row = need_output_check
            ? min(out_start + row_local, out_features - 1)
            : out_start + row_local;
        const int block0 = k_iter * MMQ_Q8_BLOCKS + lane / 8;
        const int qword = lane & 7;
        const uint8_t* raw_row = qweight + static_cast<int64_t>(output_row) * row_bytes;
        weight_tile[row_local * MMQ_WEIGHT_STRIDE_I32 + lane] =
            *reinterpret_cast<const int*>(raw_row + block0 * Q8_BLOCK_BYTES + 2 + qword * 4);
        weight_tile[row_local * MMQ_WEIGHT_STRIDE_I32 + MMQ_QS_I32 / 2 + lane] =
            *reinterpret_cast<const int*>(
                raw_row + (block0 + 4) * Q8_BLOCK_BYTES + 2 + qword * 4);
      }
    }

    // Eight FP16 block scales per output row become FP32 entries in the padded
    // shared row. WAVES*4 rows are covered at once.
#pragma unroll
    for (int i0 = 0; i0 < MMQ_M_T; i0 += WAVES * 4) {
      const int row_local = i0 + wave * 4 + lane / 8;
      const int local_block = lane & 7;
      const int output_row = need_output_check
          ? min(out_start + row_local, out_features - 1)
          : out_start + row_local;
      const uint8_t* block =
          qweight + static_cast<int64_t>(output_row) * row_bytes +
          (k_iter * MMQ_Q8_BLOCKS + local_block) * Q8_BLOCK_BYTES;
      reinterpret_cast<float*>(weight_tile)[
          row_local * MMQ_WEIGHT_STRIDE_I32 + MMQ_QS_I32 + local_block] =
          fp16_bytes_to_float(block);
    }
    }
    __syncthreads();

#pragma unroll
    for (int k_half = 0; k_half < 2; ++k_half) {
#pragma unroll 1
      for (int activation_pass = 0; activation_pass < activation_passes; ++activation_pass) {
        const int64_t plane_offset =
            static_cast<int64_t>(activation_pass) * (hidden / 128) * rows;
        for (int linear = tid; linear < ACTIVATION_I32; linear += THREADS) {
          const int token_local = linear / MMQ_D4_STRIDE_I32;
          const int elem = linear - token_local * MMQ_D4_STRIDE_I32;
          const int token = min(token_start + token_local, rows - 1);
          const int* src = reinterpret_cast<const int*>(
              x_d4 + plane_offset + static_cast<int64_t>(k_iter * 2 + k_half) * rows + token);
          activation_tile[linear] = src[elem];
        }
        __syncthreads();

        const int output_row_base = (wave / 2) * 32;
        const int token_wave_offset = (wave & 1) * 16;
        // Match llama.cpp: keep the four 32-K scale intervals rolled. Unrolling
        // this loop duplicates every token/output fragment and spills hundreds
        // of VGPRs on gfx1100.
        for (int subblock = 0; subblock < 4; ++subblock) {
          const int weight_kpack = k_half * 32 + subblock * 8;
          i32x4_t avec[2][2];
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            const int row_local = output_row_base + n * 16 + lane16;
            const int* weight_fragment =
                weight_tile + row_local * MMQ_WEIGHT_STRIDE_I32 + weight_kpack;
            avec[n][0] = *reinterpret_cast<const i32x4_t*>(weight_fragment);
            avec[n][1] = *reinterpret_cast<const i32x4_t*>(weight_fragment + 4);
          }

#pragma unroll
          for (int token_group = 0; token_group < MMQ_N_T; token_group += 32) {
            const int token_local = token_wave_offset + token_group + lane16;
            const int* activation_fragment =
                activation_tile + token_local * MMQ_D4_STRIDE_I32 + 4 + subblock * 8;
            i32x4_t bvec[2];
            bvec[0] = *reinterpret_cast<const i32x4_t*>(activation_fragment);
            bvec[1] = *reinterpret_cast<const i32x4_t*>(activation_fragment + 4);
            const float d_activation = reinterpret_cast<const float*>(activation_tile)[
                token_local * MMQ_D4_STRIDE_I32 + subblock];

#pragma unroll
            for (int n = 0; n < 2; ++n) {
              i32x8_t dot = i32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
              dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                  true, avec[n][0], true, bvec[0], dot, true);
              dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                  true, avec[n][1], true, bvec[1], dot, true);
#pragma unroll
              for (int l = 0; l < 8; ++l) {
                const int output_row_local =
                    output_row_base + n * 16 + 2 * l + half_wave;
                const float d_weight = reinterpret_cast<const float*>(weight_tile)[
                    output_row_local * MMQ_WEIGHT_STRIDE_I32 + MMQ_QS_I32 +
                    k_half * 4 + subblock];
                acc[(token_group / 16 + n) * 8 + l] +=
                    static_cast<float>(dot[l]) * d_weight * d_activation;
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }

#pragma unroll
  for (int token_group = 0; token_group < MMQ_N_T; token_group += 32) {
    const int token_local = (wave & 1) * 16 + token_group + lane16;
    const int token = token_start + token_local;
    if (token >= rows) continue;
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        const int output_row_local =
            (wave / 2) * 32 + n * 16 + 2 * l + half_wave;
        const int output_col = out_start + output_row_local;
        if (!need_output_check || output_col < out_features) {
          const float value = acc[(token_group / 16 + n) * 8 + l];
          const int64_t output_idx =
              static_cast<int64_t>(token) * out_features + output_col;
          if constexpr (sizeof(output_t) == sizeof(float)) {
            out[output_idx] = value;
          } else {
            out[output_idx] = float_to_bf16_bits(value);
          }
          if constexpr (collect_risks) {
            const bool at_risk =
                bf16_rounding_boundary_distance(value) <= risk_threshold;
            const unsigned long long risk_mask = __ballot(at_risk);
            if (risk_mask != 0) {
              const int leader = __ffsll(risk_mask) - 1;
              int wave_base = 0;
              if (lane == leader) {
                wave_base = atomicAdd(risk_count, __popcll(risk_mask));
              }
              wave_base = __shfl(wave_base, leader, 32);
              if (at_risk) {
                const unsigned long long lower_lanes =
                    lane == 0 ? 0 : ((1ULL << lane) - 1ULL);
                const int slot =
                    wave_base + __popcll(risk_mask & lower_lanes);
                if (slot < max_risks) {
                  risk_indices[slot] = static_cast<int>(output_idx);
                }
              }
            }
          }
        }
      }
    }
  }
}

__launch_bounds__(128, 2)
__global__ void q8_0_sparse_exact_correct_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    const int* __restrict__ risk_count,
    const int* __restrict__ risk_indices,
    int max_risks,
    int rows,
    int hidden,
    int out_features) {
  const int count = min(max(*risk_count, 0), max_risks);
  const int row_bytes = (hidden / Q8_QK) * Q8_BLOCK_BYTES;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int wave = static_cast<int>(threadIdx.x) >> 5;
  __shared__ float wave_sums[4];

  for (int risk_slot = static_cast<int>(blockIdx.x);
       risk_slot < count;
       risk_slot += static_cast<int>(gridDim.x)) {
    const int output_idx = risk_indices[risk_slot];
    if (output_idx < 0 || output_idx >= rows * out_features) continue;

    const int row = output_idx / out_features;
    const int output_col = output_idx - row * out_features;
    const uint8_t* weight_row =
        qweight + static_cast<int64_t>(output_col) * row_bytes;
    const uint16_t* x_row = x + static_cast<int64_t>(row) * hidden;

    float acc = 0.0f;
    for (int k = static_cast<int>(threadIdx.x); k < hidden; k += 128) {
      const float xv = bf16_bits_to_float(x_row[k]);
      acc += xv * q8_0_weight(weight_row, k);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_down(acc, offset);
    }

    if (lane == 0) wave_sums[wave] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
#pragma unroll
      for (int group = 0; group < 4; ++group) total += wave_sums[group];
      out[output_idx] = float_to_bf16_bits(total);
    }
    __syncthreads();
  }
}

int launch_quantize(
    const uint16_t* x,
    block_q8_1_mmq_d4* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (x == nullptr || out == nullptr || rows <= 0 || hidden <= 0 ||
      (hidden % 128) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>((hidden + 1023) / 1024));
  hipLaunchKernelGGL(
      q8_0_mmq128_quantize_bf16_d4_kernel,
      grid, block, 0, stream, x, out,
      static_cast<int>(rows), static_cast<int>(hidden));
  return static_cast<int>(hipGetLastError());
}

template <int MMQ_M_T, int MMQ_N_T, int activation_passes, typename output_t, bool collect_risks = false, bool PREPACKED = false>
int launch_mmq_tile(
    const block_q8_1_mmq_d4* x_d4,
    const uint8_t* qweight,
    output_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream,
    int* risk_count = nullptr,
    int* risk_indices = nullptr,
    int64_t max_risks = 0,
    float risk_threshold = 0.0f) {
  constexpr int WAVES = MMQ_M_T / 16;
  constexpr int SHARED_BYTES =
      (MMQ_N_T + MMQ_N_T * MMQ_D4_STRIDE_I32 + MMQ_M_T * MMQ_WEIGHT_STRIDE_I32) *
      static_cast<int>(sizeof(int));
  if (x_d4 == nullptr || qweight == nullptr || out == nullptr ||
      rows <= 0 || hidden <= 0 || out_features <= 0 ||
      (hidden % MMQ_K) != 0 || (out_features % 16) != 0 ||
      (collect_risks &&
       (risk_count == nullptr || risk_indices == nullptr || max_risks <= 0 ||
        max_risks > INT32_MAX || risk_threshold < 0.0f))) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(32, WAVES, 1);
  const dim3 grid(
      static_cast<unsigned int>((out_features + MMQ_M_T - 1) / MMQ_M_T),
      static_cast<unsigned int>((rows + MMQ_N_T - 1) / MMQ_N_T),
      1);
  if ((out_features % MMQ_M_T) == 0) {
    hipLaunchKernelGGL(
        (q8_0_raw_mmq_tile_q8_1_d4_kernel<
            MMQ_M_T, MMQ_N_T, false, activation_passes, output_t, collect_risks, PREPACKED>),
        grid, block, SHARED_BYTES, stream,
        x_d4, qweight, out, risk_count, risk_indices,
        static_cast<int>(max_risks), risk_threshold,
        static_cast<int>(rows), static_cast<int>(hidden),
        static_cast<int>(out_features));
  } else {
    hipLaunchKernelGGL(
        (q8_0_raw_mmq_tile_q8_1_d4_kernel<
            MMQ_M_T, MMQ_N_T, true, activation_passes, output_t, collect_risks, PREPACKED>),
        grid, block, SHARED_BYTES, stream,
        x_d4, qweight, out, risk_count, risk_indices,
        static_cast<int>(max_risks), risk_threshold,
        static_cast<int>(rows), static_cast<int>(hidden),
        static_cast<int>(out_features));
  }
  return static_cast<int>(hipGetLastError());
}

template <int activation_passes, typename output_t, bool collect_risks = false>
int launch_mmq(
    const block_q8_1_mmq_d4* x_d4,
    const uint8_t* qweight,
    output_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream,
    int* risk_count = nullptr,
    int* risk_indices = nullptr,
    int64_t max_risks = 0,
    float risk_threshold = 0.0f) {
  return launch_mmq_tile<MMQ_M, MMQ_N, activation_passes, output_t, collect_risks>(
      x_d4, qweight, out, rows, hidden, out_features, stream,
      risk_count, risk_indices, max_risks, risk_threshold);
}

__launch_bounds__(128, 2)
__global__ void q8_0_sparse_exact_correct_f32_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    float* __restrict__ out,
    const int* __restrict__ risk_count,
    const int* __restrict__ risk_indices,
    int max_risks,
    int rows,
    int hidden,
    int out_features) {
  const int count = min(max(*risk_count, 0), max_risks);
  const int row_bytes = (hidden / Q8_QK) * Q8_BLOCK_BYTES;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int wave = static_cast<int>(threadIdx.x) >> 5;
  __shared__ float wave_sums[4];

  for (int risk_slot = static_cast<int>(blockIdx.x);
       risk_slot < count;
       risk_slot += static_cast<int>(gridDim.x)) {
    const int output_idx = risk_indices[risk_slot];
    if (output_idx < 0 || output_idx >= rows * out_features) continue;

    const int row = output_idx / out_features;
    const int output_col = output_idx - row * out_features;
    const uint8_t* weight_row =
        qweight + static_cast<int64_t>(output_col) * row_bytes;
    const float* x_row = x + static_cast<int64_t>(row) * hidden;

    float acc = 0.0f;
    for (int k = static_cast<int>(threadIdx.x); k < hidden; k += 128) {
      acc += x_row[k] * q8_0_weight(weight_row, k);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_down(acc, offset);
    }

    if (lane == 0) wave_sums[wave] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
#pragma unroll
      for (int group = 0; group < 4; ++group) total += wave_sums[group];
      out[output_idx] = total;
    }
    __syncthreads();
  }
}

int launch_sparse_exact_correct(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    const int* risk_count,
    const int* risk_indices,
    int64_t max_risks,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr ||
      risk_count == nullptr || risk_indices == nullptr ||
      max_risks <= 0 || max_risks > INT32_MAX ||
      rows <= 0 || hidden <= 0 || out_features <= 0 ||
      rows * out_features > INT32_MAX || (hidden % Q8_QK) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const int64_t grid_blocks =
      max_risks < SPARSE_EXACT_MAX_BLOCKS ? max_risks : SPARSE_EXACT_MAX_BLOCKS;
  const dim3 grid(static_cast<unsigned int>(grid_blocks));
  hipLaunchKernelGGL(
      q8_0_sparse_exact_correct_bf16_kernel,
      grid, block, 0, stream,
      x, qweight, out, risk_count, risk_indices,
      static_cast<int>(max_risks), static_cast<int>(rows),
      static_cast<int>(hidden), static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q8_0_mmq128_quantize_bf16_d4(
    const uint16_t* x,
    void* out_d4,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_quantize(
      x, reinterpret_cast<block_q8_1_mmq_d4*>(out_d4), rows, hidden, stream);
}

extern "C" int hipengine_gguf_q8_0_mmq128_quantize_bf16_d4x2(
    const uint16_t* x,
    void* out_d4,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (x == nullptr || out_d4 == nullptr || rows <= 0 || hidden <= 0 ||
      (hidden % 128) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>((hidden + 1023) / 1024));
  hipLaunchKernelGGL(
      (q8_0_mmq128_quantize_bf16_residual_d4_kernel<2>),
      grid, block, 0, stream, x,
      reinterpret_cast<block_q8_1_mmq_d4*>(out_d4),
      static_cast<int>(rows), static_cast<int>(hidden));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q8_0_mmq128_quantize_bf16_d4x3(
    const uint16_t* x,
    void* out_d4,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (x == nullptr || out_d4 == nullptr || rows <= 0 || hidden <= 0 ||
      (hidden % 128) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>((hidden + 1023) / 1024));
  hipLaunchKernelGGL(
      (q8_0_mmq128_quantize_bf16_residual_d4_kernel<3>),
      grid, block, 0, stream, x,
      reinterpret_cast<block_q8_1_mmq_d4*>(out_d4),
      static_cast<int>(rows), static_cast<int>(hidden));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4_bf16_bf16_out(
    const void* x_d4,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_mmq<1, uint16_t>(
      reinterpret_cast<const block_q8_1_mmq_d4*>(x_d4),
      qweight, out, rows, hidden, out_features, stream);
}

extern "C" int hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4x2_bf16_bf16_out(
    const void* x_d4,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_mmq<2, uint16_t>(
      reinterpret_cast<const block_q8_1_mmq_d4*>(x_d4),
      qweight, out, rows, hidden, out_features, stream);
}

extern "C" int hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4x3_bf16_bf16_out(
    const void* x_d4,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_mmq<3, uint16_t>(
      reinterpret_cast<const block_q8_1_mmq_d4*>(x_d4),
      qweight, out, rows, hidden, out_features, stream);
}

extern "C" int hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out(
    const void* x_d4,
    const uint8_t* qweight,
    uint16_t* out,
    int* risk_count,
    int* risk_indices,
    int64_t max_risks,
    float risk_threshold,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_mmq<3, uint16_t, true>(
      reinterpret_cast<const block_q8_1_mmq_d4*>(x_d4),
      qweight, out, rows, hidden, out_features, stream,
      risk_count, risk_indices, max_risks, risk_threshold);
}

extern "C" int hipengine_gguf_q8_0_mmq128_sparse_exact_correct_bf16(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    const int* risk_count,
    const int* risk_indices,
    int64_t max_risks,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_sparse_exact_correct(
      x, qweight, out, risk_count, risk_indices, max_risks,
      rows, hidden, out_features, stream);
}

extern "C" int hipengine_gguf_q8_0_mmq128_quantize_f32_d4x3(
    const float* x,
    void* out_d4,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (x == nullptr || out_d4 == nullptr || rows <= 0 || hidden <= 0 ||
      (hidden % 128) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>((hidden + 1023) / 1024));
  hipLaunchKernelGGL(
      (q8_0_mmq128_quantize_f32_residual_d4_kernel<3>),
      grid, block, 0, stream, x,
      reinterpret_cast<block_q8_1_mmq_d4*>(out_d4),
      static_cast<int>(rows), static_cast<int>(hidden));
  return static_cast<int>(hipGetLastError());
}

// PF-1d plane-reduction candidate: double-plane f32 residual quantize.
extern "C" int hipengine_gguf_q8_0_mmq128_quantize_f32_d4x2(
    const float* x,
    void* out_d4,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (x == nullptr || out_d4 == nullptr || rows <= 0 || hidden <= 0 ||
      (hidden % 128) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>((hidden + 1023) / 1024));
  hipLaunchKernelGGL(
      (q8_0_mmq128_quantize_f32_residual_d4_kernel<2>),
      grid, block, 0, stream, x,
      reinterpret_cast<block_q8_1_mmq_d4*>(out_d4),
      static_cast<int>(rows), static_cast<int>(hidden));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out(
    const void* x_d4,
    const uint8_t* qweight,
    float* out,
    int* risk_count,
    int* risk_indices,
    int64_t max_risks,
    float risk_threshold,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_mmq<3, float, true>(
      reinterpret_cast<const block_q8_1_mmq_d4*>(x_d4),
      qweight, out, rows, hidden, out_features, stream,
      risk_count, risk_indices, max_risks, risk_threshold);
}

extern "C" int hipengine_gguf_q8_0_mmq_pack_weights(
    const void* raw, void* packed, int64_t hidden, int64_t outputs,
    hipStream_t stream) {
  if (!raw || !packed || hidden <= 0 || hidden % 256 || outputs <= 0)
    return static_cast<int>(hipErrorInvalidValue);
  const int64_t words = hidden / 256 * ((outputs + 127) / 128 * 128) * 76;
  const uint32_t grid = static_cast<uint32_t>(min((words + 255) / 256, int64_t{4096}));
  hipLaunchKernelGGL(q8_0_mmq_pack_weights_kernel, dim3(grid), dim3(256), 0, stream,
      static_cast<const uint8_t*>(raw), static_cast<int*>(packed), hidden, outputs);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q8_0_mmq128_prepacked_q8_1_d4x3_guarded_f32_f32_out(
    const void* x_d4, const uint8_t* qweight, float* out,
    int* risk_count, int* risk_indices, int64_t max_risks,
    float risk_threshold, int64_t rows, int64_t hidden,
    int64_t out_features, hipStream_t stream) {
  return launch_mmq_tile<128, 128, 3, float, true, true>(
      reinterpret_cast<const block_q8_1_mmq_d4*>(x_d4),
      qweight, out, rows, hidden, out_features, stream,
      risk_count, risk_indices, max_risks, risk_threshold);
}

// PF-1d plane-reduction candidate: guarded f32 chain with two activation
// planes. Not bit-identical to d4x3; admission requires the production
// numerics gate.
extern "C" int hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4x2_guarded_f32_f32_out(
    const void* x_d4,
    const uint8_t* qweight,
    float* out,
    int* risk_count,
    int* risk_indices,
    int64_t max_risks,
    float risk_threshold,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_mmq<2, float, true>(
      reinterpret_cast<const block_q8_1_mmq_d4*>(x_d4),
      qweight, out, rows, hidden, out_features, stream,
      risk_count, risk_indices, max_risks, risk_threshold);
}
// shared — two resident blocks per gfx1151 CU while keeping the 128-token
// tile (weight global traffic unchanged vs the 128x128 reference).

extern "C" int hipengine_gguf_q8_0_mmq128_sparse_exact_correct_f32(
    const float* x,
    const uint8_t* qweight,
    float* out,
    const int* risk_count,
    const int* risk_indices,
    int64_t max_risks,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr ||
      risk_count == nullptr || risk_indices == nullptr ||
      max_risks <= 0 || max_risks > INT32_MAX ||
      rows <= 0 || hidden <= 0 || out_features <= 0 ||
      rows * out_features > INT32_MAX || (hidden % Q8_QK) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const int64_t grid_blocks =
      max_risks < SPARSE_EXACT_MAX_BLOCKS ? max_risks : SPARSE_EXACT_MAX_BLOCKS;
  const dim3 grid(static_cast<unsigned int>(grid_blocks));
  hipLaunchKernelGGL(
      q8_0_sparse_exact_correct_f32_kernel,
      grid, block, 0, stream,
      x, qweight, out, risk_count, risk_indices,
      static_cast<int>(max_risks), static_cast<int>(rows),
      static_cast<int>(hidden), static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4x3_bf16_f32_out(
    const void* x_d4,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_mmq<3, float>(
      reinterpret_cast<const block_q8_1_mmq_d4*>(x_d4),
      qweight, out, rows, hidden, out_features, stream);
}
