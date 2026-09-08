// hipEngine-native raw GGUF Q5_1 selected-expert GEMV for Qwen4Exp.

#include <hip/hip_runtime.h>
#include <cfloat>
#include <stdint.h>

namespace {

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union { uint32_t u32; float f32; } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16_bits(float value) {
  union { float f32; uint32_t u32; } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ inline float load_fp16(const uint8_t* pointer) {
  return static_cast<float>(*reinterpret_cast<const _Float16*>(pointer));
}

__device__ inline float dequant_q5_1(const uint8_t* block, int index) {
  const float d = load_fp16(block);
  const float m = load_fp16(block + 2);
  const uint32_t qh = *reinterpret_cast<const uint32_t*>(block + 4);
  const uint8_t packed = block[8 + (index & 15)];
  const uint8_t low = index < 16 ? (packed & 0x0fU) : (packed >> 4);
  const uint8_t high = static_cast<uint8_t>((qh >> index) & 1U);
  return d * static_cast<float>(low | (high << 4)) + m;
}

template <int ROW_BATCH, int OUT_BATCH, bool SHARED_METADATA = false>
__global__ void q5_1_selected_grouped_prefill_bf16_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ expert_start,
    const uint8_t* __restrict__ weights,
    uint16_t* __restrict__ output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t row_bytes) {
  extern __shared__ float partial[];
  const int64_t blocks_per_row = in_features / 32;
  float* shared_d = partial + ROW_BATCH * blockDim.x;
  float* shared_m = shared_d + blocks_per_row;
  uint32_t* shared_qh = reinterpret_cast<uint32_t*>(shared_m + blocks_per_row);
  const int64_t out_base = blockIdx.x * OUT_BATCH;
  if (out_base >= out_features) return;
  // A full Y grid remains the original one-expert CTA. A smaller fixed worker
  // grid iterates experts while preserving every output's 256-thread reduction.
  for (int64_t expert = blockIdx.y; expert < num_experts; expert += gridDim.y) {
    const int64_t start = expert_start[expert];
    const int64_t end = expert_start[expert + 1];
    if (start < 0 || end < start || end > compact_rows || start == end) continue;
#pragma unroll
    for (int out_offset = 0; out_offset < OUT_BATCH; ++out_offset) {
      const int64_t out = out_base + out_offset;
      if (out >= out_features) break;
      const uint8_t* weight_row = weights
          + (expert * out_features + out) * row_bytes;
      if (SHARED_METADATA) {
        if (threadIdx.x < blocks_per_row) {
          const uint8_t* block = weight_row + threadIdx.x * 24;
          shared_d[threadIdx.x] = load_fp16(block);
          shared_m[threadIdx.x] = load_fp16(block + 2);
          shared_qh[threadIdx.x] =
              *reinterpret_cast<const uint32_t*>(block + 4);
        }
        __syncthreads();
      }
      for (int64_t base = start; base < end; base += ROW_BATCH) {
        float value[ROW_BATCH] = {};
        for (int64_t column = threadIdx.x; column < in_features; column += blockDim.x) {
          const int64_t block_index = column / 32;
          const int within = static_cast<int>(column & 31);
          float weight;
          if (SHARED_METADATA) {
            const uint8_t* block = weight_row + block_index * 24;
            const uint8_t packed = block[8 + (within & 15)];
            const uint8_t low =
                within < 16 ? (packed & 0x0fU) : (packed >> 4);
            const uint8_t high = static_cast<uint8_t>(
                (shared_qh[block_index] >> within) & 1U);
            weight = shared_d[block_index]
                * static_cast<float>(low | (high << 4))
                + shared_m[block_index];
          } else {
            weight = dequant_q5_1(weight_row + block_index * 24, within);
          }
#pragma unroll
          for (int row = 0; row < ROW_BATCH; ++row) {
            const int64_t compact_row = base + row;
            if (compact_row < end) {
              value[row] += bf16_bits_to_float(
                  input[compact_row * in_features + column]) * weight;
            }
          }
        }
#pragma unroll
        for (int row = 0; row < ROW_BATCH; ++row) {
          partial[row * blockDim.x + threadIdx.x] = value[row];
        }
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
          if (threadIdx.x < stride) {
#pragma unroll
            for (int row = 0; row < ROW_BATCH; ++row) {
              partial[row * blockDim.x + threadIdx.x] +=
                  partial[row * blockDim.x + threadIdx.x + stride];
            }
          }
          __syncthreads();
        }
        if (threadIdx.x == 0) {
#pragma unroll
          for (int row = 0; row < ROW_BATCH; ++row) {
            const int64_t compact_row = base + row;
            if (compact_row < end) {
              output[compact_row * out_features + out] =
                  float_to_bf16_bits(partial[row * blockDim.x]);
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

// Exact logical-lane contraction: 128 physical threads materialize the same
// 256 strict partials, then execute the original stride-128 -> 1 tree.
template <int ROW_BATCH, int OUT_BATCH>
__global__ void q5_1_selected_grouped_prefill_logical256_bf16_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ expert_start,
    const uint8_t* __restrict__ weights,
    uint16_t* __restrict__ output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t row_bytes) {
  constexpr int LOGICAL_THREADS = 256;
  extern __shared__ float partial[];
  const int64_t blocks_per_row = in_features / 32;
  float* shared_d = partial + ROW_BATCH * LOGICAL_THREADS;
  float* shared_m = shared_d + blocks_per_row;
  uint32_t* shared_qh = reinterpret_cast<uint32_t*>(shared_m + blocks_per_row);
  const int64_t out_base = blockIdx.x * OUT_BATCH;
  if (out_base >= out_features) return;
  for (int64_t expert = blockIdx.y; expert < num_experts; expert += gridDim.y) {
    const int64_t start = expert_start[expert];
    const int64_t end = expert_start[expert + 1];
    if (start < 0 || end < start || end > compact_rows || start == end) continue;
#pragma unroll
    for (int out_offset = 0; out_offset < OUT_BATCH; ++out_offset) {
      const int64_t out = out_base + out_offset;
      if (out >= out_features) break;
      const uint8_t* weight_row = weights
          + (expert * out_features + out) * row_bytes;
      if (threadIdx.x < blocks_per_row) {
        const uint8_t* block = weight_row + threadIdx.x * 24;
        shared_d[threadIdx.x] = load_fp16(block);
        shared_m[threadIdx.x] = load_fp16(block + 2);
        shared_qh[threadIdx.x] =
            *reinterpret_cast<const uint32_t*>(block + 4);
      }
      __syncthreads();
      for (int64_t base = start; base < end; base += ROW_BATCH) {
        float value0[ROW_BATCH] = {};
        float value1[ROW_BATCH] = {};
        const int logical0 = threadIdx.x;
        const int logical1 = threadIdx.x + blockDim.x;
        for (int64_t column = logical0; column < in_features;
             column += LOGICAL_THREADS) {
          const int64_t block_index = column / 32;
          const int within = static_cast<int>(column & 31);
          const uint8_t* block = weight_row + block_index * 24;
          const uint8_t packed = block[8 + (within & 15)];
          const uint8_t low =
              within < 16 ? (packed & 0x0fU) : (packed >> 4);
          const uint8_t high = static_cast<uint8_t>(
              (shared_qh[block_index] >> within) & 1U);
          const float weight = shared_d[block_index]
              * static_cast<float>(low | (high << 4))
              + shared_m[block_index];
#pragma unroll
          for (int row = 0; row < ROW_BATCH; ++row) {
            const int64_t compact_row = base + row;
            if (compact_row < end) {
              value0[row] += bf16_bits_to_float(
                  input[compact_row * in_features + column]) * weight;
            }
          }
        }
        for (int64_t column = logical1; column < in_features;
             column += LOGICAL_THREADS) {
          const int64_t block_index = column / 32;
          const int within = static_cast<int>(column & 31);
          const uint8_t* block = weight_row + block_index * 24;
          const uint8_t packed = block[8 + (within & 15)];
          const uint8_t low =
              within < 16 ? (packed & 0x0fU) : (packed >> 4);
          const uint8_t high = static_cast<uint8_t>(
              (shared_qh[block_index] >> within) & 1U);
          const float weight = shared_d[block_index]
              * static_cast<float>(low | (high << 4))
              + shared_m[block_index];
#pragma unroll
          for (int row = 0; row < ROW_BATCH; ++row) {
            const int64_t compact_row = base + row;
            if (compact_row < end) {
              value1[row] += bf16_bits_to_float(
                  input[compact_row * in_features + column]) * weight;
            }
          }
        }
#pragma unroll
        for (int row = 0; row < ROW_BATCH; ++row) {
          partial[row * LOGICAL_THREADS + logical0] = value0[row];
          partial[row * LOGICAL_THREADS + logical1] = value1[row];
        }
        __syncthreads();
        for (int stride = LOGICAL_THREADS / 2; stride > 0; stride >>= 1) {
          if (threadIdx.x < stride) {
#pragma unroll
            for (int row = 0; row < ROW_BATCH; ++row) {
              partial[row * LOGICAL_THREADS + threadIdx.x] +=
                  partial[row * LOGICAL_THREADS + threadIdx.x + stride];
            }
          }
          __syncthreads();
        }
        if (threadIdx.x == 0) {
#pragma unroll
          for (int row = 0; row < ROW_BATCH; ++row) {
            const int64_t compact_row = base + row;
            if (compact_row < end) {
              output[compact_row * out_features + out] =
                  float_to_bf16_bits(partial[row * LOGICAL_THREADS]);
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

// PF-3 M1 candidate: fused single-loop variant of the logical256 kernel.
// Thread t alternates column t (accumulator value0) and column t+128
// (accumulator value1) within one loop, so each accumulator sees exactly the
// same column sequence in the same order as the incumbent's two sequential
// loops (t, t+256, ... and t+128, t+384, ...) and the 256-wide strict
// reduction tree is unchanged; only the loop-schedule changes.
template <int ROW_BATCH, int OUT_BATCH>
__global__ void q5_1_selected_grouped_prefill_logical256_m1_bf16_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ expert_start,
    const uint8_t* __restrict__ weights,
    uint16_t* __restrict__ output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t row_bytes) {
  constexpr int LOGICAL_THREADS = 256;
  extern __shared__ float partial[];
  const int64_t blocks_per_row = in_features / 32;
  float* shared_d = partial + ROW_BATCH * LOGICAL_THREADS;
  float* shared_m = shared_d + blocks_per_row;
  uint32_t* shared_qh = reinterpret_cast<uint32_t*>(shared_m + blocks_per_row);
  const int64_t out_base = blockIdx.x * OUT_BATCH;
  if (out_base >= out_features) return;
  for (int64_t expert = blockIdx.y; expert < num_experts; expert += gridDim.y) {
    const int64_t start = expert_start[expert];
    const int64_t end = expert_start[expert + 1];
    if (start < 0 || end < start || end > compact_rows || start == end) continue;
#pragma unroll
    for (int out_offset = 0; out_offset < OUT_BATCH; ++out_offset) {
      const int64_t out = out_base + out_offset;
      if (out >= out_features) break;
      const uint8_t* weight_row = weights
          + (expert * out_features + out) * row_bytes;
      if (threadIdx.x < blocks_per_row) {
        const uint8_t* block = weight_row + threadIdx.x * 24;
        shared_d[threadIdx.x] = load_fp16(block);
        shared_m[threadIdx.x] = load_fp16(block + 2);
        shared_qh[threadIdx.x] =
            *reinterpret_cast<const uint32_t*>(block + 4);
      }
      __syncthreads();
      for (int64_t base = start; base < end; base += ROW_BATCH) {
        float value0[ROW_BATCH] = {};
        float value1[ROW_BATCH] = {};
        const int logical0 = threadIdx.x;
        const int logical1 = threadIdx.x + blockDim.x;
        for (int64_t column = logical0; column < in_features;
             column += LOGICAL_THREADS) {
          const int64_t column1 = column + blockDim.x;
          const int64_t block_index0 = column / 32;
          const int within0 = static_cast<int>(column & 31);
          const uint8_t* block0 = weight_row + block_index0 * 24;
          const uint8_t packed0 = block0[8 + (within0 & 15)];
          const uint8_t low0 =
              within0 < 16 ? (packed0 & 0x0fU) : (packed0 >> 4);
          const uint8_t high0 = static_cast<uint8_t>(
              (shared_qh[block_index0] >> within0) & 1U);
          const float weight0 = shared_d[block_index0]
              * static_cast<float>(low0 | (high0 << 4))
              + shared_m[block_index0];
          float weight1 = 0.0f;
          if (column1 < in_features) {
            const int64_t block_index1 = column1 / 32;
            const int within1 = static_cast<int>(column1 & 31);
            const uint8_t* block1 = weight_row + block_index1 * 24;
            const uint8_t packed1 = block1[8 + (within1 & 15)];
            const uint8_t low1 =
                within1 < 16 ? (packed1 & 0x0fU) : (packed1 >> 4);
            const uint8_t high1 = static_cast<uint8_t>(
                (shared_qh[block_index1] >> within1) & 1U);
            weight1 = shared_d[block_index1]
                * static_cast<float>(low1 | (high1 << 4))
                + shared_m[block_index1];
          }
#pragma unroll
          for (int row = 0; row < ROW_BATCH; ++row) {
            const int64_t compact_row = base + row;
            if (compact_row < end) {
              value0[row] += bf16_bits_to_float(
                  input[compact_row * in_features + column]) * weight0;
              if (column1 < in_features) {
                value1[row] += bf16_bits_to_float(
                    input[compact_row * in_features + column1]) * weight1;
              }
            }
          }
        }
#pragma unroll
        for (int row = 0; row < ROW_BATCH; ++row) {
          partial[row * LOGICAL_THREADS + logical0] = value0[row];
          partial[row * LOGICAL_THREADS + logical1] = value1[row];
        }
        __syncthreads();
        for (int stride = LOGICAL_THREADS / 2; stride > 0; stride >>= 1) {
          if (threadIdx.x < stride) {
#pragma unroll
            for (int row = 0; row < ROW_BATCH; ++row) {
              partial[row * LOGICAL_THREADS + threadIdx.x] +=
                  partial[row * LOGICAL_THREADS + threadIdx.x + stride];
            }
          }
          __syncthreads();
        }
        if (threadIdx.x == 0) {
#pragma unroll
          for (int row = 0; row < ROW_BATCH; ++row) {
            const int64_t compact_row = base + row;
            if (compact_row < end) {
              output[compact_row * out_features + out] =
                  float_to_bf16_bits(partial[row * LOGICAL_THREADS]);
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

__device__ inline float q51_cached_weight(
    const uint8_t* row, int column, const float* d, const float* m,
    const uint32_t* qh) {
  const int block = column / 32;
  const int within = column & 31;
  const uint8_t packed = row[block * 24 + 8 + (within & 15)];
  const int low = within < 16 ? (packed & 15) : (packed >> 4);
  const int high = (qh[block] >> within) & 1;
  return d[block] * static_cast<float>(low | (high << 4)) + m[block];
}

template <bool FOLD128 = false, bool PAIR_REDUCE = false, bool REGISTER_CACHE = false, bool ROW_PUBLISH = false>
__global__ void q5_1_selected_grouped_prefill_pair2_bf16_kernel(
    const uint16_t* __restrict__ input, const int64_t* __restrict__ starts,
    const uint8_t* __restrict__ weights, uint16_t* __restrict__ output,
    int64_t rows, int64_t experts, int64_t k, int64_t n) {
  constexpr int R = 8, L = 256;
  constexpr int S = FOLD128 ? 128 : L;
  extern __shared__ float partial[];
  const int blocks = k / 32;
  const int row_bytes = blocks * 24;
  float* d = partial + R * S * (PAIR_REDUCE ? 2 : 1);
  float* m = d + blocks * 2;
  uint32_t* qh = reinterpret_cast<uint32_t*>(m + blocks * 2);
  const int out_base = blockIdx.x * 8;
  for (int64_t expert = blockIdx.y; expert < experts; expert += gridDim.y) {
    const int64_t begin = starts[expert], end = starts[expert + 1];
    if (begin < 0 || end < begin || end > rows || begin == end) continue;
    for (int offset = 0; offset < 8 && out_base + offset < n; offset += 2) {
      const uint8_t* wr[2];
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        wr[p] = weights + (expert * n + out_base + offset + p) * row_bytes;
        if (out_base + offset + p < n && threadIdx.x < blocks) {
          const uint8_t* block = wr[p] + threadIdx.x * 24;
          const int idx = p * blocks + threadIdx.x;
          d[idx] = load_fp16(block);
          m[idx] = load_fp16(block + 2);
          qh[idx] = *reinterpret_cast<const uint32_t*>(block + 4);
        }
      }
      __syncthreads();
      float cached[2][REGISTER_CACHE ? 5 : 1] = {};
      if constexpr (REGISTER_CACHE) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
#pragma unroll
          for (int i = 0; i < 5; ++i) {
            if (out_base+offset+p < n)
              cached[p][i] = q51_cached_weight(
                  wr[p], threadIdx.x+i*128, d+p*blocks, m+p*blocks, qh+p*blocks);
          }
        }
      }
      for (int64_t base = begin; base < end; base += R) {
        float acc0[2][R] = {}, acc1[2][R] = {};
        if constexpr (ROW_PUBLISH) {
          static_assert(REGISTER_CACHE && PAIR_REDUCE && FOLD128);
          // Finish both logical256 streams per row before publishing to LDS.
#pragma unroll
          for (int r = 0; r < R; ++r) {
            float a[2] = {}, b[2] = {};
            if (base+r < end) {
#pragma unroll
              for (int pass = 0; pass < 3; ++pass) {
                const int column = threadIdx.x+pass*256;
                const float x0 = bf16_bits_to_float(input[(base+r)*k+column]);
                const float x1 = pass < 2
                    ? bf16_bits_to_float(input[(base+r)*k+column+128]) : 0.0f;
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                  a[p] += x0*cached[p][pass*2];
                  if (pass < 2) b[p] += x1*cached[p][pass*2+1];
                }
              }
            }
#pragma unroll
            for (int p = 0; p < 2; ++p)
              partial[(p*R+r)*S+threadIdx.x] = a[p]+b[p];
          }
        } else if constexpr (REGISTER_CACHE) {
          // K640 has five physical-lane partials: preserve the two logical256 streams.
#pragma unroll
          for (int pass = 0; pass < 3; ++pass) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
              if (base+r < end) {
                const int column = threadIdx.x+pass*256;
                const float x0 = bf16_bits_to_float(input[(base+r)*k+column]);
                const float x1 = pass < 2
                    ? bf16_bits_to_float(input[(base+r)*k+column+128]) : 0.0f;
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                  acc0[p][r] += x0*cached[p][pass*2];
                  if (pass < 2) acc1[p][r] += x1*cached[p][pass*2+1];
                }
              }
            }
          }
        } else {
        for (int column = threadIdx.x; column < k; column += L) {
          const int column1 = column + 128;
          float w0[2] = {}, w1[2] = {};
#pragma unroll
          for (int p = 0; p < 2; ++p) {
            if (out_base + offset + p < n) {
              w0[p] = q51_cached_weight(wr[p], column, d+p*blocks, m+p*blocks, qh+p*blocks);
              if (column1 < k)
                w1[p] = q51_cached_weight(wr[p], column1, d+p*blocks, m+p*blocks, qh+p*blocks);
            }
          }
#pragma unroll
          for (int r = 0; r < R; ++r) {
            if (base + r < end) {
              const float x0 = bf16_bits_to_float(input[(base+r)*k+column]);
              const float x1 = column1 < k ? bf16_bits_to_float(input[(base+r)*k+column1]) : 0.0f;
#pragma unroll
              for (int p = 0; p < 2; ++p) {
                acc0[p][r] += x0 * w0[p];
                if (column1 < k) acc1[p][r] += x1 * w1[p];
              }
            }
          }
        }
        }
        if constexpr (PAIR_REDUCE) {
          static_assert(FOLD128);
          if constexpr (!ROW_PUBLISH) {
#pragma unroll
          for (int p = 0; p < 2; ++p) {
#pragma unroll
            for (int r = 0; r < R; ++r)
              partial[(p*R+r)*S+threadIdx.x] = acc0[p][r] + acc1[p][r];
          }
          }
          __syncthreads();
          for (int stride = 64; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
#pragma unroll
              for (int pr = 0; pr < 2*R; ++pr)
                partial[pr*S+threadIdx.x] += partial[pr*S+threadIdx.x+stride];
            }
            __syncthreads();
          }
          if (threadIdx.x < 2*R) {
            const int p = threadIdx.x / R, r = threadIdx.x % R;
            if (base+r < end && out_base+offset+p < n)
              output[(base+r)*n+out_base+offset+p] =
                  float_to_bf16_bits(partial[(p*R+r)*S]);
          }
          __syncthreads();
        } else {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          if (out_base + offset + p >= n) continue;
#pragma unroll
          for (int r = 0; r < R; ++r) {
            if constexpr (FOLD128) {
              partial[r*S+threadIdx.x] = acc0[p][r] + acc1[p][r];
            } else {
              partial[r*S+threadIdx.x] = acc0[p][r];
              partial[r*S+threadIdx.x+128] = acc1[p][r];
            }
          }
          __syncthreads();
          for (int stride = S / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
#pragma unroll
              for (int r = 0; r < R; ++r)
                partial[r*S+threadIdx.x] += partial[r*S+threadIdx.x+stride];
            }
            __syncthreads();
          }
          if (threadIdx.x == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r)
              if (base+r < end) output[(base+r)*n+out_base+offset+p] = float_to_bf16_bits(partial[r*S]);
          }
          __syncthreads();
        }
        }
      }
    }
  }
}

template <int TILE_M, int TILE_N>
__launch_bounds__(32, 8)
__global__ void q5_1_selected_grouped_wmma_prefill_bf16_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ weights,
    uint16_t* __restrict__ output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t row_bytes) {
  constexpr int TM = TILE_M / 16;
  constexpr int TN = TILE_N / 16;
  const int tid = threadIdx.x;
  const int out_start = blockIdx.x * TILE_M;
  const int64_t row_tile = blockIdx.y;
  if (out_start >= out_features) return;
  const int64_t expert = tile_expert[row_tile];
  if (expert < 0 || expert >= num_experts) return;
  const int64_t compact_start = expert_start_compact[expert];
  const int64_t compact_end = expert_start_compact[expert + 1];
  const int64_t expert_rows = compact_end - compact_start;
  const int64_t local_start = row_tile * 16 - expert_start_wmma[expert];
  if (expert_rows <= 0 || compact_start < 0 || compact_end > compact_rows) return;

  int my_out[TM];
  int safe_out[TM];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    my_out[tm] = out_start + tm * 16 + (tid & 15);
    safe_out[tm] = my_out[tm] < out_features ? my_out[tm] : out_features - 1;
  }
  int64_t local_row[TN];
  int64_t safe_row[TN];
#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    local_row[tn] = local_start + tn * 16 + (tid & 15);
    const int64_t bounded = local_row[tn] < expert_rows
        ? local_row[tn] : expert_rows - 1;
    safe_row[tn] = compact_start + bounded;
  }
  float8_t acc[TM][TN];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      acc[tm][tn] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }
  const int blocks_per_row = in_features / 32;
  for (int blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* block_ptr[TM];
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      block_ptr[tm] = weights
          + (expert * out_features + safe_out[tm]) * row_bytes + blk * 24;
    }
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
      const int k_base = blk * 32 + kt * 16;
      half16_t activation[TN];
#pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          activation[tn][kk] = static_cast<half_t>(bf16_bits_to_float(
              input[safe_row[tn] * in_features + k_base + kk]));
        }
      }
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        half16_t weight;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          weight[kk] = static_cast<half_t>(
              dequant_q5_1(block_ptr[tm], kt * 16 + kk));
        }
#pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              weight, activation[tn], acc[tm][tn]);
        }
      }
    }
  }
#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    if (local_row[tn] >= expert_rows) continue;
    const int64_t out_row = compact_start + local_row[tn];
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = out_start + tm * 16 + 2 * j + (tid >> 4);
        if (col < out_features) {
          output[out_row * out_features + col] =
              float_to_bf16_bits(static_cast<float>(acc[tm][tn][j]));
        }
      }
    }
  }
}

__global__ void qwen4_exp_gather_bf16_lanes_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ sorted_lanes,
    uint16_t* __restrict__ output,
    int64_t rows,
    int64_t features) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= rows * features) return;
  const int64_t row = idx / features;
  const int64_t col = idx - row * features;
  const int64_t source = sorted_lanes[row];
  if (source >= 0 && source < rows) {
    output[idx] = input[source * features + col];
  }
}

__launch_bounds__(64, 4)
__global__ void q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ weights,
    const float* __restrict__ routing_weights,
    uint16_t* __restrict__ output,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t row_bytes) {
  constexpr int LOGICAL_THREADS = 256;
  extern __shared__ float partial[];
  const int64_t out = blockIdx.x;
  if (out >= out_features) return;
  float combined = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t expert = selected[row];
    if (expert < 0 || expert >= num_experts) return;
    const uint8_t* weight_row = weights
        + (expert * out_features + out) * row_bytes;
    float value[4] = {};
#pragma unroll
    for (int logical_group = 0; logical_group < 4; ++logical_group) {
      const int logical = threadIdx.x + logical_group * 64;
      for (int64_t column = logical; column < in_features;
           column += LOGICAL_THREADS) {
        const int64_t block_index = column / 32;
        const int within = static_cast<int>(column & 31);
        value[logical_group] +=
            bf16_bits_to_float(input[row * in_features + column])
            * dequant_q5_1(weight_row + block_index * 24, within);
      }
      partial[logical] = value[logical_group];
    }
    __syncthreads();
    partial[threadIdx.x] += partial[threadIdx.x + 128];
    partial[threadIdx.x + 64] += partial[threadIdx.x + 192];
    __syncthreads();
    partial[threadIdx.x] += partial[threadIdx.x + 64];
    __syncthreads();
    if (threadIdx.x < 32) {
      partial[threadIdx.x] += partial[threadIdx.x + 32];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      float reduced = partial[threadIdx.x];
      for (int offset = 16; offset > 0; offset >>= 1) {
        reduced += __shfl_down(reduced, offset, 32);
      }
      if (threadIdx.x == 0) {
        const float rounded = bf16_bits_to_float(float_to_bf16_bits(reduced));
        combined = fmaf(rounded, routing_weights[row], combined);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[out] = float_to_bf16_bits(combined);
}

__launch_bounds__(64, 4)
__global__ void q5_1_selected_gemv_logical256_t64_bf16_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ weights,
    uint16_t* __restrict__ output,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t row_bytes) {
  constexpr int LOGICAL_THREADS = 256;
  extern __shared__ float partial[];
  const int64_t pair = blockIdx.x;
  const int64_t row = pair / out_features;
  const int64_t out = pair - row * out_features;
  const int64_t input_row = row / (rows / x_rows);
  const int64_t expert = selected[row];
  if (expert < 0 || expert >= num_experts) return;
  const uint8_t* weight_row = weights
      + (expert * out_features + out) * row_bytes;
  float value[4] = {};
#pragma unroll
  for (int logical_group = 0; logical_group < 4; ++logical_group) {
    const int logical = threadIdx.x + logical_group * 64;
    for (int64_t column = logical; column < in_features;
         column += LOGICAL_THREADS) {
      const int64_t block_index = column / 32;
      const int within = static_cast<int>(column & 31);
      value[logical_group] +=
          bf16_bits_to_float(input[input_row * in_features + column])
          * dequant_q5_1(weight_row + block_index * 24, within);
    }
    partial[logical] = value[logical_group];
  }
  __syncthreads();
  partial[threadIdx.x] += partial[threadIdx.x + 128];
  partial[threadIdx.x + 64] += partial[threadIdx.x + 192];
  __syncthreads();
  partial[threadIdx.x] += partial[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x < 32) {
    partial[threadIdx.x] += partial[threadIdx.x + 32];
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    float reduced = partial[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1) {
      reduced += __shfl_down(reduced, offset, 32);
    }
    if (threadIdx.x == 0) output[pair] = float_to_bf16_bits(reduced);
  }
}

__launch_bounds__(128, 2)
__global__ void q5_1_selected_gemv_logical256_t128_bf16_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ weights,
    uint16_t* __restrict__ output,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t row_bytes) {
  constexpr int LOGICAL_THREADS = 256;
  extern __shared__ float partial[];
  const int64_t pair = blockIdx.x;
  const int64_t row = pair / out_features;
  const int64_t out = pair - row * out_features;
  const int64_t input_row = row / (rows / x_rows);
  const int64_t expert = selected[row];
  if (expert < 0 || expert >= num_experts) return;
  const uint8_t* weight_row = weights
      + (expert * out_features + out) * row_bytes;
  float value0 = 0.0f;
  float value1 = 0.0f;
  const int logical0 = threadIdx.x;
  const int logical1 = threadIdx.x + blockDim.x;
  for (int64_t column = logical0; column < in_features;
       column += LOGICAL_THREADS) {
    const int64_t block_index = column / 32;
    const int within = static_cast<int>(column & 31);
    value0 += bf16_bits_to_float(input[input_row * in_features + column])
        * dequant_q5_1(weight_row + block_index * 24, within);
  }
  for (int64_t column = logical1; column < in_features;
       column += LOGICAL_THREADS) {
    const int64_t block_index = column / 32;
    const int within = static_cast<int>(column & 31);
    value1 += bf16_bits_to_float(input[input_row * in_features + column])
        * dequant_q5_1(weight_row + block_index * 24, within);
  }
  partial[logical0] = value0;
  partial[logical1] = value1;
  __syncthreads();
  for (int stride = LOGICAL_THREADS / 2; stride >= 32; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    float reduced = partial[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1) {
      reduced += __shfl_down(reduced, offset, 32);
    }
    if (threadIdx.x == 0) output[pair] = float_to_bf16_bits(reduced);
  }
}

__global__ void q5_1_selected_gemv_bf16_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ weights,
    uint16_t* __restrict__ output,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t row_bytes) {
  extern __shared__ float partial[];
  const int64_t pair = blockIdx.x;
  const int64_t row = pair / out_features;
  const int64_t out = pair % out_features;
  const int64_t input_row = row / (rows / x_rows);
  const int64_t expert = selected[row];
  if (expert < 0 || expert >= num_experts) {
    return;
  }
  const uint8_t* weight_row = weights
      + (static_cast<int64_t>(expert) * out_features + out) * row_bytes;
  float value = 0.0f;
  for (int64_t column = threadIdx.x; column < in_features; column += blockDim.x) {
    const int64_t block_index = column / 32;
    const int within = static_cast<int>(column & 31);
    const float weight = dequant_q5_1(weight_row + block_index * 24, within);
    value += bf16_bits_to_float(input[input_row * in_features + column]) * weight;
  }
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    float reduced = partial[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1) {
      reduced += __shfl_down(reduced, offset, 32);
    }
    if (threadIdx.x == 0) {
      output[pair] = float_to_bf16_bits(reduced);
    }
  }
}

// ---------------------------------------------------------------------------
// Weight-exact iu8-WMMA selected Q5_1 down prefill with risk collection.
//
// Mirrors the promoted Q4_K iu8-risk structure: three residual activation
// planes staged per 32-element block in shared memory (fp32 scales, no fp16
// rounding), the raw 5-bit Q5_1 codes as the integer B operand, the m-offset
// term reconstructed per 32-block from the staged plane-0-extended q8 sums
// (ds-trick), and a Kahan-compensated accumulation-error probe with the
// row-level at-risk guards (nonfinite values; subblock amax below 2^-80).
// The m-offset sign is PLUS for Q5_1 (w = d*q5 + m) versus the Q4_K minus.
// ---------------------------------------------------------------------------

constexpr int Q51_IU8_ACT_PLANES = 3;
constexpr int Q51_IU8_LDS_ROW_STRIDE = 256 + 16;  // bank-conflict padding
constexpr float Q51_IU8_ATRISK_AMAX_FLOOR = 0x1p-80f;
typedef int __attribute__((vector_size(8 * sizeof(int)))) q51_iu8_int32x8;
typedef int __attribute__((vector_size(4 * sizeof(int)))) q51_iu8_int32x4;

__device__ inline float q51_bf16_rounding_boundary_distance(float value) {
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

__global__ __launch_bounds__(128, 2)
void q5_1_selected_wmma_iu8_risk_prefill_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int* __restrict__ risk_count,
    int* __restrict__ risk_indices,
    int64_t max_risks,
    float risk_multiplier,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  __shared__ int lds_row_atrisk[16];
  __shared__ int8_t lds_q8a[Q51_IU8_ACT_PLANES * 16 * Q51_IU8_LDS_ROW_STRIDE];
  __shared__ float lds_da[Q51_IU8_ACT_PLANES * 16 * 8];
  __shared__ int lds_sumq8a[Q51_IU8_ACT_PLANES * 16 * 8];
  __shared__ int64_t meta[4];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
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
  const bool col_valid = global_col < out_features;
  const int64_t blocks_per_row = in_features / 32;
  const int64_t row_bytes = blocks_per_row * 24;
  const bool wcol_valid = col_valid;
  const uint8_t* block_base = wcol_valid
      ? (qweight + expert_id * (out_features * row_bytes) +
         global_col * row_bytes)
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

  const int64_t chunks = (in_features + 255) / 256;
  for (int64_t chunk = 0; chunk < chunks; ++chunk) {
    // Cooperative activation staging: thread -> (actrow = tid/8, sub = tid%8),
    // one 32-element Q5_1 block per (actrow, sub).
    {
      const int actrow = tid >> 3;
      const int sub = tid & 7;
      const int64_t col0 = chunk * 256 + static_cast<int64_t>(sub) * 32;
      const int64_t local_row = tile_row_in_expert + actrow;
      const bool row_valid = local_row < expert_count;
      const bool sub_valid = col0 + 32 <= in_features;
      const int64_t compact_row = row_valid ? (compact_base + local_row) : 0;
      const uint16_t* row_ptr = sub_valid
          ? (x + compact_row * in_features + col0)
          : nullptr;
      float vals[32];
      float amax = 0.0f;
#pragma unroll
      for (int l = 0; l < 32; ++l) {
        const float v = (row_valid && sub_valid)
            ? bf16_bits_to_float(row_ptr[l]) : 0.0f;
        vals[l] = v;
        amax = fmaxf(amax, fabsf(v));
        if (!isfinite(v)) {
          lds_row_atrisk[actrow] = 1;
        }
      }
      if (row_valid && sub_valid && amax > 0.0f &&
          amax < Q51_IU8_ATRISK_AMAX_FLOOR) {
        lds_row_atrisk[actrow] = 1;
      }
      float work[32];
#pragma unroll
      for (int l = 0; l < 32; ++l) work[l] = vals[l];
#pragma unroll
      for (int plane = 0; plane < Q51_IU8_ACT_PLANES; ++plane) {
        float plane_max = 0.0f;
#pragma unroll
        for (int l = 0; l < 32; ++l)
          plane_max = fmaxf(plane_max, fabsf(work[l]));
        const float da = plane_max > 0.0f ? plane_max / 127.0f : 0.0f;
        lds_da[plane * 16 * 8 + actrow * 8 + sub] = da;
        int8_t* dst = lds_q8a + plane * 16 * Q51_IU8_LDS_ROW_STRIDE +
                      actrow * Q51_IU8_LDS_ROW_STRIDE + sub * 32;
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
        lds_sumq8a[plane * 16 * 8 + actrow * 8 + sub] = qsum;
      }
    }
    __syncthreads();

#pragma unroll 1
    for (int sub = 0; sub < 8; ++sub) {
      const int64_t col0 = chunk * 256 + static_cast<int64_t>(sub) * 32;
      const bool sub_valid = col0 + 32 <= in_features;
      const uint8_t* block_ptr = sub_valid && wcol_valid
          ? (block_base + (col0 / 32) * 24)
          : nullptr;
      // Raw 5-bit Q5_1 codes (0..31) as the integer B operand; the weight
      // path has no requantization error at all.
      int8_t wq8[32];
      if (block_ptr != nullptr) {
        const uint32_t qh = *reinterpret_cast<const uint32_t*>(block_ptr + 4);
        const uint8_t* qs = block_ptr + 8;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
          const uint8_t packed = qs[l & 15];
          const uint8_t low = l < 16 ? (packed & 0x0fU) : (packed >> 4);
          const uint8_t high = static_cast<uint8_t>((qh >> l) & 1U);
          wq8[l] = static_cast<int8_t>(low | (high << 4));
        }
      } else {
#pragma unroll
        for (int l = 0; l < 32; ++l) wq8[l] = 0;
      }
      const float dsub = sub_valid && block_ptr != nullptr
          ? load_fp16(block_ptr) : 0.0f;
      const float msub = sub_valid && block_ptr != nullptr
          ? load_fp16(block_ptr + 2) : 0.0f;

#pragma unroll
      for (int g = 0; g < 2; ++g) {
        q51_iu8_int32x8 acc[Q51_IU8_ACT_PLANES];
#pragma unroll
        for (int plane = 0; plane < Q51_IU8_ACT_PLANES; ++plane) {
          acc[plane] = {0, 0, 0, 0, 0, 0, 0, 0};
        }
#pragma unroll
        for (int khalf = 0; khalf < 2; ++khalf) {
          signed char bbytes[16];
#pragma unroll
          for (int j = 0; j < 16; ++j) bbytes[j] = wq8[khalf * 16 + j];
          q51_iu8_int32x4 b_frag;
          {
            int32_t words[4];
            memcpy(words, bbytes, 16);
            b_frag = (q51_iu8_int32x4){words[0], words[1], words[2], words[3]};
          }
#pragma unroll
          for (int plane = 0; plane < Q51_IU8_ACT_PLANES; ++plane) {
            signed char abytes[16];
            if (a_row >= 0) {
              const int8_t* src = lds_q8a +
                  plane * 16 * Q51_IU8_LDS_ROW_STRIDE +
                  (g * 8 + a_row) * Q51_IU8_LDS_ROW_STRIDE +
                  sub * 32 + khalf * 16;
#pragma unroll
              for (int j = 0; j < 16; ++j) abytes[j] = src[j];
            } else {
#pragma unroll
              for (int j = 0; j < 16; ++j) abytes[j] = 0;
            }
            q51_iu8_int32x4 a_frag;
            {
              int32_t words[4];
              memcpy(words, abytes, 16);
              a_frag = (q51_iu8_int32x4){words[0], words[1], words[2], words[3]};
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
          for (int plane = 0; plane < Q51_IU8_ACT_PLANES; ++plane) {
            const float da = lds_da[plane * 16 * 8 + r * 8 + sub];
            const int asumb = lds_sumq8a[plane * 16 * 8 + r * 8 + sub];
            const int* cr = reinterpret_cast<const int*>(&acc[plane]);
            main_sum += da * static_cast<float>(cr[i]);
            act_sum += da * static_cast<float>(asumb);
          }
          const float term = dsub * main_sum + msub * act_sum;
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
        out[compact_row * out_features + global_col] =
            float_to_bf16_bits(value);
        const float bound = risk_multiplier *
            (fabsf(kahan_err[r]) + FLT_EPSILON * abs_terms[r]);
        const bool at_risk = lds_row_atrisk[r] != 0 ||
            !isfinite(value) ||
            q51_bf16_rounding_boundary_distance(value) < bound;
        if (at_risk) {
          const int slot = atomicAdd(risk_count, 1);
          if (slot < max_risks) {
            risk_indices[slot] = static_cast<int>(
                compact_row * out_features + global_col);
          }
        }
      }
    }
  }
}

// Sparse exact repair for the iu8-WMMA Q5_1 down: reproduces the production
// pair2 row-publish arithmetic (q5_1_selected_grouped_prefill_pair2_row_
// publish: 128 logical lanes; per lane the even-stream accumulator covers
// columns lane + i*128 for even i, the odd stream covers odd i, and the
// published value is even + odd after the 128-entry halving tree).
__launch_bounds__(128, 2)
__global__ void q5_1_selected_sparse_exact_repair_row_publish_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ expert_start,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    const int* __restrict__ risk_count,
    const int* __restrict__ risk_indices,
    int64_t max_risks,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  __shared__ float partial[128];
  const int count = static_cast<int>(
      max(static_cast<int64_t>(*risk_count), static_cast<int64_t>(0)));
  const int bounded = static_cast<int>(
      min(static_cast<int64_t>(count), max_risks));
  const int64_t blocks_per_row = in_features / 32;
  const int64_t row_bytes = blocks_per_row * 24;
  const int lanes = static_cast<int>(in_features / 128);
  for (int slot = static_cast<int>(blockIdx.x); slot < bounded;
       slot += static_cast<int>(gridDim.x)) {
    const int64_t idx = risk_indices[slot];
    if (idx < 0 || idx >= compact_rows * out_features) continue;
    const int64_t compact_row = idx / out_features;
    const int64_t out_col = idx - compact_row * out_features;
    // Locate the owning expert with a binary search over group starts.
    int64_t lo = 0;
    int64_t hi = num_experts;
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
    const uint8_t* weight_row =
        qweight + (expert * out_features + out_col) * row_bytes;
    const uint16_t* x_row = input + compact_row * in_features;
    const int thread = static_cast<int>(threadIdx.x);
    float acc_even = 0.0f;
    float acc_odd = 0.0f;
    for (int i = 0; i < lanes; ++i) {
      const int64_t column = thread + static_cast<int64_t>(i) * 128;
      const int64_t block = column / 32;
      const int within = static_cast<int>(column & 31);
      const uint8_t* blk = weight_row + block * 24;
      const uint8_t packed = blk[8 + (within & 15)];
      const uint8_t low = within < 16 ? (packed & 0x0fU) : (packed >> 4);
      const uint8_t high = static_cast<uint8_t>(
          (*reinterpret_cast<const uint32_t*>(blk + 4) >> within) & 1U);
      const float weight = load_fp16(blk) *
          static_cast<float>(low | (high << 4)) + load_fp16(blk + 2);
      const float xv = bf16_bits_to_float(x_row[column]);
      if ((i & 1) == 0) {
        acc_even += xv * weight;
      } else {
        acc_odd += xv * weight;
      }
    }
    partial[threadIdx.x] = acc_even + acc_odd;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      out[compact_row * out_features + out_col] =
          float_to_bf16_bits(partial[0]);
    }
    __syncthreads();
  }
}

static int launch_q5_1_wmma_iu8_risk(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int* risk_count,
    int* risk_indices,
    int64_t max_risks,
    float risk_multiplier,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (x == nullptr || expert_start_compact == nullptr ||
      expert_start_wmma == nullptr || tile_expert == nullptr ||
      qweight == nullptr || out == nullptr || risk_count == nullptr ||
      risk_indices == nullptr || compact_rows <= 0 || num_experts <= 0 ||
      in_features <= 0 || (in_features % 32) != 0 || out_features <= 0 ||
      wmma_total_rows < compact_rows || wmma_total_rows % 16 != 0 ||
      max_risks < 0 || max_risks > INT32_MAX || risk_multiplier < 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>((out_features + 127) / 128),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      (q5_1_selected_wmma_iu8_risk_prefill_kernel),
      grid, block, 0, stream,
      x, expert_start_compact, expert_start_wmma, tile_expert, qweight,
      out, risk_count, risk_indices, max_risks, risk_multiplier,
      compact_rows, in_features, out_features, num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

static int launch_q5_1_sparse_exact_repair(
    const uint16_t* input,
    const int64_t* expert_start,
    const uint8_t* qweight,
    uint16_t* out,
    const int* risk_count,
    const int* risk_indices,
    int64_t max_risks,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t grid_blocks,
    hipStream_t stream) {
  if (input == nullptr || expert_start == nullptr || qweight == nullptr ||
      out == nullptr || risk_count == nullptr || risk_indices == nullptr ||
      max_risks < 0 || compact_rows <= 0 || num_experts <= 0 ||
      in_features <= 0 || (in_features % 128) != 0 || out_features <= 0 ||
      grid_blocks <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (q5_1_selected_sparse_exact_repair_row_publish_kernel),
      dim3(static_cast<unsigned int>(grid_blocks)), dim3(128), 0, stream,
      input, expert_start, qweight, out, risk_count, risk_indices, max_risks,
      compact_rows, in_features, out_features, num_experts);
  return static_cast<int>(hipGetLastError());
}
}  // namespace

extern "C" int hipengine_qwen4_exp_gather_bf16_lanes(
    const void* input,
    const void* sorted_lanes,
    void* output,
    int64_t rows,
    int64_t features,
    hipStream_t stream) {
  if (!input || !sorted_lanes || !output || rows <= 0 || features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t elements = rows * features;
  hipLaunchKernelGGL(
      qwen4_exp_gather_bf16_lanes_kernel,
      dim3(static_cast<uint32_t>((elements + threads - 1) / threads)),
      dim3(threads), 0, stream, static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(sorted_lanes), static_cast<uint16_t*>(output),
      rows, features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out(
    const void* input,
    const void* expert_start,
    const void* weights,
    void* output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !expert_start || !weights || !output || compact_rows <= 0 ||
      num_experts <= 0 || in_features <= 0 || (in_features % 32) ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  constexpr int row_batch = 8;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      (q5_1_selected_grouped_prefill_bf16_bf16_kernel<row_batch, 1>),
      dim3(static_cast<uint32_t>(out_features), static_cast<uint32_t>(num_experts)),
      dim3(threads), row_batch * threads * sizeof(float), stream,
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      compact_rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out(
    const void* input,
    const void* expert_start,
    const void* weights,
    void* output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !expert_start || !weights || !output || compact_rows <= 0 ||
      num_experts <= 0 || in_features <= 0 || (in_features % 32) ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  constexpr int row_batch = 8;
  constexpr int out_batch = 8;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      (q5_1_selected_grouped_prefill_bf16_bf16_kernel<row_batch, out_batch>),
      dim3(static_cast<uint32_t>((out_features + out_batch - 1) / out_batch),
           static_cast<uint32_t>(num_experts)),
      dim3(threads), row_batch * threads * sizeof(float), stream,
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      compact_rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out(
    const void* input,
    const void* expert_start,
    const void* weights,
    void* output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !expert_start || !weights || !output || compact_rows <= 0 ||
      num_experts <= 0 || in_features <= 0 || (in_features % 32) ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  constexpr int row_batch = 8;
  constexpr int out_batch = 8;
  constexpr int expert_workers = 64;
  const int64_t row_bytes = (in_features / 32) * 24;
  const int64_t worker_count = num_experts < expert_workers ? num_experts : expert_workers;
  hipLaunchKernelGGL(
      (q5_1_selected_grouped_prefill_logical256_bf16_bf16_kernel<row_batch, out_batch>),
      dim3(static_cast<uint32_t>((out_features + out_batch - 1) / out_batch),
           static_cast<uint32_t>(worker_count)),
      dim3(128),
      row_batch * threads * sizeof(float) + (in_features / 32) * 12,
      stream,
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      compact_rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out(
    const void* input,
    const void* expert_start,
    const void* weights,
    void* output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  // PF-3 M1 candidate: fused single-loop logical256 variant under the same
  // 64-CTA expert grid, 128 threads, and 256-wide strict reduction tree as
  // the production owner. Bit-exact by construction; only the per-thread
  // loop schedule changes.
  if (!input || !expert_start || !weights || !output || compact_rows <= 0 ||
      num_experts <= 0 || in_features <= 0 || (in_features % 32) ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  constexpr int row_batch = 8;
  constexpr int out_batch = 8;
  constexpr int expert_workers = 64;
  const int64_t row_bytes = (in_features / 32) * 24;
  const int64_t worker_count = num_experts < expert_workers ? num_experts : expert_workers;
  hipLaunchKernelGGL(
      (q5_1_selected_grouped_prefill_logical256_m1_bf16_bf16_kernel<row_batch, out_batch>),
      dim3(static_cast<uint32_t>((out_features + out_batch - 1) / out_batch),
           static_cast<uint32_t>(worker_count)),
      dim3(128),
      row_batch * threads * sizeof(float) + (in_features / 32) * 12,
      stream,
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      compact_rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

template <bool FOLD128, bool PAIR_REDUCE = false, bool REGISTER_CACHE = false, bool ROW_PUBLISH = false>
int launch_q51_pair2(
    const void* input, const void* starts, const void* weights, void* output,
    int64_t rows, int64_t experts, int64_t k, int64_t n, hipStream_t stream) {
  if (!input || !starts || !weights || !output || rows <= 0 || experts <= 0 ||
      k <= 0 || k % 32 || k > 4096 || n <= 0 || (REGISTER_CACHE && k != 640))
    return static_cast<int>(hipErrorInvalidValue);
  hipLaunchKernelGGL((q5_1_selected_grouped_prefill_pair2_bf16_kernel<FOLD128, PAIR_REDUCE, REGISTER_CACHE, ROW_PUBLISH>),
      dim3(static_cast<uint32_t>((n+7)/8), static_cast<uint32_t>(experts < 64 ? experts : 64)),
      dim3(128), (PAIR_REDUCE ? 2 : 1)*8*(FOLD128 ? 128 : 256)*sizeof(float)+(k/32)*24, stream,
      static_cast<const uint16_t*>(input), static_cast<const int64_t*>(starts),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      rows, experts, k, n);
  return static_cast<int>(hipGetLastError());
}

#define Q51_PAIR_ENTRY(NAME, FOLD, PAIR) \
extern "C" int NAME(const void* x, const void* starts, const void* w, void* y, \
    int64_t rows, int64_t experts, int64_t k, int64_t n, hipStream_t s) { \
  return launch_q51_pair2<FOLD, PAIR>(x, starts, w, y, rows, experts, k, n, s); \
}
Q51_PAIR_ENTRY(hipengine_qwen4_exp_q5_1_selected_grouped_prefill_pair2_bf16_bf16_out, false, false)
Q51_PAIR_ENTRY(hipengine_qwen4_exp_q5_1_selected_grouped_prefill_pair2_fold128_bf16_bf16_out, true, false)
Q51_PAIR_ENTRY(hipengine_qwen4_exp_q5_1_selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out, true, true)
#undef Q51_PAIR_ENTRY

extern "C" int hipengine_qwen4_exp_q5_1_selected_grouped_prefill_pair2_register_cache_bf16_bf16_out(
    const void* x, const void* starts, const void* w, void* y,
    int64_t rows, int64_t experts, int64_t k, int64_t n, hipStream_t s) {
  return launch_q51_pair2<true, true, true>(x, starts, w, y, rows, experts, k, n, s);
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_grouped_prefill_pair2_row_publish_bf16_bf16_out(
    const void* x, const void* starts, const void* w, void* y,
    int64_t rows, int64_t experts, int64_t k, int64_t n, hipStream_t s) {
  return launch_q51_pair2<true, true, true, true>(x, starts, w, y, rows, experts, k, n, s);
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out(
    const void* input,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* weights,
    void* output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!input || !expert_start_compact || !expert_start_wmma || !tile_expert ||
      !weights || !output || compact_rows <= 0 || num_experts <= 0 ||
      in_features <= 0 || (in_features % 32) || out_features <= 0 ||
      wmma_total_rows <= 0 || (wmma_total_rows % 16)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int tile_m = 32;
  constexpr int tile_n = 16;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      (q5_1_selected_grouped_wmma_prefill_bf16_bf16_kernel<tile_m, tile_n>),
      dim3(static_cast<uint32_t>((out_features + tile_m - 1) / tile_m),
           static_cast<uint32_t>(wmma_total_rows / tile_n)),
      dim3(32), 0, stream,
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start_compact),
      static_cast<const int64_t*>(expert_start_wmma),
      static_cast<const int64_t*>(tile_expert),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      compact_rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_out(
    const void* input,
    const void* selected,
    const void* weights,
    const void* routing_weights,
    void* output,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !selected || !weights || !routing_weights || !output ||
      rows <= 0 || num_experts <= 0 || in_features <= 0 ||
      (in_features % 32) || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int physical_threads = 64;
  constexpr int logical_threads = 256;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_kernel,
      dim3(static_cast<uint32_t>(out_features)), dim3(physical_threads),
      logical_threads * sizeof(float), stream,
      static_cast<const uint16_t*>(input), static_cast<const int64_t*>(selected),
      static_cast<const uint8_t*>(weights),
      static_cast<const float*>(routing_weights), static_cast<uint16_t*>(output),
      rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out(
    const void* input,
    const void* selected,
    const void* weights,
    void* output,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !selected || !weights || !output || x_rows <= 0 || rows <= 0 ||
      rows % x_rows || num_experts <= 0 || in_features <= 0 ||
      (in_features % 32) || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int physical_threads = 64;
  constexpr int logical_threads = 256;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      q5_1_selected_gemv_logical256_t64_bf16_bf16_kernel,
      dim3(static_cast<uint32_t>(rows * out_features)),
      dim3(physical_threads), logical_threads * sizeof(float), stream,
      static_cast<const uint16_t*>(input), static_cast<const int64_t*>(selected),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      x_rows, rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_gemv_logical256_t128_bf16_bf16_out(
    const void* input,
    const void* selected,
    const void* weights,
    void* output,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !selected || !weights || !output || x_rows <= 0 || rows <= 0 ||
      rows % x_rows || num_experts <= 0 || in_features <= 0 ||
      (in_features % 32) || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int physical_threads = 128;
  constexpr int logical_threads = 256;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      q5_1_selected_gemv_logical256_t128_bf16_bf16_kernel,
      dim3(static_cast<uint32_t>(rows * out_features)),
      dim3(physical_threads), logical_threads * sizeof(float), stream,
      static_cast<const uint16_t*>(input), static_cast<const int64_t*>(selected),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      x_rows, rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out(
    const void* input,
    const void* selected,
    const void* weights,
    void* output,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !selected || !weights || !output || x_rows <= 0 || rows <= 0 ||
      rows % x_rows || num_experts <= 0 || in_features <= 0 ||
      (in_features % 32) || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 64;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      q5_1_selected_gemv_bf16_bf16_kernel,
      dim3(static_cast<uint32_t>(rows * out_features)), dim3(threads),
      threads * sizeof(float), stream,
      static_cast<const uint16_t*>(input), static_cast<const int64_t*>(selected),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      x_rows, rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_gemv_bf16_bf16_out(
    const void* input,
    const void* selected,
    const void* weights,
    void* output,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !selected || !weights || !output || x_rows <= 0 || rows <= 0 ||
      rows % x_rows || num_experts <= 0 || in_features <= 0 ||
      (in_features % 32) || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      q5_1_selected_gemv_bf16_bf16_kernel,
      dim3(static_cast<uint32_t>(rows * out_features)), dim3(threads),
      threads * sizeof(float), stream,
      static_cast<const uint16_t*>(input), static_cast<const int64_t*>(selected),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      x_rows, rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out(
    const void* x,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* qweight,
    void* output,
    void* risk_count,
    void* risk_indices,
    int64_t max_risks,
    double risk_multiplier,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_q5_1_wmma_iu8_risk(
      static_cast<const uint16_t*>(x),
      static_cast<const int64_t*>(expert_start_compact),
      static_cast<const int64_t*>(expert_start_wmma),
      static_cast<const int64_t*>(tile_expert),
      static_cast<const uint8_t*>(qweight),
      static_cast<uint16_t*>(output),
      static_cast<int*>(risk_count),
      static_cast<int*>(risk_indices),
      max_risks,
      static_cast<float>(risk_multiplier),
      compact_rows,
      in_features,
      out_features,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_sparse_exact_repair_row_publish_bf16(
    const void* input,
    const void* expert_start,
    const void* qweight,
    void* output,
    const void* risk_count,
    const void* risk_indices,
    int64_t max_risks,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t grid_blocks,
    hipStream_t stream) {
  return launch_q5_1_sparse_exact_repair(
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start),
      static_cast<const uint8_t*>(qweight),
      static_cast<uint16_t*>(output),
      static_cast<const int*>(risk_count),
      static_cast<const int*>(risk_indices),
      max_risks,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      grid_blocks,
      stream);
}
