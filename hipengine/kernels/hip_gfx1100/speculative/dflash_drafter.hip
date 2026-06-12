// DFlash drafter root/query preparation kernels.
//
// The DFlash BF16 drafter reuses the target embedding table for the root token
// plus mask-token query rows.  This kernel materializes the compact native
// input slab on device: token ids, absolute query positions, and BF16 embedding
// rows.  It intentionally does not materialize context K/V; that remains the
// separate DFlash context-cache task.

#include <hip/hip_runtime.h>
#include <stdint.h>

using half_t = _Float16;

namespace {

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

template <typename embed_t>
__device__ inline uint16_t embed_to_bf16(embed_t value) {
  return float_to_bf16_bits(static_cast<float>(value));
}

template <>
__device__ inline uint16_t embed_to_bf16<uint16_t>(uint16_t value) {
  return value;
}

template <typename embed_t>
__global__ void dflash_prepare_noise_inputs_bf16_i32_kernel(
    const int32_t* __restrict__ root_tokens,
    const int32_t* __restrict__ root_positions,
    const embed_t* __restrict__ embed_tokens,
    int32_t* __restrict__ noise_token_ids,
    int32_t* __restrict__ position_ids,
    uint16_t* __restrict__ noise_embeddings,
    int64_t request_count,
    int64_t block_size,
    int64_t hidden_size,
    int64_t vocab_size,
    int32_t mask_token_id) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t col = static_cast<int64_t>(blockIdx.y);
  if (row >= request_count || col >= block_size) {
    return;
  }
  const int64_t slab_row = row * block_size + col;
  const int32_t token = (col == 0) ? root_tokens[row] : mask_token_id;
  const int32_t position = root_positions[row] + static_cast<int32_t>(col);
  if (threadIdx.x == 0) {
    noise_token_ids[slab_row] = token;
    position_ids[slab_row] = position;
  }
  const bool valid_token = token >= 0 && static_cast<int64_t>(token) < vocab_size;
  for (int64_t h = static_cast<int64_t>(threadIdx.x); h < hidden_size; h += blockDim.x) {
    uint16_t value = 0;
    if (valid_token) {
      value = embed_to_bf16(embed_tokens[static_cast<int64_t>(token) * hidden_size + h]);
    }
    noise_embeddings[slab_row * hidden_size + h] = value;
  }
}

bool valid_shape(int64_t request_count, int64_t block_size, int64_t hidden_size, int64_t vocab_size) {
  return request_count > 0 && block_size > 0 && hidden_size > 0 && vocab_size > 0;
}

bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256;
}

__global__ void dflash_add_bf16_kernel(
    const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ b,
    uint16_t* __restrict__ out,
    int64_t elements) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = float_to_bf16_bits(bf16_bits_to_float(a[i]) + bf16_bits_to_float(b[i]));
  }
}

template <typename scalar_t>
__global__ void dflash_concat_context_query_rows_kernel(
    const scalar_t* __restrict__ context_rows,
    const scalar_t* __restrict__ query_rows,
    scalar_t* __restrict__ out_rows,
    int64_t batch_size,
    int64_t context_len,
    int64_t query_len,
    int64_t features) {
  const int64_t total = batch_size * (context_len + query_len) * features;
  const int64_t rows_per_batch = context_len + query_len;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t feature = idx % features;
    const int64_t row_linear = idx / features;
    const int64_t row = row_linear % rows_per_batch;
    const int64_t batch = row_linear / rows_per_batch;
    if (row < context_len) {
      out_rows[idx] = context_rows[(batch * context_len + row) * features + feature];
    } else {
      out_rows[idx] = query_rows[(batch * query_len + (row - context_len)) * features + feature];
    }
  }
}

__global__ void dflash_rmsnorm_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ weight,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t hidden_size,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  extern __shared__ float partial[];
  const int64_t base = row * hidden_size;
  float sumsq = 0.0f;
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < hidden_size; dim += blockDim.x) {
    const float value = bf16_bits_to_float(x[base + dim]);
    sumsq += value * value;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < hidden_size; dim += blockDim.x) {
    const float value = bf16_bits_to_float(x[base + dim]);
    const float scale = bf16_bits_to_float(weight[dim]);
    out[base + dim] = float_to_bf16_bits(value * inv_rms * scale);
  }
}

// R3.6 C1: drafter post-attention add+rmsnorm fusion.
//
// Replaces the unfused chain ``dflash_add_bf16(query_in, attn_proj -> hidden_attn);
// dflash_rmsnorm_bf16(hidden_attn, post_norm_weight -> post)``.  Keeps the unfused
// fallback registered so a flag flip is exact-equivalent.
//
// One workgroup per row: 256 threads compute the residual sum (also stored to
// ``hidden_out``) and the RMSNorm reduction in a single LDS sweep, then a second
// pass loads ``hidden_out`` from registers (cached values), applies the RMS scale
// and per-element weight, and writes ``norm_out``.  Numerically equivalent to the
// unfused chain because both kernels accumulate the per-element sum in FP32 with
// the same per-element BF16 round at the residual store boundary.
//
// Layout assumptions:
//   input/residual: [rows, hidden_size] BF16 row-major
//   weight:         [hidden_size]       BF16
//   hidden_out:     [rows, hidden_size] BF16 row-major (residual sum, used by MLP residual path)
//   norm_out:       [rows, hidden_size] BF16 row-major (normalized output, MLP input)
//
// We require ``hidden_size <= 4096`` so the per-block scratch fits in 32 KiB of
// LDS at FP32 precision; current drafter has hidden_size=2048.  Use the unfused
// path for any larger model.

constexpr int64_t kDflashAddRmsNormMaxHidden = 4096;

__global__ void dflash_add_rmsnorm_bf16_kernel(
    const uint16_t* __restrict__ input,
    const uint16_t* __restrict__ residual,
    const uint16_t* __restrict__ weight,
    uint16_t* __restrict__ hidden_out,
    uint16_t* __restrict__ norm_out,
    int64_t rows,
    int64_t hidden_size,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  extern __shared__ float scratch[];
  // scratch[0..hidden_size)        : per-element residual sum (FP32 cache for second pass)
  // scratch[hidden_size..hidden_size + blockDim.x) : per-thread reduction tile
  float * sum_cache = scratch;
  float * partial = scratch + hidden_size;

  const int64_t base = row * hidden_size;
  float sumsq = 0.0f;
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < hidden_size; dim += blockDim.x) {
    const float a = bf16_bits_to_float(input[base + dim]);
    const float b = bf16_bits_to_float(residual[base + dim]);
    // Match unfused order exactly: round the residual sum to BF16 first (so the
    // RMS reduction sees the same value the unfused dflash_rmsnorm_bf16 would
    // have read back from HBM), then upcast to FP32 for the reduction.  This
    // keeps the fused output bit-identical to the unfused chain on every input.
    const uint16_t s_bits = float_to_bf16_bits(a + b);
    hidden_out[base + dim] = s_bits;
    const float s = bf16_bits_to_float(s_bits);
    sum_cache[dim] = s;
    sumsq += s * s;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < hidden_size; dim += blockDim.x) {
    const float scale = bf16_bits_to_float(weight[dim]);
    norm_out[base + dim] = float_to_bf16_bits(sum_cache[dim] * inv_rms * scale);
  }
}

__global__ void dflash_silu_mul_bf16_kernel(
    const uint16_t* __restrict__ gate,
    const uint16_t* __restrict__ up,
    uint16_t* __restrict__ out,
    int64_t elements) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const float g = bf16_bits_to_float(gate[i]);
    const float u = bf16_bits_to_float(up[i]);
    const float silu = g / (1.0f + expf(-g));
    out[i] = float_to_bf16_bits(silu * u);
  }
}

__global__ void dflash_silu_mul_gate_up_routes_bf16_kernel(
    const uint16_t* __restrict__ gate_up_routes,
    uint16_t* __restrict__ out_routes,
    int64_t routes,
    int64_t features) {
  const int64_t total = routes * features;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t route = idx / features;
    const int64_t dim = idx - route * features;
    const int64_t base = route * 2 * features;
    const float g = bf16_bits_to_float(gate_up_routes[base + dim]);
    const float u = bf16_bits_to_float(gate_up_routes[base + features + dim]);
    const float silu = g / (1.0f + expf(-g));
    out_routes[route * features + dim] = float_to_bf16_bits(silu * u);
  }
}

__global__ void dflash_dense_bf16_to_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ weight,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_col = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows || out_col >= out_features) {
    return;
  }
  extern __shared__ float partial[];
  const int64_t x_base = row * in_features;
  const int64_t w_base = out_col * in_features;
  float acc = 0.0f;
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < in_features; dim += blockDim.x) {
    acc = fmaf(bf16_bits_to_float(x[x_base + dim]), bf16_bits_to_float(weight[w_base + dim]), acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[row * out_features + out_col] = float_to_bf16_bits(partial[0]);
  }
}

// Expert-indexed dense GEMV: weight base resolved on-device from a routed
// expert id (`expert_ids[route]`), so the MoE expert loop never reads the
// router output back to the host (graph-capture-safe drafting).
__global__ void dflash_dense_bf16_to_bf16_expert_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ expert_weights_base,
    const int32_t* __restrict__ expert_ids,
    uint16_t* __restrict__ out,
    int64_t route,
    int64_t expert_stride_elems,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_col = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows || out_col >= out_features) {
    return;
  }
  const uint16_t* weight = expert_weights_base + static_cast<int64_t>(expert_ids[route]) * expert_stride_elems;
  extern __shared__ float partial[];
  const int64_t x_base = row * in_features;
  const int64_t w_base = out_col * in_features;
  float acc = 0.0f;
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < in_features; dim += blockDim.x) {
    acc = fmaf(bf16_bits_to_float(x[x_base + dim]), bf16_bits_to_float(weight[w_base + dim]), acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[row * out_features + out_col] = float_to_bf16_bits(partial[0]);
  }
}

__global__ void dflash_dense_bf16_to_bf16_expert_routes_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ expert_weights_base,
    const int32_t* __restrict__ expert_ids,
    uint16_t* __restrict__ out,
    int64_t routes,
    int64_t x_route_stride_elems,
    int64_t expert_stride_elems,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_col = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t route = static_cast<int64_t>(blockIdx.z);
  if (route >= routes || row >= rows || out_col >= out_features) {
    return;
  }
  const uint16_t* weight = expert_weights_base + static_cast<int64_t>(expert_ids[route]) * expert_stride_elems;
  const uint16_t* route_x = x_route_stride_elems > 0 ? x + route * x_route_stride_elems : x;
  extern __shared__ float partial[];
  const int64_t x_base = row * in_features;
  const int64_t w_base = out_col * in_features;
  float acc = 0.0f;
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < in_features; dim += blockDim.x) {
    acc = fmaf(bf16_bits_to_float(route_x[x_base + dim]), bf16_bits_to_float(weight[w_base + dim]), acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[(route * rows + row) * out_features + out_col] = float_to_bf16_bits(partial[0]);
  }
}

__global__ void dflash_dense_bf16_to_f32_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ weight,
    float* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_col = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows || out_col >= out_features) {
    return;
  }
  extern __shared__ float partial[];
  const int64_t x_base = row * in_features;
  const int64_t w_base = out_col * in_features;
  float acc = 0.0f;
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < in_features; dim += blockDim.x) {
    acc = fmaf(bf16_bits_to_float(x[x_base + dim]), bf16_bits_to_float(weight[w_base + dim]), acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[row * out_features + out_col] = partial[0];
  }
}

// -----------------------------------------------------------------------------
// R3.4: WMMA-tiled BF16 dense kernels for the DFlash drafter.
//
// One workgroup is one wave32 (32 lanes) and computes one 16x16 output tile
// using RDNA3's native ``v_wmma_f32_16x16x16_bf16`` instruction.  The K
// dimension must be divisible by 16; ``rows`` and ``out_features`` may be
// arbitrary (tiles round up; stores are bounds-checked).
//
// Tile mapping:
//   blockIdx.x = output column tile index (out_features tile, N direction)
//   blockIdx.y = row tile index (rows tile, output-row direction)
//   threadIdx.x in [0, 32) selects the wave lane.
//
// RDNA3 WMMA semantics (wave32):
//   D[m, n] = sum_k A[m, k] * B[k, n]
//   A operand: lane l holds 16 K elements of row ``l & 15``: A[l&15, 0..16)
//   B operand: lane l holds 16 K elements of column ``l & 15``: B[0..16, l&15)
//   D operand (FP32): lane l's element ``j`` lands at D[2*j + (l>>4), l&15]
//
// We want ``out[output_row, output_col] = sum_k x[output_row, k] * weight[output_col, k]``.
// Pack ``A := weight^T_tile`` and ``B := x^T_tile`` so the WMMA computes
// ``D[output_col_in_tile, output_row_in_tile]`` for our tile, then store with
// row = ``2*j + (l>>4)`` and col = ``l & 15``.  Concretely lane l loads:
//   a_reg[k] = weight[col_tile_base + (l & 15), k_offset + k]
//   b_reg[k] = x[row_tile_base + (l & 15), k_offset + k]
// No LDS staging is required for v1; weights and inputs are read once per
// output tile from global memory.  Tail rows / out_cols are handled by
// zero-fill on load and bounds-check on store.
// -----------------------------------------------------------------------------

typedef uint16_t bf16_t;
typedef bf16_t __attribute__((ext_vector_type(16))) bf16x16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

constexpr int kDflashWmmaWave = 32;
constexpr int kDflashWmmaTileM = 16;
constexpr int kDflashWmmaTileN = 16;
constexpr int kDflashWmmaTileK = 16;

__device__ inline float8_t dflash_wmma_zero_acc() {
  float8_t acc;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.0f;
  }
  return acc;
}

__device__ inline bf16x16_t dflash_wmma_zero_bf16x16() {
  bf16x16_t r;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    r[i] = static_cast<bf16_t>(0);
  }
  return r;
}

__device__ inline void dflash_wmma_load_operands(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ weight,
    int64_t b_row_in_x,
    int64_t a_row_in_weight,
    int64_t k_offset,
    int64_t in_features,
    bool x_in_range,
    bool w_in_range,
    bf16x16_t& a_reg,
    bf16x16_t& b_reg) {
  // A := weight tile (lane l holds weight row col_tile_base + (l & 15))
  if (w_in_range) {
    const uint16_t* w_ptr = weight + a_row_in_weight * in_features + k_offset;
    a_reg = *reinterpret_cast<const bf16x16_t*>(w_ptr);
  } else {
    a_reg = dflash_wmma_zero_bf16x16();
  }
  // B := x tile (lane l holds x row row_tile_base + (l & 15))
  if (x_in_range) {
    const uint16_t* x_ptr = x + b_row_in_x * in_features + k_offset;
    b_reg = *reinterpret_cast<const bf16x16_t*>(x_ptr);
  } else {
    b_reg = dflash_wmma_zero_bf16x16();
  }
}

__launch_bounds__(kDflashWmmaWave, 4)
__global__ void dflash_dense_bf16_to_bf16_wmma_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ weight,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int lane = static_cast<int>(threadIdx.x);
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;
  const int64_t row_tile_base = static_cast<int64_t>(blockIdx.y) * kDflashWmmaTileM;
  const int64_t col_tile_base = static_cast<int64_t>(blockIdx.x) * kDflashWmmaTileN;
  const int64_t b_row_in_x = row_tile_base + lane_lo;
  const int64_t a_row_in_weight = col_tile_base + lane_lo;
  const bool x_in_range = b_row_in_x < rows;
  const bool w_in_range = a_row_in_weight < out_features;

  float8_t acc = dflash_wmma_zero_acc();
  const int64_t k_tiles = in_features / kDflashWmmaTileK;
  for (int64_t kt = 0; kt < k_tiles; ++kt) {
    const int64_t k_offset = kt * kDflashWmmaTileK;
    bf16x16_t a_reg, b_reg;
    dflash_wmma_load_operands(x, weight, b_row_in_x, a_row_in_weight, k_offset, in_features, x_in_range, w_in_range, a_reg, b_reg);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_reg, b_reg, acc);
  }

  const int64_t store_row = row_tile_base + lane_lo;
  if (store_row >= rows) {
    return;
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t store_col = col_tile_base + (2 * j + lane_hi);
    if (store_col < out_features) {
      out[store_row * out_features + store_col] = float_to_bf16_bits(acc[j]);
    }
  }
}

__launch_bounds__(kDflashWmmaWave, 4)
__global__ void dflash_dense_bf16_to_f32_wmma_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ weight,
    float* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int lane = static_cast<int>(threadIdx.x);
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;
  const int64_t row_tile_base = static_cast<int64_t>(blockIdx.y) * kDflashWmmaTileM;
  const int64_t col_tile_base = static_cast<int64_t>(blockIdx.x) * kDflashWmmaTileN;
  const int64_t b_row_in_x = row_tile_base + lane_lo;
  const int64_t a_row_in_weight = col_tile_base + lane_lo;
  const bool x_in_range = b_row_in_x < rows;
  const bool w_in_range = a_row_in_weight < out_features;

  float8_t acc = dflash_wmma_zero_acc();
  const int64_t k_tiles = in_features / kDflashWmmaTileK;
  for (int64_t kt = 0; kt < k_tiles; ++kt) {
    const int64_t k_offset = kt * kDflashWmmaTileK;
    bf16x16_t a_reg, b_reg;
    dflash_wmma_load_operands(x, weight, b_row_in_x, a_row_in_weight, k_offset, in_features, x_in_range, w_in_range, a_reg, b_reg);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_reg, b_reg, acc);
  }

  const int64_t store_row = row_tile_base + lane_lo;
  if (store_row >= rows) {
    return;
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t store_col = col_tile_base + (2 * j + lane_hi);
    if (store_col < out_features) {
      out[store_row * out_features + store_col] = acc[j];
    }
  }
}

__host__ __device__ inline bool dflash_wmma_dense_shape_valid(int64_t rows, int64_t in_features, int64_t out_features) {
  return rows > 0 && in_features > 0 && out_features > 0 && (in_features % kDflashWmmaTileK) == 0;
}

__global__ void dflash_qkv_proj_bf16_mixed_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ q_weight,
    const uint16_t* __restrict__ k_weight,
    const uint16_t* __restrict__ v_weight,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    uint16_t* __restrict__ v_out,
    int64_t rows,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features) {
  const int64_t fused_col = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t total_features = q_features + 2 * kv_features;
  if (row >= rows || fused_col >= total_features) {
    return;
  }
  const uint16_t* weight = q_weight;
  int64_t out_col = fused_col;
  if (fused_col >= q_features + kv_features) {
    weight = v_weight;
    out_col = fused_col - q_features - kv_features;
  } else if (fused_col >= q_features) {
    weight = k_weight;
    out_col = fused_col - q_features;
  }
  extern __shared__ float partial[];
  const int64_t x_base = row * in_features;
  const int64_t w_base = out_col * in_features;
  float acc = 0.0f;
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < in_features; dim += blockDim.x) {
    acc = fmaf(bf16_bits_to_float(x[x_base + dim]), bf16_bits_to_float(weight[w_base + dim]), acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (fused_col < q_features) {
      q_out[row * q_features + out_col] = partial[0];
    } else if (fused_col < q_features + kv_features) {
      k_out[row * kv_features + out_col] = partial[0];
    } else {
      v_out[row * kv_features + out_col] = float_to_bf16_bits(partial[0]);
    }
  }
}

__global__ void dflash_qkv_proj_bf16_mixed_indexed_v_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ q_weight,
    const uint16_t* __restrict__ k_weight,
    const uint16_t* __restrict__ v_weight,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    uint16_t* __restrict__ value_cache_base,
    const int32_t* __restrict__ cache_slot,
    int64_t cache_rows,
    int64_t rows,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features) {
  const int64_t fused_col = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t total_features = q_features + 2 * kv_features;
  if (row >= rows || fused_col >= total_features) {
    return;
  }
  const uint16_t* weight = q_weight;
  int64_t out_col = fused_col;
  if (fused_col >= q_features + kv_features) {
    weight = v_weight;
    out_col = fused_col - q_features - kv_features;
  } else if (fused_col >= q_features) {
    weight = k_weight;
    out_col = fused_col - q_features;
  }
  extern __shared__ float partial[];
  const int64_t x_base = row * in_features;
  const int64_t w_base = out_col * in_features;
  float acc = 0.0f;
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < in_features; dim += blockDim.x) {
    acc = fmaf(bf16_bits_to_float(x[x_base + dim]), bf16_bits_to_float(weight[w_base + dim]), acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (fused_col < q_features) {
      q_out[row * q_features + out_col] = partial[0];
    } else if (fused_col < q_features + kv_features) {
      k_out[row * kv_features + out_col] = partial[0];
    } else {
      const int64_t slot = static_cast<int64_t>(cache_slot[0]) + row;
      if (slot >= 0 && slot < cache_rows) {
        value_cache_base[slot * kv_features + out_col] = float_to_bf16_bits(partial[0]);
      }
    }
  }
}

__device__ inline void dflash_head_rmsnorm_rotary_row_impl(
    const float* __restrict__ src,
    const uint16_t* __restrict__ weight,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ dst,
    float eps,
    int64_t head_dim,
    int64_t rotary_dim,
    float* __restrict__ partial) {
  float sumsq = 0.0f;
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < head_dim; dim += blockDim.x) {
    const float value = src[dim];
    sumsq += value * value;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf((partial[0] / static_cast<float>(head_dim)) + eps);
  const int64_t half_rotary = rotary_dim / 2;
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < head_dim; dim += blockDim.x) {
    const float value = src[dim] * inv_rms * bf16_bits_to_float(weight[dim]);
    float out_value = value;
    if (dim < rotary_dim) {
      const int64_t pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
      const float paired = src[pair_dim] * inv_rms * bf16_bits_to_float(weight[pair_dim]);
      const float rotated = dim < half_rotary ? -paired : paired;
      out_value = value * cos[dim] + rotated * sin[dim];
    }
    dst[dim] = out_value;
  }
}

__global__ void dflash_head_rmsnorm_rotary_f32_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const uint16_t* __restrict__ q_weight,
    const uint16_t* __restrict__ k_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int32_t* __restrict__ query_positions,
    const int32_t* __restrict__ key_positions,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    float eps) {
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t batch = static_cast<int64_t>(blockIdx.z);
  const bool is_query = head < num_q_heads;
  if (batch >= batch_size) {
    return;
  }
  extern __shared__ float partial[];
  if (is_query) {
    if (row >= query_len) {
      return;
    }
    const int64_t local_head = head;
    const int64_t position = static_cast<int64_t>(query_positions[batch * query_len + row]);
    if (position < 0 || position >= max_positions) {
      return;
    }
    const int64_t offset = ((batch * query_len + row) * num_q_heads + local_head) * head_dim;
    dflash_head_rmsnorm_rotary_row_impl(
        query + offset,
        q_weight,
        cos_table + position * rotary_dim,
        sin_table + position * rotary_dim,
        query_out + offset,
        eps,
        head_dim,
        rotary_dim,
        partial);
  } else {
    if (row >= kv_len) {
      return;
    }
    const int64_t local_head = head - num_q_heads;
    const int64_t position = static_cast<int64_t>(key_positions[batch * kv_len + row]);
    if (position < 0 || position >= max_positions) {
      return;
    }
    const int64_t offset = ((batch * kv_len + row) * num_kv_heads + local_head) * head_dim;
    dflash_head_rmsnorm_rotary_row_impl(
        key + offset,
        k_weight,
        cos_table + position * rotary_dim,
        sin_table + position * rotary_dim,
        key_out + offset,
        eps,
        head_dim,
        rotary_dim,
        partial);
  }
}

__global__ void dflash_head_rmsnorm_rotary_indexed_key_f32_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const uint16_t* __restrict__ q_weight,
    const uint16_t* __restrict__ k_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int32_t* __restrict__ query_positions,
    const int32_t* __restrict__ key_positions,
    float* __restrict__ query_out,
    float* __restrict__ key_cache_base,
    const int32_t* __restrict__ cache_slot,
    int64_t cache_rows,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    float eps) {
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t batch = static_cast<int64_t>(blockIdx.z);
  const bool is_query = head < num_q_heads;
  if (batch >= batch_size) {
    return;
  }
  extern __shared__ float partial[];
  if (is_query) {
    if (row >= query_len) {
      return;
    }
    const int64_t local_head = head;
    const int64_t position = static_cast<int64_t>(query_positions[batch * query_len + row]);
    if (position < 0 || position >= max_positions) {
      return;
    }
    const int64_t offset = ((batch * query_len + row) * num_q_heads + local_head) * head_dim;
    dflash_head_rmsnorm_rotary_row_impl(
        query + offset,
        q_weight,
        cos_table + position * rotary_dim,
        sin_table + position * rotary_dim,
        query_out + offset,
        eps,
        head_dim,
        rotary_dim,
        partial);
  } else {
    if (row >= kv_len) {
      return;
    }
    const int64_t local_head = head - num_q_heads;
    const int64_t position = static_cast<int64_t>(key_positions[batch * kv_len + row]);
    if (position < 0 || position >= max_positions) {
      return;
    }
    const int64_t slot = static_cast<int64_t>(cache_slot[0]) + row;
    if (slot < 0 || slot >= cache_rows) {
      return;
    }
    const int64_t src_offset = ((batch * kv_len + row) * num_kv_heads + local_head) * head_dim;
    const int64_t dst_offset = (slot * num_kv_heads + local_head) * head_dim;
    dflash_head_rmsnorm_rotary_row_impl(
        key + src_offset,
        k_weight,
        cos_table + position * rotary_dim,
        sin_table + position * rotary_dim,
        key_cache_base + dst_offset,
        eps,
        head_dim,
        rotary_dim,
        partial);
  }
}

__global__ void dflash_key_rmsnorm_rotary_f32_kernel(
    const float* __restrict__ key,
    const uint16_t* __restrict__ k_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int32_t* __restrict__ key_positions,
    float* __restrict__ key_out,
    int64_t rows,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    float eps) {
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows || head >= num_kv_heads) {
    return;
  }
  const int64_t position = static_cast<int64_t>(key_positions[row]);
  if (position < 0 || position >= max_positions) {
    return;
  }
  extern __shared__ float partial[];
  const int64_t offset = (row * num_kv_heads + head) * head_dim;
  dflash_head_rmsnorm_rotary_row_impl(
      key + offset,
      k_weight,
      cos_table + position * rotary_dim,
      sin_table + position * rotary_dim,
      key_out + offset,
      eps,
      head_dim,
      rotary_dim,
      partial);
}

__global__ void dflash_update_kv_metadata_i32_kernel(
    const int32_t* __restrict__ append_positions,
    int32_t* __restrict__ cache_positions,
    int32_t* __restrict__ live_count,
    int64_t start,
    int64_t count,
    int64_t end) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    cache_positions[start + i] = append_positions[i];
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    live_count[0] = static_cast<int32_t>(end);
  }
}

__global__ void dflash_gqa_attention_f32_bf16_bucketed_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const uint16_t* __restrict__ value,
    uint16_t* __restrict__ out,
    const int32_t* __restrict__ live_context_len,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t bucket_context_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  // Bucketed sibling of dflash_gqa_attention_f32_bf16_kernel: kv layout reserves
  // bucket_context_len rows for context (only the first *live_context_len are
  // valid; the trailing rows are padded zero/stale data) followed by query_len
  // rows at [bucket_context_len, kv_len).  We iterate only the valid rows in
  // TWO compact passes so masked rows cost nothing -- same per-thread work as
  // the unbucketed kernel when ``*live_context_len == bucket_context_len`` (no
  // padded rows are reachable in that case).  Bit-equivalent to the unbucketed
  // kernel when ``*live_context_len == bucket_context_len``.
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t q_row = static_cast<int64_t>(blockIdx.y);
  const int64_t batch = static_cast<int64_t>(blockIdx.z);
  if (batch >= batch_size || q_row >= query_len || q_head >= num_q_heads) {
    return;
  }
  extern __shared__ float shared[];
  float* partial = shared;
  float* scores = shared + blockDim.x;
  const int64_t group_size = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / group_size;
  const int64_t q_base = ((batch * query_len + q_row) * num_q_heads + q_head) * head_dim;
  const int64_t live = static_cast<int64_t>(live_context_len[0]);

  // Pass 1: valid context rows [0, live).  Score stride is contiguous.
  int64_t score_idx = 0;
  for (int64_t kv_row = 0; kv_row < live; ++kv_row) {
    const int64_t k_base = ((batch * kv_len + kv_row) * num_kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      dot += query[q_base + dim] * key[k_base + dim];
    }
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      scores[score_idx] = partial[0] * scale;
    }
    ++score_idx;
    __syncthreads();
  }
  // Pass 2: query rows [bucket_context_len, kv_len).
  for (int64_t kv_row = bucket_context_len; kv_row < kv_len; ++kv_row) {
    const int64_t k_base = ((batch * kv_len + kv_row) * num_kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      dot += query[q_base + dim] * key[k_base + dim];
    }
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      scores[score_idx] = partial[0] * scale;
    }
    ++score_idx;
    __syncthreads();
  }
  const int64_t score_count = score_idx;

  if (threadIdx.x == 0) {
    float max_score = -3.4028234663852886e+38F;
    for (int64_t i = 0; i < score_count; ++i) {
      max_score = fmaxf(max_score, scores[i]);
    }
    float denom = 0.0f;
    for (int64_t i = 0; i < score_count; ++i) {
      const float prob = expf(scores[i] - max_score);
      scores[i] = prob;
      denom += prob;
    }
    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
    for (int64_t i = 0; i < score_count; ++i) {
      scores[i] *= inv_denom;
    }
  }
  __syncthreads();

  const int64_t out_base = ((batch * query_len + q_row) * num_q_heads + q_head) * head_dim;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    int64_t i = 0;
    for (int64_t kv_row = 0; kv_row < live; ++kv_row) {
      const int64_t v_base = ((batch * kv_len + kv_row) * num_kv_heads + kv_head) * head_dim;
      acc += scores[i] * bf16_bits_to_float(value[v_base + dim]);
      ++i;
    }
    for (int64_t kv_row = bucket_context_len; kv_row < kv_len; ++kv_row) {
      const int64_t v_base = ((batch * kv_len + kv_row) * num_kv_heads + kv_head) * head_dim;
      acc += scores[i] * bf16_bits_to_float(value[v_base + dim]);
      ++i;
    }
    out[out_base + dim] = float_to_bf16_bits(acc);
  }
}

__global__ void dflash_gqa_attention_f32_bf16_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const uint16_t* __restrict__ value,
    uint16_t* __restrict__ out,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t q_row = static_cast<int64_t>(blockIdx.y);
  const int64_t batch = static_cast<int64_t>(blockIdx.z);
  if (batch >= batch_size || q_row >= query_len || q_head >= num_q_heads) {
    return;
  }
  extern __shared__ float shared[];
  float* partial = shared;
  float* scores = shared + blockDim.x;
  const int64_t group_size = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / group_size;
  const int64_t q_base = ((batch * query_len + q_row) * num_q_heads + q_head) * head_dim;

  for (int64_t kv_row = 0; kv_row < kv_len; ++kv_row) {
    const int64_t k_base = ((batch * kv_len + kv_row) * num_kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      dot += query[q_base + dim] * key[k_base + dim];
    }
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      scores[kv_row] = partial[0] * scale;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float max_score = -3.4028234663852886e+38F;
    for (int64_t kv_row = 0; kv_row < kv_len; ++kv_row) {
      max_score = fmaxf(max_score, scores[kv_row]);
    }
    float denom = 0.0f;
    for (int64_t kv_row = 0; kv_row < kv_len; ++kv_row) {
      const float prob = expf(scores[kv_row] - max_score);
      scores[kv_row] = prob;
      denom += prob;
    }
    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
    for (int64_t kv_row = 0; kv_row < kv_len; ++kv_row) {
      scores[kv_row] *= inv_denom;
    }
  }
  __syncthreads();

  const int64_t out_base = ((batch * query_len + q_row) * num_q_heads + q_head) * head_dim;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t kv_row = 0; kv_row < kv_len; ++kv_row) {
      const int64_t v_base = ((batch * kv_len + kv_row) * num_kv_heads + kv_head) * head_dim;
      acc += scores[kv_row] * bf16_bits_to_float(value[v_base + dim]);
    }
    out[out_base + dim] = float_to_bf16_bits(acc);
  }
}

bool valid_elements(int64_t elements, int64_t threads) {
  return elements > 0 && valid_threads(threads);
}

bool valid_concat_shape(int64_t batch_size, int64_t context_len, int64_t query_len, int64_t features, int64_t threads) {
  return batch_size > 0 && context_len >= 0 && query_len > 0 && features > 0 && (context_len + query_len) > 0 &&
         valid_threads(threads);
}

bool valid_dense_shape(int64_t rows, int64_t in_features, int64_t out_features, int64_t threads) {
  return rows > 0 && in_features > 0 && out_features > 0 && valid_threads(threads);
}

bool valid_qkv_proj_shape(int64_t rows, int64_t in_features, int64_t q_features, int64_t kv_features, int64_t threads) {
  return rows > 0 && in_features > 0 && q_features > 0 && kv_features > 0 && valid_threads(threads);
}

bool valid_rmsnorm_shape(int64_t rows, int64_t hidden_size, int64_t threads) {
  return rows > 0 && hidden_size > 0 && valid_threads(threads);
}

bool valid_head_rotary_shape(
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t threads) {
  return batch_size > 0 && query_len > 0 && kv_len > 0 && num_q_heads > 0 && num_kv_heads > 0 &&
         head_dim > 0 && rotary_dim > 0 && rotary_dim <= head_dim && (rotary_dim % 2) == 0 &&
         max_positions > 0 && valid_threads(threads);
}

bool valid_key_rotary_shape(
    int64_t rows,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t threads) {
  return rows > 0 && num_kv_heads > 0 && head_dim > 0 && rotary_dim > 0 && rotary_dim <= head_dim &&
         (rotary_dim % 2) == 0 && max_positions > 0 && valid_threads(threads);
}

bool valid_attention_shape(
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t threads) {
  return batch_size > 0 && query_len > 0 && kv_len > 0 && num_q_heads > 0 && num_kv_heads > 0 &&
         head_dim > 0 && (num_q_heads % num_kv_heads) == 0 && valid_threads(threads);
}

bool valid_bucketed_attention_shape(
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t bucket_context_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t threads) {
  return bucket_context_len >= 0 && bucket_context_len <= kv_len &&
         valid_attention_shape(batch_size, query_len, kv_len, num_q_heads, num_kv_heads, head_dim, threads);
}

}  // namespace

extern "C" int hipengine_dflash_prepare_noise_inputs_bf16_i32(
    const int32_t* root_tokens,
    const int32_t* root_positions,
    const uint16_t* embed_tokens,
    int32_t* noise_token_ids,
    int32_t* position_ids,
    uint16_t* noise_embeddings,
    int64_t request_count,
    int64_t block_size,
    int64_t hidden_size,
    int64_t vocab_size,
    int32_t mask_token_id,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(request_count, block_size, hidden_size, vocab_size) || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (dflash_prepare_noise_inputs_bf16_i32_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(request_count), static_cast<unsigned int>(block_size)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      root_tokens,
      root_positions,
      embed_tokens,
      noise_token_ids,
      position_ids,
      noise_embeddings,
      request_count,
      block_size,
      hidden_size,
      vocab_size,
      mask_token_id);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_add_bf16(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* out,
    int64_t elements,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_elements(elements, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (elements + threads - 1) / threads;
  const int64_t capped_blocks = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      dflash_add_bf16_kernel,
      dim3(static_cast<unsigned int>(capped_blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      a,
      b,
      out,
      elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_concat_rows_f32(
    const float* context_rows,
    const float* query_rows,
    float* out_rows,
    int64_t batch_size,
    int64_t context_len,
    int64_t query_len,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_concat_shape(batch_size, context_len, query_len, features, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t elements = batch_size * (context_len + query_len) * features;
  const int64_t blocks = (elements + threads - 1) / threads;
  const int64_t capped_blocks = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      (dflash_concat_context_query_rows_kernel<float>),
      dim3(static_cast<unsigned int>(capped_blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      context_rows,
      query_rows,
      out_rows,
      batch_size,
      context_len,
      query_len,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_concat_rows_bf16(
    const uint16_t* context_rows,
    const uint16_t* query_rows,
    uint16_t* out_rows,
    int64_t batch_size,
    int64_t context_len,
    int64_t query_len,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_concat_shape(batch_size, context_len, query_len, features, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t elements = batch_size * (context_len + query_len) * features;
  const int64_t blocks = (elements + threads - 1) / threads;
  const int64_t capped_blocks = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      (dflash_concat_context_query_rows_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(capped_blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      context_rows,
      query_rows,
      out_rows,
      batch_size,
      context_len,
      query_len,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_rmsnorm_bf16(
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_rmsnorm_shape(rows, hidden_size, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      dflash_rmsnorm_bf16_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      x,
      weight,
      out,
      rows,
      hidden_size,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_add_rmsnorm_bf16(
    const uint16_t* input,
    const uint16_t* residual,
    const uint16_t* weight,
    uint16_t* hidden_out,
    uint16_t* norm_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || hidden_size > kDflashAddRmsNormMaxHidden ||
      threads != 64 && threads != 128 && threads != 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const size_t shared_bytes =
      static_cast<size_t>(hidden_size) * sizeof(float) +
      static_cast<size_t>(threads) * sizeof(float);
  hipLaunchKernelGGL(
      dflash_add_rmsnorm_bf16_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      shared_bytes,
      stream,
      input,
      residual,
      weight,
      hidden_out,
      norm_out,
      rows,
      hidden_size,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_silu_mul_bf16(
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* out,
    int64_t elements,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_elements(elements, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (elements + threads - 1) / threads;
  const int64_t capped_blocks = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      dflash_silu_mul_bf16_kernel,
      dim3(static_cast<unsigned int>(capped_blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      gate,
      up,
      out,
      elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_silu_mul_gate_up_routes_bf16(
    const uint16_t* gate_up_routes,
    uint16_t* out_routes,
    int64_t routes,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (routes <= 0 || !valid_elements(features, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t elements = routes * features;
  const int64_t blocks = (elements + threads - 1) / threads;
  const int64_t capped_blocks = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      dflash_silu_mul_gate_up_routes_bf16_kernel,
      dim3(static_cast<unsigned int>(capped_blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      gate_up_routes,
      out_routes,
      routes,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_dense_bf16_to_bf16(
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_dense_shape(rows, in_features, out_features, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      dflash_dense_bf16_to_bf16_kernel,
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      x,
      weight,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_dense_bf16_to_bf16_expert(
    const uint16_t* x,
    const uint16_t* expert_weights_base,
    const int32_t* expert_ids,
    uint16_t* out,
    int64_t route,
    int64_t expert_stride_elems,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_dense_shape(rows, in_features, out_features, threads) || route < 0 || expert_stride_elems <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      dflash_dense_bf16_to_bf16_expert_kernel,
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      x,
      expert_weights_base,
      expert_ids,
      out,
      route,
      expert_stride_elems,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_dense_bf16_to_bf16_expert_routes(
    const uint16_t* x,
    const uint16_t* expert_weights_base,
    const int32_t* expert_ids,
    uint16_t* out,
    int64_t routes,
    int64_t x_route_stride_elems,
    int64_t expert_stride_elems,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_dense_shape(rows, in_features, out_features, threads) || routes <= 0 ||
      x_route_stride_elems < 0 || expert_stride_elems <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      dflash_dense_bf16_to_bf16_expert_routes_kernel,
      dim3(
          static_cast<unsigned int>(out_features),
          static_cast<unsigned int>(rows),
          static_cast<unsigned int>(routes)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      x,
      expert_weights_base,
      expert_ids,
      out,
      routes,
      x_route_stride_elems,
      expert_stride_elems,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_dense_bf16_to_f32(
    const uint16_t* x,
    const uint16_t* weight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_dense_shape(rows, in_features, out_features, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      dflash_dense_bf16_to_f32_kernel,
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      x,
      weight,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_dense_bf16_to_bf16_wmma(
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!dflash_wmma_dense_shape_valid(rows, in_features, out_features)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t row_tiles = (rows + kDflashWmmaTileM - 1) / kDflashWmmaTileM;
  const int64_t col_tiles = (out_features + kDflashWmmaTileN - 1) / kDflashWmmaTileN;
  hipLaunchKernelGGL(
      dflash_dense_bf16_to_bf16_wmma_kernel,
      dim3(static_cast<unsigned int>(col_tiles), static_cast<unsigned int>(row_tiles)),
      dim3(static_cast<unsigned int>(kDflashWmmaWave)),
      0,
      stream,
      x,
      weight,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_dense_bf16_to_f32_wmma(
    const uint16_t* x,
    const uint16_t* weight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!dflash_wmma_dense_shape_valid(rows, in_features, out_features)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t row_tiles = (rows + kDflashWmmaTileM - 1) / kDflashWmmaTileM;
  const int64_t col_tiles = (out_features + kDflashWmmaTileN - 1) / kDflashWmmaTileN;
  hipLaunchKernelGGL(
      dflash_dense_bf16_to_f32_wmma_kernel,
      dim3(static_cast<unsigned int>(col_tiles), static_cast<unsigned int>(row_tiles)),
      dim3(static_cast<unsigned int>(kDflashWmmaWave)),
      0,
      stream,
      x,
      weight,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_qkv_proj_bf16_mixed(
    const uint16_t* x,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const uint16_t* v_weight,
    float* q_out,
    float* k_out,
    uint16_t* v_out,
    int64_t rows,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_qkv_proj_shape(rows, in_features, q_features, kv_features, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total_features = q_features + 2 * kv_features;
  hipLaunchKernelGGL(
      dflash_qkv_proj_bf16_mixed_kernel,
      dim3(static_cast<unsigned int>(total_features), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      x,
      q_weight,
      k_weight,
      v_weight,
      q_out,
      k_out,
      v_out,
      rows,
      in_features,
      q_features,
      kv_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_qkv_proj_bf16_mixed_indexed_v(
    const uint16_t* x,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const uint16_t* v_weight,
    float* q_out,
    float* k_out,
    uint16_t* value_cache_base,
    const int32_t* cache_slot,
    int64_t cache_rows,
    int64_t rows,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_qkv_proj_shape(rows, in_features, q_features, kv_features, threads) || cache_rows <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total_features = q_features + 2 * kv_features;
  hipLaunchKernelGGL(
      dflash_qkv_proj_bf16_mixed_indexed_v_kernel,
      dim3(static_cast<unsigned int>(total_features), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      x,
      q_weight,
      k_weight,
      v_weight,
      q_out,
      k_out,
      value_cache_base,
      cache_slot,
      cache_rows,
      rows,
      in_features,
      q_features,
      kv_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_head_rmsnorm_rotary_f32(
    const float* query,
    const float* key,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int32_t* query_positions,
    const int32_t* key_positions,
    float* query_out,
    float* key_out,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_head_rotary_shape(batch_size, query_len, kv_len, num_q_heads, num_kv_heads, head_dim, rotary_dim, max_positions, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_len = query_len > kv_len ? query_len : kv_len;
  hipLaunchKernelGGL(
      dflash_head_rmsnorm_rotary_f32_kernel,
      dim3(static_cast<unsigned int>(num_q_heads + num_kv_heads), static_cast<unsigned int>(max_len), static_cast<unsigned int>(batch_size)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      query,
      key,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      query_positions,
      key_positions,
      query_out,
      key_out,
      batch_size,
      query_len,
      kv_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_head_rmsnorm_rotary_indexed_key_f32(
    const float* query,
    const float* key,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int32_t* query_positions,
    const int32_t* key_positions,
    float* query_out,
    float* key_cache_base,
    const int32_t* cache_slot,
    int64_t cache_rows,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_head_rotary_shape(batch_size, query_len, kv_len, num_q_heads, num_kv_heads, head_dim, rotary_dim, max_positions, threads) ||
      cache_rows <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_len = query_len > kv_len ? query_len : kv_len;
  hipLaunchKernelGGL(
      dflash_head_rmsnorm_rotary_indexed_key_f32_kernel,
      dim3(static_cast<unsigned int>(num_q_heads + num_kv_heads), static_cast<unsigned int>(max_len), static_cast<unsigned int>(batch_size)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      query,
      key,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      query_positions,
      key_positions,
      query_out,
      key_cache_base,
      cache_slot,
      cache_rows,
      batch_size,
      query_len,
      kv_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_key_rmsnorm_rotary_f32(
    const float* key,
    const uint16_t* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int32_t* key_positions,
    float* key_out,
    int64_t rows,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_key_rotary_shape(rows, num_kv_heads, head_dim, rotary_dim, max_positions, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      dflash_key_rmsnorm_rotary_f32_kernel,
      dim3(static_cast<unsigned int>(num_kv_heads), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      key,
      k_weight,
      cos_table,
      sin_table,
      key_positions,
      key_out,
      rows,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_update_kv_metadata_i32(
    const int32_t* append_positions,
    int32_t* cache_positions,
    int32_t* live_count,
    int64_t start,
    int64_t count,
    int64_t end,
    int64_t threads,
    hipStream_t stream) {
  if (start < 0 || count < 0 || end < start || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = count <= 0 ? 1 : ((count + threads - 1) / threads);
  const int64_t capped_blocks = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      dflash_update_kv_metadata_i32_kernel,
      dim3(static_cast<unsigned int>(capped_blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      append_positions,
      cache_positions,
      live_count,
      start,
      count,
      end);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_gqa_attention_f32_bf16_bucketed(
    const float* query,
    const float* key,
    const uint16_t* value,
    uint16_t* out,
    const int32_t* live_context_len,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t bucket_context_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_bucketed_attention_shape(
          batch_size, query_len, kv_len, bucket_context_len, num_q_heads, num_kv_heads, head_dim, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const size_t smem = static_cast<size_t>(threads + kv_len) * sizeof(float);
  hipLaunchKernelGGL(
      dflash_gqa_attention_f32_bf16_bucketed_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(query_len), static_cast<unsigned int>(batch_size)),
      dim3(static_cast<unsigned int>(threads)),
      smem,
      stream,
      query,
      key,
      value,
      out,
      live_context_len,
      batch_size,
      query_len,
      kv_len,
      bucket_context_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_gqa_attention_f32_bf16(
    const float* query,
    const float* key,
    const uint16_t* value,
    uint16_t* out,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_attention_shape(batch_size, query_len, kv_len, num_q_heads, num_kv_heads, head_dim, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const size_t smem = static_cast<size_t>(threads + kv_len) * sizeof(float);
  hipLaunchKernelGGL(
      dflash_gqa_attention_f32_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(query_len), static_cast<unsigned int>(batch_size)),
      dim3(static_cast<unsigned int>(threads)),
      smem,
      stream,
      query,
      key,
      value,
      out,
      batch_size,
      query_len,
      kv_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_prepare_noise_inputs_f16_to_bf16_i32(
    const int32_t* root_tokens,
    const int32_t* root_positions,
    const half_t* embed_tokens,
    int32_t* noise_token_ids,
    int32_t* position_ids,
    uint16_t* noise_embeddings,
    int64_t request_count,
    int64_t block_size,
    int64_t hidden_size,
    int64_t vocab_size,
    int32_t mask_token_id,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(request_count, block_size, hidden_size, vocab_size) || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (dflash_prepare_noise_inputs_bf16_i32_kernel<half_t>),
      dim3(static_cast<unsigned int>(request_count), static_cast<unsigned int>(block_size)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      root_tokens,
      root_positions,
      embed_tokens,
      noise_token_ids,
      position_ids,
      noise_embeddings,
      request_count,
      block_size,
      hidden_size,
      vocab_size,
      mask_token_id);
  return static_cast<int>(hipGetLastError());
}
