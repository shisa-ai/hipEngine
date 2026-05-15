// Ported from nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py @ 5d8f496.
// Compact-buffer PARO/AWQ WMMA kernels for grouped Qwen3.5 MoE prefill.

#include <hip/hip_runtime.h>
#include <stdint.h>

using half_t = _Float16;
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

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

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t v) {
  return bf16_bits_to_float(v);
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

__device__ inline int awq_shift_for_pack_lane(int lane) {
  const int packed_pos = (lane & 1) ? (4 + (lane >> 1)) : (lane >> 1);
  return packed_pos * 4;
}

template <typename scalar_t>
__launch_bounds__(32, 2)
__global__ void gemm_awq_selected_dual_pack8_wmma_compact_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const int32_t* __restrict__ qweight_a_t,
    const int32_t* __restrict__ qzeros_a,
    const scalar_t* __restrict__ scales_a,
    const int32_t* __restrict__ qweight_b_t,
    const int32_t* __restrict__ qzeros_b,
    const scalar_t* __restrict__ scales_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int group_size) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;

  const int64_t col_start = static_cast<int64_t>(col_tile) * 16;
  const int64_t out_features_a = out_packed_a * 8;
  const int64_t out_features_b = out_packed_b * 8;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (col_start >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const bool use_b = col_start >= out_features_a;
  const int64_t local_col_start = use_b ? (col_start - out_features_a) : col_start;
  const int64_t local_out_packed = use_b ? out_packed_b : out_packed_a;
  const int64_t local_out_features = local_out_packed * 8;
  const int32_t* __restrict__ qweight_t = use_b ? qweight_b_t : qweight_a_t;
  const int32_t* __restrict__ qzeros = use_b ? qzeros_b : qzeros_a;
  const scalar_t* __restrict__ scales = use_b ? scales_b : scales_a;

  const int64_t ew_stride = local_out_packed * in_features;
  const int32_t* __restrict__ expert_qw = qweight_t + expert_id * ew_stride;
  const int64_t ez_stride = (in_features / group_size) * local_out_packed;
  const int32_t* __restrict__ expert_qz = qzeros + expert_id * ez_stride;
  const int64_t es_stride = (in_features / group_size) * local_out_features;
  const scalar_t* __restrict__ expert_sc = scales + expert_id * es_stride;

  const int t_col = tid & 15;
  const int64_t out_col = local_col_start + t_col;
  const int64_t out_pack = out_col >> 3;
  const int out_lane = static_cast<int>(out_col & 7);
  const int shift = awq_shift_for_pack_lane(out_lane);
  const int t_row_in_tile = tid & 15;
  const int64_t local_row = tile_row_in_expert + t_row_in_tile;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;

  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int64_t groups = in_features / group_size;
  const int64_t k_tiles_per_group = group_size / 16;
  for (int64_t g = 0; g < groups; g++) {
    float sc_f = 0.f;
    int zp_int = 0;
    if (out_col < local_out_features) {
      sc_f = scalar_to_float(expert_sc[g * local_out_features + out_col]);
      const uint32_t pz = static_cast<uint32_t>(expert_qz[g * local_out_packed + out_pack]);
      zp_int = static_cast<int>((pz >> shift) & 0xF);
    }
    const _Float16 sc_h = (_Float16)sc_f;
    const _Float16 zp_h = (_Float16)(float)(-zp_int) * sc_h;
    for (int64_t kt = 0; kt < k_tiles_per_group; kt++) {
      const int64_t k_base = g * group_size + kt * 16;
      half16_t a_reg;
      if (valid_row) {
        const scalar_t* __restrict__ x_ptr = x + compact_row * in_features + k_base;
#pragma unroll
        for (int i = 0; i < 16; i++) a_reg[i] = (_Float16)scalar_to_float(x_ptr[i]);
      } else {
#pragma unroll
        for (int i = 0; i < 16; i++) a_reg[i] = (_Float16)0.0f;
      }
      half16_t b_reg;
      if (out_col < local_out_features) {
        const int32_t* __restrict__ qw_ptr = expert_qw + out_pack * in_features + k_base;
#pragma unroll
        for (int i = 0; i < 16; i++) {
          const uint32_t pw = static_cast<uint32_t>(qw_ptr[i]);
          const int w_int = static_cast<int>((pw >> shift) & 0xF);
          b_reg[i] = sc_h * (_Float16)(float)w_int + zp_h;
        }
      } else {
#pragma unroll
        for (int i = 0; i < 16; i++) b_reg[i] = (_Float16)0.0f;
      }
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
    }
  }

  const int half_id = tid >> 4;
  const int64_t global_out_col = col_start + t_col;
  if (global_out_col < out_features_total) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
      if (local_out_row < expert_count) {
        const int64_t compact_out_row = compact_base + local_out_row;
        if (compact_out_row < compact_rows) {
          out[compact_out_row * out_features_total + global_out_col] = float_to_scalar<scalar_t>(acc[j]);
        }
      }
    }
  }
}

template <typename scalar_t>
__launch_bounds__(32, 2)
__global__ void gemm_awq_selected_pack8_wmma_compact_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const int32_t* __restrict__ qweight_t,
    const int32_t* __restrict__ qzeros,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int group_size) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t col_start = static_cast<int64_t>(col_tile) * 16;
  const int64_t out_features = out_packed * 8;
  if (col_start >= out_features) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const int64_t ew_stride = out_packed * in_features;
  const int32_t* __restrict__ expert_qw = qweight_t + expert_id * ew_stride;
  const int64_t ez_stride = (in_features / group_size) * out_packed;
  const int32_t* __restrict__ expert_qz = qzeros + expert_id * ez_stride;
  const int64_t es_stride = (in_features / group_size) * out_features;
  const scalar_t* __restrict__ expert_sc = scales + expert_id * es_stride;

  const int t_col = tid & 15;
  const int64_t out_col = col_start + t_col;
  const int64_t out_pack = out_col >> 3;
  const int out_lane = static_cast<int>(out_col & 7);
  const int shift = awq_shift_for_pack_lane(out_lane);
  const int t_row_in_tile = tid & 15;
  const int64_t local_row = tile_row_in_expert + t_row_in_tile;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;

  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int64_t groups = in_features / group_size;
  const int64_t k_tiles_per_group = group_size / 16;
  for (int64_t g = 0; g < groups; g++) {
    float sc_f = 0.f;
    int zp_int = 0;
    if (out_col < out_features) {
      sc_f = scalar_to_float(expert_sc[g * out_features + out_col]);
      const uint32_t pz = static_cast<uint32_t>(expert_qz[g * out_packed + out_pack]);
      zp_int = static_cast<int>((pz >> shift) & 0xF);
    }
    const _Float16 sc_h = (_Float16)sc_f;
    const _Float16 zp_h = (_Float16)(float)(-zp_int) * sc_h;
    for (int64_t kt = 0; kt < k_tiles_per_group; kt++) {
      const int64_t k_base = g * group_size + kt * 16;
      half16_t a_reg;
      if (valid_row) {
        const scalar_t* __restrict__ x_ptr = x + compact_row * in_features + k_base;
#pragma unroll
        for (int i = 0; i < 16; i++) a_reg[i] = (_Float16)scalar_to_float(x_ptr[i]);
      } else {
#pragma unroll
        for (int i = 0; i < 16; i++) a_reg[i] = (_Float16)0.0f;
      }
      half16_t b_reg;
      if (out_col < out_features) {
        const int32_t* __restrict__ qw_ptr = expert_qw + out_pack * in_features + k_base;
#pragma unroll
        for (int i = 0; i < 16; i++) {
          const uint32_t pw = static_cast<uint32_t>(qw_ptr[i]);
          const int w_int = static_cast<int>((pw >> shift) & 0xF);
          b_reg[i] = sc_h * (_Float16)(float)w_int + zp_h;
        }
      } else {
#pragma unroll
        for (int i = 0; i < 16; i++) b_reg[i] = (_Float16)0.0f;
      }
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
    }
  }

  const int half_id = tid >> 4;
  if (out_col < out_features) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
      if (local_out_row < expert_count) {
        const int64_t compact_out_row = compact_base + local_out_row;
        if (compact_out_row < compact_rows) {
          out[compact_out_row * out_features + out_col] = float_to_scalar<scalar_t>(acc[j]);
        }
      }
    }
  }
}

bool valid_common(
    int64_t compact_rows,
    int64_t in_features,
    int64_t num_experts,
    int64_t group_size,
    int64_t wmma_total_rows) {
  return compact_rows > 0 && in_features > 0 && num_experts > 0 && group_size > 0 &&
      (group_size % 16) == 0 && (in_features % group_size) == 0 &&
      wmma_total_rows > 0 && (wmma_total_rows % 16) == 0;
}

}  // namespace

template <typename scalar_t>
int launch_dual(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const int32_t* qweight_a_t,
    const int32_t* qzeros_a,
    const scalar_t* scales_a,
    const int32_t* qweight_b_t,
    const int32_t* qzeros_b,
    const scalar_t* scales_b,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_common(compact_rows, in_features, num_experts, group_size, wmma_total_rows) ||
      out_packed_a <= 0 || out_packed_b <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = (out_packed_a + out_packed_b) * 8;
  hipLaunchKernelGGL(
      (gemm_awq_selected_dual_pack8_wmma_compact_kernel<scalar_t>),
      dim3(static_cast<unsigned int>((out_features_total + 15) / 16), static_cast<unsigned int>(wmma_total_rows / 16)),
      dim3(32),
      0,
      stream,
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a_t,
      qzeros_a,
      scales_a,
      qweight_b_t,
      qzeros_b,
      scales_b,
      out,
      compact_rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t>
int launch_single(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const int32_t* qweight_t,
    const int32_t* qzeros,
    const scalar_t* scales,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_common(compact_rows, in_features, num_experts, group_size, wmma_total_rows) || out_packed <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features = out_packed * 8;
  hipLaunchKernelGGL(
      (gemm_awq_selected_pack8_wmma_compact_kernel<scalar_t>),
      dim3(static_cast<unsigned int>((out_features + 15) / 16), static_cast<unsigned int>(wmma_total_rows / 16)),
      dim3(32),
      0,
      stream,
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_t,
      qzeros,
      scales,
      out,
      compact_rows,
      in_features,
      out_packed,
      num_experts,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemm_awq_selected_dual_pack8_wmma_compact_bf16(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const int32_t* qweight_a_t,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b_t,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_dual<uint16_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a_t,
      qzeros_a,
      scales_a,
      qweight_b_t,
      qzeros_b,
      scales_b,
      out,
      compact_rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      group_size,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gemm_awq_selected_dual_pack8_wmma_compact_fp16(
    const half_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const int32_t* qweight_a_t,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b_t,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_dual<half_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a_t,
      qzeros_a,
      scales_a,
      qweight_b_t,
      qzeros_b,
      scales_b,
      out,
      compact_rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      group_size,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gemm_awq_selected_pack8_wmma_compact_bf16(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const int32_t* qweight_t,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_single<uint16_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_t,
      qzeros,
      scales,
      out,
      compact_rows,
      in_features,
      out_packed,
      num_experts,
      group_size,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gemm_awq_selected_pack8_wmma_compact_fp16(
    const half_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const int32_t* qweight_t,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_single<half_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_t,
      qzeros,
      scales,
      out,
      compact_rows,
      in_features,
      out_packed,
      num_experts,
      group_size,
      wmma_total_rows,
      stream);
}
