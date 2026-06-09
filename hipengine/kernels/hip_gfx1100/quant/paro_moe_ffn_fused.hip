// B3: fused selected-expert PARO (AWQ W4 pack8 + incoherence rotations) MoE FFN
// megakernel. One block per selected (token, expert) row computes the *whole*
// expert FFN with both PARO rotations folded in:
//
//   x_rot   = rotate1(x)                 [shared incoherence rotation, hidden]
//   gate    = Wg_e @ x_rot ; up = Wu_e @ x_rot
//   inter   = silu(gate) * up
//   inter_r = down_rotate(inter)         [shared incoherence rotation, ffn_len]
//   down    = Wd_e @ inter_r
//
// keeping the hidden-wide x_rot and ffn_len-wide intermediate on-chip (LDS) so
// the rotate-out / gate_up-out HBM writes and the corresponding reads vanish,
// and the rotate1 + dual gate_up + silu/down-rotate + down launches collapse to
// one. Output is the per-selected-row down projection ``out[rows, hidden]``; the
// routing-weighted combine stays a separate kernel.
//
// Structure mirrors the GGUF B1 megakernel (gguf_q4_k_moe_ffn_fused.hip): each
// thread owns whole output columns (no cross-thread GEMV reduction), so the
// kernel is row-invariant by construction (a row's result is identical alone or
// batched -- the T1 verifier requirement). The cooperative Givens rotation and
// AWQ pack8 dequant math are forked from paro_awq_gemv.hip
// (gemv_awq_dual_pack8_transposed_rotate_staged_kernel / gemv_awq_pack8_kernel,
// nano-vllm-amd@59195ed) and must stay bit-for-bit faithful to the cpu_reference
// paro_moe_selected_ffn oracle within the project KL<=0.05 gate.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

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
__device__ inline float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

template <typename out_t>
__device__ inline void store_output(out_t* out, int64_t idx, float value) {
  out[idx] = static_cast<out_t>(value);
}

template <>
__device__ inline void store_output<uint16_t>(uint16_t* out, int64_t idx, float value) {
  out[idx] = float_to_bf16_bits(value);
}

// Round an FFN intermediate to the engine's working precision. The bf16 output
// variant rounds gate/up and silu(gate)*up to bf16 so the fused kernel tracks
// the unfused dual_gemv -> silu_mul -> down chain (which materializes those
// intermediates as bf16). The f32 variant keeps full fp32 as the reference path.
template <typename out_t>
__device__ inline float round_intermediate(float value) {
  return value;
}

template <>
__device__ inline float round_intermediate<uint16_t>(float value) {
  return bf16_bits_to_float(float_to_bf16_bits(value));
}

// FP16 variant: round gate/up and silu(gate)*up to fp16 so the fused kernel
// tracks the unfused fp16 dual_gemv -> silu_mul -> down chain (which
// materializes those intermediates as fp16). scalar_to_float / store_output use
// the generic _Float16 casts (native), so only the rounding needs specializing.
template <>
__device__ inline float round_intermediate<_Float16>(float value) {
  return static_cast<float>(static_cast<_Float16>(value));
}

// AWQ pack8 interleaved nibble shift (matches loading.qwen35_paro.awq_pack8_shift_for_lane
// and paro_awq_gemv.hip awq_shift_for_pack_lane).
__device__ inline int awq_shift_for_pack_lane(int lane) {
  const int packed_pos = (lane & 1) ? (4 + (lane >> 1)) : (lane >> 1);
  return packed_pos * 4;
}

// Cooperative in-block PARO single-output incoherence rotation over ``dim``
// channels held in LDS ``buf`` (raw, un-prescaled). Per group of ``group_size``
// channels: pre-scale by ``cscale`` then apply ``krot`` rounds of Givens
// rotations on calibration ``pairs`` with ``theta`` angles. Bit-faithful to
// cpu_reference.paro_rotate1: ``(b[i], b[j]) -> (c*b[i] + s*b[j], -s*b[i] +
// c*b[j])`` with group-local pair indices. ``pairs`` is ``[krot, dim]`` int16,
// ``theta`` is ``[krot, dim/2]``, ``cscale`` is ``[dim]``. Each channel belongs
// to exactly one pair per round (disjoint), so the in-place writes are race-free
// within a round; the inter-round dependency is covered by __syncthreads().
template <typename scalar_t>
__device__ inline void paro_rotate_groups_inplace(
    float* __restrict__ buf,
    const int16_t* __restrict__ pairs,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ cscale,
    int dim,
    int group_size,
    int krot) {
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    buf[i] *= scalar_to_float<scalar_t>(cscale[i]);
  }
  __syncthreads();

  const int half = group_size >> 1;
  const int groups = dim / group_size;
  const int work = groups * half;
  const int half_dim = dim >> 1;
  for (int r = 0; r < krot; ++r) {
    for (int w = threadIdx.x; w < work; w += blockDim.x) {
      const int group = w / half;
      const int lane = w - group * half;
      const int base = group * group_size;
      const int pair_base = r * dim + base + 2 * lane;
      const int i = static_cast<int>(pairs[pair_base + 0]);
      const int j = static_cast<int>(pairs[pair_base + 1]);
      const float angle = scalar_to_float<scalar_t>(theta[r * half_dim + group * half + lane]);
      float s, c;
      sincosf(angle, &s, &c);
      const float xi = buf[base + i];
      const float xj = buf[base + j];
      buf[base + i] = fmaf(xj, s, xi * c);
      buf[base + j] = fmaf(xi, -s, xj * c);
    }
    __syncthreads();
  }
}

// AWQ pack8 transposed dot product: out column ``c`` against staged ``act[0..in)``.
// qweight ``[out/8, in]`` (row = pack, col = k); qzeros ``[groups, out/8]``;
// scales ``[groups, out]``. Dequant matches paro_awq_gemv decode path:
// ``(q - z) * scale`` accumulated in f32.
template <typename scalar_t>
__device__ inline float awq_pack8_col_dot(
    const float* __restrict__ act,
    const int32_t* __restrict__ qweight_row,  // qweight_e + pack * in_features
    const int32_t* __restrict__ qzeros_e,
    const scalar_t* __restrict__ scales_e,
    int pack,
    int shift,
    int col,
    int in_features,
    int out_features,
    int out_packed,
    int group_size) {
  // Hoist per-group qzeros[group] and scales[group,col] (loop-invariant across
  // the 128-wide group) out of the inner k loop, and vectorize the qweight /
  // activation loads by 4. The per-k fmaf(act, (q-z)*sc, acc) stays k-ascending
  // and bit-identical to the scalar path (f32 exact, row-invariant).
  float acc = 0.0f;
  const int groups = in_features / group_size;
  for (int gi = 0; gi < groups; ++gi) {
    const uint32_t pz = static_cast<uint32_t>(qzeros_e[static_cast<int64_t>(gi) * out_packed + pack]);
    const int z = static_cast<int>((pz >> shift) & 0xFu);
    const float sc = scalar_to_float<scalar_t>(scales_e[static_cast<int64_t>(gi) * out_features + col]);
    const int k0 = gi * group_size;
    const int32_t* __restrict__ wptr = qweight_row + k0;
    int kk = 0;
    for (; kk + 4 <= group_size; kk += 4) {
      const int4 w4 = *reinterpret_cast<const int4*>(wptr + kk);
      const float4 a4 = *reinterpret_cast<const float4*>(act + k0 + kk);
      acc = fmaf(a4.x, static_cast<float>(static_cast<int>((static_cast<uint32_t>(w4.x) >> shift) & 0xFu) - z) * sc, acc);
      acc = fmaf(a4.y, static_cast<float>(static_cast<int>((static_cast<uint32_t>(w4.y) >> shift) & 0xFu) - z) * sc, acc);
      acc = fmaf(a4.z, static_cast<float>(static_cast<int>((static_cast<uint32_t>(w4.z) >> shift) & 0xFu) - z) * sc, acc);
      acc = fmaf(a4.w, static_cast<float>(static_cast<int>((static_cast<uint32_t>(w4.w) >> shift) & 0xFu) - z) * sc, acc);
    }
    for (; kk < group_size; ++kk) {
      const uint32_t pw = static_cast<uint32_t>(wptr[kk]);
      acc = fmaf(act[k0 + kk], static_cast<float>(static_cast<int>((pw >> shift) & 0xFu) - z) * sc, acc);
    }
  }
  return acc;
}

// Fused gate+up AWQ pack8 dot for one output column: shares the staged activation
// load across both projections (same x_s, same k order). Per-output per-k fmaf
// order is identical to two separate awq_pack8_col_dot calls (f32 exact,
// row-invariant); only the activation load and group walk are shared.
template <typename scalar_t>
__device__ inline void awq_pack8_dual_col_dot(
    const float* __restrict__ act,
    const int32_t* __restrict__ gate_w_row,
    const int32_t* __restrict__ gate_qz_e,
    const scalar_t* __restrict__ gate_sc_e,
    const int32_t* __restrict__ up_w_row,
    const int32_t* __restrict__ up_qz_e,
    const scalar_t* __restrict__ up_sc_e,
    int pack,
    int shift,
    int col,
    int in_features,
    int out_features,
    int out_packed,
    int group_size,
    float* __restrict__ g_out,
    float* __restrict__ u_out) {
  float gacc = 0.0f;
  float uacc = 0.0f;
  const int groups = in_features / group_size;
  for (int gi = 0; gi < groups; ++gi) {
    const int64_t zoff = static_cast<int64_t>(gi) * out_packed + pack;
    const int64_t soff = static_cast<int64_t>(gi) * out_features + col;
    const int gz = static_cast<int>((static_cast<uint32_t>(gate_qz_e[zoff]) >> shift) & 0xFu);
    const int uz = static_cast<int>((static_cast<uint32_t>(up_qz_e[zoff]) >> shift) & 0xFu);
    const float gsc = scalar_to_float<scalar_t>(gate_sc_e[soff]);
    const float usc = scalar_to_float<scalar_t>(up_sc_e[soff]);
    const int k0 = gi * group_size;
    const int32_t* __restrict__ gw = gate_w_row + k0;
    const int32_t* __restrict__ uw = up_w_row + k0;
    int kk = 0;
    for (; kk + 4 <= group_size; kk += 4) {
      const float4 a4 = *reinterpret_cast<const float4*>(act + k0 + kk);
      const int4 gw4 = *reinterpret_cast<const int4*>(gw + kk);
      const int4 uw4 = *reinterpret_cast<const int4*>(uw + kk);
      gacc = fmaf(a4.x, static_cast<float>(static_cast<int>((static_cast<uint32_t>(gw4.x) >> shift) & 0xFu) - gz) * gsc, gacc);
      gacc = fmaf(a4.y, static_cast<float>(static_cast<int>((static_cast<uint32_t>(gw4.y) >> shift) & 0xFu) - gz) * gsc, gacc);
      gacc = fmaf(a4.z, static_cast<float>(static_cast<int>((static_cast<uint32_t>(gw4.z) >> shift) & 0xFu) - gz) * gsc, gacc);
      gacc = fmaf(a4.w, static_cast<float>(static_cast<int>((static_cast<uint32_t>(gw4.w) >> shift) & 0xFu) - gz) * gsc, gacc);
      uacc = fmaf(a4.x, static_cast<float>(static_cast<int>((static_cast<uint32_t>(uw4.x) >> shift) & 0xFu) - uz) * usc, uacc);
      uacc = fmaf(a4.y, static_cast<float>(static_cast<int>((static_cast<uint32_t>(uw4.y) >> shift) & 0xFu) - uz) * usc, uacc);
      uacc = fmaf(a4.z, static_cast<float>(static_cast<int>((static_cast<uint32_t>(uw4.z) >> shift) & 0xFu) - uz) * usc, uacc);
      uacc = fmaf(a4.w, static_cast<float>(static_cast<int>((static_cast<uint32_t>(uw4.w) >> shift) & 0xFu) - uz) * usc, uacc);
    }
    for (; kk < group_size; ++kk) {
      const float a = act[k0 + kk];
      gacc = fmaf(a, static_cast<float>(static_cast<int>((static_cast<uint32_t>(gw[kk]) >> shift) & 0xFu) - gz) * gsc, gacc);
      uacc = fmaf(a, static_cast<float>(static_cast<int>((static_cast<uint32_t>(uw[kk]) >> shift) & 0xFu) - uz) * usc, uacc);
    }
  }
  *g_out = gacc;
  *u_out = uacc;
}

// AWQ pack8 dot for a whole pack (8 output columns col0..col0+7) against staged
// ``act``: loads each packed weight word ONCE and extracts all 8 nibbles, so the
// per-word weight traffic is 8x lower than thread-owns-column. Per-column k
// order is unchanged (f32 exact, row-invariant). out8[j] is column (pack*8 + j).
template <typename scalar_t>
__device__ inline void awq_pack8_pack_dot(
    const float* __restrict__ act,
    const int32_t* __restrict__ qweight_pack_row,  // qweight_e + pack * in_features
    const int32_t* __restrict__ qzeros_e,
    const scalar_t* __restrict__ scales_e,
    int pack,
    int in_features,
    int out_features,
    int out_packed,
    int group_size,
    float* __restrict__ out8) {
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
  const int col0 = pack << 3;
  const int groups = in_features / group_size;
  for (int gi = 0; gi < groups; ++gi) {
    const uint32_t pz = static_cast<uint32_t>(qzeros_e[static_cast<int64_t>(gi) * out_packed + pack]);
    int z[8];
    float sc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int sh = awq_shift_for_pack_lane(j);
      z[j] = static_cast<int>((pz >> sh) & 0xFu);
      sc[j] = scalar_to_float<scalar_t>(scales_e[static_cast<int64_t>(gi) * out_features + col0 + j]);
    }
    const int k0 = gi * group_size;
    const int32_t* __restrict__ wptr = qweight_pack_row + k0;
    int kk = 0;
    for (; kk + 4 <= group_size; kk += 4) {
      const int4 w4 = *reinterpret_cast<const int4*>(wptr + kk);
      const float4 a4 = *reinterpret_cast<const float4*>(act + k0 + kk);
      const uint32_t ws[4] = {static_cast<uint32_t>(w4.x), static_cast<uint32_t>(w4.y),
                              static_cast<uint32_t>(w4.z), static_cast<uint32_t>(w4.w)};
      const float as[4] = {a4.x, a4.y, a4.z, a4.w};
#pragma unroll
      for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int sh = awq_shift_for_pack_lane(j);
          acc[j] = fmaf(as[t], static_cast<float>(static_cast<int>((ws[t] >> sh) & 0xFu) - z[j]) * sc[j], acc[j]);
        }
      }
    }
    for (; kk < group_size; ++kk) {
      const uint32_t pw = static_cast<uint32_t>(wptr[kk]);
      const float a = act[k0 + kk];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int sh = awq_shift_for_pack_lane(j);
        acc[j] = fmaf(a, static_cast<float>(static_cast<int>((pw >> sh) & 0xFu) - z[j]) * sc[j], acc[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) out8[j] = acc[j];
}

// Dual gate+up AWQ pack8 dot for a whole pack (8 gate + 8 up output columns):
// loads each packed gate/up word ONCE and extracts all 8 nibbles, and shares the
// staged-activation load across both projections. Per-column k order unchanged
// (f32 exact, row-invariant). g8/u8[j] is column (pack*8 + j).
template <typename scalar_t>
__device__ inline void awq_pack8_dual_pack_dot(
    const float* __restrict__ act,
    const int32_t* __restrict__ gate_pack_row,
    const int32_t* __restrict__ gate_qz_e,
    const scalar_t* __restrict__ gate_sc_e,
    const int32_t* __restrict__ up_pack_row,
    const int32_t* __restrict__ up_qz_e,
    const scalar_t* __restrict__ up_sc_e,
    int pack,
    int in_features,
    int out_features,
    int out_packed,
    int group_size,
    float* __restrict__ g8,
    float* __restrict__ u8) {
  float gacc[8];
  float uacc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) { gacc[j] = 0.0f; uacc[j] = 0.0f; }
  const int col0 = pack << 3;
  const int groups = in_features / group_size;
  for (int gi = 0; gi < groups; ++gi) {
    const int64_t zoff = static_cast<int64_t>(gi) * out_packed + pack;
    const int64_t soff = static_cast<int64_t>(gi) * out_features + col0;
    const uint32_t gpz = static_cast<uint32_t>(gate_qz_e[zoff]);
    const uint32_t upz = static_cast<uint32_t>(up_qz_e[zoff]);
    int gz[8];
    int uz[8];
    float gsc[8];
    float usc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int sh = awq_shift_for_pack_lane(j);
      gz[j] = static_cast<int>((gpz >> sh) & 0xFu);
      uz[j] = static_cast<int>((upz >> sh) & 0xFu);
      gsc[j] = scalar_to_float<scalar_t>(gate_sc_e[soff + j]);
      usc[j] = scalar_to_float<scalar_t>(up_sc_e[soff + j]);
    }
    const int k0 = gi * group_size;
    const int32_t* __restrict__ gw = gate_pack_row + k0;
    const int32_t* __restrict__ uw = up_pack_row + k0;
    int kk = 0;
    for (; kk + 4 <= group_size; kk += 4) {
      const float4 a4 = *reinterpret_cast<const float4*>(act + k0 + kk);
      const int4 gw4 = *reinterpret_cast<const int4*>(gw + kk);
      const int4 uw4 = *reinterpret_cast<const int4*>(uw + kk);
      const uint32_t gws[4] = {static_cast<uint32_t>(gw4.x), static_cast<uint32_t>(gw4.y),
                               static_cast<uint32_t>(gw4.z), static_cast<uint32_t>(gw4.w)};
      const uint32_t uws[4] = {static_cast<uint32_t>(uw4.x), static_cast<uint32_t>(uw4.y),
                               static_cast<uint32_t>(uw4.z), static_cast<uint32_t>(uw4.w)};
      const float as[4] = {a4.x, a4.y, a4.z, a4.w};
#pragma unroll
      for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int sh = awq_shift_for_pack_lane(j);
          gacc[j] = fmaf(as[t], static_cast<float>(static_cast<int>((gws[t] >> sh) & 0xFu) - gz[j]) * gsc[j], gacc[j]);
          uacc[j] = fmaf(as[t], static_cast<float>(static_cast<int>((uws[t] >> sh) & 0xFu) - uz[j]) * usc[j], uacc[j]);
        }
      }
    }
    for (; kk < group_size; ++kk) {
      const uint32_t gpw = static_cast<uint32_t>(gw[kk]);
      const uint32_t upw = static_cast<uint32_t>(uw[kk]);
      const float a = act[k0 + kk];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int sh = awq_shift_for_pack_lane(j);
        gacc[j] = fmaf(a, static_cast<float>(static_cast<int>((gpw >> sh) & 0xFu) - gz[j]) * gsc[j], gacc[j]);
        uacc[j] = fmaf(a, static_cast<float>(static_cast<int>((upw >> sh) & 0xFu) - uz[j]) * usc[j], uacc[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) { g8[j] = gacc[j]; u8[j] = uacc[j]; }
}

template <typename scalar_t, typename out_t>
__launch_bounds__(256, 2)
__global__ void paro_selected_ffn_fused_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ gate_qw,
    const int32_t* __restrict__ gate_qz,
    const scalar_t* __restrict__ gate_sc,
    const int32_t* __restrict__ up_qw,
    const int32_t* __restrict__ up_qz,
    const scalar_t* __restrict__ up_sc,
    const int32_t* __restrict__ down_qw,
    const int32_t* __restrict__ down_qz,
    const scalar_t* __restrict__ down_sc,
    const int16_t* __restrict__ r1_pairs,
    const scalar_t* __restrict__ r1_theta,
    const scalar_t* __restrict__ r1_cscale,
    const int16_t* __restrict__ dr_pairs,
    const scalar_t* __restrict__ dr_theta,
    const scalar_t* __restrict__ dr_cscale,
    out_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int hidden,
    int ffn_len,
    int group_size,
    int krot) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  out_t* __restrict__ out_row = out + static_cast<int64_t>(row) * hidden;
  // Inactive / out-of-range expert lanes emit zeros so the downstream combine
  // (routing weight * down) is well defined.
  if (expert_id < 0 || expert_id >= num_experts) {
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
      store_output<out_t>(out_row, c, 0.0f);
    }
    return;
  }

  const int lanes_per_x_row = rows / x_rows;
  const int x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const scalar_t* __restrict__ x_row_ptr = x + static_cast<int64_t>(x_row) * hidden;

  // Dynamic LDS: hidden f32 (staged + rotated x) + ffn_len f32 (silu*mul + down-rotated).
  extern __shared__ float smem[];
  float* x_s = smem;
  float* inter_s = smem + hidden;

  // 1. stage x into LDS, then rotate1 in place over hidden.
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    x_s[i] = scalar_to_float<scalar_t>(x_row_ptr[i]);
  }
  __syncthreads();
  paro_rotate_groups_inplace<scalar_t>(x_s, r1_pairs, r1_theta, r1_cscale, hidden, group_size, krot);

  // 2. dual AWQ gate/up GEMV -> silu*mul -> inter_s (ffn_len wide).
  const int hidden_groups = hidden / group_size;
  const int gate_out_packed = ffn_len >> 3;
  const int64_t gate_w_expert = static_cast<int64_t>(gate_out_packed) * hidden;
  const int64_t gate_z_expert = static_cast<int64_t>(hidden_groups) * gate_out_packed;
  const int64_t gate_s_expert = static_cast<int64_t>(hidden_groups) * ffn_len;
  const int32_t* __restrict__ gate_qw_e = gate_qw + expert_id * gate_w_expert;
  const int32_t* __restrict__ gate_qz_e = gate_qz + expert_id * gate_z_expert;
  const scalar_t* __restrict__ gate_sc_e = gate_sc + expert_id * gate_s_expert;
  const int32_t* __restrict__ up_qw_e = up_qw + expert_id * gate_w_expert;
  const int32_t* __restrict__ up_qz_e = up_qz + expert_id * gate_z_expert;
  const scalar_t* __restrict__ up_sc_e = up_sc + expert_id * gate_s_expert;

  // Thread-owns-pack: each thread owns one pack (8 gate + 8 up cols), loading
  // each packed gate/up word once. ffn_len/8 packs.
  for (int p = threadIdx.x; p < gate_out_packed; p += blockDim.x) {
    float g8[8], u8[8];
    awq_pack8_dual_pack_dot<scalar_t>(
        x_s,
        gate_qw_e + static_cast<int64_t>(p) * hidden, gate_qz_e, gate_sc_e,
        up_qw_e + static_cast<int64_t>(p) * hidden, up_qz_e, up_sc_e,
        p, hidden, ffn_len, gate_out_packed, group_size, g8, u8);
    const int col0 = p << 3;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float gb = round_intermediate<out_t>(g8[j]);
      const float ub = round_intermediate<out_t>(u8[j]);
      const float silu = gb / (1.0f + expf(-gb));
      inter_s[col0 + j] = round_intermediate<out_t>(silu * ub);
    }
  }
  __syncthreads();

  // 3. down-rotate inter_s in place over ffn_len.
  paro_rotate_groups_inplace<scalar_t>(inter_s, dr_pairs, dr_theta, dr_cscale, ffn_len, group_size, krot);

  // 4. AWQ down GEMV -> out_row (hidden wide).
  const int ffn_groups = ffn_len / group_size;
  const int down_out_packed = hidden >> 3;
  const int64_t down_w_expert = static_cast<int64_t>(down_out_packed) * ffn_len;
  const int64_t down_z_expert = static_cast<int64_t>(ffn_groups) * down_out_packed;
  const int64_t down_s_expert = static_cast<int64_t>(ffn_groups) * hidden;
  const int32_t* __restrict__ down_qw_e = down_qw + expert_id * down_w_expert;
  const int32_t* __restrict__ down_qz_e = down_qz + expert_id * down_z_expert;
  const scalar_t* __restrict__ down_sc_e = down_sc + expert_id * down_s_expert;

  // Thread-owns-pack: each thread owns one pack (8 hidden columns), loading each
  // packed down weight word once. hidden=256-packs maps 1:1 to a 256-thread block.
  for (int p = threadIdx.x; p < down_out_packed; p += blockDim.x) {
    float out8[8];
    awq_pack8_pack_dot<scalar_t>(
        inter_s, down_qw_e + static_cast<int64_t>(p) * ffn_len, down_qz_e, down_sc_e,
        p, ffn_len, hidden, down_out_packed, group_size, out8);
    const int col0 = p << 3;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      store_output<out_t>(out_row, col0 + j, out8[j]);
    }
  }
}

template <typename scalar_t, typename out_t>
int launch_paro_selected_ffn_fused_out(
    const scalar_t* x,
    const int64_t* selected,
    const int32_t* gate_qw,
    const int32_t* gate_qz,
    const scalar_t* gate_sc,
    const int32_t* up_qw,
    const int32_t* up_qz,
    const scalar_t* up_sc,
    const int32_t* down_qw,
    const int32_t* down_qz,
    const scalar_t* down_sc,
    const int16_t* r1_pairs,
    const scalar_t* r1_theta,
    const scalar_t* r1_cscale,
    const int16_t* dr_pairs,
    const scalar_t* dr_theta,
    const scalar_t* dr_cscale,
    out_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t hidden,
    int64_t ffn_len,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || gate_qw == nullptr || gate_qz == nullptr ||
      gate_sc == nullptr || up_qw == nullptr || up_qz == nullptr || up_sc == nullptr ||
      down_qw == nullptr || down_qz == nullptr || down_sc == nullptr ||
      r1_pairs == nullptr || r1_theta == nullptr || r1_cscale == nullptr ||
      dr_pairs == nullptr || dr_theta == nullptr || dr_cscale == nullptr || out == nullptr ||
      x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || num_experts <= 0 ||
      hidden <= 0 || ffn_len <= 0 || group_size <= 0 || krot <= 0 ||
      (hidden % group_size) != 0 || (ffn_len % group_size) != 0 ||
      (hidden % 8) != 0 || (ffn_len % 8) != 0 ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(rows));
  const size_t shared_bytes = static_cast<size_t>(hidden + ffn_len) * sizeof(float);
  paro_selected_ffn_fused_kernel<scalar_t, out_t><<<grid, block, shared_bytes, stream>>>(
      x, selected,
      gate_qw, gate_qz, gate_sc,
      up_qw, up_qz, up_sc,
      down_qw, down_qz, down_sc,
      r1_pairs, r1_theta, r1_cscale,
      dr_pairs, dr_theta, dr_cscale,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(hidden),
      static_cast<int>(ffn_len),
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_paro_selected_ffn_fused_bf16_bf16_out(
    const void* x, const void* selected,
    const void* gate_qw, const void* gate_qz, const void* gate_sc,
    const void* up_qw, const void* up_qz, const void* up_sc,
    const void* down_qw, const void* down_qz, const void* down_sc,
    const void* r1_pairs, const void* r1_theta, const void* r1_cscale,
    const void* dr_pairs, const void* dr_theta, const void* dr_cscale,
    void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t hidden, int64_t ffn_len, int64_t group_size, int64_t krot,
    int64_t threads, void* stream) {
  return launch_paro_selected_ffn_fused_out<uint16_t, uint16_t>(
      static_cast<const uint16_t*>(x),
      static_cast<const int64_t*>(selected),
      static_cast<const int32_t*>(gate_qw),
      static_cast<const int32_t*>(gate_qz),
      static_cast<const uint16_t*>(gate_sc),
      static_cast<const int32_t*>(up_qw),
      static_cast<const int32_t*>(up_qz),
      static_cast<const uint16_t*>(up_sc),
      static_cast<const int32_t*>(down_qw),
      static_cast<const int32_t*>(down_qz),
      static_cast<const uint16_t*>(down_sc),
      static_cast<const int16_t*>(r1_pairs),
      static_cast<const uint16_t*>(r1_theta),
      static_cast<const uint16_t*>(r1_cscale),
      static_cast<const int16_t*>(dr_pairs),
      static_cast<const uint16_t*>(dr_theta),
      static_cast<const uint16_t*>(dr_cscale),
      static_cast<uint16_t*>(out),
      x_rows, rows, num_experts, hidden, ffn_len, group_size, krot, threads,
      static_cast<hipStream_t>(stream));
}

extern "C" int hipengine_paro_selected_ffn_fused_fp16_fp16_out(
    const void* x, const void* selected,
    const void* gate_qw, const void* gate_qz, const void* gate_sc,
    const void* up_qw, const void* up_qz, const void* up_sc,
    const void* down_qw, const void* down_qz, const void* down_sc,
    const void* r1_pairs, const void* r1_theta, const void* r1_cscale,
    const void* dr_pairs, const void* dr_theta, const void* dr_cscale,
    void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t hidden, int64_t ffn_len, int64_t group_size, int64_t krot,
    int64_t threads, void* stream) {
  return launch_paro_selected_ffn_fused_out<_Float16, _Float16>(
      static_cast<const _Float16*>(x),
      static_cast<const int64_t*>(selected),
      static_cast<const int32_t*>(gate_qw),
      static_cast<const int32_t*>(gate_qz),
      static_cast<const _Float16*>(gate_sc),
      static_cast<const int32_t*>(up_qw),
      static_cast<const int32_t*>(up_qz),
      static_cast<const _Float16*>(up_sc),
      static_cast<const int32_t*>(down_qw),
      static_cast<const int32_t*>(down_qz),
      static_cast<const _Float16*>(down_sc),
      static_cast<const int16_t*>(r1_pairs),
      static_cast<const _Float16*>(r1_theta),
      static_cast<const _Float16*>(r1_cscale),
      static_cast<const int16_t*>(dr_pairs),
      static_cast<const _Float16*>(dr_theta),
      static_cast<const _Float16*>(dr_cscale),
      static_cast<_Float16*>(out),
      x_rows, rows, num_experts, hidden, ffn_len, group_size, krot, threads,
      static_cast<hipStream_t>(stream));
}

extern "C" int hipengine_paro_selected_ffn_fused_f32_f32_out(
    const void* x, const void* selected,
    const void* gate_qw, const void* gate_qz, const void* gate_sc,
    const void* up_qw, const void* up_qz, const void* up_sc,
    const void* down_qw, const void* down_qz, const void* down_sc,
    const void* r1_pairs, const void* r1_theta, const void* r1_cscale,
    const void* dr_pairs, const void* dr_theta, const void* dr_cscale,
    void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t hidden, int64_t ffn_len, int64_t group_size, int64_t krot,
    int64_t threads, void* stream) {
  return launch_paro_selected_ffn_fused_out<float, float>(
      static_cast<const float*>(x),
      static_cast<const int64_t*>(selected),
      static_cast<const int32_t*>(gate_qw),
      static_cast<const int32_t*>(gate_qz),
      static_cast<const float*>(gate_sc),
      static_cast<const int32_t*>(up_qw),
      static_cast<const int32_t*>(up_qz),
      static_cast<const float*>(up_sc),
      static_cast<const int32_t*>(down_qw),
      static_cast<const int32_t*>(down_qz),
      static_cast<const float*>(down_sc),
      static_cast<const int16_t*>(r1_pairs),
      static_cast<const float*>(r1_theta),
      static_cast<const float*>(r1_cscale),
      static_cast<const int16_t*>(dr_pairs),
      static_cast<const float*>(dr_theta),
      static_cast<const float*>(dr_cscale),
      static_cast<float*>(out),
      x_rows, rows, num_experts, hidden, ffn_len, group_size, krot, threads,
      static_cast<hipStream_t>(stream));
}
