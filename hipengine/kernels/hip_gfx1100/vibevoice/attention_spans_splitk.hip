// Split-K span attention for a decode step.
//
// The strict span kernel (``vv_prefill_attention_kernel``) launches one 256-thread
// block per query head and reads the KV rows with a per-token stride, so its warps
// issue uncoalesced 2-byte loads and the grid is as small as the head count. At a
// product-depth decode context that leaves most of the device idle and costs about
// 0.32 ms per layer-branch.
//
// This kernel keeps the same span ABI and the same per-chunk softmax semantics, but
// reorganizes the work:
//
//   * one block per (KV head, KV split), so the grid is num_kv_heads * splits
//     instead of num_q_heads, and every query head in a KV group is folded into
//     that block, which reads each KV row once for all of them (YuE2 has two query
//     heads per KV head, so this halves KV traffic; the split count is what fills
//     the device);
//   * a warp covers head_dim between its lanes, so each KV row is one coalesced
//     run and the score reduction is a 32-lane shuffle instead of a serial dot;
//   * the split's scores are staged in shared memory, so the value pass reads them
//     from shared instead of recomputing.
//
// Arithmetic differs from the strict kernel only in the reduction order (per-split
// maxima and sums combined afterwards instead of one whole-context pass), which is
// the standard split-softmax reassociation. It is therefore a production variant,
// not a strict one: see docs/EXECUTION-PROFILES.md.

#define VV_SPLITK_MAX_CHUNK 256
#define VV_SPLITK_THREADS 256

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

// Grid: (num_kv_heads, num_splits, rows). Block: VV_SPLITK_THREADS.
template <int64_t kQPerKv, int64_t kHeadDim>
__global__ void vv_attention_spans_splitk_kernel(
    const float* __restrict__ query,          // [rows, q_heads, kHeadDim]
    const uint16_t* __restrict__ key_cache,   // [capacity, kv_heads, kHeadDim]
    const uint16_t* __restrict__ value_cache, // [capacity, kv_heads, kHeadDim]
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const bool* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    float* __restrict__ partial_out, // [rows, q_heads, num_splits, kHeadDim]
    float* __restrict__ partial_m,   // [rows, q_heads, num_splits]
    float* __restrict__ partial_l,   // [rows, q_heads, num_splits]
    int64_t rows,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t max_ctx,
    int64_t num_splits,
    float scale) {
  const int64_t kv_head = blockIdx.x;
  const int64_t split_idx = blockIdx.y;
  const int64_t row = blockIdx.z;
  if (kv_head >= num_kv_heads || split_idx >= num_splits || row >= rows) {
    return;
  }
  const int64_t q_base = kv_head * kQPerKv;
  const int64_t lane_id = threadIdx.x % warpSize;
  const int64_t warp_id = threadIdx.x / warpSize;

  // Same context derivation as the strict kernel: the row's own position, the live
  // span count, and the kernel's context bound.
  const int64_t requested = row_positions[row] + 1;
  const int64_t ctx = live_counts ? min(min(requested, live_counts[row]), max_ctx) : requested;
  const int64_t chunk = ctx > 0 ? (ctx + num_splits - 1) / num_splits : 0;
  const int64_t token_start = split_idx * chunk;
  const int64_t token_end = min(token_start + chunk, ctx);

  extern __shared__ float shared[];
  float* scores = shared;                                  // [kQPerKv, VV_SPLITK_MAX_CHUNK]
  float* partial = scores + kQPerKv * VV_SPLITK_MAX_CHUNK; // [kQPerKv, warps]
  float* q_shared = partial + kQPerKv * (VV_SPLITK_THREADS / 32); // [kQPerKv, kHeadDim]

  const int64_t num_warps = blockDim.x / warpSize;

  // A split past the context contributes nothing: publish the identity so the
  // combine pass can skip it without special-casing.
  if (token_start >= token_end) {
    if (threadIdx.x < kQPerKv) {
      const int64_t idx = (row * num_q_heads + q_base + threadIdx.x) * num_splits + split_idx;
      partial_m[idx] = -3.4028234663852886e38f;
      partial_l[idx] = 0.0f;
    }
    for (int64_t h = 0; h < kQPerKv; ++h) {
      float* out_row = partial_out
          + ((row * num_q_heads + q_base + h) * num_splits + split_idx) * kHeadDim;
      for (int64_t d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        out_row[d] = 0.0f;
      }
    }
    return;
  }

  for (int64_t h = 0; h < kQPerKv; ++h) {
    const float* q_row = query + (row * num_q_heads + q_base + h) * kHeadDim;
    for (int64_t d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
      q_shared[h * kHeadDim + d] = q_row[d];
    }
  }
  __syncthreads();

  const int64_t chunk_len = token_end - token_start;
  const int64_t position_limit = row_positions[row];

  // Pass 1: scores. Warp w owns tokens w, w + num_warps, ...; its 32 lanes cover
  // head_dim, so each KV row is a single coalesced read shared by every head in the
  // group.
  float local_max[kQPerKv];
#pragma unroll
  for (int64_t h = 0; h < kQPerKv; ++h) {
    local_max[h] = -3.4028234663852886e38f;
  }
  for (int64_t lt = warp_id; lt < chunk_len; lt += num_warps) {
    const int64_t token = token_start + lt;
    const int64_t slot = base_offsets ? base_offsets[token] : token;
    const bool masked = base_offsets
        && (slot < 0 || slot >= max_ctx || evict_mask[slot]
            || token_positions[slot] < 0 || token_positions[slot] > position_limit);
    if (masked) {
#pragma unroll
      for (int64_t h = 0; h < kQPerKv; ++h) {
        scores[h * VV_SPLITK_MAX_CHUNK + lt] = -INFINITY;
      }
      continue;
    }
    const int64_t k_offset = (slot * num_kv_heads + kv_head) * kHeadDim;
    float acc[kQPerKv];
#pragma unroll
    for (int64_t h = 0; h < kQPerKv; ++h) {
      acc[h] = 0.0f;
    }
    for (int64_t d = lane_id * 4; d < kHeadDim; d += warpSize * 4) {
      const float k0 = bf16_bits_to_float(key_cache[k_offset + d + 0]);
      const float k1 = bf16_bits_to_float(key_cache[k_offset + d + 1]);
      const float k2 = bf16_bits_to_float(key_cache[k_offset + d + 2]);
      const float k3 = bf16_bits_to_float(key_cache[k_offset + d + 3]);
#pragma unroll
      for (int64_t h = 0; h < kQPerKv; ++h) {
        const float* q = q_shared + h * kHeadDim + d;
        acc[h] += q[0] * k0;
        acc[h] += q[1] * k1;
        acc[h] += q[2] * k2;
        acc[h] += q[3] * k3;
      }
    }
#pragma unroll
    for (int64_t h = 0; h < kQPerKv; ++h) {
      float reduced = acc[h];
#pragma unroll
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        reduced += __shfl_xor(reduced, offset);
      }
      const float score = reduced * scale;
      scores[h * VV_SPLITK_MAX_CHUNK + lt] = score;
      local_max[h] = fmaxf(local_max[h], score);
    }
  }

  // Chunk maximum per head. Every lane of a warp walked the same token set and so
  // holds the same value: this is a per-warp quantity, not a per-lane one, and
  // reducing it across lanes would multiply the sum below by the warp size.
  if (lane_id == 0) {
#pragma unroll
    for (int64_t h = 0; h < kQPerKv; ++h) {
      partial[h * num_warps + warp_id] = local_max[h];
    }
  }
  __syncthreads();
  float chunk_max[kQPerKv];
#pragma unroll
  for (int64_t h = 0; h < kQPerKv; ++h) {
    float value = -3.4028234663852886e38f;
    for (int64_t w = 0; w < num_warps; ++w) {
      value = fmaxf(value, partial[h * num_warps + w]);
    }
    chunk_max[h] = value;
  }
  // `partial` is reused for the sums below, so every thread must finish reading the
  // maxima before any of them overwrites the slot it just read.
  __syncthreads();

  // Pass 2: exponentials in place, and the chunk sum per head.
  float local_sum[kQPerKv];
#pragma unroll
  for (int64_t h = 0; h < kQPerKv; ++h) {
    local_sum[h] = 0.0f;
  }
  for (int64_t lt = warp_id; lt < chunk_len; lt += num_warps) {
#pragma unroll
    for (int64_t h = 0; h < kQPerKv; ++h) {
      const float e = expf(scores[h * VV_SPLITK_MAX_CHUNK + lt] - chunk_max[h]);
      scores[h * VV_SPLITK_MAX_CHUNK + lt] = e;
      local_sum[h] += e;
    }
  }
  if (lane_id == 0) {
#pragma unroll
    for (int64_t h = 0; h < kQPerKv; ++h) {
      partial[h * num_warps + warp_id] = local_sum[h];
    }
  }
  __syncthreads();
  float chunk_sum[kQPerKv];
#pragma unroll
  for (int64_t h = 0; h < kQPerKv; ++h) {
    float value = 0.0f;
    for (int64_t w = 0; w < num_warps; ++w) {
      value += partial[h * num_warps + w];
    }
    chunk_sum[h] = value;
  }

  // Pass 3: weighted values. One thread per output dim, and the value row is loaded
  // once per token and applied to every head in the group, which is what makes the
  // eight head accumulators independent instead of one dependent chain per head.
  if (threadIdx.x < kHeadDim) {
    const int64_t dim = threadIdx.x;
    float acc[kQPerKv];
#pragma unroll
    for (int64_t h = 0; h < kQPerKv; ++h) {
      acc[h] = 0.0f;
    }
    for (int64_t lt = 0; lt < chunk_len; ++lt) {
      const int64_t token = token_start + lt;
      const int64_t slot = base_offsets ? base_offsets[token] : token;
      const int64_t v_offset = (slot * num_kv_heads + kv_head) * kHeadDim + dim;
      const float v = bf16_bits_to_float(value_cache[v_offset]);
#pragma unroll
      for (int64_t h = 0; h < kQPerKv; ++h) {
        acc[h] = fmaf(scores[h * VV_SPLITK_MAX_CHUNK + lt], v, acc[h]);
      }
    }
#pragma unroll
    for (int64_t h = 0; h < kQPerKv; ++h) {
      partial_out[((row * num_q_heads + q_base + h) * num_splits + split_idx) * kHeadDim + dim] =
          acc[h];
    }
  }
  if (threadIdx.x < kQPerKv) {
    const int64_t idx = (row * num_q_heads + q_base + threadIdx.x) * num_splits + split_idx;
    partial_m[idx] = chunk_max[threadIdx.x];
    partial_l[idx] = chunk_sum[threadIdx.x];
  }
}

// Combines the splits: out = sum_s acc_s * e^(m_s - M) / sum_s l_s * e^(m_s - M).
// Grid: (num_q_heads, rows). Block: 128 threads (one per output dim at head_dim 128).
__global__ void vv_attention_spans_splitk_combine_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    float* __restrict__ out,
    int64_t rows,
    int64_t num_q_heads,
    int64_t head_dim,
    int64_t num_splits) {
  const int64_t q_head = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (q_head >= num_q_heads || row >= rows) {
    return;
  }
  const int64_t base = (row * num_q_heads + q_head) * num_splits;
  float running_max = -3.4028234663852886e38f;
  for (int64_t s = 0; s < num_splits; ++s) {
    running_max = fmaxf(running_max, partial_m[base + s]);
  }
  // The partial rows are unnormalized sums (sum_t e^(s_t - m) * v_t), so the
  // numerator carries only the exponential rescale; the split's own sum belongs in
  // the denominator alone. Weighting the numerator by l as well would multiply the
  // result by the split's normalizer.
  float denominator = 0.0f;
  for (int64_t s = 0; s < num_splits; ++s) {
    denominator += partial_l[base + s] * expf(partial_m[base + s] - running_max);
  }
  const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
  float* out_row = out + (row * num_q_heads + q_head) * head_dim;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t s = 0; s < num_splits; ++s) {
      acc = fmaf(expf(partial_m[base + s] - running_max),
                 partial_out[(base + s) * head_dim + d], acc);
    }
    out_row[d] = acc * inv;
  }
}

} // namespace

extern "C" int hipengine_vv_attention_spans_splitk(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    float* workspace,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const bool* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t max_ctx,
    int64_t num_splits,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0
      || num_splits <= 0 || num_splits > 64 || workspace == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q_per_kv = q_heads / kv_heads;
  if (q_heads % kv_heads != 0 || head_dim != 128
      || (q_per_kv != 1 && q_per_kv != 2 && q_per_kv != 4 && q_per_kv != 8)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t partial_count = rows * q_heads * num_splits;
  float* partial_out = workspace;
  float* partial_m = partial_out + partial_count * head_dim;
  float* partial_l = partial_m + partial_count;

  // One instantiation per query-heads-per-KV-head ratio, so the head fold is a
  // compile-time trip count instead of a loop bound.
#define HIPENGINE_LAUNCH_SPLITK(Q_PER_KV)                                                     \
  do {                                                                                        \
    const size_t shared_bytes =                                                               \
        static_cast<size_t>(q_per_kv * VV_SPLITK_MAX_CHUNK + q_per_kv * (VV_SPLITK_THREADS / 32) \
                            + q_per_kv * head_dim) * sizeof(float);                           \
    hipLaunchKernelGGL(                                                                       \
        HIP_KERNEL_NAME(vv_attention_spans_splitk_kernel<Q_PER_KV, 128>),                      \
        dim3(static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(num_splits),       \
             static_cast<unsigned int>(rows)),                                                 \
        dim3(VV_SPLITK_THREADS),                                                              \
        shared_bytes,                                                                         \
        stream,                                                                               \
        query,                                                                                \
        key_cache,                                                                            \
        value_cache,                                                                          \
        base_offsets,                                                                         \
        live_counts,                                                                          \
        token_positions,                                                                      \
        evict_mask,                                                                           \
        row_positions,                                                                        \
        partial_out,                                                                          \
        partial_m,                                                                            \
        partial_l,                                                                            \
        rows,                                                                                 \
        q_heads,                                                                              \
        kv_heads,                                                                             \
        max_ctx,                                                                              \
        num_splits,                                                                           \
        scale);                                                                               \
  } while (0)
  if (q_per_kv == 8) {
    HIPENGINE_LAUNCH_SPLITK(8);
  } else if (q_per_kv == 4) {
    HIPENGINE_LAUNCH_SPLITK(4);
  } else if (q_per_kv == 2) {
    HIPENGINE_LAUNCH_SPLITK(2);
  } else {
    HIPENGINE_LAUNCH_SPLITK(1);
  }
#undef HIPENGINE_LAUNCH_SPLITK
  int status = static_cast<int>(hipGetLastError());
  if (status != 0) {
    return status;
  }
  const int64_t combine_threads = head_dim < 128 ? head_dim : 128;
  hipLaunchKernelGGL(
      vv_attention_spans_splitk_combine_kernel,
      dim3(static_cast<unsigned int>(q_heads), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(combine_threads)),
      0,
      stream,
      partial_out,
      partial_m,
      partial_l,
      out,
      rows,
      q_heads,
      head_dim,
      num_splits);
  return static_cast<int>(hipGetLastError());
}

// The workspace this entry point needs, so callers can size their allocation.
extern "C" int64_t hipengine_vv_attention_spans_splitk_workspace_floats(
    int64_t rows, int64_t q_heads, int64_t head_dim, int64_t num_splits) {
  return rows * q_heads * num_splits * (head_dim + 2);
}
