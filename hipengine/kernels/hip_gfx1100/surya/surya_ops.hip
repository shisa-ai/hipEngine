// Surya OCR 2 fp32 text-stack helper kernels (gfx1100/gfx1151 JIT).
//
// Everything else in the Surya GPU path reuses the family kernels already
// shipped for EVIE (elementwise, rope, GDN conv/recurrence/gates) and
// rocBLAS SGEMM; this unit only adds the two pieces with no parameterized
// parent:
//   1. per-head q|gate deinterleave for the fused attention q_proj output
//      (Surya fuses q and the sigmoid gate per head: (nq, 2*head_dim), the
//      EVIE parent hardcodes 16 heads / 256 dim);
//   2. causal score masking + scaling for prompt prefill attention (the
//      EVIE parent is bidirectional; softmax itself reuses
//      hipengine_evie_softmax_rows_f32).

#include <hip/hip_runtime.h>
#include <math.h>

__device__ inline float surya_softplus_f32(float x) {
  return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

// Strided-source q/K l2-normalization with PLAIN output rows.
//   src row: (src_stride,) floats; q plane at [+0, key_dim), k plane at
//   [+k_offset, +k_offset + key_dim). Output (tokens, heads, head_dim)
//   contiguous per head — the layout the fp32 prefill recurrence kernels
//   read. (The EVIE repeat variant writes a doubled per-head layout whose
//   read convention does not match this family's recurrence.)
__global__ void surya_gdn_l2norm_kernel(
    const float* __restrict__ src,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float q_scale,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    int64_t src_stride,
    int64_t k_offset) {
  const int64_t rows = tokens * heads;
  const int64_t idx = static_cast<int64_t>(blockIdx.x);
  if (idx >= rows) {
    return;
  }
  const int64_t token = idx / heads;
  const int64_t head = idx % heads;
  const float* qr = src + token * src_stride + head * head_dim;
  const float* kr = src + token * src_stride + k_offset + head * head_dim;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  float sq = 0.0f;
  float sk = 0.0f;
  for (int64_t i = tid; i < head_dim; i += nthr) {
    sq += qr[i] * qr[i];
    sk += kr[i] * kr[i];
  }
  for (int off = 16; off > 0; off >>= 1) {
    sq += __shfl_xor(sq, off);
    sk += __shfl_xor(sk, off);
  }
  const float iq = rsqrtf(sq + 1e-6f) * q_scale;
  const float ik = rsqrtf(sk + 1e-6f);
  float* qo = q_out + idx * head_dim;
  float* ko = k_out + idx * head_dim;
  for (int64_t i = tid; i < head_dim; i += nthr) {
    qo[i] = qr[i] * iq;
    ko[i] = kr[i] * ik;
  }
}

extern "C" int hipengine_surya_gdn_l2norm_f32(
    const float* src,
    float* q_out,
    float* k_out,
    float q_scale,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    int64_t src_stride,
    int64_t k_offset,
    hipStream_t stream) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0 || src_stride <= 0 ||
      k_offset < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t rows = tokens * heads;
  hipLaunchKernelGGL(
      surya_gdn_l2norm_kernel,
      dim3(static_cast<unsigned int>(rows)), dim3(32), 0, stream,
      src, q_out, k_out, q_scale, tokens, heads, head_dim, src_stride,
      k_offset);
  return static_cast<int>(hipGetLastError());
}

// Plain-weight RMSNorm (Qwen convention: out = x * rsqrt(mean(x^2)+eps) * w).
// The EVIE parent hardcodes (1+w), which does not match Surya's weights.
__global__ void surya_rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int64_t rows,
    int64_t hidden,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* xr = x + row * hidden;
  float total = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    total += xr[i] * xr[i];
  }
  __shared__ float partial[256];
  partial[threadIdx.x] = total;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  __shared__ float s_inv;
  if (threadIdx.x == 0) {
    s_inv = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
  }
  __syncthreads();
  const float inv = s_inv;
  float* orow = out + row * hidden;
  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    orow[i] = xr[i] * inv * w[i];
  }
}

extern "C" int hipengine_surya_rmsnorm_f32(
    const float* x,
    const float* w,
    float* out,
    int64_t rows,
    int64_t hidden,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      surya_rmsnorm_kernel, dim3(static_cast<unsigned int>(rows)), dim3(256),
      0, stream, x, w, out, rows, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

// Per-head deinterleave of the fused q_proj output.
//   src (tokens, nq * 2 * head_dim): head h occupies [h*2*head_dim,
//   (h+1)*2*head_dim) with q at [+0, head_dim) and gate at [+head_dim,
//   +2*head_dim). q_dst (tokens, nq, head_dim); gate_dst (tokens, nq *
//   head_dim).
__global__ void surya_split_qgate_kernel(
    const float* __restrict__ src,
    float* __restrict__ q_dst,
    float* __restrict__ gate_dst,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = tokens * heads * head_dim;
  if (idx >= n) {
    return;
  }
  const int64_t d = idx % head_dim;
  const int64_t head = (idx / head_dim) % heads;
  const int64_t token = idx / (head_dim * heads);
  const int64_t src_base = token * heads * 2 * head_dim + head * 2 * head_dim;
  q_dst[idx] = src[src_base + d];
  gate_dst[idx] = src[src_base + head_dim + d];
}

extern "C" int hipengine_surya_split_qgate_f32(
    const float* src,
    float* q_dst,
    float* gate_dst,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    hipStream_t stream) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t n = tokens * heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      surya_split_qgate_kernel, dim3(blocks), dim3(threads), 0, stream,
      src, q_dst, gate_dst, tokens, heads, head_dim);
  return static_cast<int>(hipGetLastError());
}

// Scatter per-token K/V rows into persistent cache planes.
//   src (tokens, nk * head_dim) row-major (the k/v GEMM output);
//   dst (nk, max_seq, head_dim) per-head planes; token t lands at
//   dst[h, token_offset + t, :].
__global__ void surya_scatter_kv_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t tokens,
    int64_t token_offset,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t max_seq) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = tokens * kv_heads * head_dim;
  if (idx >= n) {
    return;
  }
  const int64_t d = idx % head_dim;
  const int64_t head = (idx / head_dim) % kv_heads;
  const int64_t token = idx / (head_dim * kv_heads);
  dst[head * max_seq * head_dim + (token_offset + token) * head_dim + d] =
      src[token * kv_heads * head_dim + head * head_dim + d];
}

extern "C" int hipengine_surya_scatter_kv_f32(
    const float* src,
    float* dst,
    int64_t tokens,
    int64_t token_offset,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t max_seq,
    hipStream_t stream) {
  if (tokens <= 0 || kv_heads <= 0 || head_dim <= 0 || max_seq <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (token_offset < 0 || token_offset + tokens > max_seq) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t n = tokens * kv_heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      surya_scatter_kv_kernel, dim3(blocks), dim3(threads), 0, stream,
      src, dst, tokens, token_offset, kv_heads, head_dim, max_seq);
  return static_cast<int>(hipGetLastError());
}

// Causal mask + 1/sqrt(d_k) scale over batched score rows.
//   scores layout: (heads, queries, head_stride) row-major; entry
//   (h, i, j) with j > query_offset + i becomes -inf, everything else is
//   scaled. The dense prefill is the case queries == keys == head_stride and
//   query_offset == 0; a query-row tile passes its first query as
//   query_offset, because the causal condition is relative to the tile's
//   first query, not to the tile's own row index.
__global__ void surya_causal_mask_scale_kernel(
    float* __restrict__ scores,
    float scale,
    int64_t heads,
    int64_t queries,
    int64_t head_stride,
    int64_t query_offset) {
  const int64_t row = blockIdx.x;
  if (row >= heads * queries) {
    return;
  }
  const int64_t head = row / queries;
  const int64_t query = row % queries;
  float* sr = scores + head * queries * head_stride + query * head_stride;
  const int64_t causal = query_offset + query;
  for (int64_t j = threadIdx.x; j < head_stride; j += blockDim.x) {
    sr[j] = (j <= causal) ? sr[j] * scale : -INFINITY;
  }
}

extern "C" int hipengine_surya_causal_mask_scale_f32(
    float* scores,
    float scale,
    int64_t heads,
    int64_t queries,
    int64_t head_stride,
    int64_t query_offset,
    hipStream_t stream) {
  if (heads <= 0 || queries <= 0 || head_stride <= 0 || query_offset < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      surya_causal_mask_scale_kernel,
      dim3(static_cast<unsigned int>(heads * queries)), dim3(256), 0, stream,
      scores, scale, heads, queries, head_stride, query_offset);
  return static_cast<int>(hipGetLastError());
}

// ---------------------------------------------------------------------------
// KVLiveSpans ABI: spans-aware dense scatter + fused fp32 decode attention.
//
// The pre-spans Surya path wrote into a bespoke dense ``(nk, max_seq, hd)``
// plane from a host-supplied ``(tokens, token_offset)`` pair and decoded
// through three rocBLAS batched SGEMMs plus two elementwise kernels over a
// materialized ``(nq, max_seq)`` score row.  Both kernels below read the same
// ``(base_offsets, live_counts, token_positions, evict_mask)`` span metadata the
// rest of the tree uses, so a dense uniform fill is a policy choice rather than
// a separate ABI.  The dense fp32 cache layout (head-major, ``head_dim``
// contiguous) is unchanged, so the prefill path keeps addressing the same
// planes the scatter writes.
//
// The decode kernel is a GQA-fused split-K producer: one block per
// ``(kv_head, context chunk)`` computes all ``q_per_kv`` query heads of that KV
// head, so the K/V planes are read once per KV head instead of once per query
// head (the parent batched GEMM re-reads each plane ``q_per_kv`` times).  Each
// split publishes ``(partial_out, partial_m, partial_l)`` and
// ``hipengine_surya_full_attn_decode_split_k_reduce_f32`` combines them.
// Split-K keeps the grid wide (``num_kv_heads * num_splits`` blocks) without
// materializing a ``capacity``-sized LDS score row: at Surya's 16384-token
// default that row would need 66560 B, above the gfx11 64 KiB workgroup limit.
// ---------------------------------------------------------------------------

// Physical slot for a logical token index under a spans page table.
__device__ __forceinline__ int64_t surya_spans_physical_slot(
    const int32_t* __restrict__ base_offsets,
    int64_t logical,
    int64_t block_size) {
  const int64_t logical_block = logical / block_size;
  return static_cast<int64_t>(base_offsets[logical_block]) * block_size +
         (logical - logical_block * block_size);
}

__device__ __forceinline__ bool surya_spans_token_visible(
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    int64_t logical,
    int64_t query_position) {
  if (token_positions != nullptr) {
    const int64_t position = token_positions[logical];
    if (position < 0 || position > query_position) {
      return false;
    }
  }
  if (evict_mask != nullptr && evict_mask[logical] != 0) {
    return false;
  }
  return true;
}

// Scatter per-token K/V rows into persistent cache planes using KVLiveSpans.
//   src (tokens, nk * head_dim) row-major (the k/v GEMM output);
//   dst (nk, max_seq, head_dim) per-head planes;
//   logical token lt = token_offset + i lands at physical slot
//   base_offsets[lt / block_size] * block_size + (lt % block_size), and is
//   skipped when lt >= live_counts[0], token_positions[lt] < 0, or
//   evict_mask[lt] is set.
__global__ void surya_scatter_kv_spans_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    int64_t tokens,
    int64_t token_offset,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t max_seq,
    int64_t block_size,
    int64_t block_table_len) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = tokens * kv_heads * head_dim;
  if (idx >= n) {
    return;
  }
  const int64_t d = idx % head_dim;
  const int64_t head = (idx / head_dim) % kv_heads;
  const int64_t token = idx / (head_dim * kv_heads);
  const int64_t logical = token_offset + token;
  const int64_t live = (live_counts != nullptr) ? live_counts[0] : max_seq;
  if (logical < 0 || logical >= live) {
    return;
  }
  if (!surya_spans_token_visible(token_positions, evict_mask, logical, max_seq - 1)) {
    return;
  }
  const int64_t logical_block = logical / block_size;
  if (logical_block < 0 || logical_block >= block_table_len) {
    return;
  }
  const int64_t physical =
      surya_spans_physical_slot(base_offsets, logical, block_size);
  if (physical < 0 || physical >= max_seq) {
    return;
  }
  dst[head * max_seq * head_dim + physical * head_dim + d] =
      src[token * kv_heads * head_dim + head * head_dim + d];
}

extern "C" int hipengine_surya_scatter_kv_f32_spans(
    const float* src,
    float* dst,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    int64_t tokens,
    int64_t token_offset,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t max_seq,
    int64_t block_size,
    int64_t block_table_len,
    hipStream_t stream) {
  if (tokens <= 0 || kv_heads <= 0 || head_dim <= 0 || max_seq <= 0 ||
      block_size <= 0 || block_table_len <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (token_offset < 0 || token_offset + tokens > max_seq) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if ((max_seq + block_size - 1) / block_size > block_table_len) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t n = tokens * kv_heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      surya_scatter_kv_spans_kernel, dim3(blocks), dim3(threads), 0, stream,
      src, dst, base_offsets, live_counts, token_positions, evict_mask,
      tokens, token_offset, kv_heads, head_dim, max_seq, block_size,
      block_table_len);
  return static_cast<int>(hipGetLastError());
}

// GQA-fused split-K decode producer.  Grid (num_kv_heads, num_splits);
// 256 threads; one LDS ``chunk_size`` score row per query head of the KV group.
__global__ void surya_full_attn_decode_split_k_spans_kernel(
    const float* __restrict__ query,         // [num_q_heads, 256]
    const float* __restrict__ key_cache,     // [num_kv_heads, capacity, 256]
    const float* __restrict__ value_cache,
    float* __restrict__ partial_out,         // [num_q_heads, num_splits, 256]
    float* __restrict__ partial_m,           // [num_q_heads, num_splits]
    float* __restrict__ partial_l,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    int64_t capacity,
    int64_t chunk_size,
    int64_t num_splits,
    int64_t block_size,
    int64_t block_table_len,
    float scale) {
  constexpr int64_t head_dim = 256;
  constexpr int64_t q_per_kv = 4;
  const int64_t kv_head = blockIdx.x;
  const int64_t split_idx = blockIdx.y;

  const int64_t context_len =
      live_counts[0] < capacity ? live_counts[0] : capacity;
  const int64_t query_position = row_positions[0];
  const int64_t token_start = split_idx * chunk_size;
  const int64_t q_base = kv_head * q_per_kv;

  if (token_start >= context_len) {
    if (threadIdx.x == 0) {
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const int64_t partial_idx = (q_base + h) * num_splits + split_idx;
        partial_m[partial_idx] = -3.4028234663852886e38f;
        partial_l[partial_idx] = 0.0f;
      }
    }
    if (static_cast<int64_t>(threadIdx.x) < head_dim) {
      const int64_t dim = threadIdx.x;
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const int64_t partial_idx = (q_base + h) * num_splits + split_idx;
        partial_out[partial_idx * head_dim + dim] = 0.0f;
      }
    }
    return;
  }

  const int64_t token_end_raw = token_start + chunk_size;
  const int64_t token_end =
      (token_end_raw < context_len) ? token_end_raw : context_len;
  const int64_t chunk_len = token_end - token_start;

  extern __shared__ float shared[];
  float* scores = shared;                           // [q_per_kv, chunk_size]
  const int num_warps = blockDim.x / warpSize;
  float* partial = scores + q_per_kv * chunk_size;  // [q_per_kv, num_warps]
  float* q_shared = partial + q_per_kv * num_warps; // [q_per_kv, head_dim]
  int32_t* token_offsets = reinterpret_cast<int32_t*>(
      q_shared + q_per_kv * head_dim);               // [chunk_size]

  for (int64_t h = 0; h < q_per_kv; ++h) {
    q_shared[h * head_dim + threadIdx.x] =
        query[(q_base + h) * head_dim + threadIdx.x];
  }
  __syncthreads();

  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  float local_max[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    local_max[h] = -3.4028234663852886e38f;
  }

  for (int64_t lt = warp_id; lt < chunk_len; lt += num_warps) {
    const int64_t logical = token_start + lt;
    const bool visible = surya_spans_token_visible(
        token_positions, evict_mask, logical, query_position);
    float acc[q_per_kv];
    for (int64_t h = 0; h < q_per_kv; ++h) {
      acc[h] = 0.0f;
    }
    int64_t physical = -1;
    if (visible) {
      physical = surya_spans_physical_slot(base_offsets, logical, block_size);
      const int64_t k_offset = kv_head * capacity * head_dim +
                               physical * head_dim;
      for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
        const float k0 = key_cache[k_offset + dim + 0];
        const float k1 = key_cache[k_offset + dim + 1];
        const float k2 = key_cache[k_offset + dim + 2];
        const float k3 = key_cache[k_offset + dim + 3];
        for (int64_t h = 0; h < q_per_kv; ++h) {
          const float* q = q_shared + h * head_dim + dim;
          acc[h] += q[0] * k0;
          acc[h] += q[1] * k1;
          acc[h] += q[2] * k2;
          acc[h] += q[3] * k3;
        }
      }
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      for (int64_t h = 0; h < q_per_kv; ++h) {
        acc[h] += __shfl_down(acc[h], offset);
      }
    }
    if (lane_id == 0) {
      token_offsets[lt] = static_cast<int32_t>(physical);
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const float score =
            visible ? acc[h] * scale : -3.4028234663852886e38f;
        scores[h * chunk_size + lt] = score;
        local_max[h] = fmaxf(local_max[h], score);
      }
    }
  }
  if (lane_id == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      partial[h * num_warps + warp_id] = local_max[h];
    }
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      float warp_max = (threadIdx.x < num_warps)
                           ? partial[h * num_warps + threadIdx.x]
                           : -3.4028234663852886e38f;
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_down(warp_max, offset));
      }
      if (threadIdx.x == 0) {
        partial[h * num_warps] = warp_max;
      }
    }
  }
  __syncthreads();

  float chunk_max[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    chunk_max[h] = partial[h * num_warps];
    // A chunk whose every token is invisible (empty slot or evicted) has no
    // maximum; clamp the sentinel so expf(-inf - sentinel) stays 0 instead of
    // turning into a NaN weight.
    if (!(chunk_max[h] > -3.0e38f)) {
      chunk_max[h] = 0.0f;
    }
  }
  __syncthreads();

  float local_sum[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    local_sum[h] = 0.0f;
    for (int64_t lt = threadIdx.x; lt < chunk_len; lt += blockDim.x) {
      const float e = expf(scores[h * chunk_size + lt] - chunk_max[h]);
      scores[h * chunk_size + lt] = e;
      local_sum[h] += e;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      local_sum[h] += __shfl_down(local_sum[h], offset);
    }
  }
  if (lane_id == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      partial[h * num_warps + warp_id] = local_sum[h];
    }
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      float warp_sum = (threadIdx.x < num_warps)
                           ? partial[h * num_warps + threadIdx.x]
                           : 0.0f;
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down(warp_sum, offset);
      }
      if (threadIdx.x == 0) {
        partial[h * num_warps] = warp_sum;
      }
    }
  }
  __syncthreads();

  float chunk_sum[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    chunk_sum[h] = partial[h * num_warps];
  }
  __syncthreads();

  if (static_cast<int64_t>(threadIdx.x) < head_dim) {
    const int64_t dim = threadIdx.x;
    float acc[q_per_kv];
    for (int64_t h = 0; h < q_per_kv; ++h) {
      acc[h] = 0.0f;
    }
    for (int64_t lt = 0; lt < chunk_len; ++lt) {
      const int32_t physical = token_offsets[lt];
      if (physical < 0) {
        continue;
      }
      const float v =
          value_cache[kv_head * capacity * head_dim +
                      static_cast<int64_t>(physical) * head_dim + dim];
      for (int64_t h = 0; h < q_per_kv; ++h) {
        acc[h] += scores[h * chunk_size + lt] * v;
      }
    }
    for (int64_t h = 0; h < q_per_kv; ++h) {
      const int64_t partial_idx = (q_base + h) * num_splits + split_idx;
      partial_out[partial_idx * head_dim + dim] = acc[h];
    }
  }

  if (threadIdx.x == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      const int64_t partial_idx = (q_base + h) * num_splits + split_idx;
      partial_m[partial_idx] = chunk_max[h];
      partial_l[partial_idx] = chunk_sum[h];
    }
  }
}

extern "C" int hipengine_surya_full_attn_decode_split_k_f32_spans(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    float* partial_out,
    float* partial_m,
    float* partial_l,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t chunk_size,
    int64_t num_splits,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  constexpr int64_t q_per_kv = 4;
  if (capacity <= 0 || chunk_size <= 0 || num_splits <= 0 || block_size <= 0 ||
      block_table_len <= 0 || head_dim != 256 || num_kv_heads <= 0 ||
      num_q_heads != q_per_kv * num_kv_heads) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if ((capacity + block_size - 1) / block_size > block_table_len) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int num_warps = threads / 32;
  const size_t shared_bytes =
      static_cast<size_t>(q_per_kv * chunk_size + q_per_kv * num_warps +
                          q_per_kv * head_dim) *
          sizeof(float) +
      static_cast<size_t>(chunk_size) * sizeof(int32_t);
  hipLaunchKernelGGL(
      surya_full_attn_decode_split_k_spans_kernel,
      dim3(static_cast<unsigned int>(num_kv_heads),
           static_cast<unsigned int>(num_splits)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      partial_out,
      partial_m,
      partial_l,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      chunk_size,
      num_splits,
      block_size,
      block_table_len,
      scale);
  return static_cast<int>(hipGetLastError());
}

// Combine split-K partials: o = sum_s o_s * exp(m_s - m_global) / l_global.
__global__ void surya_full_attn_decode_split_k_reduce_kernel(
    const float* __restrict__ partial_out,  // [num_q_heads, num_splits, head_dim]
    const float* __restrict__ partial_m,    // [num_q_heads, num_splits]
    const float* __restrict__ partial_l,
    float* __restrict__ out,                // [num_q_heads, head_dim]
    int64_t num_splits,
    int64_t head_dim) {
  const int64_t q_head = blockIdx.x;
  extern __shared__ float sred[];
  float* sm = sred;
  float* sl = sred + num_splits;
  float* sw = sl + num_splits;

  for (int64_t i = threadIdx.x; i < num_splits; i += blockDim.x) {
    sm[i] = partial_m[q_head * num_splits + i];
    sl[i] = partial_l[q_head * num_splits + i];
  }
  __syncthreads();

  if (static_cast<int64_t>(threadIdx.x) >= head_dim) {
    return;
  }
  const int64_t dim = threadIdx.x;

  if (threadIdx.x == 0) {
    float m_global = -3.4028234663852886e38f;
    for (int64_t s = 0; s < num_splits; ++s) {
      m_global = fmaxf(m_global, sm[s]);
    }
    float l_global = 0.0f;
    for (int64_t s = 0; s < num_splits; ++s) {
      const float li = sl[s];
      const float weight = (li > 0.0f) ? expf(sm[s] - m_global) : 0.0f;
      sw[s] = weight;
      l_global += li * weight;
    }
    const float inv_l = 1.0f / fmaxf(l_global, 1.0e-20f);
    for (int64_t s = 0; s < num_splits; ++s) {
      sw[s] *= inv_l;
    }
  }
  __syncthreads();

  float acc = 0.0f;
  for (int64_t s = 0; s < num_splits; ++s) {
    acc += partial_out[(q_head * num_splits + s) * head_dim + dim] * sw[s];
  }
  out[q_head * head_dim + dim] = acc;
}

extern "C" int hipengine_surya_full_attn_decode_split_k_reduce_f32(
    const float* partial_out,
    const float* partial_m,
    const float* partial_l,
    float* out,
    int64_t num_q_heads,
    int64_t num_splits,
    int64_t head_dim,
    hipStream_t stream) {
  if (num_q_heads <= 0 || num_splits <= 0 || head_dim <= 0 || head_dim > 1024) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int reduce_threads = static_cast<int>(head_dim);
  const size_t shared_bytes = static_cast<size_t>(num_splits) * 3 * sizeof(float);
  hipLaunchKernelGGL(
      surya_full_attn_decode_split_k_reduce_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(reduce_threads),
      shared_bytes,
      stream,
      partial_out,
      partial_m,
      partial_l,
      out,
      num_splits,
      head_dim);
  return static_cast<int>(hipGetLastError());
}
