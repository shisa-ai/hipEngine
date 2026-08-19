// DFlash2 drafter kernels (grouped dynamic conv, top-16, candidate selector).
//
// Strict exact/parent-parity RED contract vs kernels/cpu_reference/dflash2.py
// (grouped_dynamic_convolve, candidate_selector_select, dflash2_topk).  The
// kernels read BF16 weights/hidden and accumulate FP32; the RED tests run the
// CPU oracle on the BF16-rounded inputs and assert a small FP32 tolerance.
//
// Kernel bodies take raw device pointers only (no torch::Tensor).

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

// ---------------------------------------------------------------------------
// Grouped dynamic causal conv (two taps, zero-padded at the sequence start).
//   out[t,c] = (base[0,c] + dyn[t,0,g]) * h[t,c]
//            + (t>0 ? (base[1,c] + dyn[t,1,g]) * h[t-1,c] : 0)
// where g = c / group_size.  ``dyn`` is a strided view of the dynamic
// coefficient block: row t begins at ``dyn[t*dyn_stride + dyn_offset]`` with
// ``2*groups`` usable elements (tap-major: tap o, group g at offset o*groups+g).
// ``base`` is (2, hidden) BF16 (a view into the (2, 2, hidden) base_kernel at
// the input or output side).
// ---------------------------------------------------------------------------
__global__ void dflash2_grouped_conv_kernel(
    const uint16_t* __restrict__ hidden,
    const uint16_t* __restrict__ dyn,
    const uint16_t* __restrict__ base,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t hidden_size,
    int32_t group_size,
    int32_t groups,
    int64_t dyn_offset,
    int64_t dyn_stride) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * hidden_size;
  if (idx >= total) {
    return;
  }
  const int64_t t = idx / hidden_size;
  const int64_t c = idx % hidden_size;
  const int32_t g = static_cast<int32_t>(c / group_size);
  const float b0 = bf16_bits_to_float(base[0 * hidden_size + c]);
  const float b1 = bf16_bits_to_float(base[1 * hidden_size + c]);
  const int64_t drow = t * dyn_stride + dyn_offset;
  const float d0 = bf16_bits_to_float(dyn[drow + 0LL * groups + g]);
  const float d1 = bf16_bits_to_float(dyn[drow + 1LL * groups + g]);
  const float h0 = bf16_bits_to_float(hidden[t * hidden_size + c]);
  float acc = (b0 + d0) * h0;
  if (t > 0) {
    const float h1 = bf16_bits_to_float(hidden[(t - 1) * hidden_size + c]);
    acc += (b1 + d1) * h1;
  }
  out[idx] = float_to_bf16_bits(acc);
}

// ---------------------------------------------------------------------------
// Top-16 over (rows, vocab) FP32 logits, descending, one block per row.
// ---------------------------------------------------------------------------
#define DFLASH2_TOPK 16

__device__ inline void dflash2_insert_topk_pair(
    float value, int32_t index, float* values, int32_t* indices, int32_t top_k) {
  for (int pos = 0; pos < top_k; ++pos) {
    if (value > values[pos]) {
      for (int shift = top_k - 1; shift > pos; --shift) {
        values[shift] = values[shift - 1];
        indices[shift] = indices[shift - 1];
      }
      values[pos] = value;
      indices[pos] = index;
      break;
    }
  }
}

__global__ void dflash2_top16_rows_kernel(
    const float* __restrict__ logits,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_values,
    int64_t rows,
    int64_t vocab_size,
    int32_t top_k) {
  extern __shared__ unsigned char shared_raw[];
  float* shared_values = reinterpret_cast<float*>(shared_raw);
  int32_t* shared_indices = reinterpret_cast<int32_t*>(shared_values + static_cast<int64_t>(blockDim.x) * top_k);
  const int tid = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  float local_values[DFLASH2_TOPK];
  int32_t local_indices[DFLASH2_TOPK];
  for (int pos = 0; pos < top_k; ++pos) {
    local_values[pos] = -3.4028234663852886e+38F;
    local_indices[pos] = INT32_MAX;
  }
  const int64_t base = row * vocab_size;
  for (int64_t i = tid; i < vocab_size; i += blockDim.x) {
    dflash2_insert_topk_pair(logits[base + i], static_cast<int32_t>(i), local_values, local_indices, top_k);
  }
  for (int pos = 0; pos < top_k; ++pos) {
    const int64_t off = static_cast<int64_t>(tid) * top_k + pos;
    shared_values[off] = local_values[pos];
    shared_indices[off] = local_indices[pos];
  }
  __syncthreads();
  if (tid == 0) {
    float final_values[DFLASH2_TOPK];
    int32_t final_indices[DFLASH2_TOPK];
    for (int pos = 0; pos < top_k; ++pos) {
      final_values[pos] = -3.4028234663852886e+38F;
      final_indices[pos] = INT32_MAX;
    }
    for (int thread = 0; thread < blockDim.x; ++thread) {
      for (int pos = 0; pos < top_k; ++pos) {
        const int64_t off = static_cast<int64_t>(thread) * top_k + pos;
        dflash2_insert_topk_pair(shared_values[off], shared_indices[off], final_values, final_indices, top_k);
      }
    }
    for (int pos = 0; pos < top_k; ++pos) {
      out_ids[row * top_k + pos] = final_indices[pos];
      out_values[row * top_k + pos] = final_values[pos];
    }
  }
}

// ---------------------------------------------------------------------------
// Candidate selector: h = hidden @ hidden_projection.T, then a greedy walk
// over K candidates per position using the low-rank bilinear scores
//   sc[k] = unary[t,k] + sum_r codebook_a[pred, r] * h[t, r] * codebook_b[cand, r]
// The walk is sequential over positions (position t's predecessor is the
// chosen candidate of position t-1), so one thread performs it.
// ---------------------------------------------------------------------------
__global__ void dflash2_selector_proj_kernel(
    const uint16_t* __restrict__ hidden,
    const uint16_t* __restrict__ hidden_projection,
    float* __restrict__ h_out,
    int64_t rows,
    int64_t hidden_size,
    int32_t rank) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int idx = threadIdx.x;  // 0..rank-1
  if (row >= rows || idx >= rank) {
    return;
  }
  float acc = 0.0f;
  const uint16_t* hptr = hidden + row * hidden_size;
  const uint16_t* wptr = hidden_projection + idx * hidden_size;
  for (int64_t k = 0; k < hidden_size; ++k) {
    acc += bf16_bits_to_float(hptr[k]) * bf16_bits_to_float(wptr[k]);
  }
  h_out[row * rank + idx] = acc;
}

__global__ void dflash2_selector_walk_kernel(
    const float* __restrict__ h,
    const int32_t* __restrict__ cand_ids,
    const float* __restrict__ cand_vals,
    const int32_t* __restrict__ anchor_ids,
    const uint16_t* __restrict__ codebook_a,
    const uint16_t* __restrict__ codebook_b,
    int32_t* __restrict__ path,
    float* __restrict__ scores,
    int64_t rows,
    int32_t rank,
    int32_t top_k,
    int64_t vocab) {
  if (threadIdx.x != 0) {
    return;
  }
  int32_t predecessor = anchor_ids[0];
  for (int64_t t = 0; t < rows; ++t) {
    const float* hrow = h + t * rank;
    const int32_t* ids = cand_ids + t * top_k;
    const float* vals = cand_vals + t * top_k;
    // Gather the predecessor codebook row into registers.
    const uint16_t* arow = codebook_a + static_cast<int64_t>(predecessor) * rank;
    float best = -3.4028234663852886e+38F;
    int32_t best_index = 0;
    int32_t best_id = -1;
    for (int k = 0; k < top_k; ++k) {
      const int64_t id = static_cast<int64_t>(ids[k]);
      const uint16_t* brow = codebook_b + id * rank;
      float acc = vals[k];
      for (int r = 0; r < rank; ++r) {
        acc += bf16_bits_to_float(arow[r]) * hrow[r] * bf16_bits_to_float(brow[r]);
      }
      scores[t * top_k + k] = acc;
      if (acc > best) {
        best = acc;
        best_index = k;
        best_id = ids[k];
      }
    }
    path[t] = best_id;
    predecessor = best_id;
    (void)best_index;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host-callable wrappers
// ---------------------------------------------------------------------------

extern "C" int hipengine_dflash2_grouped_conv(
    const uint16_t* hidden,
    const uint16_t* dyn,
    const uint16_t* base,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    int32_t group_size,
    int32_t groups,
    int64_t dyn_offset,
    int64_t dyn_stride,
    int64_t threads,
    hipStream_t stream) {
  const int64_t total = rows * hidden_size;
  const int64_t block = (threads > 0) ? threads : 256;
  const int64_t grid = (total + block - 1) / block;
  hipLaunchKernelGGL(dflash2_grouped_conv_kernel,
                     dim3(static_cast<unsigned int>(grid)),
                     dim3(static_cast<unsigned int>(block)),
                     0, stream,
                     hidden, dyn, base, out, rows, hidden_size, group_size, groups,
                     dyn_offset, dyn_stride);
  return 0;
}

extern "C" int hipengine_dflash2_top16_rows(
    const float* logits,
    int32_t* out_ids,
    float* out_values,
    int64_t rows,
    int64_t vocab_size,
    int32_t top_k,
    int64_t threads,
    hipStream_t stream) {
  const int64_t block = (threads > 0) ? threads : 256;
  const size_t shared = static_cast<size_t>(block) * static_cast<size_t>(top_k) *
                        (sizeof(float) + sizeof(int32_t));
  hipLaunchKernelGGL(dflash2_top16_rows_kernel,
                     dim3(static_cast<unsigned int>(rows)),
                     dim3(static_cast<unsigned int>(block)),
                     shared, stream,
                     logits, out_ids, out_values, rows, vocab_size, top_k);
  return 0;
}

extern "C" int hipengine_dflash2_selector(
    const uint16_t* hidden,
    const uint16_t* hidden_projection,
    const int32_t* cand_ids,
    const float* cand_vals,
    const int32_t* anchor_ids,
    const uint16_t* codebook_a,
    const uint16_t* codebook_b,
    float* h_scratch,
    int32_t* path,
    float* scores,
    int64_t rows,
    int64_t hidden_size,
    int32_t rank,
    int32_t top_k,
    int64_t vocab,
    int64_t threads,
    hipStream_t stream) {
  hipLaunchKernelGGL(dflash2_selector_proj_kernel,
                     dim3(static_cast<unsigned int>(rows)),
                     dim3(static_cast<unsigned int>(rank)),
                     0, stream,
                     hidden, hidden_projection, h_scratch, rows, hidden_size, rank);
  hipLaunchKernelGGL(dflash2_selector_walk_kernel,
                     dim3(1), dim3(1), 0, stream,
                     h_scratch, cand_ids, cand_vals, anchor_ids, codebook_a, codebook_b,
                     path, scores, rows, rank, top_k, vocab);
  return 0;
}
