// Net-new Maple kernels derived from the public DeepGrove format/math contract.
// Format reference: deepgrove-ai/mlx-lm-deepgrove/mlx_lm/ternary.py (LSB-first).

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float bf16_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ inline float reduce_sum(float value, float* scratch) {
  const int tid = static_cast<int>(threadIdx.x);
  scratch[tid] = value;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ inline float ternary_dot(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    int64_t in_features) {
  const int64_t words = in_features / 16;
  float sum = 0.0f;
  for (int64_t word_idx = threadIdx.x; word_idx < words; word_idx += blockDim.x) {
    const uint32_t word = weight[word_idx];
    const int64_t base = word_idx * 16;
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
      const int code = static_cast<int>((word >> (2 * lane)) & 0x3U) - 1;
      sum += bf16_to_float(x[base + lane]) * static_cast<float>(code);
    }
  }
  return sum;
}

__global__ void maple_ternary_gemv_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ row_alpha,
    uint16_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  const int64_t row = blockIdx.x;
  if (row >= out_features) {
    return;
  }
  extern __shared__ float scratch[];
  const int64_t words = in_features / 16;
  const float partial = ternary_dot(x, weight + row * words, in_features);
  const float total = reduce_sum(partial, scratch);
  if (threadIdx.x == 0) {
    out[row] = float_to_bf16(total * bf16_to_float(row_alpha[row]));
  }
}

__global__ void maple_ternary_qkv_gemv_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ q_weight,
    const uint16_t* __restrict__ q_alpha,
    const uint32_t* __restrict__ k_weight,
    const uint16_t* __restrict__ k_alpha,
    const uint32_t* __restrict__ v_weight,
    const uint16_t* __restrict__ v_alpha,
    uint16_t* __restrict__ out,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features) {
  const int64_t row = blockIdx.x;
  const int64_t total_rows = q_features + 2 * kv_features;
  if (row >= total_rows) {
    return;
  }
  const int64_t words = in_features / 16;
  const uint32_t* selected_weight;
  const uint16_t* selected_alpha;
  int64_t selected_row;
  if (row < q_features) {
    selected_weight = q_weight;
    selected_alpha = q_alpha;
    selected_row = row;
  } else if (row < q_features + kv_features) {
    selected_weight = k_weight;
    selected_alpha = k_alpha;
    selected_row = row - q_features;
  } else {
    selected_weight = v_weight;
    selected_alpha = v_alpha;
    selected_row = row - q_features - kv_features;
  }
  extern __shared__ float scratch[];
  const float partial = ternary_dot(
      x,
      selected_weight + selected_row * words,
      in_features);
  const float total = reduce_sum(partial, scratch);
  if (threadIdx.x == 0) {
    out[row] = float_to_bf16(total * bf16_to_float(selected_alpha[selected_row]));
  }
}

__global__ void maple_selected_ternary_dual_gemv_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ a_weight,
    const uint16_t* __restrict__ a_alpha,
    const uint32_t* __restrict__ b_weight,
    const uint16_t* __restrict__ b_alpha,
    const int32_t* __restrict__ selected_experts,
    uint16_t* __restrict__ a_out,
    uint16_t* __restrict__ b_out,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features) {
  const int64_t block = blockIdx.x;
  const int64_t route = block / out_features;
  const int64_t row = block - route * out_features;
  if (route >= top_k) {
    return;
  }
  const int32_t expert = selected_experts[route];
  if (expert < 0 || expert >= num_experts) {
    if (threadIdx.x == 0) {
      a_out[route * out_features + row] = float_to_bf16(0.0f);
      b_out[route * out_features + row] = float_to_bf16(0.0f);
    }
    return;
  }
  const int64_t words = in_features / 16;
  const int64_t matrix_row = static_cast<int64_t>(expert) * out_features + row;
  float a_partial = 0.0f;
  float b_partial = 0.0f;
  const uint32_t* a_row = a_weight + matrix_row * words;
  const uint32_t* b_row = b_weight + matrix_row * words;
  for (int64_t word_idx = threadIdx.x; word_idx < words; word_idx += blockDim.x) {
    const uint32_t a_word = a_row[word_idx];
    const uint32_t b_word = b_row[word_idx];
    const int64_t base = word_idx * 16;
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
      const float xv = bf16_to_float(x[base + lane]);
      const int a_code = static_cast<int>((a_word >> (2 * lane)) & 0x3U) - 1;
      const int b_code = static_cast<int>((b_word >> (2 * lane)) & 0x3U) - 1;
      a_partial += xv * static_cast<float>(a_code);
      b_partial += xv * static_cast<float>(b_code);
    }
  }
  extern __shared__ float scratch[];
  float* a_scratch = scratch;
  float* b_scratch = scratch + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  a_scratch[tid] = a_partial;
  b_scratch[tid] = b_partial;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      a_scratch[tid] += a_scratch[tid + stride];
      b_scratch[tid] += b_scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const int64_t out_idx = route * out_features + row;
    a_out[out_idx] = float_to_bf16(
        a_scratch[0] * bf16_to_float(a_alpha[matrix_row]));
    b_out[out_idx] = float_to_bf16(
        b_scratch[0] * bf16_to_float(b_alpha[matrix_row]));
  }
}

__global__ void maple_selected_ternary_gemv_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ row_alpha,
    const int32_t* __restrict__ selected_experts,
    uint16_t* __restrict__ out,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features) {
  const int64_t block = blockIdx.x;
  const int64_t route = block / out_features;
  const int64_t row = block - route * out_features;
  if (route >= top_k) {
    return;
  }
  const int32_t expert = selected_experts[route];
  if (expert < 0 || expert >= num_experts) {
    if (threadIdx.x == 0) {
      out[route * out_features + row] = float_to_bf16(0.0f);
    }
    return;
  }
  const int64_t words = in_features / 16;
  const int64_t matrix_row = static_cast<int64_t>(expert) * out_features + row;
  const float partial = ternary_dot(
      x + route * in_features,
      weight + matrix_row * words,
      in_features);
  extern __shared__ float scratch[];
  const float total = reduce_sum(partial, scratch);
  if (threadIdx.x == 0) {
    out[route * out_features + row] = float_to_bf16(
        total * bf16_to_float(row_alpha[matrix_row]));
  }
}

__global__ void maple_affine4_embed_kernel(
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ biases,
    uint16_t* __restrict__ out,
    int64_t token_id,
    int64_t hidden_size) {
  const int64_t words_per_row = hidden_size / 8;
  const int64_t groups_per_row = hidden_size / 64;
  const uint32_t* row = weight + token_id * words_per_row;
  const uint16_t* row_scales = scales + token_id * groups_per_row;
  const uint16_t* row_biases = biases + token_id * groups_per_row;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const uint32_t word = row[idx / 8];
    const int code = static_cast<int>((word >> (4 * (idx & 7))) & 0xFU);
    const int64_t group = idx / 64;
    const float value = static_cast<float>(code) * bf16_to_float(row_scales[group])
        + bf16_to_float(row_biases[group]);
    out[idx] = float_to_bf16(value);
  }
}

__global__ void maple_affine4_gemv_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ biases,
    float* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  const int64_t row_idx = blockIdx.x;
  if (row_idx >= out_features) {
    return;
  }
  const int64_t words = in_features / 8;
  const int64_t groups = in_features / 64;
  const uint32_t* row = weight + row_idx * words;
  const uint16_t* row_scales = scales + row_idx * groups;
  const uint16_t* row_biases = biases + row_idx * groups;
  float partial = 0.0f;
  for (int64_t word_idx = threadIdx.x; word_idx < words; word_idx += blockDim.x) {
    const uint32_t word = row[word_idx];
    const int64_t base = word_idx * 8;
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      const int64_t idx = base + lane;
      const int code = static_cast<int>((word >> (4 * lane)) & 0xFU);
      const int64_t group = idx / 64;
      const float w = static_cast<float>(code) * bf16_to_float(row_scales[group])
          + bf16_to_float(row_biases[group]);
      partial += bf16_to_float(x[idx]) * w;
    }
  }
  extern __shared__ float scratch[];
  const float total = reduce_sum(partial, scratch);
  if (threadIdx.x == 0) {
    out[row_idx] = total;
  }
}

inline int launch_threads_for_words(int64_t words) {
  if (words >= 128) {
    return 128;
  }
  if (words >= 64) {
    return 64;
  }
  return 32;
}

}  // namespace

extern "C" int hipengine_maple_ternary_gemv_bf16(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* row_alpha,
    uint16_t* out,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (in_features <= 0 || out_features <= 0 || in_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  hipLaunchKernelGGL(
      maple_ternary_gemv_kernel,
      dim3(static_cast<unsigned int>(out_features)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      x,
      weight,
      row_alpha,
      out,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

// Batched ternary GEMM (prefill P1): computes
//   out[t, r] = sum_k x[t, k] * ternary(W[r, k]) * alpha[r]
// for t in [0, rows), r in [0, out_features).  One block owns one output row
// and a tile of TOKENS_PER_BLOCK prompt rows, loads that weight row into LDS
// ONCE, and reuses it across the token tile — the core prefill weight-reuse
// win (the c1 GEMV re-reads the full weight row per token).  Accumulation
// order per (token,row) is word-strided threads + lane loop + tree reduce,
// identical to maple_ternary_gemv_kernel, so bf16 outputs match within the
// gate.
template <int TOKENS_PER_BLOCK>
__global__ void maple_ternary_gemm_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ row_alpha,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t r = blockIdx.x;
  const int64_t tile = blockIdx.y;
  const int64_t t0 = tile * TOKENS_PER_BLOCK;
  if (r >= out_features) {
    return;
  }
  const int64_t words = in_features / 16;
  const int tid = static_cast<int>(threadIdx.x);
  extern __shared__ float xs[];
  // Load this output row's weights into LDS once (packed ternary codes).
  __shared__ uint32_t wrow[128];
  const uint32_t* wbase = weight + r * words;
  for (int64_t w = tid; w < words; w += blockDim.x) {
    wrow[w] = wbase[w];
  }
  __syncthreads();  // publish wrow before the token loop reads it
  // Scratch for tree reductions: [blockDim.x] floats per token slice.
  float* scratch = xs;
  const float alpha = bf16_to_float(row_alpha[r]);
  for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
    const int64_t tok = t0 + t;
    if (tok >= rows) {
      break;
    }
    const uint16_t* xrow = x + tok * in_features;
    float partial = 0.0f;
    for (int64_t w = tid; w < words; w += blockDim.x) {
      const uint32_t word = wrow[w];
      const int64_t base = w * 16;
#pragma unroll
      for (int lane = 0; lane < 16; ++lane) {
        const int code = static_cast<int>((word >> (2 * lane)) & 0x3U) - 1;
        partial += bf16_to_float(xrow[base + lane]) * static_cast<float>(code);
      }
    }
    float* slice = scratch + t * blockDim.x;
    slice[tid] = partial;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        slice[tid] += slice[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      out[tok * out_features + r] = float_to_bf16(slice[0] * alpha);
    }
    __syncthreads();  // protect scratch reuse across tokens in the tile
  }
}

extern "C" int hipengine_maple_ternary_gemm_bf16(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* row_alpha,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0 || in_features % 16 != 0
      || in_features / 16 > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  const int64_t words = in_features / 16;
  const int toks_per_block = 8;
  const int64_t blocks_x = out_features;
  const int64_t blocks_y = (rows + toks_per_block - 1) / toks_per_block;
  const int64_t shared = toks_per_block * threads * sizeof(float);
  hipLaunchKernelGGL(
      (maple_ternary_gemm_bf16_kernel<8>),
      dim3(static_cast<unsigned int>(blocks_x),
           static_cast<unsigned int>(blocks_y)),
      dim3(threads),
      static_cast<unsigned int>(shared),
      stream,
      x,
      weight,
      row_alpha,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

// Batched QKV ternary GEMM (P1 prefill): computes the full [rows, q_features +
// 2*kv_features] qkv buffer from the q/k/v ternary projections of a [rows, in]
// input. Grid is (total_output_rows x token-tiles); each block owns one output
// row and an 8-token tile, loading that output row's packed weight codes into
// LDS once and reusing them across the tile (the prefill weight-reuse win).
// Output column layout matches the single-row qkv GEMV: [q | k | v].
template <int TOKENS_PER_BLOCK>
__global__ void maple_ternary_qkv_gemm_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ q_weight,
    const uint16_t* __restrict__ q_alpha,
    const uint32_t* __restrict__ k_weight,
    const uint16_t* __restrict__ k_alpha,
    const uint32_t* __restrict__ v_weight,
    const uint16_t* __restrict__ v_alpha,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features) {
  const int64_t total_rows = q_features + 2 * kv_features;
  const int64_t r = blockIdx.x;
  const int64_t tile = blockIdx.y;
  const int64_t t0 = tile * TOKENS_PER_BLOCK;
  if (r >= total_rows) {
    return;
  }
  const uint32_t* selected_weight;
  const uint16_t* selected_alpha;
  int64_t matrix_row;
  int64_t col;
  if (r < q_features) {
    selected_weight = q_weight;
    selected_alpha = q_alpha;
    matrix_row = r;
    col = r;
  } else if (r < q_features + kv_features) {
    selected_weight = k_weight;
    selected_alpha = k_alpha;
    matrix_row = r - q_features;
    col = q_features + matrix_row;
  } else {
    selected_weight = v_weight;
    selected_alpha = v_alpha;
    matrix_row = r - q_features - kv_features;
    col = q_features + kv_features + matrix_row;
  }
  const int64_t words = in_features / 16;
  const int tid = static_cast<int>(threadIdx.x);
  extern __shared__ float xs[];
  __shared__ uint32_t wrow[128];
  const uint32_t* wbase = selected_weight + matrix_row * words;
  for (int64_t w = tid; w < words; w += blockDim.x) {
    wrow[w] = wbase[w];
  }
  __syncthreads();
  float* scratch = xs;
  const float alpha = bf16_to_float(selected_alpha[matrix_row]);
  for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
    const int64_t tok = t0 + t;
    if (tok >= rows) {
      break;
    }
    const uint16_t* xrow = x + tok * in_features;
    float partial = 0.0f;
    for (int64_t w = tid; w < words; w += blockDim.x) {
      const uint32_t word = wrow[w];
      const int64_t base = w * 16;
#pragma unroll
      for (int lane = 0; lane < 16; ++lane) {
        const int code = static_cast<int>((word >> (2 * lane)) & 0x3U) - 1;
        partial += bf16_to_float(xrow[base + lane]) * static_cast<float>(code);
      }
    }
    float* slice = scratch + t * blockDim.x;
    slice[tid] = partial;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        slice[tid] += slice[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      out[tok * total_rows + col] = float_to_bf16(slice[0] * alpha);
    }
    __syncthreads();
  }
}

extern "C" int hipengine_maple_ternary_qkv_gemm_bf16(
    const uint16_t* x,
    const uint32_t* q_weight,
    const uint16_t* q_alpha,
    const uint32_t* k_weight,
    const uint16_t* k_alpha,
    const uint32_t* v_weight,
    const uint16_t* v_alpha,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || q_features <= 0 || kv_features <= 0
      || in_features % 16 != 0 || in_features / 16 > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  const int toks_per_block = 8;
  const int64_t total_rows = q_features + 2 * kv_features;
  const int64_t blocks_x = total_rows;
  const int64_t blocks_y = (rows + toks_per_block - 1) / toks_per_block;
  const int64_t shared = toks_per_block * threads * sizeof(float);
  hipLaunchKernelGGL(
      (maple_ternary_qkv_gemm_kernel<8>),
      dim3(static_cast<unsigned int>(blocks_x),
           static_cast<unsigned int>(blocks_y)),
      dim3(threads),
      static_cast<unsigned int>(shared),
      stream,
      x,
      q_weight,
      q_alpha,
      k_weight,
      k_alpha,
      v_weight,
      v_alpha,
      out,
      rows,
      in_features,
      q_features,
      kv_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_ternary_qkv_gemv_bf16(
    const uint16_t* x,
    const uint32_t* q_weight,
    const uint16_t* q_alpha,
    const uint32_t* k_weight,
    const uint16_t* k_alpha,
    const uint32_t* v_weight,
    const uint16_t* v_alpha,
    uint16_t* out,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features,
    hipStream_t stream) {
  if (in_features <= 0 || q_features <= 0 || kv_features <= 0 || in_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  const int64_t blocks = q_features + 2 * kv_features;
  hipLaunchKernelGGL(
      maple_ternary_qkv_gemv_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      x,
      q_weight,
      q_alpha,
      k_weight,
      k_alpha,
      v_weight,
      v_alpha,
      out,
      in_features,
      q_features,
      kv_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_selected_ternary_dual_gemv_bf16(
    const uint16_t* x,
    const uint32_t* a_weight,
    const uint16_t* a_alpha,
    const uint32_t* b_weight,
    const uint16_t* b_alpha,
    const int32_t* selected_experts,
    uint16_t* a_out,
    uint16_t* b_out,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (num_experts <= 0 || top_k <= 0 || in_features <= 0 || out_features <= 0
      || in_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  const int64_t blocks = top_k * out_features;
  hipLaunchKernelGGL(
      maple_selected_ternary_dual_gemv_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      2 * threads * sizeof(float),
      stream,
      x,
      a_weight,
      a_alpha,
      b_weight,
      b_alpha,
      selected_experts,
      a_out,
      b_out,
      num_experts,
      top_k,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_selected_ternary_gemv_bf16(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* row_alpha,
    const int32_t* selected_experts,
    uint16_t* out,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (num_experts <= 0 || top_k <= 0 || in_features <= 0 || out_features <= 0
      || in_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  const int64_t blocks = top_k * out_features;
  hipLaunchKernelGGL(
      maple_selected_ternary_gemv_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      x,
      weight,
      row_alpha,
      selected_experts,
      out,
      num_experts,
      top_k,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_affine4_embed_bf16(
    const uint32_t* weight,
    const uint16_t* scales,
    const uint16_t* biases,
    uint16_t* out,
    int64_t token_id,
    int64_t hidden_size,
    hipStream_t stream) {
  if (token_id < 0 || hidden_size <= 0 || hidden_size % 64 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      maple_affine4_embed_kernel,
      dim3(1),
      dim3(256),
      0,
      stream,
      weight,
      scales,
      biases,
      out,
      token_id,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_affine4_gemv_f32(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* scales,
    const uint16_t* biases,
    float* out,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (in_features <= 0 || out_features <= 0 || in_features % 64 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 8);
  hipLaunchKernelGGL(
      maple_affine4_gemv_kernel,
      dim3(static_cast<unsigned int>(out_features)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      x,
      weight,
      scales,
      biases,
      out,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}
