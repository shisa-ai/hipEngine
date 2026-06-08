// Minimal GPU lm-head + argmax bring-up kernel for Qwen3.5/PARO E2E smokes.
// Hidden is BF16 bits, weight is checkpoint FP16 bits, logits are FP32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
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

__device__ inline bool better_pair(float value, int64_t index, float best_value, int64_t best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

__global__ void lm_head_fp16_logits_kernel(
    const uint16_t* __restrict__ hidden_bf16,
    const uint16_t* __restrict__ weight_fp16,
    float* __restrict__ logits,
    int64_t hidden_size,
    int64_t vocab_size) {
  const int64_t vocab_row = static_cast<int64_t>(blockIdx.x);
  if (vocab_row >= vocab_size) {
    return;
  }
  const int tid = threadIdx.x;
  const __half* weight = reinterpret_cast<const __half*>(weight_fp16);
  const int64_t weight_base = vocab_row * hidden_size;
  float acc = 0.0f;
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (hidden_size / vec_stride) * vec_stride;
  for (int64_t k = static_cast<int64_t>(tid) * 8; k < main_end; k += vec_stride) {
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 0]), __half2float(weight[weight_base + k + 0]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 1]), __half2float(weight[weight_base + k + 1]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 2]), __half2float(weight[weight_base + k + 2]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 3]), __half2float(weight[weight_base + k + 3]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 4]), __half2float(weight[weight_base + k + 4]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 5]), __half2float(weight[weight_base + k + 5]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 6]), __half2float(weight[weight_base + k + 6]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 7]), __half2float(weight[weight_base + k + 7]), acc);
  }
  for (int64_t k = main_end + tid; k < hidden_size; k += blockDim.x) {
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k]), __half2float(weight[weight_base + k]), acc);
  }

  extern __shared__ float partial[];
  partial[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    logits[vocab_row] = partial[0];
  }
}

__global__ void lm_head_fp16_logits_rows_kernel(
    const uint16_t* __restrict__ hidden_bf16,
    const uint16_t* __restrict__ weight_fp16,
    float* __restrict__ logits,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size) {
  const int64_t vocab_row = static_cast<int64_t>(blockIdx.x);
  const int64_t verify_row = static_cast<int64_t>(blockIdx.y);
  if (vocab_row >= vocab_size || verify_row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const __half* weight = reinterpret_cast<const __half*>(weight_fp16);
  const int64_t hidden_base = verify_row * hidden_size;
  const int64_t weight_base = vocab_row * hidden_size;
  float acc = 0.0f;
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (hidden_size / vec_stride) * vec_stride;
  for (int64_t k = static_cast<int64_t>(tid) * 8; k < main_end; k += vec_stride) {
    acc = fmaf(bf16_bits_to_float(hidden_bf16[hidden_base + k + 0]), __half2float(weight[weight_base + k + 0]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[hidden_base + k + 1]), __half2float(weight[weight_base + k + 1]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[hidden_base + k + 2]), __half2float(weight[weight_base + k + 2]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[hidden_base + k + 3]), __half2float(weight[weight_base + k + 3]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[hidden_base + k + 4]), __half2float(weight[weight_base + k + 4]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[hidden_base + k + 5]), __half2float(weight[weight_base + k + 5]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[hidden_base + k + 6]), __half2float(weight[weight_base + k + 6]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[hidden_base + k + 7]), __half2float(weight[weight_base + k + 7]), acc);
  }
  for (int64_t k = main_end + tid; k < hidden_size; k += blockDim.x) {
    acc = fmaf(bf16_bits_to_float(hidden_bf16[hidden_base + k]), __half2float(weight[weight_base + k]), acc);
  }

  extern __shared__ float partial[];
  partial[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    logits[verify_row * vocab_size + vocab_row] = partial[0];
  }
}

__global__ void argmax_stage1_kernel(
    const float* __restrict__ logits,
    float* __restrict__ block_values,
    int64_t* __restrict__ block_indices,
    int64_t vocab_size) {
  extern __shared__ unsigned char shared_raw[];
  float* values = reinterpret_cast<float*>(shared_raw);
  int64_t* indices = reinterpret_cast<int64_t*>(values + blockDim.x);
  const int tid = threadIdx.x;
  const int64_t chunk = static_cast<int64_t>(blockDim.x) * 4;
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * chunk;
  const int64_t end = min(begin + chunk, vocab_size);

  float best_value = -3.4028234663852886e+38F;
  int64_t best_index = INT64_MAX;
  for (int64_t i = begin + tid; i < end; i += blockDim.x) {
    const float value = logits[i];
    if (better_pair(value, i, best_value, best_index)) {
      best_value = value;
      best_index = i;
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && better_pair(values[tid + stride], indices[tid + stride], values[tid], indices[tid])) {
      values[tid] = values[tid + stride];
      indices[tid] = indices[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_values[blockIdx.x] = values[0];
    block_indices[blockIdx.x] = indices[0];
  }
}

__global__ void argmax_stage2_kernel(
    const float* __restrict__ block_values,
    const int64_t* __restrict__ block_indices,
    int64_t num_blocks,
    int64_t* __restrict__ out_index,
    float* __restrict__ out_value) {
  extern __shared__ unsigned char shared_raw[];
  float* values = reinterpret_cast<float*>(shared_raw);
  int64_t* indices = reinterpret_cast<int64_t*>(values + blockDim.x);
  const int tid = threadIdx.x;
  float best_value = -3.4028234663852886e+38F;
  int64_t best_index = INT64_MAX;
  for (int64_t i = tid; i < num_blocks; i += blockDim.x) {
    const float value = block_values[i];
    const int64_t index = block_indices[i];
    if (better_pair(value, index, best_value, best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && better_pair(values[tid + stride], indices[tid + stride], values[tid], indices[tid])) {
      values[tid] = values[tid + stride];
      indices[tid] = indices[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *out_index = indices[0];
    *out_value = values[0];
  }
}

__device__ inline bool better_pair_i32(float value, int32_t index, float best_value, int32_t best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

// R3.7: Fused W8A16 LM-head GEMV + per-block argmax stage 1.
//
// Eliminates the [rows x vocab_size] FP32 logits buffer that the unfused path
// (`hipengine_w8a16_linear_bf16_f32_multi_row` + `argmax_rows_stage1_i32_kernel`)
// materializes in HBM.  Each block of 256 threads owns a vocab tile of
// `chunk = blockDim.x * 4` rows for one verifier row; the block iterates over
// its vocab tile and uses the SAME 256-way cooperative dot-product pattern as
// `w8a16_linear_multi_row_kernel` (each thread strides across hidden_size with
// 8x unroll, then LDS reduces 256 partials to one).  Thread 0 keeps the
// running (max_value, min_index_on_tie) and writes one (value, index) pair to
// per-block scratch that the unchanged ``argmax_rows_stage2_i32_kernel``
// finalizes.  The cooperative dot pattern preserves HBM coalescing across
// threads (consecutive bytes within one vocab row) -- the alternative of
// each-thread-owns-its-vocab-row was tried first and ran 22 ms/cycle SLOWER
// because of scattered access across 256 different vocab rows simultaneously.
//
// Numerics: bit-exact vs the unfused path.  Per-vocab-row reduction order is
// identical to ``w8a16_linear_multi_row_kernel`` (256 partial accumulators
// followed by an 8-step LDS reduce, scale applied at the end), so the FP32
// per-row logit is bit-equal.  The block-level argmax reduction follows
// ``argmax_rows_stage1_i32_kernel`` semantics with the same
// ``better_pair_i32`` tiebreak.
__global__ void w8a16_lm_head_argmax_rows_stage1_kernel(
    const uint16_t* __restrict__ hidden,
    const int8_t* __restrict__ weight,
    const float* __restrict__ weight_scale,
    float* __restrict__ block_values,
    int32_t* __restrict__ block_indices,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t stage1_blocks) {
  extern __shared__ float partial[];

  const int tid = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows) {
    return;
  }
  const int64_t chunk = static_cast<int64_t>(blockDim.x) * 4;
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * chunk;
  const int64_t end = min(begin + chunk, vocab_size);
  const int64_t hidden_row = row * hidden_size;
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;

  float best_value = -3.4028234663852886e+38F;
  int32_t best_index = INT32_MAX;

  for (int64_t v = begin; v < end; ++v) {
    const int64_t weight_row_off = v * hidden_size;
    float acc = 0.0f;
    for (int64_t k = static_cast<int64_t>(tid) * 8; k + 7 < hidden_size; k += vec_stride) {
      const int64_t h_off = hidden_row + k;
      const int64_t w_off = weight_row_off + k;
      acc += bf16_bits_to_float(hidden[h_off + 0]) * static_cast<float>(static_cast<int>(weight[w_off + 0]));
      acc += bf16_bits_to_float(hidden[h_off + 1]) * static_cast<float>(static_cast<int>(weight[w_off + 1]));
      acc += bf16_bits_to_float(hidden[h_off + 2]) * static_cast<float>(static_cast<int>(weight[w_off + 2]));
      acc += bf16_bits_to_float(hidden[h_off + 3]) * static_cast<float>(static_cast<int>(weight[w_off + 3]));
      acc += bf16_bits_to_float(hidden[h_off + 4]) * static_cast<float>(static_cast<int>(weight[w_off + 4]));
      acc += bf16_bits_to_float(hidden[h_off + 5]) * static_cast<float>(static_cast<int>(weight[w_off + 5]));
      acc += bf16_bits_to_float(hidden[h_off + 6]) * static_cast<float>(static_cast<int>(weight[w_off + 6]));
      acc += bf16_bits_to_float(hidden[h_off + 7]) * static_cast<float>(static_cast<int>(weight[w_off + 7]));
    }
    for (int64_t k = (hidden_size & ~7) + tid; k < hidden_size; k += blockDim.x) {
      acc += bf16_bits_to_float(hidden[hidden_row + k]) *
          static_cast<float>(static_cast<int>(weight[weight_row_off + k]));
    }
    partial[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        partial[tid] += partial[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      const float logit = partial[0] * weight_scale[v];
      const int32_t idx = static_cast<int32_t>(v);
      if (better_pair_i32(logit, idx, best_value, best_index)) {
        best_value = logit;
        best_index = idx;
      }
    }
    // Ensure partial[0] is safe to overwrite on the next iteration.
    __syncthreads();
  }

  if (tid == 0) {
    const int64_t off = row * stage1_blocks + static_cast<int64_t>(blockIdx.x);
    block_values[off] = best_value;
    block_indices[off] = best_index;
  }
}

__global__ void argmax_rows_stage1_i32_kernel(
    const float* __restrict__ logits,
    float* __restrict__ block_values,
    int32_t* __restrict__ block_indices,
    int64_t rows,
    int64_t vocab_size,
    int64_t stage1_blocks) {
  extern __shared__ unsigned char shared_raw[];
  float* values = reinterpret_cast<float*>(shared_raw);
  int32_t* indices = reinterpret_cast<int32_t*>(values + blockDim.x);
  const int tid = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows) {
    return;
  }
  const int64_t chunk = static_cast<int64_t>(blockDim.x) * 4;
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * chunk;
  const int64_t end = min(begin + chunk, vocab_size);
  const int64_t logits_base = row * vocab_size;

  float best_value = -3.4028234663852886e+38F;
  int32_t best_index = INT32_MAX;
  for (int64_t i = begin + tid; i < end; i += blockDim.x) {
    const float value = logits[logits_base + i];
    const int32_t index = static_cast<int32_t>(i);
    if (better_pair_i32(value, index, best_value, best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && better_pair_i32(values[tid + stride], indices[tid + stride], values[tid], indices[tid])) {
      values[tid] = values[tid + stride];
      indices[tid] = indices[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const int64_t block_offset = row * stage1_blocks + static_cast<int64_t>(blockIdx.x);
    block_values[block_offset] = values[0];
    block_indices[block_offset] = indices[0];
  }
}

__global__ void argmax_rows_stage2_i32_kernel(
    const float* __restrict__ block_values,
    const int32_t* __restrict__ block_indices,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_values,
    int64_t rows,
    int64_t num_blocks) {
  extern __shared__ unsigned char shared_raw[];
  float* values = reinterpret_cast<float*>(shared_raw);
  int32_t* indices = reinterpret_cast<int32_t*>(values + blockDim.x);
  const int tid = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int64_t block_base = row * num_blocks;
  float best_value = -3.4028234663852886e+38F;
  int32_t best_index = INT32_MAX;
  for (int64_t i = tid; i < num_blocks; i += blockDim.x) {
    const float value = block_values[block_base + i];
    const int32_t index = block_indices[block_base + i];
    if (better_pair_i32(value, index, best_value, best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && better_pair_i32(values[tid + stride], indices[tid + stride], values[tid], indices[tid])) {
      values[tid] = values[tid + stride];
      indices[tid] = indices[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_indices[row] = indices[0];
    if (out_values != nullptr) {
      out_values[row] = values[0];
    }
  }
}

constexpr int kMaxTopK = 8;

__device__ inline void insert_topk_pair(
    float value,
    int32_t index,
    float* top_values,
    int32_t* top_indices,
    int top_k) {
  for (int pos = 0; pos < top_k; ++pos) {
    if (better_pair_i32(value, index, top_values[pos], top_indices[pos])) {
      for (int shift = top_k - 1; shift > pos; --shift) {
        top_values[shift] = top_values[shift - 1];
        top_indices[shift] = top_indices[shift - 1];
      }
      top_values[pos] = value;
      top_indices[pos] = index;
      return;
    }
  }
}

__global__ void topk_rows_i32_kernel(
    const float* __restrict__ logits,
    float* __restrict__ out_values,
    int32_t* __restrict__ out_indices,
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
  float local_values[kMaxTopK];
  int32_t local_indices[kMaxTopK];
  for (int pos = 0; pos < kMaxTopK; ++pos) {
    local_values[pos] = -3.4028234663852886e+38F;
    local_indices[pos] = INT32_MAX;
  }
  const int64_t base = row * vocab_size;
  for (int64_t i = tid; i < vocab_size; i += blockDim.x) {
    insert_topk_pair(logits[base + i], static_cast<int32_t>(i), local_values, local_indices, top_k);
  }
  for (int pos = 0; pos < top_k; ++pos) {
    const int64_t off = static_cast<int64_t>(tid) * top_k + pos;
    shared_values[off] = local_values[pos];
    shared_indices[off] = local_indices[pos];
  }
  __syncthreads();
  if (tid == 0) {
    float final_values[kMaxTopK];
    int32_t final_indices[kMaxTopK];
    for (int pos = 0; pos < kMaxTopK; ++pos) {
      final_values[pos] = -3.4028234663852886e+38F;
      final_indices[pos] = INT32_MAX;
    }
    for (int thread = 0; thread < blockDim.x; ++thread) {
      for (int pos = 0; pos < top_k; ++pos) {
        const int64_t off = static_cast<int64_t>(thread) * top_k + pos;
        insert_topk_pair(shared_values[off], shared_indices[off], final_values, final_indices, top_k);
      }
    }
    const int64_t out_base = row * top_k;
    for (int pos = 0; pos < top_k; ++pos) {
      out_indices[out_base + pos] = final_indices[pos];
      if (out_values != nullptr) {
        out_values[out_base + pos] = final_values[pos];
      }
    }
  }
}

bool valid_threads(int64_t threads) {
  return threads == 128 || threads == 256 || threads == 512;
}

}  // namespace

extern "C" int hipengine_argmax_f32(
    const float* logits,
    float* block_values,
    int64_t* block_indices,
    int64_t* out_index,
    float* out_value,
    int64_t vocab_size,
    int64_t threads,
    hipStream_t stream) {
  if (vocab_size <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int t = static_cast<unsigned int>(threads);
  const int64_t stage1_blocks = (vocab_size + threads * 4 - 1) / (threads * 4);
  const size_t argmax_smem = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int64_t));
  hipLaunchKernelGGL(
      argmax_stage1_kernel,
      dim3(static_cast<unsigned int>(stage1_blocks)),
      dim3(t),
      argmax_smem,
      stream,
      logits,
      block_values,
      block_indices,
      vocab_size);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }

  hipLaunchKernelGGL(
      argmax_stage2_kernel,
      dim3(1),
      dim3(t),
      argmax_smem,
      stream,
      block_values,
      block_indices,
      stage1_blocks,
      out_index,
      out_value);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_lm_head_fp16_argmax_bf16(
    const uint16_t* hidden_bf16,
    const uint16_t* weight_fp16,
    float* logits,
    float* block_values,
    int64_t* block_indices,
    int64_t* out_index,
    float* out_value,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t threads,
    hipStream_t stream) {
  if (hidden_size <= 0 || vocab_size <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int t = static_cast<unsigned int>(threads);
  hipLaunchKernelGGL(
      lm_head_fp16_logits_kernel,
      dim3(static_cast<unsigned int>(vocab_size)),
      dim3(t),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_bf16,
      weight_fp16,
      logits,
      hidden_size,
      vocab_size);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }

  const int64_t stage1_blocks = (vocab_size + threads * 4 - 1) / (threads * 4);
  const size_t argmax_smem = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int64_t));
  hipLaunchKernelGGL(
      argmax_stage1_kernel,
      dim3(static_cast<unsigned int>(stage1_blocks)),
      dim3(t),
      argmax_smem,
      stream,
      logits,
      block_values,
      block_indices,
      vocab_size);
  err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }

  hipLaunchKernelGGL(
      argmax_stage2_kernel,
      dim3(1),
      dim3(t),
      argmax_smem,
      stream,
      block_values,
      block_indices,
      stage1_blocks,
      out_index,
      out_value);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_argmax_f32_rows_i32(
    const float* logits,
    float* block_values,
    int32_t* block_indices,
    int32_t* out_indices,
    float* out_values,
    int64_t rows,
    int64_t vocab_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || vocab_size <= 0 || vocab_size > INT32_MAX || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int t = static_cast<unsigned int>(threads);
  const int64_t stage1_blocks = (vocab_size + threads * 4 - 1) / (threads * 4);
  const size_t argmax_smem = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int32_t));
  hipLaunchKernelGGL(
      argmax_rows_stage1_i32_kernel,
      dim3(static_cast<unsigned int>(stage1_blocks), static_cast<unsigned int>(rows)),
      dim3(t),
      argmax_smem,
      stream,
      logits,
      block_values,
      block_indices,
      rows,
      vocab_size,
      stage1_blocks);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }

  hipLaunchKernelGGL(
      argmax_rows_stage2_i32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(t),
      argmax_smem,
      stream,
      block_values,
      block_indices,
      out_indices,
      out_values,
      rows,
      stage1_blocks);
  return static_cast<int>(hipGetLastError());
}

// R3.7: Fused W8A16 LM-head + argmax-rows.  Replaces the unfused pair
//   `hipengine_w8a16_linear_bf16_f32_multi_row` -> `hipengine_argmax_f32_rows_i32`
// in the DFlash chain verifier window so the per-cycle full-vocab FP32 logits
// buffer is never written to HBM.  Bit-equivalent to the unfused path because
// per-vocab-row dot product math, scale order, and tiebreak are preserved.
extern "C" int hipengine_w8a16_lm_head_argmax_rows_bf16(
    const uint16_t* hidden,
    const int8_t* weight,
    const float* weight_scale,
    float* block_values,
    int32_t* block_indices,
    int32_t* out_indices,
    float* out_values,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || vocab_size <= 0 || vocab_size > INT32_MAX || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int t = static_cast<unsigned int>(threads);
  const int64_t stage1_blocks = (vocab_size + threads * 4 - 1) / (threads * 4);
  const size_t stage1_smem = static_cast<size_t>(threads) * sizeof(float);
  const size_t stage2_smem = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int32_t));
  hipLaunchKernelGGL(
      w8a16_lm_head_argmax_rows_stage1_kernel,
      dim3(static_cast<unsigned int>(stage1_blocks), static_cast<unsigned int>(rows)),
      dim3(t),
      stage1_smem,
      stream,
      hidden,
      weight,
      weight_scale,
      block_values,
      block_indices,
      rows,
      hidden_size,
      vocab_size,
      stage1_blocks);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }

  hipLaunchKernelGGL(
      argmax_rows_stage2_i32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(t),
      stage2_smem,
      stream,
      block_values,
      block_indices,
      out_indices,
      out_values,
      rows,
      stage1_blocks);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_topk_f32_rows_i32(
    const float* logits,
    float* out_values,
    int32_t* out_indices,
    int64_t rows,
    int64_t vocab_size,
    int64_t top_k,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || vocab_size <= 0 || vocab_size > INT32_MAX || top_k <= 0 || top_k > kMaxTopK || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const size_t smem = static_cast<size_t>(threads) * static_cast<size_t>(top_k) * (sizeof(float) + sizeof(int32_t));
  hipLaunchKernelGGL(
      topk_rows_i32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      smem,
      stream,
      logits,
      out_values,
      out_indices,
      rows,
      vocab_size,
      static_cast<int32_t>(top_k));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_lm_head_fp16_argmax_bf16_rows_i32(
    const uint16_t* hidden_bf16,
    const uint16_t* weight_fp16,
    float* logits,
    float* block_values,
    int32_t* block_indices,
    int32_t* out_indices,
    float* out_values,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || vocab_size <= 0 || vocab_size > INT32_MAX || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int t = static_cast<unsigned int>(threads);
  hipLaunchKernelGGL(
      lm_head_fp16_logits_rows_kernel,
      dim3(static_cast<unsigned int>(vocab_size), static_cast<unsigned int>(rows)),
      dim3(t),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_bf16,
      weight_fp16,
      logits,
      rows,
      hidden_size,
      vocab_size);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }

  return hipengine_argmax_f32_rows_i32(
      logits,
      block_values,
      block_indices,
      out_indices,
      out_values,
      rows,
      vocab_size,
      threads,
      stream);
}

__global__ void batch_argmax_stage1_kernel(
    const float* __restrict__ logits,
    float* __restrict__ block_values,
    int64_t* __restrict__ block_indices,
    int64_t vocab_size,
    int64_t stage1_blocks) {
  extern __shared__ unsigned char shared_raw[];
  float* values = reinterpret_cast<float*>(shared_raw);
  int64_t* indices = reinterpret_cast<int64_t*>(values + blockDim.x);
  const int tid = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  const int64_t chunk = static_cast<int64_t>(blockDim.x) * 4;
  const int64_t begin = block * chunk;
  const int64_t end = min(begin + chunk, vocab_size);
  const float* row_logits = logits + row * vocab_size;

  float best_value = -3.4028234663852886e+38F;
  int64_t best_index = INT64_MAX;
  for (int64_t i = begin + tid; i < end; i += blockDim.x) {
    const float value = row_logits[i];
    if (better_pair(value, i, best_value, best_index)) {
      best_value = value;
      best_index = i;
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && better_pair(values[tid + stride], indices[tid + stride], values[tid], indices[tid])) {
      values[tid] = values[tid + stride];
      indices[tid] = indices[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const int64_t out = row * stage1_blocks + block;
    block_values[out] = values[0];
    block_indices[out] = indices[0];
  }
}

__global__ void batch_argmax_stage2_kernel(
    const float* __restrict__ block_values,
    const int64_t* __restrict__ block_indices,
    int64_t num_blocks,
    int64_t* __restrict__ out_index,
    float* __restrict__ out_value) {
  extern __shared__ unsigned char shared_raw[];
  float* values = reinterpret_cast<float*>(shared_raw);
  int64_t* indices = reinterpret_cast<int64_t*>(values + blockDim.x);
  const int tid = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t row_base = row * num_blocks;
  float best_value = -3.4028234663852886e+38F;
  int64_t best_index = INT64_MAX;
  for (int64_t i = tid; i < num_blocks; i += blockDim.x) {
    const float value = block_values[row_base + i];
    const int64_t index = block_indices[row_base + i];
    if (better_pair(value, index, best_value, best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && better_pair(values[tid + stride], indices[tid + stride], values[tid], indices[tid])) {
      values[tid] = values[tid + stride];
      indices[tid] = indices[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_index[row] = indices[0];
    out_value[row] = values[0];
  }
}

extern "C" int hipengine_batch_argmax_f32(
    const float* logits,
    float* block_values,
    int64_t* block_indices,
    int64_t* out_index,
    float* out_value,
    int64_t rows,
    int64_t vocab_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || vocab_size <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t stage1_blocks = (vocab_size + threads * 4 - 1) / (threads * 4);
  if (rows > 2147483647LL || stage1_blocks > 2147483647LL) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int t = static_cast<unsigned int>(threads);
  const size_t argmax_smem = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int64_t));
  hipLaunchKernelGGL(
      batch_argmax_stage1_kernel,
      dim3(static_cast<unsigned int>(stage1_blocks), static_cast<unsigned int>(rows)),
      dim3(t),
      argmax_smem,
      stream,
      logits,
      block_values,
      block_indices,
      vocab_size,
      stage1_blocks);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }

  hipLaunchKernelGGL(
      batch_argmax_stage2_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(t),
      argmax_smem,
      stream,
      block_values,
      block_indices,
      stage1_blocks,
      out_index,
      out_value);
  return static_cast<int>(hipGetLastError());
}
