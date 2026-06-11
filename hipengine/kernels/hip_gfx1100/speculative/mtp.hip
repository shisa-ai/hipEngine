// Native MTP proposal helper kernels.
//
// These kernels implement provider-neutral pieces of the Qwen3.6/Qwen3.5 MTP
// head.  The first boundary normalizes the current token embedding and the
// target hidden row independently, concatenating the two BF16 streams so the
// existing dense BF16 GEMV kernel can apply mtp.fc.

#include <hip/hip_runtime.h>
#include <limits.h>
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

__device__ inline float warp_reduce_sum(float value) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  return value;
}

__device__ inline float block_reduce_sum(float value, float* shared) {
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + warpSize - 1) / warpSize;
    float total = 0.0f;
    for (int i = 0; i < warps; ++i) {
      total += shared[i];
    }
    shared[0] = total;
  }
  __syncthreads();
  return shared[0];
}

bool valid_shape(int64_t rows, int64_t hidden_size, int64_t vocab_size, int64_t threads) {
  return rows > 0 && hidden_size > 0 && vocab_size > 0 && (threads == 64 || threads == 128 || threads == 256);
}

bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256;
}

__device__ inline float sigmoid_f32(float x) {
  return 1.0f / (1.0f + expf(-x));
}

constexpr int kMtpRouterExperts = 256;
constexpr int kMtpRouterTopK = 8;

__device__ inline bool better_pair_i32(float value, int32_t index, float best_value, int32_t best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

__device__ inline void insert_topk_pair_i32(
    float value,
    int32_t index,
    float* top_values,
    int32_t* top_indices) {
  for (int pos = 0; pos < kMtpRouterTopK; ++pos) {
    if (better_pair_i32(value, index, top_values[pos], top_indices[pos])) {
      for (int shift = kMtpRouterTopK - 1; shift > pos; --shift) {
        top_values[shift] = top_values[shift - 1];
        top_indices[shift] = top_indices[shift - 1];
      }
      top_values[pos] = value;
      top_indices[pos] = index;
      return;
    }
  }
}

__global__ void mtp_fuse_inputs_f16_bf16_kernel(
    const int64_t* __restrict__ token_ids,
    const half_t* __restrict__ embedding_f16,
    const uint16_t* __restrict__ target_hidden_bf16,
    const uint16_t* __restrict__ embed_norm_weight_bf16,
    const uint16_t* __restrict__ hidden_norm_weight_bf16,
    uint16_t* __restrict__ out_concat_bf16,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int stream = static_cast<int>(blockIdx.y);  // 0: token embedding, 1: target hidden
  if (row >= rows || stream < 0 || stream > 1) {
    return;
  }
  extern __shared__ float scratch[];

  const int64_t token = token_ids[row];
  const bool valid_token = token >= 0 && token < vocab_size;
  float sum_sq = 0.0f;
  for (int64_t col = static_cast<int64_t>(threadIdx.x); col < hidden_size; col += blockDim.x) {
    float value = 0.0f;
    if (stream == 0) {
      value = valid_token ? static_cast<float>(embedding_f16[token * hidden_size + col]) : 0.0f;
    } else {
      value = bf16_bits_to_float(target_hidden_bf16[row * hidden_size + col]);
    }
    sum_sq += value * value;
  }
  const float variance_sum = block_reduce_sum(sum_sq, scratch);
  const float inv_rms = rsqrtf(variance_sum / static_cast<float>(hidden_size) + eps);

  const uint16_t* weight = (stream == 0) ? embed_norm_weight_bf16 : hidden_norm_weight_bf16;
  const int64_t out_base = row * (2 * hidden_size) + static_cast<int64_t>(stream) * hidden_size;
  for (int64_t col = static_cast<int64_t>(threadIdx.x); col < hidden_size; col += blockDim.x) {
    float value = 0.0f;
    if (stream == 0) {
      value = valid_token ? static_cast<float>(embedding_f16[token * hidden_size + col]) : 0.0f;
    } else {
      value = bf16_bits_to_float(target_hidden_bf16[row * hidden_size + col]);
    }
    const float scale = 1.0f + bf16_bits_to_float(weight[col]);
    out_concat_bf16[out_base + col] = float_to_bf16_bits(value * inv_rms * scale);
  }
}

__global__ void mtp_rmsnorm_bf16_oneplus_kernel(
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
  extern __shared__ float scratch[];
  const int64_t base = row * hidden_size;
  float sum_sq = 0.0f;
  for (int64_t col = static_cast<int64_t>(threadIdx.x); col < hidden_size; col += blockDim.x) {
    const float value = bf16_bits_to_float(x[base + col]);
    sum_sq += value * value;
  }
  const float variance_sum = block_reduce_sum(sum_sq, scratch);
  const float inv_rms = rsqrtf(variance_sum / static_cast<float>(hidden_size) + eps);
  for (int64_t col = static_cast<int64_t>(threadIdx.x); col < hidden_size; col += blockDim.x) {
    const float value = bf16_bits_to_float(x[base + col]);
    const float scale = 1.0f + bf16_bits_to_float(weight[col]);
    out[base + col] = float_to_bf16_bits(value * inv_rms * scale);
  }
}

__global__ void mtp_add_rmsnorm_bf16_oneplus_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ residual,
    const uint16_t* __restrict__ weight,
    uint16_t* __restrict__ out,
    uint16_t* __restrict__ residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  extern __shared__ float scratch[];
  const int64_t base = row * hidden_size;
  float sum_sq = 0.0f;
  for (int64_t col = static_cast<int64_t>(threadIdx.x); col < hidden_size; col += blockDim.x) {
    const uint16_t combined_bits = float_to_bf16_bits(bf16_bits_to_float(x[base + col]) + bf16_bits_to_float(residual[base + col]));
    const float combined = bf16_bits_to_float(combined_bits);
    sum_sq += combined * combined;
  }
  const float variance_sum = block_reduce_sum(sum_sq, scratch);
  const float inv_rms = rsqrtf(variance_sum / static_cast<float>(hidden_size) + eps);
  for (int64_t col = static_cast<int64_t>(threadIdx.x); col < hidden_size; col += blockDim.x) {
    const uint16_t combined_bits = float_to_bf16_bits(bf16_bits_to_float(x[base + col]) + bf16_bits_to_float(residual[base + col]));
    const float combined = bf16_bits_to_float(combined_bits);
    residual_out[base + col] = combined_bits;
    const float scale = 1.0f + bf16_bits_to_float(weight[col]);
    out[base + col] = float_to_bf16_bits(combined * inv_rms * scale);
  }
}

__global__ void mtp_split_q_gate_f32_bf16_kernel(
    const float* __restrict__ q_proj,
    float* __restrict__ query_out,
    uint16_t* __restrict__ gate_out,
    int64_t rows,
    int64_t num_q_heads,
    int64_t head_dim) {
  const int64_t total = rows * num_q_heads * head_dim;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t dim = idx % head_dim;
    const int64_t head = (idx / head_dim) % num_q_heads;
    const int64_t row = idx / (head_dim * num_q_heads);
    const int64_t src = (row * num_q_heads + head) * (2 * head_dim) + dim;
    const int64_t gate_src = src + head_dim;
    query_out[idx] = q_proj[src];
    gate_out[idx] = float_to_bf16_bits(q_proj[gate_src]);
  }
}

__global__ void mtp_gate_mul_bf16_kernel(
    const uint16_t* __restrict__ attn,
    const uint16_t* __restrict__ gate,
    uint16_t* __restrict__ out,
    int64_t elements) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < elements;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[idx] = float_to_bf16_bits(bf16_bits_to_float(attn[idx]) * sigmoid_f32(bf16_bits_to_float(gate[idx])));
  }
}

__global__ void mtp_softmax_topk_f32_kernel(
    const float* __restrict__ values,
    float* __restrict__ routing,
    int64_t rows,
    int64_t top_k) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows || threadIdx.x != 0) {
    return;
  }
  const int64_t base = row * top_k;
  float max_value = values[base];
  for (int64_t i = 1; i < top_k; ++i) {
    max_value = fmaxf(max_value, values[base + i]);
  }
  float denom = 0.0f;
  for (int64_t i = 0; i < top_k; ++i) {
    const float e = expf(values[base + i] - max_value);
    routing[base + i] = e;
    denom += e;
  }
  const float inv = 1.0f / denom;
  for (int64_t i = 0; i < top_k; ++i) {
    routing[base + i] *= inv;
  }
}

__global__ void mtp_router_topk_softmax_256x8_f32_kernel(
    const float* __restrict__ logits,
    float* __restrict__ out_values,
    int32_t* __restrict__ out_indices,
    float* __restrict__ routing) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  float top_values[kMtpRouterTopK];
  int32_t top_indices[kMtpRouterTopK];
  for (int i = 0; i < kMtpRouterTopK; ++i) {
    top_values[i] = -3.4028234663852886e+38F;
    top_indices[i] = INT32_MAX;
  }

  for (int i = 0; i < kMtpRouterExperts; ++i) {
    insert_topk_pair_i32(logits[i], static_cast<int32_t>(i), top_values, top_indices);
  }

  float max_value = top_values[0];
  for (int i = 1; i < kMtpRouterTopK; ++i) {
    max_value = fmaxf(max_value, top_values[i]);
  }

  float denom = 0.0f;
  for (int i = 0; i < kMtpRouterTopK; ++i) {
    out_values[i] = top_values[i];
    out_indices[i] = top_indices[i];
    const float e = expf(top_values[i] - max_value);
    routing[i] = e;
    denom += e;
  }
  const float inv = 1.0f / denom;
  for (int i = 0; i < kMtpRouterTopK; ++i) {
    routing[i] *= inv;
  }
}

__global__ void mtp_accumulate_route_bf16_to_f32_kernel(
    const uint16_t* __restrict__ src,
    const float* __restrict__ routing,
    float* __restrict__ accum,
    int64_t elements,
    int64_t route_index,
    int64_t reset_output) {
  const float scale = routing[route_index];
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < elements;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = scale * bf16_bits_to_float(src[idx]);
    accum[idx] = reset_output ? value : accum[idx] + value;
  }
}

__global__ void mtp_accumulate_routes_bf16_to_f32_kernel(
    const uint16_t* __restrict__ src_routes,
    const float* __restrict__ routing,
    float* __restrict__ accum,
    int64_t routes,
    int64_t elements) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < elements;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float total = 0.0f;
    for (int64_t route = 0; route < routes; ++route) {
      const volatile float value = routing[route] * bf16_bits_to_float(src_routes[route * elements + idx]);
      if (route == 0) {
        total = value;
      } else {
        const volatile float previous = total;
        total = previous + value;
      }
    }
    accum[idx] = total;
  }
}

__global__ void mtp_accumulate_sigmoid_gate_bf16_to_f32_kernel(
    const uint16_t* __restrict__ src,
    const float* __restrict__ gate,
    float* __restrict__ accum,
    int64_t elements) {
  const float scale = sigmoid_f32(gate[0]);
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < elements;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    accum[idx] += scale * bf16_bits_to_float(src[idx]);
  }
}

__global__ void mtp_finalize_f32_to_bf16_kernel(
    const float* __restrict__ src,
    uint16_t* __restrict__ out,
    int64_t elements) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < elements;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[idx] = float_to_bf16_bits(src[idx]);
  }
}

}  // namespace

extern "C" int hipengine_mtp_fuse_inputs_f16_bf16(
    const int64_t* token_ids,
    const half_t* embedding_f16,
    const uint16_t* target_hidden_bf16,
    const uint16_t* embed_norm_weight_bf16,
    const uint16_t* hidden_norm_weight_bf16,
    uint16_t* out_concat_bf16,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(rows, hidden_size, vocab_size, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      mtp_fuse_inputs_f16_bf16_kernel,
      dim3(static_cast<unsigned int>(rows), 2U),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<unsigned int>(((threads + 31) / 32) * sizeof(float)),
      stream,
      token_ids,
      embedding_f16,
      target_hidden_bf16,
      embed_norm_weight_bf16,
      hidden_norm_weight_bf16,
      out_concat_bf16,
      rows,
      hidden_size,
      vocab_size,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_mtp_rmsnorm_bf16_oneplus(
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      mtp_rmsnorm_bf16_oneplus_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<unsigned int>(((threads + 31) / 32) * sizeof(float)),
      stream,
      x,
      weight,
      out,
      rows,
      hidden_size,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_mtp_add_rmsnorm_bf16_oneplus(
    const uint16_t* x,
    const uint16_t* residual,
    const uint16_t* weight,
    uint16_t* out,
    uint16_t* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      mtp_add_rmsnorm_bf16_oneplus_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<unsigned int>(((threads + 31) / 32) * sizeof(float)),
      stream,
      x,
      residual,
      weight,
      out,
      residual_out,
      rows,
      hidden_size,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_mtp_split_q_gate_f32_bf16(
    const float* q_proj,
    float* query_out,
    uint16_t* gate_out,
    int64_t rows,
    int64_t num_q_heads,
    int64_t head_dim,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || num_q_heads <= 0 || head_dim <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t elements = rows * num_q_heads * head_dim;
  const int64_t blocks = (elements + threads - 1) / threads;
  hipLaunchKernelGGL(
      mtp_split_q_gate_f32_bf16_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      q_proj,
      query_out,
      gate_out,
      rows,
      num_q_heads,
      head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_mtp_gate_mul_bf16(
    const uint16_t* attn,
    const uint16_t* gate,
    uint16_t* out,
    int64_t elements,
    int64_t threads,
    hipStream_t stream) {
  if (elements <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (elements + threads - 1) / threads;
  hipLaunchKernelGGL(
      mtp_gate_mul_bf16_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      attn,
      gate,
      out,
      elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_mtp_softmax_topk_f32(
    const float* values,
    float* routing,
    int64_t rows,
    int64_t top_k,
    hipStream_t stream) {
  if (rows <= 0 || top_k <= 0 || top_k > 8) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      mtp_softmax_topk_f32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(1U),
      0,
      stream,
      values,
      routing,
      rows,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_mtp_router_topk_softmax_256x8_f32(
    const float* logits,
    float* out_values,
    int32_t* out_indices,
    float* routing,
    int64_t num_experts,
    int64_t top_k,
    hipStream_t stream) {
  if (num_experts != kMtpRouterExperts || top_k != kMtpRouterTopK) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      mtp_router_topk_softmax_256x8_f32_kernel,
      dim3(1U),
      dim3(1U),
      0,
      stream,
      logits,
      out_values,
      out_indices,
      routing);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_mtp_accumulate_route_bf16_to_f32(
    const uint16_t* src,
    const float* routing,
    float* accum,
    int64_t elements,
    int64_t route_index,
    int64_t reset_output,
    int64_t threads,
    hipStream_t stream) {
  if (elements <= 0 || route_index < 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (elements + threads - 1) / threads;
  hipLaunchKernelGGL(
      mtp_accumulate_route_bf16_to_f32_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      src,
      routing,
      accum,
      elements,
      route_index,
      reset_output);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_mtp_accumulate_routes_bf16_to_f32(
    const uint16_t* src_routes,
    const float* routing,
    float* accum,
    int64_t routes,
    int64_t elements,
    int64_t threads,
    hipStream_t stream) {
  if (routes <= 0 || elements <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (elements + threads - 1) / threads;
  hipLaunchKernelGGL(
      mtp_accumulate_routes_bf16_to_f32_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      src_routes,
      routing,
      accum,
      routes,
      elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_mtp_accumulate_sigmoid_gate_bf16_to_f32(
    const uint16_t* src,
    const float* gate,
    float* accum,
    int64_t elements,
    int64_t threads,
    hipStream_t stream) {
  if (elements <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (elements + threads - 1) / threads;
  hipLaunchKernelGGL(
      mtp_accumulate_sigmoid_gate_bf16_to_f32_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      src,
      gate,
      accum,
      elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_mtp_finalize_f32_to_bf16(
    const float* src,
    uint16_t* out,
    int64_t elements,
    int64_t threads,
    hipStream_t stream) {
  if (elements <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (elements + threads - 1) / threads;
  hipLaunchKernelGGL(
      mtp_finalize_f32_to_bf16_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      src,
      out,
      elements);
  return static_cast<int>(hipGetLastError());
}
