// Net-new Maple MoE kernels from the public DeepGrove inference contract.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

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

__global__ void maple_router_topk_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ weight,
    int32_t* __restrict__ selected_experts,
    float* __restrict__ selected_weights,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k) {
  const int tid = static_cast<int>(threadIdx.x);
  __shared__ float logits[256];
  __shared__ float probabilities[256];
  __shared__ float reduction[256];
  __shared__ uint8_t selected_mask[256];

  float logit = -INFINITY;
  if (tid < num_experts) {
    logit = 0.0f;
    const uint16_t* row = weight + static_cast<int64_t>(tid) * hidden_size;
    for (int64_t idx = 0; idx < hidden_size; ++idx) {
      logit += bf16_to_float(x[idx]) * bf16_to_float(row[idx]);
    }
  }
  logits[tid] = logit;
  reduction[tid] = logit;
  selected_mask[tid] = 0;
  __syncthreads();

  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  const float max_logit = reduction[0];
  const float exponential = tid < num_experts ? expf(logits[tid] - max_logit) : 0.0f;
  probabilities[tid] = exponential;
  reduction[tid] = exponential;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  if (tid < num_experts) {
    probabilities[tid] /= reduction[0];
  }
  __syncthreads();

  if (tid == 0) {
    float selected_sum = 0.0f;
    for (int64_t rank = 0; rank < top_k; ++rank) {
      int best_id = -1;
      float best_probability = -1.0f;
      for (int expert = 0; expert < num_experts; ++expert) {
        const float probability = probabilities[expert];
        if (!selected_mask[expert]
            && (probability > best_probability
                || (probability == best_probability && expert < best_id))) {
          best_probability = probability;
          best_id = expert;
        }
      }
      selected_experts[rank] = best_id;
      selected_weights[rank] = best_probability;
      selected_mask[best_id] = 1;
      selected_sum += best_probability;
    }
    for (int64_t rank = 0; rank < top_k; ++rank) {
      selected_weights[rank] /= selected_sum + 1.0e-20f;
    }
  }
}

__global__ void maple_router_logits_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ weight,
    float* __restrict__ logits_out,
    int64_t hidden_size,
    int64_t num_experts) {
  // One block per expert: coalesced strided dot, then a block tree reduce. The
  // logits land in a global [num_experts] scratch consumed by
  // maple_router_softmax_topk_kernel. Tree reduction order may differ from the
  // reference sequential accumulation by a few ULP; IDs and the model gate are
  // robust to that (the exact single-block variant remains the reference).
  const int e = static_cast<int>(blockIdx.x);
  const int t = static_cast<int>(threadIdx.x);
  float acc = 0.0f;
  const uint16_t* row = weight + static_cast<int64_t>(e) * hidden_size;
  for (int64_t i = t; i < hidden_size; i += blockDim.x) {
    acc += bf16_to_float(x[i]) * bf16_to_float(row[i]);
  }
  __shared__ float red[256];
  red[t] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  if (t == 0 && e < num_experts) logits_out[e] = red[0];
}

__global__ void maple_router_softmax_topk_kernel(
    const float* __restrict__ logits_in,
    int32_t* __restrict__ selected_experts,
    float* __restrict__ selected_weights,
    int64_t num_experts,
    int64_t top_k) {
  // Single block, one thread per expert: max/exp/sum softmax, then a parallel
  // rank-count stable top-k (desc by probability, ties by ascending expert id),
  // then renormalize by the selected sum. No serial per-expert scan.
  const int t = static_cast<int>(threadIdx.x);
  __shared__ float logits[256];
  __shared__ float probs[256];
  __shared__ float red[256];
  __shared__ uint8_t sel_mask[256];

  const float logit = (t < num_experts) ? logits_in[t] : -INFINITY;
  logits[t] = logit;
  red[t] = logit;
  sel_mask[t] = 0;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (t < s) red[t] = fmaxf(red[t], red[t + s]);
    __syncthreads();
  }
  const float max_logit = red[0];
  const float ex = (t < num_experts) ? expf(logits[t] - max_logit) : 0.0f;
  probs[t] = ex;
  red[t] = ex;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  const float inv_sum = 1.0f / red[0];
  if (t < num_experts) probs[t] *= inv_sum;
  __syncthreads();

  if (t < num_experts) {
    const float my = probs[t];
    int rank = 0;
    for (int e2 = 0; e2 < num_experts; ++e2) {
      const float p2 = probs[e2];
      if (p2 > my || (p2 == my && e2 < t)) ++rank;
    }
    if (rank < top_k) {
      selected_experts[rank] = t;
      sel_mask[t] = 1;
    }
  }
  __syncthreads();

  float sum = 0.0f;
  for (int e2 = 0; e2 < num_experts; ++e2) {
    if (sel_mask[e2]) sum += probs[e2];
  }
  if (t < num_experts && sel_mask[t]) {
    const float my = probs[t];
    int rank = 0;
    for (int e2 = 0; e2 < num_experts; ++e2) {
      const float p2 = probs[e2];
      if (p2 > my || (p2 == my && e2 < t)) ++rank;
    }
    selected_weights[rank] = probs[t] / (sum + 1.0e-20f);
  }
}

// D0 one-dispatch c1 router. Every expert block writes one tree-reduced logit,
// then the last block to increment completion_counter performs the unchanged
// softmax/stable-top-k body. atomicInc wraps the counter back to zero on the
// last ticket, so sequential calls on the same stream require no reset launch.
__global__ void maple_router_topk_single_dispatch_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ weight,
    int32_t* __restrict__ selected_experts,
    float* __restrict__ selected_weights,
    float* __restrict__ logits_out,
    unsigned int* __restrict__ completion_counter,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k) {
  const int e = static_cast<int>(blockIdx.x);
  const int t = static_cast<int>(threadIdx.x);
  __shared__ float logits[256];
  __shared__ float probs[256];
  __shared__ float red[256];
  __shared__ uint8_t sel_mask[256];
  __shared__ int last_block;

  float acc = 0.0f;
  const uint16_t* row = weight + static_cast<int64_t>(e) * hidden_size;
  for (int64_t i = t; i < hidden_size; i += blockDim.x) {
    acc += bf16_to_float(x[i]) * bf16_to_float(row[i]);
  }
  red[t] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  if (t == 0) {
    logits_out[e] = red[0];
    __threadfence();
    const unsigned int ticket = atomicInc(
        completion_counter, static_cast<unsigned int>(num_experts - 1));
    last_block = ticket == static_cast<unsigned int>(num_experts - 1);
  }
  __syncthreads();
  if (!last_block) {
    return;
  }

  const float logit = (t < num_experts) ? logits_out[t] : -INFINITY;
  logits[t] = logit;
  red[t] = logit;
  sel_mask[t] = 0;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (t < s) red[t] = fmaxf(red[t], red[t + s]);
    __syncthreads();
  }
  const float max_logit = red[0];
  const float ex = (t < num_experts) ? expf(logits[t] - max_logit) : 0.0f;
  probs[t] = ex;
  red[t] = ex;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  const float inv_sum = 1.0f / red[0];
  if (t < num_experts) probs[t] *= inv_sum;
  __syncthreads();

  if (t < num_experts) {
    const float my = probs[t];
    int rank = 0;
    for (int e2 = 0; e2 < num_experts; ++e2) {
      const float p2 = probs[e2];
      if (p2 > my || (p2 == my && e2 < t)) ++rank;
    }
    if (rank < top_k) {
      selected_experts[rank] = t;
      sel_mask[t] = 1;
    }
  }
  __syncthreads();

  float sum = 0.0f;
  for (int e2 = 0; e2 < num_experts; ++e2) {
    if (sel_mask[e2]) sum += probs[e2];
  }
  if (t < num_experts && sel_mask[t]) {
    const float my = probs[t];
    int rank = 0;
    for (int e2 = 0; e2 < num_experts; ++e2) {
      const float p2 = probs[e2];
      if (p2 > my || (p2 == my && e2 < t)) ++rank;
    }
    selected_weights[rank] = probs[t] / (sum + 1.0e-20f);
  }
}

// Batched router (P3 prefill): same two-kernel design as the c1 parallel router,
// but with a row dimension over T prompt rows.  logits_out is [T, num_experts];
// selected_experts / selected_weights are [T, top_k].
__global__ void maple_router_logits_batched_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ weight,
    float* __restrict__ logits_out,
    int64_t hidden_size,
    int64_t num_experts) {
  const int e = static_cast<int>(blockIdx.x);
  const int64_t row = blockIdx.y;
  const int t = static_cast<int>(threadIdx.x);
  float acc = 0.0f;
  const uint16_t* xrow = x + row * hidden_size;
  const uint16_t* wrow = weight + static_cast<int64_t>(e) * hidden_size;
  for (int64_t i = t; i < hidden_size; i += blockDim.x) {
    acc += bf16_to_float(xrow[i]) * bf16_to_float(wrow[i]);
  }
  __shared__ float red[256];
  red[t] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  if (t == 0 && e < num_experts) logits_out[row * num_experts + e] = red[0];
}

__global__ void maple_router_softmax_topk_batched_kernel(
    const float* __restrict__ logits_in,
    int32_t* __restrict__ selected_experts,
    float* __restrict__ selected_weights,
    int64_t num_experts,
    int64_t top_k) {
  const int64_t row = blockIdx.x;
  const int t = static_cast<int>(threadIdx.x);
  __shared__ float logits[256];
  __shared__ float probs[256];
  __shared__ float red[256];
  __shared__ uint8_t sel_mask[256];

  const float* lrow = logits_in + row * num_experts;
  const float logit = (t < num_experts) ? lrow[t] : -INFINITY;
  logits[t] = logit;
  red[t] = logit;
  sel_mask[t] = 0;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (t < s) red[t] = fmaxf(red[t], red[t + s]);
    __syncthreads();
  }
  const float max_logit = red[0];
  const float ex = (t < num_experts) ? expf(logits[t] - max_logit) : 0.0f;
  probs[t] = ex;
  red[t] = ex;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  const float inv_sum = 1.0f / red[0];
  if (t < num_experts) probs[t] *= inv_sum;
  __syncthreads();

  if (t < num_experts) {
    const float my = probs[t];
    int rank = 0;
    for (int e2 = 0; e2 < num_experts; ++e2) {
      const float p2 = probs[e2];
      if (p2 > my || (p2 == my && e2 < t)) ++rank;
    }
    if (rank < top_k) {
      selected_experts[row * top_k + rank] = t;
      sel_mask[t] = 1;
    }
  }
  __syncthreads();

  float sum = 0.0f;
  for (int e2 = 0; e2 < num_experts; ++e2) {
    if (sel_mask[e2]) sum += probs[e2];
  }
  if (t < num_experts && sel_mask[t]) {
    const float my = probs[t];
    int rank = 0;
    for (int e2 = 0; e2 < num_experts; ++e2) {
      const float p2 = probs[e2];
      if (p2 > my || (p2 == my && e2 < t)) ++rank;
    }
    selected_weights[row * top_k + rank] = probs[t] / (sum + 1.0e-20f);
  }
}

__global__ void maple_clamped_swiglu_kernel(
    const uint16_t* __restrict__ gate,
    const uint16_t* __restrict__ up,
    uint16_t* __restrict__ out,
    int64_t elements) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < elements;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float gate_value = fminf(bf16_to_float(gate[idx]), 7.0f);
    const float up_value = fminf(fmaxf(bf16_to_float(up[idx]), -7.0f), 7.0f);
    const float silu = gate_value / (1.0f + expf(-gate_value));
    out[idx] = float_to_bf16(silu * up_value);
  }
}

__global__ void maple_weighted_residual_kernel(
    const uint16_t* __restrict__ residual,
    const uint16_t* __restrict__ expert_outputs,
    const float* __restrict__ routing_weights,
    uint16_t* __restrict__ out,
    int64_t top_k,
    int64_t hidden_size) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < hidden_size;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float combined = 0.0f;
    for (int64_t route = 0; route < top_k; ++route) {
      combined += routing_weights[route]
          * bf16_to_float(expert_outputs[route * hidden_size + idx]);
    }
    const uint16_t combined_bits = float_to_bf16(combined);
    out[idx] = float_to_bf16(
        bf16_to_float(residual[idx]) + bf16_to_float(combined_bits));
  }
}

// Batched weighted residual (P3 prefill): combines T rows' top-k expert
// outputs into the layer residual.  residual/expert_outputs/out are
// [T, hidden_size]; routing_weights is [T, top_k].  Same rounding order as
// the c1 kernel (round combined to bf16, then add residual).
__global__ void maple_weighted_residual_batched_kernel(
    const uint16_t* __restrict__ residual,
    const uint16_t* __restrict__ expert_outputs,
    const float* __restrict__ routing_weights,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t top_k,
    int64_t hidden_size) {
  const int64_t total = rows * hidden_size;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = idx / hidden_size;
    const int64_t hid = idx - row * hidden_size;
    float combined = 0.0f;
    const float* wrow = routing_weights + row * top_k;
    const uint16_t* erow = expert_outputs + row * top_k * hidden_size;
    for (int64_t route = 0; route < top_k; ++route) {
      combined += wrow[route] * bf16_to_float(erow[route * hidden_size + hid]);
    }
    const uint16_t combined_bits = float_to_bf16(combined);
    out[idx] = float_to_bf16(
        bf16_to_float(residual[idx]) + bf16_to_float(combined_bits));
  }
}

}  // namespace

extern "C" int hipengine_maple_router_topk_bf16(
    const uint16_t* x,
    const uint16_t* weight,
    int32_t* selected_experts,
    float* selected_weights,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k,
    hipStream_t stream) {
  if (hidden_size <= 0 || num_experts <= 0 || num_experts > 256
      || top_k <= 0 || top_k > 16 || top_k > num_experts) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      maple_router_topk_kernel,
      dim3(1),
      dim3(256),
      0,
      stream,
      x,
      weight,
      selected_experts,
      selected_weights,
      hidden_size,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_router_topk_parallel_bf16(
    const uint16_t* x,
    const uint16_t* weight,
    int32_t* selected_experts,
    float* selected_weights,
    float* logits_scratch,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k,
    hipStream_t stream) {
  if (hidden_size <= 0 || num_experts <= 0 || num_experts > 256
      || top_k <= 0 || top_k > 16 || top_k > num_experts) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      maple_router_logits_kernel,
      dim3(static_cast<unsigned int>(num_experts)),
      dim3(256),
      0,
      stream,
      x,
      weight,
      logits_scratch,
      hidden_size,
      num_experts);
  hipLaunchKernelGGL(
      maple_router_softmax_topk_kernel,
      dim3(1),
      dim3(256),
      0,
      stream,
      logits_scratch,
      selected_experts,
      selected_weights,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_router_topk_single_dispatch_bf16(
    const uint16_t* x,
    const uint16_t* weight,
    int32_t* selected_experts,
    float* selected_weights,
    float* logits_scratch,
    unsigned int* completion_counter,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k,
    hipStream_t stream) {
  if (hidden_size <= 0 || num_experts <= 0 || num_experts > 256
      || top_k <= 0 || top_k > 16 || top_k > num_experts
      || completion_counter == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      maple_router_topk_single_dispatch_kernel,
      dim3(static_cast<unsigned int>(num_experts)),
      dim3(256),
      0,
      stream,
      x,
      weight,
      selected_experts,
      selected_weights,
      logits_scratch,
      completion_counter,
      hidden_size,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_router_topk_parallel_batched_bf16(
    const uint16_t* x,
    const uint16_t* weight,
    int32_t* selected_experts,
    float* selected_weights,
    float* logits_scratch,
    int64_t rows,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || num_experts <= 0 || num_experts > 256
      || top_k <= 0 || top_k > 16 || top_k > num_experts) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      maple_router_logits_batched_kernel,
      dim3(static_cast<unsigned int>(num_experts),
           static_cast<unsigned int>(rows)),
      dim3(256),
      0,
      stream,
      x,
      weight,
      logits_scratch,
      hidden_size,
      num_experts);
  hipLaunchKernelGGL(
      maple_router_softmax_topk_batched_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(256),
      0,
      stream,
      logits_scratch,
      selected_experts,
      selected_weights,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_clamped_swiglu_bf16(
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* out,
    int64_t rows,
    int64_t features,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t elements = rows * features;
  const int blocks = static_cast<int>((elements + 255) / 256);
  hipLaunchKernelGGL(
      maple_clamped_swiglu_kernel,
      dim3(blocks),
      dim3(256),
      0,
      stream,
      gate,
      up,
      out,
      elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_weighted_residual_bf16(
    const uint16_t* residual,
    const uint16_t* expert_outputs,
    const float* routing_weights,
    uint16_t* out,
    int64_t top_k,
    int64_t hidden_size,
    hipStream_t stream) {
  if (top_k <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int blocks = static_cast<int>((hidden_size + 255) / 256);
  hipLaunchKernelGGL(
      maple_weighted_residual_kernel,
      dim3(blocks),
      dim3(256),
      0,
      stream,
      residual,
      expert_outputs,
      routing_weights,
      out,
      top_k,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_weighted_residual_batched_bf16(
    const uint16_t* residual,
    const uint16_t* expert_outputs,
    const float* routing_weights,
    uint16_t* out,
    int64_t rows,
    int64_t top_k,
    int64_t hidden_size,
    hipStream_t stream) {
  if (rows <= 0 || top_k <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * hidden_size;
  const int blocks = static_cast<int>((total + 255) / 256);
  hipLaunchKernelGGL(
      maple_weighted_residual_batched_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(256),
      0,
      stream,
      residual,
      expert_outputs,
      routing_weights,
      out,
      rows,
      top_k,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}
