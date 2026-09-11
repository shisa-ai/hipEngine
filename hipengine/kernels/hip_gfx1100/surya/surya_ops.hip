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
//   scores layout: (heads, tokens, head_stride) row-major; entry
//   (h, i, j) with j > i becomes -inf, everything else is scaled.
__global__ void surya_causal_mask_scale_kernel(
    float* __restrict__ scores,
    float scale,
    int64_t heads,
    int64_t tokens,
    int64_t head_stride) {
  const int64_t row = blockIdx.x;
  if (row >= heads * tokens) {
    return;
  }
  const int64_t head = row / tokens;
  const int64_t query = row % tokens;
  float* sr = scores + head * tokens * head_stride + query * head_stride;
  for (int64_t j = threadIdx.x; j < head_stride; j += blockDim.x) {
    sr[j] = (j <= query) ? sr[j] * scale : -INFINITY;
  }
}

extern "C" int hipengine_surya_causal_mask_scale_f32(
    float* scores,
    float scale,
    int64_t heads,
    int64_t tokens,
    int64_t head_stride,
    hipStream_t stream) {
  if (heads <= 0 || tokens <= 0 || head_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      surya_causal_mask_scale_kernel,
      dim3(static_cast<unsigned int>(heads * tokens)), dim3(256), 0, stream,
      scores, scale, heads, tokens, head_stride);
  return static_cast<int>(hipGetLastError());
}
