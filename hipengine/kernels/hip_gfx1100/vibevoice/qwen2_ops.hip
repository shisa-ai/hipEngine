// VibeVoice Qwen2 backbone primitives (gfx1100).
// F16 storage with F32 accumulation, matching the CPU reference contract:
// RMSNorm, rotate-half RoPE, tiled GEMM, GQA decode attention, chunked
// causal prefill attention. Raw device pointers only; the host wrapper
// converts.
#include <stdint.h>
#include <hip/hip_runtime.h>

static __device__ float vv_f16_to_f32(unsigned short h) {
    unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1Fu;
    unsigned int man = h & 0x3FFu;
    unsigned int bits;
    if (exp == 0u) {
        if (man == 0u) {
            bits = sign;
        } else {
            // Subnormal: normalize.
            int e = -1;
            unsigned int m = man;
            do { m <<= 1; e++; } while ((m & 0x400u) == 0u);
            m &= 0x3FFu;
            bits = sign | ((unsigned int)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    return __int_as_float((int)bits);
}

static __device__ unsigned short vv_f32_to_f16(float f) {
    unsigned int x = (unsigned int)__float_as_int(f);
    unsigned int sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xFFu) - 127 + 15;
    unsigned int man = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) {  // inf/nan
        return (unsigned short)(sign | 0x7C00u | (man ? 0x200u : 0u));
    }
    if (exp >= 31) return (unsigned short)(sign | 0x7C00u);   // overflow -> inf
    if (exp <= 0) {                                            // subnormal/zero
        if (exp < -10) return (unsigned short)sign;
        man |= 0x800000u;
        unsigned int shift = (unsigned int)(1 - exp);
        unsigned int man16 = man >> (13 + shift);
        unsigned int rem = man & ((1u << (13 + shift)) - 1u);
        unsigned int half = (1u << (12 + shift));
        if (rem > half || (rem == half && (man16 & 1u))) man16++;
        return (unsigned short)(sign | man16);
    }
    unsigned int man16 = man >> 13;
    unsigned int rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (man16 & 1u))) man16++;
    if (man16 == 0x800u) { man16 = 0; exp++; }
    return (unsigned short)(sign | ((unsigned int)exp << 10) | man16);
}

__global__ void vv_qwen2_rmsnorm_f16_k(
    const unsigned short* __restrict__ x,   // [rows, hidden]
    const float* __restrict__ weight,       // [hidden]
    unsigned short* __restrict__ out,       // [rows, hidden]
    int hidden, float eps) {
    int row = blockIdx.x;
    const unsigned short* xr = x + (size_t)row * hidden;
    unsigned short* orow = out + (size_t)row * hidden;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = vv_f16_to_f32(xr[i]);
        sum += v * v;
    }
    __shared__ float red[1024];
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)hidden + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        orow[i] = vv_f32_to_f16(vv_f16_to_f32(xr[i]) * inv * weight[i]);
    }
}

__global__ void vv_qwen2_rope_f16_k(
    unsigned short* __restrict__ x,     // [tokens, heads, head_dim] in place
    const float* __restrict__ cos_t,    // [tokens, head_dim] (halves duplicated)
    const float* __restrict__ sin_t,
    int heads, int head_dim) {
    int token = blockIdx.y;
    int head = blockIdx.x;
    int half = head_dim / 2;
    unsigned short* row = x + ((size_t)token * heads + head) * head_dim;
    int i = threadIdx.x;
    if (i < half) {
        float a = vv_f16_to_f32(row[i]);
        float b = vv_f16_to_f32(row[i + half]);
        float c = cos_t[token * head_dim + i];
        float s = sin_t[token * head_dim + i];
        row[i] = vv_f32_to_f16(a * c - b * s);
        row[i + half] = vv_f32_to_f16(b * c + a * s);
    }
}

// out[m,n] = sum_k x[m,k] * w[n,k]  (weight is [N, K], row-major), F32 accum.
__global__ void vv_qwen2_gemm_f16_k(
    const unsigned short* __restrict__ x,   // [M, K]
    const unsigned short* __restrict__ w,   // [N, K]
    const float* __restrict__ bias,         // [N] or null
    unsigned short* __restrict__ out,       // [M, N]
    int m, int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    const unsigned short* xr = x + (size_t)row * k;
    const unsigned short* wr = w + (size_t)col * k;
    float acc = 0.0f;
    for (int i = 0; i < k; i++) {
        acc += vv_f16_to_f32(xr[i]) * vv_f16_to_f32(wr[i]);
    }
    if (bias) acc += bias[col];
    out[(size_t)row * n + col] = vv_f32_to_f16(acc);
}

// Single-token decode attention with GQA grouping.
// q: [q_heads, head_dim] f16; k/v caches: [capacity, kv_heads, head_dim] f16;
// attends positions 0..pos inclusive; out: [q_heads, head_dim] f16.
__global__ void vv_qwen2_decode_attn_f16_k(
    const unsigned short* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    unsigned short* __restrict__ out,
    int kv_heads, int group, int head_dim, int capacity, int pos,
    float scale) {
    int qh = blockIdx.x;                  // one block per query head
    (void)cos_t; (void)sin_t;
    int kvh = qh / group;
    __shared__ float sq[128];
    __shared__ float smax[1];
    __shared__ float ssum[1];
    // q arrives pre-roped (rotate-half applied by the host rope kernel).
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) sq[i] = vv_f16_to_f32(q[qh * head_dim + i]);
    __syncthreads();
    // Pass 1: scores + max.
    float local_max = -1e30f;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        const unsigned short* kr = k_cache + ((size_t)p * kv_heads + kvh) * head_dim;
        float acc = 0.0f;
        for (int i = 0; i < head_dim; i++) acc += sq[i] * vv_f16_to_f32(kr[i]);
        float score = acc * scale;
        if (score > local_max) local_max = score;
    }
    __shared__ float red[1024];
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && red[threadIdx.x + s] > red[threadIdx.x]) red[threadIdx.x] = red[threadIdx.x + s];
        __syncthreads();
    }
    smax[0] = red[0];
    __syncthreads();
    // Pass 2: sum of exp.
    float local_sum = 0.0f;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        const unsigned short* kr = k_cache + ((size_t)p * kv_heads + kvh) * head_dim;
        float acc = 0.0f;
        for (int i = 0; i < head_dim; i++) acc += sq[i] * vv_f16_to_f32(kr[i]);
        local_sum += __expf(acc * scale - smax[0]);
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    ssum[0] = red[0];
    __syncthreads();
    // Pass 3: weighted value sum, only head-dim threads write.
    if (threadIdx.x < head_dim) {
        float acc = 0.0f;
        for (int p = 0; p <= pos; p++) {
            const unsigned short* vr = v_cache + ((size_t)p * kv_heads + kvh) * head_dim;
            const unsigned short* kr = k_cache + ((size_t)p * kv_heads + kvh) * head_dim;
            float d = 0.0f;
            for (int i = 0; i < head_dim; i++) d += sq[i] * vv_f16_to_f32(kr[i]);
            acc += __expf(d * scale - smax[0]) * vv_f16_to_f32(vr[threadIdx.x]);
        }
        out[qh * head_dim + threadIdx.x] = vv_f32_to_f16(acc / ssum[0]);
    }
}

// Chunked causal prefill with GQA over a cache that already holds `past`
// positions; new k/v are NOT yet in the cache (passed separately, pre-rope).
// q: [T, q_heads, head_dim]; k_new/v_new: [T, kv_heads, head_dim];
// cache: [capacity, kv_heads, head_dim]; out: [T, q_heads, head_dim].
__global__ void vv_qwen2_prefill_attn_f16_k(
    const unsigned short* __restrict__ q,
    const unsigned short* __restrict__ k_new,
    const unsigned short* __restrict__ v_new,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    unsigned short* __restrict__ out,
    int kv_heads, int group, int head_dim, int capacity, int past, int tokens,
    float scale) {
    int qh = blockIdx.x;
    int t = blockIdx.y;
    int half = head_dim / 2;
    int kvh = qh / group;
    int total = past + tokens;
    __shared__ float sq[128];
    __shared__ float red[1024];
    // q arrives pre-roped (rotate-half applied by the host rope kernel).
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) sq[i] = vv_f16_to_f32(q[((size_t)t * gridDim.x + qh) * head_dim + i]);
    __syncthreads();
    // Pass 1: max over allowed positions (past + [0..t]).
    float local_max = -1e30f;
    for (int p = threadIdx.x; p < past + t + 1; p += blockDim.x) {
        const unsigned short* kr;
        if (p < past) {
            kr = k_cache + ((size_t)p * kv_heads + kvh) * head_dim;
        } else {
            // New token (p - past) has its RoPE already applied per token t.
            kr = k_new + ((size_t)(p - past) * kv_heads + kvh) * head_dim;
        }
        float acc = 0.0f;
        for (int i = 0; i < head_dim; i++) acc += sq[i] * vv_f16_to_f32(kr[i]);
        if (acc * scale > local_max) local_max = acc * scale;
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && red[threadIdx.x + s] > red[threadIdx.x]) red[threadIdx.x] = red[threadIdx.x + s];
        __syncthreads();
    }
    __shared__ float smax[1];
    if (threadIdx.x == 0) smax[0] = red[0];
    __syncthreads();
    // Pass 2: denominator.
    float local_sum = 0.0f;
    for (int p = threadIdx.x; p < past + t + 1; p += blockDim.x) {
        const unsigned short* kr = (p < past)
            ? k_cache + ((size_t)p * kv_heads + kvh) * head_dim
            : k_new + ((size_t)(p - past) * kv_heads + kvh) * head_dim;
        float acc = 0.0f;
        for (int i = 0; i < head_dim; i++) acc += sq[i] * vv_f16_to_f32(kr[i]);
        local_sum += __expf(acc * scale - smax[0]);
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    __shared__ float ssum[1];
    if (threadIdx.x == 0) ssum[0] = red[0];
    __syncthreads();
    // Pass 3: context.
    if (threadIdx.x < head_dim) {
        float acc = 0.0f;
        for (int p = 0; p < past + t + 1; p++) {
            const unsigned short* kr = (p < past)
                ? k_cache + ((size_t)p * kv_heads + kvh) * head_dim
                : k_new + ((size_t)(p - past) * kv_heads + kvh) * head_dim;
            const unsigned short* vr = (p < past)
                ? v_cache + ((size_t)p * kv_heads + kvh) * head_dim
                : v_new + ((size_t)(p - past) * kv_heads + kvh) * head_dim;
            float d = 0.0f;
            for (int i = 0; i < head_dim; i++) d += sq[i] * vv_f16_to_f32(kr[i]);
            acc += __expf(d * scale - smax[0]) * vv_f16_to_f32(vr[threadIdx.x]);
        }
        out[((size_t)t * gridDim.x + qh) * head_dim + threadIdx.x] = vv_f32_to_f16(acc / ssum[0]);
    }
}

// Append one token's rope-applied k/v to the caches.
__global__ void vv_qwen2_append_kv_f16_k(
    const unsigned short* __restrict__ k_rope,  // [tokens, kv_heads, head_dim]
    const unsigned short* __restrict__ v,       // [tokens, kv_heads, head_dim]
    unsigned short* __restrict__ k_cache,       // [capacity, kv_heads, head_dim]
    unsigned short* __restrict__ v_cache,
    int kv_heads, int head_dim, int start, int tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * kv_heads * head_dim;
    if (idx >= total) return;
    int t = idx / (kv_heads * head_dim);
    int rest = idx % (kv_heads * head_dim);
    k_cache[((size_t)(start + t) * kv_heads * head_dim) + rest] = k_rope[idx];
    v_cache[((size_t)(start + t) * kv_heads * head_dim) + rest] = v[idx];
}

__global__ void vv_qwen2_bias_add_f16_k(
    unsigned short* __restrict__ out, const float* __restrict__ bias,
    int64_t n, int64_t total) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int64_t col = i % n;
  out[i] = vv_f32_to_f16(vv_f16_to_f32(out[i]) + bias[col]);
}

extern "C" {

hipError_t vv_qwen2_rmsnorm_f16(
    const unsigned short* x, const float* w, unsigned short* out,
    int64_t rows, int64_t hidden, float eps, hipStream_t stream) {
  (void)rows;
  hipLaunchKernelGGL(vv_qwen2_rmsnorm_f16_k, dim3(static_cast<unsigned>(rows)),
                     dim3(256), 0, stream, x, w, out, static_cast<int>(hidden), eps);
  return hipGetLastError();
}

hipError_t vv_qwen2_rope_f16(
    unsigned short* x, const float* cos_t, const float* sin_t,
    int64_t tokens, int64_t heads, int64_t head_dim, hipStream_t stream) {
  dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(tokens));
  hipLaunchKernelGGL(vv_qwen2_rope_f16_k, grid, dim3(64), 0, stream,
                     x, cos_t, sin_t, static_cast<int>(heads), static_cast<int>(head_dim));
  return hipGetLastError();
}

hipError_t vv_qwen2_gemm_f16(
    const unsigned short* x, const unsigned short* w, const float* bias,
    unsigned short* out, int64_t m, int64_t k, int64_t n, hipStream_t stream) {
  dim3 block(16, 16);
  dim3 grid(static_cast<unsigned>((n + 15) / 16), static_cast<unsigned>((m + 15) / 16));
  hipLaunchKernelGGL(vv_qwen2_gemm_f16_k, grid, block, 0, stream,
                     x, w, bias, out, static_cast<int>(m), static_cast<int>(k), static_cast<int>(n));
  return hipGetLastError();
}

hipError_t vv_qwen2_decode_attn_f16(
    const unsigned short* q, const unsigned short* kc, const unsigned short* vc,
    const float* cos_t, const float* sin_t, unsigned short* out,
    int64_t q_heads, int64_t kv_heads, int64_t head_dim, int64_t capacity,
    int64_t pos, float scale, hipStream_t stream) {
  (void)capacity;
  hipLaunchKernelGGL(vv_qwen2_decode_attn_f16_k, dim3(static_cast<unsigned>(q_heads)),
                     dim3(256), 0, stream, q, kc, vc, cos_t, sin_t, out,
                     static_cast<int>(kv_heads),
                     static_cast<int>(q_heads / kv_heads), static_cast<int>(head_dim),
                     static_cast<int>(capacity), static_cast<int>(pos), scale);
  return hipGetLastError();
}

hipError_t vv_qwen2_prefill_attn_f16(
    const unsigned short* q, const unsigned short* kn, const unsigned short* vn,
    const unsigned short* kc, const unsigned short* vc,
    const float* cos_t, const float* sin_t, unsigned short* out,
    int64_t q_heads, int64_t kv_heads, int64_t head_dim, int64_t capacity,
    int64_t past, int64_t tokens, float scale, hipStream_t stream) {
  (void)capacity;
  dim3 grid(static_cast<unsigned>(q_heads), static_cast<unsigned>(tokens));
  hipLaunchKernelGGL(vv_qwen2_prefill_attn_f16_k, grid, dim3(256), 0, stream,
                     q, kn, vn, kc, vc, cos_t, sin_t, out,
                     static_cast<int>(kv_heads),
                     static_cast<int>(q_heads / kv_heads), static_cast<int>(head_dim),
                     static_cast<int>(capacity), static_cast<int>(past),
                     static_cast<int>(tokens), scale);
  return hipGetLastError();
}

hipError_t vv_qwen2_append_kv_f16(
    const unsigned short* k, const unsigned short* v, unsigned short* kc,
    unsigned short* vc, int64_t kv_heads, int64_t head_dim, int64_t start,
    int64_t tokens, hipStream_t stream) {
  const int64_t total = tokens * kv_heads * head_dim;
  if (total <= 0) return hipSuccess;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(vv_qwen2_append_kv_f16_k, dim3(blocks), dim3(threads), 0, stream,
                     k, v, kc, vc, static_cast<int>(kv_heads), static_cast<int>(head_dim),
                     static_cast<int>(start), static_cast<int>(tokens));
  return hipGetLastError();
}

hipError_t vv_qwen2_bias_add_f16(
    unsigned short* out, const float* bias, int64_t n, int64_t total,
    hipStream_t stream) {
  if (total <= 0) return hipSuccess;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(vv_qwen2_bias_add_f16_k, dim3(blocks), dim3(threads), 0,
                     stream, out, bias, n, total);
  return hipGetLastError();
}

}  // extern "C"

// out = a + b elementwise (f16 storage, f32 arithmetic).
__global__ void vv_qwen2_add_f16_k(
    const unsigned short* __restrict__ a, const unsigned short* __restrict__ b,
    unsigned short* __restrict__ out, int64_t total) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  out[i] = vv_f32_to_f16(vv_f16_to_f32(a[i]) + vv_f16_to_f32(b[i]));
}

// gate = silu(gate) * up elementwise.
__global__ void vv_qwen2_silu_mul_f16_k(
    unsigned short* __restrict__ gate, const unsigned short* __restrict__ up,
    int64_t total) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float g = vv_f16_to_f32(gate[i]);
  float u = vv_f16_to_f32(up[i]);
  gate[i] = vv_f32_to_f16(g / (1.0f + __expf(-g)) * u);
}


// out[row, n] += bias[n] (f16 storage, f32 arithmetic).

extern "C" {

hipError_t vv_qwen2_add_f16(
    const unsigned short* a, const unsigned short* b, unsigned short* out,
    int64_t total, hipStream_t stream) {
  if (total <= 0) return hipSuccess;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(vv_qwen2_add_f16_k, dim3(blocks), dim3(threads), 0, stream,
                     a, b, out, total);
  return hipGetLastError();
}

hipError_t vv_qwen2_silu_mul_f16(
    unsigned short* gate, const unsigned short* up, int64_t total,
    hipStream_t stream) {
  if (total <= 0) return hipSuccess;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(vv_qwen2_silu_mul_f16_k, dim3(blocks), dim3(threads), 0,
                     stream, gate, up, total);
  return hipGetLastError();
}

}  // extern "C"
