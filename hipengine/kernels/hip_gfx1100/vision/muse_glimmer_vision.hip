// Muse Glimmer vision primitives. Equations follow clip_graph_muse_glimmer
// (llama.cpp mtmd); all operands are raw FP32 device pointers. Linear and
// attention contractions use rocBLAS. Registered for both gfx11 backends.
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// LayerNorm with weight and bias: (x - mean) / sqrt(var + eps) * w + b,
// population variance, one block per row.
__global__ void muse_glimmer_vision_layernorm_kernel(const float* x, const float* w,
                                                     const float* b, float* y,
                                                     int width, float eps) {
    __shared__ float partial[256];
    __shared__ float mean, inv;
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * width;
    float sum = 0;
    for (int c = threadIdx.x; c < width; c += blockDim.x) sum += xr[c];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (!threadIdx.x) mean = partial[0] / width;
    __syncthreads();
    float var = 0;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        float d = xr[c] - mean;
        var += d * d;
    }
    partial[threadIdx.x] = var;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (!threadIdx.x) inv = rsqrtf(partial[0] / width + eps);
    __syncthreads();
    for (int c = threadIdx.x; c < width; c += blockDim.x)
        y[(size_t)row * width + c] = (xr[c] - mean) * inv * w[c] + b[c];
}

// op 0: y = x + b (elementwise residual); op 1: exact-erf GELU;
// op 2: y = x * lo (scalar scale).
__global__ void muse_glimmer_vision_elementwise_kernel(const float* x, const float* b,
                                                       float* y, int n, int op, float lo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (op == 0) v += b[i];
    if (op == 1) v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    if (op == 2) v *= lo;
    y[i] = v;
}

// Per-channel bias broadcast over rows: y[r, c] = x[r, c] + b[c].
__global__ void muse_glimmer_vision_bias_add_kernel(const float* x, const float* b,
                                                    float* y, int width, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] + b[i % width];
}

// 2D interleaved RoPE, token-major [tokens, heads, dim]. Dims [0, dim/2) rotate
// by pos_w, [dim/2, dim) by pos_h; each half has dim/4 adjacent pairs with
// inv_freq = theta^(-2i/(dim/2)). Positions are 1-indexed int32 per token.
__global__ void muse_glimmer_vision_rope2d_kernel(const float* x, const int32_t* pos_w,
                                                  const int32_t* pos_h, float* y,
                                                  int tokens, int heads, int dim,
                                                  float theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = dim / 2;
    int n = tokens * heads * dim;
    if (i >= n) return;
    int c = i % dim, h = (i / dim) % heads, t = i / (dim * heads);
    int axis = c / half, local = c % half;
    int pair = local / 2;
    float pos = axis == 0 ? (float)pos_w[t] : (float)pos_h[t];
    float angle = pos * powf(theta, -2.0f * pair / half);
    float cs = cosf(angle), sn = sinf(angle);
    int base = (t * heads + h) * dim + axis * half;
    float a = x[base + 2 * pair], b = x[base + 2 * pair + 1];
    y[i] = (local % 2 == 0) ? a * cs - b * sn : a * sn + b * cs;
}

// Token-major [tokens, heads, dim] <-> head-major [heads, tokens, dim].
// dir 0: token->head; dir 1: head->token.
__global__ void muse_glimmer_vision_transpose_kernel(const float* x, float* y,
                                                     int tokens, int heads, int dim,
                                                     int dir) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = tokens * heads * dim;
    if (i >= n) return;
    int c = i % dim;
    if (dir == 0) {
        // output head-major [heads, tokens, dim]
        int t = (i / dim) % tokens, h = i / (dim * tokens);
        y[i] = x[(t * heads + h) * dim + c];
    } else {
        // output token-major [tokens, heads, dim]
        int h = (i / dim) % heads, t = i / (dim * heads);
        y[i] = x[(h * tokens + t) * dim + c];
    }
}

// In-place softmax over the last dim of x [rows, cols], with an additive
// mask [nq, cols] broadcast across heads (row r reads mask row r % nq).
// mask == nullptr is the plain (global-layer) softmax.
__global__ void muse_glimmer_vision_softmax_kernel(float* x, const float* mask,
                                                   int cols, int nq) {
    __shared__ float reduce[256];
    __shared__ float maximum, inverse;
    int row = blockIdx.x;
    const float* mr = mask ? mask + (size_t)(row % nq) * cols : nullptr;
    float* xr = x + (size_t)row * cols;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        m = fmaxf(m, xr[c] + (mr ? mr[c] : 0.0f));
    reduce[threadIdx.x] = m;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (threadIdx.x < s) reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + s]);
        __syncthreads();
    }
    if (!threadIdx.x) maximum = reduce[0];
    __syncthreads();
    float sum = 0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        sum += expf(xr[c] + (mr ? mr[c] : 0.0f) - maximum);
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (threadIdx.x < s) reduce[threadIdx.x] += reduce[threadIdx.x + s];
        __syncthreads();
    }
    if (!threadIdx.x) inverse = 1.0f / reduce[0];
    __syncthreads();
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        xr[c] = expf(xr[c] + (mr ? mr[c] : 0.0f) - maximum) * inverse;
}

// Row gather: y[i, :] = x[perm[i], :].
__global__ void muse_glimmer_vision_gather_kernel(const float* x, const int32_t* perm,
                                                  float* y, int width, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = i / width, c = i % width;
    y[i] = x[(size_t)perm[r] * width + c];
}

// 2x2 (merge x merge) pixel-shuffle, channel-outer:
// y[o, c*f*f + s] = x[src(o, s), c] with src the raster f*f neighborhood.
__global__ void muse_glimmer_vision_shuffle_kernel(const float* x, float* y,
                                                   int gh, int gw, int hidden, int f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = gw / f, oh = gh / f;
    int n = oh * ow * hidden * f * f;
    if (i >= n) return;
    int s = i % (f * f), c = (i / (f * f)) % hidden, o = i / (f * f * hidden);
    int oy = o / ow, ox = o % ow, ry = s / f, rx = s % f;
    int src = (oy * f + ry) * gw + (ox * f + rx);
    y[i] = x[(size_t)src * hidden + c];
}

extern "C" int hipengine_muse_glimmer_vision_layernorm(const float* x, const float* w,
    const float* b, float* y, int rows, int width, float eps, hipStream_t stream) {
    if (!x || !w || !b || !y || rows <= 0 || width <= 0 || eps <= 0) return hipErrorInvalidValue;
    muse_glimmer_vision_layernorm_kernel<<<rows, 256, 0, stream>>>(x, w, b, y, width, eps);
    return hipGetLastError();
}
extern "C" int hipengine_muse_glimmer_vision_elementwise(const float* x, const float* b,
    float* y, int n, int op, float lo, hipStream_t stream) {
    if (!x || !y || n <= 0 || op < 0 || op > 2 || (op == 0 && !b)) return hipErrorInvalidValue;
    muse_glimmer_vision_elementwise_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, b, y, n, op, lo);
    return hipGetLastError();
}
extern "C" int hipengine_muse_glimmer_vision_bias_add(const float* x, const float* b,
    float* y, int rows, int width, hipStream_t stream) {
    if (!x || !b || !y || rows <= 0 || width <= 0) return hipErrorInvalidValue;
    int n = rows * width;
    muse_glimmer_vision_bias_add_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, b, y, width, n);
    return hipGetLastError();
}
extern "C" int hipengine_muse_glimmer_vision_rope2d(const float* x, const int32_t* pw,
    const int32_t* ph, float* y, int tokens, int heads, int dim, float theta, hipStream_t stream) {
    if (!x || !pw || !ph || !y || tokens <= 0 || heads <= 0 || dim <= 0 || dim % 4 || theta <= 0)
        return hipErrorInvalidValue;
    int n = tokens * heads * dim;
    muse_glimmer_vision_rope2d_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, pw, ph, y, tokens, heads, dim, theta);
    return hipGetLastError();
}
extern "C" int hipengine_muse_glimmer_vision_transpose(const float* x, float* y,
    int tokens, int heads, int dim, int dir, hipStream_t stream) {
    if (!x || !y || tokens <= 0 || heads <= 0 || dim <= 0 || dir < 0 || dir > 1)
        return hipErrorInvalidValue;
    int n = tokens * heads * dim;
    muse_glimmer_vision_transpose_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, tokens, heads, dim, dir);
    return hipGetLastError();
}
extern "C" int hipengine_muse_glimmer_vision_softmax(float* x, const float* mask,
    int rows, int cols, int nq, hipStream_t stream) {
    if (!x || rows <= 0 || cols <= 0 || nq <= 0 || rows % nq) return hipErrorInvalidValue;
    muse_glimmer_vision_softmax_kernel<<<rows, 256, 0, stream>>>(x, mask, cols, nq);
    return hipGetLastError();
}
extern "C" int hipengine_muse_glimmer_vision_gather(const float* x, const int32_t* perm,
    float* y, int rows, int width, hipStream_t stream) {
    if (!x || !perm || !y || rows <= 0 || width <= 0) return hipErrorInvalidValue;
    int n = rows * width;
    muse_glimmer_vision_gather_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, perm, y, width, n);
    return hipGetLastError();
}
extern "C" int hipengine_muse_glimmer_vision_shuffle(const float* x, float* y,
    int gh, int gw, int hidden, int f, hipStream_t stream) {
    if (!x || !y || gh <= 0 || gw <= 0 || hidden <= 0 || f <= 0 || gh % f || gw % f)
        return hipErrorInvalidValue;
    int n = (gh / f) * (gw / f) * hidden * f * f;
    muse_glimmer_vision_shuffle_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, gh, gw, hidden, f);
    return hipGetLastError();
}