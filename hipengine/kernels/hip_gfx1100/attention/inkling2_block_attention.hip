#include <hip/hip_runtime.h>

// One block per (query, query head). The caller selects the visible key block.
__global__ void inkling2_block_attention(const float* q, const float* k, const float* v,
                                          float* out, int nq, int nk, int dim, int keys) {
    constexpr int threads = 128;
    const int lane = threadIdx.x;
    const int head = blockIdx.x;
    const int kh = head / (nq / nk);
    const size_t row = (size_t(blockIdx.y) * nq + head) * dim;
    const float qi = lane < dim ? q[row + lane] : 0.0f;
    const float scale = rsqrtf(float(dim));
    __shared__ float reduce[threads], factors[2];
    float maximum = -INFINITY, denom = 0.0f, acc = 0.0f;
    for (int i = 0; i < keys; ++i) {
        const size_t base = (size_t(i) * nk + kh) * dim;
        reduce[lane] = lane < dim ? qi * k[base + lane] : 0.0f;
        __syncthreads();
        for (int stride = threads / 2; stride > 0; stride /= 2) {
            if (lane < stride) reduce[lane] += reduce[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            const float score = reduce[0] * scale;
            const float next = fmaxf(maximum, score);
            factors[0] = expf(maximum - next);
            factors[1] = expf(score - next);
            denom = denom * factors[0] + factors[1];
            maximum = next;
        }
        __syncthreads();
        if (lane < dim) acc = acc * factors[0] + factors[1] * v[base + lane];
        __syncthreads();
    }
    if (lane == 0) factors[0] = denom;
    __syncthreads();
    if (lane < dim) out[row + lane] = acc / factors[0];
}

extern "C" int hipengine_inkling2_block_attention(const float* q, const float* k, const float* v,
                                                  float* out, int queries, int keys, int nq, int nk, int dim) {
    if (!q || !k || !v || !out || queries <= 0 || queries > 65535 || keys <= 0 ||
        nq <= 0 || nk <= 0 || nq % nk || dim <= 0 || dim > 128) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_block_attention, dim3(nq, queries), dim3(128), 0, 0,
                       q, k, v, out, nq, nk, dim, keys);
    return hipGetLastError();
}
