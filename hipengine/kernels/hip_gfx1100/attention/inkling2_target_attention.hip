#include <hip/hip_runtime.h>
#include <stdint.h>

__global__ void inkling2_target_attention(const float* q, const float* k, const float* v,
                                           const float* bias, const float* sink, float* out,
                                           const int32_t* offsets, const int32_t* count,
                                           const int32_t* positions, const uint8_t* evicted,
                                           const int32_t* row, int nq, int nk, int dim, int extent) {
    constexpr int tile = 128;
    const int lane = threadIdx.x;
    const int head = blockIdx.x;
    const int kh = head / (nq / nk);
    const float query = lane < dim ? q[head * dim + lane] : 0.0f;
    __shared__ float reduce[tile];
    __shared__ float factors[2];
    float acc = 0.0f;
    float max_score = sink[head];
    float denom = 1.0f;
    for (int i = 0; i < *count; ++i) {
        const int lag = *row - positions[i];
        if (evicted[i] || lag < 0) continue;
        const size_t base = (size_t(offsets[i]) * nk + kh) * dim;
        reduce[lane] = lane < dim ? query * k[base + lane] : 0.0f;
        __syncthreads();
        for (int stride = tile / 2; stride > 0; stride /= 2) {
            if (lane < stride) reduce[lane] += reduce[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            const float score = reduce[0] * rsqrtf(float(dim)) + (lag < extent ? bias[head * extent + lag] : 0.0f);
            const float next = fmaxf(max_score, score);
            factors[0] = expf(max_score - next);
            factors[1] = expf(score - next);
            denom = denom * factors[0] + factors[1];
            max_score = next;
        }
        __syncthreads();
        if (lane < dim) acc = acc * factors[0] + factors[1] * v[base + lane];
        __syncthreads();
    }
    if (lane == 0) factors[0] = denom;
    __syncthreads();
    if (lane < dim) out[head * dim + lane] = acc / factors[0];
}

extern "C" int hipengine_inkling2_target_attention(const float* q, const float* k, const float* v,
                                                   const float* bias, const float* sink, float* out,
                                                   const int32_t* offsets, const int32_t* count,
                                                   const int32_t* positions, const uint8_t* evicted,
                                                   const int32_t* row, int nq, int nk, int dim, int extent) {
    if (nq <= 0 || nk <= 0 || nq % nk || dim <= 0 || dim > 128 || extent <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_target_attention, dim3(nq), dim3(128), 0, 0,
                       q, k, v, bias, sink, out, offsets, count, positions, evicted, row, nq, nk, dim, extent);
    return hipGetLastError();
}
