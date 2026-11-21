#include <hip/hip_runtime.h>
#include <stdint.h>

// Each wave computes one score with the same 128-lane tree as the tiled kernel.
__global__ void split_scores(const float* q, const float* k, const float* bias, float* scores,
                             const int32_t* offsets, const int32_t* count, const int32_t* positions,
                             const uint8_t* evicted, const int32_t* row, int nq, int nk, int dim,
                             int extent, int stride) {
    const int lane = threadIdx.x % 32;
    const int i = blockIdx.x * 4 + threadIdx.x / 32;
    const int head = blockIdx.y;
    if (i >= *count) return;
    const int lag = *row - positions[i];
    if (evicted[i] || lag < 0) return;
    const int kh = head / (nq / nk);
    const float* key = k + (size_t(offsets[i]) * nk + kh) * dim;
    q += head * dim;
    float part[4];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int col = lane + 32 * g;
        part[g] = col < dim ? __fmul_rn(q[col], key[col]) : 0.0f;
    }
    float sum = (part[0] + part[2]) + (part[1] + part[3]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down(sum, offset, 32);
    if (!lane) scores[size_t(head) * stride + i] = sum * rsqrtf(float(dim)) +
        (lag < extent ? bias[head * extent + lag] : 0.0f);
}

__global__ void split_values(const float* value, const float* scores, const float* sink, float* out,
                             const int32_t* offsets, const int32_t* count, const int32_t* positions,
                             const uint8_t* evicted, const int32_t* row, int nq, int nk, int dim, int stride) {
    constexpr int tile = 32;
    const int tid = threadIdx.x, head = blockIdx.x;
    const int kh = head / (nq / nk);
    const int live = *count, position = *row;
    scores += size_t(head) * stride;
    __shared__ float previous[tile], current[tile], final_denom;
    __shared__ int physical[tile];
    float hi = sink[head], denom = 1.0f, acc = 0.0f;
    for (int start = 0; start < live; start += tile) {
        const int n = min(tile, live - start);
        if (!tid) {
            for (int j = 0; j < n; ++j) {
                const int i = start + j;
                const bool active = !evicted[i] && position >= positions[i];
                physical[j] = active ? offsets[i] : -1;
                if (!active) continue;
                const float score = scores[i];
                const float next = fmaxf(hi, score);
                previous[j] = expf(hi - next);
                current[j] = expf(score - next);
                denom = denom * previous[j] + current[j];
                hi = next;
            }
        }
        __syncthreads();
        if (tid < dim) {
            for (int j = 0; j < n; ++j) {
                if (physical[j] < 0) continue;
                const size_t base = (size_t(physical[j]) * nk + kh) * dim;
                acc = acc * previous[j] + current[j] * value[base + tid];
            }
        }
        __syncthreads();
    }
    if (!tid) final_denom = denom;
    __syncthreads();
    if (tid < dim) out[head * dim + tid] = acc / final_denom;
}

extern "C" int hipengine_inkling2_split_attention(
    const float* q, const float* k, const float* v, const float* bias, const float* sink, float* out,
    const int32_t* offsets, const int32_t* count, const int32_t* positions, const uint8_t* evicted,
    const int32_t* row, float* scores, int nq, int nk, int dim, int extent, int live) {
    if (live <= 0 || nq <= 0 || nk <= 0 || nq % nk || dim <= 0 || dim > 128) return hipErrorInvalidValue;
    hipLaunchKernelGGL(split_scores, dim3(1 + (live - 1) / 4, nq), dim3(128), 0, 0,
                       q, k, bias, scores, offsets, count, positions, evicted, row, nq, nk, dim, extent, live);
    int status = hipGetLastError();
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(split_values, dim3(nq), dim3(128), 0, 0,
                       v, scores, sink, out, offsets, count, positions, evicted, row, nq, nk, dim, live);
    return hipGetLastError();
}
