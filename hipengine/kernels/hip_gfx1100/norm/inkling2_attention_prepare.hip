#include <hip/hip_runtime.h>
#include <stdint.h>

// One wave per (row, head). The 128-way tree matches the tiled attention kernel:
// combine +64, then +32, then the wave.
__device__ __forceinline__ float wave_square_sum(const float* x, const float* scale, int lane, int dim, float* v) {
    float part[4];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int col = lane + 32 * g;
        v[g] = col < dim ? (scale ? __fmul_rn(x[col], scale[col]) : x[col]) : 0.0f;
        part[g] = __fmul_rn(v[g], v[g]);
    }
    float sum = (part[0] + part[2]) + (part[1] + part[3]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down(sum, offset, 32);
    // Every lane scales its own elements, so all of them need the total.
    return __shfl(sum, 0);
}

__device__ __forceinline__ void wave_store(float* out, const float* v, int lane, int dim) {
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int col = lane + 32 * g;
        if (col < dim) out[col] = v[g];
    }
}

// out = clip(x * scale, threshold), times the log position scale when requested.
__global__ void inkling2_prepare_clip(const float* x, float* out, const float* scale, int dim, int heads,
                                      float threshold, float alpha, double floor, int first, int logscale) {
    const int lane = threadIdx.x % 32;
    const int waves = blockDim.x / 32;
    const int row = blockIdx.x;
    float factor_s = 1.0f;
    if (logscale) {
        const double position = first + row;
        factor_s = (float)(1.0 + (double)alpha * log(fmax(1.0, position / floor)));
    }
    for (int head = threadIdx.x / 32; head < heads; head += waves) {
        const size_t base = (size_t(row) * heads + head) * dim;
        float v[4];
        const float norm = sqrtf(wave_square_sum(x + base, scale ? scale + head * dim : nullptr, lane, dim, v));
        const float factor = __fmul_rn(fminf(1.0f, threshold / (norm + 1e-6f)), factor_s);
        #pragma unroll
        for (int g = 0; g < 4; ++g) v[g] = __fmul_rn(v[g], factor);
        wave_store(out + base, v, lane, dim);
    }
}

// out = x / sqrt(mean(x * x) + eps), per head.
__global__ void inkling2_prepare_rms(const float* x, float* out, int dim, int heads, float eps) {
    const int lane = threadIdx.x % 32;
    const int waves = blockDim.x / 32;
    const int row = blockIdx.x;
    for (int head = threadIdx.x / 32; head < heads; head += waves) {
        const size_t base = (size_t(row) * heads + head) * dim;
        float v[4];
        const float sum = wave_square_sum(x + base, nullptr, lane, dim, v);
        const float divisor = sqrtf(sum / float(dim) + eps);
        #pragma unroll
        for (int g = 0; g < 4; ++g) v[g] = __fdiv_rn(v[g], divisor);
        wave_store(out + base, v, lane, dim);
    }
}

extern "C" int hipengine_inkling2_attention_prepare_clip(
    const float* x, float* out, const float* scale, int rows, int dim, int heads, float threshold, float alpha,
    double floor, int first, int logscale) {
    if (!x || !out || rows <= 0 || dim <= 0 || dim > 128 || heads <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_prepare_clip, dim3(rows), dim3(128), 0, 0,
                       x, out, scale, dim, heads, threshold, alpha, floor, first, logscale ? 1 : 0);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_attention_prepare_rms(const float* x, float* out, int rows, int dim, int heads,
                                                        float eps) {
    if (!x || !out || rows <= 0 || dim <= 0 || dim > 128 || heads <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_prepare_rms, dim3(rows), dim3(128), 0, 0, x, out, dim, heads, eps);
    return hipGetLastError();
}
