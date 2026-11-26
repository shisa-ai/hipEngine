// Narrow-output FP32 GEMM for prompt blocks: out[rows, M] = x[rows, K] . W[M, K]^T.
//
// The products and sums are FP32 throughout, as in the rocBLAS GEMM this replaces; only the order
// of the sum over K differs. A wave32 holds R rows per lane and splits its work two ways: the
// lanes of a wave cover S slices of K and MP parts of the M outputs, so a lane keeps R rows by
// M / MP partial sums for its K-slice and the slices are added in a fixed xor tree at the end.
// The weights of a chunk, 4 * S * T inputs of every output, are read once by the block into shared
// memory and read back by its waves in contiguous 16-byte pieces, with the next chunk fetched
// under the products of the current one. A block of eight waves covers 16 rows and reads every
// weight once; a wave on its own reads them once per four rows, and with the weights cold in L2
// the re-reads cost DRAM time.
//
// The outputs are few: 24, 32, 64 or 128, and K is a multiple of the chunk. Other shapes are
// refused before any launch and keep their caller's GEMM.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

// `T` is the number of 16-byte pieces of K a lane takes from each output row in a chunk.
template <int M, int MP, int R, int WAVES, int T>
__launch_bounds__(32 * WAVES) __global__ void narrow_f32_kernel(const float* __restrict__ x,
                                                                       const float* __restrict__ w,
                                                                       float* __restrict__ out, int rows, int64_t k) {
    constexpr int S = 32 / MP;          // slices of K across the lanes of a wave
    constexpr int C = M / MP;           // outputs a lane covers
    constexpr int CHUNK = 4 * S * T;    // inputs in a chunk
    constexpr int PIECES = M * S * T / (32 * WAVES);   // 16-byte pieces of the chunk a thread fetches
    constexpr int THREADS = 32 * WAVES;
    __shared__ float tile[2][M][CHUNK];
    const int tid = threadIdx.x, lane = tid & 31, j = lane & (S - 1), mp = lane / S;
    const int row0 = (blockIdx.x * WAVES + (tid >> 5)) * R;
    float acc[R][C];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
    }
    const float* xr[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        // A row past the input reads row 0 and is not stored.
        const int row = row0 + r;
        xr[r] = x + (row < rows ? row : 0) * k;
    }
    // Thread t fetches pieces p, p + THREADS, ... of the chunk: piece p is the (p / (S * T))-th
    // output row's (p % (S * T))-th 16 bytes.
    const int64_t chunks = k / CHUNK;
    float4 piece[PIECES];
    #pragma unroll
    for (int p = 0; p < PIECES; ++p) {
        const int index = tid + p * THREADS;
        piece[p] = *reinterpret_cast<const float4*>(w + (index / (S * T)) * k + (index % (S * T)) * 4);
    }
    #pragma unroll
    for (int p = 0; p < PIECES; ++p) {
        reinterpret_cast<float4*>(tile[0][0])[tid + p * THREADS] = piece[p];
    }
    __syncthreads();
    #pragma unroll 1
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        const int buf = static_cast<int>(chunk & 1);
        const int64_t base = chunk * CHUNK + 4 * j;
        // The next chunk is fetched under this chunk's products and stored after them.
        const bool more = chunk + 1 < chunks;
        if (more) {
            #pragma unroll
            for (int p = 0; p < PIECES; ++p) {
                const int index = tid + p * THREADS;
                piece[p] = *reinterpret_cast<const float4*>(w + (index / (S * T)) * k + (chunk + 1) * CHUNK
                                                            + (index % (S * T)) * 4);
            }
        }
        float4 xv[R][T];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            #pragma unroll
            for (int t = 0; t < T; ++t) xv[r][t] = *reinterpret_cast<const float4*>(xr[r] + base + t * 4 * S);
        }
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            #pragma unroll
            for (int c = 0; c < C; ++c) {
                const float4 w4 = *reinterpret_cast<const float4*>(&tile[buf][mp * C + c][4 * j + t * 4 * S]);
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    acc[r][c] += xv[r][t].x * w4.x + xv[r][t].y * w4.y + xv[r][t].z * w4.z + xv[r][t].w * w4.w;
                }
            }
        }
        if (more) {
            #pragma unroll
            for (int p = 0; p < PIECES; ++p) {
                reinterpret_cast<float4*>(tile[buf ^ 1][0])[tid + p * THREADS] = piece[p];
            }
        }
        __syncthreads();
    }
    // The K-slices of the wave's lanes that share an output part, in a fixed order.
    #pragma unroll
    for (int off = S / 2; off > 0; off >>= 1) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            #pragma unroll
            for (int c = 0; c < C; ++c) acc[r][c] += __shfl_xor(acc[r][c], off);
        }
    }
    if (j == 0) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row >= rows) continue;
            float* orow = out + static_cast<int64_t>(row) * M + mp * C;
            #pragma unroll
            for (int c = 0; c < C; ++c) orow[c] = acc[r][c];
        }
    }
}

template <int M, int MP, int R, int WAVES, int T>
int launch(const void* x, const void* w, void* out, int64_t rows, int64_t k, hipStream_t stream) {
    constexpr int S = 32 / MP;
    if (!x || !w || !out || rows <= 0 || k <= 0 || k % (4 * S * T)) return hipErrorInvalidValue;
    const unsigned blocks = static_cast<unsigned>((rows + WAVES * R - 1) / (WAVES * R));
    hipLaunchKernelGGL((narrow_f32_kernel<M, MP, R, WAVES, T>), dim3(blocks), dim3(32 * WAVES), 0, stream,
                       static_cast<const float*>(x), static_cast<const float*>(w), static_cast<float*>(out),
                       static_cast<int>(rows), k);
    return hipGetLastError();
}

}  // namespace

#define INKLING2_NARROW_ENTRY(name, M, MP, R, WAVES, T)                                                             \
    extern "C" int name(const void* x, const void* w, void* out, int64_t rows, int64_t k, hipStream_t stream) { \
        return launch<M, MP, R, WAVES, T>(x, w, out, rows, k, stream);                                             \
    }
// 24 and 32 outputs: one part, 32 slices of K. 64 outputs: four parts of 16, eight slices, two
// pieces of K a chunk. 128 outputs: four parts of 32, eight slices, one piece. Two rows a lane.
INKLING2_NARROW_ENTRY(hipengine_narrow_f32_m24_f32_out, 24, 1, 2, 8, 1)
INKLING2_NARROW_ENTRY(hipengine_narrow_f32_m32_f32_out, 32, 1, 2, 8, 1)
INKLING2_NARROW_ENTRY(hipengine_narrow_f32_m64_f32_out, 64, 4, 2, 8, 2)
INKLING2_NARROW_ENTRY(hipengine_narrow_f32_m128_f32_out, 128, 4, 2, 8, 1)
