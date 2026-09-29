// Gemma 4 tiled flash prefill attention at head_dim 512.
//
// Source lineage: llama.cpp ggml/src/ggml-cuda/fattn-tile.cuh (device region)
// at the commit pinned in docs/source_lineage.json, plus the common.cuh helpers
// it depends on. The device body below is byte-identical to that file so it
// stays diffable against upstream; everything Gemma-specific lives in the host
// launcher at the end of this file. If you edit the device region, record the
// drift in docs/source_lineage.json first.
//
// Why this exists: the exact block kernel in gemma4_attention.hip does three
// passes over the key block (max, denominator, weighted sum) and its grid is
// tokens*head_dim threads that only cooperate along head_dim. At Gemma 4's five
// global layers (head_dim 512) that spends 39.2 ms of a prefill budget, and the
// flash formulation cuts key traffic threefold with an online softmax.
//
// Three properties of this port are easy to get wrong, and each was wrong at
// some point during bring-up:
//
//   * Input layout. hipEngine packs (tokens, num_heads, head_dim) head-last, but
//     the ggml tensor this kernel indexes is (head_dim, tokens, heads), head
//     *outermost*: element h*(T*D) + t*D + d. The launcher transposes during the
//     dtype conversion. Staging in source order silently scrambles every logit
//     while still producing finite output.
//   * Strides. ggml's nb[2] is ne[1]*nb[1], so the stride to the next query head
//     is head_dim*tokens (resp. head_dim*keys for K/V) -- not
//     head_dim*num_heads. Both errors vanish at head 0, which is exactly what
//     makes them hard to see in a single-row probe.
//   * Masks. The kernel reads mask[query*keys + key] as half, with keep=0.0 and
//     drop=-inf, and it never re-derives causality: KV_max is null, so it attends
//     over every key and the caller's keep mask decides. That is the same
//     contract the exact kernel runs under.
//
// Stage exactly as upstream's launch_fattn does: bf16 -> f32 for Q, bf16 -> f16
// for K/V, u8 -> f16 for the mask, f32 -> bf16 for the output. Measured at
// Gemma geometry (tokens 512, keys 1024, 16/2 heads, head_dim 512) on a W7900
// those four conversions total 0.0817 ms per layer, 0.41 ms across all five
// global layers, 1.0% of the 39.2 ms budget -- so the kernel body is left
// untouched rather than forked to avoid them.
//
// Return codes from hipengine_gemma4_attention_tiled_prefill_bf16:
//   0 ok, 1 head_dim != 512, 2 non-positive shape, 3 keys not a multiple of 128,
//   4 num_heads not a multiple of 8, 5 num_heads not divisible by num_kv_heads,
//   6 gqa_ratio not a multiple of 8, 7 staging allocation failed,
//   8/9 launch or output conversion failed.

// Minimal stand-ins for the four common.cuh helpers fattn-tile.cuh's device
// code uses. Written fresh rather than sliced out of common.cuh, whose
// surrounding #if forest does not survive partial extraction.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#define GGML_USE_HIP
#define RDNA
#define FLASH_ATTN_AVAILABLE
#define GGML_USE_WMMA_FATTN
#define RDNA3
#define FAST_FP16_AVAILABLE
#define V_DOT2_F32_F16_AVAILABLE
typedef _Float16 ggml_fp16_t;
using half = __half;
using half2 = __half2;
// Host-side cc predicates: this port only instantiates the device config path,
// so the topology questions the host overload asks are answered for gfx1100.
#define GGML_CUDA_CC_IS_AMD(cc) true
#define GGML_CUDA_CC_IS_RDNA(cc) true
static __host__ __device__ bool fast_fp16_available(int) { return true; }
#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() { return 16; }

template <int n> struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};
template <> struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const { f(0, args...); }
};

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float v, const float u) { acc += v*u; }
static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float2 v, const float2 u) {
    acc += v.x*u.x + v.y*u.y;
}
static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const half2 v, const half2 u) {
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
}
static __device__ __forceinline__ void ggml_cuda_mad(half2 & acc, const half2 v, const half2 u) { acc += v*u; }

template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(nbytes <= 16 || alignment == 0, "misuse of ggml_cuda_memcpy_1");
    if constexpr (alignment != 0) static_assert(nbytes % alignment == 0, "bad alignment");
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;
#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1)      ((char *) dst)[i] = ((const char *) src)[i];
        else if constexpr (nb_per_cpy == 2) ((short *) dst)[i] = ((const short *) src)[i];
        else if constexpr (nb_per_cpy == 4) ((int *) dst)[i] = ((const int *) src)[i];
        else if constexpr (nb_per_cpy == 8) ((int2 *) dst)[i] = ((const int2 *) src)[i];
        else if constexpr (nb_per_cpy == 16) ((int4 *) dst)[i] = ((const int4 *) src)[i];
        else static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
    }
}

#include <cfloat>
#define NO_DEVICE_CODE
#define GGML_UNUSED_VARS(...) do { (void)sizeof((__VA_ARGS__, 0)); } while (0)
#ifndef FATTN_KQ_MAX_OFFSET
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)
#endif

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    return (hi + n) >> fastdiv_values.y;
}
static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    return make_uint2(div_val, n - div_val * fastdiv_values.z);
}
static __device__ __forceinline__ float get_alibi_slope(
        const float max_bias, const uint32_t h, const uint32_t n_head_log2,
        const float m0, const float m1) {
    if (max_bias <= 0.0f) { return 1.0f; }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? (int)h + 1 : 2*(int)(h - n_head_log2) + 1;
    return powf(base, exph);
}

#define FP16_AVAILABLE
template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1)
        x += __shfl_xor_sync(0xffffffffffffffffULL, x, offset, width);
    return x;
}
template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1)
        x += __shfl_xor_sync(0xffffffffffffffffULL, x, offset, width);
    return x;
}
template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a.x += __shfl_xor_sync(0xffffffffffffffffULL, a.x, offset, width);
        a.y += __shfl_xor_sync(0xffffffffffffffffULL, a.y, offset, width);
    }
    return a;
}
template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1)
        a = __hadd2(a, __shfl_xor_sync(0xffffffffffffffffULL, a, offset, width));
    return a;
}
template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1)
        x = fmaxf(x, __shfl_xor_sync(0xffffffffffffffffULL, x, offset, width));
    return x;
}
// ---- config tables + accessors (fattn-tile.cuh 7-347) ----

// TODO optimize kernel parameters for FP16 NVIDIA (P100)
// TODO optimize kernel parameters for head sizes 40, 72, 80, 96, 112

// The ROCm compiler cannot handle templating in __launch_bounds__.
// As a workaround, define a macro to package the kernel parameters as uint32_t:
#define GGML_CUDA_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        return ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }                                                                                                 \

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_nvidia_fp16(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  64,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  64,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  64,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  64,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  64,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  64,  72)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  64,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  64,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  64,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  64,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  64,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  64,  48)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  64,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  64,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  64,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  64,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  64,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  64,  64)

    return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_nvidia_fp32(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  32,  72)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  32,  48)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 3,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 128, 3,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 128, 3,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  32, 256)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 32, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  32,  64)

    return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 64, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 64, 256, 2,  32,  72)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 64, 256, 2,  32,  48)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 64, 256, 2,  32,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 64, 256, 2,  64,  32)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  32, 128)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 32, 512, 1, 128,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 32, 512, 1, 128,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 32, 512, 1, 128,  64)

    return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd_rdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 5, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 128, 5, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 128, 4,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 64, 128, 5,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 64, 256, 2,  32,  72)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 64, 256, 2,  32,  48)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 64, 256, 2,  32,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 8,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 128, 8,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 3, 128, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 3, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 64, 256, 3,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 6,  32, 256)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 128, 6,  32, 256)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 5,  32, 256)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 3,  64, 128)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 32, 256, 2, 128,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 4,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 32, 256, 2, 128,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 4,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 32, 256, 2, 128,  64)

    return 0;
}

static __host__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA(cc)) {
            return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
        }
        return ggml_cuda_fattn_tile_get_config_amd(DKQ, DV, ncols);
    }
    if (fast_fp16_available(cc)) {
        return ggml_cuda_fattn_tile_get_config_nvidia_fp16(DKQ, DV, ncols);
    }
    return ggml_cuda_fattn_tile_get_config_nvidia_fp32(DKQ, DV, ncols);
}

static constexpr __device__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols) {
#ifdef GGML_USE_HIP
#ifdef RDNA
    return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
#else
    return ggml_cuda_fattn_tile_get_config_amd(DKQ, DV, ncols);
#endif // RDNA
#else
#ifdef FAST_FP16_AVAILABLE
    return ggml_cuda_fattn_tile_get_config_nvidia_fp16(DKQ, DV, ncols);
#else
    return ggml_cuda_fattn_tile_get_config_nvidia_fp32(DKQ, DV, ncols);
#endif // FAST_FP16_AVAILABLE
#endif // GGML_USE_HIP
}

static __host__ int ggml_cuda_fattn_tile_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

static __host__ int ggml_cuda_fattn_tile_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

static __host__ int ggml_cuda_fattn_tile_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

static __host__ int ggml_cuda_fattn_tile_get_nbatch_K(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_get_nbatch_K(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}
// ---- device helpers + kernel (fattn-tile.cuh 348-1115) ----

// TODO: deduplicate with mma-f16
template<int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int stride_KV, const int i_sup) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    auto load = [&] __device__ (const int n) {
        const int stride_j = warp_size >> n;

        if (stride_j == 0) {
            return;
        }

        const int j0_start = stride_j == warp_size ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        const int j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        const int stride_i = warp_size / stride_j;

        if (j0_start == j0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_j == warp_size ? 0 : threadIdx.x / stride_j);

            if (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    const int j = j0*cpy_ne + (stride_j == warp_size ? threadIdx.x : threadIdx.x % stride_j)*cpy_ne;

                    const __align__(16) half2 zero[cpy_ne] = {{0.0f, 0.0f}};
                    ggml_cuda_memcpy_1<cpy_nb>(
                        tile_KV + i*(J/2 + J_padding) + j,
                        !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);
                }
            }
        }
    };
    // 1: max 64*16=512 bytes, 512 half
    // 2: max 32*16=512 bytes, 256 half
    // 3: max 16*16=256 bytes, 128 half
    // 4: max  8*16=128 bytes,  64 half
    // 5: max  4*16= 64 bytes,  32 half
    // 6: max  2*16= 32 bytes,  16 half
    // 7: max  1*16= 16 bytes,   8 half
    static_assert(J % 8 == 0, "bad J");
    static_assert((J/2) % cpy_ne == 0, "bad J");
    ggml_cuda_unroll<7>{}(load);
}

template<int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_load_tile(
        const half2 * const __restrict__ KV, float * const __restrict__ tile_KV, const int stride_KV, const int i_sup) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    auto load = [&] __device__ (const int n) {
        const int stride_j = warp_size >> n;

        if (stride_j == 0) {
            return;
        }

        const int j0_start = stride_j == warp_size ? 0 : (J/cpy_ne) - (J/cpy_ne) % (2*stride_j);
        const int j0_stop  =                             (J/cpy_ne) - (J/cpy_ne) % (1*stride_j);
        const int stride_i = warp_size / stride_j;

        if (j0_start == j0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_j == warp_size ? 0 : threadIdx.x / stride_j);

            if (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    const int j = j0*(cpy_ne/2) + (stride_j == warp_size ? threadIdx.x : threadIdx.x % stride_j)*(cpy_ne/2);

                    const half2 zero[cpy_ne/2] = {{0.0f, 0.0f}};
                    __align__(16) half2 tmp_h2[cpy_ne/2];
                    ggml_cuda_memcpy_1<sizeof(tmp_h2)>(
                        tmp_h2, !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);

                    __align__(16) float2 tmp_f2[cpy_ne/2];
#pragma unroll
                    for (int l = 0; l < cpy_ne/2; ++l) {
                        tmp_f2[l] = __half22float2(tmp_h2[l]);
                    }
                    ggml_cuda_memcpy_1<sizeof(tmp_f2)>(tile_KV + i*(J + J_padding) + 2*j, tmp_f2);
                }
            }
        }
    };
    // 1: max 32*16=512 bytes, 128 float
    // 2: max 16*16=256 bytes,  64 float
    // 3: max  8*16=128 bytes,  32 float
    // 4: max  4*16= 64 bytes,  16 float
    // 5: max  2*16= 32 bytes,   8 float
    static_assert(J % 8 == 0, "bad J");
    static_assert(J % cpy_ne == 0, "bad J");
    ggml_cuda_unroll<5>{}(load);
}

// Function that performs a single iteration in for the KQ matrix multiplication:
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check, typename T_vec_dot>
static __device__ __forceinline__ void flash_attn_tile_iter_KQ(
        T_vec_dot   * const Q_tmp,
        const half2 * const __restrict__ K_h2,
        T_vec_dot   * const KV_tmp,
        const int stride_K2,
        const int k_VKQ_0,
        const int k_VKQ_sup,
        const int k_KQ_0,
        float * KQ_acc) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    flash_attn_tile_load_tile<warp_size, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    __syncthreads();

#ifdef FAST_FP16_AVAILABLE
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __align__(16) half2 K_k[nbatch_fa/(np*warp_size)][cpy_ne];
        __align__(16) half2 Q_k[cpw][cpy_ne];
#else
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __align__(16) float K_k[nbatch_fa/(np*warp_size)][cpy_ne];
        __align__(16) float Q_k[cpw][cpy_ne];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;

#ifdef FAST_FP16_AVAILABLE
            ggml_cuda_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#else
            ggml_cuda_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#endif // FAST_FP16_AVAILABLE
        }
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (threadIdx.y / np)*cpw;

#ifdef FAST_FP16_AVAILABLE
            ggml_cuda_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#else
            ggml_cuda_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#endif // FAST_FP16_AVAILABLE
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                for (int k = 0; k < cpy_ne; ++k) {
                    ggml_cuda_mad(KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0], K_k[i_KQ_0/(np*warp_size)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    if (k_KQ_0 + nbatch_K < DKQ) {
        __syncthreads(); // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int DV, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check, typename T_vec_dot, typename T_KQ, typename T_acc>
static __device__ __forceinline__ void flash_attn_tile_iter(
        T_vec_dot * const Q_tmp,
        const half2 * const __restrict__ K_h2,
        const half2 * const __restrict__ V_h2,
        const half  * const __restrict__ mask,
        const uint3 ne01,
        const float logit_softcap,
        const float slope,
        T_KQ      * const KQ,
        T_vec_dot * const KV_tmp,
        const int stride_K2,
        const int stride_V2,
        const int stride_mask,
        float * const KQ_max,
        float * const KQ_sum,
        T_acc * const VKQ,
        const int k_VKQ_0,
        const int k_VKQ_max,
        const int col_Q_0) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size.

    // KQ_cs == KQ chunk size, number of KQ values in j direction to store as one contiguous chunk in memory.
    // KQ is originally 2D but uses a Z-shaped 3D memory pattern like KQ[ncols/KQ_cs][DVp][KQ_cs].
#ifdef FAST_FP16_AVAILABLE
    constexpr int KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#else
    constexpr int KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#endif // FAST_FP16_AVAILABLE
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    const int k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    float KQ_max_new[cpw];
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    float KQ_acc[nbatch_fa/(np*warp_size) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr int nbatch_K_last = DKQ % nbatch_K;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_last; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    if (nbatch_K_last > 0) {
        constexpr int k_KQ_0 = DKQ - nbatch_K_last;
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_last, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int j = fastmodulo(col_Q_0 + (jc0 + (threadIdx.y / np)*cpw)/ncols2, ne01);

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;

#if defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0] *= 4.0f;
#endif // defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)

            if (use_logit_softcap) {
                KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] = logit_softcap * tanhf(KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0]);
            }

            if (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] += (ncols2 > 1 || mask) ?
                    slope*__half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;

                KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] + FATTN_KQ_MAX_OFFSET);
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
    }

    if constexpr (np == 1) {
        __syncthreads();
    } else {
        static_assert(cpw == 1, "bad cpw");
        __shared__ float KQ_max_new_shared[nwarps];
        if (threadIdx.x == 0) {
            KQ_max_new_shared[threadIdx.y] = KQ_max_new[0];
        }
        __syncthreads();
        KQ_max_new[0] = KQ_max_new_shared[(threadIdx.y & ~(np-1)) + threadIdx.x % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#ifdef FAST_FP16_AVAILABLE
        __align__(16) half  tmp[nbatch_fa/(np*warp_size)][KQ_cs];
#else
        __align__(16) float tmp[nbatch_fa/(np*warp_size)][KQ_cs];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
        for (int jc1 = 0; jc1 < KQ_cs; ++jc1) {
            const int jc = jc0 + jc1;

            const float KQ_max_scale = expf(KQ_max[jc] - KQ_max_new[jc]);
            KQ_max[jc] = KQ_max_new[jc];

            float KQ_sum_add = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
                const float val = !oob_check || i0 + (threadIdx.y % np)*warp_size + threadIdx.x < static_cast<uint32_t>(k_VKQ_sup) ?
                    expf(KQ_acc[(i0/(np*warp_size))*cpw + jc] - KQ_max[jc]) : 0.0f;
                KQ_sum_add += val;
                tmp[i0/(np*warp_size)][jc1] = val;
            }
            KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].x *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
            const int i = i0 + (threadIdx.y % np)*warp_size + threadIdx.x;

            ggml_cuda_memcpy_1<sizeof(tmp[0])>(
                KQ + (jc0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs))*(nbatch_fa*KQ_cs) + i*KQ_cs,
                tmp[i0/(np*warp_size)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr int nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        flash_attn_tile_load_tile<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        __syncthreads();

#ifdef FAST_FP16_AVAILABLE
#pragma unroll
        for (int k1 = 0; k1 < nbatch_V; k1 += np) {
            __align__(16) half2 V_k[(DVp/2)/warp_size];
            __align__(16) half2 KQ_k[cpw];

            constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/warp_size], &KV_tmp[(k1 + threadIdx.y % np)*(DV/2) + i0 + threadIdx.x*cpy_ne_D]);
            }
#pragma unroll
            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                const int jc_KQ = jc_VKQ_0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs);

                __align__(16) half tmp[KQ_cs];
                ggml_cuda_memcpy_1<KQ_cs*sizeof(half)>(
                    &tmp, KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + threadIdx.y % np)*KQ_cs);
#pragma unroll
                for (int jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0+jc_VKQ_1] = __half2half2(tmp[jc_VKQ_1]);
                }
            }

#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size] += V_k[i0/warp_size]*KQ_k[jc_VKQ_0];
                }
            }
        }
#else
#pragma unroll
        for (int k1 = 0; k1 < nbatch_V; k1 += np) {
            __align__(16) float2 V_k[(DVp/2)/warp_size];
            __align__(16) float  KQ_k[cpw];

            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_size)], &KV_tmp[(k1 + threadIdx.y % np)*DV + i0 + threadIdx.x*cpy_ne_D]);
            }
#pragma unroll
            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                const int jc_KQ = jc_VKQ_0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs);

                ggml_cuda_memcpy_1<KQ_cs*sizeof(float)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + threadIdx.y % np)*KQ_cs);
            }

#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].x += V_k[i0/warp_size].x*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].y += V_k[i0/warp_size].y*KQ_k[jc_VKQ_0];
                }
            }
        }
#endif // FAST_FP16_AVAILABLE

        __syncthreads();
    }
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_tile_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_tile(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:

    if (
#ifdef GGML_USE_WMMA_FATTN
            (ncols2 != 1 && DV != 40 && DV != 72 && DV != 512) ||
#endif // GGML_USE_WMMA_FATTN
            (use_logit_softcap && !(DV == 128 || DV == 256 || DV == 512))
    ) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    static_assert(ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");

    constexpr int ncols     = ncols1*ncols2;
    constexpr int warp_size = 32;
    constexpr int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_size;
    constexpr int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr int nbatch_K  = ggml_cuda_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int col_Q_0 = blockIdx.x * ncols1; // Index of the first Q column for this CUDA block to work on.

    const int sequence = blockIdx.z / (ne02/ncols2);
    const int head0 = blockIdx.z*ncols2 - sequence*ne02; // == blockIdx.z % (ne02/ncols2)
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0);
    const half2 * K_h2 = (const half2 *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
    const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio)); // K and V have same shape

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33)) : nullptr;

    const int stride_K2   = nb11 / sizeof(half2);
    const int stride_V2   = nb21 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr int np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.
    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_size) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr int DKQp = (DKQ + 2*warp_size - 1) & ~(2*warp_size - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr int DVp  = (DV  + 2*warp_size - 1) & ~(2*warp_size - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.
#ifdef FAST_FP16_AVAILABLE
    __shared__ half2 Q_tmp[ncols * DKQ/2];
    __shared__ half2 KV_tmp[nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV];
    __shared__ half  KQ[ncols * nbatch_fa];
    __align__(16) half2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};
#else
    __shared__ float Q_tmp[ncols * DKQ];
    __shared__ float KV_tmp[nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV];
    __shared__ float KQ[ncols * nbatch_fa];
    __align__(16) float2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};
#endif // FAST_FP16_AVAILABLE

    float KQ_max[cpw];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float KQ_sum[cpw] = {0.0f};

    // Load Q data, convert to FP16 if fast:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y / np)*cpw;

        const int j = jc / ncols2;
        const int c = jc % ncols2;

        constexpr int cpy_ne_D = cpy_ne < DKQp/warp_size ? cpy_ne : DKQp/warp_size;

#pragma unroll
        for (int i0 = 0; i0 < DKQp; i0 += np*warp_size*cpy_ne_D) {
            if (i0 + np*warp_size*cpy_ne_D <= DKQ || i0 + (threadIdx.y % np)*(warp_size*cpy_ne_D) + threadIdx.x*cpy_ne_D < DKQ) {
                __align__(16) float tmp_f[cpy_ne_D] = {0.0f};
                ggml_cuda_memcpy_1<sizeof(tmp_f)>
                    (tmp_f, &Q_f[c*(nb02/sizeof(float)) + fastmodulo(col_Q_0 + j, ne01)*(nb01/sizeof(float))
                                 + i0 + (threadIdx.y % np)*(warp_size*cpy_ne_D) + threadIdx.x*cpy_ne_D]);

#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#ifdef FAST_FP16_AVAILABLE
                __align__(16) half2 tmp_h2[cpy_ne_D/2];
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#if defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1/2] *= make_half2(0.25f, 0.25f);
#endif // defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
                }
                ggml_cuda_memcpy_1<sizeof(tmp_h2)>(
                    &Q_tmp[jc*(DKQ/2) + i0/2 + (threadIdx.y % np)*(warp_size*cpy_ne_D/2) + threadIdx.x*(cpy_ne_D/2)],
                    tmp_h2);
#else
                ggml_cuda_memcpy_1<sizeof(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (threadIdx.y % np)*(warp_size*cpy_ne_D)   + threadIdx.x* cpy_ne_D],
                    tmp_f);
#endif // FAST_FP16_AVAILABLE
            }
        }
    }

    __syncthreads();

    // Main loop over KV cache:
    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    if (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        int k_VKQ_0 = blockIdx.y*nbatch_fa;
        while (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
            constexpr bool oob_check = false;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
                stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0);
            k_VKQ_0 += gridDim.y*nbatch_fa;
        }
        if (k_VKQ_0 < k_VKQ_max) {
            constexpr bool oob_check = true;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
                stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0);
        }
    } else {
        // Branch without out-of-bounds checks.
        for (int k_VKQ_0 = blockIdx.y*nbatch_fa; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nbatch_fa) {
            constexpr bool oob_check = false;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
                stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0);
        }
    }

#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_sum[jc0] = warp_reduce_sum<warp_size>(KQ_sum[jc0]);
    }

    if constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#ifdef FAST_FP16_AVAILABLE
        half2 * VKQ_combine    = (half2 *) KV_tmp;
#else
        float * VKQ_combine    = (float *) KV_tmp;
#endif // FAST_FP16_AVAILABLE
        float * KQ_sum_combine = (float *) Q_tmp;

        if (threadIdx.y % np != 0) {
#ifdef FAST_FP16_AVAILABLE
            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&VKQ_combine[threadIdx.y*(DVp/2) + i0 + threadIdx.x*cpy_ne_D], &VKQ[i0/warp_size]);
            }
#else
            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[threadIdx.y*DVp + i0 + threadIdx.x*cpy_ne_D], ((const float *) VKQ) + i0/warp_size);
            }
#endif // FAST_FP16_AVAILABLE

            if (threadIdx.x == 0) {
                KQ_sum_combine[threadIdx.y] = KQ_sum[0];
            }

            return;
        }

        __syncthreads();

#pragma unroll
        for (int ip = 1; ip < np; ++ip) {
#ifdef FAST_FP16_AVAILABLE
            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                __align__(16) half2 tmp[cpy_ne_D];
                ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(threadIdx.y + ip)*(DVp/2) + i0 + threadIdx.x*cpy_ne_D]);
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_size + i1] += tmp[i1];
                }
            }
#else
            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                __align__(16) float tmp[cpy_ne_D];
                ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(threadIdx.y + ip)*DVp + i0 + threadIdx.x*cpy_ne_D]);
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((float *)VKQ)[i0/warp_size + i1] += tmp[i1];
                }
            }
#endif // FAST_FP16_AVAILABLE

            KQ_sum[0] += KQ_sum_combine[threadIdx.y + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (threadIdx.y/np)*cpw;
            const float sink = ((const float *) sinks)[head0 + jc % ncols2];

            float KQ_max_new_j = fmaxf(KQ_max[jc0], sink);
            const float KQ_max_scale = expf(KQ_max[jc0] - KQ_max_new_j);
            KQ_max[jc0] = KQ_max_new_j;

            const float val = expf(sink - KQ_max[jc0]);
            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + val;

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].x *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }
    }

    // Write back results:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y/np)*cpw;

        const int j = jc / ncols2;
        const int c = jc % ncols2;

        if (ncols1 > 1 && col_Q_0 + j >= int(ne01.z)) {
            return;
        }

        const float scale = gridDim.y == 1 ? 1.0f/KQ_sum[jc0] : 1.0f;

        const int j_dst_unrolled = ((sequence*int(ne01.z) + col_Q_0 + j)*ne02 + head0 + c)*gridDim.y + blockIdx.y;

#ifdef FAST_FP16_AVAILABLE
        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
            __align__(16) float2 tmp[cpy_ne_D];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = __half22float2(VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size + i1]);
                tmp[i1].x *= scale;
                tmp[i1].y *= scale;
            }
            if (i0 + warp_size*cpy_ne_D <= DV/2 || i0 + threadIdx.x*cpy_ne_D < DV/2) {
                ggml_cuda_memcpy_1<sizeof(tmp)>(&dst[j_dst_unrolled*DV + 2*i0 + threadIdx.x*(2*cpy_ne_D)], tmp);
            }
        }
#else
        constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
            if (i0 + warp_size*cpy_ne_D <= DV || i0 + threadIdx.x*cpy_ne_D < DV) {
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].x *= scale;
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].y *= scale;
                }
                ggml_cuda_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + threadIdx.x*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size)]);
            }
        }
#endif // FAST_FP16_AVAILABLE

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_sum[jc0]);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

// ---------------------------------------------------------------------------
// Host launcher: Gemma 4 prefill attention at head_dim 512, tiled route.
//
// Ported from llama.cpp ggml/src/ggml-cuda/fattn-tile.cuh (device region) with
// its common.cuh helpers reimplemented. The device kernel body is UNCHANGED
// from upstream so it stays verifiable against it; everything Gemma-specific
// lives here.
//
// Gemma ABI in, ggml ABI at the kernel boundary:
//   query      (tokens, num_heads, head_dim)   bf16
//   key/value  (keys,   num_kv_heads, head_dim) bf16
//   keep_mask  (tokens, keys)                  uint8, 1 = attend
//   out        (tokens, num_heads, head_dim)   bf16
//
// launch_fattn asserts F32 Q, F32 out, F16 mask and stages F16 KV, so this
// launcher performs exactly those conversions. Measured at this geometry on a
// W7900 they total 0.0817 ms per layer, 1.0% of the 39.2 ms budget.
//
// Launch geometry, mirroring launch_fattn with parallel_blocks forced to 1:
//   grid  = (ceil(tokens/ncols1), 1, ntiles_z_gqa * num_kv_heads)
//   block = (32, nwarps)
// Forcing y = 1 keeps the kernel off its partial-tile combine path, so
// dst_meta may be null and dst is written directly (the write is guarded by
// `gridDim.y != 1`). KV_max is null, so k_VKQ_max = ne11 = keys and the Gemma
// keep mask alone decides what is attended -- the same contract the exact
// block kernel runs under.
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstddef>

// Staging for one launch, grown monotonically and reused. Attention launches
// are ordered on a stream, so a single static block is enough and avoids a
// hipMalloc per layer per block.
namespace {
struct Staging {
    void*  ptr = nullptr;
    size_t cap = 0;
};
Staging g_stage;

bool ensure(size_t n) {
    if (n <= g_stage.cap) return true;
    if (g_stage.ptr) hipFree(g_stage.ptr);
    g_stage.ptr = nullptr;
    g_stage.cap = 0;
    if (hipMalloc(&g_stage.ptr, n) != hipSuccess) return false;
    g_stage.cap = n;
    return true;
}

namespace {
// hipEngine packs (tokens, heads, head_dim) head-last; the ggml tensor this
// kernel indexes is (head_dim, tokens, heads), head-outermost:
//   ggml = h*(T*D) + t*D + d,  src = (t*H + h)*D + d
// K/V follow the same rule with (keys, kv_heads, head_dim) -> kh*(K*D) + k*D + d.
// The conversion kernels transpose src -> ggml so the kernel body can stay
// byte-identical to upstream.
__global__ void k_q_bf16_to_f32_ggml(const uint16_t* __restrict__ in, float* __restrict__ out,
                                     int T, int H, int D) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t n = (size_t)T * H * D;
    if (i >= n) return;
    const int  d = (int)(i % D);
    const int  r = (int)(i / D);
    const int  h = r % H;
    const int  t = r / H;
    out[(size_t)h * T * D + (size_t)t * D + d] = __uint_as_float((uint32_t)in[i] << 16);
}

__global__ void k_kv_bf16_to_f16_ggml(const uint16_t* __restrict__ in, __half* __restrict__ out,
                                      int K, int NH, int D) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t n = (size_t)K * NH * D;
    if (i >= n) return;
    const int  d = (int)(i % D);
    const int  r = (int)(i / D);
    const int  kh = r % NH;
    const int  k = r / NH;
    out[(size_t)kh * K * D + (size_t)k * D + d] = __float2half_rn(__uint_as_float((uint32_t)in[i] << 16));
}
} // namespace
__global__ void k_f32_to_bf16(const float* __restrict__ in, uint16_t* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        const uint32_t b = __float_as_uint(in[i]);
        const uint32_t up = b >> 16, rem = b & 0xFFFFu;
        const uint32_t inc = (rem > 0x8000u) || (rem == 0x8000u && (up & 1u));  // round-to-nearest-even
        out[i] = (uint16_t)(up + inc);
    }
}
__global__ void k_bf16_to_f16(const uint16_t* __restrict__ in, __half* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t f = (uint32_t)in[i] << 16;
        out[i] = __float2half_rn(__uint_as_float(f));
    }
}
__global__ void k_mask_u8_to_f16(const uint8_t* __restrict__ in, __half* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] ? __float2half_rn(0.0f) : __float2half_rn(-INFINITY);
}

inline unsigned grid_for(size_t n) { return (unsigned)((n + 255) / 256); }
} // namespace

// Configuration constants carried over from the config tables.
static constexpr int kTiledNthreads = 256;   // CASE(512,512,32,...,256,...)
static constexpr int kWarpSize      = 32;
static constexpr int kNwarps        = kTiledNthreads / kWarpSize; // 8
static constexpr int kNcols1        = 4;
static constexpr int kNcols2        = 8;
static constexpr int kNbatchFA      = 128;   // CASE nbatch_fa field
static constexpr int kNbatchK       = 64;    // CASE nbatch_K field
static constexpr int kHeadDim       = 512;
static constexpr int kKeysStride    = 256;   // FATTN_KQ_STRIDE

extern "C" int hipengine_gemma4_attention_tiled_prefill_bf16(
    const uint16_t* query,
    const uint16_t* key,
    const uint16_t* value,
    const uint8_t* keep_mask,
    uint16_t* out,
    int64_t tokens, int64_t num_heads, int64_t num_kv_heads,
    int64_t head_dim, float scale, hipStream_t stream, int64_t keys) {

    // Capability checks. Each names a property the device code actually needs;
    // none of them is a quality judgement, and each is stated in the return code
    // documentation below so a caller can satisfy or work around it.
    if (head_dim != kHeadDim) return 1;                    // template is DKQ=DV=512
    if (tokens <= 0 || keys <= 0) return 2;
    // ncols2 > 1 selects the "branch without out-of-bounds checks" KV loop: it
    // walks k_VKQ_0 in steps of nbatch_fa and loads a full nbatch_fa tile each
    // time, so the key count must be a multiple of nbatch_fa or the final tile
    // reads past the K/V/mask buffers. Pad keys to a multiple of 128.
    if (keys % kNbatchFA != 0) return 3;
    // grid.z decoding is sequence = z/(ne02/ncols2), head0 = (z % (ne02/ncols2))*ncols2,
    // while the launch computes grid.z = ntiles_z_gqa*ne12. Those agree only when
    // ncols2 divides gqa_ratio; otherwise sequence runs past ne03 and writes off
    // the end of the output.
    if (num_heads < kNcols2 || num_heads % kNcols2 != 0) return 4;
    if (num_heads % num_kv_heads != 0) return 5;
    if ((num_heads / num_kv_heads) % kNcols2 != 0) return 6;

    const size_t q_elems   = (size_t)tokens * num_heads * head_dim;
    const size_t kv_elems  = (size_t)keys * num_kv_heads * head_dim;
    const size_t mask_elems = (size_t)tokens * keys;
    const size_t need = q_elems * sizeof(float) * 2     // Q staging + out staging
                      + kv_elems * sizeof(__half) * 2   // K staging + V staging
                      + mask_elems * sizeof(__half);
    if (!ensure(need)) return 7;

    char* p = (char*)g_stage.ptr;
    float*   q_f32 = (float*)p;                       p += q_elems * sizeof(float);
    float*   o_f32 = (float*)p;                       p += q_elems * sizeof(float);
    __half*  k_f16 = (__half*)p;                      p += kv_elems * sizeof(__half);
    __half*  v_f16 = (__half*)p;                      p += kv_elems * sizeof(__half);
    __half*  m_f16 = (__half*)p;                      p += mask_elems * sizeof(__half);

    const hipStream_t s = stream;

    k_q_bf16_to_f32_ggml<<<grid_for(q_elems), 256, 0, s>>>(
        (const uint16_t*)query, q_f32, (int)tokens, (int)num_heads, (int)head_dim);
    k_kv_bf16_to_f16_ggml<<<grid_for(kv_elems), 256, 0, s>>>(
        (const uint16_t*)key, k_f16, (int)keys, (int)num_kv_heads, (int)head_dim);
    k_kv_bf16_to_f16_ggml<<<grid_for(kv_elems), 256, 0, s>>>(
        (const uint16_t*)value, v_f16, (int)keys, (int)num_kv_heads, (int)head_dim);
    k_mask_u8_to_f16<<<grid_for(mask_elems), 256, 0, s>>>((const uint8_t*)keep_mask, m_f16, mask_elems);

    // init_fastdiv_values(Q->ne[1]): mp, L, divisor, packed as upstream does.
    const uint32_t d = (uint32_t)tokens;
    uint32_t L = 0;
    while (L < 32 && ((uint32_t)1 << L) < d) L++;
    const uint32_t mp = (uint32_t)(((uint64_t)1 << 32) * (((uint64_t)1 << L) - d) / d + 1);
    const uint3 ne01 = make_uint3(mp, L, d);

    const int32_t ne00 = (int32_t)head_dim;
    const int32_t ne02 = (int32_t)num_heads;
    const int32_t ne03 = 1;
    const int32_t nb01 = (int32_t)(head_dim * (int64_t)sizeof(float));
    const int32_t nb02 = (int32_t)(head_dim * tokens * (int64_t)sizeof(float));   // ggml: nb[2] = ne[1]*nb[1]
    const int32_t nb03 = (int32_t)(head_dim * tokens * num_heads * (int64_t)sizeof(float));

    const int32_t ne10 = (int32_t)head_dim;
    const int32_t ne11 = (int32_t)keys;
    const int32_t ne12 = (int32_t)num_kv_heads;
    const int32_t ne13 = 1;
    const int32_t nb11 = (int32_t)(head_dim * (int64_t)sizeof(__half));
    const int32_t nb12 = (int32_t)(head_dim * keys * (int64_t)sizeof(__half));    // ggml: nb[2] = ne[1]*nb[1]
    const int32_t nb13 = (int32_t)(head_dim * keys * num_kv_heads * (int64_t)sizeof(__half));
    const int32_t nb21 = nb11, nb22 = nb12;
    const int64_t nb23 = nb13;

    const int32_t ne31 = (int32_t)tokens;
    const int32_t ne32 = 1, ne33 = 1;
    const int32_t nb31 = (int32_t)(keys * (int64_t)sizeof(__half));
    const int32_t nb32 = 0;
    const int64_t nb33 = 0;

    const uint32_t n_head      = (uint32_t)num_heads;
    const uint32_t n_head_log2 = 1u << (uint32_t)(31 - __builtin_clz(n_head));

    const int ntiles_x     = (int)((tokens + kNcols1 - 1) / kNcols1);
    const int gqa_ratio    = (int)(num_heads / num_kv_heads);
    const int ntiles_z_gqa = (gqa_ratio + kNcols2 - 1) / kNcols2;

    const dim3 grid((unsigned)ntiles_x, 1u, (unsigned)(ntiles_z_gqa * num_kv_heads));
    const dim3 block((unsigned)kWarpSize, (unsigned)kNwarps, 1u);

    hipLaunchKernelGGL(
        (flash_attn_tile<kHeadDim, kHeadDim, kNcols1, kNcols2, false>),
        grid, block, 0, s,
        (const char*)q_f32, (const char*)k_f16, (const char*)v_f16,
        (const char*)m_f16,
        (const char*)nullptr,        // sinks (unused by Gemma)
        (const int*)nullptr,         // KV_max -> k_VKQ_max = ne11 = keys
        o_f32, (float2*)nullptr,      // dst, dst_meta (meta unused at grid.y==1)
        scale, 0.0f, 1.0f, 1.0f, n_head_log2, 0.0f,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23,
        ne31, ne32, ne33, nb31, nb32, nb33);

    if (hipGetLastError() != hipSuccess) return 8;

    k_f32_to_bf16<<<grid_for(q_elems), 256, 0, s>>>(o_f32, (uint16_t*)out, q_elems);
    return hipGetLastError() == hipSuccess ? 0 : 9;
}