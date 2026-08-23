// Comparative gfx11 arithmetic ceiling screen for MTP verifier research.
//
// This is an instruction-throughput harness, not an operation-complete GEMM.
// One wave32 workgroup runs several independent accumulator chains so latency
// does not masquerade as throughput. Inputs stay in registers; HIP events time
// only the kernel launch. The Python driver owns compilation and artifacts.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using f16x16_t = _Float16 __attribute__((ext_vector_type(16)));
using bf16x16_t = short __attribute__((ext_vector_type(16)));
using f32x8_t = float __attribute__((ext_vector_type(8)));
using i32x2_t = int __attribute__((ext_vector_type(2)));
using i32x4_t = int __attribute__((ext_vector_type(4)));
using i32x8_t = int __attribute__((ext_vector_type(8)));

#define HIP_CHECK(call)                                                                    \
    do {                                                                                   \
        const hipError_t status_ = (call);                                                  \
        if (status_ != hipSuccess) {                                                        \
            std::fprintf(stderr, "%s failed: %s\n", #call, hipGetErrorString(status_));   \
            std::exit(2);                                                                  \
        }                                                                                  \
    } while (0)

enum class Lane : int {
    fp16_wmma,
    bf16_wmma,
    i8i8_wmma,
    u8s8_wmma,
    i4i4_wmma,
    u4s4_wmma,
    u4u4_wmma,
    u8s8_dot4,
    u4s4_dot8,
};

template <int Chains>
__global__ __launch_bounds__(32) void fp16_wmma_kernel(float* output, int iterations, uint32_t seed) {
    const int lane = static_cast<int>(threadIdx.x);
    f16x16_t a{};
    f16x16_t b{};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int av = static_cast<int>((seed + lane + i) & 7u) + 1;
        const int bv = static_cast<int>((seed + lane * 3 + i * 5) & 7u) + 1;
        a[i] = static_cast<_Float16>(static_cast<float>(av) * (1.0f / 1024.0f));
        b[i] = static_cast<_Float16>(static_cast<float>(bv) * (1.0f / 1024.0f));
    }
    f32x8_t acc[Chains];
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[chain][i] = static_cast<float>(chain + i) * 1.0e-7f;
    }
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
        for (int chain = 0; chain < Chains; ++chain) {
            acc[chain] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[chain]);
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) sum += acc[chain][lane & 7];
    output[static_cast<size_t>(blockIdx.x) * 32 + lane] = sum;
}

template <int Chains>
__global__ __launch_bounds__(32) void bf16_wmma_kernel(float* output, int iterations, uint32_t seed) {
    const int lane = static_cast<int>(threadIdx.x);
    bf16x16_t a{};
    bf16x16_t b{};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        // Exact positive BF16 powers near 2^-10, represented as raw bits.
        a[i] = static_cast<short>(0x3a80u + (((seed + lane + i) & 3u) << 7));
        b[i] = static_cast<short>(0x3a80u + (((seed + lane * 3 + i * 5) & 3u) << 7));
    }
    f32x8_t acc[Chains];
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[chain][i] = static_cast<float>(chain + i) * 1.0e-7f;
    }
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
        for (int chain = 0; chain < Chains; ++chain) {
            acc[chain] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[chain]);
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) sum += acc[chain][lane & 7];
    output[static_cast<size_t>(blockIdx.x) * 32 + lane] = sum;
}

template <int Chains, bool SignedA, bool SignedB, bool FourBit>
__global__ __launch_bounds__(32) void integer_wmma_kernel(int* output, int iterations, uint32_t seed) {
    const int lane = static_cast<int>(threadIdx.x);
    i32x8_t acc[Chains];
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[chain][i] = chain * 17 + i;
    }
    if constexpr (FourBit) {
        i32x2_t a{};
        i32x2_t b{};
        a[0] = static_cast<int>(0x12345678u ^ seed ^ static_cast<uint32_t>(lane));
        a[1] = static_cast<int>(0x23456781u ^ (seed << 1) ^ static_cast<uint32_t>(lane * 3));
        b[0] = static_cast<int>(0xfedcba98u ^ (seed << 2) ^ static_cast<uint32_t>(lane * 5));
        b[1] = static_cast<int>(0xedcba987u ^ (seed << 3) ^ static_cast<uint32_t>(lane * 7));
#pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
            for (int chain = 0; chain < Chains; ++chain) {
                acc[chain] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    SignedA, a, SignedB, b, acc[chain], false);
            }
        }
    } else {
        i32x4_t a{};
        i32x4_t b{};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[i] = static_cast<int>((0x12345678u + static_cast<uint32_t>(i) * 0x11111111u) ^
                                    seed ^ static_cast<uint32_t>(lane * (i + 1)));
            b[i] = static_cast<int>((0xfedcba98u - static_cast<uint32_t>(i) * 0x11111111u) ^
                                    (seed << 1) ^ static_cast<uint32_t>(lane * (i + 3)));
        }
#pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
            for (int chain = 0; chain < Chains; ++chain) {
                acc[chain] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    SignedA, a, SignedB, b, acc[chain], false);
            }
        }
    }
    int sink = 0;
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) sink ^= acc[chain][lane & 7];
    output[static_cast<size_t>(blockIdx.x) * 32 + lane] = sink;
}

template <int Chains, bool FourBit>
__global__ __launch_bounds__(32) void integer_dot_kernel(int* output, int iterations, uint32_t seed) {
    const int lane = static_cast<int>(threadIdx.x);
    const int a = static_cast<int>(0x12345678u ^ seed ^ static_cast<uint32_t>(lane));
    const int b = static_cast<int>(0xfedcba98u ^ (seed << 1) ^ static_cast<uint32_t>(lane * 3));
    int acc[Chains];
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) acc[chain] = chain * 17 + lane;
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
        for (int chain = 0; chain < Chains; ++chain) {
            if constexpr (FourBit) {
                acc[chain] = __builtin_amdgcn_sudot8(false, a, true, b, acc[chain], false);
            } else {
                acc[chain] = __builtin_amdgcn_sudot4(false, a, true, b, acc[chain], false);
            }
        }
    }
    int sink = 0;
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) sink ^= acc[chain];
    output[static_cast<size_t>(blockIdx.x) * 32 + lane] = sink;
}

template <int Chains>
void launch_lane(Lane lane, void* output, int blocks, int iterations, uint32_t seed, hipStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(blocks));
    const dim3 block(32);
    switch (lane) {
        case Lane::fp16_wmma:
            hipLaunchKernelGGL((fp16_wmma_kernel<Chains>), grid, block, 0, stream,
                               static_cast<float*>(output), iterations, seed);
            break;
        case Lane::bf16_wmma:
            hipLaunchKernelGGL((bf16_wmma_kernel<Chains>), grid, block, 0, stream,
                               static_cast<float*>(output), iterations, seed);
            break;
        case Lane::i8i8_wmma:
            hipLaunchKernelGGL((integer_wmma_kernel<Chains, true, true, false>), grid, block, 0, stream,
                               static_cast<int*>(output), iterations, seed);
            break;
        case Lane::u8s8_wmma:
            hipLaunchKernelGGL((integer_wmma_kernel<Chains, false, true, false>), grid, block, 0, stream,
                               static_cast<int*>(output), iterations, seed);
            break;
        case Lane::i4i4_wmma:
            hipLaunchKernelGGL((integer_wmma_kernel<Chains, true, true, true>), grid, block, 0, stream,
                               static_cast<int*>(output), iterations, seed);
            break;
        case Lane::u4s4_wmma:
            hipLaunchKernelGGL((integer_wmma_kernel<Chains, false, true, true>), grid, block, 0, stream,
                               static_cast<int*>(output), iterations, seed);
            break;
        case Lane::u4u4_wmma:
            hipLaunchKernelGGL((integer_wmma_kernel<Chains, false, false, true>), grid, block, 0, stream,
                               static_cast<int*>(output), iterations, seed);
            break;
        case Lane::u8s8_dot4:
            hipLaunchKernelGGL((integer_dot_kernel<Chains, false>), grid, block, 0, stream,
                               static_cast<int*>(output), iterations, seed);
            break;
        case Lane::u4s4_dot8:
            hipLaunchKernelGGL((integer_dot_kernel<Chains, true>), grid, block, 0, stream,
                               static_cast<int*>(output), iterations, seed);
            break;
    }
    HIP_CHECK(hipGetLastError());
}

void launch(Lane lane, int chains, void* output, int blocks, int iterations, uint32_t seed, hipStream_t stream) {
    switch (chains) {
        case 2: launch_lane<2>(lane, output, blocks, iterations, seed, stream); break;
        case 4: launch_lane<4>(lane, output, blocks, iterations, seed, stream); break;
        case 8: launch_lane<8>(lane, output, blocks, iterations, seed, stream); break;
        default:
            std::fprintf(stderr, "chains must be one of 2, 4, 8\n");
            std::exit(2);
    }
}

const char* lane_name(Lane lane) {
    switch (lane) {
        case Lane::fp16_wmma: return "fp16_wmma";
        case Lane::bf16_wmma: return "bf16_wmma";
        case Lane::i8i8_wmma: return "i8i8_wmma";
        case Lane::u8s8_wmma: return "u8s8_wmma";
        case Lane::i4i4_wmma: return "i4i4_wmma";
        case Lane::u4s4_wmma: return "u4s4_wmma";
        case Lane::u4u4_wmma: return "u4u4_wmma";
        case Lane::u8s8_dot4: return "u8s8_dot4";
        case Lane::u4s4_dot8: return "u4s4_dot8";
    }
    return "unknown";
}

bool is_wmma(Lane lane) {
    return lane != Lane::u8s8_dot4 && lane != Lane::u4s4_dot8;
}

double operations_per_wave_instruction(Lane lane) {
    if (is_wmma(lane)) return 2.0 * 16.0 * 16.0 * 16.0;
    if (lane == Lane::u8s8_dot4) return 32.0 * 2.0 * 4.0;
    return 32.0 * 2.0 * 8.0;
}

int parse_int_arg(int argc, char** argv, const char* name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
    }
    return fallback;
}

int main(int argc, char** argv) {
    const int iterations = parse_int_arg(argc, argv, "--iterations", 65536);
    const int samples = parse_int_arg(argc, argv, "--samples", 7);
    const int warmups = parse_int_arg(argc, argv, "--warmups", 2);
    if (iterations <= 0 || samples <= 0 || warmups < 0) {
        std::fprintf(stderr, "iterations/samples must be positive and warmups non-negative\n");
        return 2;
    }

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    const int blocks_arg = parse_int_arg(argc, argv, "--blocks", 0);
    const int blocks = blocks_arg > 0 ? blocks_arg : properties.multiProcessorCount * 16;
    if (blocks <= 0) {
        std::fprintf(stderr, "computed block count must be positive\n");
        return 2;
    }

    void* output = nullptr;
    HIP_CHECK(hipMalloc(&output, static_cast<size_t>(blocks) * 32 * sizeof(int)));
    hipStream_t stream{};
    hipEvent_t start{};
    hipEvent_t stop{};
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    constexpr Lane lanes[] = {
        Lane::fp16_wmma,
        Lane::bf16_wmma,
        Lane::i8i8_wmma,
        Lane::u8s8_wmma,
        Lane::i4i4_wmma,
        Lane::u4s4_wmma,
        Lane::u4u4_wmma,
        Lane::u8s8_dot4,
        Lane::u4s4_dot8,
    };
    constexpr int chain_counts[] = {2, 4, 8};

    std::printf("{\n");
    std::printf("  \"schema_version\": 1,\n");
    std::printf("  \"device\": {\"name\": \"%s\", \"arch\": \"%s\", \"multiprocessor_count\": %d, \"clock_rate_khz\": %d},\n",
                properties.name, properties.gcnArchName, properties.multiProcessorCount, properties.clockRate);
    std::printf("  \"parameters\": {\"blocks\": %d, \"threads\": 32, \"iterations\": %d, \"warmups\": %d, \"samples\": %d},\n",
                blocks, iterations, warmups, samples);
    std::printf("  \"measurements\": [\n");

    bool first_measurement = true;
    uint32_t seed = 0x31415926u;
    for (Lane lane : lanes) {
        for (int chains : chain_counts) {
            for (int warmup = 0; warmup < warmups; ++warmup) {
                launch(lane, chains, output, blocks, iterations, seed + static_cast<uint32_t>(warmup), stream);
            }
            HIP_CHECK(hipStreamSynchronize(stream));

            if (!first_measurement) std::printf(",\n");
            first_measurement = false;
            const double instruction_ops = operations_per_wave_instruction(lane);
            const double total_ops = static_cast<double>(blocks) * static_cast<double>(iterations) *
                                     static_cast<double>(chains) * instruction_ops;
            std::printf("    {\"lane\": \"%s\", \"chains\": %d, \"instruction_ops\": %.0f, \"total_ops\": %.0f, \"milliseconds\": [",
                        lane_name(lane), chains, instruction_ops, total_ops);
            for (int sample = 0; sample < samples; ++sample) {
                HIP_CHECK(hipEventRecord(start, stream));
                launch(lane, chains, output, blocks, iterations,
                       seed + static_cast<uint32_t>(sample + warmups), stream);
                HIP_CHECK(hipEventRecord(stop, stream));
                HIP_CHECK(hipEventSynchronize(stop));
                float milliseconds = 0.0f;
                HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
                if (sample != 0) std::printf(", ");
                std::printf("%.6f", static_cast<double>(milliseconds));
            }
            std::printf("]}");
            seed += 0x9e3779b9u;
        }
    }
    std::printf("\n  ]\n}\n");

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(output));
    return 0;
}
