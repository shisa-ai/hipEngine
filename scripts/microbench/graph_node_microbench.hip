// HIP side of the v2 narrow dispatch/grid-floor timing contract.
//
// A trivial kernel (one block, writes memory so it cannot be elided) is launched
// back-to-back `count` times from a C loop. Issuing the whole burst from C removes
// all Python/ctypes per-node overhead, so the wall time measures the *pure HIP*
// host-issue + GPU-dispatch cost of N serial launches.
//
// The matched serial path replays a pre-recorded graph, while the independent
// path partitions disjoint outputs over nonblocking streams.
//
// Not a production kernel: no registry entry, no KVLiveSpans ABI. Diagnostic only.

#include <hip/hip_runtime.h>
#include <stdint.h>

#include <algorithm>
#include <exception>
#include <vector>

#include "../../benchmarks/micro/runners/micro_timing_hip.hpp"

extern "C" __global__ void gmb_noop_kernel(float* __restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Real (tiny) memory write so the launch cannot be optimized away.
    out[idx] = out[idx] + 1.0f;
  }
}

// Wide-argument variant: same trivial compute, but the kernel takes 16 arguments
// (8 pointers + 8 int64) so hipLaunchKernelGGL must marshal a realistically large
// kernel-arg buffer per launch — matching the hot W4 GEMV / fused launchers, which
// carry ~10-20 args, rather than the 2-arg trivial kernel above. All pointer args
// alias the same buffer; the ints are touched so they cannot be dead-code-elided.
extern "C" __global__ void gmb_wide_kernel(
    float* __restrict__ p0,
    const float* __restrict__ p1,
    const float* __restrict__ p2,
    const float* __restrict__ p3,
    const float* __restrict__ p4,
    const float* __restrict__ p5,
    const float* __restrict__ p6,
    const float* __restrict__ p7,
    int64_t a0,
    int64_t a1,
    int64_t a2,
    int64_t a3,
    int64_t a4,
    int64_t a5,
    int64_t a6,
    int64_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Touch every arg so none is elided (forces real marshaling of all 16).
    float acc = p0[idx] + p1[idx] + p2[idx] + p3[idx] + p4[idx] + p5[idx] + p6[idx] + p7[idx];
    int64_t bits = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6;
    p0[idx] = acc + static_cast<float>(bits & 1);
  }
}

// grid_blocks <= 0 -> auto (ceil(n/256)); otherwise force that many workgroups
// so the kernel's launch/dispatch cost can be measured against grid size while
// keeping the compute trivial (blocks past ceil(n/256) return immediately).
static inline unsigned int gmb_grid(int n, int grid_blocks) {
  if (grid_blocks > 0) {
    return static_cast<unsigned int>(grid_blocks);
  }
  constexpr int kBlock = 256;
  unsigned int grid_x = static_cast<unsigned int>((n + kBlock - 1) / kBlock);
  return grid_x == 0 ? 1u : grid_x;
}

extern "C" int gmb_launch_wide_n(float* buf, int n, int count, int grid_blocks, hipStream_t stream) {
  if (n < 0 || count < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kBlock = 256;
  unsigned int grid_x = gmb_grid(n, grid_blocks);
  for (int i = 0; i < count; ++i) {
    hipLaunchKernelGGL(
        gmb_wide_kernel,
        dim3(grid_x),
        dim3(kBlock),
        0,
        stream,
        buf, buf, buf, buf, buf, buf, buf, buf,
        static_cast<int64_t>(i + 1),
        static_cast<int64_t>(2),
        static_cast<int64_t>(3),
        static_cast<int64_t>(4),
        static_cast<int64_t>(5),
        static_cast<int64_t>(6),
        static_cast<int64_t>(7),
        static_cast<int64_t>(n));
  }
  return static_cast<int>(hipGetLastError());
}

// Issue `count` back-to-back launches on `stream`. When called inside a stream
// capture, this records `count` kernel nodes with fixed arguments.
extern "C" int gmb_launch_n(float* out, int n, int count, int grid_blocks, hipStream_t stream) {
  if (n < 0 || count < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kBlock = 256;
  unsigned int grid_x = gmb_grid(n, grid_blocks);
  for (int i = 0; i < count; ++i) {
    hipLaunchKernelGGL(
        gmb_noop_kernel,
        dim3(grid_x),
        dim3(kBlock),
        0,
        stream,
        out,
        n);
  }
  return static_cast<int>(hipGetLastError());
}

namespace {

struct ScopedGraph {
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;

  ~ScopedGraph() {
    if (exec != nullptr) {
      (void)hipGraphExecDestroy(exec);
    }
    if (graph != nullptr) {
      (void)hipGraphDestroy(graph);
    }
  }
};

void capture_graph(
    ScopedGraph& captured,
    float* out,
    int n,
    int count,
    int grid_blocks) {
  hipStream_t stream = nullptr;
  hipengine::micro::timing_hip_check(
      hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
      "hipStreamCreateWithFlags capture");
  try {
    hipengine::micro::timing_hip_check(
        hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed),
        "hipStreamBeginCapture");
    hipengine::micro::timing_hip_check(
        static_cast<hipError_t>(gmb_launch_n(out, n, count, grid_blocks, stream)),
        "gmb_launch_n capture");
    hipengine::micro::timing_hip_check(
        hipStreamEndCapture(stream, &captured.graph),
        "hipStreamEndCapture");
    hipGraphNode_t error_node = nullptr;
    char log_buffer[4096] = {};
    hipengine::micro::timing_hip_check(
        hipGraphInstantiate(
            &captured.exec,
            captured.graph,
            &error_node,
            log_buffer,
            sizeof(log_buffer)),
        "hipGraphInstantiate");
  } catch (...) {
    (void)hipStreamDestroy(stream);
    throw;
  }
  hipengine::micro::timing_hip_check(
      hipStreamDestroy(stream), "hipStreamDestroy capture");
}

void zero_output(float* out, size_t elements) {
  hipengine::micro::timing_hip_check(
      hipMemset(out, 0, elements * sizeof(float)), "hipMemset output");
  hipengine::micro::timing_hip_check(
      hipDeviceSynchronize(), "hipDeviceSynchronize after memset");
}

uint32_t copy_and_count_mismatches(
    const float* out,
    size_t elements,
    float expected) {
  std::vector<float> host(elements);
  hipengine::micro::timing_hip_check(
      hipMemcpy(
          host.data(),
          out,
          elements * sizeof(float),
          hipMemcpyDeviceToHost),
      "hipMemcpy validation output");
  return static_cast<uint32_t>(std::count_if(
      host.begin(), host.end(), [expected](float value) { return value != expected; }));
}

void copy_timing_samples(
    const hipengine::micro::TimingSamples& samples,
    double* gpu_us,
    double* host_us) {
  std::copy(samples.gpu_sequence_us.begin(), samples.gpu_sequence_us.end(), gpu_us);
  std::copy(samples.host_sequence_us.begin(), samples.host_sequence_us.end(), host_us);
}

}  // namespace

extern "C" int gmb_measure_graph_contract(
    float* out,
    int n,
    int count,
    int grid_blocks,
    int samples,
    int warmup,
    double* single_gpu_us,
    double* single_host_us,
    double* burst_gpu_us,
    double* burst_host_us,
    uint32_t* single_mismatches,
    uint32_t* burst_mismatches) {
  if (out == nullptr || n <= 0 || count <= 0 || samples <= 0 || warmup < 0 ||
      single_gpu_us == nullptr || single_host_us == nullptr || burst_gpu_us == nullptr ||
      burst_host_us == nullptr || single_mismatches == nullptr ||
      burst_mismatches == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  try {
    ScopedGraph single_graph;
    ScopedGraph burst_graph;
    capture_graph(single_graph, out, n, 1, grid_blocks);
    capture_graph(burst_graph, out, n, count, grid_blocks);
    hipengine::micro::HipSequenceTimer timer(
        hipengine::micro::TimingMode::SerialLatency, 1);

    auto run_graph = [](hipGraphExec_t exec, hipStream_t stream) {
      hipengine::micro::timing_hip_check(
          hipGraphLaunch(exec, stream), "hipGraphLaunch");
    };
    zero_output(out, static_cast<size_t>(n));
    timer.run_and_wait(1, [&](uint32_t, hipStream_t stream) {
      run_graph(single_graph.exec, stream);
    });
    *single_mismatches = copy_and_count_mismatches(out, n, 1.0f);

    zero_output(out, static_cast<size_t>(n));
    timer.run_and_wait(1, [&](uint32_t, hipStream_t stream) {
      run_graph(burst_graph.exec, stream);
    });
    *burst_mismatches =
        copy_and_count_mismatches(out, n, static_cast<float>(count));

    if (warmup > 0) {
      timer.run_and_wait(static_cast<uint32_t>(warmup), [&](uint32_t, hipStream_t stream) {
        run_graph(single_graph.exec, stream);
      });
    }
    auto single = timer.measure(1, static_cast<uint32_t>(samples), [&](uint32_t, hipStream_t stream) {
      run_graph(single_graph.exec, stream);
    });
    auto burst = timer.measure(1, static_cast<uint32_t>(samples), [&](uint32_t, hipStream_t stream) {
      run_graph(burst_graph.exec, stream);
    });
    copy_timing_samples(single, single_gpu_us, single_host_us);
    copy_timing_samples(burst, burst_gpu_us, burst_host_us);
    return static_cast<int>(hipSuccess);
  } catch (const std::exception&) {
    return static_cast<int>(hipErrorUnknown);
  }
}

extern "C" int gmb_measure_independent_contract(
    float* out,
    int n,
    int count,
    int grid_blocks,
    int independent_streams,
    int samples,
    int warmup,
    double* single_gpu_us,
    double* single_host_us,
    double* burst_gpu_us,
    double* burst_host_us,
    uint32_t* single_mismatches,
    uint32_t* burst_mismatches) {
  if (out == nullptr || n <= 0 || count <= 0 || independent_streams <= 0 ||
      samples <= 0 || warmup < 0 || single_gpu_us == nullptr ||
      single_host_us == nullptr || burst_gpu_us == nullptr ||
      burst_host_us == nullptr || single_mismatches == nullptr ||
      burst_mismatches == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  try {
    const uint32_t stream_count = static_cast<uint32_t>(
        std::min(count, independent_streams));
    hipengine::micro::HipSequenceTimer timer(
        hipengine::micro::TimingMode::IndependentThroughput, stream_count);
    const unsigned int grid_x = gmb_grid(n, grid_blocks);
    auto launch = [&](uint32_t rep, hipStream_t stream) {
      hipLaunchKernelGGL(
          gmb_noop_kernel,
          dim3(grid_x),
          dim3(256),
          0,
          stream,
          out + static_cast<size_t>(rep) * static_cast<size_t>(n),
          n);
      hipengine::micro::timing_hip_check(
          hipGetLastError(), "gmb_noop_kernel launch");
    };

    zero_output(out, static_cast<size_t>(count) * static_cast<size_t>(n));
    timer.run_and_wait(1, launch);
    *single_mismatches = copy_and_count_mismatches(out, n, 1.0f);

    zero_output(out, static_cast<size_t>(count) * static_cast<size_t>(n));
    timer.run_and_wait(static_cast<uint32_t>(count), launch);
    *burst_mismatches = copy_and_count_mismatches(
        out, static_cast<size_t>(count) * static_cast<size_t>(n), 1.0f);

    if (warmup > 0) {
      timer.run_and_wait(static_cast<uint32_t>(warmup), launch);
    }
    auto single = timer.measure(1, static_cast<uint32_t>(samples), launch);
    auto burst = timer.measure(
        static_cast<uint32_t>(count), static_cast<uint32_t>(samples), launch);
    copy_timing_samples(single, single_gpu_us, single_host_us);
    copy_timing_samples(burst, burst_gpu_us, burst_host_us);
    return static_cast<int>(hipSuccess);
  } catch (const std::exception&) {
    return static_cast<int>(hipErrorUnknown);
  }
}
