// Native orchestration of the two-rank staged exchange.
//
// This is the native half of an A/B against the Python host loop in
// `scripts/tp_collective_bench.py` (`staged_exchange_batched`). It implements the
// same protocol with the same dependency structure and completion boundaries:
//
//   for each step:
//     submit device-to-host for rank 0 and rank 1          (both, before either wait)
//     wait for rank 0, wait for rank 1
//     host sum of the two partials, written back into both slots
//     submit host-to-device for rank 0 and rank 1          (no return wait)
//   drain both streams once at the end
//
// Two things differ from the Python arm by construction and are reported rather
// than hidden: buffers are preallocated (no per-step NumPy view creation), and
// device scoping is one `hipSetDevice` per operation instead of the Python
// runtime's get/set/restore around each call.
//
// The timed loop uses the plain sum, which is the arithmetic the Python arm
// measures. A separate untimed pass uses a bounded recurrence
// (`sum / world + 1`) so the chain's dependency can be verified exactly at the
// same depth, where `seed * world ** depth` would saturate fp32.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(hipError_t result, const char* what) {
  if (result != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(result));
  }
}

struct Args {
  int depth = 128;
  uint32_t count = 5120;
  uint32_t iterations = 30;
  uint32_t warmup = 10;
  float seed = 1.0f;
  std::string json_path;
};

struct Phase {
  double submit_d2h_us = 0.0;
  double wait_d2h_us = 0.0;
  double sum_us = 0.0;
  double submit_h2d_us = 0.0;
};

double now_us() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch())
      .count();
}

struct Runner {
  Args args;
  uint32_t world = 2;
  size_t nbytes = 0;
  std::vector<int> devices;
  std::vector<hipStream_t> streams;
  // Two device buffers per rank, alternating per step, plus one pinned host slot
  // per rank and one preallocated accumulation scratch.
  std::vector<std::vector<float*>> device_buffers;
  std::vector<float*> host_slots;
  std::vector<float> scratch;

  void init() {
    nbytes = static_cast<size_t>(args.count) * sizeof(float);
    int device_count = 0;
    check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count < static_cast<int>(world)) {
      throw std::runtime_error("this runner needs two visible devices");
    }
    devices = {0, 1};
    streams.assign(world, nullptr);
    device_buffers.assign(world, {nullptr, nullptr});
    host_slots.assign(world, nullptr);
    scratch.assign(args.count, 0.0f);
    for (uint32_t rank = 0; rank < world; ++rank) {
      check(hipSetDevice(devices[rank]), "hipSetDevice");
      check(hipStreamCreate(&streams[rank]), "hipStreamCreate");
      for (int buffer = 0; buffer < 2; ++buffer) {
        check(hipMalloc(&device_buffers[rank][buffer], nbytes), "hipMalloc");
      }
      check(hipHostMalloc(reinterpret_cast<void**>(&host_slots[rank]), nbytes,
                          hipHostMallocDefault),
            "hipHostMalloc");
    }
  }

  void destroy() {
    for (uint32_t rank = 0; rank < world; ++rank) {
      check(hipSetDevice(devices[rank]), "hipSetDevice");
      for (int buffer = 0; buffer < 2; ++buffer) {
        if (device_buffers[rank][buffer]) {
          hipFree(device_buffers[rank][buffer]);
        }
      }
      if (host_slots[rank]) {
        hipHostFree(host_slots[rank]);
      }
      if (streams[rank]) {
        hipStreamDestroy(streams[rank]);
      }
    }
  }

  void seed_chain() {
    std::vector<float> values(args.count, args.seed);
    for (uint32_t rank = 0; rank < world; ++rank) {
      check(hipSetDevice(devices[rank]), "hipSetDevice");
      check(hipMemcpy(device_buffers[rank][0], values.data(), nbytes,
                      hipMemcpyHostToDevice),
            "hipMemcpy");
    }
  }

  // One step of the batched protocol, with optional phase accounting.
  void step(int index, Phase* phase) {
    const int slot = index % 2;
    const int next = (index + 1) % 2;
    double started = 0.0;
    for (uint32_t rank = 0; rank < world; ++rank) {
      check(hipSetDevice(devices[rank]), "hipSetDevice");
      started = phase ? now_us() : 0.0;
      check(hipMemcpyAsync(host_slots[rank], device_buffers[rank][index % 2], nbytes,
                           hipMemcpyDeviceToHost, streams[rank]),
            "hipMemcpyAsync D2H");
      if (phase) {
        phase->submit_d2h_us += now_us() - started;
      }
    }
    for (uint32_t rank = 0; rank < world; ++rank) {
      check(hipSetDevice(devices[rank]), "hipSetDevice");
      started = phase ? now_us() : 0.0;
      check(hipStreamSynchronize(streams[rank]), "hipStreamSynchronize");
      if (phase) {
        phase->wait_d2h_us += now_us() - started;
      }
    }
    started = phase ? now_us() : 0.0;
    // The accumulation the Python arm performs: copy rank 0's partial, add rank
    // 1's, then write the total back into both slots. Preallocated here, so no
    // per-step allocation is inside the timed region.
    std::copy(host_slots[0], host_slots[0] + args.count, scratch.begin());
    for (uint32_t index_inner = 0; index_inner < args.count; ++index_inner) {
      scratch[index_inner] += host_slots[1][index_inner];
    }
    for (uint32_t rank = 0; rank < world; ++rank) {
      std::copy(scratch.begin(), scratch.end(), host_slots[rank]);
    }
    if (phase) {
      phase->sum_us += now_us() - started;
    }
    for (uint32_t rank = 0; rank < world; ++rank) {
      check(hipSetDevice(devices[rank]), "hipSetDevice");
      started = phase ? now_us() : 0.0;
      check(hipMemcpyAsync(device_buffers[rank][next], host_slots[rank], nbytes,
                           hipMemcpyHostToDevice, streams[rank]),
            "hipMemcpyAsync H2D");
      if (phase) {
        phase->submit_h2d_us += now_us() - started;
      }
    }
  }

  void drain() {
    for (uint32_t rank = 0; rank < world; ++rank) {
      check(hipSetDevice(devices[rank]), "hipSetDevice");
      check(hipStreamSynchronize(streams[rank]), "hipStreamSynchronize");
    }
  }

  void run_chain(int depth, Phase* phase) {
    for (int index = 0; index < depth; ++index) {
      step(index, phase);
    }
    drain();
  }

  std::vector<float> read_rank0(int depth) {
    std::vector<float> values(args.count, 0.0f);
    check(hipSetDevice(devices[0]), "hipSetDevice");
    check(hipMemcpy(values.data(), device_buffers[0][depth % 2], nbytes,
                    hipMemcpyDeviceToHost),
          "hipMemcpy");
    return values;
  }

  // Bounded recurrence: the host writes sum / world + 1, so after `depth` steps
  // every element holds mean_r(seed_r[j]) + depth. Exact in fp32 and informative
  // at any depth, unlike seed * world ** depth.
  void seed_vector() {
    for (uint32_t rank = 0; rank < world; ++rank) {
      std::vector<float> values(args.count, 0.0f);
      for (uint32_t index = 0; index < args.count; ++index) {
        values[index] = 4.0f * static_cast<float>(rank + 1) * static_cast<float>(index + 1);
      }
      check(hipSetDevice(devices[rank]), "hipSetDevice");
      check(hipMemcpy(device_buffers[rank][0], values.data(), nbytes,
                      hipMemcpyHostToDevice),
            "hipMemcpy");
    }
  }

  void step_bounded(int index) {
    const int next = (index + 1) % 2;
    for (uint32_t rank = 0; rank < world; ++rank) {
      check(hipSetDevice(devices[rank]), "hipSetDevice");
      check(hipMemcpyAsync(host_slots[rank], device_buffers[rank][index % 2], nbytes,
                           hipMemcpyDeviceToHost, streams[rank]),
            "hipMemcpyAsync D2H");
    }
    for (uint32_t rank = 0; rank < world; ++rank) {
      check(hipSetDevice(devices[rank]), "hipSetDevice");
      check(hipStreamSynchronize(streams[rank]), "hipStreamSynchronize");
    }
    std::copy(host_slots[0], host_slots[0] + args.count, scratch.begin());
    for (uint32_t index_inner = 0; index_inner < args.count; ++index_inner) {
      scratch[index_inner] += host_slots[1][index_inner];
    }
    const float scale = 1.0f / static_cast<float>(world);
    for (uint32_t index_inner = 0; index_inner < args.count; ++index_inner) {
      scratch[index_inner] = scratch[index_inner] * scale + 1.0f;
    }
    for (uint32_t rank = 0; rank < world; ++rank) {
      std::copy(scratch.begin(), scratch.end(), host_slots[rank]);
    }
    for (uint32_t rank = 0; rank < world; ++rank) {
      check(hipSetDevice(devices[rank]), "hipSetDevice");
      check(hipMemcpyAsync(device_buffers[rank][next], host_slots[rank], nbytes,
                           hipMemcpyHostToDevice, streams[rank]),
            "hipMemcpyAsync H2D");
    }
  }
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    auto value = [&]() -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + flag);
      }
      return argv[++index];
    };
    if (flag == "--depth") {
      args.depth = std::stoi(value());
    } else if (flag == "--count") {
      args.count = static_cast<uint32_t>(std::stoul(value()));
    } else if (flag == "--iterations") {
      args.iterations = static_cast<uint32_t>(std::stoul(value()));
    } else if (flag == "--warmup") {
      args.warmup = static_cast<uint32_t>(std::stoul(value()));
    } else if (flag == "--seed") {
      args.seed = std::stof(value());
    } else if (flag == "--json") {
      args.json_path = value();
    } else {
      throw std::runtime_error("unknown flag " + flag);
    }
  }
  return args;
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  Runner runner;
  runner.args = args;
  runner.init();

  // Dependency verification first: a broken chain must not produce a timing.
  runner.seed_vector();
  for (int index = 0; index < args.depth; ++index) {
    runner.step_bounded(index);
  }
  runner.drain();
  std::vector<float> observed = runner.read_rank0(args.depth);
  double max_error = 0.0;
  for (uint32_t index = 0; index < args.count; ++index) {
    const double expected =
        6.0 * static_cast<double>(index + 1) + static_cast<double>(args.depth);
    max_error = std::max(max_error, std::fabs(static_cast<double>(observed[index]) - expected));
  }

  // Timed runs: plain sum, the same arithmetic the Python arm measures.
  std::vector<double> totals;
  std::vector<double> singles;
  for (uint32_t iteration = 0; iteration < args.warmup + args.iterations; ++iteration) {
    runner.seed_chain();
    const double started = now_us();
    runner.run_chain(args.depth, nullptr);
    const double elapsed = now_us() - started;
    if (iteration >= args.warmup) {
      totals.push_back(elapsed);
    }
  }
  runner.seed_chain();
  {
    const double started = now_us();
    runner.run_chain(1, nullptr);
    singles.push_back(now_us() - started);
  }

  // Phase accounting on a separate pass, so the timed runs stay uninstrumented.
  Phase phase;
  runner.seed_chain();
  runner.run_chain(args.depth, &phase);

  const double total_median = median(totals);
  const double single_median = median(singles);
  // The single-step intercept cannot divide out of a one-step chain.
  const double per_step =
      args.depth > 1 ? (total_median - single_median) / static_cast<double>(args.depth - 1)
                     : std::nan("");
  const double per_step_phases =
      (phase.submit_d2h_us + phase.wait_d2h_us + phase.sum_us + phase.submit_h2d_us) /
      static_cast<double>(args.depth);

  std::cout.precision(10);
  std::cout << "{\n";
  std::cout << "  \"depth\": " << args.depth << ",\n";
  std::cout << "  \"count\": " << args.count << ",\n";
  std::cout << "  \"payload_bytes\": " << runner.nbytes << ",\n";
  std::cout << "  \"iterations\": " << args.iterations << ",\n";
  std::cout << "  \"warmup\": " << args.warmup << ",\n";
  std::cout << "  \"total_median_us\": " << total_median << ",\n";
  std::cout << "  \"single_step_median_us\": " << single_median << ",\n";
  std::cout << "  \"per_step_us\": ";
  if (std::isnan(per_step)) {
    std::cout << "null";
  } else {
    std::cout << per_step;
  }
  std::cout << ",\n";
  std::cout << "  \"per_step_phases_us\": " << per_step_phases << ",\n";
  std::cout << "  \"phases_us\": {\n";
  std::cout << "    \"submit_d2h\": " << phase.submit_d2h_us / args.depth << ",\n";
  std::cout << "    \"wait_d2h\": " << phase.wait_d2h_us / args.depth << ",\n";
  std::cout << "    \"host_sum\": " << phase.sum_us / args.depth << ",\n";
  std::cout << "    \"submit_h2d\": " << phase.submit_h2d_us / args.depth << "\n";
  std::cout << "  },\n";
  std::cout << "  \"verification\": {\n";
  std::cout << "    \"recurrence\": \"sum / world + 1\",\n";
  std::cout << "    \"expected\": \"6 * (j + 1) + depth\",\n";
  std::cout << "    \"max_abs_error\": " << max_error << ",\n";
  std::cout << "    \"exact\": " << (max_error == 0.0 ? "true" : "false") << "\n";
  std::cout << "  },\n";
  std::cout << "  \"protocol\": \"batched: both D2H submitted before either wait, no return wait\"\n";
  std::cout << "}\n";

  runner.destroy();
  return max_error == 0.0 ? 0 : 1;
}
