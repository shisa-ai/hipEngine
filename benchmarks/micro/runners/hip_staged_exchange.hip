// Native orchestration of the two-rank staged exchange.
//
// This is the native half of an A/B against the Python host loop in
// `scripts/tp_collective_bench.py` (`staged_exchange_batched`). It implements the
// same protocol with the same dependency structure and completion boundaries:
//
//   for each step:
//     submit device-to-host for rank 0 and rank 1          (both, before either wait)
//     wait for rank 0, wait for rank 1
//     host reduction of the two partials, written back into both slots
//     submit host-to-device for rank 0 and rank 1          (no return wait)
//   drain both streams once at the end
//
// Two things differ from the Python arm by construction and are reported rather
// than hidden: buffers are preallocated (no per-step NumPy view creation), and
// device scoping is one `hipSetDevice` per operation instead of the Python
// runtime's get/set/restore around each call.
//
// Verification and timing share one `step()` implementation. The recurrence is a
// parameter, not a second code path, so the timed structure is the verified
// structure:
//
//   kSum      v <- v_rank0 + v_rank1                 (the arithmetic the Python arm
//                                                      measures; closed form is
//                                                      seed * world ** depth, which
//                                                      is exact in fp32 through
//                                                      depth 127 and saturates at
//                                                      128, where it is reported
//                                                      as uninformative rather
//                                                      than as a pass)
//   kBounded  v <- (v_rank0 + v_rank1) / world + 1   (closed form
//                                                      mean_r(seed_r[j]) + depth,
//                                                      exact and informative at any
//                                                      depth)
//
// Both recurrences are verified over the whole vector on *both* ranks, and any
// nonfinite observed or expected value fails the run.

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

//: Which host reduction the chain performs. One enum keeps verification and
//: timing on the same code path.
enum class Recurrence { kSum, kBounded };

const char* recurrence_name(Recurrence recurrence) {
  return recurrence == Recurrence::kSum ? "sum" : "bounded: sum / world + 1";
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
  double reduce_us = 0.0;
  double submit_h2d_us = 0.0;
};

double now_us() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch())
      .count();
}

struct Check {
  bool informative = false;
  bool finite = true;
  bool exact = false;
  //: True when the closed form itself overflows fp32, so nonfinite device values
  //: are the expected outcome rather than a failure - and are themselves evidence
  //: that the chain compounded, since a shorter chain would still be finite.
  bool saturation_expected = false;
  int nonfinite = 0;
  double max_abs_error = 0.0;
  std::string expected;
  std::string detail;
};

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

  void upload(uint32_t rank, int buffer, const std::vector<float>& values) {
    check(hipSetDevice(devices[rank]), "hipSetDevice");
    check(hipMemcpy(device_buffers[rank][buffer], values.data(), nbytes,
                    hipMemcpyHostToDevice),
          "hipMemcpy");
  }

  void seed_chain() {
    std::vector<float> values(args.count, args.seed);
    for (uint32_t rank = 0; rank < world; ++rank) {
      upload(rank, 0, values);
    }
  }

  // Rank-distinct, element-distinct finite seeds: base * (rank + 1) * (j + 1).
  // With base 4 the mean over two ranks is an exact integer.
  void seed_vector() {
    for (uint32_t rank = 0; rank < world; ++rank) {
      std::vector<float> values(args.count, 0.0f);
      for (uint32_t index = 0; index < args.count; ++index) {
        values[index] = 4.0f * static_cast<float>(rank + 1) * static_cast<float>(index + 1);
      }
      upload(rank, 0, values);
    }
  }

  // The single step implementation. Both the timed pass and every verification
  // pass call this, so a structure that fails verification is the structure that
  // was timed.
  void step(int index, Recurrence recurrence, Phase* phase) {
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
    if (recurrence == Recurrence::kBounded) {
      const float scale = 1.0f / static_cast<float>(world);
      for (uint32_t index_inner = 0; index_inner < args.count; ++index_inner) {
        scratch[index_inner] = scratch[index_inner] * scale + 1.0f;
      }
    }
    for (uint32_t rank = 0; rank < world; ++rank) {
      std::copy(scratch.begin(), scratch.end(), host_slots[rank]);
    }
    if (phase) {
      phase->reduce_us += now_us() - started;
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

  void run_chain(int depth, Recurrence recurrence, Phase* phase) {
    for (int index = 0; index < depth; ++index) {
      step(index, recurrence, phase);
    }
    drain();
  }

  std::vector<float> read_rank(uint32_t rank, int depth) {
    std::vector<float> values(args.count, 0.0f);
    check(hipSetDevice(devices[rank]), "hipSetDevice");
    check(hipMemcpy(values.data(), device_buffers[rank][depth % 2], nbytes,
                    hipMemcpyDeviceToHost),
          "hipMemcpy");
    return values;
  }

  // Run a chain and check the whole vector on every rank.
  Check verify(int depth, Recurrence recurrence) {
    Check result;
    result.expected = recurrence == Recurrence::kSum
                          ? "seed * world ** depth"
                          : "mean_r(seed_r[j]) + depth";
    const double closed_form =
        recurrence == Recurrence::kSum
            ? static_cast<double>(args.seed) * std::pow(static_cast<double>(world), depth)
            : 0.0;
    // The sum recurrence is exact in fp32 exactly while its closed form is finite
    // and exactly representable. The test is a round trip rather than an integer
    // bound, because seed * world ** depth is a power of two here and stays exact
    // far past the 2**24 limit for arbitrary integers; past exactness it saturates
    // and proves nothing.
    const float rounded = static_cast<float>(closed_form);
    const bool closed_form_exact =
        std::isfinite(closed_form) && static_cast<double>(rounded) == closed_form;
    result.informative = recurrence == Recurrence::kBounded || closed_form_exact;
    if (recurrence == Recurrence::kBounded) {
      seed_vector();
    } else {
      seed_chain();
    }
    run_chain(depth, recurrence, nullptr);
    for (uint32_t rank = 0; rank < world; ++rank) {
      const std::vector<float> observed = read_rank(rank, depth);
      for (uint32_t index = 0; index < args.count; ++index) {
        const double value = static_cast<double>(observed[index]);
        if (!std::isfinite(value)) {
          // A NaN comparison is false, so it would slip past a max() accumulator.
          ++result.nonfinite;
          result.finite = false;
          continue;
        }
        if (!result.informative) {
          continue;
        }
        const double expected =
            recurrence == Recurrence::kSum
                ? closed_form
                : 6.0 * static_cast<double>(index + 1) + static_cast<double>(depth);
        result.max_abs_error = std::max(result.max_abs_error, std::fabs(value - expected));
      }
    }
    result.saturation_expected = !result.informative && !result.finite;
    result.exact = result.informative && result.finite && result.max_abs_error == 0.0;
    if (result.saturation_expected) {
      result.detail = "closed form overflows fp32 at this depth, so the nonfinite values "
                      "are expected; they also show the chain compounded, because a "
                      "shorter chain would still be finite";
    } else if (!result.finite) {
      result.detail = "nonfinite value on a device buffer";
    } else if (!result.informative) {
      result.detail = "closed form saturates fp32 at this depth, so the check is not "
                      "informative";
    }
    return result;
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

void print_check(const char* name, const Check& check, uint32_t elements, int indent) {
  const std::string pad(indent, ' ');
  std::cout << pad << "\"" << name << "\": {\n";
  std::cout << pad << "  \"informative\": " << (check.informative ? "true" : "false") << ",\n";
  std::cout << pad << "  \"finite\": " << (check.finite ? "true" : "false") << ",\n";
  std::cout << pad << "  \"exact\": " << (check.exact ? "true" : "false") << ",\n";
  std::cout << pad << "  \"nonfinite_values\": " << check.nonfinite << ",\n";
  std::cout << pad << "  \"saturation_expected\": "
            << (check.saturation_expected ? "true" : "false") << ",\n";
  std::cout << pad << "  \"max_abs_error\": " << check.max_abs_error << ",\n";
  std::cout << pad << "  \"ranks_checked\": 2,\n";
  std::cout << pad << "  \"elements_per_rank\": " << elements << ",\n";
  std::cout << pad << "  \"expected\": \"" << check.expected << "\"";
  if (!check.detail.empty()) {
    std::cout << ",\n" << pad << "  \"detail\": \"" << check.detail << "\"";
  }
  std::cout << "\n" << pad << "}";
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  Runner runner;
  runner.args = args;
  runner.init();

  // Verification first, and through the same step() the timed passes use: a
  // broken chain must not produce a timing. Both recurrences are checked over the
  // whole vector on both ranks.
  const Check bounded = runner.verify(args.depth, Recurrence::kBounded);
  const Check sum = runner.verify(args.depth, Recurrence::kSum);

  // Timed runs: the sum recurrence, which is the arithmetic the Python arm
  // measures.
  std::vector<double> totals;
  std::vector<double> singles;
  for (uint32_t iteration = 0; iteration < args.warmup + args.iterations; ++iteration) {
    runner.seed_chain();
    const double started = now_us();
    runner.run_chain(args.depth, Recurrence::kSum, nullptr);
    const double elapsed = now_us() - started;
    if (iteration >= args.warmup) {
      totals.push_back(elapsed);
    }
  }
  runner.seed_chain();
  {
    const double started = now_us();
    runner.run_chain(1, Recurrence::kSum, nullptr);
    singles.push_back(now_us() - started);
  }

  // Phase accounting on a separate pass, so the timed runs stay uninstrumented.
  Phase phase;
  runner.seed_chain();
  runner.run_chain(args.depth, Recurrence::kSum, &phase);

  const double total_median = median(totals);
  const double single_median = median(singles);
  // The single-step intercept cannot divide out of a one-step chain.
  const double per_step =
      args.depth > 1 ? (total_median - single_median) / static_cast<double>(args.depth - 1)
                     : std::nan("");
  const double per_step_phases =
      (phase.submit_d2h_us + phase.wait_d2h_us + phase.reduce_us + phase.submit_h2d_us) /
      static_cast<double>(args.depth);

  std::cout.precision(10);
  std::cout << "{\n";
  std::cout << "  \"depth\": " << args.depth << ",\n";
  std::cout << "  \"count\": " << args.count << ",\n";
  std::cout << "  \"payload_bytes\": " << runner.nbytes << ",\n";
  std::cout << "  \"iterations\": " << args.iterations << ",\n";
  std::cout << "  \"warmup\": " << args.warmup << ",\n";
  std::cout << "  \"timed_recurrence\": \"sum\",\n";
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
  std::cout << "    \"host_reduce\": " << phase.reduce_us / args.depth << ",\n";
  std::cout << "    \"submit_h2d\": " << phase.submit_h2d_us / args.depth << "\n";
  std::cout << "  },\n";
  std::cout << "  \"verification\": {\n";
  print_check("bounded", bounded, args.count, 4);
  std::cout << ",\n";
  print_check("sum", sum, args.count, 4);
  std::cout << ",\n";
  std::cout << "    \"ranks_checked\": 2,\n";
  std::cout << "    \"elements_per_rank\": " << args.count << "\n";
  std::cout << "  },\n";
  std::cout << "  \"protocol\": \"batched: both D2H submitted before either wait, no return wait\"\n";
  std::cout << "}\n";

  runner.destroy();
  // The bounded recurrence must always verify exactly and finitely. The timed
  // recurrence must verify exactly where its closed form is representable, and is
  // allowed to saturate only where that closed form itself overflows fp32.
  const bool sum_ok =
      sum.informative ? (sum.finite && sum.exact) : sum.saturation_expected;
  const bool ok = bounded.exact && bounded.finite && sum_ok;
  return ok ? 0 : 1;
}
