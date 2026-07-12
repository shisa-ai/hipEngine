#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "micro_timing_hip.hpp"

#ifndef HIPENGINE_DOT_MODE
#define HIPENGINE_DOT_MODE 0
#endif

#ifndef HIPENGINE_DOT_GROUPS
#define HIPENGINE_DOT_GROUPS 16
#endif

#ifndef HIPENGINE_DOT_FIXED_BLOCK
#define HIPENGINE_DOT_FIXED_BLOCK 0
#endif

#ifndef HIPENGINE_BLOCK_SIZE
#define HIPENGINE_BLOCK_SIZE 256
#endif

namespace {

constexpr uint32_t kBlockSize = HIPENGINE_BLOCK_SIZE;
constexpr uint32_t kModeQ8Signed = 0;
constexpr uint32_t kModeQ4Unsigned = 1;
constexpr uint32_t kModeQ6Zero = 2;
constexpr uint32_t kModeScalarDequant = 3;

#if HIPENGINE_DOT_FIXED_BLOCK
#define HIPENGINE_DOT_LAUNCH_BOUNDS __launch_bounds__(kBlockSize)
#else
#define HIPENGINE_DOT_LAUNCH_BOUNDS
#endif

struct Args {
  std::string json_path;
  uint32_t n = 32768;
  uint32_t body_iters = 128;
  uint32_t reps = 20;
  uint32_t warmup = 5;
  uint32_t samples = 7;
  std::string timing_mode = "serial_latency";
  uint32_t independent_streams = 4;
  int device_index = 0;
};

struct Row {
  std::string mode;
  uint32_t groups;
  uint32_t n;
  uint32_t body_iters;
  uint32_t block_size;
  uint32_t data_elems;
  double bytes_per_dispatch;
  double integer_ops_per_dispatch;
  double median_us;
  double p05_us;
  double p95_us;
  double min_us;
  double max_us;
  double bandwidth_gbps;
  double gops;
  double max_abs;
  double max_rel;
  bool correctness_pass;
  bool timed_sequence_correctness_pass;
  uint32_t stream_count;
  hipengine::micro::TimingSamples single_timing;
  hipengine::micro::TimingSamples burst_timing;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void check(hipError_t result, const char* what) {
  if (result != hipSuccess) {
    std::ostringstream oss;
    oss << what << " failed: " << hipGetErrorString(result);
    fail(oss.str());
  }
}

std::string require_value(int& index, int argc, char** argv, const std::string& flag) {
  if (index + 1 >= argc) {
    fail(flag + " requires a value");
  }
  ++index;
  return argv[index];
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    if (flag == "--json") {
      args.json_path = require_value(i, argc, argv, flag);
    } else if (flag == "--n") {
      args.n = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--body-iters") {
      args.body_iters = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--reps") {
      args.reps = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--warmup") {
      args.warmup = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--samples") {
      args.samples = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--timing-mode") {
      args.timing_mode = require_value(i, argc, argv, flag);
    } else if (flag == "--independent-streams") {
      args.independent_streams =
          static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--device-index") {
      args.device_index = std::stoi(require_value(i, argc, argv, flag));
    } else {
      fail("unknown argument: " + flag);
    }
  }
  if (args.n == 0 || args.body_iters == 0 || args.reps == 0 || args.samples == 0 ||
      args.independent_streams == 0) {
    fail("--n, --body-iters, --reps, --samples, and --independent-streams must be positive");
  }
  (void)hipengine::micro::parse_timing_mode(args.timing_mode);
  return args;
}

const char* mode_name() {
#if HIPENGINE_DOT_MODE == 0
  return "q8_signed";
#elif HIPENGINE_DOT_MODE == 1
  return "q4_unsigned";
#elif HIPENGINE_DOT_MODE == 2
  return "q6_zero";
#elif HIPENGINE_DOT_MODE == 3
  return "scalar_dequant";
#else
  return "unknown";
#endif
}

uint32_t next_power_of_two(uint64_t value) {
  if (value <= 1) {
    return 1;
  }
  --value;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  ++value;
  if (value > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    fail("required data size exceeds uint32 range");
  }
  return static_cast<uint32_t>(value);
}

uint32_t required_data_elems(uint32_t n, uint32_t body_iters) {
  uint64_t elems = static_cast<uint64_t>(n) * body_iters * HIPENGINE_DOT_GROUPS;
  return next_power_of_two(std::max<uint64_t>(1024, elems));
}

double bytes_per_dispatch(uint32_t n, uint32_t body_iters) {
  return static_cast<double>(n) * body_iters * HIPENGINE_DOT_GROUPS * 2.0 * sizeof(uint32_t);
}

double ops_per_dispatch(uint32_t n, uint32_t body_iters) {
  double dots = static_cast<double>(n) * body_iters * HIPENGINE_DOT_GROUPS;
#if HIPENGINE_DOT_MODE == 2
  return dots * 17.0;
#else
  return dots * 8.0;
#endif
}

__host__ __device__ uint32_t hash_u32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

__host__ __device__ uint8_t activation_byte(uint32_t i, uint32_t lane) {
  uint32_t bits = hash_u32(i * 1664525u + lane * 1013904223u + 0x9e3779b9u);
  return static_cast<uint8_t>(bits & 0xffu);
}

__host__ __device__ uint8_t weight_byte(uint32_t i, uint32_t lane) {
  uint32_t bits = hash_u32(i * 747796405u + lane * 2891336453u + 0x85ebca6bu);
#if HIPENGINE_DOT_MODE == 0
  return static_cast<uint8_t>(bits & 0xffu);
#elif HIPENGINE_DOT_MODE == 1 || HIPENGINE_DOT_MODE == 3
  return static_cast<uint8_t>(bits & 0x0fu);
#elif HIPENGINE_DOT_MODE == 2
  return static_cast<uint8_t>(bits & 0x3fu);
#else
  return static_cast<uint8_t>(bits & 0xffu);
#endif
}

__host__ __device__ uint32_t pack_weight_word(uint32_t i) {
  uint32_t packed = 0;
#pragma unroll
  for (uint32_t lane = 0; lane < 4; ++lane) {
    packed |= static_cast<uint32_t>(weight_byte(i, lane)) << (lane * 8);
  }
  return packed;
}

__host__ __device__ uint32_t pack_activation_word(uint32_t i) {
  uint32_t packed = 0;
#pragma unroll
  for (uint32_t lane = 0; lane < 4; ++lane) {
    packed |= static_cast<uint32_t>(activation_byte(i, lane)) << (lane * 8);
  }
  return packed;
}

void fill_inputs(std::vector<uint32_t>& weights, std::vector<uint32_t>& activations, uint32_t n, uint32_t body_iters) {
  uint32_t data_elems = required_data_elems(n, body_iters);
  weights.resize(data_elems);
  activations.resize(data_elems);
  for (uint32_t i = 0; i < data_elems; ++i) {
    weights[i] = pack_weight_word(i);
    activations[i] = pack_activation_word(i);
  }
}

__host__ __device__ int dot_q8_s8_s8(uint32_t a, uint32_t b, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(true, static_cast<int>(a), true, static_cast<int>(b), c, false);
#else
  int acc = c;
#pragma unroll
  for (uint32_t lane = 0; lane < 4; ++lane) {
    int av = static_cast<int>(static_cast<int8_t>((a >> (lane * 8)) & 0xffu));
    int bv = static_cast<int>(static_cast<int8_t>((b >> (lane * 8)) & 0xffu));
    acc += av * bv;
  }
  return acc;
#endif
}

__host__ __device__ int dot_u8_s8(uint32_t a, uint32_t b, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(true, static_cast<int>(a), true, static_cast<int>(b), c, false);
#else
  int acc = c;
#pragma unroll
  for (uint32_t lane = 0; lane < 4; ++lane) {
    int av = static_cast<int>((a >> (lane * 8)) & 0xffu);
    int bv = static_cast<int>(static_cast<int8_t>((b >> (lane * 8)) & 0xffu));
    acc += av * bv;
  }
  return acc;
#endif
}

__host__ __device__ int scalar_q4_dequant(uint32_t a, uint32_t b, int c) {
  int acc = c;
#pragma unroll
  for (uint32_t lane = 0; lane < 4; ++lane) {
    int q = static_cast<int>((a >> (lane * 8)) & 0x0fu) - 8;
    int x = static_cast<int>(static_cast<int8_t>((b >> (lane * 8)) & 0xffu));
    acc += q * x;
  }
  return acc;
}

__host__ __device__ int run_value(
    const uint32_t* __restrict__ weights,
    const uint32_t* __restrict__ activations,
    uint32_t idx,
    uint32_t n,
    uint32_t body_iters,
    uint32_t data_mask) {
  int sum = 0;
  for (uint32_t iter = 0; iter < body_iters; ++iter) {
    uint32_t base = ((iter * n + idx) * HIPENGINE_DOT_GROUPS) & data_mask;
#pragma unroll
    for (uint32_t group = 0; group < HIPENGINE_DOT_GROUPS; ++group) {
      uint32_t offset = (base + group) & data_mask;
      uint32_t a = weights[offset];
      uint32_t b = activations[offset];
#if HIPENGINE_DOT_MODE == 0
      sum = dot_q8_s8_s8(a, b, sum);
#elif HIPENGINE_DOT_MODE == 1
      sum = dot_u8_s8(a, b, sum);
#elif HIPENGINE_DOT_MODE == 2
      int dot = dot_u8_s8(a, b, 0);
      int q8_sum = dot_u8_s8(0x01010101u, b, 0);
      sum += dot - 32 * q8_sum;
#elif HIPENGINE_DOT_MODE == 3
      sum = scalar_q4_dequant(a, b, sum);
#endif
    }
  }
  return sum;
}

HIPENGINE_DOT_LAUNCH_BOUNDS
__global__ void dot_path_kernel(
    const uint32_t* __restrict__ weights,
    const uint32_t* __restrict__ activations,
    int32_t* __restrict__ out,
    uint32_t n,
    uint32_t body_iters,
    uint32_t data_mask,
    uint32_t sequence_id) {
  uint32_t idx =
#if HIPENGINE_DOT_FIXED_BLOCK
      blockIdx.x * kBlockSize + threadIdx.x;
#else
      blockIdx.x * blockDim.x + threadIdx.x;
#endif
  if (idx >= n) {
    return;
  }
  out[idx] =
      run_value(weights, activations, idx, n, body_iters, data_mask) +
      static_cast<int32_t>(sequence_id);
}

double percentile(std::vector<double> values, double q) {
  if (values.empty()) {
    fail("cannot compute percentile of empty vector");
  }
  std::sort(values.begin(), values.end());
  double pos = q * static_cast<double>(values.size() - 1);
  size_t lo = static_cast<size_t>(std::floor(pos));
  size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) {
    return values[lo];
  }
  double t = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - t) + values[hi] * t;
}

Row run_config(const Args& args) {
  std::vector<uint32_t> weights;
  std::vector<uint32_t> activations;
  fill_inputs(weights, activations, args.n, args.body_iters);
  const auto timing_mode = hipengine::micro::parse_timing_mode(args.timing_mode);
  const uint32_t output_slots =
      timing_mode == hipengine::micro::TimingMode::IndependentThroughput
          ? std::max(args.reps, args.warmup)
          : 1u;
  std::vector<int32_t> actual(static_cast<size_t>(args.n) * output_slots, 0);
  uint32_t data_mask = static_cast<uint32_t>(weights.size() - 1);

  uint32_t* d_weights = nullptr;
  uint32_t* d_activations = nullptr;
  int32_t* d_out = nullptr;
  size_t weights_bytes = sizeof(uint32_t) * weights.size();
  size_t activations_bytes = sizeof(uint32_t) * activations.size();
  size_t out_bytes = sizeof(int32_t) * actual.size();

  check(hipMalloc(&d_weights, weights_bytes), "hipMalloc weights");
  check(hipMalloc(&d_activations, activations_bytes), "hipMalloc activations");
  check(hipMalloc(&d_out, out_bytes), "hipMalloc out");
  check(hipMemcpy(d_weights, weights.data(), weights_bytes, hipMemcpyHostToDevice), "hipMemcpy weights");
  check(hipMemcpy(d_activations, activations.data(), activations_bytes, hipMemcpyHostToDevice),
        "hipMemcpy activations");
  check(hipMemset(d_out, 0, out_bytes), "hipMemset out");
  check(hipDeviceSynchronize(), "initial output sync");

  dim3 block(kBlockSize);
  dim3 grid((args.n + kBlockSize - 1) / kBlockSize);
  hipengine::micro::HipSequenceTimer timer(timing_mode, args.independent_streams);
  auto launch = [&](uint32_t rep, hipStream_t stream) {
    const uint32_t slot =
        timing_mode == hipengine::micro::TimingMode::IndependentThroughput ? rep : 0u;
    dot_path_kernel<<<grid, block, 0, stream>>>(
        d_weights,
        d_activations,
        d_out + static_cast<size_t>(slot) * args.n,
        args.n,
        args.body_iters,
        data_mask,
        rep);
  };

  timer.run_and_wait(1, launch);
  check(hipMemcpy(actual.data(), d_out, sizeof(int32_t) * args.n, hipMemcpyDeviceToHost),
        "copy single out");
  double max_abs = 0.0;
  double max_rel = 0.0;
  const uint32_t checked = std::min<uint32_t>(args.n, 64);
  auto check_slot = [&](uint32_t slot, uint32_t sequence_id) {
    for (uint32_t i = 0; i < checked; ++i) {
      int expected =
          run_value(weights.data(), activations.data(), i, args.n, args.body_iters, data_mask) +
          static_cast<int32_t>(sequence_id);
      const int32_t observed = actual[static_cast<size_t>(slot) * args.n + i];
      double diff = std::abs(static_cast<double>(observed - expected));
      max_abs = std::max(max_abs, diff);
      max_rel =
          std::max(max_rel, diff / std::max(1.0, std::abs(static_cast<double>(expected))));
    }
  };
  check_slot(0, 0);
  const bool single_pass = max_abs == 0.0;

  if (args.warmup > 0) {
    timer.run_and_wait(args.warmup, launch);
  }
  hipengine::micro::TimingSamples single_timing = timer.measure(1, args.samples, launch);
  hipengine::micro::TimingSamples burst_timing =
      timer.measure(args.reps, args.samples, launch);

  check(hipMemset(d_out, 0, out_bytes), "hipMemset exact burst out");
  check(hipDeviceSynchronize(), "exact burst memset sync");
  timer.run_and_wait(args.reps, launch);
  check(hipMemcpy(actual.data(), d_out, out_bytes, hipMemcpyDeviceToHost),
        "copy exact burst out");
  if (timing_mode == hipengine::micro::TimingMode::IndependentThroughput) {
    for (uint32_t slot = 0; slot < args.reps; ++slot) {
      check_slot(slot, slot);
    }
  } else {
    check_slot(0, args.reps - 1);
  }
  const bool burst_pass = max_abs == 0.0;

  check(hipFree(d_weights), "hipFree weights");
  check(hipFree(d_activations), "hipFree activations");
  check(hipFree(d_out), "hipFree out");

  std::vector<double> samples;
  samples.reserve(burst_timing.gpu_sequence_us.size());
  for (double sample : burst_timing.gpu_sequence_us) {
    samples.push_back(sample / args.reps);
  }
  double median_us = percentile(samples, 0.5);
  double bytes = bytes_per_dispatch(args.n, args.body_iters);
  double ops = ops_per_dispatch(args.n, args.body_iters);
  return Row{
      mode_name(),
      HIPENGINE_DOT_GROUPS,
      args.n,
      args.body_iters,
      kBlockSize,
      static_cast<uint32_t>(weights.size()),
      bytes,
      ops,
      median_us,
      percentile(samples, 0.05),
      percentile(samples, 0.95),
      *std::min_element(samples.begin(), samples.end()),
      *std::max_element(samples.begin(), samples.end()),
      bytes / median_us / 1000.0,
      ops / median_us / 1000.0,
      max_abs,
      max_rel,
      single_pass && burst_pass,
      burst_pass,
      timer.stream_count(),
      std::move(single_timing),
      std::move(burst_timing),
  };
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (char ch : text) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

void write_samples(std::ostream& out, const std::vector<double>& samples) {
  out << "[";
  for (size_t i = 0; i < samples.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << samples[i];
  }
  out << "]";
}

void write_timing_raw(
    std::ostream& out,
    const char* name,
    uint32_t logical_iterations,
    const hipengine::micro::TimingSamples& timing,
    bool trailing_comma) {
  out << "      \"" << name << "\": {\n";
  out << "        \"logical_iterations\": " << logical_iterations << ",\n";
  out << "        \"dispatches_per_iteration\": 1,\n";
  out << "        \"gpu_samples_us\": ";
  write_samples(out, timing.gpu_sequence_us);
  out << ",\n";
  out << "        \"host_samples_us\": ";
  write_samples(out, timing.host_sequence_us);
  out << "\n      }" << (trailing_comma ? "," : "") << "\n";
}

void write_json(const Args& args, const hipDeviceProp_t& prop, const Row& row) {
  std::ostream* out = &std::cout;
  std::ofstream file;
  if (!args.json_path.empty()) {
    file.open(args.json_path);
    if (!file) {
      fail("could not open JSON path: " + args.json_path);
    }
    out = &file;
  }
  *out << std::setprecision(10);
  *out << "{\n";
  *out << "  \"run_tag\": \"hip-dot-path\",\n";
  *out << "  \"status\": \"diagnostic\",\n";
  *out << "  \"backend\": \"hip\",\n";
  *out << "  \"hardware\": {\n";
  *out << "    \"device_name\": \"" << json_escape(prop.name) << "\",\n";
  *out << "    \"gcn_arch_name\": \"" << json_escape(prop.gcnArchName) << "\",\n";
  *out << "    \"multi_processor_count\": " << prop.multiProcessorCount << "\n";
  *out << "  },\n";
  *out << "  \"config\": {\n";
  *out << "    \"mode\": \"" << json_escape(row.mode) << "\",\n";
  *out << "    \"mode_id\": " << HIPENGINE_DOT_MODE << ",\n";
  *out << "    \"groups\": " << row.groups << ",\n";
  *out << "    \"n\": " << row.n << ",\n";
  *out << "    \"body_iters\": " << row.body_iters << ",\n";
  *out << "    \"block_size\": " << row.block_size << ",\n";
  *out << "    \"fixed_block_indexing\": "
       << (HIPENGINE_DOT_FIXED_BLOCK ? "true" : "false") << ",\n";
  *out << "    \"data_elems\": " << row.data_elems << ",\n";
  *out << "    \"timing_mode\": \"" << json_escape(args.timing_mode) << "\",\n";
  *out << "    \"independent_streams\": " << args.independent_streams << ",\n";
  *out << "    \"reps\": " << args.reps << ",\n";
  *out << "    \"warmup\": " << args.warmup << ",\n";
  *out << "    \"samples\": " << args.samples << ",\n";
  *out << "    \"method\": \"packed int8 dot-path diagnostic; sampled exact CPU oracle\"\n";
  *out << "  },\n";
  *out << "  \"rows\": [\n";
  *out << "    {\n";
  *out << "      \"mode\": \"" << json_escape(row.mode) << "\",\n";
  *out << "      \"groups\": " << row.groups << ",\n";
  *out << "      \"n\": " << row.n << ",\n";
  *out << "      \"body_iters\": " << row.body_iters << ",\n";
  *out << "      \"block_size\": " << row.block_size << ",\n";
  *out << "      \"data_elems\": " << row.data_elems << ",\n";
  *out << "      \"bytes_per_dispatch\": " << row.bytes_per_dispatch << ",\n";
  *out << "      \"integer_ops_per_dispatch\": " << row.integer_ops_per_dispatch << ",\n";
  *out << "      \"median_us\": " << row.median_us << ",\n";
  *out << "      \"p05_us\": " << row.p05_us << ",\n";
  *out << "      \"p95_us\": " << row.p95_us << ",\n";
  *out << "      \"min_us\": " << row.min_us << ",\n";
  *out << "      \"max_us\": " << row.max_us << ",\n";
  *out << "      \"bandwidth_gbps\": " << row.bandwidth_gbps << ",\n";
  *out << "      \"gops\": " << row.gops << ",\n";
  *out << "      \"timing_mode\": \"" << json_escape(args.timing_mode) << "\",\n";
  *out << "      \"queue_or_stream_count\": " << row.stream_count << ",\n";
  *out << "      \"gpu_timestamps_supported\": true,\n";
  *out << "      \"timed_sequence_correctness_pass\": "
       << (row.timed_sequence_correctness_pass ? "true" : "false") << ",\n";
  *out << "      \"synchronization_pass\": "
       << (row.timed_sequence_correctness_pass ? "true" : "false") << ",\n";
  *out << "      \"barrier_count\": 0,\n";
  *out << "      \"timing_raw\": {\n";
  write_timing_raw(*out, "single", 1, row.single_timing, true);
  write_timing_raw(*out, "burst", args.reps, row.burst_timing, false);
  *out << "      },\n";
  *out << "      \"max_abs\": " << row.max_abs << ",\n";
  *out << "      \"max_rel\": " << row.max_rel << ",\n";
  *out << "      \"correctness_pass\": " << (row.correctness_pass ? "true" : "false") << "\n";
  *out << "    }\n";
  *out << "  ]\n";
  *out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    check(hipSetDevice(args.device_index), "hipSetDevice");
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, args.device_index), "hipGetDeviceProperties");
    Row row = run_config(args);
    std::cerr << "[hip] mode=" << row.mode << " groups=" << row.groups
              << " median=" << row.median_us
              << " us correctness=" << (row.correctness_pass ? "pass" : "fail")
              << "\n";
    write_json(args, prop, row);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
