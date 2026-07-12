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
#include <vector>

#include "micro_timing_hip.hpp"

#ifndef HIPENGINE_ARGMAX_WG
#define HIPENGINE_ARGMAX_WG 256
#endif

#ifndef HIPENGINE_ARGMAX_TOPK
#define HIPENGINE_ARGMAX_TOPK 1
#endif

namespace {

constexpr uint32_t kWorkgroupSize = HIPENGINE_ARGMAX_WG;
constexpr uint32_t kTopK = HIPENGINE_ARGMAX_TOPK;
static_assert(kTopK > 0, "HIPENGINE_ARGMAX_TOPK must be positive");
static_assert(kTopK <= kWorkgroupSize, "HIPENGINE_ARGMAX_TOPK must be <= HIPENGINE_ARGMAX_WG");

struct Args {
  std::string json_path;
  uint32_t rows = 4;
  uint32_t vocab = 32768;
  uint32_t reps = 50;
  uint32_t warmup = 10;
  uint32_t samples = 9;
  std::string timing_mode = "serial_latency";
  uint32_t independent_streams = 4;
  int device_index = 0;
};

struct Row {
  uint32_t rows;
  uint32_t vocab;
  uint32_t workgroup_size;
  uint32_t top_k;
  double bytes_per_dispatch;
  double comparisons_per_dispatch;
  double median_us;
  double p05_us;
  double p95_us;
  double min_us;
  double max_us;
  double bandwidth_gbps;
  double gcomparisons_per_s;
  std::vector<double> single_gpu_samples_us;
  std::vector<double> single_host_samples_us;
  std::vector<double> burst_gpu_samples_us;
  std::vector<double> burst_host_samples_us;
  uint32_t single_mismatches;
  uint32_t burst_mismatches;
  float max_abs;
  bool correctness_pass;
  uint32_t stream_count;
};

struct Pair {
  float value;
  uint32_t index;
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
    } else if (flag == "--rows") {
      args.rows = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--vocab") {
      args.vocab = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
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
  if (args.rows == 0 || args.vocab == 0 || args.reps == 0 || args.samples == 0 ||
      args.independent_streams == 0) {
    fail("--rows, --vocab, --reps, --samples, and --independent-streams must be positive");
  }
  (void)hipengine::micro::parse_timing_mode(args.timing_mode);
  return args;
}

__host__ __device__ uint32_t hash_u32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

__host__ __device__ uint32_t peak_index(uint32_t row, uint32_t vocab) {
  return hash_u32(row * 747796405u + 2891336453u) % vocab;
}

__host__ __device__ float logit_value(uint32_t row, uint32_t col, uint32_t vocab) {
  if (col == peak_index(row, vocab)) {
    return 64.0f + static_cast<float>(row) * 0.25f;
  }
  uint32_t bits = hash_u32(row * 1664525u + col * 1013904223u);
  int value = static_cast<int>(bits & 0xffffu) - 32768;
  return static_cast<float>(value) * 0.000030517578125f;
}

void fill_logits(std::vector<float>& logits, uint32_t rows, uint32_t vocab) {
  logits.resize(static_cast<size_t>(rows) * vocab);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < vocab; ++col) {
      logits[static_cast<size_t>(row) * vocab + col] = logit_value(row, col, vocab);
    }
  }
}

__host__ __device__ bool better_pair(Pair a, Pair b) {
  return (a.value > b.value) || (a.value == b.value && a.index < b.index);
}

__host__ __device__ void insert_topk(Pair* top, Pair candidate) {
  for (uint32_t i = 0; i < kTopK; ++i) {
    if (better_pair(candidate, top[i])) {
      for (uint32_t j = kTopK - 1; j > i; --j) {
        top[j] = top[j - 1];
      }
      top[i] = candidate;
      return;
    }
  }
}

void cpu_topk(const std::vector<float>& logits, uint32_t row, uint32_t vocab, std::vector<Pair>& out) {
  out.assign(kTopK, Pair{-std::numeric_limits<float>::infinity(), 0xffffffffu});
  size_t base = static_cast<size_t>(row) * vocab;
  for (uint32_t col = 0; col < vocab; ++col) {
    insert_topk(out.data(), Pair{logits[base + col], col});
  }
}

__global__ __launch_bounds__(kWorkgroupSize) void sampler_argmax_kernel(
    const float* __restrict__ logits,
    uint32_t* __restrict__ out_indices,
    float* __restrict__ out_values,
    uint32_t rows,
    uint32_t vocab,
    size_t output_base,
    uint32_t sequence_id) {
  uint32_t row = blockIdx.x;
  uint32_t tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ uint32_t selected_indices[kTopK];
  __shared__ float selected_values[kTopK];
  if (tid < kTopK) {
    selected_indices[tid] = 0xffffffffu;
    selected_values[tid] = -std::numeric_limits<float>::infinity();
  }
  __syncthreads();

  size_t base = static_cast<size_t>(row) * vocab;

  __shared__ float shared_values[kWorkgroupSize];
  __shared__ uint32_t shared_indices[kWorkgroupSize];
  for (uint32_t pass = 0; pass < kTopK; ++pass) {
    Pair best{-std::numeric_limits<float>::infinity(), 0xffffffffu};
    for (uint32_t col = tid; col < vocab; col += kWorkgroupSize) {
      bool already_selected = false;
#pragma unroll
      for (uint32_t i = 0; i < kTopK; ++i) {
        if (i < pass && selected_indices[i] == col) {
          already_selected = true;
        }
      }
      if (already_selected) {
        continue;
      }
      Pair candidate{logits[base + col], col};
      if (better_pair(candidate, best)) {
        best = candidate;
      }
    }

    shared_values[tid] = best.value;
    shared_indices[tid] = best.index;
    __syncthreads();

    for (uint32_t offset = kWorkgroupSize / 2; offset > 0; offset >>= 1) {
      if (tid < offset) {
        Pair current{shared_values[tid], shared_indices[tid]};
        Pair other{shared_values[tid + offset], shared_indices[tid + offset]};
        if (better_pair(other, current)) {
          shared_values[tid] = other.value;
          shared_indices[tid] = other.index;
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      selected_indices[pass] = shared_indices[0];
      selected_values[pass] = shared_values[0];
    }
    __syncthreads();
  }

  if (tid < kTopK) {
    size_t output_index = output_base + static_cast<size_t>(row) * kTopK + tid;
    out_indices[output_index] = selected_indices[tid];
    out_values[output_index] =
        selected_values[tid] + static_cast<float>(sequence_id) * 0.125f;
  }
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

struct CorrectnessResult {
  uint32_t mismatches = 0;
  float max_abs = 0.0f;
};

CorrectnessResult validate_outputs(
    const std::vector<float>& logits,
    const std::vector<uint32_t>& actual_indices,
    const std::vector<float>& actual_values,
    uint32_t rows,
    uint32_t vocab,
    uint32_t copies,
    bool independent,
    uint32_t final_sequence_id) {
  CorrectnessResult result;
  std::vector<Pair> expected;
  const size_t output_elems = static_cast<size_t>(rows) * kTopK;
  for (uint32_t copy = 0; copy < copies; ++copy) {
    for (uint32_t row = 0; row < rows; ++row) {
      cpu_topk(logits, row, vocab, expected);
      for (uint32_t i = 0; i < kTopK; ++i) {
        size_t offset = static_cast<size_t>(copy) * output_elems +
                        static_cast<size_t>(row) * kTopK + i;
        uint32_t sequence_id = independent ? copy : final_sequence_id;
        float tagged_expected =
            expected[i].value + static_cast<float>(sequence_id) * 0.125f;
        float diff = std::abs(actual_values[offset] - tagged_expected);
        result.max_abs = std::max(result.max_abs, diff);
        if (actual_indices[offset] != expected[i].index || diff > 0.0f) {
          ++result.mismatches;
        }
      }
    }
  }
  return result;
}

Row run_config(const Args& args) {
  if (kTopK > args.vocab) {
    fail("top-k must be <= vocab");
  }
  const auto timing_mode = hipengine::micro::parse_timing_mode(args.timing_mode);
  const bool independent =
      timing_mode == hipengine::micro::TimingMode::IndependentThroughput;
  std::vector<float> logits;
  fill_logits(logits, args.rows, args.vocab);
  const size_t output_elems = static_cast<size_t>(args.rows) * kTopK;
  const uint32_t output_copies = independent ? args.reps : 1u;
  const uint32_t allocation_copies =
      independent ? std::max(args.reps, args.warmup) : 1u;
  const size_t allocation_elems = output_elems * allocation_copies;
  std::vector<uint32_t> actual_indices(allocation_elems, 0);
  std::vector<float> actual_values(allocation_elems, 0.0f);

  float* d_logits = nullptr;
  uint32_t* d_indices = nullptr;
  float* d_values = nullptr;
  size_t logits_bytes = sizeof(float) * logits.size();
  size_t index_bytes = sizeof(uint32_t) * allocation_elems;
  size_t value_bytes = sizeof(float) * allocation_elems;

  check(hipMalloc(&d_logits, logits_bytes), "hipMalloc logits");
  check(hipMalloc(&d_indices, index_bytes), "hipMalloc indices");
  check(hipMalloc(&d_values, value_bytes), "hipMalloc values");
  check(hipMemcpy(d_logits, logits.data(), logits_bytes, hipMemcpyHostToDevice), "hipMemcpy logits");
  check(hipMemset(d_indices, 0, index_bytes), "hipMemset indices");
  check(hipMemset(d_values, 0, value_bytes), "hipMemset values");

  dim3 block(kWorkgroupSize);
  dim3 grid(args.rows);
  hipengine::micro::HipSequenceTimer timer(timing_mode, args.independent_streams);
  auto launch = [&](uint32_t rep, hipStream_t stream) {
    size_t output_base = independent ? static_cast<size_t>(rep) * output_elems : 0u;
    sampler_argmax_kernel<<<grid, block, 0, stream>>>(
        d_logits, d_indices, d_values, args.rows, args.vocab, output_base, rep);
    check(hipGetLastError(), "sampler launch");
  };

  check(hipMemset(d_indices, 0, index_bytes), "reset indices for single correctness");
  check(hipMemset(d_values, 0, value_bytes), "reset values for single correctness");
  timer.run_and_wait(1, launch);
  check(
      hipMemcpy(actual_indices.data(), d_indices, sizeof(uint32_t) * output_elems, hipMemcpyDeviceToHost),
      "copy single indices");
  check(
      hipMemcpy(actual_values.data(), d_values, sizeof(float) * output_elems, hipMemcpyDeviceToHost),
      "copy single values");
  CorrectnessResult single_correctness =
      validate_outputs(logits, actual_indices, actual_values, args.rows, args.vocab, 1, independent, 0);

  check(hipMemset(d_indices, 0, index_bytes), "reset indices for burst correctness");
  check(hipMemset(d_values, 0, value_bytes), "reset values for burst correctness");
  timer.run_and_wait(args.reps, launch);
  check(hipMemcpy(actual_indices.data(), d_indices, index_bytes, hipMemcpyDeviceToHost), "copy burst indices");
  check(hipMemcpy(actual_values.data(), d_values, value_bytes, hipMemcpyDeviceToHost), "copy burst values");
  CorrectnessResult burst_correctness = validate_outputs(
      logits,
      actual_indices,
      actual_values,
      args.rows,
      args.vocab,
      output_copies,
      independent,
      args.reps - 1);

  if (args.warmup > 0) {
    timer.run_and_wait(args.warmup, launch);
  }
  auto single_timing = timer.measure(1, args.samples, launch);
  auto burst_timing = timer.measure(args.reps, args.samples, launch);

  check(hipFree(d_logits), "hipFree logits");
  check(hipFree(d_indices), "hipFree indices");
  check(hipFree(d_values), "hipFree values");

  std::vector<double> burst_per_iteration_us;
  burst_per_iteration_us.reserve(burst_timing.gpu_sequence_us.size());
  for (double value : burst_timing.gpu_sequence_us) {
    burst_per_iteration_us.push_back(value / args.reps);
  }
  double median_us = percentile(burst_per_iteration_us, 0.5);
  double scanned_values =
      static_cast<double>(args.rows) *
      (static_cast<double>(kTopK) * args.vocab - static_cast<double>(kTopK * (kTopK - 1)) / 2.0);
  double bytes = scanned_values * sizeof(float);
  double comparisons = scanned_values;
  bool pass = single_correctness.mismatches == 0 && burst_correctness.mismatches == 0;
  return Row{
      args.rows,
      args.vocab,
      kWorkgroupSize,
      kTopK,
      bytes,
      comparisons,
      median_us,
      percentile(burst_per_iteration_us, 0.05),
      percentile(burst_per_iteration_us, 0.95),
      *std::min_element(burst_per_iteration_us.begin(), burst_per_iteration_us.end()),
      *std::max_element(burst_per_iteration_us.begin(), burst_per_iteration_us.end()),
      bytes / median_us / 1000.0,
      comparisons / median_us / 1000.0,
      std::move(single_timing.gpu_sequence_us),
      std::move(single_timing.host_sequence_us),
      std::move(burst_timing.gpu_sequence_us),
      std::move(burst_timing.host_sequence_us),
      single_correctness.mismatches,
      burst_correctness.mismatches,
      std::max(single_correctness.max_abs, burst_correctness.max_abs),
      pass,
      timer.stream_count(),
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
    out << (i == 0 ? "" : ", ") << samples[i];
  }
  out << "]";
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
  *out << "  \"run_tag\": \"hip-sampler-argmax\",\n";
  *out << "  \"status\": \"diagnostic\",\n";
  *out << "  \"backend\": \"hip\",\n";
  *out << "  \"hardware\": {\n";
  *out << "    \"device_name\": \"" << json_escape(prop.name) << "\",\n";
  *out << "    \"gcn_arch_name\": \"" << json_escape(prop.gcnArchName) << "\",\n";
  *out << "    \"multi_processor_count\": " << prop.multiProcessorCount << "\n";
  *out << "  },\n";
  *out << "  \"config\": {\n";
  *out << "    \"rows\": " << row.rows << ",\n";
  *out << "    \"vocab\": " << row.vocab << ",\n";
  *out << "    \"workgroup_size\": " << row.workgroup_size << ",\n";
  *out << "    \"top_k\": " << row.top_k << ",\n";
  *out << "    \"reps\": " << args.reps << ",\n";
  *out << "    \"warmup\": " << args.warmup << ",\n";
  *out << "    \"samples\": " << args.samples << ",\n";
  *out << "    \"timing_mode\": \"" << json_escape(args.timing_mode) << "\",\n";
  *out << "    \"independent_streams\": " << args.independent_streams << ",\n";
  *out << "    \"actual_streams\": " << row.stream_count << ",\n";
  *out << "    \"method\": \"HIP events and host wall for one-dispatch and exact-burst controls; CPU oracle validates the timed burst\"\n";
  *out << "  },\n";
  *out << "  \"rows\": [\n";
  *out << "    {\n";
  *out << "      \"rows\": " << row.rows << ",\n";
  *out << "      \"vocab\": " << row.vocab << ",\n";
  *out << "      \"workgroup_size\": " << row.workgroup_size << ",\n";
  *out << "      \"top_k\": " << row.top_k << ",\n";
  *out << "      \"bytes_per_dispatch\": " << row.bytes_per_dispatch << ",\n";
  *out << "      \"comparisons_per_dispatch\": " << row.comparisons_per_dispatch << ",\n";
  *out << "      \"median_us\": " << row.median_us << ",\n";
  *out << "      \"p05_us\": " << row.p05_us << ",\n";
  *out << "      \"p95_us\": " << row.p95_us << ",\n";
  *out << "      \"min_us\": " << row.min_us << ",\n";
  *out << "      \"max_us\": " << row.max_us << ",\n";
  *out << "      \"bandwidth_gbps\": " << row.bandwidth_gbps << ",\n";
  *out << "      \"gcomparisons_per_s\": " << row.gcomparisons_per_s << ",\n";
  *out << "      \"timing_mode\": \"" << json_escape(args.timing_mode) << "\",\n";
  *out << "      \"stream_count\": " << row.stream_count << ",\n";
  *out << "      \"single_gpu_samples_us\": ";
  write_samples(*out, row.single_gpu_samples_us);
  *out << ",\n      \"single_host_samples_us\": ";
  write_samples(*out, row.single_host_samples_us);
  *out << ",\n      \"burst_gpu_samples_us\": ";
  write_samples(*out, row.burst_gpu_samples_us);
  *out << ",\n      \"burst_host_samples_us\": ";
  write_samples(*out, row.burst_host_samples_us);
  *out << ",\n";
  *out << "      \"single_mismatches\": " << row.single_mismatches << ",\n";
  *out << "      \"burst_mismatches\": " << row.burst_mismatches << ",\n";
  *out << "      \"mismatches\": " << (row.single_mismatches + row.burst_mismatches) << ",\n";
  *out << "      \"max_abs\": " << row.max_abs << ",\n";
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
    std::cerr << "[hip] rows=" << row.rows << " vocab=" << row.vocab
              << " wg=" << row.workgroup_size << " median=" << row.median_us
              << " us correctness=" << (row.correctness_pass ? "pass" : "fail")
              << "\n";
    write_json(args, prop, row);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
