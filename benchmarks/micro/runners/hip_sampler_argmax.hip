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

#ifndef HIPENGINE_ARGMAX_WG
#define HIPENGINE_ARGMAX_WG 256
#endif

namespace {

constexpr uint32_t kWorkgroupSize = HIPENGINE_ARGMAX_WG;

struct Args {
  std::string json_path;
  uint32_t rows = 4;
  uint32_t vocab = 32768;
  uint32_t reps = 50;
  uint32_t warmup = 10;
  uint32_t samples = 9;
  int device_index = 0;
};

struct Row {
  uint32_t rows;
  uint32_t vocab;
  uint32_t workgroup_size;
  double bytes_per_dispatch;
  double comparisons_per_dispatch;
  double median_us;
  double p05_us;
  double p95_us;
  double min_us;
  double max_us;
  double bandwidth_gbps;
  double gcomparisons_per_s;
  uint32_t mismatches;
  float max_abs;
  bool correctness_pass;
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
    } else if (flag == "--device-index") {
      args.device_index = std::stoi(require_value(i, argc, argv, flag));
    } else {
      fail("unknown argument: " + flag);
    }
  }
  if (args.rows == 0 || args.vocab == 0 || args.reps == 0 || args.samples == 0) {
    fail("--rows, --vocab, --reps, and --samples must be positive");
  }
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

__global__ __launch_bounds__(kWorkgroupSize) void sampler_argmax_kernel(
    const float* __restrict__ logits,
    uint32_t* __restrict__ out_indices,
    float* __restrict__ out_values,
    uint32_t rows,
    uint32_t vocab) {
  uint32_t row = blockIdx.x;
  uint32_t tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  Pair best{-std::numeric_limits<float>::infinity(), 0xffffffffu};
  size_t base = static_cast<size_t>(row) * vocab;
  for (uint32_t col = tid; col < vocab; col += kWorkgroupSize) {
    Pair candidate{logits[base + col], col};
    if (better_pair(candidate, best)) {
      best = candidate;
    }
  }

  __shared__ float shared_values[kWorkgroupSize];
  __shared__ uint32_t shared_indices[kWorkgroupSize];
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
    out_indices[row] = shared_indices[0];
    out_values[row] = shared_values[0];
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

Row run_config(const Args& args) {
  std::vector<float> logits;
  fill_logits(logits, args.rows, args.vocab);
  std::vector<uint32_t> actual_indices(args.rows, 0);
  std::vector<float> actual_values(args.rows, 0.0f);

  float* d_logits = nullptr;
  uint32_t* d_indices = nullptr;
  float* d_values = nullptr;
  size_t logits_bytes = sizeof(float) * logits.size();
  size_t index_bytes = sizeof(uint32_t) * actual_indices.size();
  size_t value_bytes = sizeof(float) * actual_values.size();

  check(hipMalloc(&d_logits, logits_bytes), "hipMalloc logits");
  check(hipMalloc(&d_indices, index_bytes), "hipMalloc indices");
  check(hipMalloc(&d_values, value_bytes), "hipMalloc values");
  check(hipMemcpy(d_logits, logits.data(), logits_bytes, hipMemcpyHostToDevice), "hipMemcpy logits");
  check(hipMemset(d_indices, 0, index_bytes), "hipMemset indices");
  check(hipMemset(d_values, 0, value_bytes), "hipMemset values");

  dim3 block(kWorkgroupSize);
  dim3 grid(args.rows);
  sampler_argmax_kernel<<<grid, block>>>(d_logits, d_indices, d_values, args.rows, args.vocab);
  check(hipGetLastError(), "correctness launch");
  check(hipDeviceSynchronize(), "correctness sync");
  check(hipMemcpy(actual_indices.data(), d_indices, index_bytes, hipMemcpyDeviceToHost), "copy indices");
  check(hipMemcpy(actual_values.data(), d_values, value_bytes, hipMemcpyDeviceToHost), "copy values");

  uint32_t mismatches = 0;
  float max_abs = 0.0f;
  for (uint32_t row = 0; row < args.rows; ++row) {
    uint32_t expected_index = peak_index(row, args.vocab);
    float expected_value = logit_value(row, expected_index, args.vocab);
    float diff = std::abs(actual_values[row] - expected_value);
    max_abs = std::max(max_abs, diff);
    if (actual_indices[row] != expected_index || diff > 0.0f) {
      ++mismatches;
    }
  }
  bool pass = mismatches == 0;

  for (uint32_t i = 0; i < args.warmup; ++i) {
    sampler_argmax_kernel<<<grid, block>>>(d_logits, d_indices, d_values, args.rows, args.vocab);
  }
  check(hipDeviceSynchronize(), "warmup sync");

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  check(hipEventCreate(&start), "hipEventCreate start");
  check(hipEventCreate(&stop), "hipEventCreate stop");
  std::vector<double> samples;
  samples.reserve(args.samples);
  for (uint32_t sample = 0; sample < args.samples; ++sample) {
    check(hipEventRecord(start, 0), "hipEventRecord start");
    for (uint32_t rep = 0; rep < args.reps; ++rep) {
      sampler_argmax_kernel<<<grid, block>>>(d_logits, d_indices, d_values, args.rows, args.vocab);
    }
    check(hipEventRecord(stop, 0), "hipEventRecord stop");
    check(hipEventSynchronize(stop), "hipEventSynchronize stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");
    samples.push_back(static_cast<double>(elapsed_ms) * 1000.0 / args.reps);
  }

  check(hipEventDestroy(start), "hipEventDestroy start");
  check(hipEventDestroy(stop), "hipEventDestroy stop");
  check(hipFree(d_logits), "hipFree logits");
  check(hipFree(d_indices), "hipFree indices");
  check(hipFree(d_values), "hipFree values");

  double median_us = percentile(samples, 0.5);
  double bytes = static_cast<double>(args.rows) * args.vocab * sizeof(float);
  double comparisons = static_cast<double>(args.rows) * args.vocab;
  return Row{
      args.rows,
      args.vocab,
      kWorkgroupSize,
      bytes,
      comparisons,
      median_us,
      percentile(samples, 0.05),
      percentile(samples, 0.95),
      *std::min_element(samples.begin(), samples.end()),
      *std::max_element(samples.begin(), samples.end()),
      bytes / median_us / 1000.0,
      comparisons / median_us / 1000.0,
      mismatches,
      max_abs,
      pass,
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
  *out << "    \"reps\": " << args.reps << ",\n";
  *out << "    \"warmup\": " << args.warmup << ",\n";
  *out << "    \"samples\": " << args.samples << ",\n";
  *out << "    \"method\": \"one workgroup per row top-1 argmax reduction; CPU oracle\"\n";
  *out << "  },\n";
  *out << "  \"rows\": [\n";
  *out << "    {\n";
  *out << "      \"rows\": " << row.rows << ",\n";
  *out << "      \"vocab\": " << row.vocab << ",\n";
  *out << "      \"workgroup_size\": " << row.workgroup_size << ",\n";
  *out << "      \"bytes_per_dispatch\": " << row.bytes_per_dispatch << ",\n";
  *out << "      \"comparisons_per_dispatch\": " << row.comparisons_per_dispatch << ",\n";
  *out << "      \"median_us\": " << row.median_us << ",\n";
  *out << "      \"p05_us\": " << row.p05_us << ",\n";
  *out << "      \"p95_us\": " << row.p95_us << ",\n";
  *out << "      \"min_us\": " << row.min_us << ",\n";
  *out << "      \"max_us\": " << row.max_us << ",\n";
  *out << "      \"bandwidth_gbps\": " << row.bandwidth_gbps << ",\n";
  *out << "      \"gcomparisons_per_s\": " << row.gcomparisons_per_s << ",\n";
  *out << "      \"mismatches\": " << row.mismatches << ",\n";
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
