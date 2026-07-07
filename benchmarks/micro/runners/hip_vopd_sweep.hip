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

#ifndef HIPENGINE_VOPD_MODE
#define HIPENGINE_VOPD_MODE 0
#endif

#ifndef HIPENGINE_VOPD_ACCUMS
#define HIPENGINE_VOPD_ACCUMS 4
#endif

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kModeIndependentFma = 0;
constexpr uint32_t kModeDependentFma = 1;
constexpr uint32_t kModeMixedIntFloat = 2;
constexpr uint32_t kModeDequantLike = 3;

struct Args {
  std::string json_path;
  uint32_t n = 65536;
  uint32_t body_iters = 2048;
  uint32_t reps = 20;
  uint32_t warmup = 5;
  uint32_t samples = 7;
  int device_index = 0;
};

struct Row {
  std::string mode;
  uint32_t accums;
  uint32_t n;
  uint32_t body_iters;
  uint32_t block_size;
  double median_us;
  double p05_us;
  double p95_us;
  double min_us;
  double max_us;
  double gops;
  float max_abs;
  float max_rel;
  bool correctness_pass;
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
    } else if (flag == "--device-index") {
      args.device_index = std::stoi(require_value(i, argc, argv, flag));
    } else {
      fail("unknown argument: " + flag);
    }
  }
  if (args.n == 0 || args.body_iters == 0 || args.reps == 0 || args.samples == 0) {
    fail("--n, --body-iters, --reps, and --samples must be positive");
  }
  return args;
}

const char* mode_name() {
#if HIPENGINE_VOPD_MODE == 0
  return "independent_fma";
#elif HIPENGINE_VOPD_MODE == 1
  return "dependent_fma";
#elif HIPENGINE_VOPD_MODE == 2
  return "mixed_int_float";
#elif HIPENGINE_VOPD_MODE == 3
  return "dequant_like";
#else
  return "unknown";
#endif
}

__host__ __device__ float init_accum(uint32_t idx, uint32_t lane) {
  uint32_t bits = (idx * 747796405u) ^ (lane * 2891336453u) ^ 0x9e3779b9u;
  return 0.25f + static_cast<float>(bits & 0xffu) * 0.0009765625f;
}

__host__ __device__ float lane_bias(uint32_t idx, uint32_t iter, uint32_t lane) {
  uint32_t bits = idx * 1664525u + iter * 1013904223u + lane * 2246822519u;
  return static_cast<float>((bits >> 24) & 0x1fu) * 0.000001f + 0.000003f;
}

__host__ __device__ uint32_t hash_step(uint32_t value, uint32_t iter, uint32_t lane) {
  value ^= iter * 0x9e3779b9u + lane * 0x85ebca6bu;
  value = value * 1664525u + 1013904223u;
  value ^= value >> 16;
  return value;
}

__host__ __device__ float run_value(uint32_t idx, uint32_t body_iters) {
  float a0 = init_accum(idx, 0);
  float a1 = init_accum(idx, 1);
  float a2 = init_accum(idx, 2);
  float a3 = init_accum(idx, 3);
  float a4 = init_accum(idx, 4);
  float a5 = init_accum(idx, 5);
  float a6 = init_accum(idx, 6);
  float a7 = init_accum(idx, 7);
  uint32_t u0 = idx * 747796405u + 2891336453u;
  uint32_t u1 = idx ^ 0xa5a5a5a5u;

  for (uint32_t iter = 0; iter < body_iters; ++iter) {
#if HIPENGINE_VOPD_MODE == 0
    a0 = fmaf(a0, 1.000001f, lane_bias(idx, iter, 0));
#if HIPENGINE_VOPD_ACCUMS >= 2
    a1 = fmaf(a1, 0.999999f, lane_bias(idx, iter, 1));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 3
    a2 = fmaf(a2, 1.000002f, lane_bias(idx, iter, 2));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 4
    a3 = fmaf(a3, 0.999998f, lane_bias(idx, iter, 3));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 5
    a4 = fmaf(a4, 1.000003f, lane_bias(idx, iter, 4));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 6
    a5 = fmaf(a5, 0.999997f, lane_bias(idx, iter, 5));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 7
    a6 = fmaf(a6, 1.000004f, lane_bias(idx, iter, 6));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 8
    a7 = fmaf(a7, 0.999996f, lane_bias(idx, iter, 7));
#endif
#elif HIPENGINE_VOPD_MODE == 1
    a0 = fmaf(a0, 1.000001f, lane_bias(idx, iter, 0));
#if HIPENGINE_VOPD_ACCUMS >= 2
    a0 = fmaf(a0, 0.999999f, lane_bias(idx, iter, 1));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 3
    a0 = fmaf(a0, 1.000002f, lane_bias(idx, iter, 2));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 4
    a0 = fmaf(a0, 0.999998f, lane_bias(idx, iter, 3));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 5
    a0 = fmaf(a0, 1.000003f, lane_bias(idx, iter, 4));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 6
    a0 = fmaf(a0, 0.999997f, lane_bias(idx, iter, 5));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 7
    a0 = fmaf(a0, 1.000004f, lane_bias(idx, iter, 6));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 8
    a0 = fmaf(a0, 0.999996f, lane_bias(idx, iter, 7));
#endif
#elif HIPENGINE_VOPD_MODE == 2
    u0 = hash_step(u0, iter, 0);
    a0 = fmaf(a0, 1.000001f, static_cast<float>(u0 & 0x1fu) * 0.000001f);
#if HIPENGINE_VOPD_ACCUMS >= 2
    u1 = hash_step(u1, iter, 1);
    a1 = fmaf(a1, 0.999999f, static_cast<float>((u1 >> 3) & 0x1fu) * 0.000001f);
#endif
#if HIPENGINE_VOPD_ACCUMS >= 3
    u0 = hash_step(u0, iter, 2);
    a2 = fmaf(a2, 1.000002f, static_cast<float>((u0 >> 7) & 0x1fu) * 0.000001f);
#endif
#if HIPENGINE_VOPD_ACCUMS >= 4
    u1 = hash_step(u1, iter, 3);
    a3 = fmaf(a3, 0.999998f, static_cast<float>((u1 >> 11) & 0x1fu) * 0.000001f);
#endif
#elif HIPENGINE_VOPD_MODE == 3
    u0 = hash_step(u0, iter, 0);
    int q0 = static_cast<int>((u0 >> ((iter + 0u) & 15u)) & 0xffu) - 128;
    a0 = fmaf(static_cast<float>(q0) * 0.0078125f, 0.03125f, a0);
#if HIPENGINE_VOPD_ACCUMS >= 2
    u1 = hash_step(u1, iter, 1);
    int q1 = static_cast<int>((u1 >> ((iter + 3u) & 15u)) & 0xffu) - 128;
    a1 = fmaf(static_cast<float>(q1) * 0.0078125f, 0.03125f, a1);
#endif
#if HIPENGINE_VOPD_ACCUMS >= 3
    u0 = hash_step(u0, iter, 2);
    int q2 = static_cast<int>((u0 >> ((iter + 5u) & 15u)) & 0xffu) - 128;
    a2 = fmaf(static_cast<float>(q2) * 0.0078125f, 0.03125f, a2);
#endif
#if HIPENGINE_VOPD_ACCUMS >= 4
    u1 = hash_step(u1, iter, 3);
    int q3 = static_cast<int>((u1 >> ((iter + 7u) & 15u)) & 0xffu) - 128;
    a3 = fmaf(static_cast<float>(q3) * 0.0078125f, 0.03125f, a3);
#endif
#endif
  }

  float out = a0;
#if HIPENGINE_VOPD_ACCUMS >= 2
  out += a1;
#endif
#if HIPENGINE_VOPD_ACCUMS >= 3
  out += a2;
#endif
#if HIPENGINE_VOPD_ACCUMS >= 4
  out += a3;
#endif
#if HIPENGINE_VOPD_ACCUMS >= 5
  out += a4;
#endif
#if HIPENGINE_VOPD_ACCUMS >= 6
  out += a5;
#endif
#if HIPENGINE_VOPD_ACCUMS >= 7
  out += a6;
#endif
#if HIPENGINE_VOPD_ACCUMS >= 8
  out += a7;
#endif
  return out;
}

__global__ void vopd_sweep_kernel(float* __restrict__ out, uint32_t n, uint32_t body_iters) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = run_value(idx, body_iters);
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

double ops_per_element_iter() {
  return static_cast<double>(HIPENGINE_VOPD_ACCUMS);
}

Row run_config(const Args& args) {
  float* d_out = nullptr;
  std::vector<float> actual(args.n, 0.0f);
  size_t out_bytes = sizeof(float) * actual.size();
  check(hipMalloc(&d_out, out_bytes), "hipMalloc out");
  check(hipMemset(d_out, 0, out_bytes), "hipMemset out");

  dim3 block(kBlockSize);
  dim3 grid((args.n + kBlockSize - 1) / kBlockSize);
  vopd_sweep_kernel<<<grid, block>>>(d_out, args.n, args.body_iters);
  check(hipGetLastError(), "correctness launch");
  check(hipDeviceSynchronize(), "correctness sync");
  check(hipMemcpy(actual.data(), d_out, out_bytes, hipMemcpyDeviceToHost), "copy out");

  float max_abs = 0.0f;
  float max_rel = 0.0f;
  uint32_t checked = std::min<uint32_t>(args.n, 64);
  for (uint32_t i = 0; i < checked; ++i) {
    float expected = run_value(i, args.body_iters);
    float diff = std::abs(actual[i] - expected);
    max_abs = std::max(max_abs, diff);
    max_rel = std::max(max_rel, diff / std::max(1.0e-6f, std::abs(expected)));
  }
  bool pass = max_abs <= 2.5e-3f || max_rel <= 2.5e-4f;

  for (uint32_t i = 0; i < args.warmup; ++i) {
    vopd_sweep_kernel<<<grid, block>>>(d_out, args.n, args.body_iters);
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
      vopd_sweep_kernel<<<grid, block>>>(d_out, args.n, args.body_iters);
    }
    check(hipEventRecord(stop, 0), "hipEventRecord stop");
    check(hipEventSynchronize(stop), "hipEventSynchronize stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");
    samples.push_back(static_cast<double>(elapsed_ms) * 1000.0 / args.reps);
  }

  check(hipEventDestroy(start), "hipEventDestroy start");
  check(hipEventDestroy(stop), "hipEventDestroy stop");
  check(hipFree(d_out), "hipFree out");

  double median_us = percentile(samples, 0.5);
  double ops = static_cast<double>(args.n) * args.body_iters * ops_per_element_iter();
  return Row{
      mode_name(),
      HIPENGINE_VOPD_ACCUMS,
      args.n,
      args.body_iters,
      kBlockSize,
      median_us,
      percentile(samples, 0.05),
      percentile(samples, 0.95),
      *std::min_element(samples.begin(), samples.end()),
      *std::max_element(samples.begin(), samples.end()),
      ops / median_us / 1000.0,
      max_abs,
      max_rel,
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
  *out << "  \"run_tag\": \"hip-vopd-sweep\",\n";
  *out << "  \"status\": \"diagnostic\",\n";
  *out << "  \"backend\": \"hip\",\n";
  *out << "  \"hardware\": {\n";
  *out << "    \"device_name\": \"" << json_escape(prop.name) << "\",\n";
  *out << "    \"gcn_arch_name\": \"" << json_escape(prop.gcnArchName) << "\",\n";
  *out << "    \"multi_processor_count\": " << prop.multiProcessorCount << "\n";
  *out << "  },\n";
  *out << "  \"config\": {\n";
  *out << "    \"mode\": \"" << json_escape(row.mode) << "\",\n";
  *out << "    \"mode_id\": " << HIPENGINE_VOPD_MODE << ",\n";
  *out << "    \"accums\": " << row.accums << ",\n";
  *out << "    \"n\": " << row.n << ",\n";
  *out << "    \"body_iters\": " << row.body_iters << ",\n";
  *out << "    \"block_size\": " << row.block_size << ",\n";
  *out << "    \"reps\": " << args.reps << ",\n";
  *out << "    \"warmup\": " << args.warmup << ",\n";
  *out << "    \"samples\": " << args.samples << ",\n";
  *out << "    \"method\": \"pure VALU VOPD diagnostic; one output element per lane; sampled CPU oracle\"\n";
  *out << "  },\n";
  *out << "  \"rows\": [\n";
  *out << "    {\n";
  *out << "      \"mode\": \"" << json_escape(row.mode) << "\",\n";
  *out << "      \"accums\": " << row.accums << ",\n";
  *out << "      \"n\": " << row.n << ",\n";
  *out << "      \"body_iters\": " << row.body_iters << ",\n";
  *out << "      \"block_size\": " << row.block_size << ",\n";
  *out << "      \"median_us\": " << row.median_us << ",\n";
  *out << "      \"p05_us\": " << row.p05_us << ",\n";
  *out << "      \"p95_us\": " << row.p95_us << ",\n";
  *out << "      \"min_us\": " << row.min_us << ",\n";
  *out << "      \"max_us\": " << row.max_us << ",\n";
  *out << "      \"gops\": " << row.gops << ",\n";
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
    std::cerr << "[hip] mode=" << row.mode << " accums=" << row.accums
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
