#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef HIPENGINE_FIXED_WORKGROUP_SIZE
#define HIPENGINE_FIXED_WORKGROUP_SIZE 0
#endif

namespace {

#if HIPENGINE_FIXED_WORKGROUP_SIZE
#define HIPENGINE_GEOMETRY_LAUNCH_BOUNDS __launch_bounds__(HIPENGINE_FIXED_WORKGROUP_SIZE)
#else
#define HIPENGINE_GEOMETRY_LAUNCH_BOUNDS
#endif

struct Args {
  std::string json_path;
  std::vector<uint32_t> k_list{512, 2048, 8192};
  std::vector<uint32_t> rows_list{1, 4, 8};
  std::vector<uint32_t> workgroups{32, 64, 128, 256};
  uint32_t body_repeats = 128;
  uint32_t reps = 20;
  uint32_t warmup = 5;
  uint32_t samples = 11;
  int device_index = 0;
};

struct Row {
  uint32_t k;
  uint32_t rows;
  uint32_t workgroup_size;
  uint32_t body_repeats;
  double median_us;
  double p05_us;
  double p95_us;
  double min_us;
  double max_us;
  double gflops;
  double bytes_per_us;
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

std::vector<uint32_t> parse_u32_list(const std::string& text) {
  std::vector<uint32_t> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty()) {
      continue;
    }
    unsigned long parsed = std::stoul(item);
    if (parsed == 0 || parsed > UINT32_MAX) {
      fail("list values must be positive uint32");
    }
    values.push_back(static_cast<uint32_t>(parsed));
  }
  if (values.empty()) {
    fail("list must contain at least one value");
  }
  return values;
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
    } else if (flag == "--k-list") {
      args.k_list = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--rows-list") {
      args.rows_list = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--workgroups") {
      args.workgroups = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--body-repeats") {
      args.body_repeats = static_cast<uint32_t>(
          std::stoul(require_value(i, argc, argv, flag)));
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
  if (args.body_repeats == 0 || args.reps == 0 || args.samples == 0) {
    fail("--body-repeats, --reps, and --samples must be positive");
  }
  for (uint32_t wg : args.workgroups) {
    if (wg == 0 || wg > 256 || (wg & (wg - 1)) != 0) {
      fail("workgroup sizes must be powers of two in [1, 256]");
    }
#if HIPENGINE_FIXED_WORKGROUP_SIZE
    if (wg != HIPENGINE_FIXED_WORKGROUP_SIZE) {
      fail("fixed-workgroup build can only run its compiled workgroup size");
    }
#endif
  }
  return args;
}

float x_value(uint32_t i) {
  int v = static_cast<int>((i * 17u + 5u) % 29u) - 14;
  return static_cast<float>(v) * 0.03125f;
}

float w_value(uint32_t row, uint32_t i) {
  int v = static_cast<int>((row * 11u + i * 13u + 3u) % 31u) - 15;
  return static_cast<float>(v) * 0.015625f;
}

void fill_inputs(std::vector<float>& x, std::vector<float>& w, uint32_t k, uint32_t rows) {
  x.resize(k);
  w.resize(static_cast<size_t>(k) * rows);
  for (uint32_t i = 0; i < k; ++i) {
    x[i] = x_value(i);
  }
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t i = 0; i < k; ++i) {
      w[static_cast<size_t>(row) * k + i] = w_value(row, i);
    }
  }
}

std::vector<float> cpu_reference(
    const std::vector<float>& x,
    const std::vector<float>& w,
    uint32_t k,
    uint32_t rows,
    uint32_t workgroup_size,
    uint32_t body_repeats) {
  std::vector<float> out(rows, 0.0f);
  std::vector<float> scratch(workgroup_size, 0.0f);
  for (uint32_t row = 0; row < rows; ++row) {
    std::fill(scratch.begin(), scratch.end(), 0.0f);
    for (uint32_t lid = 0; lid < workgroup_size; ++lid) {
      float sum = 0.0f;
      for (uint32_t repeat = 0; repeat < body_repeats; ++repeat) {
        for (uint32_t i = lid; i < k; i += workgroup_size) {
          uint32_t j = (i + repeat) % k;
          sum = std::fma(x[j], w[static_cast<size_t>(row) * k + j], sum);
        }
      }
      scratch[lid] = sum;
    }
    for (uint32_t offset = workgroup_size >> 1; offset > 0; offset >>= 1) {
      for (uint32_t lid = 0; lid < offset; ++lid) {
        scratch[lid] += scratch[lid + offset];
      }
    }
    out[row] = scratch[0];
  }
  return out;
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

template <typename T>
void print_json_array(std::ostream& out, const std::vector<T>& values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << values[i];
  }
  out << "]";
}

HIPENGINE_GEOMETRY_LAUNCH_BOUNDS
__global__ void f32_gemv_geometry_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    uint32_t k,
    uint32_t body_repeats) {
  extern __shared__ float scratch[];
  uint32_t row = blockIdx.x;
  uint32_t lid = threadIdx.x;
  uint32_t wg =
#if HIPENGINE_FIXED_WORKGROUP_SIZE
      HIPENGINE_FIXED_WORKGROUP_SIZE;
#else
      blockDim.x;
#endif
  float sum = 0.0f;
  for (uint32_t repeat = 0; repeat < body_repeats; ++repeat) {
    for (uint32_t i = lid; i < k; i += wg) {
      uint32_t j = (i + repeat) % k;
      sum = fmaf(x[j], w[static_cast<size_t>(row) * k + j], sum);
    }
  }
  scratch[lid] = sum;
  __syncthreads();
  for (uint32_t offset = wg >> 1; offset > 0; offset >>= 1) {
    if (lid < offset) {
      scratch[lid] += scratch[lid + offset];
    }
    __syncthreads();
  }
  if (lid == 0) {
    out[row] = scratch[0];
  }
}

Row run_config(
    const Args& args,
    uint32_t k,
    uint32_t rows,
    uint32_t workgroup_size) {
  std::vector<float> x;
  std::vector<float> w;
  fill_inputs(x, w, k, rows);
  std::vector<float> expected =
      cpu_reference(x, w, k, rows, workgroup_size, args.body_repeats);
  std::vector<float> actual(rows, 0.0f);

  float* d_x = nullptr;
  float* d_w = nullptr;
  float* d_out = nullptr;
  size_t x_bytes = sizeof(float) * x.size();
  size_t w_bytes = sizeof(float) * w.size();
  size_t out_bytes = sizeof(float) * actual.size();

  check(hipMalloc(&d_x, x_bytes), "hipMalloc x");
  check(hipMalloc(&d_w, w_bytes), "hipMalloc w");
  check(hipMalloc(&d_out, out_bytes), "hipMalloc out");
  check(hipMemcpy(d_x, x.data(), x_bytes, hipMemcpyHostToDevice), "hipMemcpy x");
  check(hipMemcpy(d_w, w.data(), w_bytes, hipMemcpyHostToDevice), "hipMemcpy w");
  check(hipMemset(d_out, 0, out_bytes), "hipMemset out");

  dim3 grid(rows);
  dim3 block(workgroup_size);
  size_t shared_bytes = sizeof(float) * workgroup_size;
  f32_gemv_geometry_kernel<<<grid, block, shared_bytes>>>(
      d_x, d_w, d_out, k, args.body_repeats);
  check(hipGetLastError(), "correctness launch");
  check(hipDeviceSynchronize(), "correctness sync");
  check(hipMemcpy(actual.data(), d_out, out_bytes, hipMemcpyDeviceToHost), "copy out");

  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (uint32_t i = 0; i < rows; ++i) {
    float diff = std::abs(actual[i] - expected[i]);
    max_abs = std::max(max_abs, diff);
    max_rel = std::max(max_rel, diff / std::max(1.0e-6f, std::abs(expected[i])));
  }
  bool pass = max_abs <= 5.0e-3f || max_rel <= 5.0e-4f;

  for (uint32_t i = 0; i < args.warmup; ++i) {
    f32_gemv_geometry_kernel<<<grid, block, shared_bytes>>>(
        d_x, d_w, d_out, k, args.body_repeats);
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
      f32_gemv_geometry_kernel<<<grid, block, shared_bytes>>>(
          d_x, d_w, d_out, k, args.body_repeats);
    }
    check(hipEventRecord(stop, 0), "hipEventRecord stop");
    check(hipEventSynchronize(stop), "hipEventSynchronize stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");
    samples.push_back(static_cast<double>(elapsed_ms) * 1000.0 / args.reps);
  }

  check(hipEventDestroy(start), "hipEventDestroy start");
  check(hipEventDestroy(stop), "hipEventDestroy stop");
  check(hipFree(d_x), "hipFree x");
  check(hipFree(d_w), "hipFree w");
  check(hipFree(d_out), "hipFree out");

  double median_us = percentile(samples, 0.5);
  double ops = static_cast<double>(rows) * k * args.body_repeats * 2.0;
  double bytes = static_cast<double>(sizeof(float)) * (k + (static_cast<double>(rows) * k));
  return Row{
      k,
      rows,
      workgroup_size,
      args.body_repeats,
      median_us,
      percentile(samples, 0.05),
      percentile(samples, 0.95),
      *std::min_element(samples.begin(), samples.end()),
      *std::max_element(samples.begin(), samples.end()),
      ops / median_us / 1000.0,
      bytes / median_us,
      max_abs,
      max_rel,
      pass,
  };
}

void write_json(
    const Args& args,
    const hipDeviceProp_t& prop,
    const std::vector<Row>& rows) {
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
  *out << "  \"run_tag\": \"hip-geometry-sweep\",\n";
  *out << "  \"status\": \"diagnostic\",\n";
  *out << "  \"backend\": \"hip\",\n";
  *out << "  \"hardware\": {\n";
  *out << "    \"device_name\": \"" << prop.name << "\",\n";
  *out << "    \"gcn_arch_name\": \"" << prop.gcnArchName << "\",\n";
  *out << "    \"multi_processor_count\": " << prop.multiProcessorCount << "\n";
  *out << "  },\n";
  *out << "  \"config\": {\n";
  *out << "    \"k_list\": ";
  print_json_array(*out, args.k_list);
  *out << ",\n";
  *out << "    \"rows_list\": ";
  print_json_array(*out, args.rows_list);
  *out << ",\n";
  *out << "    \"workgroups\": ";
  print_json_array(*out, args.workgroups);
  *out << ",\n";
  *out << "    \"body_repeats\": " << args.body_repeats << ",\n";
  *out << "    \"hip_fixed_workgroup_size\": " << HIPENGINE_FIXED_WORKGROUP_SIZE << ",\n";
  *out << "    \"reps\": " << args.reps << ",\n";
  *out << "    \"warmup\": " << args.warmup << ",\n";
  *out << "    \"samples\": " << args.samples << ",\n";
  *out << "    \"method\": \"one workgroup per row; repeat-shifted strided f32 FMA inner loop; shared-memory tree reduction\"\n";
  *out << "  },\n";
  *out << "  \"rows\": [\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row& row = rows[i];
    *out << "    {\n";
    *out << "      \"k\": " << row.k << ",\n";
    *out << "      \"rows\": " << row.rows << ",\n";
    *out << "      \"workgroup_size\": " << row.workgroup_size << ",\n";
    *out << "      \"body_repeats\": " << row.body_repeats << ",\n";
    *out << "      \"median_us\": " << row.median_us << ",\n";
    *out << "      \"p05_us\": " << row.p05_us << ",\n";
    *out << "      \"p95_us\": " << row.p95_us << ",\n";
    *out << "      \"min_us\": " << row.min_us << ",\n";
    *out << "      \"max_us\": " << row.max_us << ",\n";
    *out << "      \"gflops\": " << row.gflops << ",\n";
    *out << "      \"bytes_per_us\": " << row.bytes_per_us << ",\n";
    *out << "      \"max_abs\": " << row.max_abs << ",\n";
    *out << "      \"max_rel\": " << row.max_rel << ",\n";
    *out << "      \"correctness_pass\": " << (row.correctness_pass ? "true" : "false") << "\n";
    *out << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
  }
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

    std::vector<Row> rows;
    for (uint32_t k : args.k_list) {
      for (uint32_t row_count : args.rows_list) {
        for (uint32_t workgroup_size : args.workgroups) {
          Row row = run_config(args, k, row_count, workgroup_size);
          rows.push_back(row);
          std::cerr << "[hip] K=" << k << " rows=" << row_count
                    << " wg=" << workgroup_size << " median=" << row.median_us
                    << " us correctness=" << (row.correctness_pass ? "pass" : "fail")
                    << "\n";
        }
      }
    }
    write_json(args, prop, rows);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
