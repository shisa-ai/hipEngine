#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
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

namespace {

constexpr uint32_t QK_K = 256;
constexpr uint32_t Q8_1_WORDS = 9;
constexpr uint32_t Q4_K_BLOCK_BYTES = 144;

struct Args {
  std::string quantize_spirv_path;
  std::string dot_spirv_path;
  std::string json_path;
  std::string x_bf16_path;
  uint32_t x_rows = 4;
  uint32_t rows = 32;
  uint32_t experts = 256;
  uint32_t in_features = 2048;
  uint32_t out_features = 512;
  uint32_t local_size = 64;
  uint32_t reps = 120;
  uint32_t warmup = 30;
  uint32_t samples = 9;
  uint32_t device_index = 0;
  float input_scale = 0.1f;
};

struct PushConstants {
  uint32_t rows;
  uint32_t in_features;
  uint32_t out_features;
  uint32_t experts;
  uint32_t q8_blocks_per_row;
  uint32_t out_packed;
  uint32_t blocks_per_row;
  uint32_t x_rows;
};

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize size = 0;
};

struct TimingStats {
  double median_us = 0.0;
  double p05_us = 0.0;
  double p95_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
  std::vector<double> samples_us;
};

struct Correctness {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double top1 = 0.0;
  uint32_t exact_bf16_mismatches = 0;
  bool pass = false;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void check(VkResult result, const char* what) {
  if (result != VK_SUCCESS) {
    std::ostringstream oss;
    oss << what << " failed with VkResult " << static_cast<int>(result);
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
    if (flag == "--quantize-spirv") {
      args.quantize_spirv_path = require_value(i, argc, argv, flag);
    } else if (flag == "--dot-spirv") {
      args.dot_spirv_path = require_value(i, argc, argv, flag);
    } else if (flag == "--json") {
      args.json_path = require_value(i, argc, argv, flag);
    } else if (flag == "--x-bf16") {
      args.x_bf16_path = require_value(i, argc, argv, flag);
    } else if (flag == "--x-rows") {
      args.x_rows = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--rows") {
      args.rows = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--experts") {
      args.experts = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--in-features") {
      args.in_features = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--out-features") {
      args.out_features = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--local-size") {
      args.local_size = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--reps") {
      args.reps = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--warmup") {
      args.warmup = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--samples") {
      args.samples = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--device-index") {
      args.device_index = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--input-scale") {
      args.input_scale = std::stof(require_value(i, argc, argv, flag));
    } else {
      fail("unknown argument: " + flag);
    }
  }
  if (args.quantize_spirv_path.empty() || args.dot_spirv_path.empty()) {
    fail("--quantize-spirv and --dot-spirv are required");
  }
  if (args.x_rows == 0 || args.rows == 0 || args.experts == 0 || args.in_features == 0 ||
      args.out_features == 0 || args.reps == 0 || args.samples == 0) {
    fail("x_rows, rows, experts, features, reps, and samples must be positive");
  }
  if ((args.rows % args.x_rows) != 0) {
    fail("rows must be divisible by x_rows");
  }
  if ((args.in_features % QK_K) != 0) {
    fail("in_features must be divisible by 256");
  }
  if (args.local_size != 64 && args.local_size != 128 && args.local_size != 256) {
    fail("--local-size must be 64, 128, or 256");
  }
  return args;
}

uint32_t hash_u32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

uint16_t float_to_bf16_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

float bf16_bits_to_float(uint32_t bits) {
  uint32_t word = (bits & 0xffffu) << 16;
  float value = 0.0f;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

uint16_t float_to_half_bits(float value) {
  uint32_t f = 0;
  std::memcpy(&f, &value, sizeof(f));
  uint32_t sign = (f >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((f >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = f & 0x7fffffu;
  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mant |= 0x800000u;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half_mant = mant >> shift;
    if ((mant >> (shift - 1)) & 1u) {
      ++half_mant;
    }
    return static_cast<uint16_t>(sign | half_mant);
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  if (mant & 0x1000u) {
    ++half;
  }
  return static_cast<uint16_t>(half);
}

float half_bits_to_float(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x03ffu;
  uint32_t f = 0;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x03ffu;
      f = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000u | (mant << 13);
  } else {
    f = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &f, sizeof(value));
  return value;
}

void set_byte(std::vector<uint32_t>& words, uint64_t byte_index, uint8_t value) {
  uint64_t word_index = byte_index >> 2;
  uint32_t shift = static_cast<uint32_t>((byte_index & 3u) * 8u);
  uint32_t mask = 0xffu << shift;
  words.at(static_cast<size_t>(word_index)) =
      (words.at(static_cast<size_t>(word_index)) & ~mask) | (static_cast<uint32_t>(value) << shift);
}

uint8_t get_byte(const std::vector<uint32_t>& words, uint64_t byte_index) {
  uint32_t word = words.at(static_cast<size_t>(byte_index >> 2));
  return static_cast<uint8_t>((word >> ((byte_index & 3u) * 8u)) & 0xffu);
}

uint16_t get_u16(const std::vector<uint32_t>& words, uint64_t byte_index) {
  return static_cast<uint16_t>(get_byte(words, byte_index)) |
         (static_cast<uint16_t>(get_byte(words, byte_index + 1)) << 8);
}

uint32_t get_u32(const std::vector<uint32_t>& words, uint64_t byte_index) {
  return static_cast<uint32_t>(get_byte(words, byte_index)) |
         (static_cast<uint32_t>(get_byte(words, byte_index + 1)) << 8) |
         (static_cast<uint32_t>(get_byte(words, byte_index + 2)) << 16) |
         (static_cast<uint32_t>(get_byte(words, byte_index + 3)) << 24);
}

int byte_to_i8(uint8_t value) {
  return static_cast<int>(static_cast<int8_t>(value));
}

int dot_u8_s8(uint32_t a, uint32_t b, int c) {
  int acc = c;
  for (uint32_t lane = 0; lane < 4; ++lane) {
    int av = static_cast<int>((a >> (lane * 8)) & 0xffu);
    int bv = static_cast<int>(static_cast<int8_t>((b >> (lane * 8)) & 0xffu));
    acc += av * bv;
  }
  return acc;
}

std::vector<uint32_t> make_x_bf16(const Args& args) {
  std::vector<uint32_t> x(static_cast<size_t>(args.x_rows) * args.in_features);
  if (!args.x_bf16_path.empty()) {
    std::ifstream file(args.x_bf16_path, std::ios::binary | std::ios::ate);
    if (!file) {
      fail("could not open x BF16 file: " + args.x_bf16_path);
    }
    std::streamsize size = file.tellg();
    size_t expected_bytes = x.size() * sizeof(uint16_t);
    if (size != static_cast<std::streamsize>(expected_bytes)) {
      std::ostringstream oss;
      oss << "x BF16 file has " << size << " bytes, expected " << expected_bytes;
      fail(oss.str());
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint16_t> packed(x.size());
    if (!file.read(reinterpret_cast<char*>(packed.data()), size)) {
      fail("could not read x BF16 file: " + args.x_bf16_path);
    }
    for (size_t i = 0; i < packed.size(); ++i) {
      x[i] = packed[i];
    }
    return x;
  }
  for (uint32_t row = 0; row < args.x_rows; ++row) {
    for (uint32_t k = 0; k < args.in_features; ++k) {
      uint32_t bits = hash_u32(row * 1315423911u + k * 2654435761u + 0x9e3779b9u);
      float centered = static_cast<float>(static_cast<int32_t>(bits & 0xffffu) - 32768) / 32768.0f;
      float value = centered * args.input_scale;
      x[static_cast<size_t>(row) * args.in_features + k] = float_to_bf16_bits(value);
    }
  }
  return x;
}

std::vector<uint32_t> make_selected(const Args& args) {
  std::vector<uint32_t> selected(args.rows);
  for (uint32_t row = 0; row < args.rows; ++row) {
    selected[row] = row % args.experts;
  }
  return selected;
}

void pack_q4_k_scales(const uint8_t* scales, const uint8_t* mins, uint8_t* out) {
  std::fill(out, out + 12, static_cast<uint8_t>(0));
  for (uint32_t i = 0; i < 4; ++i) {
    out[i] = static_cast<uint8_t>((scales[i] & 0x3fu) | ((scales[i + 4] & 0x30u) << 2));
    out[4 + i] = static_cast<uint8_t>((mins[i] & 0x3fu) | ((mins[i + 4] & 0x30u) << 2));
    out[8 + i] = static_cast<uint8_t>((scales[i + 4] & 0x0fu) | ((mins[i + 4] & 0x0fu) << 4));
  }
}

void make_q4_block(uint32_t out_idx, uint32_t block_idx, uint8_t* block) {
  std::fill(block, block + Q4_K_BLOCK_BYTES, static_cast<uint8_t>(0));
  uint16_t d = float_to_half_bits(0.015625f * static_cast<float>(1 + (out_idx % 5)));
  uint16_t dmin = float_to_half_bits(0.0078125f * static_cast<float>(1 + (block_idx % 3)));
  block[0] = static_cast<uint8_t>(d & 0xffu);
  block[1] = static_cast<uint8_t>(d >> 8);
  block[2] = static_cast<uint8_t>(dmin & 0xffu);
  block[3] = static_cast<uint8_t>(dmin >> 8);
  uint8_t scales[8];
  uint8_t mins[8];
  for (uint32_t i = 0; i < 8; ++i) {
    scales[i] = static_cast<uint8_t>((i * 3u + out_idx + block_idx) % 63u + 1u);
    mins[i] = static_cast<uint8_t>((i * 5u + 2u * out_idx + block_idx) % 17u);
  }
  pack_q4_k_scales(scales, mins, block + 4);
  for (uint32_t pair = 0; pair < 4; ++pair) {
    for (uint32_t lane = 0; lane < 32; ++lane) {
      uint32_t k0 = pair * 64u + lane;
      uint32_t k1 = k0 + 32u;
      uint8_t q0 = static_cast<uint8_t>((k0 + out_idx * 7u + block_idx * 11u) % 16u);
      uint8_t q1 = static_cast<uint8_t>((k1 + out_idx * 7u + block_idx * 11u) % 16u);
      block[16 + pair * 32 + lane] = static_cast<uint8_t>(q0 | (q1 << 4));
    }
  }
}

std::vector<uint32_t> make_q4_dual_weights(const Args& args) {
  uint32_t blocks_per_row = args.in_features / QK_K;
  uint64_t row_bytes = static_cast<uint64_t>(blocks_per_row) * Q4_K_BLOCK_BYTES;
  uint64_t expert_bytes = static_cast<uint64_t>(args.out_features) * row_bytes;
  uint64_t tensor_bytes = static_cast<uint64_t>(args.experts) * expert_bytes;
  std::vector<uint32_t> words(static_cast<size_t>((2 * tensor_bytes + 3) / 4), 0);
  uint8_t block[Q4_K_BLOCK_BYTES];
  for (uint32_t expert = 0; expert < args.experts; ++expert) {
    uint32_t shift_a = expert % args.out_features;
    uint32_t shift_b = (expert * 3u) % args.out_features;
    for (uint32_t out_idx = 0; out_idx < args.out_features; ++out_idx) {
      for (uint32_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        uint64_t row_base = static_cast<uint64_t>(expert) * expert_bytes +
            static_cast<uint64_t>(out_idx) * row_bytes +
            static_cast<uint64_t>(block_idx) * Q4_K_BLOCK_BYTES;
        uint32_t source_a = (out_idx + args.out_features - shift_a) % args.out_features;
        make_q4_block(source_a, block_idx, block);
        for (uint32_t byte = 0; byte < Q4_K_BLOCK_BYTES; ++byte) {
          set_byte(words, row_base + byte, block[byte]);
        }
        uint32_t source_b = (out_idx + args.out_features - shift_b) % args.out_features;
        make_q4_block(source_b, block_idx, block);
        for (uint32_t byte = 0; byte < Q4_K_BLOCK_BYTES; ++byte) {
          set_byte(words, tensor_bytes + row_base + byte, block[byte]);
        }
      }
    }
  }
  return words;
}

std::vector<uint32_t> cpu_quantize_q8_1(const std::vector<uint32_t>& x, const Args& args) {
  uint32_t q8_blocks = args.in_features / 32;
  std::vector<uint32_t> xq(static_cast<size_t>(args.x_rows) * q8_blocks * Q8_1_WORDS, 0);
  for (uint32_t row = 0; row < args.x_rows; ++row) {
    for (uint32_t block = 0; block < q8_blocks; ++block) {
      float values[32];
      float amax = 0.0f;
      float sum = 0.0f;
      for (uint32_t lane = 0; lane < 32; ++lane) {
        uint32_t k = block * 32 + lane;
        float value = bf16_bits_to_float(x[static_cast<size_t>(row) * args.in_features + k]);
        values[lane] = value;
        amax = std::max(amax, std::fabs(value));
        sum += value;
      }
      float d = amax / 127.0f;
      uint64_t base_word = (static_cast<uint64_t>(row) * q8_blocks + block) * Q8_1_WORDS;
      uint32_t dsum = static_cast<uint32_t>(float_to_half_bits(d)) |
          (static_cast<uint32_t>(float_to_half_bits(sum)) << 16);
      xq[static_cast<size_t>(base_word)] = dsum;
      for (uint32_t lane = 0; lane < 32; ++lane) {
        int q = amax == 0.0f ? 0 : static_cast<int>(std::round(values[lane] / d));
        q = std::max(-128, std::min(127, q));
        uint32_t word = 1 + (lane >> 2);
        uint32_t shift = (lane & 3u) * 8u;
        xq[static_cast<size_t>(base_word + word)] |= (static_cast<uint32_t>(q) & 0xffu) << shift;
      }
    }
  }
  return xq;
}

uint8_t q4_k_scale(const std::vector<uint32_t>& weights, uint64_t scales_byte, uint32_t subblock) {
  if (subblock < 4) {
    return static_cast<uint8_t>(get_byte(weights, scales_byte + subblock) & 0x3fu);
  }
  uint32_t idx = subblock - 4;
  return static_cast<uint8_t>(
      (get_byte(weights, scales_byte + 8 + idx) & 0x0fu) |
      ((get_byte(weights, scales_byte + idx) >> 2) & 0x30u));
}

uint8_t q4_k_min(const std::vector<uint32_t>& weights, uint64_t scales_byte, uint32_t subblock) {
  if (subblock < 4) {
    return static_cast<uint8_t>(get_byte(weights, scales_byte + 4 + subblock) & 0x3fu);
  }
  uint32_t idx = subblock - 4;
  return static_cast<uint8_t>(
      (get_byte(weights, scales_byte + 8 + idx) >> 4) |
      ((get_byte(weights, scales_byte + 4 + idx) >> 2) & 0x30u));
}

uint32_t q4_k_pack4(const std::vector<uint32_t>& weights, uint64_t block_byte, uint32_t subblock, uint32_t lane4) {
  uint32_t pair = subblock >> 1;
  uint32_t packed = get_u32(weights, block_byte + 16 + pair * 32 + lane4);
  return (subblock & 1u) ? ((packed >> 4) & 0x0f0f0f0fu) : (packed & 0x0f0f0f0fu);
}

float cpu_q4_term(
    const std::vector<uint32_t>& weights,
    uint64_t block_byte,
    uint32_t subblock,
    uint32_t lane4,
    uint32_t x_pack,
    int q8_sum,
    float xd) {
  int dot = dot_u8_s8(q4_k_pack4(weights, block_byte, subblock, lane4), x_pack, 0);
  float d = half_bits_to_float(get_u16(weights, block_byte));
  float dmin = half_bits_to_float(get_u16(weights, block_byte + 2));
  uint64_t scales = block_byte + 4;
  float scale = static_cast<float>(q4_k_scale(weights, scales, subblock));
  float minv = static_cast<float>(q4_k_min(weights, scales, subblock));
  return xd * (d * scale * static_cast<float>(dot) - dmin * minv * static_cast<float>(q8_sum));
}

std::vector<uint32_t> cpu_dot_dual(
    const std::vector<uint32_t>& xq,
    const std::vector<uint32_t>& selected,
    const std::vector<uint32_t>& weights,
    const Args& args) {
  uint32_t q8_blocks = args.in_features / 32;
  uint32_t blocks_per_row = args.in_features / QK_K;
  uint64_t row_bytes = static_cast<uint64_t>(blocks_per_row) * Q4_K_BLOCK_BYTES;
  uint64_t expert_bytes = static_cast<uint64_t>(args.out_features) * row_bytes;
  uint64_t tensor_bytes = static_cast<uint64_t>(args.experts) * expert_bytes;
  uint32_t groups4 = args.in_features >> 2;
  std::vector<uint32_t> out(static_cast<size_t>(2) * args.rows * args.out_features, 0);
  uint32_t lanes_per_x_row = args.rows / args.x_rows;
  for (uint32_t row = 0; row < args.rows; ++row) {
    uint32_t expert = selected[row];
    uint32_t x_row = args.x_rows == 1 ? 0 : (row / lanes_per_x_row);
    for (uint32_t out_col = 0; out_col < args.out_features; ++out_col) {
      float acc_a = 0.0f;
      float acc_b = 0.0f;
      uint64_t weight_a = static_cast<uint64_t>(expert) * expert_bytes +
          static_cast<uint64_t>(out_col) * row_bytes;
      uint64_t weight_b = tensor_bytes + weight_a;
      for (uint32_t group = 0; group < groups4; ++group) {
        uint32_t k4 = group << 2;
        uint32_t block_idx = k4 / QK_K;
        uint32_t within = k4 - block_idx * QK_K;
        uint32_t subblock = within >> 5;
        uint32_t lane4 = within & 31u;
        uint32_t q8_index = block_idx * 8u + subblock;
        uint64_t xq_base = (static_cast<uint64_t>(x_row) * q8_blocks + q8_index) * Q8_1_WORDS;
        uint32_t x_pack = xq[static_cast<size_t>(xq_base + 1u + (lane4 >> 2))];
        int q8_sum = dot_u8_s8(0x01010101u, x_pack, 0);
        float xd = half_bits_to_float(static_cast<uint16_t>(xq[static_cast<size_t>(xq_base)] & 0xffffu));
        acc_a += cpu_q4_term(
            weights,
            weight_a + static_cast<uint64_t>(block_idx) * Q4_K_BLOCK_BYTES,
            subblock,
            lane4,
            x_pack,
            q8_sum,
            xd);
        acc_b += cpu_q4_term(
            weights,
            weight_b + static_cast<uint64_t>(block_idx) * Q4_K_BLOCK_BYTES,
            subblock,
            lane4,
            x_pack,
            q8_sum,
            xd);
      }
      uint64_t out_idx = static_cast<uint64_t>(row) * args.out_features + out_col;
      out[static_cast<size_t>(out_idx)] = float_to_bf16_bits(acc_a);
      out[static_cast<size_t>(args.rows) * args.out_features + out_idx] = float_to_bf16_bits(acc_b);
    }
  }
  return out;
}

Correctness compare_outputs(const std::vector<uint32_t>& expected, const std::vector<uint32_t>& actual, const Args& args) {
  Correctness result{};
  double total_abs = 0.0;
  uint32_t top1_match = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    if ((expected[i] & 0xffffu) != (actual[i] & 0xffffu)) {
      ++result.exact_bf16_mismatches;
    }
    double diff = std::fabs(
        static_cast<double>(bf16_bits_to_float(expected[i])) -
        static_cast<double>(bf16_bits_to_float(actual[i])));
    result.max_abs = std::max(result.max_abs, diff);
    total_abs += diff;
  }
  result.mean_abs = total_abs / static_cast<double>(expected.size());
  for (uint32_t tensor = 0; tensor < 2; ++tensor) {
    size_t tensor_base = static_cast<size_t>(tensor) * args.rows * args.out_features;
    for (uint32_t row = 0; row < args.rows; ++row) {
      uint32_t expected_argmax = 0;
      uint32_t actual_argmax = 0;
      float expected_best = -std::numeric_limits<float>::infinity();
      float actual_best = -std::numeric_limits<float>::infinity();
      for (uint32_t col = 0; col < args.out_features; ++col) {
        size_t idx = tensor_base + static_cast<size_t>(row) * args.out_features + col;
        float ev = bf16_bits_to_float(expected[idx]);
        float av = bf16_bits_to_float(actual[idx]);
        if (ev > expected_best) {
          expected_best = ev;
          expected_argmax = col;
        }
        if (av > actual_best) {
          actual_best = av;
          actual_argmax = col;
        }
      }
      if (expected_argmax == actual_argmax) {
        ++top1_match;
      }
    }
  }
  result.top1 = static_cast<double>(top1_match) / static_cast<double>(2 * args.rows);
  result.pass = result.max_abs <= 2.0 && result.top1 == 1.0;
  return result;
}

std::vector<uint32_t> read_spirv(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    fail("could not open SPIR-V file: " + path);
  }
  std::streamsize size = file.tellg();
  if (size <= 0 || (size % 4) != 0) {
    fail("SPIR-V file size must be a positive multiple of 4");
  }
  file.seekg(0, std::ios::beg);
  std::vector<uint32_t> words(static_cast<size_t>(size) / sizeof(uint32_t));
  if (!file.read(reinterpret_cast<char*>(words.data()), size)) {
    fail("could not read SPIR-V file: " + path);
  }
  return words;
}

uint32_t find_queue_family(VkPhysicalDevice physical_device) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());
  for (uint32_t i = 0; i < count; ++i) {
    if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
      return i;
    }
  }
  fail("physical device has no compute queue family");
}

bool has_device_extension(VkPhysicalDevice physical_device, const char* extension_name) {
  uint32_t count = 0;
  check(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr),
        "vkEnumerateDeviceExtensionProperties(count)");
  std::vector<VkExtensionProperties> extensions(count);
  check(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, extensions.data()),
        "vkEnumerateDeviceExtensionProperties(list)");
  for (const VkExtensionProperties& extension : extensions) {
    if (std::strcmp(extension.extensionName, extension_name) == 0) {
      return true;
    }
  }
  return false;
}

void require_integer_dot_product(VkPhysicalDevice physical_device) {
  if (!has_device_extension(physical_device, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) {
    fail("physical device does not expose VK_KHR_shader_integer_dot_product");
  }
  VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dot_features{};
  dot_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
  VkPhysicalDeviceFeatures2 features2{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &dot_features;
  vkGetPhysicalDeviceFeatures2(physical_device, &features2);
  if (dot_features.shaderIntegerDotProduct != VK_TRUE) {
    fail("physical device reports shaderIntegerDotProduct=false");
  }
}

VkDevice create_device(VkPhysicalDevice physical_device, uint32_t queue_family, const float* queue_priority) {
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = queue_family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = queue_priority;

  VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dot_features{};
  dot_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
  dot_features.shaderIntegerDotProduct = VK_TRUE;

  VkPhysicalDeviceFeatures2 features2{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &dot_features;

  const char* extensions[] = {VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME};
  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.pNext = &features2;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount = 1;
  device_info.ppEnabledExtensionNames = extensions;

  VkDevice device = VK_NULL_HANDLE;
  check(vkCreateDevice(physical_device, &device_info, nullptr, &device), "vkCreateDevice");
  return device;
}

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_bits, VkMemoryPropertyFlags required) {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) != 0 &&
        (properties.memoryTypes[i].propertyFlags & required) == required) {
      return i;
    }
  }
  fail("no compatible memory type found");
}

Buffer create_buffer(
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    bool map) {
  Buffer buffer{};
  buffer.size = size;
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check(vkCreateBuffer(device, &buffer_info, nullptr, &buffer.buffer), "vkCreateBuffer");
  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
  VkMemoryAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex = find_memory_type(physical_device, requirements.memoryTypeBits, properties);
  check(vkAllocateMemory(device, &allocate_info, nullptr, &buffer.memory), "vkAllocateMemory");
  check(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0), "vkBindBufferMemory");
  if (map) {
    check(vkMapMemory(device, buffer.memory, 0, size, 0, &buffer.mapped), "vkMapMemory");
  }
  return buffer;
}

void destroy_buffer(VkDevice device, Buffer& buffer) {
  if (buffer.mapped != nullptr) {
    vkUnmapMemory(device, buffer.memory);
    buffer.mapped = nullptr;
  }
  if (buffer.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device, buffer.buffer, nullptr);
    buffer.buffer = VK_NULL_HANDLE;
  }
  if (buffer.memory != VK_NULL_HANDLE) {
    vkFreeMemory(device, buffer.memory, nullptr);
    buffer.memory = VK_NULL_HANDLE;
  }
}

VkCommandBuffer begin_one_time(VkDevice device, VkCommandPool command_pool) {
  VkCommandBufferAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate_info.commandPool = command_pool;
  allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate_info.commandBufferCount = 1;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  check(vkAllocateCommandBuffers(device, &allocate_info, &command_buffer), "vkAllocateCommandBuffers");
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");
  return command_buffer;
}

void submit_and_free(VkDevice device, VkQueue queue, VkCommandPool command_pool, VkCommandBuffer command_buffer) {
  check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  check(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE), "vkQueueSubmit");
  check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
  vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}

void copy_to_device(
    VkDevice device,
    VkQueue queue,
    VkCommandPool command_pool,
    const Buffer& stage,
    const Buffer& device_buffer,
    VkDeviceSize bytes) {
  VkCommandBuffer cmd = begin_one_time(device, command_pool);
  VkBufferCopy copy{};
  copy.size = bytes;
  vkCmdCopyBuffer(cmd, stage.buffer, device_buffer.buffer, 1, &copy);
  submit_and_free(device, queue, command_pool, cmd);
}

void buffer_barrier(
    VkCommandBuffer cmd,
    const Buffer& buffer,
    VkAccessFlags src_access,
    VkAccessFlags dst_access,
    VkPipelineStageFlags src_stage,
    VkPipelineStageFlags dst_stage) {
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = buffer.buffer;
  barrier.offset = 0;
  barrier.size = buffer.size;
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

VkDescriptorSetLayout create_descriptor_set_layout(VkDevice device) {
  std::vector<VkDescriptorSetLayoutBinding> bindings(5);
  for (uint32_t i = 0; i < static_cast<uint32_t>(bindings.size()); ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  create_info.bindingCount = static_cast<uint32_t>(bindings.size());
  create_info.pBindings = bindings.data();
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  check(vkCreateDescriptorSetLayout(device, &create_info, nullptr, &layout), "vkCreateDescriptorSetLayout");
  return layout;
}

VkPipelineLayout create_pipeline_layout(VkDevice device, VkDescriptorSetLayout descriptor_set_layout) {
  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = sizeof(PushConstants);
  VkPipelineLayoutCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  create_info.setLayoutCount = 1;
  create_info.pSetLayouts = &descriptor_set_layout;
  create_info.pushConstantRangeCount = 1;
  create_info.pPushConstantRanges = &push_range;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  check(vkCreatePipelineLayout(device, &create_info, nullptr, &layout), "vkCreatePipelineLayout");
  return layout;
}

VkShaderModule create_shader_module(VkDevice device, const std::vector<uint32_t>& spirv) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = spirv.size() * sizeof(uint32_t);
  create_info.pCode = spirv.data();
  VkShaderModule module = VK_NULL_HANDLE;
  check(vkCreateShaderModule(device, &create_info, nullptr, &module), "vkCreateShaderModule");
  return module;
}

VkPipeline create_pipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule module) {
  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = module;
  stage.pName = "main";
  VkComputePipelineCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  create_info.stage = stage;
  create_info.layout = layout;
  VkPipeline pipeline = VK_NULL_HANDLE;
  check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &create_info, nullptr, &pipeline),
        "vkCreateComputePipelines");
  return pipeline;
}

VkDescriptorSet create_descriptor_set(
    VkDevice device,
    VkDescriptorSetLayout descriptor_set_layout,
    const Buffer& x_device,
    const Buffer& xq_device,
    const Buffer& selected_device,
    const Buffer& tiles_device,
    const Buffer& out_device,
    VkDescriptorPool& descriptor_pool) {
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 5;
  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool), "vkCreateDescriptorPool");

  VkDescriptorSetAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocate_info.descriptorPool = descriptor_pool;
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &descriptor_set_layout;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  check(vkAllocateDescriptorSets(device, &allocate_info, &descriptor_set), "vkAllocateDescriptorSets");

  VkDescriptorBufferInfo infos[5] = {
      {x_device.buffer, 0, x_device.size},
      {xq_device.buffer, 0, xq_device.size},
      {selected_device.buffer, 0, selected_device.size},
      {tiles_device.buffer, 0, tiles_device.size},
      {out_device.buffer, 0, out_device.size},
  };
  std::vector<VkWriteDescriptorSet> writes(5);
  for (uint32_t i = 0; i < static_cast<uint32_t>(writes.size()); ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descriptor_set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  return descriptor_set;
}

void record_quantize(
    VkCommandBuffer cmd,
    VkPipeline pipeline,
    VkPipelineLayout layout,
    VkDescriptorSet descriptor_set,
    const PushConstants& push,
    uint32_t reps) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descriptor_set, 0, nullptr);
  vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  for (uint32_t rep = 0; rep < reps; ++rep) {
    vkCmdDispatch(cmd, push.q8_blocks_per_row, push.rows, 1);
  }
}

void record_dot(
    VkCommandBuffer cmd,
    VkPipeline pipeline,
    VkPipelineLayout layout,
    VkDescriptorSet descriptor_set,
    const PushConstants& push,
    uint32_t reps) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descriptor_set, 0, nullptr);
  vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  for (uint32_t rep = 0; rep < reps; ++rep) {
    vkCmdDispatch(cmd, push.out_packed, push.rows, 1);
  }
}

void record_quantize_dot(
    VkCommandBuffer cmd,
    VkPipeline quant_pipeline,
    VkPipeline dot_pipeline,
    VkPipelineLayout layout,
    VkDescriptorSet descriptor_set,
    const PushConstants& quant_push,
    const PushConstants& dot_push,
    const Buffer& xq_device,
    uint32_t reps) {
  for (uint32_t rep = 0; rep < reps; ++rep) {
    record_quantize(cmd, quant_pipeline, layout, descriptor_set, quant_push, 1);
    buffer_barrier(
        cmd,
        xq_device,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    record_dot(cmd, dot_pipeline, layout, descriptor_set, dot_push, 1);
  }
}

double percentile(std::vector<double> values, double q) {
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

double submit_once(VkDevice device, VkQueue queue, VkCommandBuffer command_buffer, VkFence fence) {
  check(vkResetFences(device, 1, &fence), "vkResetFences");
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  auto t0 = std::chrono::steady_clock::now();
  check(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
  check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

TimingStats time_command(
    VkDevice device,
    VkQueue queue,
    VkCommandBuffer command_buffer,
    VkFence fence,
    uint32_t reps,
    uint32_t warmup,
    uint32_t samples) {
  for (uint32_t i = 0; i < warmup; ++i) {
    (void)submit_once(device, queue, command_buffer, fence);
  }
  TimingStats stats{};
  stats.samples_us.reserve(samples);
  for (uint32_t sample = 0; sample < samples; ++sample) {
    stats.samples_us.push_back(submit_once(device, queue, command_buffer, fence) / reps);
  }
  stats.median_us = percentile(stats.samples_us, 0.5);
  stats.p05_us = percentile(stats.samples_us, 0.05);
  stats.p95_us = percentile(stats.samples_us, 0.95);
  stats.min_us = *std::min_element(stats.samples_us.begin(), stats.samples_us.end());
  stats.max_us = *std::max_element(stats.samples_us.begin(), stats.samples_us.end());
  return stats;
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

std::string version_string(uint32_t version) {
  std::ostringstream out;
  out << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "." << VK_VERSION_PATCH(version);
  return out.str();
}

void write_timing_json(const char* name, const TimingStats& stats, std::ostream& out, const char* suffix) {
  out << "    \"" << name << "\": {\n";
  out << "      \"median_us\": " << stats.median_us << ",\n";
  out << "      \"median_ms\": " << stats.median_us / 1000.0 << ",\n";
  out << "      \"p05_us\": " << stats.p05_us << ",\n";
  out << "      \"p95_us\": " << stats.p95_us << ",\n";
  out << "      \"min_us\": " << stats.min_us << ",\n";
  out << "      \"max_us\": " << stats.max_us << ",\n";
  out << "      \"samples_us\": [";
  for (size_t i = 0; i < stats.samples_us.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << stats.samples_us[i];
  }
  out << "]\n";
  out << "    }" << suffix << "\n";
}

std::string command_line(int argc, char** argv) {
  std::ostringstream out;
  for (int i = 0; i < argc; ++i) {
    if (i != 0) {
      out << " ";
    }
    out << argv[i];
  }
  return out.str();
}

void write_json(
    const Args& args,
    int argc,
    char** argv,
    const VkPhysicalDeviceProperties& properties,
    uint32_t queue_family,
    const Correctness& correctness,
    const TimingStats& quantize_stats,
    const TimingStats& dot_stats,
    const TimingStats& combined_stats,
    std::ostream& out) {
  out << std::setprecision(10);
  out << "{\n";
  out << "  \"schema\": \"hipengine.micro.vulkan_q4_selected_dual.v1\",\n";
  out << "  \"backend\": \"vulkan\",\n";
  out << "  \"classification\": \"real_slice_probe\",\n";
  out << "  \"command\": \"" << json_escape(command_line(argc, argv)) << "\",\n";
  out << "  \"hardware\": {\n";
  out << "    \"device_name\": \"" << json_escape(properties.deviceName) << "\",\n";
  out << "    \"vendor_id\": " << properties.vendorID << ",\n";
  out << "    \"device_id\": " << properties.deviceID << ",\n";
  out << "    \"device_type\": " << properties.deviceType << ",\n";
  out << "    \"api_version\": \"" << version_string(properties.apiVersion) << "\",\n";
  out << "    \"driver_version_raw\": " << properties.driverVersion << ",\n";
  out << "    \"queue_family\": " << queue_family << ",\n";
  out << "    \"shader_integer_dot_product\": true\n";
  out << "  },\n";
  out << "  \"shape\": {\n";
  out << "    \"quant\": \"q4_k\",\n";
  out << "    \"x_rows\": " << args.x_rows << ",\n";
  out << "    \"rows\": " << args.rows << ",\n";
  out << "    \"experts\": " << args.experts << ",\n";
  out << "    \"in_features\": " << args.in_features << ",\n";
  out << "    \"out_features\": " << args.out_features << ",\n";
  out << "    \"input_scale\": " << args.input_scale << ",\n";
  out << "    \"local_size\": " << args.local_size << ",\n";
  out << "    \"q8_blocks_per_row\": " << args.in_features / 32 << ",\n";
  out << "    \"blocks_per_row\": " << args.in_features / QK_K << "\n";
  out << "  },\n";
  out << "  \"timing_config\": {\n";
  out << "    \"reps\": " << args.reps << ",\n";
  out << "    \"warmup\": " << args.warmup << ",\n";
  out << "    \"samples\": " << args.samples << ",\n";
  out << "    \"method\": \"pre-recorded Vulkan command buffer, host wall divided by reps; transfer and pipeline creation excluded\"\n";
  out << "  },\n";
  out << "  \"timing\": {\n";
  write_timing_json("q8_1_quantize", quantize_stats, out, ",");
  write_timing_json("selected_dual_dp4a_dot_prequantized", dot_stats, out, ",");
  write_timing_json("selected_dual_dp4a_quantize_plus_dot", combined_stats, out, "");
  out << "  },\n";
  out << "  \"correctness_vs_cpu\": {\n";
  out << "    \"oracle\": \"full CPU reference for q8_1 quantize plus Q4_K selected-dual gate/up dp4a, bf16 output bits compared after shader run\",\n";
  out << "    \"max_abs\": " << correctness.max_abs << ",\n";
  out << "    \"mean_abs\": " << correctness.mean_abs << ",\n";
  out << "    \"top1\": " << correctness.top1 << ",\n";
  out << "    \"exact_bf16_mismatches\": " << correctness.exact_bf16_mismatches << ",\n";
  out << "    \"pass\": " << (correctness.pass ? "true" : "false") << "\n";
  out << "  },\n";
  out << "  \"notes\": \"Matched production-shaped Vulkan probe for the retained HIP Q4_K selected-dual gate/up q8_1+dp4a slice; synthetic deterministic data, same GGUF Q4_K byte layout and expert-row rotations.\"\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    std::vector<uint32_t> quantize_spirv = read_spirv(args.quantize_spirv_path);
    std::vector<uint32_t> dot_spirv = read_spirv(args.dot_spirv_path);
    std::vector<uint32_t> x = make_x_bf16(args);
    std::vector<uint32_t> selected = make_selected(args);
    std::vector<uint32_t> weights = make_q4_dual_weights(args);
    std::vector<uint32_t> expected = cpu_dot_dual(cpu_quantize_q8_1(x, args), selected, weights, args);
    std::vector<uint32_t> actual(expected.size(), 0);

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "hipEngine Vulkan Q4 selected-dual";
    app_info.applicationVersion = 1;
    app_info.pEngineName = "hipEngine microbench";
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    VkInstance instance = VK_NULL_HANDLE;
    check(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

    uint32_t physical_count = 0;
    check(vkEnumeratePhysicalDevices(instance, &physical_count, nullptr),
          "vkEnumeratePhysicalDevices(count)");
    if (physical_count == 0) {
      fail("no Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    check(vkEnumeratePhysicalDevices(instance, &physical_count, physical_devices.data()),
          "vkEnumeratePhysicalDevices(list)");
    if (args.device_index >= physical_count) {
      fail("--device-index is outside the physical-device list");
    }
    VkPhysicalDevice physical_device = physical_devices[args.device_index];
    require_integer_dot_product(physical_device);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    uint32_t queue_family = find_queue_family(physical_device);

    float queue_priority = 1.0f;
    VkDevice device = create_device(physical_device, queue_family, &queue_priority);
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkShaderModule quant_module = create_shader_module(device, quantize_spirv);
    VkShaderModule dot_module = create_shader_module(device, dot_spirv);
    VkDescriptorSetLayout descriptor_layout = create_descriptor_set_layout(device);
    VkPipelineLayout pipeline_layout = create_pipeline_layout(device, descriptor_layout);
    VkPipeline quant_pipeline = create_pipeline(device, pipeline_layout, quant_module);
    VkPipeline dot_pipeline = create_pipeline(device, pipeline_layout, dot_module);

    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.queueFamilyIndex = queue_family;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    check(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");

    VkDeviceSize x_bytes = sizeof(uint32_t) * x.size();
    VkDeviceSize selected_bytes = sizeof(uint32_t) * selected.size();
    VkDeviceSize weights_bytes = sizeof(uint32_t) * weights.size();
    VkDeviceSize xq_bytes = sizeof(uint32_t) * static_cast<uint64_t>(args.x_rows) *
        (args.in_features / 32) * Q8_1_WORDS;
    VkDeviceSize out_bytes = sizeof(uint32_t) * actual.size();

    Buffer x_stage = create_buffer(
        physical_device,
        device,
        x_bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        true);
    Buffer selected_stage = create_buffer(
        physical_device,
        device,
        selected_bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        true);
    Buffer weights_stage = create_buffer(
        physical_device,
        device,
        weights_bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        true);
    Buffer out_stage = create_buffer(
        physical_device,
        device,
        out_bytes,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        true);
    Buffer x_device = create_buffer(
        physical_device,
        device,
        x_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        false);
    Buffer selected_device = create_buffer(
        physical_device,
        device,
        selected_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        false);
    Buffer weights_device = create_buffer(
        physical_device,
        device,
        weights_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        false);
    Buffer xq_device = create_buffer(
        physical_device,
        device,
        xq_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        false);
    Buffer out_device = create_buffer(
        physical_device,
        device,
        out_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        false);

    std::memcpy(x_stage.mapped, x.data(), static_cast<size_t>(x_bytes));
    std::memcpy(selected_stage.mapped, selected.data(), static_cast<size_t>(selected_bytes));
    std::memcpy(weights_stage.mapped, weights.data(), static_cast<size_t>(weights_bytes));
    copy_to_device(device, queue, command_pool, x_stage, x_device, x_bytes);
    copy_to_device(device, queue, command_pool, selected_stage, selected_device, selected_bytes);
    copy_to_device(device, queue, command_pool, weights_stage, weights_device, weights_bytes);

    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = create_descriptor_set(
        device,
        descriptor_layout,
        x_device,
        xq_device,
        selected_device,
        weights_device,
        out_device,
        descriptor_pool);

    PushConstants quant_push{
        args.x_rows,
        args.in_features,
        args.out_features,
        args.experts,
        args.in_features / 32,
        args.out_features,
        args.in_features / QK_K,
        args.x_rows};
    PushConstants dot_push{
        args.rows,
        args.in_features,
        args.out_features,
        args.experts,
        args.in_features / 32,
        args.out_features,
        args.in_features / QK_K,
        args.x_rows};

    VkCommandBuffer correctness_cmd = begin_one_time(device, command_pool);
    record_quantize_dot(
        correctness_cmd,
        quant_pipeline,
        dot_pipeline,
        pipeline_layout,
        descriptor_set,
        quant_push,
        dot_push,
        xq_device,
        1);
    buffer_barrier(
        correctness_cmd,
        out_device,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferCopy out_copy{};
    out_copy.size = out_bytes;
    vkCmdCopyBuffer(correctness_cmd, out_device.buffer, out_stage.buffer, 1, &out_copy);
    submit_and_free(device, queue, command_pool, correctness_cmd);
    std::memcpy(actual.data(), out_stage.mapped, static_cast<size_t>(out_bytes));
    Correctness correctness = compare_outputs(expected, actual, args);
    if (!correctness.pass) {
      std::cerr << "warning: correctness failed max_abs=" << correctness.max_abs
                << " top1=" << correctness.top1
                << " bf16_mismatches=" << correctness.exact_bf16_mismatches << "\n";
    }

    VkCommandBuffer quant_cmd = begin_one_time(device, command_pool);
    record_quantize(quant_cmd, quant_pipeline, pipeline_layout, descriptor_set, quant_push, args.reps);
    check(vkEndCommandBuffer(quant_cmd), "vkEndCommandBuffer quantize");

    VkCommandBuffer dot_cmd = begin_one_time(device, command_pool);
    record_dot(dot_cmd, dot_pipeline, pipeline_layout, descriptor_set, dot_push, args.reps);
    check(vkEndCommandBuffer(dot_cmd), "vkEndCommandBuffer dot");

    VkCommandBuffer combined_cmd = begin_one_time(device, command_pool);
    record_quantize_dot(
        combined_cmd,
        quant_pipeline,
        dot_pipeline,
        pipeline_layout,
        descriptor_set,
        quant_push,
        dot_push,
        xq_device,
        args.reps);
    check(vkEndCommandBuffer(combined_cmd), "vkEndCommandBuffer combined");

    TimingStats quantize_stats = time_command(device, queue, quant_cmd, fence, args.reps, args.warmup, args.samples);
    TimingStats dot_stats = time_command(device, queue, dot_cmd, fence, args.reps, args.warmup, args.samples);
    TimingStats combined_stats = time_command(device, queue, combined_cmd, fence, args.reps, args.warmup, args.samples);

    std::cout << "[vulkan-q4-selected-dual] quantize=" << quantize_stats.median_us / 1000.0
              << " ms dot=" << dot_stats.median_us / 1000.0
              << " ms combined=" << combined_stats.median_us / 1000.0
              << " ms correctness=" << (correctness.pass ? "pass" : "fail") << "\n";

    if (args.json_path.empty()) {
      write_json(args, argc, argv, properties, queue_family, correctness, quantize_stats, dot_stats, combined_stats, std::cout);
    } else {
      std::ofstream output(args.json_path);
      if (!output) {
        fail("could not open JSON path: " + args.json_path);
      }
      write_json(args, argc, argv, properties, queue_family, correctness, quantize_stats, dot_stats, combined_stats, output);
    }

    vkFreeCommandBuffers(device, command_pool, 1, &quant_cmd);
    vkFreeCommandBuffers(device, command_pool, 1, &dot_cmd);
    vkFreeCommandBuffers(device, command_pool, 1, &combined_cmd);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    destroy_buffer(device, x_stage);
    destroy_buffer(device, selected_stage);
    destroy_buffer(device, weights_stage);
    destroy_buffer(device, out_stage);
    destroy_buffer(device, x_device);
    destroy_buffer(device, selected_device);
    destroy_buffer(device, weights_device);
    destroy_buffer(device, xq_device);
    destroy_buffer(device, out_device);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyPipeline(device, quant_pipeline, nullptr);
    vkDestroyPipeline(device, dot_pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    vkDestroyShaderModule(device, quant_module, nullptr);
    vkDestroyShaderModule(device, dot_module, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return correctness.pass ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
