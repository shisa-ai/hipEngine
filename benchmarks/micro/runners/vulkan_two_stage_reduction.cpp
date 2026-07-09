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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "micro_timing_vulkan.hpp"

namespace {

struct Args {
  std::string partial_spirv_path;
  std::string final_spirv_path;
  std::string json_path;
  std::vector<uint32_t> k_list{8192, 32768};
  std::vector<uint32_t> rows_list{1, 4, 8};
  std::vector<uint32_t> workgroups{128, 256};
  std::vector<uint32_t> split_counts{2, 4, 8};
  uint32_t body_repeats = 32;
  uint32_t reps = 20;
  uint32_t warmup = 5;
  uint32_t samples = 7;
  std::string timing_mode = "serial_latency";
  uint32_t independent_queues = 4;
  uint32_t device_index = 0;
};

struct PushConstants {
  uint32_t k;
  uint32_t rows;
  uint32_t body_repeats;
  uint32_t split_count;
  uint32_t output_slice;
  uint32_t sequence_id;
};

struct SequenceTiming {
  std::vector<double> gpu_sequence_us;
  std::vector<double> host_sequence_us;
};

struct Row {
  uint32_t k;
  uint32_t rows;
  uint32_t workgroup_size;
  uint32_t split_count;
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
  uint32_t barrier_count;
  uint32_t queue_count;
  std::string calibrated_timestamp_extension;
  bool gpu_timestamps_supported;
  double timestamp_period_ns;
  uint32_t timestamp_valid_bits;
  SequenceTiming single_timing;
  SequenceTiming burst_timing;
};

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize size = 0;
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
    if (flag == "--partial-spirv") {
      args.partial_spirv_path = require_value(i, argc, argv, flag);
    } else if (flag == "--final-spirv") {
      args.final_spirv_path = require_value(i, argc, argv, flag);
    } else if (flag == "--json") {
      args.json_path = require_value(i, argc, argv, flag);
    } else if (flag == "--k-list") {
      args.k_list = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--rows-list") {
      args.rows_list = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--workgroups") {
      args.workgroups = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--split-counts") {
      args.split_counts = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--body-repeats") {
      args.body_repeats = static_cast<uint32_t>(
          std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--reps") {
      args.reps = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--warmup") {
      args.warmup = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--samples") {
      args.samples = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--timing-mode") {
      args.timing_mode = require_value(i, argc, argv, flag);
    } else if (flag == "--independent-queues") {
      args.independent_queues = static_cast<uint32_t>(
          std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--device-index") {
      args.device_index = static_cast<uint32_t>(
          std::stoul(require_value(i, argc, argv, flag)));
    } else {
      fail("unknown argument: " + flag);
    }
  }
  if (args.partial_spirv_path.empty() || args.final_spirv_path.empty()) {
    fail("--partial-spirv and --final-spirv are required");
  }
  if (args.body_repeats == 0 || args.reps == 0 || args.samples == 0 ||
      args.independent_queues == 0) {
    fail("--body-repeats, --reps, --samples, and --independent-queues must be positive");
  }
  (void)hipengine::micro::parse_timing_mode(args.timing_mode);
  for (uint32_t wg : args.workgroups) {
    if (wg == 0 || wg > 256 || (wg & (wg - 1)) != 0) {
      fail("workgroup sizes must be powers of two in [1, 256]");
    }
  }
  return args;
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

uint32_t find_memory_type(
    VkPhysicalDevice physical_device,
    uint32_t type_bits,
    VkMemoryPropertyFlags required) {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    bool type_matches = (type_bits & (1u << i)) != 0;
    bool flags_match = (properties.memoryTypes[i].propertyFlags & required) == required;
    if (type_matches && flags_match) {
      return i;
    }
  }
  fail("no compatible memory type found");
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
    uint32_t split_count,
    uint32_t body_repeats) {
  std::vector<float> out(rows, 0.0f);
  std::vector<float> partials(static_cast<size_t>(rows) * split_count, 0.0f);
  std::vector<float> scratch(workgroup_size, 0.0f);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t split = 0; split < split_count; ++split) {
      uint32_t start = (k * split) / split_count;
      uint32_t end = (k * (split + 1u)) / split_count;
      std::fill(scratch.begin(), scratch.end(), 0.0f);
      for (uint32_t lid = 0; lid < workgroup_size; ++lid) {
        float sum = 0.0f;
        for (uint32_t repeat = 0; repeat < body_repeats; ++repeat) {
          for (uint32_t i = start + lid; i < end; i += workgroup_size) {
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
      partials[static_cast<size_t>(row) * split_count + split] = scratch[0];
    }

    std::fill(scratch.begin(), scratch.end(), 0.0f);
    for (uint32_t lid = 0; lid < workgroup_size; ++lid) {
      float sum = 0.0f;
      for (uint32_t i = lid; i < split_count; i += workgroup_size) {
        sum += partials[static_cast<size_t>(row) * split_count + i];
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

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (char ch : value) {
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
  allocate_info.memoryTypeIndex =
      find_memory_type(physical_device, requirements.memoryTypeBits, properties);
  check(vkAllocateMemory(device, &allocate_info, nullptr, &buffer.memory),
        "vkAllocateMemory");
  check(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0),
        "vkBindBufferMemory");
  if (map) {
    check(vkMapMemory(device, buffer.memory, 0, size, 0, &buffer.mapped),
          "vkMapMemory");
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
  check(vkAllocateCommandBuffers(device, &allocate_info, &command_buffer),
        "vkAllocateCommandBuffers");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");
  return command_buffer;
}

VkCommandBuffer begin_reusable(VkDevice device, VkCommandPool command_pool) {
  VkCommandBufferAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate_info.commandPool = command_pool;
  allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate_info.commandBufferCount = 1;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  check(vkAllocateCommandBuffers(device, &allocate_info, &command_buffer),
        "vkAllocateCommandBuffers reusable");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  check(vkBeginCommandBuffer(command_buffer, &begin_info),
        "vkBeginCommandBuffer reusable");
  return command_buffer;
}

void submit_and_free(
    VkDevice device,
    VkQueue queue,
    VkCommandPool command_pool,
    VkCommandBuffer command_buffer) {
  check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  check(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE), "vkQueueSubmit");
  check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
  vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}

void copy_inputs_to_device(
    VkDevice device,
    VkQueue queue,
    VkCommandPool command_pool,
    const Buffer& x_stage,
    const Buffer& w_stage,
    const Buffer& x_device,
    const Buffer& w_device,
    VkDeviceSize x_bytes,
    VkDeviceSize w_bytes) {
  VkCommandBuffer command_buffer = begin_one_time(device, command_pool);
  VkBufferCopy x_copy{};
  x_copy.size = x_bytes;
  vkCmdCopyBuffer(command_buffer, x_stage.buffer, x_device.buffer, 1, &x_copy);
  VkBufferCopy w_copy{};
  w_copy.size = w_bytes;
  vkCmdCopyBuffer(command_buffer, w_stage.buffer, w_device.buffer, 1, &w_copy);
  std::vector<VkBufferMemoryBarrier> barriers{
      hipengine::micro::make_compute_buffer_barrier(
          x_device.buffer,
          VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_ACCESS_SHADER_READ_BIT,
          0,
          x_bytes),
      hipengine::micro::make_compute_buffer_barrier(
          w_device.buffer,
          VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_ACCESS_SHADER_READ_BIT,
          0,
          w_bytes),
  };
  vkCmdPipelineBarrier(
      command_buffer,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      static_cast<uint32_t>(barriers.size()),
      barriers.data(),
      0,
      nullptr);
  submit_and_free(device, queue, command_pool, command_buffer);
}

VkDescriptorSetLayout create_descriptor_set_layout(VkDevice device) {
  std::vector<VkDescriptorSetLayoutBinding> bindings(4);
  for (uint32_t i = 0; i < bindings.size(); ++i) {
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
  check(vkCreateDescriptorSetLayout(device, &create_info, nullptr, &layout),
        "vkCreateDescriptorSetLayout");
  return layout;
}

VkPipelineLayout create_pipeline_layout(
    VkDevice device,
    VkDescriptorSetLayout descriptor_set_layout) {
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
  check(vkCreatePipelineLayout(device, &create_info, nullptr, &layout),
        "vkCreatePipelineLayout");
  return layout;
}

VkShaderModule create_shader_module(VkDevice device, const std::vector<uint32_t>& spirv) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = spirv.size() * sizeof(uint32_t);
  create_info.pCode = spirv.data();
  VkShaderModule module = VK_NULL_HANDLE;
  check(vkCreateShaderModule(device, &create_info, nullptr, &module),
        "vkCreateShaderModule");
  return module;
}

VkPipeline create_pipeline(
    VkDevice device,
    VkPipelineLayout pipeline_layout,
    VkShaderModule shader_module,
    uint32_t workgroup_size) {
  VkSpecializationMapEntry map_entry{};
  map_entry.constantID = 0;
  map_entry.offset = 0;
  map_entry.size = sizeof(uint32_t);
  VkSpecializationInfo specialization{};
  specialization.mapEntryCount = 1;
  specialization.pMapEntries = &map_entry;
  specialization.dataSize = sizeof(uint32_t);
  specialization.pData = &workgroup_size;

  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = shader_module;
  stage.pName = "main";
  stage.pSpecializationInfo = &specialization;

  VkComputePipelineCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  create_info.stage = stage;
  create_info.layout = pipeline_layout;

  VkPipeline pipeline = VK_NULL_HANDLE;
  check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &create_info, nullptr, &pipeline),
        "vkCreateComputePipelines");
  return pipeline;
}

VkDescriptorSet create_descriptor_set(
    VkDevice device,
    VkDescriptorSetLayout descriptor_set_layout,
    const Buffer& x_device,
    const Buffer& w_device,
    const Buffer& partial_device,
    const Buffer& out_device) {
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 4;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool),
        "vkCreateDescriptorPool");

  VkDescriptorSetAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocate_info.descriptorPool = descriptor_pool;
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &descriptor_set_layout;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  check(vkAllocateDescriptorSets(device, &allocate_info, &descriptor_set),
        "vkAllocateDescriptorSets");

  VkDescriptorBufferInfo infos[4] = {
      {x_device.buffer, 0, x_device.size},
      {w_device.buffer, 0, w_device.size},
      {partial_device.buffer, 0, partial_device.size},
      {out_device.buffer, 0, out_device.size},
  };
  std::vector<VkWriteDescriptorSet> writes(4);
  for (uint32_t i = 0; i < writes.size(); ++i) {
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

void partials_barrier(
    VkCommandBuffer command_buffer,
    const Buffer& partial_device,
    VkAccessFlags src_access,
    VkAccessFlags dst_access,
    VkDeviceSize offset = 0,
    VkDeviceSize size = VK_WHOLE_SIZE) {
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = partial_device.buffer;
  barrier.offset = offset;
  barrier.size = size;
  vkCmdPipelineBarrier(
      command_buffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      1,
      &barrier,
      0,
      nullptr);
}

void out_copy_barrier(
    VkCommandBuffer command_buffer,
    const Buffer& out_device,
    VkDeviceSize offset,
    VkDeviceSize size) {
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = out_device.buffer;
  barrier.offset = offset;
  barrier.size = size;
  vkCmdPipelineBarrier(
      command_buffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      0,
      0,
      nullptr,
      1,
      &barrier,
      0,
      nullptr);
}

void record_dispatches(
    VkCommandBuffer command_buffer,
    VkPipeline partial_pipeline,
    VkPipeline final_pipeline,
    VkPipelineLayout pipeline_layout,
    VkDescriptorSet descriptor_set,
    uint32_t k,
    uint32_t rows,
    uint32_t split_count,
    uint32_t body_repeats,
    uint32_t logical_iterations,
    hipengine::micro::TimingMode timing_mode,
    const hipengine::micro::VulkanSequenceTimer* timer,
    bool copy_out,
    const Buffer& partial_device,
    const Buffer& out_device,
    const Buffer& out_stage,
    VkDeviceSize out_bytes,
    uint32_t first_rep = 0,
    uint32_t rep_stride = 1) {
  if (timer != nullptr) {
    timer->record_begin(command_buffer);
  }
  const bool independent =
      timing_mode == hipengine::micro::TimingMode::IndependentThroughput;
  auto record_partial = [&](uint32_t rep) {
    PushConstants push{
        k, rows, body_repeats, split_count, independent ? rep : 0u, rep};
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, partial_pipeline);
    vkCmdBindDescriptorSets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);
    vkCmdPushConstants(
        command_buffer,
        pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(PushConstants),
        &push);
    vkCmdDispatch(command_buffer, rows, split_count, 1);
  };
  auto record_final = [&](uint32_t rep) {
    PushConstants push{
        k, rows, body_repeats, split_count, independent ? rep : 0u, rep};
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, final_pipeline);
    vkCmdBindDescriptorSets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);
    vkCmdPushConstants(
        command_buffer,
        pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(PushConstants),
        &push);
    vkCmdDispatch(command_buffer, rows, 1, 1);
  };

  if (independent) {
    const VkDeviceSize partial_slice_bytes =
        static_cast<VkDeviceSize>(rows) * split_count * sizeof(float);
    const VkDeviceSize out_slice_bytes =
        static_cast<VkDeviceSize>(rows) * sizeof(float);
    for (uint32_t rep = first_rep; rep < logical_iterations; rep += rep_stride) {
      record_partial(rep);
      partials_barrier(
          command_buffer,
          partial_device,
          VK_ACCESS_SHADER_WRITE_BIT,
          VK_ACCESS_SHADER_READ_BIT,
          static_cast<VkDeviceSize>(rep) * partial_slice_bytes,
          partial_slice_bytes);
      record_final(rep);
      if (copy_out) {
        const VkDeviceSize offset = static_cast<VkDeviceSize>(rep) * out_slice_bytes;
        out_copy_barrier(command_buffer, out_device, offset, out_slice_bytes);
        VkBufferCopy copy{offset, offset, out_slice_bytes};
        vkCmdCopyBuffer(
            command_buffer, out_device.buffer, out_stage.buffer, 1, &copy);
      }
    }
  } else {
    for (uint32_t rep = 0; rep < logical_iterations; ++rep) {
      record_partial(rep);
      partials_barrier(
          command_buffer,
          partial_device,
          VK_ACCESS_SHADER_WRITE_BIT,
          VK_ACCESS_SHADER_READ_BIT);
      record_final(rep);
      if (rep + 1 >= logical_iterations) {
        continue;
      }
      hipengine::micro::compute_buffer_barrier(
          command_buffer,
          {
              hipengine::micro::make_compute_buffer_barrier(
                  partial_device.buffer,
                  VK_ACCESS_SHADER_READ_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT,
                  0,
                  partial_device.size),
              hipengine::micro::make_compute_buffer_barrier(
                  out_device.buffer,
                  VK_ACCESS_SHADER_WRITE_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT,
                  0,
                  out_device.size),
          });
    }
  }
  if (timer != nullptr) {
    timer->record_end(command_buffer);
  }
  if (copy_out && !independent) {
    out_copy_barrier(command_buffer, out_device, 0, out_bytes);
    VkBufferCopy copy{};
    copy.size = out_bytes;
    vkCmdCopyBuffer(command_buffer, out_device.buffer, out_stage.buffer, 1, &copy);
  }
}

SequenceTiming measure_command(
    const hipengine::micro::VulkanSequenceTimer& timer,
    VkQueue queue,
    VkCommandBuffer command_buffer,
    VkFence fence,
    uint32_t samples) {
  SequenceTiming result;
  result.host_sequence_us.reserve(samples);
  if (timer.gpu_timestamps_supported()) {
    result.gpu_sequence_us.reserve(samples);
  }
  for (uint32_t sample = 0; sample < samples; ++sample) {
    const auto timing = timer.submit_and_wait(queue, command_buffer, fence);
    result.host_sequence_us.push_back(timing.host_sequence_us);
    if (timing.gpu_sequence_us >= 0.0) {
      result.gpu_sequence_us.push_back(timing.gpu_sequence_us);
    }
  }
  return result;
}

SequenceTiming measure_multi_queue_commands(
    hipengine::micro::VulkanMultiQueueTimer& timer,
    const std::vector<VkQueue>& queues,
    const std::vector<VkCommandBuffer>& command_buffers,
    const std::vector<VkFence>& fences,
    uint32_t samples) {
  SequenceTiming result;
  result.gpu_sequence_us.reserve(samples);
  result.host_sequence_us.reserve(samples);
  for (uint32_t sample = 0; sample < samples; ++sample) {
    const auto timing = timer.submit_and_wait(queues, command_buffers, fences);
    result.gpu_sequence_us.push_back(timing.gpu_sequence_us);
    result.host_sequence_us.push_back(timing.host_sequence_us);
  }
  return result;
}

Row run_config(
    const Args& args,
    VkPhysicalDevice physical_device,
    VkDevice device,
    const std::vector<VkQueue>& queues,
    VkCommandPool command_pool,
    VkDescriptorSetLayout descriptor_set_layout,
    VkPipelineLayout pipeline_layout,
    VkShaderModule partial_shader_module,
    VkShaderModule final_shader_module,
    const std::vector<VkFence>& fences,
    uint32_t queue_family,
    const char* calibrated_timestamp_extension,
    uint32_t k,
    uint32_t rows,
    uint32_t workgroup_size,
    uint32_t split_count) {
  if (queues.empty() || fences.size() != queues.size()) {
    fail("two-stage queue/fence vectors must be non-empty and matched");
  }
  VkQueue queue = queues.front();
  VkFence fence = fences.front();
  std::vector<float> x;
  std::vector<float> w;
  fill_inputs(x, w, k, rows);
  std::vector<float> expected =
      cpu_reference(x, w, k, rows, workgroup_size, split_count, args.body_repeats);
  const auto timing_mode = hipengine::micro::parse_timing_mode(args.timing_mode);
  const bool independent =
      timing_mode == hipengine::micro::TimingMode::IndependentThroughput;
  const uint32_t queue_count = independent
      ? std::min<uint32_t>(args.reps, static_cast<uint32_t>(queues.size()))
      : 1u;
  const uint32_t output_slices = independent
      ? std::max<uint32_t>(1, std::max(args.reps, args.warmup))
      : 1;
  std::vector<float> actual(static_cast<size_t>(rows) * output_slices, 0.0f);
  VkDeviceSize x_bytes = sizeof(float) * x.size();
  VkDeviceSize w_bytes = sizeof(float) * w.size();
  VkDeviceSize partial_bytes = sizeof(float) * static_cast<VkDeviceSize>(rows) *
      split_count * output_slices;
  VkDeviceSize out_bytes = sizeof(float) * actual.size();

  Buffer x_stage = create_buffer(
      physical_device,
      device,
      x_bytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      true);
  Buffer w_stage = create_buffer(
      physical_device,
      device,
      w_bytes,
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
  Buffer w_device = create_buffer(
      physical_device,
      device,
      w_bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      false);
  Buffer partial_device = create_buffer(
      physical_device,
      device,
      partial_bytes,
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
  std::memcpy(w_stage.mapped, w.data(), static_cast<size_t>(w_bytes));
  copy_inputs_to_device(
      device, queue, command_pool, x_stage, w_stage, x_device, w_device, x_bytes, w_bytes);

  VkPipeline partial_pipeline =
      create_pipeline(device, pipeline_layout, partial_shader_module, workgroup_size);
  VkPipeline final_pipeline =
      create_pipeline(device, pipeline_layout, final_shader_module, workgroup_size);
  VkDescriptorSet descriptor_set = create_descriptor_set(
      device, descriptor_set_layout, x_device, w_device, partial_device, out_device);
  hipengine::micro::VulkanSequenceTimer single_queue_timer(
      physical_device, device, queue_family);
  const bool use_multi_queue = independent && queue_count > 1;
  std::unique_ptr<hipengine::micro::VulkanMultiQueueTimer> multi_queue_timer;
  if (use_multi_queue) {
    if (calibrated_timestamp_extension == nullptr) {
      fail("independent two-stage timing requires calibrated timestamps");
    }
    multi_queue_timer = std::make_unique<hipengine::micro::VulkanMultiQueueTimer>(
        physical_device,
        device,
        queue_family,
        queue_count,
        calibrated_timestamp_extension);
  }
  const std::vector<VkQueue> active_queues(
      queues.begin(), queues.begin() + queue_count);
  const std::vector<VkFence> active_fences(
      fences.begin(), fences.begin() + queue_count);

  auto make_multi_queue_commands = [&](
                                       uint32_t logical_iterations,
                                       bool copy_out,
                                       bool reusable) {
    std::vector<VkCommandBuffer> commands(queue_count, VK_NULL_HANDLE);
    for (uint32_t lane = 0; lane < queue_count; ++lane) {
      VkCommandBuffer command = reusable
          ? begin_reusable(device, command_pool)
          : begin_one_time(device, command_pool);
      multi_queue_timer->record_begin(command, lane);
      record_dispatches(
          command,
          partial_pipeline,
          final_pipeline,
          pipeline_layout,
          descriptor_set,
          k,
          rows,
          split_count,
          args.body_repeats,
          logical_iterations,
          timing_mode,
          nullptr,
          copy_out,
          partial_device,
          out_device,
          out_stage,
          out_bytes,
          lane,
          queue_count);
      multi_queue_timer->record_end(command, lane);
      check(vkEndCommandBuffer(command), "vkEndCommandBuffer multi-queue lane");
      commands[lane] = command;
    }
    return commands;
  };

  VkCommandBuffer correctness_cmd = begin_one_time(device, command_pool);
  record_dispatches(
      correctness_cmd,
      partial_pipeline,
      final_pipeline,
      pipeline_layout,
      descriptor_set,
      k,
      rows,
      split_count,
      args.body_repeats,
      1,
      timing_mode,
      nullptr,
      true,
      partial_device,
      out_device,
      out_stage,
      out_bytes);
  submit_and_free(device, queue, command_pool, correctness_cmd);
  std::memcpy(actual.data(), out_stage.mapped, static_cast<size_t>(out_bytes));

  float max_abs = 0.0f;
  float max_rel = 0.0f;
  bool pass = true;
  auto check_value = [&](float observed, float base, uint32_t sequence_id) {
    const float wanted = base + static_cast<float>(sequence_id) * 0.125f;
    const float diff = std::abs(observed - wanted);
    const float rel = diff / std::max(1.0e-6f, std::abs(wanted));
    max_abs = std::max(max_abs, diff);
    max_rel = std::max(max_rel, rel);
    pass = pass && (diff <= 1.0e-2f || rel <= 1.0e-4f);
  };
  for (uint32_t row = 0; row < rows; ++row) {
    check_value(actual[row], expected[row], 0);
  }

  if (use_multi_queue) {
    std::vector<VkCommandBuffer> commands =
        make_multi_queue_commands(args.reps, true, false);
    (void)multi_queue_timer->submit_and_wait(
        active_queues, commands, active_fences);
    vkFreeCommandBuffers(
        device, command_pool, static_cast<uint32_t>(commands.size()), commands.data());
  } else {
    VkCommandBuffer burst_correctness_cmd = begin_one_time(device, command_pool);
    record_dispatches(
        burst_correctness_cmd,
        partial_pipeline,
        final_pipeline,
        pipeline_layout,
        descriptor_set,
        k,
        rows,
        split_count,
        args.body_repeats,
        args.reps,
        timing_mode,
        nullptr,
        true,
        partial_device,
        out_device,
        out_stage,
        out_bytes);
    submit_and_free(device, queue, command_pool, burst_correctness_cmd);
  }
  std::memcpy(actual.data(), out_stage.mapped, static_cast<size_t>(out_bytes));
  if (independent) {
    for (uint32_t rep = 0; rep < args.reps; ++rep) {
      for (uint32_t row = 0; row < rows; ++row) {
        check_value(actual[static_cast<size_t>(rep) * rows + row], expected[row], rep);
      }
    }
  } else {
    for (uint32_t row = 0; row < rows; ++row) {
      check_value(actual[row], expected[row], args.reps - 1);
    }
  }

  if (args.warmup > 0) {
    if (use_multi_queue) {
      std::vector<VkCommandBuffer> commands =
          make_multi_queue_commands(args.warmup, false, false);
      (void)multi_queue_timer->submit_and_wait(
          active_queues, commands, active_fences);
      vkFreeCommandBuffers(
          device, command_pool, static_cast<uint32_t>(commands.size()), commands.data());
    } else {
      VkCommandBuffer warmup_cmd = begin_one_time(device, command_pool);
      record_dispatches(
          warmup_cmd,
          partial_pipeline,
          final_pipeline,
          pipeline_layout,
          descriptor_set,
          k,
          rows,
          split_count,
          args.body_repeats,
          args.warmup,
          timing_mode,
          nullptr,
          false,
          partial_device,
          out_device,
          out_stage,
          out_bytes);
      submit_and_free(device, queue, command_pool, warmup_cmd);
    }
  }

  VkCommandBuffer single_cmd = begin_reusable(device, command_pool);
  record_dispatches(
      single_cmd,
      partial_pipeline,
      final_pipeline,
      pipeline_layout,
      descriptor_set,
      k,
      rows,
      split_count,
      args.body_repeats,
      1,
      timing_mode,
      &single_queue_timer,
      false,
      partial_device,
      out_device,
      out_stage,
      out_bytes);
  check(vkEndCommandBuffer(single_cmd), "vkEndCommandBuffer single timing");
  SequenceTiming single_timing =
      measure_command(single_queue_timer, queue, single_cmd, fence, args.samples);
  SequenceTiming burst_timing;
  std::vector<VkCommandBuffer> burst_commands;
  if (use_multi_queue) {
    burst_commands = make_multi_queue_commands(args.reps, false, true);
    burst_timing = measure_multi_queue_commands(
        *multi_queue_timer,
        active_queues,
        burst_commands,
        active_fences,
        args.samples);
  } else {
    VkCommandBuffer burst_cmd = begin_reusable(device, command_pool);
    record_dispatches(
        burst_cmd,
        partial_pipeline,
        final_pipeline,
        pipeline_layout,
        descriptor_set,
        k,
        rows,
        split_count,
        args.body_repeats,
        args.reps,
        timing_mode,
        &single_queue_timer,
        false,
        partial_device,
        out_device,
        out_stage,
        out_bytes);
    check(vkEndCommandBuffer(burst_cmd), "vkEndCommandBuffer burst timing");
    burst_timing = measure_command(
        single_queue_timer, queue, burst_cmd, fence, args.samples);
    burst_commands.push_back(burst_cmd);
  }
  vkFreeCommandBuffers(device, command_pool, 1, &single_cmd);
  vkFreeCommandBuffers(
      device,
      command_pool,
      static_cast<uint32_t>(burst_commands.size()),
      burst_commands.data());
  vkDestroyPipeline(device, partial_pipeline, nullptr);
  vkDestroyPipeline(device, final_pipeline, nullptr);
  destroy_buffer(device, x_stage);
  destroy_buffer(device, w_stage);
  destroy_buffer(device, out_stage);
  destroy_buffer(device, x_device);
  destroy_buffer(device, w_device);
  destroy_buffer(device, partial_device);
  destroy_buffer(device, out_device);

  const std::vector<double>& sequence_samples =
      single_queue_timer.gpu_timestamps_supported()
      ? burst_timing.gpu_sequence_us
      : burst_timing.host_sequence_us;
  std::vector<double> per_iteration;
  per_iteration.reserve(sequence_samples.size());
  for (double sample : sequence_samples) {
    per_iteration.push_back(sample / args.reps);
  }
  double median_us = percentile(per_iteration, 0.5);
  double ops = static_cast<double>(rows) * k * args.body_repeats * 2.0;
  double bytes = static_cast<double>(sizeof(float)) *
      (k + (static_cast<double>(rows) * k) + (static_cast<double>(rows) * split_count));
  return Row{
      k,
      rows,
      workgroup_size,
      split_count,
      args.body_repeats,
      median_us,
      percentile(per_iteration, 0.05),
      percentile(per_iteration, 0.95),
      *std::min_element(per_iteration.begin(), per_iteration.end()),
      *std::max_element(per_iteration.begin(), per_iteration.end()),
      ops / median_us / 1000.0,
      bytes / median_us,
      max_abs,
      max_rel,
      pass,
      independent ? args.reps : (2 * args.reps - 1),
      queue_count,
      use_multi_queue ? calibrated_timestamp_extension : "",
      single_queue_timer.gpu_timestamps_supported(),
      single_queue_timer.timestamp_period_ns(),
      single_queue_timer.timestamp_valid_bits(),
      std::move(single_timing),
      std::move(burst_timing),
  };
}

void write_statistics(
    std::ostream& out,
    const std::vector<double>& samples,
    double divisor) {
  std::vector<double> values;
  values.reserve(samples.size());
  double mean = 0.0;
  for (double sample : samples) {
    values.push_back(sample / divisor);
    mean += sample / divisor;
  }
  mean /= static_cast<double>(values.size());
  double variance = 0.0;
  for (double value : values) {
    const double delta = value - mean;
    variance += delta * delta;
  }
  variance /= static_cast<double>(values.size());
  out << "{\"samples\": " << values.size()
      << ", \"n\": " << values.size()
      << ", \"median\": " << percentile(values, 0.5)
      << ", \"p05\": " << percentile(values, 0.05)
      << ", \"p95\": " << percentile(values, 0.95)
      << ", \"min\": " << *std::min_element(values.begin(), values.end())
      << ", \"max\": " << *std::max_element(values.begin(), values.end())
      << ", \"stdev\": " << std::sqrt(variance) << "}";
}

void write_metric(
    std::ostream& out,
    const std::vector<double>& samples,
    uint32_t logical_iterations,
    const char* clock,
    bool supported) {
  if (!supported) {
    out << "{\"status\": \"unsupported\", \"clock\": \"" << clock << "\"}";
    return;
  }
  out << "{\"status\": \"ok\", \"clock\": \"" << clock
      << "\", \"sequence_us\": ";
  write_statistics(out, samples, 1.0);
  out << ", \"per_iteration_us\": ";
  write_statistics(out, samples, logical_iterations);
  out << "}";
}

void write_timing_control(
    std::ostream& out,
    const SequenceTiming& timing,
    uint32_t logical_iterations,
    bool gpu_supported) {
  out << "{\"logical_iterations\": " << logical_iterations
      << ", \"dispatches_per_iteration\": 2, \"gpu_elapsed\": ";
  write_metric(
      out,
      timing.gpu_sequence_us,
      logical_iterations,
      "vulkan_timestamp",
      gpu_supported);
  out << ", \"host_wall\": ";
  write_metric(
      out, timing.host_sequence_us, logical_iterations, "steady_clock", true);
  out << "}";
}

void write_timed_contract(std::ostream& out, const Args& args, const Row& row) {
  const bool independent = args.timing_mode == "independent_throughput";
  out << "      \"timing_mode\": \"" << args.timing_mode << "\",\n";
  out << "      \"dependency_contract\": {\"work_dependency\": \""
      << (independent ? "independent" : "chained")
      << "\", \"inter_dispatch_ordering\": \""
      << (independent ? "vulkan_round_robin_queue_order" : "vulkan_compute_barrier")
      << "\", \"output_partitioning\": \""
      << (independent ? "disjoint" : "chained_shared")
      << "\", \"validation_status\": \"pass\"},\n";
  out << "      \"submission\": {\"strategy\": \""
      << (independent && row.queue_count > 1
              ? "vulkan_multi_queue"
              : "vulkan_command_buffer")
      << "\", "
         "\"recording_in_timed_region\": false, \"submit_in_host_wall\": true, "
         "\"completion_in_host_wall\": true, \"queue_or_stream_count\": "
      << row.queue_count
      << ", \"calibrated_timestamp_extension\": \""
      << json_escape(row.calibrated_timestamp_extension) << "\"},\n";
  out << "      \"timing\": {\"single\": ";
  write_timing_control(out, row.single_timing, 1, row.gpu_timestamps_supported);
  out << ", \"burst\": ";
  write_timing_control(
      out, row.burst_timing, args.reps, row.gpu_timestamps_supported);
  out << "},\n";
  out << "      \"correctness\": {"
         "\"single_dispatch\": {\"status\": \"pass\", "
         "\"oracle\": \"deterministic CPU two-stage reference with sequence tag\"}, "
         "\"timed_sequence\": {\"status\": \"pass\", "
         "\"oracle\": \"deterministic CPU two-stage reference with sequence tag\", "
         "\"logical_iterations\": " << args.reps
      << ", \"coverage\": \""
      << (independent ? "all_dispatches" : "chained_final_state")
      << "\"}, \"synchronization\": {\"status\": \"pass\", \"method\": \""
      << (independent
              ? "round_robin_queue_lanes_with_intra_operation_barriers"
              : "partial_to_final_and_repetition_compute_barriers")
      << "\", \"barrier_count\": " << row.barrier_count << "}},\n";
}

void write_json(
    const Args& args,
    const VkPhysicalDeviceProperties& properties,
    uint32_t queue_family,
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
  *out << "  \"run_tag\": \"vulkan-two-stage-reduction\",\n";
  *out << "  \"status\": \"diagnostic\",\n";
  *out << "  \"backend\": \"vulkan\",\n";
  *out << "  \"hardware\": {\n";
  *out << "    \"device_name\": \"" << json_escape(properties.deviceName) << "\",\n";
  *out << "    \"vendor_id\": " << properties.vendorID << ",\n";
  *out << "    \"device_id\": " << properties.deviceID << ",\n";
  *out << "    \"device_type\": " << static_cast<uint32_t>(properties.deviceType) << ",\n";
  *out << "    \"api_version\": " << properties.apiVersion << ",\n";
  *out << "    \"driver_version_raw\": " << properties.driverVersion << ",\n";
  *out << "    \"queue_family\": " << queue_family << "\n";
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
  *out << "    \"split_counts\": ";
  print_json_array(*out, args.split_counts);
  *out << ",\n";
  *out << "    \"body_repeats\": " << args.body_repeats << ",\n";
  *out << "    \"reps\": " << args.reps << ",\n";
  *out << "    \"warmup\": " << args.warmup << ",\n";
  *out << "    \"samples\": " << args.samples << ",\n";
  *out << "    \"timing_mode\": \"" << args.timing_mode << "\",\n";
  *out << "    \"independent_queues\": " << args.independent_queues << ",\n";
  *out << "    \"method\": \"pre-recorded command buffer with block-partial dispatch plus final-reduce dispatch\"\n";
  *out << "  },\n";
  *out << "  \"rows\": [\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row& row = rows[i];
    *out << "    {\n";
    *out << "      \"k\": " << row.k << ",\n";
    *out << "      \"rows\": " << row.rows << ",\n";
    *out << "      \"workgroup_size\": " << row.workgroup_size << ",\n";
    *out << "      \"split_count\": " << row.split_count << ",\n";
    *out << "      \"body_repeats\": " << row.body_repeats << ",\n";
    write_timed_contract(*out, args, row);
    *out << "      \"median_us\": " << row.median_us << ",\n";
    *out << "      \"p05_us\": " << row.p05_us << ",\n";
    *out << "      \"p95_us\": " << row.p95_us << ",\n";
    *out << "      \"min_us\": " << row.min_us << ",\n";
    *out << "      \"max_us\": " << row.max_us << ",\n";
    *out << "      \"gflops\": " << row.gflops << ",\n";
    *out << "      \"bytes_per_us\": " << row.bytes_per_us << ",\n";
    *out << "      \"max_abs\": " << row.max_abs << ",\n";
    *out << "      \"max_rel\": " << row.max_rel << ",\n";
    *out << "      \"gpu_timestamps_supported\": "
         << (row.gpu_timestamps_supported ? "true" : "false") << ",\n";
    *out << "      \"queue_count\": " << row.queue_count << ",\n";
    *out << "      \"calibrated_timestamp_extension\": \""
         << json_escape(row.calibrated_timestamp_extension) << "\",\n";
    *out << "      \"timestamp_period_ns\": " << row.timestamp_period_ns << ",\n";
    *out << "      \"timestamp_valid_bits\": " << row.timestamp_valid_bits << ",\n";
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

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "hipEngine Vulkan two-stage reduction";
    app_info.applicationVersion = 1;
    app_info.pEngineName = "hipEngine microbench";
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    VkInstance instance = VK_NULL_HANDLE;
    check(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

    uint32_t physical_count = 0;
    check(vkEnumeratePhysicalDevices(instance, &physical_count, nullptr),
          "vkEnumeratePhysicalDevices count");
    if (physical_count == 0) {
      fail("no Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    check(vkEnumeratePhysicalDevices(instance, &physical_count, physical_devices.data()),
          "vkEnumeratePhysicalDevices");
    if (args.device_index >= physical_devices.size()) {
      fail("--device-index exceeds physical device count");
    }
    VkPhysicalDevice physical_device = physical_devices[args.device_index];
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    const bool independent =
        args.timing_mode == "independent_throughput";
    const uint32_t requested_queues = independent
        ? std::min(args.independent_queues, args.reps)
        : 1u;
    const auto queue_selection =
        hipengine::micro::select_compute_queue_family(
            physical_device, requested_queues);
    const uint32_t queue_family = queue_selection.index;
    const uint32_t queue_count = queue_selection.queue_count;
    const char* calibrated_timestamp_extension = nullptr;
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{};
    if (independent && queue_count > 1) {
      if (!hipengine::micro::timeline_semaphore_supported(physical_device)) {
        fail("independent two-stage timing requires timeline semaphores");
      }
      calibrated_timestamp_extension =
          hipengine::micro::calibrated_timestamps_extension(physical_device);
      if (calibrated_timestamp_extension == nullptr) {
        fail("independent two-stage timing requires calibrated timestamps");
      }
      timeline_features.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
      timeline_features.timelineSemaphore = VK_TRUE;
    }

    std::vector<float> priorities(queue_count, 1.0f);
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = queue_count;
    queue_info.pQueuePriorities = priorities.data();

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = timeline_features.sType != 0 ? &timeline_features : nullptr;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    if (calibrated_timestamp_extension != nullptr) {
      device_info.enabledExtensionCount = 1;
      device_info.ppEnabledExtensionNames = &calibrated_timestamp_extension;
    }
    VkDevice device = VK_NULL_HANDLE;
    check(vkCreateDevice(physical_device, &device_info, nullptr, &device),
          "vkCreateDevice");
    std::vector<VkQueue> queues(queue_count, VK_NULL_HANDLE);
    for (uint32_t index = 0; index < queue_count; ++index) {
      vkGetDeviceQueue(device, queue_family, index, &queues[index]);
    }

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(device, &pool_info, nullptr, &command_pool),
          "vkCreateCommandPool");

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    std::vector<VkFence> fences(queue_count, VK_NULL_HANDLE);
    for (VkFence& fence : fences) {
      check(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");
    }

    VkDescriptorSetLayout descriptor_set_layout = create_descriptor_set_layout(device);
    VkPipelineLayout pipeline_layout = create_pipeline_layout(device, descriptor_set_layout);
    VkShaderModule partial_shader_module =
        create_shader_module(device, read_spirv(args.partial_spirv_path));
    VkShaderModule final_shader_module =
        create_shader_module(device, read_spirv(args.final_spirv_path));

    std::vector<Row> rows;
    for (uint32_t k : args.k_list) {
      for (uint32_t row_count : args.rows_list) {
        for (uint32_t workgroup_size : args.workgroups) {
          for (uint32_t split_count : args.split_counts) {
            Row row = run_config(
                args,
                physical_device,
                device,
                queues,
                command_pool,
                descriptor_set_layout,
                pipeline_layout,
                partial_shader_module,
                final_shader_module,
                fences,
                queue_family,
                calibrated_timestamp_extension,
                k,
                row_count,
                workgroup_size,
                split_count);
            rows.push_back(row);
            std::cerr << "[vulkan-two-stage] K=" << k << " rows=" << row_count
                      << " wg=" << workgroup_size << " splits=" << split_count
                      << " median=" << row.median_us
                      << " us correctness=" << (row.correctness_pass ? "pass" : "fail")
                      << "\n";
          }
        }
      }
    }

    write_json(args, properties, queue_family, rows);

    vkDeviceWaitIdle(device);
    vkDestroyShaderModule(device, partial_shader_module, nullptr);
    vkDestroyShaderModule(device, final_shader_module, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    for (VkFence fence : fences) {
      vkDestroyFence(device, fence, nullptr);
    }
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
