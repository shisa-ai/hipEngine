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
#include <utility>
#include <vector>

#include "micro_timing_vulkan.hpp"

#ifndef HIPENGINE_DOT_MODE
#define HIPENGINE_DOT_MODE 0
#endif

#ifndef HIPENGINE_DOT_GROUPS
#define HIPENGINE_DOT_GROUPS 16
#endif

#ifndef HIPENGINE_BLOCK_SIZE
#define HIPENGINE_BLOCK_SIZE 256
#endif

namespace {

constexpr uint32_t kBlockSize = HIPENGINE_BLOCK_SIZE;

struct Args {
  std::string spirv_path;
  std::string json_path;
  uint32_t n = 32768;
  uint32_t body_iters = 128;
  uint32_t reps = 20;
  uint32_t warmup = 5;
  uint32_t samples = 7;
  std::string timing_mode = "serial_latency";
  uint32_t device_index = 0;
};

struct PushConstants {
  uint32_t n;
  uint32_t body_iters;
  uint32_t data_mask;
  uint32_t output_offset;
  uint32_t sequence_id;
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
  bool gpu_timestamps_supported;
  std::vector<double> single_gpu_samples_us;
  std::vector<double> single_host_samples_us;
  std::vector<double> burst_gpu_samples_us;
  std::vector<double> burst_host_samples_us;
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
    if (flag == "--spirv") {
      args.spirv_path = require_value(i, argc, argv, flag);
    } else if (flag == "--json") {
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
    } else if (flag == "--device-index") {
      args.device_index = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else {
      fail("unknown argument: " + flag);
    }
  }
  if (args.spirv_path.empty()) {
    fail("--spirv is required");
  }
  if (args.n == 0 || args.body_iters == 0 || args.reps == 0 || args.samples == 0) {
    fail("--n, --body-iters, --reps, and --samples must be positive");
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

uint32_t hash_u32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

uint8_t activation_byte(uint32_t i, uint32_t lane) {
  uint32_t bits = hash_u32(i * 1664525u + lane * 1013904223u + 0x9e3779b9u);
  return static_cast<uint8_t>(bits & 0xffu);
}

uint8_t weight_byte(uint32_t i, uint32_t lane) {
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

uint32_t pack_weight_word(uint32_t i) {
  uint32_t packed = 0;
  for (uint32_t lane = 0; lane < 4; ++lane) {
    packed |= static_cast<uint32_t>(weight_byte(i, lane)) << (lane * 8);
  }
  return packed;
}

uint32_t pack_activation_word(uint32_t i) {
  uint32_t packed = 0;
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

int dot_q8_s8_s8(uint32_t a, uint32_t b, int c) {
  int acc = c;
  for (uint32_t lane = 0; lane < 4; ++lane) {
    int av = static_cast<int>(static_cast<int8_t>((a >> (lane * 8)) & 0xffu));
    int bv = static_cast<int>(static_cast<int8_t>((b >> (lane * 8)) & 0xffu));
    acc += av * bv;
  }
  return acc;
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

int scalar_q4_dequant(uint32_t a, uint32_t b, int c) {
  int acc = c;
  for (uint32_t lane = 0; lane < 4; ++lane) {
    int q = static_cast<int>((a >> (lane * 8)) & 0x0fu) - 8;
    int x = static_cast<int>(static_cast<int8_t>((b >> (lane * 8)) & 0xffu));
    acc += q * x;
  }
  return acc;
}

int run_value(
    const uint32_t* weights,
    const uint32_t* activations,
    uint32_t idx,
    uint32_t n,
    uint32_t body_iters,
    uint32_t data_mask) {
  int sum = 0;
  for (uint32_t iter = 0; iter < body_iters; ++iter) {
    uint32_t base = ((iter * n + idx) * HIPENGINE_DOT_GROUPS) & data_mask;
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
  if (count == 0) {
    fail("physical device has no queue families");
  }
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

VkDevice create_device_with_integer_dot(
    VkPhysicalDevice physical_device,
    uint32_t queue_family,
    const float* queue_priority) {
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
  check(vkAllocateCommandBuffers(device, &allocate_info, &command_buffer),
        "vkAllocateCommandBuffers");
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");
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
    const Buffer& weights_stage,
    const Buffer& activations_stage,
    const Buffer& weights_device,
    const Buffer& activations_device,
    VkDeviceSize weights_bytes,
    VkDeviceSize activations_bytes) {
  VkCommandBuffer command_buffer = begin_one_time(device, command_pool);
  VkBufferCopy weights_copy{};
  weights_copy.size = weights_bytes;
  vkCmdCopyBuffer(command_buffer, weights_stage.buffer, weights_device.buffer, 1, &weights_copy);
  VkBufferCopy activations_copy{};
  activations_copy.size = activations_bytes;
  vkCmdCopyBuffer(
      command_buffer, activations_stage.buffer, activations_device.buffer, 1, &activations_copy);
  submit_and_free(device, queue, command_pool, command_buffer);
}

VkDescriptorSetLayout create_descriptor_set_layout(VkDevice device) {
  std::vector<VkDescriptorSetLayoutBinding> bindings(3);
  for (uint32_t i = 0; i < 3; ++i) {
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
  check(vkCreateShaderModule(device, &create_info, nullptr, &module), "vkCreateShaderModule");
  return module;
}

VkPipeline create_pipeline(VkDevice device, VkPipelineLayout pipeline_layout, VkShaderModule shader_module) {
  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = shader_module;
  stage.pName = "main";
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
    const Buffer& weights_device,
    const Buffer& activations_device,
    const Buffer& out_device,
    VkDescriptorPool& descriptor_pool) {
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 3;
  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
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

  VkDescriptorBufferInfo infos[3] = {
      {weights_device.buffer, 0, weights_device.size},
      {activations_device.buffer, 0, activations_device.size},
      {out_device.buffer, 0, out_device.size},
  };
  std::vector<VkWriteDescriptorSet> writes(3);
  for (uint32_t i = 0; i < 3; ++i) {
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

uint32_t grid_blocks(uint32_t n) {
  return std::max<uint32_t>(1, (n + kBlockSize - 1) / kBlockSize);
}

void record_dispatches(
    VkCommandBuffer command_buffer,
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    VkDescriptorSet descriptor_set,
    const Args& args,
    uint32_t data_mask,
    uint32_t reps,
    hipengine::micro::TimingMode timing_mode,
    const hipengine::micro::VulkanSequenceTimer* timer,
    bool copy_out,
    const Buffer& out_device,
    const Buffer& out_stage,
    VkDeviceSize out_bytes) {
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      command_buffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline_layout,
      0,
      1,
      &descriptor_set,
      0,
      nullptr);
  if (timer != nullptr) {
    timer->record_begin(command_buffer);
  }
  for (uint32_t rep = 0; rep < reps; ++rep) {
    const uint32_t output_offset =
        timing_mode == hipengine::micro::TimingMode::IndependentThroughput
            ? rep * args.n
            : 0u;
    PushConstants push{args.n, args.body_iters, data_mask, output_offset, rep};
    vkCmdPushConstants(
        command_buffer,
        pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(PushConstants),
        &push);
    vkCmdDispatch(command_buffer, grid_blocks(args.n), 1, 1);
    if (timing_mode == hipengine::micro::TimingMode::SerialLatency && rep + 1 < reps) {
      hipengine::micro::compute_buffer_barrier(
          command_buffer,
          {hipengine::micro::make_compute_buffer_barrier(
              out_device.buffer,
              VK_ACCESS_SHADER_WRITE_BIT,
              VK_ACCESS_SHADER_WRITE_BIT)});
    }
  }
  if (timer != nullptr) {
    timer->record_end(command_buffer);
  }
  if (copy_out) {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = out_device.buffer;
    barrier.offset = 0;
    barrier.size = out_bytes;
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
    VkBufferCopy copy{};
    copy.size = out_bytes;
    vkCmdCopyBuffer(command_buffer, out_device.buffer, out_stage.buffer, 1, &copy);
  }
}

double percentile(std::vector<double> values, double q) {
  if (values.empty()) {
    fail("cannot compute percentile of empty values");
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
    const std::vector<double>& gpu_samples,
    const std::vector<double>& host_samples,
    bool trailing_comma) {
  out << "      \"" << name << "\": {\n";
  out << "        \"logical_iterations\": " << logical_iterations << ",\n";
  out << "        \"dispatches_per_iteration\": 1,\n";
  out << "        \"gpu_samples_us\": ";
  write_samples(out, gpu_samples);
  out << ",\n";
  out << "        \"host_samples_us\": ";
  write_samples(out, host_samples);
  out << "\n      }" << (trailing_comma ? "," : "") << "\n";
}

std::string version_string(uint32_t version) {
  std::ostringstream out;
  out << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "."
      << VK_VERSION_PATCH(version);
  return out.str();
}

Row run_config(
    const Args& args,
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkQueue queue,
    uint32_t queue_family,
    VkCommandPool command_pool,
    VkDescriptorSetLayout descriptor_set_layout,
    VkPipelineLayout pipeline_layout,
    VkShaderModule shader_module,
    VkFence fence) {
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
  VkDeviceSize weights_bytes = sizeof(uint32_t) * weights.size();
  VkDeviceSize activations_bytes = sizeof(uint32_t) * activations.size();
  VkDeviceSize out_bytes = sizeof(int32_t) * actual.size();

  Buffer weights_stage = create_buffer(
      physical_device,
      device,
      weights_bytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      true);
  Buffer activations_stage = create_buffer(
      physical_device,
      device,
      activations_bytes,
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
  Buffer weights_device = create_buffer(
      physical_device,
      device,
      weights_bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      false);
  Buffer activations_device = create_buffer(
      physical_device,
      device,
      activations_bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      false);
  Buffer out_device = create_buffer(
      physical_device,
      device,
      out_bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      false);

  std::memcpy(weights_stage.mapped, weights.data(), static_cast<size_t>(weights_bytes));
  std::memcpy(activations_stage.mapped, activations.data(), static_cast<size_t>(activations_bytes));
  copy_inputs_to_device(
      device,
      queue,
      command_pool,
      weights_stage,
      activations_stage,
      weights_device,
      activations_device,
      weights_bytes,
      activations_bytes);

  VkPipeline pipeline = create_pipeline(device, pipeline_layout, shader_module);
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set = create_descriptor_set(
      device, descriptor_set_layout, weights_device, activations_device, out_device, descriptor_pool);

  VkCommandBuffer correctness_cmd = begin_one_time(device, command_pool);
  record_dispatches(
      correctness_cmd,
      pipeline,
      pipeline_layout,
      descriptor_set,
      args,
      data_mask,
      1,
      timing_mode,
      nullptr,
      true,
      out_device,
      out_stage,
      out_bytes);
  submit_and_free(device, queue, command_pool, correctness_cmd);
  std::memcpy(actual.data(), out_stage.mapped, static_cast<size_t>(out_bytes));

  double max_abs = 0.0;
  double max_rel = 0.0;
  const uint32_t checked = std::min<uint32_t>(args.n, 64);
  auto check_slot = [&](uint32_t slot, uint32_t sequence_id) {
    for (uint32_t i = 0; i < checked; ++i) {
      int expected =
          run_value(weights.data(), activations.data(), i, args.n, args.body_iters, data_mask) +
          static_cast<int32_t>(sequence_id);
      int32_t observed = actual[static_cast<size_t>(slot) * args.n + i];
      double diff = std::abs(static_cast<double>(observed - expected));
      max_abs = std::max(max_abs, diff);
      max_rel =
          std::max(max_rel, diff / std::max(1.0, std::abs(static_cast<double>(expected))));
    }
  };
  check_slot(0, 0);
  const bool single_pass = max_abs == 0.0;

  hipengine::micro::VulkanSequenceTimer timer(physical_device, device, queue_family);
  VkCommandBuffer single_cmd = begin_one_time(device, command_pool);
  record_dispatches(
      single_cmd,
      pipeline,
      pipeline_layout,
      descriptor_set,
      args,
      data_mask,
      1,
      timing_mode,
      &timer,
      false,
      out_device,
      out_stage,
      out_bytes);
  check(vkEndCommandBuffer(single_cmd), "vkEndCommandBuffer single timing");

  VkCommandBuffer burst_cmd = begin_one_time(device, command_pool);
  record_dispatches(
      burst_cmd,
      pipeline,
      pipeline_layout,
      descriptor_set,
      args,
      data_mask,
      args.reps,
      timing_mode,
      &timer,
      false,
      out_device,
      out_stage,
      out_bytes);
  check(vkEndCommandBuffer(burst_cmd), "vkEndCommandBuffer burst timing");

  if (args.warmup > 0) {
    VkCommandBuffer warmup_cmd = begin_one_time(device, command_pool);
    record_dispatches(
        warmup_cmd,
        pipeline,
        pipeline_layout,
        descriptor_set,
        args,
        data_mask,
        args.warmup,
        timing_mode,
        nullptr,
        false,
        out_device,
        out_stage,
        out_bytes);
    submit_and_free(device, queue, command_pool, warmup_cmd);
  }
  std::vector<double> single_gpu_samples;
  std::vector<double> single_host_samples;
  for (uint32_t sample = 0; sample < args.samples; ++sample) {
    auto timing = timer.submit_and_wait(queue, single_cmd, fence);
    single_gpu_samples.push_back(timing.gpu_sequence_us);
    single_host_samples.push_back(timing.host_sequence_us);
  }
  std::vector<double> burst_gpu_samples;
  std::vector<double> burst_host_samples;
  for (uint32_t sample = 0; sample < args.samples; ++sample) {
    auto timing = timer.submit_and_wait(queue, burst_cmd, fence);
    burst_gpu_samples.push_back(timing.gpu_sequence_us);
    burst_host_samples.push_back(timing.host_sequence_us);
  }

  VkCommandBuffer burst_correctness_cmd = begin_one_time(device, command_pool);
  record_dispatches(
      burst_correctness_cmd,
      pipeline,
      pipeline_layout,
      descriptor_set,
      args,
      data_mask,
      args.reps,
      timing_mode,
      nullptr,
      true,
      out_device,
      out_stage,
      out_bytes);
  submit_and_free(device, queue, command_pool, burst_correctness_cmd);
  std::memcpy(actual.data(), out_stage.mapped, static_cast<size_t>(out_bytes));
  if (timing_mode == hipengine::micro::TimingMode::IndependentThroughput) {
    for (uint32_t slot = 0; slot < args.reps; ++slot) {
      check_slot(slot, slot);
    }
  } else {
    check_slot(0, args.reps - 1);
  }
  const bool burst_pass = max_abs == 0.0;

  vkFreeCommandBuffers(device, command_pool, 1, &single_cmd);
  vkFreeCommandBuffers(device, command_pool, 1, &burst_cmd);
  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  destroy_buffer(device, weights_stage);
  destroy_buffer(device, activations_stage);
  destroy_buffer(device, out_stage);
  destroy_buffer(device, weights_device);
  destroy_buffer(device, activations_device);
  destroy_buffer(device, out_device);

  std::vector<double> samples;
  samples.reserve(args.samples);
  const std::vector<double>& burst_source =
      timer.gpu_timestamps_supported() ? burst_gpu_samples : burst_host_samples;
  for (double sample : burst_source) {
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
      timer.gpu_timestamps_supported(),
      std::move(single_gpu_samples),
      std::move(single_host_samples),
      std::move(burst_gpu_samples),
      std::move(burst_host_samples),
  };
}

void write_json(
    const Args& args,
    const VkPhysicalDeviceProperties& properties,
    uint32_t queue_family,
    const Row& row,
    std::ostream& out) {
  out << std::setprecision(10);
  out << "{\n";
  out << "  \"run_tag\": \"vulkan-dot-path\",\n";
  out << "  \"status\": \"diagnostic\",\n";
  out << "  \"backend\": \"vulkan\",\n";
  out << "  \"hardware\": {\n";
  out << "    \"device_name\": \"" << json_escape(properties.deviceName) << "\",\n";
  out << "    \"vendor_id\": " << properties.vendorID << ",\n";
  out << "    \"device_id\": " << properties.deviceID << ",\n";
  out << "    \"device_type\": " << properties.deviceType << ",\n";
  out << "    \"api_version\": \"" << version_string(properties.apiVersion) << "\",\n";
  out << "    \"driver_version_raw\": " << properties.driverVersion << ",\n";
  out << "    \"queue_family\": " << queue_family << ",\n";
  out << "    \"shader_integer_dot_product\": true,\n";
  out << "    \"device_extension\": \"VK_KHR_shader_integer_dot_product\"\n";
  out << "  },\n";
  out << "  \"config\": {\n";
  out << "    \"mode\": \"" << json_escape(row.mode) << "\",\n";
  out << "    \"mode_id\": " << HIPENGINE_DOT_MODE << ",\n";
  out << "    \"groups\": " << row.groups << ",\n";
  out << "    \"n\": " << row.n << ",\n";
  out << "    \"body_iters\": " << row.body_iters << ",\n";
  out << "    \"block_size\": " << row.block_size << ",\n";
  out << "    \"data_elems\": " << row.data_elems << ",\n";
  out << "    \"timing_mode\": \"" << json_escape(args.timing_mode) << "\",\n";
  out << "    \"reps\": " << args.reps << ",\n";
  out << "    \"warmup\": " << args.warmup << ",\n";
  out << "    \"samples\": " << args.samples << ",\n";
  out << "    \"method\": \"pre-recorded Vulkan command buffer; VK_KHR_shader_integer_dot_product dot-path diagnostic; sampled exact CPU oracle\"\n";
  out << "  },\n";
  out << "  \"rows\": [\n";
  out << "    {\n";
  out << "      \"mode\": \"" << json_escape(row.mode) << "\",\n";
  out << "      \"groups\": " << row.groups << ",\n";
  out << "      \"n\": " << row.n << ",\n";
  out << "      \"body_iters\": " << row.body_iters << ",\n";
  out << "      \"block_size\": " << row.block_size << ",\n";
  out << "      \"data_elems\": " << row.data_elems << ",\n";
  out << "      \"bytes_per_dispatch\": " << row.bytes_per_dispatch << ",\n";
  out << "      \"integer_ops_per_dispatch\": " << row.integer_ops_per_dispatch << ",\n";
  out << "      \"median_us\": " << row.median_us << ",\n";
  out << "      \"p05_us\": " << row.p05_us << ",\n";
  out << "      \"p95_us\": " << row.p95_us << ",\n";
  out << "      \"min_us\": " << row.min_us << ",\n";
  out << "      \"max_us\": " << row.max_us << ",\n";
  out << "      \"bandwidth_gbps\": " << row.bandwidth_gbps << ",\n";
  out << "      \"gops\": " << row.gops << ",\n";
  out << "      \"timing_mode\": \"" << json_escape(args.timing_mode) << "\",\n";
  out << "      \"queue_or_stream_count\": 1,\n";
  out << "      \"gpu_timestamps_supported\": "
      << (row.gpu_timestamps_supported ? "true" : "false") << ",\n";
  out << "      \"timed_sequence_correctness_pass\": "
      << (row.timed_sequence_correctness_pass ? "true" : "false") << ",\n";
  out << "      \"synchronization_pass\": "
      << (row.timed_sequence_correctness_pass ? "true" : "false") << ",\n";
  out << "      \"barrier_count\": "
      << (args.timing_mode == "serial_latency" ? args.reps - 1 : 0) << ",\n";
  out << "      \"timing_raw\": {\n";
  write_timing_raw(
      out,
      "single",
      1,
      row.single_gpu_samples_us,
      row.single_host_samples_us,
      true);
  write_timing_raw(
      out,
      "burst",
      args.reps,
      row.burst_gpu_samples_us,
      row.burst_host_samples_us,
      false);
  out << "      },\n";
  out << "      \"max_abs\": " << row.max_abs << ",\n";
  out << "      \"max_rel\": " << row.max_rel << ",\n";
  out << "      \"correctness_pass\": " << (row.correctness_pass ? "true" : "false") << "\n";
  out << "    }\n";
  out << "  ]\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    std::vector<uint32_t> spirv = read_spirv(args.spirv_path);

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "hipEngine Vulkan dot path";
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
    VkDevice device = create_device_with_integer_dot(physical_device, queue_family, &queue_priority);

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queue_family, 0, &queue);
    VkShaderModule shader_module = create_shader_module(device, spirv);
    VkDescriptorSetLayout descriptor_layout = create_descriptor_set_layout(device);
    VkPipelineLayout pipeline_layout = create_pipeline_layout(device, descriptor_layout);

    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.queueFamilyIndex = queue_family;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool),
          "vkCreateCommandPool");

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    check(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");

    Row row = run_config(
        args,
        physical_device,
        device,
        queue,
        queue_family,
        command_pool,
        descriptor_layout,
        pipeline_layout,
        shader_module,
        fence);
    std::cout << "[vulkan] mode=" << row.mode << " groups=" << row.groups
              << " median=" << row.median_us
              << " us correctness=" << (row.correctness_pass ? "pass" : "fail")
              << "\n";

    if (args.json_path.empty()) {
      write_json(args, properties, queue_family, row, std::cout);
    } else {
      std::ofstream output(args.json_path);
      if (!output) {
        fail("could not open JSON path: " + args.json_path);
      }
      write_json(args, properties, queue_family, row, output);
    }

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    vkDestroyShaderModule(device, shader_module, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
