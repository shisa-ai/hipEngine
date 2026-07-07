#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kLocalSizeX = 256;

struct Args {
  std::string spirv_path;
  std::string json_path;
  std::vector<uint32_t> counts{1, 50, 200, 941};
  std::vector<uint32_t> grid_sweep;
  uint32_t grid_sweep_count = 941;
  uint32_t n = 256;
  uint32_t reps = 50;
  uint32_t warmup = 10;
  uint32_t device_index = 0;
};

struct PushConstants {
  uint32_t n;
  uint32_t salt[16];
};

struct Row {
  uint32_t dispatch_count;
  uint32_t grid_blocks;
  double burst_us_median;
  double us_per_dispatch;
  double burst_us_min;
  uint32_t reps;
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
  if (text.empty()) {
    return values;
  }
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
    if (flag == "--spirv") {
      args.spirv_path = require_value(i, argc, argv, flag);
    } else if (flag == "--json") {
      args.json_path = require_value(i, argc, argv, flag);
    } else if (flag == "--counts") {
      args.counts = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--grid-sweep") {
      args.grid_sweep = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--grid-sweep-count") {
      args.grid_sweep_count = static_cast<uint32_t>(
          std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--n") {
      args.n = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--reps") {
      args.reps = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--warmup") {
      args.warmup = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--device-index") {
      args.device_index = static_cast<uint32_t>(
          std::stoul(require_value(i, argc, argv, flag)));
    } else {
      fail("unknown argument: " + flag);
    }
  }
  if (args.spirv_path.empty()) {
    fail("--spirv is required");
  }
  if (args.counts.empty()) {
    fail("--counts must not be empty");
  }
  if (args.n == 0 || args.reps == 0) {
    fail("--n and --reps must be positive");
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

uint32_t find_queue_family(VkPhysicalDevice physical_device) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
  if (count == 0) {
    fail("physical device has no queue families");
  }
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());
  for (uint32_t i = 0; i < count; ++i) {
    if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      return i;
    }
  }
  fail("physical device has no compute queue family");
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

uint32_t auto_grid(uint32_t n) {
  return std::max<uint32_t>(1, (n + kLocalSizeX - 1) / kLocalSizeX);
}

double median(std::vector<double> values) {
  if (values.empty()) {
    fail("cannot compute median of empty values");
  }
  std::sort(values.begin(), values.end());
  size_t mid = values.size() / 2;
  if ((values.size() % 2) == 1) {
    return values[mid];
  }
  return (values[mid - 1] + values[mid]) * 0.5;
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

VkCommandBuffer record_command_buffer(
    VkDevice device,
    VkCommandPool command_pool,
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    VkDescriptorSet descriptor_set,
    uint32_t dispatch_count,
    uint32_t grid_blocks,
    uint32_t n) {
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
  check(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");
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
  for (uint32_t i = 0; i < dispatch_count; ++i) {
    PushConstants push{};
    push.n = n;
    for (uint32_t s = 0; s < 16; ++s) {
      push.salt[s] = i + s + 1;
    }
    vkCmdPushConstants(
        command_buffer,
        pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(PushConstants),
        &push);
    vkCmdDispatch(command_buffer, grid_blocks, 1, 1);
  }
  check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
  return command_buffer;
}

Row measure_row(
    VkDevice device,
    VkQueue queue,
    VkCommandPool command_pool,
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    VkDescriptorSet descriptor_set,
    VkFence fence,
    const Args& args,
    uint32_t dispatch_count,
    uint32_t grid_blocks) {
  VkCommandBuffer command_buffer = record_command_buffer(
      device,
      command_pool,
      pipeline,
      pipeline_layout,
      descriptor_set,
      dispatch_count,
      grid_blocks,
      args.n);

  for (uint32_t i = 0; i < args.warmup; ++i) {
    (void)submit_once(device, queue, command_buffer, fence);
  }
  std::vector<double> samples;
  samples.reserve(args.reps);
  for (uint32_t i = 0; i < args.reps; ++i) {
    samples.push_back(submit_once(device, queue, command_buffer, fence));
  }
  check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
  vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);

  double burst_median = median(samples);
  double burst_min = *std::min_element(samples.begin(), samples.end());
  return Row{
      dispatch_count,
      grid_blocks,
      burst_median,
      burst_median / static_cast<double>(dispatch_count),
      burst_min,
      args.reps};
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
  out << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "."
      << VK_VERSION_PATCH(version);
  return out.str();
}

void write_rows(std::ostream& out, const std::vector<Row>& rows, int indent) {
  std::string pad(static_cast<size_t>(indent), ' ');
  out << "[\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row& row = rows[i];
    out << pad << "  {\n";
    out << pad << "    \"dispatch_count\": " << row.dispatch_count << ",\n";
    out << pad << "    \"grid_blocks\": " << row.grid_blocks << ",\n";
    out << pad << "    \"burst_us_median\": " << row.burst_us_median << ",\n";
    out << pad << "    \"us_per_dispatch\": " << row.us_per_dispatch << ",\n";
    out << pad << "    \"burst_us_min\": " << row.burst_us_min << ",\n";
    out << pad << "    \"reps\": " << row.reps << "\n";
    out << pad << "  }" << (i + 1 == rows.size() ? "\n" : ",\n");
  }
  out << pad << "]";
}

void write_json(
    const Args& args,
    const VkPhysicalDeviceProperties& properties,
    uint32_t queue_family,
    const std::vector<Row>& rows,
    const std::vector<Row>& grid_rows,
    std::ostream& out) {
  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"run_tag\": \"vulkan-dispatch-floor\",\n";
  out << "  \"status\": \"diagnostic\",\n";
  out << "  \"hardware\": {\n";
  out << "    \"device_name\": \"" << json_escape(properties.deviceName) << "\",\n";
  out << "    \"vendor_id\": " << properties.vendorID << ",\n";
  out << "    \"device_id\": " << properties.deviceID << ",\n";
  out << "    \"device_type\": " << properties.deviceType << ",\n";
  out << "    \"api_version\": \"" << version_string(properties.apiVersion) << "\",\n";
  out << "    \"driver_version_raw\": " << properties.driverVersion << ",\n";
  out << "    \"queue_family\": " << queue_family << "\n";
  out << "  },\n";
  out << "  \"config\": {\n";
  out << "    \"counts\": [";
  for (size_t i = 0; i < args.counts.size(); ++i) {
    out << (i == 0 ? "" : ", ") << args.counts[i];
  }
  out << "],\n";
  out << "    \"grid_sweep\": [";
  for (size_t i = 0; i < args.grid_sweep.size(); ++i) {
    out << (i == 0 ? "" : ", ") << args.grid_sweep[i];
  }
  out << "],\n";
  out << "    \"grid_sweep_count\": " << args.grid_sweep_count << ",\n";
  out << "    \"n_elements\": " << args.n << ",\n";
  out << "    \"reps\": " << args.reps << ",\n";
  out << "    \"warmup\": " << args.warmup << ",\n";
  out << "    \"local_size_x\": " << kLocalSizeX << ",\n";
  out << "    \"method\": \"pre-recorded Vulkan command buffer with N compute"
         " dispatches; wall time around vkQueueSubmit+vkWaitForFences; shader"
         " writes a storage buffer to prevent empty-shader elimination\"\n";
  out << "  },\n";
  out << "  \"rows\": ";
  write_rows(out, rows, 2);
  out << ",\n";
  out << "  \"grid_sweep_rows\": ";
  write_rows(out, grid_rows, 2);
  out << "\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    std::vector<uint32_t> spirv = read_spirv(args.spirv_path);

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "hipEngine Vulkan dispatch floor";
    app_info.applicationVersion = 1;
    app_info.pEngineName = "hipEngine microbench";
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_0;

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
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    uint32_t queue_family = find_queue_family(physical_device);

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkDevice device = VK_NULL_HANDLE;
    check(vkCreateDevice(physical_device, &device_info, nullptr, &device), "vkCreateDevice");

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spirv.size() * sizeof(uint32_t);
    shader_info.pCode = spirv.data();
    VkShaderModule shader_module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device, &shader_info, nullptr, &shader_module),
          "vkCreateShaderModule");

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = static_cast<VkDeviceSize>(args.n) * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    check(vkCreateBuffer(device, &buffer_info, nullptr, &buffer), "vkCreateBuffer");

    VkMemoryRequirements memory_requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);
    uint32_t memory_type = find_memory_type(
        physical_device,
        memory_requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex = memory_type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    check(vkAllocateMemory(device, &allocate_info, nullptr, &memory), "vkAllocateMemory");
    check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");

    void* mapped = nullptr;
    check(vkMapMemory(device, memory, 0, memory_requirements.size, 0, &mapped), "vkMapMemory");
    std::memset(mapped, 0, static_cast<size_t>(memory_requirements.size));
    vkUnmapMemory(device, memory);

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptor_layout_info{};
    descriptor_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout_info.bindingCount = 1;
    descriptor_layout_info.pBindings = &binding;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    check(vkCreateDescriptorSetLayout(
              device, &descriptor_layout_info, nullptr, &descriptor_layout),
          "vkCreateDescriptorSetLayout");

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout),
          "vkCreatePipelineLayout");

    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    check(vkCreateComputePipelines(
              device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
          "vkCreateComputePipelines");

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 1;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool),
          "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo descriptor_allocate_info{};
    descriptor_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_allocate_info.descriptorPool = descriptor_pool;
    descriptor_allocate_info.descriptorSetCount = 1;
    descriptor_allocate_info.pSetLayouts = &descriptor_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    check(vkAllocateDescriptorSets(device, &descriptor_allocate_info, &descriptor_set),
          "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo descriptor_buffer_info{};
    descriptor_buffer_info.buffer = buffer;
    descriptor_buffer_info.offset = 0;
    descriptor_buffer_info.range = buffer_info.size;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &descriptor_buffer_info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

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

    std::vector<Row> rows;
    rows.reserve(args.counts.size());
    for (uint32_t count : args.counts) {
      rows.push_back(measure_row(
          device,
          queue,
          command_pool,
          pipeline,
          pipeline_layout,
          descriptor_set,
          fence,
          args,
          count,
          auto_grid(args.n)));
      const Row& row = rows.back();
      std::cout << "[vulkan] N=" << row.dispatch_count
                << " grid=" << row.grid_blocks
                << " submit=" << row.us_per_dispatch << " us/dispatch\n";
    }

    std::vector<Row> grid_rows;
    grid_rows.reserve(args.grid_sweep.size());
    for (uint32_t grid_blocks : args.grid_sweep) {
      grid_rows.push_back(measure_row(
          device,
          queue,
          command_pool,
          pipeline,
          pipeline_layout,
          descriptor_set,
          fence,
          args,
          args.grid_sweep_count,
          grid_blocks));
      const Row& row = grid_rows.back();
      std::cout << "[grid] blocks=" << row.grid_blocks
                << " submit=" << row.us_per_dispatch << " us/dispatch\n";
    }

    if (args.json_path.empty()) {
      write_json(args, properties, queue_family, rows, grid_rows, std::cout);
    } else {
      std::ofstream output(args.json_path);
      if (!output) {
        fail("could not open JSON output: " + args.json_path);
      }
      write_json(args, properties, queue_family, rows, grid_rows, output);
      std::cout << "wrote " << args.json_path << "\n";
    }

    check(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
    vkDestroyShaderModule(device, shader_module, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
