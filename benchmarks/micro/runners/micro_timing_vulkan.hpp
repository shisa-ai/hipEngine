#pragma once

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipengine::micro {

enum class TimingMode { SerialLatency, IndependentThroughput };

inline TimingMode parse_timing_mode(const std::string& value) {
  if (value == "serial_latency") {
    return TimingMode::SerialLatency;
  }
  if (value == "independent_throughput") {
    return TimingMode::IndependentThroughput;
  }
  throw std::runtime_error(
      "timing mode must be serial_latency or independent_throughput");
}

inline const char* timing_mode_name(TimingMode mode) {
  return mode == TimingMode::SerialLatency ? "serial_latency" : "independent_throughput";
}

inline void timing_vk_check(VkResult result, const char* what) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        std::string(what) + " failed with VkResult " +
        std::to_string(static_cast<int>(result)));
  }
}

inline void compute_buffer_barrier(
    VkCommandBuffer command_buffer,
    const std::vector<VkBufferMemoryBarrier>& barriers) {
  if (barriers.empty()) {
    return;
  }
  vkCmdPipelineBarrier(
      command_buffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      static_cast<uint32_t>(barriers.size()),
      barriers.data(),
      0,
      nullptr);
}

inline VkBufferMemoryBarrier make_compute_buffer_barrier(
    VkBuffer buffer,
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
  barrier.buffer = buffer;
  barrier.offset = offset;
  barrier.size = size;
  return barrier;
}

struct VulkanTimingSample {
  double gpu_sequence_us;
  double host_sequence_us;
};

class VulkanSequenceTimer {
 public:
  VulkanSequenceTimer(
      VkPhysicalDevice physical_device,
      VkDevice device,
      uint32_t queue_family_index)
      : device_(device) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    timestamp_period_ns_ = static_cast<double>(properties.limits.timestampPeriod);

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device, &family_count, families.data());
    if (queue_family_index >= family_count) {
      throw std::runtime_error("invalid Vulkan queue family index");
    }
    timestamp_valid_bits_ = families[queue_family_index].timestampValidBits;
    if (timestamp_valid_bits_ == 0) {
      return;
    }
    VkQueryPoolCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    create_info.queryCount = 2;
    timing_vk_check(
        vkCreateQueryPool(device_, &create_info, nullptr, &query_pool_),
        "vkCreateQueryPool");
  }

  VulkanSequenceTimer(const VulkanSequenceTimer&) = delete;
  VulkanSequenceTimer& operator=(const VulkanSequenceTimer&) = delete;

  ~VulkanSequenceTimer() {
    if (query_pool_ != VK_NULL_HANDLE) {
      vkDestroyQueryPool(device_, query_pool_, nullptr);
    }
  }

  bool gpu_timestamps_supported() const {
    return query_pool_ != VK_NULL_HANDLE;
  }

  double timestamp_period_ns() const {
    return timestamp_period_ns_;
  }

  uint32_t timestamp_valid_bits() const {
    return timestamp_valid_bits_;
  }

  void record_begin(VkCommandBuffer command_buffer) const {
    if (!gpu_timestamps_supported()) {
      return;
    }
    vkCmdResetQueryPool(command_buffer, query_pool_, 0, 2);
    vkCmdWriteTimestamp(
        command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 0);
  }

  void record_end(VkCommandBuffer command_buffer) const {
    if (!gpu_timestamps_supported()) {
      return;
    }
    vkCmdWriteTimestamp(
        command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 1);
  }

  VulkanTimingSample submit_and_wait(
      VkQueue queue,
      VkCommandBuffer command_buffer,
      VkFence fence) const {
    timing_vk_check(vkResetFences(device_, 1, &fence), "vkResetFences");
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    const auto host_start = std::chrono::steady_clock::now();
    timing_vk_check(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
    timing_vk_check(
        vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
        "vkWaitForFences");
    const auto host_stop = std::chrono::steady_clock::now();

    double gpu_us = -1.0;
    if (gpu_timestamps_supported()) {
      uint64_t timestamps[2] = {};
      timing_vk_check(
          vkGetQueryPoolResults(
              device_,
              query_pool_,
              0,
              2,
              sizeof(timestamps),
              timestamps,
              sizeof(uint64_t),
              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
          "vkGetQueryPoolResults");
      uint64_t delta = 0;
      if (timestamp_valid_bits_ >= 64) {
        delta = timestamps[1] - timestamps[0];
      } else {
        const uint64_t mask = (uint64_t{1} << timestamp_valid_bits_) - 1;
        delta = (timestamps[1] - timestamps[0]) & mask;
      }
      gpu_us = static_cast<double>(delta) * timestamp_period_ns_ / 1000.0;
    }
    return {
        gpu_us,
        std::chrono::duration<double, std::micro>(host_stop - host_start).count(),
    };
  }

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueryPool query_pool_ = VK_NULL_HANDLE;
  double timestamp_period_ns_ = 0.0;
  uint32_t timestamp_valid_bits_ = 0;
};

}  // namespace hipengine::micro
