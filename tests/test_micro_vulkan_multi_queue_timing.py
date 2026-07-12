from __future__ import annotations

import ctypes.util
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HEADER = REPO_ROOT / "benchmarks" / "micro" / "runners" / "micro_timing_vulkan.hpp"


def test_multi_queue_header_exposes_coordinated_lane_contract() -> None:
    source = HEADER.read_text(encoding="utf-8")

    assert "select_compute_queue_family" in source
    assert "timeline_semaphore_supported" in source
    assert "calibrated_timestamps_extension" in source
    assert "class VulkanMultiQueueTimer" in source
    assert "VK_SEMAPHORE_TYPE_TIMELINE" in source
    assert "vkSignalSemaphore" in source
    assert "VK_PIPELINE_STAGE_ALL_COMMANDS_BIT" in source
    assert "2 * lane_count_" in source
    assert "earliest" in source and "latest" in source


def test_multi_queue_header_compiles_and_probes_queue_selection(tmp_path: Path) -> None:
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None or ctypes.util.find_library("vulkan") is None:
        pytest.skip("Vulkan compiler/runtime is unavailable")

    source = tmp_path / "probe.cpp"
    executable = tmp_path / "probe"
    source.write_text(
        """
#include <vulkan/vulkan.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "benchmarks/micro/runners/micro_timing_vulkan.hpp"

int main() {
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo create{};
  create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create.pApplicationInfo = &app;
  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&create, nullptr, &instance) != VK_SUCCESS) return 2;
  uint32_t count = 0;
  if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) {
    vkDestroyInstance(instance, nullptr);
    return 3;
  }
  std::vector<VkPhysicalDevice> devices(count);
  if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return 4;
  const auto selected = hipengine::micro::select_compute_queue_family(devices[0], 4);
  const bool timeline = hipengine::micro::timeline_semaphore_supported(devices[0]);
  const char* calibrated = hipengine::micro::calibrated_timestamps_extension(devices[0]);
  if (selected.queue_count < 2 || !timeline || calibrated == nullptr) {
    std::cout << "unsupported " << selected.queue_count << " " << timeline << std::endl;
    vkDestroyInstance(instance, nullptr);
    return 0;
  }

  float priorities[2] = {1.0f, 1.0f};
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = selected.index;
  queue_info.queueCount = 2;
  queue_info.pQueuePriorities = priorities;
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{};
  timeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  timeline_features.timelineSemaphore = VK_TRUE;
  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.pNext = &timeline_features;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount = 1;
  device_info.ppEnabledExtensionNames = &calibrated;
  VkDevice device = VK_NULL_HANDLE;
  if (vkCreateDevice(devices[0], &device_info, nullptr, &device) != VK_SUCCESS) return 5;
  std::vector<VkQueue> queues(2, VK_NULL_HANDLE);
  vkGetDeviceQueue(device, selected.index, 0, &queues[0]);
  vkGetDeviceQueue(device, selected.index, 1, &queues[1]);

  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = selected.index;
  VkCommandPool pool = VK_NULL_HANDLE;
  if (vkCreateCommandPool(device, &pool_info, nullptr, &pool) != VK_SUCCESS) return 6;
  std::vector<VkCommandBuffer> commands(2, VK_NULL_HANDLE);
  VkCommandBufferAllocateInfo allocate{};
  allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate.commandPool = pool;
  allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate.commandBufferCount = 2;
  if (vkAllocateCommandBuffers(device, &allocate, commands.data()) != VK_SUCCESS) return 7;
  std::vector<VkFence> fences(2, VK_NULL_HANDLE);
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  for (VkFence& fence : fences) {
    if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS) return 8;
  }

  double gpu_us = -1.0;
  double host_us = -1.0;
  {
    hipengine::micro::VulkanMultiQueueTimer timer(
        devices[0], device, selected.index, 2, calibrated);
    for (uint32_t lane = 0; lane < 2; ++lane) {
      VkCommandBufferBeginInfo begin{};
      begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
      if (vkBeginCommandBuffer(commands[lane], &begin) != VK_SUCCESS) return 9;
      timer.record_begin(commands[lane], lane);
      timer.record_end(commands[lane], lane);
      if (vkEndCommandBuffer(commands[lane]) != VK_SUCCESS) return 10;
    }
    for (uint32_t sample_index = 0; sample_index < 2; ++sample_index) {
      const auto sample = timer.submit_and_wait(queues, commands, fences);
      gpu_us = sample.gpu_sequence_us;
      host_us = sample.host_sequence_us;
    }
  }
  std::cout << selected.index << " " << selected.queue_count << " " << timeline
            << " " << gpu_us << " " << host_us << std::endl;
  for (VkFence fence : fences) vkDestroyFence(device, fence, nullptr);
  vkDestroyCommandPool(device, pool, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  return gpu_us >= 0.0 && host_us > 0.0 ? 0 : 11;
}
""".lstrip(),
        encoding="utf-8",
    )
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(REPO_ROOT),
            str(source),
            "-lvulkan",
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True
    )
    assert run_result.returncode == 0, run_result.stderr
    fields = run_result.stdout.strip().split()
    if fields[0] == "unsupported":
        pytest.skip("Vulkan device lacks two timestamp-capable queues or timeline semaphores")
    family, queues, timeline, gpu_us, host_us = fields
    assert int(family) >= 0
    assert int(queues) >= 2
    assert timeline == "1"
    assert float(gpu_us) >= 0.0
    assert float(host_us) > 0.0
