#ifndef VG_HPP
#define VG_HPP

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

namespace vg {

class Window {
public:
  Window(const std::string& title, int width, int height);

  void run_continuous(std::function<void()> f);
  void destroy();

  operator GLFWwindow*() const {
    return m_window;
  }

private:
  GLFWwindow* m_window;
};

struct SurfaceDetails {
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> present_modes;
  vk::SurfaceCapabilitiesKHR caps;
};

struct RenderGroup {
  vk::PhysicalDevice dev;
  std::uint32_t qfam_idx;
  SurfaceDetails surf_details;
};

class Renderer {
public:
  Renderer(Window window);
  void destroy();

  void draw();

private:
  Window window;
  size_t frame_idx {0};

  vk::Instance inst;
  void createInstance();

  vk::SurfaceKHR surf;
  void createSurface();

  RenderGroup rend_group;
  void chooseRenderGroup();

  vk::Device dev;
  void createDevice();

  vk::Queue gfx_q;

  vk::SurfaceFormatKHR format;
  SurfaceDetails getSurfaceDetails(vk::PhysicalDevice dev);
  void chooseSurfaceFormat();

  std::uint32_t img_count;
  void chooseImageCount();

  vk::Extent2D extent;
  void chooseSwapExtent();

  vk::SwapchainKHR swapchain;
  vk::PresentModeKHR choosePresentMode();
  void createSwapchain();

  void createSwapchainDependents();
  void destroySwapchainDependents();
  void recreateSwapchain();

  std::vector<vk::Image> images;

  std::vector<vk::ImageView> image_views;
  void createImageViews();

  vk::RenderPass render_pass;
  void createRenderPass();

  vk::Pipeline pipeline;
  vk::PipelineLayout layout;
  void createPipeline();

  std::vector<vk::Framebuffer> framebuffers;
  void createFramebuffers();

  vk::CommandPool cmd_pool;
  std::vector<vk::CommandBuffer> cmd_bufs;

  void recordCommandBuffers();

  std::vector<vk::Semaphore> image_available;
  std::vector<vk::Semaphore> render_finished;
  std::vector<vk::Fence> frame_inflight;
  std::vector<vk::Fence> image_inflight;
  void createSyncPrimitives();
};

} // namespace vg

#endif // VG_HPP
