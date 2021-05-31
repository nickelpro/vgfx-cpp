#include <fstream>
#include <stdexcept>

#include "vg.hpp"

namespace vg {

static std::vector<char> readFile(const std::string& file_name) {
  std::ifstream ifs {file_name, std::ios::ate | std::ios::binary};
  if(!ifs)
    throw std::runtime_error {"failed to open file: " + file_name};

  auto size {ifs.tellg()};
  std::vector<char> buf(size);
  ifs.seekg(0);
  ifs.read(buf.data(), size);
  return buf;
}

Window::Window(const std::string& title, int width, int height) {
  if(!glfwInit())
    throw std::runtime_error("Failed to init glfw");

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  m_window = glfwCreateWindow(width, height, title.data(), nullptr, nullptr);
}

void Window::run_continuous(std::function<void()> f) {
  while(!glfwWindowShouldClose(m_window)) {
    glfwPollEvents();
    f();
  }
}

void Window::destroy() {
  glfwDestroyWindow(m_window);
  glfwTerminate();
}

Renderer::Renderer(Window window) : window {window} {

  createInstance();
  createSurface();
  chooseRenderGroup();
  createDevice();
  gfx_q = dev.getQueue(rend_group.qfam_idx, 0);

  chooseSurfaceFormat();
  chooseImageCount();
  chooseSwapExtent();
  createSwapchain();
  images = dev.getSwapchainImagesKHR(swapchain);

  createImageViews();
  createRenderPass();
  createPipeline();
  createFramebuffers();
  cmd_pool = dev.createCommandPool({.queueFamilyIndex {rend_group.qfam_idx}});
  cmd_bufs = dev.allocateCommandBuffers({
      .commandPool {cmd_pool},
      .commandBufferCount {static_cast<std::uint32_t>(framebuffers.size())},
  });
  recordCommandBuffers();

  createSyncPrimitives();
}

void Renderer::destroy() {
  dev.waitIdle();

  for(size_t i {0}; i < img_count; i++) {
    dev.destroy(frame_inflight[i]);
    dev.destroy(image_available[i]);
    dev.destroy(render_finished[i]);
  }

  dev.destroy(cmd_pool);
  for(auto fb : framebuffers)
    dev.destroy(fb);
  dev.destroy(pipeline);
  dev.destroy(layout);
  dev.destroy(render_pass);
  for(auto image_view : image_views)
    dev.destroy(image_view);

  dev.destroy(swapchain);

  dev.destroy();
  inst.destroy(surf);
  inst.destroy();
}

void Renderer::draw() {
  if(dev.waitForFences(std::array {frame_inflight[frame_idx]}, true,
         UINT64_MAX) != vk::Result::eSuccess)
    throw std::runtime_error {"wait failure or timeout"};

  // clang-format off
  auto img_idx {dev.acquireNextImageKHR(
      swapchain, UINT64_MAX, image_available[frame_idx]).value};
  // clang-format on

  if(image_inflight[img_idx] &&
      dev.waitForFences(std::array {image_inflight[img_idx]}, true,
          UINT64_MAX) != vk::Result::eSuccess)
    throw std::runtime_error {"wait failure or timeout"};
  image_inflight[img_idx] = frame_inflight[frame_idx];

  vk::PipelineStageFlags flags {
      vk::PipelineStageFlagBits::eColorAttachmentOutput};
  std::array submit_info {vk::SubmitInfo {
      .waitSemaphoreCount {1},
      .pWaitSemaphores {&image_available[frame_idx]},
      .pWaitDstStageMask {&flags},
      .commandBufferCount {1},
      .pCommandBuffers {&cmd_bufs[img_idx]},
      .signalSemaphoreCount {1},
      .pSignalSemaphores {&render_finished[frame_idx]},
  }};

  dev.resetFences(std::array {frame_inflight[frame_idx]});
  gfx_q.submit(submit_info, frame_inflight[frame_idx]);
  if(gfx_q.presentKHR({
         .waitSemaphoreCount {1},
         .pWaitSemaphores {&render_finished[frame_idx]},
         .swapchainCount {1},
         .pSwapchains {&swapchain},
         .pImageIndices {&img_idx},
     }) != vk::Result::eSuccess)
    throw std::runtime_error {"failed to Present"};

  ++frame_idx %= img_count;
}

void Renderer::createInstance() {
  const char* validation_layer {"VK_LAYER_KHRONOS_validation"};
  std::uint32_t glfw_count;
  const char** glfw_exts {glfwGetRequiredInstanceExtensions(&glfw_count)};
  std::vector<const char*> extensions(glfw_exts, glfw_exts + glfw_count);
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  const vk::ApplicationInfo app_info {
      .apiVersion {VK_API_VERSION_1_2},
  };
  inst = vk::createInstance({
      .pApplicationInfo {&app_info},
      .enabledLayerCount {1},
      .ppEnabledLayerNames {&validation_layer},
      .enabledExtensionCount {static_cast<std::uint32_t>(extensions.size())},
      .ppEnabledExtensionNames {extensions.data()},
  });
}

void Renderer::createSurface() {
  VkSurfaceKHR _surf;
  if(glfwCreateWindowSurface(inst, window, nullptr, &_surf) != VK_SUCCESS)
    throw std::runtime_error {"failed to create window surface"};
  surf = _surf;
}

SurfaceDetails Renderer::getSurfaceDetails(vk::PhysicalDevice phy_dev) {
  return {
      .formats {phy_dev.getSurfaceFormatsKHR(surf)},
      .present_modes {phy_dev.getSurfacePresentModesKHR(surf)},
      .caps {phy_dev.getSurfaceCapabilitiesKHR(surf)},
  };
}

void Renderer::chooseRenderGroup() {
  std::vector<RenderGroup> valid_groups;
  for(const auto dev : inst.enumeratePhysicalDevices()) {

    auto surf_details {getSurfaceDetails(dev)};
    if(surf_details.formats.empty() || surf_details.present_modes.empty())
      continue;

    auto qfams {dev.getQueueFamilyProperties()};
    for(std::uint32_t i {0}; i < qfams.size(); i++)
      if(qfams[i].queueFlags & vk::QueueFlagBits::eGraphics &&
          dev.getSurfaceSupportKHR(i, surf)) {
        rend_group = {dev, i, surf_details};
        if(dev.getProperties().deviceType ==
            vk::PhysicalDeviceType::eDiscreteGpu)
          return;
        valid_groups.push_back(rend_group);
      }
  }
  if(valid_groups.empty())
    throw std::runtime_error {"no suitable device group found"};
  rend_group = valid_groups[0];
}

void Renderer::createDevice() {
  const float one {1.0f};
  const auto feats {rend_group.dev.getFeatures()};
  const vk::DeviceQueueCreateInfo q_info {
      .queueFamilyIndex {rend_group.qfam_idx},
      .queueCount {1},
      .pQueuePriorities {&one},
  };
  const char* swap_ext {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  dev = rend_group.dev.createDevice({
      .queueCreateInfoCount {1},
      .pQueueCreateInfos {&q_info},
      .enabledExtensionCount {1},
      .ppEnabledExtensionNames {&swap_ext},
      .pEnabledFeatures {&feats},
  });
}

void Renderer::chooseSurfaceFormat() {
  for(const auto& fmt : rend_group.surf_details.formats)
    if(fmt.format == vk::Format::eB8G8R8A8Srgb &&
        fmt.colorSpace == vk::ColorSpaceKHR::eVkColorspaceSrgbNonlinear) {
      format = fmt;
      return;
    }
  format = rend_group.surf_details.formats[0];
}

void Renderer::chooseImageCount() {
  img_count = rend_group.surf_details.caps.minImageCount + 1;
  if(rend_group.surf_details.caps.maxImageCount &&
      img_count > rend_group.surf_details.caps.maxImageCount)
    img_count = rend_group.surf_details.caps.maxImageCount;
}

void Renderer::chooseSwapExtent() {
  if(rend_group.surf_details.caps.currentExtent.width != UINT32_MAX)
    extent = rend_group.surf_details.caps.currentExtent;
  else {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    extent.width = {std::clamp(static_cast<std::uint32_t>(width),
        rend_group.surf_details.caps.minImageExtent.width,
        rend_group.surf_details.caps.maxImageExtent.width)};
    extent.height = {std::clamp(static_cast<std::uint32_t>(height),
        rend_group.surf_details.caps.minImageExtent.height,
        rend_group.surf_details.caps.maxImageExtent.height)};
  }
}

vk::PresentModeKHR Renderer::choosePresentMode() {
  vk::PresentModeKHR ret {vk::PresentModeKHR::eFifo};
  for(auto present_mode : rend_group.surf_details.present_modes) {
    if(present_mode == vk::PresentModeKHR::eMailbox)
      return vk::PresentModeKHR::eMailbox;
    else if(present_mode == vk::PresentModeKHR::eImmediate)
      ret = present_mode;
  }
  return ret;
}

void Renderer::createSwapchain() {
  swapchain = dev.createSwapchainKHR({
      .surface {surf},
      .minImageCount {img_count},
      .imageFormat {format.format},
      .imageColorSpace {format.colorSpace},
      .imageExtent {extent},
      .imageArrayLayers {1},
      .imageUsage {vk::ImageUsageFlagBits::eColorAttachment},
      .imageSharingMode {vk::SharingMode::eExclusive},
      .preTransform {rend_group.surf_details.caps.currentTransform},
      .compositeAlpha {vk::CompositeAlphaFlagBitsKHR::eOpaque},
      .presentMode {choosePresentMode()},
      .clipped {true},
  });
}

void Renderer::createImageViews() {
  image_views.resize(images.size());
  for(size_t i {0}; i < images.size(); i++)
    image_views[i] = dev.createImageView({
        .image {images[i]},
        .viewType {vk::ImageViewType::e2D},
        .format {format.format},
        .subresourceRange {
            .aspectMask {vk::ImageAspectFlagBits::eColor},
            .baseMipLevel {0},
            .levelCount {1},
            .baseArrayLayer {0},
            .layerCount {1},
        },
    });
}

void Renderer::createRenderPass() {
  vk::AttachmentDescription attach_dec {
      .format {format.format},
      .samples {vk::SampleCountFlagBits::e1},
      .stencilLoadOp {vk::AttachmentLoadOp::eDontCare},
      .stencilStoreOp {vk::AttachmentStoreOp::eDontCare},
      .finalLayout {vk::ImageLayout::ePresentSrcKHR},
  };

  vk::AttachmentReference attach_ref {
      .attachment {0},
      .layout {vk::ImageLayout::eColorAttachmentOptimal},
  };

  vk::SubpassDescription subpass_desc {
      .colorAttachmentCount {1},
      .pColorAttachments {&attach_ref},
  };

  vk::SubpassDependency subpass_dep {
      .srcSubpass {VK_SUBPASS_EXTERNAL},
      .dstSubpass {0},
      .srcStageMask {vk::PipelineStageFlagBits::eColorAttachmentOutput},
      .dstStageMask {vk::PipelineStageFlagBits::eColorAttachmentOutput},
      .dstAccessMask {vk::AccessFlagBits::eColorAttachmentWrite},
  };

  render_pass = dev.createRenderPass({
      .attachmentCount {1},
      .pAttachments {&attach_dec},
      .subpassCount {1},
      .pSubpasses {&subpass_desc},
      .dependencyCount {1},
      .pDependencies {&subpass_dep},
  });
}

void Renderer::createPipeline() {
  auto vert_code {readFile("shaders/shader.vert.spv")};
  auto frag_code {readFile("shaders/shader.frag.spv")};

  auto vert_module {dev.createShaderModule({
      .codeSize {vert_code.size()},
      .pCode {reinterpret_cast<const std::uint32_t*>(vert_code.data())},
  })};
  auto frag_module {dev.createShaderModule({
      .codeSize {frag_code.size()},
      .pCode {reinterpret_cast<const std::uint32_t*>(frag_code.data())},
  })};

  std::array shader_stages {
      vk::PipelineShaderStageCreateInfo {
          .stage {vk::ShaderStageFlagBits::eVertex},
          .module {vert_module},
          .pName {"main"},
      },
      vk::PipelineShaderStageCreateInfo {
          .stage {vk::ShaderStageFlagBits::eFragment},
          .module {frag_module},
          .pName {"main"},
      },
  };

  vk::PipelineVertexInputStateCreateInfo pipe_vert_info {};

  vk::PipelineInputAssemblyStateCreateInfo pipe_input_asm_info {
      .topology {vk::PrimitiveTopology::eTriangleList},
  };

  vk::Viewport viewport {
      .width {static_cast<float>(extent.width)},
      .height {static_cast<float>(extent.height)},
      .maxDepth {1.0f},
  };

  vk::Rect2D scissor {
      .extent {extent},
  };

  vk::PipelineViewportStateCreateInfo viewport_state {
      .viewportCount {1},
      .pViewports {&viewport},
      .scissorCount {1},
      .pScissors {&scissor},
  };

  vk::PipelineRasterizationStateCreateInfo rast_state {
      .cullMode {vk::CullModeFlagBits::eBack},
      .frontFace {vk::FrontFace::eClockwise},
      .lineWidth {1.0f},
  };

  vk::PipelineMultisampleStateCreateInfo mm_sample {
      .rasterizationSamples {vk::SampleCountFlagBits::e1},
      .minSampleShading {1.0f},
  };

  vk::PipelineColorBlendAttachmentState color_blend_attach {
      .colorWriteMask {
          vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA},
  };

  vk::PipelineColorBlendStateCreateInfo color_blend_state {
      .attachmentCount {1},
      .pAttachments {&color_blend_attach},
  };

  layout = dev.createPipelineLayout({});

  // clang-format off
  pipeline = dev.createGraphicsPipeline(VK_NULL_HANDLE, {
      .stageCount {shader_stages.size()},
      .pStages {shader_stages.data()},
      .pVertexInputState {&pipe_vert_info},
      .pInputAssemblyState {&pipe_input_asm_info},
      .pViewportState {&viewport_state},
      .pRasterizationState {&rast_state},
      .pMultisampleState {&mm_sample},
      .pColorBlendState {&color_blend_state},
      .layout {layout},
      .renderPass {render_pass},
  }).value;
  // clang-format on

  dev.destroy(vert_module);
  dev.destroy(frag_module);
}

void Renderer::createFramebuffers() {
  framebuffers.resize(image_views.size());
  for(size_t i {0}; i < image_views.size(); i++)
    framebuffers[i] = dev.createFramebuffer({
        .renderPass {render_pass},
        .attachmentCount {1},
        .pAttachments {&image_views[i]},
        .width {extent.width},
        .height {extent.height},
        .layers {1},
    });
}

void Renderer::recordCommandBuffers() {
  const vk::ClearValue clear_color {std::array {0.0f, 0.0f, 0.0f, 1.0f}};
  for(size_t i {0}; i < cmd_bufs.size(); i++) {
    cmd_bufs[i].begin(vk::CommandBufferBeginInfo {});
    cmd_bufs[i].beginRenderPass(
        {
            .renderPass {render_pass},
            .framebuffer {framebuffers[i]},
            .renderArea {.extent {extent}},
            .clearValueCount {1},
            .pClearValues {&clear_color},
        },
        vk::SubpassContents::eInline);

    cmd_bufs[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
    cmd_bufs[i].draw(3, 1, 0, 0);
    cmd_bufs[i].endRenderPass();

    cmd_bufs[i].end();
  }
}

void Renderer::createSyncPrimitives() {
  image_available.resize(img_count);
  render_finished.resize(img_count);
  frame_inflight.resize(img_count);
  image_inflight.resize(img_count);

  for(size_t i {0}; i < img_count; i++) {
    image_available[i] = dev.createSemaphore({});
    render_finished[i] = dev.createSemaphore({});
    frame_inflight[i] =
        dev.createFence({.flags {vk::FenceCreateFlagBits::eSignaled}});
  }
}

} // namespace vg
