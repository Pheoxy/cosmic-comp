pub mod gles;
pub mod pixman;

use smithay::backend::drm::DrmDeviceFd;

#[cfg(not(feature = "renderer_vulkan"))]
pub type KmsGraphics = gles::GbmGlowBackend<DrmDeviceFd>;

#[cfg(feature = "renderer_vulkan")]
pub type KmsGraphics =
    smithay::backend::renderer::multigpu::vulkan::VulkanGbmBackend<DrmDeviceFd>;
