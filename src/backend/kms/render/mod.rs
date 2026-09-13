pub mod gles;
pub mod pixman;

use anyhow::Context;
use smithay::backend::{
    drm::DrmDeviceFd,
    renderer::{
        Renderer,
        multigpu::{ApiDevice, Error as MultiError, GpuManager, MultiFrame, MultiRenderer, vulkan::VulkanGbmBackend},
    },
    vulkan::{Instance, version::Version},
};

pub type GlesGraphics = gles::GbmGlowBackend<DrmDeviceFd>;
pub type VulkanGraphics = VulkanGbmBackend<DrmDeviceFd>;

/// Always-available (not feature-gated) concrete per-backend renderer/frame/error types, for the
/// handful of call sites that genuinely need to name one backend's type specifically - e.g. because
/// they hold both backends' renderers live at once behind a runtime match (`SurfaceThreadState::redraw`
/// and its 3 direct helpers: `take_screencopy_frames`, `send_screencopy_result`, `postprocess_elements`).
/// Most rendering code should stay generic over `R` instead of naming either of these.
pub type GlesMultiRenderer<'a> = MultiRenderer<'a, 'a, GlesGraphics, GlesGraphics>;
pub type GlesMultiFrame<'a, 'frame, 'buffer> =
    MultiFrame<'a, 'a, 'frame, 'buffer, GlesGraphics, GlesGraphics>;
pub type GlesMultiError = MultiError<GlesGraphics, GlesGraphics>;
pub type VulkanMultiRenderer<'a> = MultiRenderer<'a, 'a, VulkanGraphics, VulkanGraphics>;
pub type VulkanMultiFrame<'a, 'frame, 'buffer> =
    MultiFrame<'a, 'a, 'frame, 'buffer, VulkanGraphics, VulkanGraphics>;
pub type VulkanMultiError = MultiError<VulkanGraphics, VulkanGraphics>;

/// Which renderer backend to use for the KMS/DRM path, chosen once at startup.
///
/// Kept as a plain enum (rather than a Cargo feature) so both backends are always compiled in and
/// selectable at runtime via `COSMIC_RENDERER`. `KmsApi` mirrors this split for the two concrete
/// `GpuManager` types that actually need to name one backend or the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KmsBackendKind {
    Gles,
    Vulkan,
}

impl KmsBackendKind {
    /// Reads `COSMIC_RENDERER` (`gles` or `vulkan`, case-insensitive). Defaults to `Gles`.
    ///
    /// Every construction site for a per-device or per-surface renderer backend must agree on this
    /// choice, so callers should read it once per process rather than caching it independently.
    pub fn selected() -> Self {
        match std::env::var("COSMIC_RENDERER") {
            Ok(val) if val.eq_ignore_ascii_case("vulkan") => KmsBackendKind::Vulkan,
            Ok(val) if val.eq_ignore_ascii_case("gles") => KmsBackendKind::Gles,
            Ok(val) => {
                tracing::warn!(
                    value = %val,
                    "Unrecognized COSMIC_RENDERER value, expected \"gles\" or \"vulkan\"; defaulting to gles"
                );
                KmsBackendKind::Gles
            }
            Err(_) => KmsBackendKind::Gles,
        }
    }
}

/// Runtime-selected replacement for the old `KmsGraphics` compile-time type alias.
///
/// Wraps whichever [`GpuManager`] backend [`KmsBackendKind::selected`] picked. Held by both
/// `KmsState` (device/node management) and `SurfaceThreadState` (per-output rendering) - both must
/// be constructed with the same [`KmsBackendKind`].
#[derive(Debug)]
pub enum KmsApi {
    Gles(GpuManager<GlesGraphics>),
    Vulkan(GpuManager<VulkanGraphics>),
}

impl KmsApi {
    pub fn new(kind: KmsBackendKind) -> anyhow::Result<Self> {
        match kind {
            KmsBackendKind::Gles => Ok(KmsApi::Gles(
                GpuManager::new(gles::GbmGlowBackend::new())
                    .context("Failed to initialize GLES gpu backend")?,
            )),
            KmsBackendKind::Vulkan => {
                // Default loader enumerates every ICD, including NVIDIA. That is required
                // for HDMI on the dGPU. If RM is still in kgspInitRm, this ioctl can D-state
                // the compositor; skip the dGPU with COSMIC_DRM_BLOCK_DEVICES=0x10de:0x249d
                // rather than filtering ICDs (that would also lose NVIDIA scanout).
                let instance = Instance::new(Version::VERSION_1_3, None)
                    .context("Failed to create Vulkan instance")?;
                Ok(KmsApi::Vulkan(
                    GpuManager::new(VulkanGbmBackend::new(instance).with_wayland_linux_dmabuf_interop(true))
                        .context("Failed to initialize Vulkan gpu backend")?,
                ))
            }
        }
    }

    pub fn kind(&self) -> KmsBackendKind {
        match self {
            KmsApi::Gles(_) => KmsBackendKind::Gles,
            KmsApi::Vulkan(_) => KmsBackendKind::Vulkan,
        }
    }


    /// Drops any renderer state held for `node` (e.g. on device removal or GPU loss).
    pub fn remove_node(&mut self, node: &smithay::backend::drm::DrmNode) {
        match self {
            KmsApi::Gles(api) => {
                api.as_mut().remove_node(node);
            }
            KmsApi::Vulkan(api) => {
                api.as_mut().remove_node(node);
            }
        }
    }

    /// Triggers `GpuManager`'s lazy device (re-)enumeration for `node`, discarding the renderer.
    /// Used after a topology change where nothing needs to render immediately, but the device list
    /// must be refreshed before the next real acquisition.
    pub fn trigger_enumeration(&mut self, node: &smithay::backend::drm::DrmNode) {
        match self {
            KmsApi::Gles(api) => {
                let _ = api.single_renderer(node);
            }
            KmsApi::Vulkan(api) => {
                let _ = api.single_renderer(node);
            }
        }
    }

    /// Triggers `GpuManager`'s lazy device (re-)enumeration, discarding the device list.
    pub fn trigger_devices_enumeration(&mut self) {
        match self {
            KmsApi::Gles(api) => {
                let _ = api.devices();
            }
            KmsApi::Vulkan(api) => {
                let _ = api.devices();
            }
        }
    }

    /// Runs `GpuManager::devices_mut().cleanup_texture_cache()` for every known device.
    pub fn cleanup_texture_caches(&mut self) -> anyhow::Result<()> {
        match self {
            KmsApi::Gles(api) => {
                for device in api.devices_mut()? {
                    device.renderer_mut().cleanup_texture_cache()?;
                }
            }
            KmsApi::Vulkan(api) => {
                for device in api.devices_mut()? {
                    device.renderer_mut().cleanup_texture_cache()?;
                }
            }
        }
        Ok(())
    }
}
