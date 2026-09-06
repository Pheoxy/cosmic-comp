// SPDX-License-Identifier: GPL-3.0-only

//! Nested Vulkan window present path.
//!
//! GLES winit stays in [`super::winit`]. This module is the counterpart of
//! smithay's `WinitVulkanGraphicsBackend`: Vulkan WSI, `Transform::Normal`,
//! linux-dmabuf via `wayland_sampled_dmabuf_formats` / `sampled_dmabuf_import_supported`,
//! no `wl_drm`.

use crate::{
    backend::render,
    config::ScreenFilter,
    state::{BackendData, Common, State},
};
use anyhow::{Context, Result, anyhow};
use cosmic_comp_config::output::comp::{OutputConfig, TransformDef};
use smithay::{
    backend::{
        renderer::{
            damage::{OutputDamageTracker, RenderOutputResult},
            vulkan::VulkanRenderer,
        },
        winit::{self, WinitVulkanGraphicsBackend},
    },
    output::{Mode, Output, PhysicalProperties, Scale, Subpixel},
    reexports::{
        calloop::{EventLoop, ping},
        wayland_protocols::wp::presentation_time::server::wp_presentation_feedback,
        wayland_server::DisplayHandle,
        winit::{dpi::LogicalSize, event_loop::pump_events::PumpStatus, window::WindowAttributes},
    },
    utils::Transform,
    wayland::{dmabuf::DmabufFeedbackBuilder, presentation::Refresh},
};
use std::{cell::RefCell, time::Duration};
use tracing::{error, info, warn};

use super::render::{CursorMode, ScreenFilterStorage};

#[derive(Debug)]
pub struct WinitVulkanState {
    pub backend: WinitVulkanGraphicsBackend,
    pub(crate) output: Output,
    damage_tracker: OutputDamageTracker,
    screen_filter_state: ScreenFilterStorage,
}

impl WinitVulkanState {
    #[profiling::function]
    pub fn render_output(&mut self, state: &mut Common) -> Result<()> {
        let age = self.backend.buffer_age().unwrap_or(0);
        let (renderer, mut fb) = self
            .backend
            .bind()
            .with_context(|| "Failed to bind Vulkan swapchain image")?;
        match render::render_output(
            None,
            renderer,
            &mut fb,
            &mut self.damage_tracker,
            age,
            &state.shell,
            state.clock.now(),
            &self.output,
            CursorMode::NotDefault,
            &mut self.screen_filter_state,
            &state.event_loop_handle,
        ) {
            Ok(RenderOutputResult { damage, states, .. }) => {
                std::mem::drop(fb);
                self.backend
                    .submit(damage.map(|x| x.as_slice()))
                    .with_context(|| "Failed to submit Vulkan swapchain image")?;
                state.send_frames(&self.output, None);
                state.update_primary_output(&self.output, &states);
                state.send_dmabuf_feedback(&self.output, &states, |_| None);
                if damage.is_some() {
                    let mut output_presentation_feedback = state
                        .shell
                        .read()
                        .take_presentation_feedback(&self.output, &states);
                    output_presentation_feedback.presented(
                        state.clock.now(),
                        self.output
                            .current_mode()
                            .map(|mode| {
                                Refresh::Fixed(Duration::from_secs_f64(
                                    1_000.0 / mode.refresh as f64,
                                ))
                            })
                            .unwrap_or(Refresh::Unknown),
                        0,
                        wp_presentation_feedback::Kind::Vsync,
                    );
                }
            }
            Err(err) => {
                anyhow::bail!("Rendering failed: {}", err);
            }
        };

        Ok(())
    }

    pub fn all_outputs(&self) -> Vec<Output> {
        vec![self.output.clone()]
    }

    pub fn apply_config_for_outputs(&mut self, test_only: bool) -> Result<(), anyhow::Error> {
        let size = self.backend.window_size();
        let mut config = self
            .output
            .user_data()
            .get::<RefCell<OutputConfig>>()
            .unwrap()
            .borrow_mut();
        if config.mode.0 != (size.w, size.h) {
            if !test_only {
                config.mode = ((size.w, size.h), None);
            }
            Err(anyhow::anyhow!("Cannot set window size"))
        } else {
            Ok(())
        }
    }

    pub fn update_screen_filter(&mut self, screen_filter: &ScreenFilter) -> Result<()> {
        self.screen_filter_state.filter = screen_filter.clone();
        Ok(())
    }
}

pub fn init_backend(
    dh: &DisplayHandle,
    event_loop: &mut EventLoop<State>,
    state: &mut State,
) -> Result<()> {
    let (mut backend, mut input) = winit::init_vulkan_from_attributes(
        WindowAttributes::default()
            .with_surface_size(LogicalSize::new(1280.0, 800.0))
            .with_title("COSMIC (Vulkan)")
            .with_visible(true),
    )
    .map_err(|e| anyhow!("Failed to initialize Vulkan winit backend: {e:?}"))?;

    backend.renderer().set_wayland_linux_dmabuf_interop(true);

    init_linux_dmabuf(dh, state, backend.renderer())?;

    let name = "WINIT-VULKAN-0".to_string();
    let size = backend.window_size();
    let props = PhysicalProperties {
        size: (0, 0).into(),
        subpixel: Subpixel::Unknown,
        make: "COSMIC".to_string(),
        model: name.clone(),
        serial_number: "Unknown".to_string(),
    };
    let mode = Mode {
        size: (size.w, size.h).into(),
        refresh: 60_000,
    };
    let output = Output::new(name, props);
    output.add_mode(mode);
    output.set_preferred(mode);
    // Vulkan WSI is top-left, matching Wayland. Do not copy GLES Flipped180.
    output.change_current_state(
        Some(mode),
        Some(Transform::Normal),
        Some(Scale::Integer(1)),
        Some((0, 0).into()),
    );
    output.user_data().insert_if_missing(|| {
        RefCell::new(OutputConfig {
            mode: ((size.w, size.h), None),
            transform: TransformDef::Normal,
            ..Default::default()
        })
    });

    let (event_ping, event_source) =
        ping::make_ping().with_context(|| "Failed to init eventloop timer for winit-vulkan")?;
    let (render_ping, render_source) =
        ping::make_ping().with_context(|| "Failed to init eventloop timer for winit-vulkan")?;
    let event_ping_handle = event_ping.clone();
    let render_ping_handle = render_ping.clone();
    let mut token = Some(
        event_loop
            .handle()
            .insert_source(render_source, move |_, _, state| {
                if let Err(err) = state
                    .backend
                    .winit_vulkan()
                    .render_output(&mut state.common)
                {
                    error!(?err, "Failed to render Vulkan winit frame.");
                    render_ping.ping();
                }
                profiling::finish_frame!();
            })
            .map_err(|_| anyhow::anyhow!("Failed to init eventloop timer for winit-vulkan"))?,
    );
    let event_loop_handle = event_loop.handle();
    event_loop
        .handle()
        .insert_source(event_source, move |_, _, state| {
            match input
                .dispatch_new_events(|event| state.process_winit_event(event, &render_ping_handle))
            {
                PumpStatus::Continue => {
                    event_ping_handle.ping();
                    render_ping_handle.ping();
                }
                PumpStatus::Exit(_) => {
                    let output = state.backend.winit_vulkan().output.clone();
                    state.common.remove_output(&output);
                    if let Some(token) = token.take() {
                        event_loop_handle.remove(token);
                    }
                }
            };
        })
        .map_err(|_| anyhow::anyhow!("Failed to init eventloop timer for winit-vulkan"))?;
    event_ping.ping();

    state.backend = BackendData::WinitVulkan(WinitVulkanState {
        backend,
        output: output.clone(),
        damage_tracker: OutputDamageTracker::from_output(&output),
        screen_filter_state: ScreenFilterStorage::default(),
    });

    state
        .common
        .output_configuration_state
        .add_heads(std::iter::once(&output));
    {
        state.common.add_output(&output);
        if let Err(err) = state.common.config.read_outputs(
            &mut state.common.output_configuration_state,
            &mut state.backend,
            &state.common.shell,
            &state.common.event_loop_handle,
            &mut state.common.workspace_state.update(),
            &state.common.xdg_activation_state,
            state.common.startup_done.clone(),
            &state.common.clock,
        ) {
            error!("Unrecoverable output config error: {}", err);
        }
        state.common.refresh();
    }

    if state.common.with_xwayland {
        state.launch_xwayland(None);
    } else {
        state.notify_ready();
    }

    info!("Vulkan winit backend initialized (nested window present path).");
    Ok(())
}

fn init_linux_dmabuf(
    dh: &DisplayHandle,
    state: &mut State,
    renderer: &mut VulkanRenderer,
) -> Result<()> {
    let dmabuf_formats = renderer.wayland_sampled_dmabuf_formats();
    let main_device = renderer
        .physical_device()
        .and_then(|device| {
            device
                .render_node()
                .ok()
                .flatten()
                .or_else(|| device.primary_node().ok().flatten())
        })
        .map(|node| node.dev_id())
        .unwrap_or(0);

    match DmabufFeedbackBuilder::new(main_device, dmabuf_formats.clone()).build() {
        Ok(feedback) => {
            let _global = state
                .common
                .dmabuf_state
                .create_global_with_default_feedback::<State>(dh, &feedback);
            info!("linux-dmabuf advertised from Vulkan sampled formats.");
        }
        Err(err) => {
            warn!(
                ?err,
                "Failed to build dmabuf feedback; advertising sampled formats without default feedback."
            );
            let _global = state
                .common
                .dmabuf_state
                .create_global::<State>(dh, dmabuf_formats);
        }
    }
    Ok(())
}
