// SPDX-License-Identifier: GPL-3.0-only

use calloop::LoopHandle;
use smithay::{
    backend::{
        allocator::{Buffer, Fourcc},
        renderer::{
            Bind, BufferType, Color32F, ExportMem, ImportAll, ImportMem, Offscreen,
            buffer_dimensions, buffer_type,
            damage::{Error as DTError, OutputDamageTracker, RenderOutputResult},
            element::{
                RenderElement, UnderlyingStorage,
                utils::{Relocate, RelocateRenderElement},
            },
            gles::GlesError,
            sync::SyncPoint,
            utils::with_renderer_surface_state,
        },
    },
    desktop::space::SpaceElement,
    input::Seat,
    output::{Output, OutputNoMode},
    reexports::wayland_server::protocol::wl_buffer::WlBuffer,
    utils::{
        Buffer as BufferCoords, IsAlive, Logical, Physical, Point, Rectangle, Scale, Size,
        Transform,
    },
    wayland::{
        dmabuf::get_dmabuf,
        image_copy_capture::{CaptureFailureReason, CursorSessionRef, Frame, SessionRef},
        seat::WaylandFocus,
        shm::{shm_format_to_fourcc, with_buffer_contents, with_buffer_contents_mut},
    },
};
use std::{cell::RefCell, rc::Rc};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::warn;

use smithay::backend::allocator::format::get_transparent;
use smithay::backend::renderer::gles::GlesRenderbuffer;

use crate::{
    backend::render::{
        CursorMode, ElementFilter, RendererRef,
        cursor::{self, CursorRenderElement},
        element::{AsGlowRenderer, CosmicElement, DamageElement},
        render_workspace,
        wayland::SurfaceRenderElement,
    },
    shell::{CosmicMappedRenderElement, CosmicSurface, WorkspaceRenderElement},
    state::{Common, KmsNodes, State},
    utils::prelude::{PointExt, PointGlobalExt, RectExt, RectLocalExt, SeatExt},
    wayland::{
        handlers::image_copy_capture::{
            SessionData, SessionOffscreen, SessionUserData, constraints_for_output,
            constraints_for_toplevel,
        },
        protocols::workspace::WorkspaceHandle,
    },
};

use super::{
    super::data_device::get_dnd_icon, cursor_capture_constraints, user_data::SessionHolder,
};

use smithay::backend::renderer::{Renderer, vulkan::VulkanRenderTarget};

pub fn render_element_buffers<R, E>(
    renderer: &mut R,
    elements: &[E],
) -> Vec<smithay::backend::renderer::utils::Buffer>
where
    R: AsGlowRenderer,
    E: RenderElement<R>,
{
    elements
        .iter()
        .filter_map(|elem| match elem.underlying_storage(renderer) {
            Some(UnderlyingStorage::Wayland(buffer)) => Some(buffer.clone()),
            Some(UnderlyingStorage::Memory(_)) | None => None,
        })
        .collect()
}

pub struct PendingImageCopyData {
    pub frame: Frame,
    pub damage: Vec<smithay::utils::Rectangle<i32, BufferCoords>>,
    pub sync: SyncPoint,
    // Hold reference so `wl_buffer` isn't released and sync point isn't signaled
    // until image copy completes.
    _buffers: Vec<smithay::backend::renderer::utils::Buffer>,
    // The GPU->CPU readback for an SHM capture, still to be done once `sync` is reached. `None`
    // for a direct dmabuf-target render, which needs no separate copy step. Only ever `Some` on
    // the Vulkan path (see `SessionUserData::copy_pending`); GLES stays eager/synchronous.
    shm_copy: Option<PendingShmCopy>,
}

/// The session types `submit_buffer`/`render_session` are called for. Workspace and toplevel
/// captures use [`SessionRef`]; cursor captures use the separate [`CursorSessionRef`] - both
/// expose an equivalent `user_data()`/`SessionData`, just as different Rust types.
#[derive(Clone)]
pub enum CaptureSessionRef {
    Session(SessionRef),
    Cursor(CursorSessionRef),
}

impl CaptureSessionRef {
    fn session_data(&self) -> Option<&SessionData> {
        match self {
            Self::Session(session) => session.user_data().get::<SessionData>(),
            Self::Cursor(session) => session.user_data().get::<SessionData>(),
        }
    }
}

/// Work still needed to complete an SHM capture: `submit_buffer` renders into the session's
/// offscreen target but does not wait for or copy out of it (that would block the compositor's
/// single event-loop thread on every capture poll). This carries what a later, fence-driven
/// completion needs to redo the same bind and finish the copy: the exact renderer selection used
/// for the render (so the readback happens on the same device), and the session/destination
/// buffer to copy into.
struct PendingShmCopy {
    session: CaptureSessionRef,
    nodes: Option<KmsNodes>,
    buffer: WlBuffer,
    buffer_size: Size<i32, BufferCoords>,
}

/// Finish a deferred SHM capture: re-bind the session's offscreen target on the same renderer
/// used to render it, copy out, and write into the client's buffer.
///
/// Must only be called once `sync` for the render has already been reached - this does not wait.
fn finish_shm_copy(state: &mut State, shm_copy: &PendingShmCopy) -> Result<(), CaptureFailureReason> {
    let Some(session_data) = shm_copy.session.session_data() else {
        return Err(CaptureFailureReason::Unknown);
    };
    let mut session_user_data = session_data.lock().unwrap();
    session_user_data.copy_pending = false;
    let Some(target) = session_user_data.offscreen.as_mut() else {
        return Err(CaptureFailureReason::Unknown);
    };

    let nodes = shm_copy.nodes;
    let renderer = state
        .backend
        .offscreen_renderer(move |_| nodes)
        .map_err(|_| CaptureFailureReason::Unknown)?;

    // Each `RendererRef` arm is a different concrete renderer type with its own `Error`, so
    // normalize to `CaptureFailureReason` inside the match instead of returning `R::Error` from
    // it - the two arms would otherwise be different, non-unifiable `Result` types.
    let result: Result<(), CaptureFailureReason> = match renderer {
        RendererRef::Glow(r) => {
            copy_shm_target(r, target, &shm_copy.buffer, shm_copy.buffer_size).map_err(|err| {
                warn!(?err, "Failed to finish deferred SHM capture");
                CaptureFailureReason::Unknown
            })
        }
        RendererRef::MultiGles(mut r) => {
            copy_shm_target(&mut r, target, &shm_copy.buffer, shm_copy.buffer_size).map_err(|err| {
                warn!(?err, "Failed to finish deferred SHM capture");
                CaptureFailureReason::Unknown
            })
        }
        RendererRef::MultiVulkan(mut r) => {
            copy_shm_target(&mut r, target, &shm_copy.buffer, shm_copy.buffer_size).map_err(|err| {
                warn!(?err, "Failed to finish deferred SHM capture");
                CaptureFailureReason::Unknown
            })
        }
    };

    // `session_user_data` still borrows `target`; drop it explicitly so the borrow ends here,
    // not just at end of scope, in case a future change adds code after this point that needs
    // the session lock again.
    drop(session_user_data);

    result
}

/// Re-bind `target` on `renderer`, copy it out, and memcpy the result into `buffer`.
fn copy_shm_target<R>(
    renderer: &mut R,
    target: &mut SessionOffscreen,
    buffer: &WlBuffer,
    buffer_size: Size<i32, BufferCoords>,
) -> Result<(), R::Error>
where
    R: CaptureRenderer + ExportMem,
{
    let fb = renderer.bind_shm_offscreen(target)?;
    with_buffer_contents_mut(buffer, |ptr, len, data| {
        let offset = data.offset;
        let width = data.width;
        let height = data.height;
        let stride = data.stride;
        let format = shm_format_to_fourcc(data.format)
            .expect("We should be able to convert all hardcoded shm screencopy formats");
        let pixelsize = 4i32;
        assert!((offset + (height - 1) * stride + width * pixelsize) as usize <= len);

        let mapping = renderer.copy_framebuffer(&fb, Rectangle::from_size(buffer_size), format)?;
        let gl_data = renderer.map_texture(&mapping)?;
        assert!((width * height * pixelsize) as usize <= gl_data.len());

        for i in 0..height {
            unsafe {
                std::ptr::copy_nonoverlapping::<u8>(
                    gl_data.as_ptr().offset((width * pixelsize * i) as isize),
                    ptr.offset((offset + stride * i) as isize),
                    (width * pixelsize) as usize,
                );
            }
        }

        Ok(())
    })
    .map_err(|err| R::from_gles_error(GlesError::BufferAccessError(err)))
    .and_then(|x: Result<(), R::Error>| x)?;
    drop(fb);
    Ok(())
}

/// Bundle handed between the fd/timeout/idle sources racing to complete one deferred SHM copy.
/// Whichever fires first `take()`s it; the rest find `None` and no-op.
type PendingShmCopyBundle = (
    Frame,
    Vec<smithay::utils::Rectangle<i32, BufferCoords>>,
    PendingShmCopy,
    Vec<smithay::backend::renderer::utils::Buffer>,
);

fn run_pending_shm_copy(
    bundle: &Rc<RefCell<Option<PendingShmCopyBundle>>>,
    state: &mut State,
    transform: Transform,
    presented: Duration,
    timed_out: bool,
) {
    let Some((frame, damage, shm_copy, _buffers)) = bundle.borrow_mut().take() else {
        return;
    };
    if timed_out {
        warn!("Timed out waiting on SHM capture render fence");
        if let Some(session_data) = shm_copy.session.session_data() {
            session_data.lock().unwrap().copy_pending = false;
        }
        frame.fail(CaptureFailureReason::Unknown);
        return;
    }
    match finish_shm_copy(state, &shm_copy) {
        Ok(()) => frame.success(transform, damage, presented),
        Err(reason) => frame.fail(reason),
    }
    // `_buffers` (the render's source wl_buffers) drops here, after the copy has actually read
    // from the offscreen target they were drawn into.
}

impl PendingImageCopyData {
    /// Send `success` to image copy frame, once sync point is reached.
    ///
    /// Generic over `LoopData` because callers run on different event loops: workspace/window/
    /// cursor captures run on the compositor's main `State` loop, output-based captures run on a
    /// per-surface `SurfaceThreadState` loop. `self.shm_copy` (vulkan only) is only ever `Some`
    /// when `submit_buffer` was told (`defer_shm_copy: true`) that this call site uses `State` -
    /// every such call site in this codebase does - so the downcast below always succeeds in
    /// practice; it exists to fail closed instead of panicking if that invariant is ever violated.
    pub fn send_success_when_ready<LoopData: 'static>(
        self,
        transform: Transform,
        loop_handle: &LoopHandle<'static, LoopData>,
        presented: impl Into<Duration>,
    ) {
        let presented = presented.into();

        if self.shm_copy.is_some() {
            let loop_handle_any: &dyn std::any::Any = loop_handle;
            return match loop_handle_any.downcast_ref::<LoopHandle<'static, State>>() {
                Some(state_loop_handle) => self.finish_deferred(transform, state_loop_handle, presented),
                None => {
                    warn!(
                        "SHM capture was deferred for a non-`State` event loop; failing instead of \
                         skipping the copy"
                    );
                    self.frame.fail(CaptureFailureReason::Unknown);
                }
            };
        }

        if self.sync.is_reached() {
            self.frame.success(transform, self.damage, presented);
        } else if let Some(fence_fd) = self.sync.export() {
            let source = calloop::generic::Generic::new(
                fence_fd,
                calloop::Interest::READ,
                calloop::Mode::OneShot,
            );
            let mut data = Some(self);
            loop_handle
                .insert_source(source, move |_, _, _: &mut LoopData| {
                    let data = data.take().unwrap();
                    data.frame.success(transform, data.damage, presented);
                    Ok(calloop::PostAction::Remove)
                })
                .expect("Failed to wait on sync point");
        } else {
            // Should be able to export fence; but otherwise wait
            let _ = self.sync.wait();
            self.frame.success(transform, self.damage, presented);
        }
    }

    /// Complete an SHM capture once its render's sync point is reached, or fail it if that never
    /// happens within a safety-net timeout (a hung/lost-device fence would otherwise leave the fd
    /// source registered forever with the client's capture request never answered).
    ///
    /// `finish_shm_copy` needs `&mut State` (to re-derive the renderer), which this method does
    /// not have - only calloop callbacks do - so even the already-reached case is finished via
    /// `insert_idle` rather than called inline.
    fn finish_deferred(
        self,
        transform: Transform,
        loop_handle: &LoopHandle<'static, State>,
        presented: Duration,
    ) {
        let PendingImageCopyData {
            frame,
            damage,
            sync,
            _buffers,
            shm_copy,
        } = self;
        let shm_copy = shm_copy.expect("checked by caller");

        let bundle: Rc<RefCell<Option<PendingShmCopyBundle>>> =
            Rc::new(RefCell::new(Some((frame, damage, shm_copy, _buffers))));

        if sync.is_reached() {
            let bundle = bundle.clone();
            let _ = loop_handle
                .insert_idle(move |state| run_pending_shm_copy(&bundle, state, transform, presented, false));
            return;
        }

        let Some(fence_fd) = sync.export() else {
            // No exportable fence: fall back to a blocking wait, same as the non-SHM path.
            let _ = sync.wait();
            let bundle = bundle.clone();
            let _ = loop_handle
                .insert_idle(move |state| run_pending_shm_copy(&bundle, state, transform, presented, false));
            return;
        };

        let source =
            calloop::generic::Generic::new(fence_fd, calloop::Interest::READ, calloop::Mode::OneShot);
        {
            let bundle = bundle.clone();
            loop_handle
                .insert_source(source, move |_, _, state: &mut State| {
                    run_pending_shm_copy(&bundle, state, transform, presented, false);
                    Ok(calloop::PostAction::Remove)
                })
                .expect("Failed to wait on sync point");
        }

        // Safety-net timeout: if the fence never signals (device lost, driver bug), fail the
        // frame instead of leaving the client's capture request hanging forever.
        let timer = calloop::timer::Timer::from_duration(Duration::from_secs(2));
        loop_handle
            .insert_source(timer, move |_, _, state: &mut State| {
                run_pending_shm_copy(&bundle, state, transform, presented, true);
                calloop::timer::TimeoutAction::Drop
            })
            .expect("Failed to arm SHM capture timeout");
    }
}

pub fn submit_buffer<R>(
    frame: Frame,
    renderer: &mut R,
    offscreen: Option<&mut R::Framebuffer<'_>>,
    transform: Transform,
    damage: Option<&[Rectangle<i32, Physical>]>,
    sync: SyncPoint,
    buffers: Vec<smithay::backend::renderer::utils::Buffer>,
    session_ref: CaptureSessionRef,
    nodes: Option<KmsNodes>,
    // Only the workspace/window/cursor capture paths run on the `State`-based event loop the
    // deferred completion (`send_success_when_ready`'s vulkan branch) needs; output-based capture
    // runs on a per-surface `SurfaceThreadState` loop that can't provide it, so that caller passes
    // `false` here and keeps the original eager (blocking) copy for the vulkan SHM case too.
    defer_shm_copy: bool,
) -> Result<Option<PendingImageCopyData>, R::Error>
where
    R: ExportMem + AsGlowRenderer,
{
    let _ = &session_ref;
    let Some(damage) = damage else {
        frame.success(
            transform,
            None,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO),
        );
        return Ok(None);
    };

    let buffer = frame.buffer();
    let buffer_size = buffer_dimensions(&buffer).unwrap();

    let mut shm_copy = None;
    let mut sync = sync;

    // GLES's copy is already synchronous/eager and never stalls the compositor the way Vulkan's
    // explicit-fence readback could, so it always takes the eager path below regardless of
    // `defer_shm_copy` - matching this renderer's behavior before runtime selection existed.
    let is_gles = renderer.glow_renderer().is_some();

    if let Some(fb) = offscreen {
        assert!(matches!(buffer_type(&buffer), Some(BufferType::Shm)));
        if defer_shm_copy && !is_gles {
            // Validate the destination buffer now so a protocol violation fails fast; the actual
            // GPU->CPU copy is deferred until `sync` is reached (`send_success_when_ready`)
            // instead of blocking this render dispatch on the compositor's event-loop thread.
            if let Err(err) = with_buffer_contents(&buffer, |_, len, data| {
                let offset = data.offset;
                let width = data.width;
                let height = data.height;
                let stride = data.stride;
                let pixelsize = 4i32;
                assert!((offset + (height - 1) * stride + width * pixelsize) as usize <= len);
                let _ = shm_format_to_fourcc(data.format)
                    .expect("We should be able to convert all hardcoded shm screencopy formats");
            }) {
                frame.fail(CaptureFailureReason::Unknown);
                return Err(R::from_gles_error(GlesError::BufferAccessError(err)));
            }
            shm_copy = Some(PendingShmCopy {
                session: session_ref.clone(),
                nodes,
                buffer: buffer.clone(),
                buffer_size,
            });
        } else if let Err(err) = with_buffer_contents_mut(&buffer, |ptr, len, data| {
            let offset = data.offset;
            let width = data.width;
            let height = data.height;
            let stride = data.stride;
            let format = shm_format_to_fourcc(data.format)
                .expect("We should be able to convert all hardcoded shm screencopy formats");
            let pixelsize = 4i32;
            assert!((offset + (height - 1) * stride + width * pixelsize) as usize <= len);

            renderer.wait(&sync)?;

            // `get_transparent` is a GLES-specific SHM readback format adjustment; harmless no-op
            // for a format it doesn't recognize, so applying it unconditionally is safe even
            // though the Vulkan path only ever reaches here when `defer_shm_copy` is false.
            let copy_format = get_transparent(format).unwrap_or(format);
            let mapping =
                renderer.copy_framebuffer(fb, Rectangle::from_size(buffer_size), copy_format)?;
            let gl_data = renderer.map_texture(&mapping)?;
            assert!((width * height * pixelsize) as usize <= gl_data.len());

            for i in 0..height {
                unsafe {
                    std::ptr::copy_nonoverlapping::<u8>(
                        gl_data.as_ptr().offset((width * pixelsize * i) as isize),
                        ptr.offset((offset + stride * i) as isize),
                        (width * pixelsize) as usize,
                    );
                }
            }

            sync = SyncPoint::signaled();

            Ok(())
        })
        .map_err(|err| R::from_gles_error(GlesError::BufferAccessError(err)))
        .and_then(|x| x)
        {
            frame.fail(CaptureFailureReason::Unknown);
            return Err(err);
        }
    }

    Ok(Some(PendingImageCopyData {
        frame,
        damage: damage
            .iter()
            .map(|rect| {
                let logical = rect.to_logical(1);
                logical.to_buffer(1, transform.invert(), &buffer_size.to_logical(1, transform))
            })
            .collect(),
        sync,
        _buffers: buffers,
        shm_copy,
    }))
}

/// SHM capture offscreen, real per-backend behavior always compiled in (runtime-selected via
/// `KmsApi`/`COSMIC_RENDERER`, not the `renderer_vulkan` Cargo feature). GLES uses Glow
/// `Offscreen<GlesRenderbuffer>` via [`AsGlowRenderer`]; Vulkan uses smithay [`Offscreen`] /
/// [`Bind`] of [`VulkanRenderTarget`]. Each impl wraps its result in the matching
/// [`SessionOffscreen`] variant so callers don't need to know which backend produced it.
pub trait CaptureRenderer: AsGlowRenderer {
    fn create_shm_offscreen(
        &mut self,
        format: smithay::backend::allocator::Fourcc,
        size: smithay::utils::Size<i32, smithay::utils::Buffer>,
    ) -> Result<SessionOffscreen, Self::Error>;

    fn bind_shm_offscreen<'a>(
        &mut self,
        target: &'a mut SessionOffscreen,
    ) -> Result<Self::Framebuffer<'a>, Self::Error>;
}

impl CaptureRenderer for crate::backend::kms::render::GlesMultiRenderer<'_> {
    fn create_shm_offscreen(
        &mut self,
        format: smithay::backend::allocator::Fourcc,
        size: smithay::utils::Size<i32, smithay::utils::Buffer>,
    ) -> Result<SessionOffscreen, Self::Error> {
        let context_id = self.glow_renderer().unwrap().context_id();
        let renderbuffer = Offscreen::<GlesRenderbuffer>::create_buffer(self, format, size)?;
        Ok(SessionOffscreen::Gles(context_id, renderbuffer))
    }
    fn bind_shm_offscreen<'a>(
        &mut self,
        target: &'a mut SessionOffscreen,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        match target {
            SessionOffscreen::Gles(_, renderbuffer) => self.bind(renderbuffer),
            SessionOffscreen::Vulkan(_) => {
                unreachable!("SessionOffscreen::Vulkan bound on a GLES renderer")
            }
        }
    }
}

impl CaptureRenderer for crate::backend::kms::render::VulkanMultiRenderer<'_> {
    fn create_shm_offscreen(
        &mut self,
        format: smithay::backend::allocator::Fourcc,
        size: smithay::utils::Size<i32, smithay::utils::Buffer>,
    ) -> Result<SessionOffscreen, Self::Error> {
        Offscreen::<VulkanRenderTarget<'static>>::create_buffer(self, format, size)
            .map(SessionOffscreen::Vulkan)
    }
    fn bind_shm_offscreen<'a>(
        &mut self,
        target: &'a mut SessionOffscreen,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        match target {
            SessionOffscreen::Vulkan(target) => self.bind(target),
            SessionOffscreen::Gles(..) => {
                unreachable!("SessionOffscreen::Gles bound on a Vulkan renderer")
            }
        }
    }
}

impl CaptureRenderer for smithay::backend::renderer::vulkan::VulkanRenderer {
    fn create_shm_offscreen(
        &mut self,
        format: smithay::backend::allocator::Fourcc,
        size: smithay::utils::Size<i32, smithay::utils::Buffer>,
    ) -> Result<SessionOffscreen, Self::Error> {
        Offscreen::<VulkanRenderTarget<'static>>::create_buffer(self, format, size)
            .map(SessionOffscreen::Vulkan)
    }
    fn bind_shm_offscreen<'a>(
        &mut self,
        target: &'a mut SessionOffscreen,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        match target {
            SessionOffscreen::Vulkan(target) => self.bind(target),
            SessionOffscreen::Gles(..) => {
                unreachable!("SessionOffscreen::Gles bound on a Vulkan renderer")
            }
        }
    }
}

impl CaptureRenderer for smithay::backend::renderer::glow::GlowRenderer {
    fn create_shm_offscreen(
        &mut self,
        format: smithay::backend::allocator::Fourcc,
        size: smithay::utils::Size<i32, smithay::utils::Buffer>,
    ) -> Result<SessionOffscreen, Self::Error> {
        let context_id = self.context_id();
        let renderbuffer = Offscreen::<GlesRenderbuffer>::create_buffer(self, format, size)?;
        Ok(SessionOffscreen::Gles(context_id, renderbuffer))
    }
    fn bind_shm_offscreen<'a>(
        &mut self,
        target: &'a mut SessionOffscreen,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        match target {
            SessionOffscreen::Gles(_, renderbuffer) => self.bind(renderbuffer),
            SessionOffscreen::Vulkan(_) => {
                unreachable!("SessionOffscreen::Vulkan bound on a GLES renderer")
            }
        }
    }
}

pub fn render_session<F, R>(
    renderer: &mut R,
    session: &SessionData,
    session_ref: CaptureSessionRef,
    nodes: Option<KmsNodes>,
    // See `submit_buffer`'s doc on this parameter.
    defer_shm_copy: bool,
    frame: Frame,
    transform: Transform,
    render_fn: F,
) -> Result<Option<PendingImageCopyData>, DTError<R::Error>>
where
    R: CaptureRenderer,
    F: for<'d> FnOnce(
        &WlBuffer,
        &mut R,
        Option<&mut R::Framebuffer<'_>>,
        &'d mut OutputDamageTracker,
        usize,
        Vec<Rectangle<i32, BufferCoords>>,
    ) -> Result<
        (
            RenderOutputResult<'d>,
            Vec<smithay::backend::renderer::utils::Buffer>,
        ),
        DTError<R::Error>,
    >,
{
    let mut session_user_data = session.lock().unwrap();

    // A previous capture on this session already rendered into `offscreen` and is waiting on its
    // fence before its deferred copy runs (see `PendingShmCopy`); re-rendering now would clobber
    // the contents that copy still expects. Fail this request rather than race it. Only ever set
    // on the Vulkan path; always false on GLES.
    if session_user_data.copy_pending {
        drop(session_user_data);
        frame.fail(CaptureFailureReason::Unknown);
        return Ok(None);
    }

    let buffer = frame.buffer();

    let mut age = 1;
    if matches!(buffer_type(&buffer), Some(BufferType::Shm)) {
        let size = buffer_dimensions(&buffer).ok_or(DTError::OutputNoMode(OutputNoMode))?;
        let format = with_buffer_contents(&buffer, |_, _, data| {
            shm_format_to_fourcc(data.format)
                .expect("We should be able to convert all hardcoded shm screencopy formats")
        })
        .map_err(|_| DTError::OutputNoMode(OutputNoMode))?;

        session_user_data.offscreen.take_if(|target| {
            target.size() != size
                || target.format() != Some(format)
                || target.is_stale_context(renderer)
        });

        if session_user_data.offscreen.is_none() {
            let target = renderer
                .create_shm_offscreen(format, size)
                .map_err(DTError::Rendering)?;
            session_user_data.offscreen = Some(target);
            age = 0;
        }
    } else {
        // If for some reason a capture session is used for shm, but then changes to dmabuf capture,
        // remove the offscreen buffer.
        session_user_data.offscreen = None;
    }

    let SessionUserData {
        dt,
        offscreen,
        copy_pending,
    } = &mut *session_user_data;
    let mut fb = if let Some(target) = offscreen.as_mut() {
        Some(
            renderer
                .bind_shm_offscreen(target)
                .map_err(DTError::Rendering)?,
        )
    } else {
        None
    };
    let (result, buffers) = render_fn(
        &frame.buffer(),
        renderer,
        fb.as_mut(),
        dt,
        age,
        frame.damage(),
    )?;

    let pending = submit_buffer(
        frame,
        renderer,
        fb.as_mut(),
        transform,
        result.damage.map(|x| x.as_slice()),
        result.sync,
        buffers,
        session_ref,
        nodes,
        defer_shm_copy,
    )
    .map_err(DTError::Rendering)?;

    if let Some(p) = &pending {
        if p.shm_copy.is_some() {
            *copy_pending = true;
        }
    }

    Ok(pending)
}

pub fn render_workspace_to_buffer(
    state: &mut State,
    session: &SessionRef,
    frame: Frame,
    handle: WorkspaceHandle,
) {
    let shell = state.common.shell.read();
    let Some(workspace) = shell.workspaces.space_for_handle(&handle) else {
        return;
    };

    let mut output = workspace.output().clone();
    let idx = shell.workspaces.idx_for_handle(&output, &handle).unwrap();
    std::mem::drop(shell);

    let mode = output
        .current_mode()
        .map(|mode| mode.size.to_logical(1).to_buffer(1, Transform::Normal));

    let buffer = frame.buffer();
    let buffer_size = buffer_dimensions(&buffer).unwrap();
    if mode != Some(buffer_size) {
        let Some(constraints) = constraints_for_output(&output, &mut state.backend) else {
            output.remove_session(session);
            return;
        };
        session.update_constraints(constraints);
        if let Some(data) = session.user_data().get::<SessionData>() {
            *data.lock().unwrap() = SessionUserData::new(OutputDamageTracker::from_output(&output));
        }
        frame.fail(CaptureFailureReason::BufferConstraints);
        return;
    }

    fn render_fn<'d, R>(
        buffer: &WlBuffer,
        renderer: &mut R,
        offscreen: Option<&mut R::Framebuffer<'_>>,
        dt: &'d mut OutputDamageTracker,
        age: usize,
        additional_damage: Vec<Rectangle<i32, BufferCoords>>,
        draw_cursor: bool,
        common: &mut Common,
        output: &Output,
        handle: (WorkspaceHandle, usize),
    ) -> Result<
        (
            RenderOutputResult<'d>,
            Vec<smithay::backend::renderer::utils::Buffer>,
        ),
        DTError<R::Error>,
    >
    where
        R: AsGlowRenderer,
        R::TextureId: Send + Clone + 'static,
        CosmicElement<R>: RenderElement<R>,
        CosmicMappedRenderElement<R>: RenderElement<R>,
        WorkspaceRenderElement<R>: RenderElement<R>,
    {
        let cursor_mode = if draw_cursor {
            CursorMode::All
        } else {
            CursorMode::None
        };

        let area = output
            .current_mode()
            .ok_or(DTError::OutputNoMode(OutputNoMode))
            .map(
                |mode| {
                    mode.size
                        .to_logical(1)
                        .to_buffer(1, Transform::Normal)
                        .to_f64()
                }, /* TODO: Mode is Buffer..., why is this Physical in the first place */
            )?;
        let additional_damage = (!additional_damage.is_empty()).then(|| {
            additional_damage
                .into_iter()
                .map(|rect| {
                    rect.to_f64()
                        .to_logical(
                            output.current_scale().fractional_scale(),
                            output.current_transform(),
                            &area,
                        )
                        .to_i32_round()
                })
                .collect()
        });

        let (res, elements) = if let Ok(dmabuf) = get_dmabuf(buffer) {
            let mut dmabuf = dmabuf.clone();
            let mut fb = renderer.bind(&mut dmabuf).map_err(DTError::Rendering)?;
            render_workspace(
                None,
                renderer,
                &mut fb,
                dt,
                age,
                additional_damage,
                &common.shell,
                None,
                common.clock.now(),
                output,
                None,
                handle,
                cursor_mode,
                ElementFilter::ExcludeWorkspaceOverview,
            )?
        } else {
            let target = offscreen.expect("shm buffers should have an offscreen target");
            render_workspace(
                None,
                renderer,
                target,
                dt,
                age,
                additional_damage,
                &common.shell,
                None,
                common.clock.now(),
                output,
                None,
                handle,
                cursor_mode,
                ElementFilter::ExcludeWorkspaceOverview,
            )?
        };

        let buffers = render_element_buffers(renderer, &elements);

        Ok((res, buffers))
    }

    let draw_cursor = session.draw_cursor();
    let transform = output.current_transform();
    let common = &mut state.common;

    let nodes_cell: std::cell::Cell<Option<KmsNodes>> = std::cell::Cell::new(None);
    let renderer = match state.backend.offscreen_renderer(|kms| {
        let render_node = kms
            .target_node_for_output(&output)
            .or(*kms.primary_node.read().unwrap())?;
        let target_node = get_dmabuf(&buffer)
            .ok()
            .and_then(|dma| dma.node())
            .unwrap_or(render_node);

        let buffer_format = match buffer_type(&buffer) {
            Some(BufferType::Dma) => Some(get_dmabuf(&buffer).unwrap().format().code),
            Some(BufferType::Shm) => {
                with_buffer_contents(&buffer, |_, _, data| shm_format_to_fourcc(data.format))
                    .unwrap()
            }
            _ => None,
        };

        let nodes = KmsNodes {
            render_node,
            target_node,
            copy_format: buffer_format.unwrap_or(Fourcc::Abgr8888),
        };
        nodes_cell.set(Some(nodes));
        Some(nodes)
    }) {
        Ok(renderer) => renderer,
        Err(err) => {
            warn!(?err, "Couldn't use node for screencopy");
            frame.fail(CaptureFailureReason::Unknown);
            return;
        }
    };
    let nodes = nodes_cell.get();
    let result = match renderer {
        RendererRef::Glow(renderer) => {
            match render_session(
                renderer,
                session.user_data().get::<SessionData>().unwrap(),
                CaptureSessionRef::Session(session.clone()),
                nodes,
                true,
                frame,
                transform,
                |buffer, renderer, offscreen, dt, age, additional_damage| {
                    render_fn(
                        buffer,
                        renderer,
                        offscreen,
                        dt,
                        age,
                        additional_damage,
                        draw_cursor,
                        common,
                        &output,
                        (handle, idx),
                    )
                },
            ) {
                Ok(frame) => frame,
                Err(err) => {
                    tracing::warn!(?err, "Failed to render to screencopy buffer");
                    None
                }
            }
        }
        RendererRef::MultiGles(mut renderer) => {
            match render_session(
                &mut renderer,
                session.user_data().get::<SessionData>().unwrap(),
                CaptureSessionRef::Session(session.clone()),
                nodes,
                true,
                frame,
                transform,
                |buffer, renderer, offscreen, dt, age, additional_damage| {
                    render_fn(
                        buffer,
                        renderer,
                        offscreen,
                        dt,
                        age,
                        additional_damage,
                        draw_cursor,
                        common,
                        &output,
                        (handle, idx),
                    )
                },
            ) {
                Ok(frame) => frame,
                Err(err) => {
                    tracing::warn!(?err, "Failed to render to screencopy buffer");
                    None
                }
            }
        }
        RendererRef::MultiVulkan(mut renderer) => {
            match render_session(
                &mut renderer,
                session.user_data().get::<SessionData>().unwrap(),
                CaptureSessionRef::Session(session.clone()),
                nodes,
                true,
                frame,
                transform,
                |buffer, renderer, offscreen, dt, age, additional_damage| {
                    render_fn(
                        buffer,
                        renderer,
                        offscreen,
                        dt,
                        age,
                        additional_damage,
                        draw_cursor,
                        common,
                        &output,
                        (handle, idx),
                    )
                },
            ) {
                Ok(frame) => frame,
                Err(err) => {
                    tracing::warn!(?err, "Failed to render to screencopy buffer");
                    None
                }
            }
        }
    };

    if let Some(pending_image_copy_data) = result {
        pending_image_copy_data.send_success_when_ready(
            transform,
            &common.event_loop_handle,
            common.clock.now(),
        );
    }
}

smithay::render_elements! {
    pub WindowCaptureElement<R> where R: ImportAll + ImportMem + AsGlowRenderer, R::TextureId: Send;
    WaylandElement=SurfaceRenderElement<R>,
    CursorElement=RelocateRenderElement<cursor::CursorRenderElement<R>>,
    DamageElement=DamageElement,
}

pub fn render_window_to_buffer(
    state: &mut State,
    session: &SessionRef,
    frame: Frame,
    toplevel: &CosmicSurface,
) {
    if !toplevel.alive() {
        toplevel.clone().remove_session(session);
        return;
    }

    let buffer = frame.buffer();
    let geometry = toplevel.geometry();
    let buffer_size = buffer_dimensions(&buffer).unwrap();
    if buffer_size != geometry.size.to_buffer(1, Transform::Normal) {
        let Some(constraints) = constraints_for_toplevel(toplevel, &mut state.backend) else {
            toplevel.clone().remove_session(session);
            return;
        };
        session.update_constraints(constraints);

        if let Some(data) = session.user_data().get::<SessionData>() {
            let size = geometry.size.to_physical(1);
            *data.lock().unwrap() =
                SessionUserData::new(OutputDamageTracker::new(size, 1.0, Transform::Normal));
        }
        frame.fail(CaptureFailureReason::BufferConstraints);
        return;
    }

    fn render_fn<'d, R>(
        buffer: &WlBuffer,
        renderer: &mut R,
        offscreen: Option<&mut R::Framebuffer<'_>>,
        dt: &'d mut OutputDamageTracker,
        age: usize,
        additional_damage: Vec<Rectangle<i32, BufferCoords>>,
        draw_cursor: bool,
        common: &mut Common,
        toplevel: &CosmicSurface,
        geometry: Rectangle<i32, Logical>,
    ) -> Result<
        (
            RenderOutputResult<'d>,
            Vec<smithay::backend::renderer::utils::Buffer>,
        ),
        DTError<R::Error>,
    >
    where
        R: AsGlowRenderer,
        R::TextureId: Send + Clone + 'static,
        CosmicElement<R>: RenderElement<R>,
        CosmicMappedRenderElement<R>: RenderElement<R>,
    {
        let mut elements: Vec<_> = additional_damage
            .into_iter()
            .filter_map(|rect| {
                let logical_rect = rect.to_logical(
                    1,
                    Transform::Normal,
                    &geometry.size.to_buffer(1, Transform::Normal),
                );
                logical_rect.intersection(Rectangle::from_size(geometry.size))
            })
            .map(DamageElement::new)
            .map(WindowCaptureElement::<R>::from)
            .collect();

        let shell = common.shell.read();
        let blur_strength = 9; // TODO
        let seat = shell.seats.last_active().clone();
        let pointer = seat.get_pointer().unwrap();
        let pointer_loc = pointer.current_location().to_i32_round().as_global();
        let mut location = None;
        if let Some(element) = shell.element_for_surface(toplevel)
            && pointer
                .current_focus()
                .and_then(|f| f.toplevel(&shell))
                .as_ref()
                == Some(toplevel)
            && element.has_active_window(toplevel)
            && let Some(workspace) = shell.space_for(element)
            && let Some(geometry) = workspace.element_geometry(element)
        {
            let mut surface_geo = element.active_window_geometry().as_local();
            surface_geo.loc += geometry.loc;
            let global_geo = surface_geo.to_global(workspace.output());
            if global_geo.contains(pointer_loc) {
                location = Some((pointer_loc - global_geo.loc).as_logical().to_f64());
            }
        };
        std::mem::drop(shell);

        if let Some(location) = location {
            if draw_cursor {
                cursor::draw_cursor(
                    renderer,
                    &seat,
                    location,
                    1.0.into(),
                    1.0,
                    common.clock.now(),
                    blur_strength,
                    true,
                    &mut |elem, hotspot| {
                        elements.push(WindowCaptureElement::CursorElement(
                            RelocateRenderElement::from_element(
                                elem,
                                Point::from((-hotspot.x, -hotspot.y)),
                                Relocate::Relative,
                            ),
                        ));
                    },
                );
            }

            // TODO cosmic-workspaces wants to omit, but metadata cursor capture in portal should
            // still include dnd surface in window capture buffer?
            if draw_cursor && let Some(dnd_icon) = get_dnd_icon(&seat) {
                cursor::draw_dnd_icon(
                    renderer,
                    &dnd_icon.surface,
                    (location + dnd_icon.offset.to_f64()).to_i32_round(),
                    1.0,
                    blur_strength,
                    &mut |elem| {
                        elements.push(
                            RelocateRenderElement::from_element(
                                CursorRenderElement::Surface(elem),
                                Point::new(0, 0),
                                Relocate::Relative,
                            )
                            .into(),
                        )
                    },
                );
            }
        }

        toplevel.push_render_elements(
            renderer,
            (-geometry.loc.x, -geometry.loc.y).into(),
            Scale::from(1.0),
            1.0,
            None,
            None,
            false,
            [0; 4],
            blur_strength,
            &mut |elem| elements.push(elem.into()),
            None,
        );

        let res = if let Ok(dmabuf) = get_dmabuf(buffer) {
            let mut dmabuf_clone = dmabuf.clone();
            let mut fb = renderer
                .bind(&mut dmabuf_clone)
                .map_err(DTError::Rendering)?;
            dt.render_output(renderer, &mut fb, age, &elements, Color32F::TRANSPARENT)?
        } else {
            let fb = offscreen.expect("shm buffer should have an offscreen target");
            dt.render_output(renderer, fb, age, &elements, Color32F::TRANSPARENT)?
        };

        let buffers = render_element_buffers(renderer, &elements);

        Ok((res, buffers))
    }

    let common = &mut state.common;
    let draw_cursor = session.draw_cursor();

    let nodes_cell: std::cell::Cell<Option<KmsNodes>> = std::cell::Cell::new(None);
    let renderer = match state.backend.offscreen_renderer(|kms| {
        let node = get_dmabuf(&buffer)
            .ok()
            .and_then(|dmabuf| dmabuf.node())
            .or_else(|| {
                toplevel
                    .wl_surface()
                    .and_then(|wl_surface| {
                        with_renderer_surface_state(&wl_surface, |state| {
                            let buffer = state.buffer()?;
                            let dmabuf = get_dmabuf(buffer).ok()?;
                            dmabuf.node()
                        })
                    })
                    .flatten()
            })
            .or(*kms.primary_node.read().unwrap());
        nodes_cell.set(node.map(KmsNodes::from));
        node
    }) {
        Ok(renderer) => renderer,
        Err(err) => {
            warn!(?err, "Couldn't use node for screencopy");
            frame.fail(CaptureFailureReason::Unknown);
            return;
        }
    };
    let nodes = nodes_cell.get();
    let result = match renderer {
        RendererRef::Glow(renderer) => match render_session(
            renderer,
            session.user_data().get::<SessionData>().unwrap(),
            CaptureSessionRef::Session(session.clone()),
            nodes,
            true,
            frame,
            Transform::Normal,
            |buffer, renderer, offscreen, dt, age, additional_damage| {
                render_fn(
                    buffer,
                    renderer,
                    offscreen,
                    dt,
                    age,
                    additional_damage,
                    draw_cursor,
                    common,
                    toplevel,
                    geometry,
                )
            },
        ) {
            Ok(frame) => frame,
            Err(err) => {
                tracing::warn!(?err, "Failed to render to screencopy buffer");
                None
            }
        },
        RendererRef::MultiGles(mut renderer) => match render_session(
            &mut renderer,
            session.user_data().get::<SessionData>().unwrap(),
            CaptureSessionRef::Session(session.clone()),
            nodes,
            true,
            frame,
            Transform::Normal,
            |buffer, renderer, offscreen, dt, age, additional_damage| {
                render_fn(
                    buffer,
                    renderer,
                    offscreen,
                    dt,
                    age,
                    additional_damage,
                    draw_cursor,
                    common,
                    toplevel,
                    geometry,
                )
            },
        ) {
            Ok(frame) => frame,
            Err(err) => {
                tracing::warn!(?err, "Failed to render to screencopy buffer");
                None
            }
        },
        RendererRef::MultiVulkan(mut renderer) => match render_session(
            &mut renderer,
            session.user_data().get::<SessionData>().unwrap(),
            CaptureSessionRef::Session(session.clone()),
            nodes,
            true,
            frame,
            Transform::Normal,
            |buffer, renderer, offscreen, dt, age, additional_damage| {
                render_fn(
                    buffer,
                    renderer,
                    offscreen,
                    dt,
                    age,
                    additional_damage,
                    draw_cursor,
                    common,
                    toplevel,
                    geometry,
                )
            },
        ) {
            Ok(frame) => frame,
            Err(err) => {
                tracing::warn!(?err, "Failed to render to screencopy buffer");
                None
            }
        },
    };

    if let Some(pending_image_copy_data) = result {
        pending_image_copy_data.send_success_when_ready(
            Transform::Normal,
            &common.event_loop_handle,
            common.clock.now(),
        );
    }
}

pub fn render_cursor_to_buffer(
    state: &mut State,
    session: &CursorSessionRef,
    frame: Frame,
    seat: &Seat<State>,
) {
    let buffer = frame.buffer();
    let cursor_geometry = seat.cursor_geometry((0.0, 0.0), state.common.clock.now());
    let constraints = cursor_capture_constraints(cursor_geometry);
    let buffer_size = buffer_dimensions(&buffer).unwrap();
    if buffer_size != constraints.size {
        session.update_constraints(constraints.clone());
        if let Some(data) = session.user_data().get::<SessionData>() {
            *data.lock().unwrap() = SessionUserData::new(OutputDamageTracker::new(
                constraints
                    .size
                    .to_logical(1, Transform::Normal)
                    .to_physical(1),
                1.0,
                Transform::Normal,
            ));
        }
        frame.fail(CaptureFailureReason::BufferConstraints);
        return;
    }

    fn render_fn<'d, R>(
        buffer: &WlBuffer,
        renderer: &mut R,
        offscreen: Option<&mut R::Framebuffer<'_>>,
        dt: &'d mut OutputDamageTracker,
        age: usize,
        additional_damage: Vec<Rectangle<i32, BufferCoords>>,
        common: &mut Common,
        seat: &Seat<State>,
    ) -> Result<
        (
            RenderOutputResult<'d>,
            Vec<smithay::backend::renderer::utils::Buffer>,
        ),
        DTError<R::Error>,
    >
    where
        R: AsGlowRenderer,
        R::TextureId: Send + Clone + 'static,
        CosmicElement<R>: RenderElement<R>,
        CosmicMappedRenderElement<R>: RenderElement<R>,
    {
        let mut elements: Vec<_> = additional_damage
            .into_iter()
            .filter_map(|rect| {
                let logical_rect = rect.to_logical(1, Transform::Normal, &Size::from((64, 64)));
                logical_rect.intersection(Rectangle::from_size((64, 64).into()))
            })
            .map(DamageElement::new)
            .map(WindowCaptureElement::from)
            .collect();

        cursor::draw_cursor(
            renderer,
            seat,
            Point::from((0.0, 0.0)),
            1.0.into(),
            1.0,
            common.clock.now(),
            0,
            true,
            &mut |elem, _| {
                elements.push(
                    RelocateRenderElement::from_element(elem, (0, 0), Relocate::Relative).into(),
                )
            },
        );

        let res = if let Ok(dmabuf) = get_dmabuf(buffer) {
            let mut dmabuf_clone = dmabuf.clone();
            let mut fb = renderer
                .bind(&mut dmabuf_clone)
                .map_err(DTError::Rendering)?;
            dt.render_output(renderer, &mut fb, age, &elements, [0.0, 0.0, 0.0, 0.0])?
        } else {
            let fb = offscreen.expect("shm buffers should have offscreen target");
            dt.render_output(renderer, fb, age, &elements, [0.0, 0.0, 0.0, 0.0])?
        };

        let buffers = render_element_buffers(renderer, &elements);

        Ok((res, buffers))
    }

    let common = &mut state.common;
    let nodes_cell: std::cell::Cell<Option<KmsNodes>> = std::cell::Cell::new(None);
    let renderer = match state.backend.offscreen_renderer(|kms| {
        let node = *kms.primary_node.read().unwrap();
        nodes_cell.set(node.map(KmsNodes::from));
        node
    }) {
        Ok(renderer) => renderer,
        Err(err) => {
            warn!(?err, "Couldn't use node for screencopy");
            frame.fail(CaptureFailureReason::Unknown);
            return;
        }
    };
    let nodes = nodes_cell.get();
    let result = match renderer {
        RendererRef::Glow(renderer) => {
            match render_session(
                renderer,
                session.user_data().get::<SessionData>().unwrap(),
                CaptureSessionRef::Cursor(session.clone()),
                nodes,
                true,
                frame,
                Transform::Normal,
                |buffer, renderer, offscreen, dt, age, additional_damage| {
                    render_fn(
                        buffer,
                        renderer,
                        offscreen,
                        dt,
                        age,
                        additional_damage,
                        common,
                        seat,
                    )
                },
            ) {
                Ok(frame) => frame,
                Err(err) => {
                    tracing::warn!(?err, "Failed to render to screencopy buffer");
                    None
                }
            }
        }
        RendererRef::MultiGles(mut renderer) => {
            match render_session(
                &mut renderer,
                session.user_data().get::<SessionData>().unwrap(),
                CaptureSessionRef::Cursor(session.clone()),
                nodes,
                true,
                frame,
                Transform::Normal,
                |buffer, renderer, offscreen, dt, age, additional_damage| {
                    render_fn(
                        buffer,
                        renderer,
                        offscreen,
                        dt,
                        age,
                        additional_damage,
                        common,
                        seat,
                    )
                },
            ) {
                Ok(frame) => frame,
                Err(err) => {
                    tracing::warn!(?err, "Failed to render to screencopy buffer");
                    None
                }
            }
        }
        RendererRef::MultiVulkan(mut renderer) => {
            match render_session(
                &mut renderer,
                session.user_data().get::<SessionData>().unwrap(),
                CaptureSessionRef::Cursor(session.clone()),
                nodes,
                true,
                frame,
                Transform::Normal,
                |buffer, renderer, offscreen, dt, age, additional_damage| {
                    render_fn(
                        buffer,
                        renderer,
                        offscreen,
                        dt,
                        age,
                        additional_damage,
                        common,
                        seat,
                    )
                },
            ) {
                Ok(frame) => frame,
                Err(err) => {
                    tracing::warn!(?err, "Failed to render to screencopy buffer");
                    None
                }
            }
        }
    };

    if let Some(pending_image_copy_data) = result {
        pending_image_copy_data.send_success_when_ready(
            Transform::Normal,
            &common.event_loop_handle,
            common.clock.now(),
        );
    }
}
