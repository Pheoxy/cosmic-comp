// SPDX-License-Identifier: GPL-3.0-only

//! Server-side `wlr-gamma-control-unstable-v1` - lets a privileged client (an external tool
//! like `gammastep`/`wlsunset`, or this compositor's own settings UI) set an output's gamma
//! ramp. Used for night light; the compositor applies the ramp via the hardware `GAMMA_LUT`
//! CRTC property (see `backend/kms/color.rs`) rather than any renderer/shader involvement, so
//! this protocol implementation has no dependency on which renderer (GLES/Vulkan) is active.

use smithay::{
    output::{Output, WeakOutput},
    reexports::{
        wayland_protocols_wlr::gamma_control::v1::server::{
            zwlr_gamma_control_manager_v1::{self, ZwlrGammaControlManagerV1},
            zwlr_gamma_control_v1::{self, ZwlrGammaControlV1},
        },
        wayland_server::{
            Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, backend::GlobalId,
        },
    },
};
use std::{collections::HashMap, fs::File, io::Read};
use tracing::warn;
use wayland_backend::server::ClientId;

pub trait GammaControlHandler {
    fn gamma_control_state(&mut self) -> &mut GammaControlState;
    /// The size of the output's `GAMMA_LUT` ramp (number of entries per channel), or `None` if
    /// the output doesn't support hardware gamma control at all.
    fn gamma_size(&mut self, output: &Output) -> Option<u32>;
    /// Sets the output's gamma ramp. `ramp` is `gamma_size * 3` u16 values: all of the red
    /// ramp, then all of the green ramp, then all of the blue ramp (matching the protocol's own
    /// wire format - not interleaved (r, g, b) triples). `None` restores the original/default
    /// ramp. Returns `None` on failure (e.g. the output has since lost gamma support).
    fn set_gamma(&mut self, output: &Output, ramp: Option<Vec<u16>>) -> Option<()>;
}

#[derive(Debug)]
pub struct GammaControlState {
    global: GlobalId,
    controls: HashMap<ZwlrGammaControlV1, WeakOutput>,
}

impl GammaControlState {
    pub fn new<D, F>(dh: &DisplayHandle, client_filter: F) -> GammaControlState
    where
        D: GlobalDispatch<ZwlrGammaControlManagerV1, GammaControlManagerGlobalData> + 'static,
        F: for<'a> Fn(&'a Client) -> bool + Clone + Send + Sync + 'static,
    {
        let global = dh.create_global::<D, ZwlrGammaControlManagerV1, _>(
            1,
            GammaControlManagerGlobalData { filter: Box::new(client_filter) },
        );

        GammaControlState { global, controls: HashMap::new() }
    }

    pub fn global_id(&self) -> GlobalId {
        self.global.clone()
    }

    /// Fails any active gamma control for an output that's gone away, matching the protocol's
    /// own `failed` event semantics ("the output doesn't support gamma tables" / has vanished).
    pub fn output_removed(&mut self, output: &Output) {
        self.controls.retain(|control, weak| {
            if weak.upgrade().as_ref() == Some(output) {
                control.failed();
                false
            } else {
                true
            }
        });
    }
}

pub struct GammaControlManagerGlobalData {
    filter: Box<dyn for<'a> Fn(&'a Client) -> bool + Send + Sync>,
}

pub struct GammaControlData {
    output: WeakOutput,
}

impl<D> GlobalDispatch<ZwlrGammaControlManagerV1, GammaControlManagerGlobalData, D>
    for GammaControlState
where
    D: GlobalDispatch<ZwlrGammaControlManagerV1, GammaControlManagerGlobalData>
        + Dispatch<ZwlrGammaControlManagerV1, ()>
        + 'static,
{
    fn bind(
        _state: &mut D,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwlrGammaControlManagerV1>,
        _global_data: &GammaControlManagerGlobalData,
        data_init: &mut DataInit<'_, D>,
    ) {
        data_init.init(resource, ());
    }

    fn can_view(client: Client, global_data: &GammaControlManagerGlobalData) -> bool {
        (global_data.filter)(&client)
    }
}

impl<D> Dispatch<ZwlrGammaControlManagerV1, (), D> for GammaControlState
where
    D: GlobalDispatch<ZwlrGammaControlManagerV1, GammaControlManagerGlobalData>
        + Dispatch<ZwlrGammaControlManagerV1, ()>
        + Dispatch<ZwlrGammaControlV1, GammaControlData>
        + GammaControlHandler
        + 'static,
{
    fn request(
        state: &mut D,
        _client: &Client,
        _obj: &ZwlrGammaControlManagerV1,
        request: zwlr_gamma_control_manager_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            zwlr_gamma_control_manager_v1::Request::GetGammaControl { id, output } => {
                let output = Output::from_resource(&output);
                let already_controlled = output.as_ref().is_some_and(|output| {
                    state
                        .gamma_control_state()
                        .controls
                        .values()
                        .any(|weak| weak.upgrade().as_ref() == Some(output))
                });
                let gamma_size = if already_controlled {
                    None
                } else {
                    output.as_ref().and_then(|o| state.gamma_size(o))
                };

                let control = data_init.init(
                    id,
                    GammaControlData {
                        output: output.as_ref().map(|o| o.downgrade()).unwrap_or_default(),
                    },
                );
                // Sent immediately on creation per the protocol, even on failure - `failed` is
                // how the client learns it never got usable control.
                match gamma_size {
                    Some(size) => {
                        control.gamma_size(size);
                        state
                            .gamma_control_state()
                            .controls
                            .insert(control, output.unwrap().downgrade());
                    }
                    None => control.failed(),
                }
            }
            zwlr_gamma_control_manager_v1::Request::Destroy => {}
            _ => unreachable!(),
        }
    }
}

impl<D> Dispatch<ZwlrGammaControlV1, GammaControlData, D> for GammaControlState
where
    D: Dispatch<ZwlrGammaControlV1, GammaControlData> + GammaControlHandler + 'static,
{
    fn request(
        state: &mut D,
        _client: &Client,
        obj: &ZwlrGammaControlV1,
        request: zwlr_gamma_control_v1::Request,
        data: &GammaControlData,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            zwlr_gamma_control_v1::Request::SetGamma { fd } => {
                let Some(output) = data.output.upgrade() else {
                    obj.failed();
                    return;
                };
                let Some(gamma_size) = state.gamma_size(&output) else {
                    obj.failed();
                    state.gamma_control_state().controls.remove(obj);
                    return;
                };

                // Per protocol: successive whole ramps (all of red, then all of green, then all
                // of blue), not interleaved (r, g, b) triples - `set_gamma` passes this straight
                // through, the KMS backend does the interleaving into the DRM blob layout.
                let mut ramp = vec![0u16; gamma_size as usize * 3];
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        ramp.as_mut_ptr() as *mut u8,
                        std::mem::size_of_val(ramp.as_slice()),
                    )
                };
                let mut file = File::from(fd);
                if let Err(err) = file.read_exact(bytes) {
                    warn!(output = %output.name(), ?err, "failed to read gamma ramp data");
                    obj.failed();
                    state.gamma_control_state().controls.remove(obj);
                    let _ = state.set_gamma(&output, None);
                    return;
                }
                // The protocol requires exactly `gamma_size * 3` u16s; reject any trailing data
                // past that rather than silently ignoring it.
                if !matches!(file.read(&mut [0u8]), Ok(0)) {
                    warn!(output = %output.name(), "gamma ramp data is larger than expected");
                    obj.failed();
                    state.gamma_control_state().controls.remove(obj);
                    let _ = state.set_gamma(&output, None);
                    return;
                }

                if state.set_gamma(&output, Some(ramp)).is_none() {
                    warn!(output = %output.name(), "failed to set gamma");
                    obj.failed();
                    state.gamma_control_state().controls.remove(obj);
                }
            }
            zwlr_gamma_control_v1::Request::Destroy => {
                if let Some(output) = data.output.upgrade() {
                    let _ = state.set_gamma(&output, None);
                }
            }
            _ => unreachable!(),
        }
    }

    fn destroyed(state: &mut D, _client: ClientId, obj: &ZwlrGammaControlV1, data: &GammaControlData) {
        state.gamma_control_state().controls.remove(obj);
        if let Some(output) = data.output.upgrade() {
            let _ = state.set_gamma(&output, None);
        }
    }
}

macro_rules! delegate_gamma_control {
    ($(@<$( $lt:tt $( : $clt:tt $(+ $dlt:tt )* )? ),+>)? $ty: ty) => {
        smithay::reexports::wayland_server::delegate_global_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            smithay::reexports::wayland_protocols_wlr::gamma_control::v1::server::zwlr_gamma_control_manager_v1::ZwlrGammaControlManagerV1: $crate::wayland::protocols::gamma_control::GammaControlManagerGlobalData
        ] => $crate::wayland::protocols::gamma_control::GammaControlState);
        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            smithay::reexports::wayland_protocols_wlr::gamma_control::v1::server::zwlr_gamma_control_manager_v1::ZwlrGammaControlManagerV1: ()
        ] => $crate::wayland::protocols::gamma_control::GammaControlState);
        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            smithay::reexports::wayland_protocols_wlr::gamma_control::v1::server::zwlr_gamma_control_v1::ZwlrGammaControlV1: $crate::wayland::protocols::gamma_control::GammaControlData
        ] => $crate::wayland::protocols::gamma_control::GammaControlState);
    };
}
pub(crate) use delegate_gamma_control;
