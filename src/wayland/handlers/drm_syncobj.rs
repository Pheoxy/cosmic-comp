// SPDX-License-Identifier: GPL-3.0-only

use crate::{
    state::{BackendData, State},
    wayland::handlers::compositor::client_compositor_state,
};
use smithay::{
    reexports::wayland_server::{DisplayHandle, Resource, protocol::wl_surface::WlSurface},
    wayland::drm_syncobj::{DrmSyncPoint, DrmSyncPointSource, DrmSyncobjHandler, DrmSyncobjState},
};

impl DrmSyncobjHandler for State {
    fn drm_syncobj_state(&mut self) -> Option<&mut DrmSyncobjState> {
        let kms = match &mut self.backend {
            BackendData::Kms(kms) => kms,
            _ => unreachable!(),
        };
        kms.syncobj_state.as_mut()
    }

    fn drm_syncobj_install_acquire_point_source(
        &mut self,
        _dh: &DisplayHandle,
        surface: &WlSurface,
        _acquire_point: &DrmSyncPoint,
        source: DrmSyncPointSource,
    ) -> bool {
        let BackendData::Kms(_) = &self.backend else {
            return false;
        };
        let Some(client) = surface.client() else {
            return false;
        };
        let acquire_source_id = self.backend.kms().next_syncobj_acquire_source_id();
        let res = self
            .common
            .event_loop_handle
            .insert_source(source, move |_, _, data| {
                if let BackendData::Kms(kms) = &mut data.backend {
                    kms.syncobj_acquire_source_tokens.remove(&acquire_source_id);
                }
                let dh = data.common.display_handle.clone();
                client_compositor_state(&client).blocker_cleared(data, &dh);
                Ok(())
            });
        match res {
            Ok(token) => {
                self.backend
                    .kms()
                    .syncobj_acquire_source_tokens
                    .insert(acquire_source_id, token);
                true
            }
            Err(_) => false,
        }
    }
}
