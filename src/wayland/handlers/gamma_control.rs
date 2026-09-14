// SPDX-License-Identifier: GPL-3.0-only

use smithay::output::Output;

use crate::{
    state::{BackendData, State},
    wayland::protocols::gamma_control::{
        GammaControlHandler, GammaControlState, delegate_gamma_control,
    },
};

impl GammaControlHandler for State {
    fn gamma_control_state(&mut self) -> &mut GammaControlState {
        &mut self.common.gamma_control_state
    }

    fn gamma_size(&mut self, output: &Output) -> Option<u32> {
        match &mut self.backend {
            BackendData::Kms(kms_state) => kms_state.gamma_size(output),
            // No hardware CTM/GAMMA_LUT outside the real KMS backend (winit/X11 dev backends).
            _ => None,
        }
    }

    fn set_gamma(&mut self, output: &Output, ramp: Option<Vec<u16>>) -> Option<()> {
        match &mut self.backend {
            BackendData::Kms(kms_state) => kms_state.set_gamma(output, ramp),
            _ => None,
        }
    }
}

delegate_gamma_control!(State);
