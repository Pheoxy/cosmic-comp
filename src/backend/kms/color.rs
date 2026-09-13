//! Hardware CRTC color-pipeline support (`CTM`/`GAMMA_LUT` atomic properties), used to apply
//! the accessibility screen filter (invert, greyscale, colorblind correction) and (eventually)
//! night light without a compositing shader pass - see the project plan this was built from.
//!
//! The math here is a direct, verifiable port of `backend/render/shaders/offscreen.frag`'s
//! colorblind-correction algorithm into a single composed 3x3 matrix per mode, since every step
//! of that shader (RGB->LMS, the per-mode simulation, LMS->error, and the final correction) is a
//! linear operation with no clamping or branching within a mode. [`shader_reference`] is a
//! direct, step-by-step port of the *shader's* formula (not the composed matrix) and exists only
//! so tests can cross-check the composed matrix produces the same result the shader does.

use crate::config::{ColorFilter, ScreenFilter};
use smithay::reexports::drm::control::{Device as ControlDevice, crtc, property};
use std::os::fd::AsFd;
use tracing::warn;

/// A 3x3 matrix, row-major: `mat_vec(m, [r, g, b])[0] == m[0][0]*r + m[0][1]*g + m[0][2]*b`.
pub type Matrix3 = [[f64; 3]; 3];

const IDENTITY: Matrix3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

// RGB -> LMS, from `offscreen.frag`'s `L`/`M`/`S` computation.
const RGB_TO_LMS: Matrix3 = [
    [17.8824, 43.5161, 4.11935],
    [3.45565, 27.1554, 3.86714],
    [0.0299566, 0.184309, 1.46709],
];

// LMS (post per-mode simulation) -> error, from `offscreen.frag`'s `error.rgb` computation.
const LMS_TO_ERROR: Matrix3 = [
    [0.0809444479, -0.130504409, 0.116721066],
    [-0.0102485335, 0.0540193266, -0.113614708],
    [-0.000365296938, -0.00412161469, 0.693511405],
];

// `correction` from `diff`, from `offscreen.frag`'s `correction.rgb` computation.
const CORRECTION_FROM_DIFF: Matrix3 = [[0.0, 0.0, 0.0], [0.7, 1.0, 0.0], [0.7, 0.0, 1.0]];

// Per-mode LMS simulation matrices, from `offscreen.frag`'s `l`/`m`/`s` per `color_mode` branch.
const PROTANOPIA_LMS: Matrix3 = [[0.0, 2.02344, -2.52581], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const DEUTERANOPIA_LMS: Matrix3 = [[1.0, 0.0, 0.0], [0.494207, 0.0, 1.24827], [0.0, 0.0, 1.0]];
const TRITANOPIA_LMS: Matrix3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.395913, 0.801109, 0.0]];

const GREYSCALE: Matrix3 = [
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
];

fn mat_mul(a: &Matrix3, b: &Matrix3) -> Matrix3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn mat_sub(a: &Matrix3, b: &Matrix3) -> Matrix3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] - b[i][j];
        }
    }
    out
}

fn mat_add(a: &Matrix3, b: &Matrix3) -> Matrix3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

pub fn mat_vec(m: &Matrix3, v: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
    }
    out
}

/// The single composed CTM for a given [`ColorFilter`] mode - see the module doc for why every
/// mode is a pure linear operation and can be collapsed into one matrix.
pub fn ctm_for(filter: ColorFilter) -> Matrix3 {
    match filter {
        ColorFilter::Greyscale => GREYSCALE,
        ColorFilter::Protanopia | ColorFilter::Deuteranopia | ColorFilter::Tritanopia => {
            let lms_mode = match filter {
                ColorFilter::Protanopia => PROTANOPIA_LMS,
                ColorFilter::Deuteranopia => DEUTERANOPIA_LMS,
                ColorFilter::Tritanopia => TRITANOPIA_LMS,
                ColorFilter::Greyscale => unreachable!(),
            };
            // error = LMS_TO_ERROR * lms_mode * RGB_TO_LMS * rgb
            let error_matrix = mat_mul(&LMS_TO_ERROR, &mat_mul(&lms_mode, &RGB_TO_LMS));
            // diff = rgb - error = (I - error_matrix) * rgb
            let diff_matrix = mat_sub(&IDENTITY, &error_matrix);
            // output = rgb + CORRECTION_FROM_DIFF * diff = (I + CORRECTION_FROM_DIFF * diff_matrix) * rgb
            mat_add(&IDENTITY, &mat_mul(&CORRECTION_FROM_DIFF, &diff_matrix))
        }
    }
}

/// Direct, step-by-step port of `offscreen.frag`'s formula (not the composed matrix) - used only
/// by tests, to cross-check [`ctm_for`]'s composed matrix against the shader's own logic.
#[cfg(test)]
fn shader_reference(filter: ColorFilter, rgb: [f64; 3]) -> [f64; 3] {
    if let ColorFilter::Greyscale = filter {
        let avg = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
        return [avg, avg, avg];
    }
    let lms = mat_vec(&RGB_TO_LMS, rgb);
    let lms_mode = match filter {
        ColorFilter::Protanopia => PROTANOPIA_LMS,
        ColorFilter::Deuteranopia => DEUTERANOPIA_LMS,
        ColorFilter::Tritanopia => TRITANOPIA_LMS,
        ColorFilter::Greyscale => unreachable!(),
    };
    let simulated = mat_vec(&lms_mode, lms);
    let error = mat_vec(&LMS_TO_ERROR, simulated);
    let diff = [rgb[0] - error[0], rgb[1] - error[1], rgb[2] - error[2]];
    let correction = mat_vec(&CORRECTION_FROM_DIFF, diff);
    [rgb[0] + correction[0], rgb[1] + correction[1], rgb[2] + correction[2]]
}

/// Encodes a [`Matrix3`] into the kernel's `struct drm_color_ctm` layout: 9 `u64`s in row-major
/// order, each S31.32 **sign-magnitude** (not two's complement) fixed point - the sign is bit 63,
/// the magnitude (31 integer + 32 fractional bits) is the rest.
pub fn encode_ctm_blob(m: &Matrix3) -> [u64; 9] {
    let mut out = [0u64; 9];
    let mut idx = 0;
    for row in m {
        for &value in row {
            let magnitude = (value.abs() * (1u64 << 32) as f64).round() as u64;
            let sign_bit = if value.is_sign_negative() { 1u64 << 63 } else { 0 };
            out[idx] = sign_bit | (magnitude & !(1u64 << 63));
            idx += 1;
        }
    }
    out
}

/// A `GAMMA_LUT` ramp of the given size (from the CRTC's `GAMMA_LUT_SIZE` property) that is the
/// identity (no-op) mapping - `(r, g, b)` triples, matching `struct drm_color_lut`'s field order.
pub fn identity_lut(size: usize) -> Vec<(u16, u16, u16)> {
    (0..size)
        .map(|i| {
            let v = ((0xFFFFu64 * i as u64) / (size as u64 - 1).max(1)) as u16;
            (v, v, v)
        })
        .collect()
}

/// A `GAMMA_LUT` ramp implementing `invert` (`offscreen.frag`'s `color.rgb = 1.0 - color.rgb`).
pub fn inverted_lut(size: usize) -> Vec<(u16, u16, u16)> {
    identity_lut(size)
        .into_iter()
        .map(|(r, g, b)| (0xFFFF - r, 0xFFFF - g, 0xFFFF - b))
        .collect()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmColorCtm {
    // S31.32 sign-magnitude fixed point, row-major - see `encode_ctm_blob`.
    matrix: [u64; 9],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmColorLut {
    red: u16,
    green: u16,
    blue: u16,
    reserved: u16,
}

/// Caches a CRTC's `CTM`/`GAMMA_LUT` property handles (and the blobs currently uploaded to
/// them) so the accessibility screen filter can be applied via one atomic-managed property set
/// per CRTC instead of a per-frame compositing shader pass.
///
/// [`Self::probe`] returning `None` means this CRTC doesn't expose both properties - the caller
/// should log that plainly and keep using the existing GLES shader path rather than attempt any
/// software/shader-based emulation of this on Vulkan (see the project plan: this mirrors GNOME's
/// own approach of a hard capability gate, not a compositing fallback).
pub struct CrtcColorProps {
    crtc: crtc::Handle,
    ctm: property::Handle,
    gamma_lut: property::Handle,
    gamma_lut_size: usize,
    previous_ctm_blob: Option<u64>,
    previous_gamma_blob: Option<u64>,
}

impl CrtcColorProps {
    pub fn probe<D: ControlDevice>(device: &D, crtc: crtc::Handle) -> Option<Self> {
        let props = device.get_properties(crtc).ok()?;
        let (handles, values) = props.as_props_and_values();

        let mut ctm = None;
        let mut gamma_lut = None;
        let mut gamma_lut_size = None;

        for (&prop, &value) in handles.iter().zip(values.iter()) {
            let Ok(info) = device.get_property(prop) else {
                continue;
            };
            let Ok(name) = info.name().to_str() else {
                continue;
            };
            match name {
                "CTM" if matches!(info.value_type(), property::ValueType::Blob) => {
                    ctm = Some(prop);
                }
                "GAMMA_LUT" if matches!(info.value_type(), property::ValueType::Blob) => {
                    gamma_lut = Some(prop);
                }
                "GAMMA_LUT_SIZE"
                    if matches!(info.value_type(), property::ValueType::UnsignedRange(_, _)) =>
                {
                    gamma_lut_size = Some(value as usize);
                }
                _ => (),
            }
        }

        // Zero-size GAMMA_LUT is not usable (there's nothing to build a ramp against).
        let gamma_lut_size = gamma_lut_size.filter(|&size| size > 0)?;

        Some(Self {
            crtc,
            ctm: ctm?,
            gamma_lut: gamma_lut?,
            gamma_lut_size,
            previous_ctm_blob: None,
            previous_gamma_blob: None,
        })
    }

    /// Uploads new `CTM`/`GAMMA_LUT` blobs for the given [`ScreenFilter`] state and sets both
    /// CRTC properties. A blob value of `0` means "no blob" (the DRM convention for resetting a
    /// blob property to its default/identity state), used here when `filter` is a no-op.
    pub fn apply<D: ControlDevice + AsFd>(&mut self, device: &D, filter: &ScreenFilter) -> Option<()> {
        let ctm_blob = filter.color_filter.map(|mode| {
            let matrix = DrmColorCtm { matrix: encode_ctm_blob(&ctm_for(mode)) };
            device.create_property_blob(&matrix)
        });
        let ctm_blob_id = match ctm_blob {
            Some(Ok(property::Value::Blob(id))) => Some(id),
            Some(Ok(_)) => unreachable!("create_property_blob always returns Value::Blob"),
            Some(Err(err)) => {
                warn!(?err, crtc = ?self.crtc, "failed to create CTM property blob");
                return None;
            }
            None => None,
        };

        let lut = if filter.inverted {
            inverted_lut(self.gamma_lut_size)
        } else {
            identity_lut(self.gamma_lut_size)
        };
        // `filter.is_noop()` (no invert, no color filter) still uploads an identity LUT rather
        // than resetting to blob 0, so a driver's own default GAMMA_LUT (which may not be a
        // pure identity ramp) doesn't change the picture when the filter is off.
        let mut lut_data: Vec<DrmColorLut> = lut
            .into_iter()
            .map(|(red, green, blue)| DrmColorLut { red, green, blue, reserved: 0 })
            .collect();
        let lut_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                lut_data.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(lut_data.as_slice()),
            )
        };
        let gamma_blob_id = match drm_ffi::mode::create_property_blob(device.as_fd(), lut_bytes) {
            Ok(blob) => u64::from(blob.blob_id),
            Err(err) => {
                warn!(?err, crtc = ?self.crtc, "failed to create GAMMA_LUT property blob");
                if let Some(id) = ctm_blob_id {
                    let _ = device.destroy_property_blob(id);
                }
                return None;
            }
        };

        if let Err(err) = device.set_property(self.crtc, self.ctm, ctm_blob_id.unwrap_or(0)) {
            warn!(?err, crtc = ?self.crtc, "failed to set CTM");
            if let Some(id) = ctm_blob_id {
                let _ = device.destroy_property_blob(id);
            }
            let _ = device.destroy_property_blob(gamma_blob_id);
            return None;
        }
        if let Err(err) = device.set_property(self.crtc, self.gamma_lut, gamma_blob_id) {
            warn!(?err, crtc = ?self.crtc, "failed to set GAMMA_LUT");
            let _ = device.destroy_property_blob(gamma_blob_id);
            return None;
        }

        if let Some(old) = self.previous_ctm_blob.replace(ctm_blob_id.unwrap_or(0)).filter(|&b| b != 0) {
            let _ = device.destroy_property_blob(old);
        }
        if let Some(old) = self.previous_gamma_blob.replace(gamma_blob_id).filter(|&b| b != 0) {
            let _ = device.destroy_property_blob(old);
        }
        Some(())
    }

    /// Re-applies the last-set blobs, for restoring state after a session resume (VT switch
    /// back, suspend/resume) - matching `pop-os/cosmic-comp#2417`'s own handling for `GAMMA_LUT`
    /// alone, extended here to also cover `CTM`.
    pub fn restore<D: ControlDevice>(&self, device: &D) -> Option<()> {
        device.set_property(self.crtc, self.ctm, self.previous_ctm_blob.unwrap_or(0)).ok()?;
        device
            .set_property(self.crtc, self.gamma_lut, self.previous_gamma_blob.unwrap_or(0))
            .ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_COLORS: [[f64; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.2, 0.6, 0.9],
    ];

    fn assert_close(a: [f64; 3], b: [f64; 3]) {
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < 1e-9,
                "component {i} differs: {:?} vs {:?}",
                a,
                b
            );
        }
    }

    #[test]
    fn composed_ctm_matches_shader_reference() {
        for filter in [
            ColorFilter::Greyscale,
            ColorFilter::Protanopia,
            ColorFilter::Deuteranopia,
            ColorFilter::Tritanopia,
        ] {
            let ctm = ctm_for(filter);
            for &color in &SAMPLE_COLORS {
                let via_matrix = mat_vec(&ctm, color);
                let via_shader = shader_reference(filter, color);
                assert_close(via_matrix, via_shader);
            }
        }
    }

    #[test]
    fn ctm_encoding_round_trips_known_values() {
        // 1.0 -> sign 0, magnitude == 1 << 32
        assert_eq!(encode_ctm_blob(&IDENTITY)[0], 1u64 << 32);
        // 0.0 -> 0
        assert_eq!(encode_ctm_blob(&IDENTITY)[1], 0);
        // -0.395913 (Tritanopia's one negative coefficient) has the sign bit set and a nonzero
        // magnitude in the lower 63 bits.
        let encoded = encode_ctm_blob(&TRITANOPIA_LMS)[6];
        assert_ne!(encoded & (1u64 << 63), 0, "sign bit should be set for a negative value");
        assert_ne!(encoded & !(1u64 << 63), 0, "magnitude should be nonzero");
    }

    #[test]
    fn identity_lut_is_identity() {
        let lut = identity_lut(256);
        assert_eq!(lut[0], (0, 0, 0));
        assert_eq!(lut[255], (0xFFFF, 0xFFFF, 0xFFFF));
    }

    #[test]
    fn inverted_lut_inverts() {
        let identity = identity_lut(256);
        let inverted = inverted_lut(256);
        for i in 0..256 {
            assert_eq!(inverted[i].0, 0xFFFF - identity[i].0);
        }
    }
}
