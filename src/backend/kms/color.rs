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

// Generated from redshift's src/colorramp.c (blackbody_color table), 1000K-25100K
// at 100K steps, to match the same color-temperature curve gammastep/redshift/
// wlsunset use, so results are consistent whether a user drives night light via
// this compositor's own toggle or an external tool.
const BLACKBODY_TABLE: [[f64; 3]; 242] = [
    [1.00000000, 0.18172716, 0.00000000],
    [1.00000000, 0.25503671, 0.00000000],
    [1.00000000, 0.30942099, 0.00000000],
    [1.00000000, 0.35357379, 0.00000000],
    [1.00000000, 0.39091524, 0.00000000],
    [1.00000000, 0.42322816, 0.00000000],
    [1.00000000, 0.45159884, 0.00000000],
    [1.00000000, 0.47675916, 0.00000000],
    [1.00000000, 0.49923747, 0.00000000],
    [1.00000000, 0.51943421, 0.00000000],
    [1.00000000, 0.54360078, 0.08679949],
    [1.00000000, 0.56618736, 0.14065513],
    [1.00000000, 0.58734976, 0.18362641],
    [1.00000000, 0.60724493, 0.22137978],
    [1.00000000, 0.62600248, 0.25591950],
    [1.00000000, 0.64373109, 0.28819679],
    [1.00000000, 0.66052319, 0.31873863],
    [1.00000000, 0.67645822, 0.34786758],
    [1.00000000, 0.69160518, 0.37579588],
    [1.00000000, 0.70602449, 0.40267128],
    [1.00000000, 0.71976951, 0.42860152],
    [1.00000000, 0.73288760, 0.45366838],
    [1.00000000, 0.74542112, 0.47793608],
    [1.00000000, 0.75740814, 0.50145662],
    [1.00000000, 0.76888303, 0.52427322],
    [1.00000000, 0.77987699, 0.54642268],
    [1.00000000, 0.79041843, 0.56793692],
    [1.00000000, 0.80053332, 0.58884417],
    [1.00000000, 0.81024551, 0.60916971],
    [1.00000000, 0.81957693, 0.62893653],
    [1.00000000, 0.82854786, 0.64816570],
    [1.00000000, 0.83717703, 0.66687674],
    [1.00000000, 0.84548188, 0.68508786],
    [1.00000000, 0.85347859, 0.70281616],
    [1.00000000, 0.86118227, 0.72007777],
    [1.00000000, 0.86860704, 0.73688797],
    [1.00000000, 0.87576611, 0.75326132],
    [1.00000000, 0.88267187, 0.76921169],
    [1.00000000, 0.88933596, 0.78475236],
    [1.00000000, 0.89576933, 0.79989606],
    [1.00000000, 0.90198230, 0.81465502],
    [1.00000000, 0.90963069, 0.82838210],
    [1.00000000, 0.91710889, 0.84190889],
    [1.00000000, 0.92441842, 0.85523742],
    [1.00000000, 0.93156127, 0.86836903],
    [1.00000000, 0.93853986, 0.88130458],
    [1.00000000, 0.94535695, 0.89404470],
    [1.00000000, 0.95201559, 0.90658983],
    [1.00000000, 0.95851906, 0.91894041],
    [1.00000000, 0.96487079, 0.93109690],
    [1.00000000, 0.97107439, 0.94305985],
    [1.00000000, 0.97713351, 0.95482993],
    [1.00000000, 0.98305189, 0.96640795],
    [1.00000000, 0.98883326, 0.97779486],
    [1.00000000, 0.99448139, 0.98899179],
    [1.00000000, 1.00000000, 1.00000000],
    [0.98947904, 0.99348723, 1.00000000],
    [0.97940448, 0.98722715, 1.00000000],
    [0.96975025, 0.98120637, 1.00000000],
    [0.96049223, 0.97541240, 1.00000000],
    [0.95160805, 0.96983355, 1.00000000],
    [0.94303638, 0.96443333, 1.00000000],
    [0.93480451, 0.95923080, 1.00000000],
    [0.92689056, 0.95421394, 1.00000000],
    [0.91927697, 0.94937330, 1.00000000],
    [0.91194747, 0.94470005, 1.00000000],
    [0.90488690, 0.94018594, 1.00000000],
    [0.89808115, 0.93582323, 1.00000000],
    [0.89151710, 0.93160469, 1.00000000],
    [0.88518247, 0.92752354, 1.00000000],
    [0.87906581, 0.92357340, 1.00000000],
    [0.87315640, 0.91974827, 1.00000000],
    [0.86744421, 0.91604254, 1.00000000],
    [0.86191983, 0.91245088, 1.00000000],
    [0.85657444, 0.90896831, 1.00000000],
    [0.85139976, 0.90559011, 1.00000000],
    [0.84638799, 0.90231183, 1.00000000],
    [0.84153180, 0.89912926, 1.00000000],
    [0.83682430, 0.89603843, 1.00000000],
    [0.83225897, 0.89303558, 1.00000000],
    [0.82782969, 0.89011714, 1.00000000],
    [0.82353066, 0.88727974, 1.00000000],
    [0.81935641, 0.88452017, 1.00000000],
    [0.81530175, 0.88183541, 1.00000000],
    [0.81136180, 0.87922257, 1.00000000],
    [0.80753191, 0.87667891, 1.00000000],
    [0.80380769, 0.87420182, 1.00000000],
    [0.80018497, 0.87178882, 1.00000000],
    [0.79665980, 0.86943756, 1.00000000],
    [0.79322843, 0.86714579, 1.00000000],
    [0.78988728, 0.86491137, 1.00000000],
    [0.78663296, 0.86273225, 1.00000000],
    [0.78346225, 0.86060650, 1.00000000],
    [0.78037207, 0.85853224, 1.00000000],
    [0.77735950, 0.85650771, 1.00000000],
    [0.77442176, 0.85453121, 1.00000000],
    [0.77155617, 0.85260112, 1.00000000],
    [0.76876022, 0.85071588, 1.00000000],
    [0.76603147, 0.84887402, 1.00000000],
    [0.76336762, 0.84707411, 1.00000000],
    [0.76076645, 0.84531479, 1.00000000],
    [0.75822586, 0.84359476, 1.00000000],
    [0.75574383, 0.84191277, 1.00000000],
    [0.75331843, 0.84026762, 1.00000000],
    [0.75094780, 0.83865816, 1.00000000],
    [0.74863017, 0.83708329, 1.00000000],
    [0.74636386, 0.83554194, 1.00000000],
    [0.74414722, 0.83403311, 1.00000000],
    [0.74197871, 0.83255582, 1.00000000],
    [0.73985682, 0.83110912, 1.00000000],
    [0.73778012, 0.82969211, 1.00000000],
    [0.73574723, 0.82830393, 1.00000000],
    [0.73375683, 0.82694373, 1.00000000],
    [0.73180765, 0.82561071, 1.00000000],
    [0.72989845, 0.82430410, 1.00000000],
    [0.72802807, 0.82302316, 1.00000000],
    [0.72619537, 0.82176715, 1.00000000],
    [0.72439927, 0.82053539, 1.00000000],
    [0.72263872, 0.81932722, 1.00000000],
    [0.72091270, 0.81814197, 1.00000000],
    [0.71922025, 0.81697905, 1.00000000],
    [0.71756043, 0.81583783, 1.00000000],
    [0.71593234, 0.81471775, 1.00000000],
    [0.71433510, 0.81361825, 1.00000000],
    [0.71276788, 0.81253878, 1.00000000],
    [0.71122987, 0.81147883, 1.00000000],
    [0.70972029, 0.81043789, 1.00000000],
    [0.70823838, 0.80941546, 1.00000000],
    [0.70678342, 0.80841109, 1.00000000],
    [0.70535469, 0.80742432, 1.00000000],
    [0.70395153, 0.80645469, 1.00000000],
    [0.70257327, 0.80550180, 1.00000000],
    [0.70121928, 0.80456522, 1.00000000],
    [0.69988894, 0.80364455, 1.00000000],
    [0.69858167, 0.80273941, 1.00000000],
    [0.69729688, 0.80184943, 1.00000000],
    [0.69603402, 0.80097423, 1.00000000],
    [0.69479255, 0.80011347, 1.00000000],
    [0.69357196, 0.79926681, 1.00000000],
    [0.69237173, 0.79843391, 1.00000000],
    [0.69119138, 0.79761446, 1.00000000],
    [0.69003044, 0.79680814, 1.00000000],
    [0.68888844, 0.79601466, 1.00000000],
    [0.68776494, 0.79523371, 1.00000000],
    [0.68665951, 0.79446502, 1.00000000],
    [0.68557173, 0.79370830, 1.00000000],
    [0.68450119, 0.79296330, 1.00000000],
    [0.68344751, 0.79222975, 1.00000000],
    [0.68241029, 0.79150740, 1.00000000],
    [0.68138918, 0.79079600, 1.00000000],
    [0.68038380, 0.79009531, 1.00000000],
    [0.67939381, 0.78940511, 1.00000000],
    [0.67841888, 0.78872517, 1.00000000],
    [0.67745866, 0.78805526, 1.00000000],
    [0.67651284, 0.78739518, 1.00000000],
    [0.67558112, 0.78674472, 1.00000000],
    [0.67466317, 0.78610368, 1.00000000],
    [0.67375872, 0.78547186, 1.00000000],
    [0.67286748, 0.78484907, 1.00000000],
    [0.67198916, 0.78423512, 1.00000000],
    [0.67112350, 0.78362984, 1.00000000],
    [0.67027024, 0.78303305, 1.00000000],
    [0.66942911, 0.78244457, 1.00000000],
    [0.66859988, 0.78186425, 1.00000000],
    [0.66778228, 0.78129191, 1.00000000],
    [0.66697610, 0.78072740, 1.00000000],
    [0.66618110, 0.78017057, 1.00000000],
    [0.66539706, 0.77962127, 1.00000000],
    [0.66462376, 0.77907934, 1.00000000],
    [0.66386098, 0.77854465, 1.00000000],
    [0.66310852, 0.77801705, 1.00000000],
    [0.66236618, 0.77749642, 1.00000000],
    [0.66163375, 0.77698261, 1.00000000],
    [0.66091106, 0.77647551, 1.00000000],
    [0.66019791, 0.77597498, 1.00000000],
    [0.65949412, 0.77548090, 1.00000000],
    [0.65879952, 0.77499315, 1.00000000],
    [0.65811392, 0.77451161, 1.00000000],
    [0.65743716, 0.77403618, 1.00000000],
    [0.65676908, 0.77356673, 1.00000000],
    [0.65610952, 0.77310316, 1.00000000],
    [0.65545831, 0.77264537, 1.00000000],
    [0.65481530, 0.77219324, 1.00000000],
    [0.65418036, 0.77174669, 1.00000000],
    [0.65355332, 0.77130560, 1.00000000],
    [0.65293404, 0.77086988, 1.00000000],
    [0.65232240, 0.77043944, 1.00000000],
    [0.65171824, 0.77001419, 1.00000000],
    [0.65112144, 0.76959404, 1.00000000],
    [0.65053187, 0.76917889, 1.00000000],
    [0.64994941, 0.76876866, 1.00000000],
    [0.64937392, 0.76836326, 1.00000000],
    [0.64880528, 0.76796263, 1.00000000],
    [0.64824339, 0.76756666, 1.00000000],
    [0.64768812, 0.76717529, 1.00000000],
    [0.64713935, 0.76678844, 1.00000000],
    [0.64659699, 0.76640603, 1.00000000],
    [0.64606092, 0.76602798, 1.00000000],
    [0.64553103, 0.76565424, 1.00000000],
    [0.64500722, 0.76528472, 1.00000000],
    [0.64448939, 0.76491935, 1.00000000],
    [0.64397745, 0.76455808, 1.00000000],
    [0.64347129, 0.76420082, 1.00000000],
    [0.64297081, 0.76384753, 1.00000000],
    [0.64247594, 0.76349813, 1.00000000],
    [0.64198657, 0.76315256, 1.00000000],
    [0.64150261, 0.76281076, 1.00000000],
    [0.64102399, 0.76247267, 1.00000000],
    [0.64055061, 0.76213824, 1.00000000],
    [0.64008239, 0.76180740, 1.00000000],
    [0.63961926, 0.76148010, 1.00000000],
    [0.63916112, 0.76115628, 1.00000000],
    [0.63870790, 0.76083590, 1.00000000],
    [0.63825953, 0.76051890, 1.00000000],
    [0.63781592, 0.76020522, 1.00000000],
    [0.63737701, 0.75989482, 1.00000000],
    [0.63694273, 0.75958764, 1.00000000],
    [0.63651299, 0.75928365, 1.00000000],
    [0.63608774, 0.75898278, 1.00000000],
    [0.63566691, 0.75868499, 1.00000000],
    [0.63525042, 0.75839025, 1.00000000],
    [0.63483822, 0.75809849, 1.00000000],
    [0.63443023, 0.75780969, 1.00000000],
    [0.63402641, 0.75752379, 1.00000000],
    [0.63362667, 0.75724075, 1.00000000],
    [0.63323097, 0.75696053, 1.00000000],
    [0.63283925, 0.75668310, 1.00000000],
    [0.63245144, 0.75640840, 1.00000000],
    [0.63206749, 0.75613641, 1.00000000],
    [0.63168735, 0.75586707, 1.00000000],
    [0.63131096, 0.75560036, 1.00000000],
    [0.63093826, 0.75533624, 1.00000000],
    [0.63056920, 0.75507467, 1.00000000],
    [0.63020374, 0.75481562, 1.00000000],
    [0.62984181, 0.75455904, 1.00000000],
    [0.62948337, 0.75430491, 1.00000000],
    [0.62912838, 0.75405319, 1.00000000],
    [0.62877678, 0.75380385, 1.00000000],
    [0.62842852, 0.75355685, 1.00000000],
    [0.62808356, 0.75331217, 1.00000000],
    [0.62774186, 0.75306977, 1.00000000],
    [0.62740336, 0.75282962, 1.00000000],
];

/// The white-point RGB scale for a given color temperature in Kelvin, via linear interpolation
/// of [`BLACKBODY_TABLE`] - a direct port of redshift's `colorramp_fill`'s white-point
/// computation (the gamma/brightness terms in that function are separate redshift features not
/// needed for a simple night-light toggle, so only the interpolation itself is ported).
/// Clamped to the table's valid interpolation range (1000K-25000K, so there is always a next
/// entry to interpolate toward). 6500K returns `[1.0, 1.0, 1.0]` (neutral, no adjustment).
pub fn white_point_for_temperature(kelvin: u32) -> [f64; 3] {
    let kelvin = kelvin.clamp(1000, 25000);
    let alpha = (kelvin % 100) as f64 / 100.0;
    let index = ((kelvin - 1000) / 100) as usize;
    let (c1, c2) = (BLACKBODY_TABLE[index], BLACKBODY_TABLE[index + 1]);
    [
        (1.0 - alpha) * c1[0] + alpha * c2[0],
        (1.0 - alpha) * c1[1] + alpha * c2[1],
        (1.0 - alpha) * c1[2] + alpha * c2[2],
    ]
}

/// A `GAMMA_LUT` ramp applying night-light warming at the given color temperature, optionally
/// composed with `invert`. Scales the identity (or inverted) ramp by [`white_point_for_temperature`]
/// per channel - this is the same composition redshift's own `colorramp_fill` uses (scaling
/// whatever gamma ramp is already there by the white point), just applied to our identity/invert
/// base instead of an existing arbitrary ramp.
pub fn temperature_lut(size: usize, kelvin: u32, invert: bool) -> Vec<(u16, u16, u16)> {
    let white_point = white_point_for_temperature(kelvin);
    let base = if invert { inverted_lut(size) } else { identity_lut(size) };
    base.into_iter()
        .map(|(r, g, b)| {
            (
                (r as f64 * white_point[0]).round() as u16,
                (g as f64 * white_point[1]).round() as u16,
                (b as f64 * white_point[2]).round() as u16,
            )
        })
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

    #[test]
    fn white_point_6500k_is_neutral() {
        let wp = white_point_for_temperature(6500);
        assert_close(wp, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn white_point_matches_table_entries_exactly() {
        // At an exact 100K table step (no interpolation needed), the result should be the
        // table entry itself, not an interpolated approximation of it.
        assert_close(white_point_for_temperature(1000), BLACKBODY_TABLE[0]);
        assert_close(white_point_for_temperature(4500), BLACKBODY_TABLE[35]);
    }

    #[test]
    fn white_point_interpolates_between_steps() {
        // Halfway between the 1000K and 1100K entries.
        let wp = white_point_for_temperature(1050);
        let expected = [
            (BLACKBODY_TABLE[0][0] + BLACKBODY_TABLE[1][0]) / 2.0,
            (BLACKBODY_TABLE[0][1] + BLACKBODY_TABLE[1][1]) / 2.0,
            (BLACKBODY_TABLE[0][2] + BLACKBODY_TABLE[1][2]) / 2.0,
        ];
        assert_close(wp, expected);
    }

    #[test]
    fn white_point_warm_temperature_reduces_blue_more_than_red() {
        // Warm (low Kelvin) night-light temperatures should scale blue down far more than red -
        // that's the entire point of the feature (reduce blue light).
        let wp = white_point_for_temperature(3000);
        assert!(wp[2] < wp[0], "blue ({}) should be reduced more than red ({})", wp[2], wp[0]);
        assert!((wp[0] - 1.0).abs() < 1e-9, "red channel should stay at full scale");
    }

    #[test]
    fn white_point_clamps_out_of_range_temperatures() {
        assert_close(white_point_for_temperature(0), BLACKBODY_TABLE[0]);
        // Clamped to 25000K (index 240), not the table's final 25100K entry (index 241) - there
        // must always be a next entry to interpolate toward.
        assert_close(white_point_for_temperature(u32::MAX), BLACKBODY_TABLE[240]);
    }

    #[test]
    fn temperature_lut_scales_identity_by_white_point() {
        let size = 256;
        let kelvin = 3500;
        let wp = white_point_for_temperature(kelvin);
        let lut = temperature_lut(size, kelvin, false);
        let identity = identity_lut(size);
        for i in [0, 128, 255] {
            let expected_r = (identity[i].0 as f64 * wp[0]).round() as u16;
            assert_eq!(lut[i].0, expected_r);
        }
    }

    #[test]
    fn temperature_lut_composes_with_invert() {
        let size = 256;
        let kelvin = 3500;
        let wp = white_point_for_temperature(kelvin);
        let lut = temperature_lut(size, kelvin, true);
        let inverted = inverted_lut(size);
        for i in [0, 128, 255] {
            let expected_r = (inverted[i].0 as f64 * wp[0]).round() as u16;
            assert_eq!(lut[i].0, expected_r);
        }
    }
}
