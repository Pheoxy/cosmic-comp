// SPDX-License-Identifier: GPL-3.0-only

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::OpenOptions, path::Path};
use tracing::{error, warn};

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OutputState {
    #[serde(rename = "true")]
    Enabled,
    #[serde(rename = "false")]
    Disabled,
    Mirroring(String),
}

fn default_state() -> OutputState {
    OutputState::Enabled
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AdaptiveSync {
    #[serde(rename = "true")]
    Enabled,
    #[serde(rename = "false")]
    Disabled,
    Force,
}

fn default_sync() -> AdaptiveSync {
    AdaptiveSync::Enabled
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputsConfig {
    pub config: HashMap<Vec<OutputInfo>, Vec<OutputConfig>>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct OutputConfig {
    pub mode: ((i32, i32), Option<u32>),
    #[serde(default = "default_sync")]
    pub vrr: AdaptiveSync,
    pub scale: f64,
    pub transform: TransformDef,
    pub position: (u32, u32),
    #[serde(default = "default_state")]
    pub enabled: OutputState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_bpc: Option<u32>,
    #[serde(default)]
    pub xwayland_primary: bool,
}

impl Default for OutputConfig {
    fn default() -> OutputConfig {
        OutputConfig {
            mode: ((0, 0), None),
            vrr: AdaptiveSync::Enabled,
            scale: 1.0,
            transform: TransformDef::Normal,
            position: (0, 0),
            enabled: OutputState::Enabled,
            max_bpc: None,
            xwayland_primary: false,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutputInfo {
    pub connector: String,
    pub make: String,
    pub model: String,
    /// EDID-derived serial (product serial string or numeric serial), when the
    /// display reports one.
    ///
    /// This is a best-effort identity hint, not a guaranteed-unique key:
    /// - Some panels (e.g. internal `eDP` displays) have no serial descriptor
    ///   at all.
    /// - Some external displays reuse a manufacturer-default numeric serial
    ///   (e.g. `0x01010101`) that is identical across every unit of that
    ///   model, so it must not be trusted as unique without also checking it
    ///   actually disambiguates the currently-connected outputs.
    /// - libdisplay-info's own docs describe this value as "informational
    ///   and not meant to be used in programmatic decisions" - so it must
    ///   always be used as a hint alongside `connector`/`make`/`model`, with
    ///   graceful fallback when absent or ambiguous, never as a sole key.
    ///
    /// `#[serde(default)]` so `outputs.ron` files written before this field
    /// existed keep deserializing (they simply get `None` here).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serial: Option<String>,
}

pub fn load_outputs(path: Option<impl AsRef<Path>>) -> OutputsConfig {
    if let Some(path) = path.as_ref() {
        let path: &Path = path.as_ref();
        if path.exists() {
            match ron::de::from_reader::<_, OutputsConfig>(
                OpenOptions::new().read(true).open(path).unwrap(),
            ) {
                Ok(mut config) => {
                    for (info, config) in config.config.iter_mut() {
                        let config_clone = config.clone();
                        for conf in config.iter_mut() {
                            if let OutputState::Mirroring(conn) = &conf.enabled {
                                if let Some((j, _)) = info
                                    .iter()
                                    .enumerate()
                                    .find(|(_, info)| &info.connector == conn)
                                {
                                    if config_clone[j].enabled != OutputState::Enabled {
                                        warn!(
                                            "Invalid Mirroring tag, overriding with `Enabled` instead"
                                        );
                                        conf.enabled = OutputState::Enabled;
                                    }
                                } else {
                                    warn!(
                                        "Invalid Mirroring tag, overriding with `Enabled` instead"
                                    );
                                    conf.enabled = OutputState::Enabled;
                                }
                            }
                        }
                    }
                    return config;
                }
                Err(err) => {
                    warn!(?err, "Failed to read output_config, resetting..");
                    if let Err(err) = std::fs::remove_file(path) {
                        error!(?err, "Failed to remove output_config.");
                    }
                }
            };
        }
    }

    OutputsConfig {
        config: HashMap::new(),
    }
}

/// Finds the saved output configuration matching the currently-connected
/// `infos`.
///
/// Tries an exact match first (covers the common case where nothing has
/// changed, and is the only path needed once every saved layout has been
/// re-saved with a `serial`). If that fails, falls back to a serial-aware,
/// connector-order-independent match against every saved layout of the same
/// output count: two entries are considered the same physical display if
/// their EDID serials agree (and that serial isn't shared by another
/// currently-connected or saved display - see `ambiguous_serials`), or
/// otherwise by today's `connector`+`make`+`model` equality. This keeps
/// pre-existing saved layouts (written before `serial` existed, or for
/// displays with no EDID serial) working exactly as before, while letting
/// serial-bearing displays keep matching across a connector rename (the bug
/// this exists to fix).
pub fn find_output_configs(
    config: &HashMap<Vec<OutputInfo>, Vec<OutputConfig>>,
    infos: &[OutputInfo],
) -> Option<Vec<OutputConfig>> {
    if let Some(configs) = config.get(infos) {
        return Some(configs.clone());
    }

    let ambiguous_current = ambiguous_serials(infos);
    for (stored_infos, stored_configs) in config {
        if stored_infos.len() != infos.len() {
            continue;
        }
        let ambiguous_stored = ambiguous_serials(stored_infos);
        if let Some(mapping) =
            match_same_displays(infos, stored_infos, &ambiguous_current, &ambiguous_stored)
        {
            return Some(mapping.into_iter().map(|j| stored_configs[j].clone()).collect());
        }
    }

    None
}

/// Serial values that appear more than once in `infos`, and therefore cannot
/// be trusted to identify a single display within this set (e.g. some
/// displays report a manufacturer-default placeholder serial shared by every
/// unit of that model, rather than a real per-unit value).
fn ambiguous_serials(infos: &[OutputInfo]) -> std::collections::HashSet<&str> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for info in infos {
        if let Some(serial) = info.serial.as_deref() {
            *counts.entry(serial).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .filter(|(_, n)| *n > 1)
        .map(|(s, _)| s)
        .collect()
}

fn same_display(
    a: &OutputInfo,
    b: &OutputInfo,
    ambiguous_a: &std::collections::HashSet<&str>,
    ambiguous_b: &std::collections::HashSet<&str>,
) -> bool {
    if let (Some(sa), Some(sb)) = (a.serial.as_deref(), b.serial.as_deref()) {
        if sa == sb && !ambiguous_a.contains(sa) && !ambiguous_b.contains(sb) {
            return true;
        }
    }
    a.connector == b.connector && a.make == b.make && a.model == b.model
}

/// Finds a one-to-one pairing between every entry in `infos` and some entry
/// in `stored` such that each pair is the same physical display (per
/// `same_display`), returning `mapping` where `mapping[i]` is the index into
/// `stored` that `infos[i]` was paired with - or `None` if no such pairing
/// exists. `infos`/`stored` are always small (a handful of connected
/// displays at most), so plain backtracking is more than fast enough.
fn match_same_displays(
    infos: &[OutputInfo],
    stored: &[OutputInfo],
    ambiguous_current: &std::collections::HashSet<&str>,
    ambiguous_stored: &std::collections::HashSet<&str>,
) -> Option<Vec<usize>> {
    fn backtrack(
        infos: &[OutputInfo],
        stored: &[OutputInfo],
        ambiguous_current: &std::collections::HashSet<&str>,
        ambiguous_stored: &std::collections::HashSet<&str>,
        i: usize,
        used: &mut [bool],
        mapping: &mut Vec<usize>,
    ) -> bool {
        if i == infos.len() {
            return true;
        }
        for j in 0..stored.len() {
            if used[j] || !same_display(&infos[i], &stored[j], ambiguous_current, ambiguous_stored)
            {
                continue;
            }
            used[j] = true;
            mapping.push(j);
            if backtrack(
                infos,
                stored,
                ambiguous_current,
                ambiguous_stored,
                i + 1,
                used,
                mapping,
            ) {
                return true;
            }
            mapping.pop();
            used[j] = false;
        }
        false
    }

    let mut used = vec![false; stored.len()];
    let mut mapping = Vec::with_capacity(infos.len());
    backtrack(
        infos,
        stored,
        ambiguous_current,
        ambiguous_stored,
        0,
        &mut used,
        &mut mapping,
    )
    .then_some(mapping)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransformDef {
    Normal,
    _90,
    _180,
    _270,
    Flipped,
    Flipped90,
    Flipped180,
    Flipped270,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shaped like a real `outputs.ron` written before the `serial` field on
    // `OutputInfo` existed (connector/make/model only, no `serial` key at
    // all). Loading it must not lose the saved layout, and every entry must
    // come back with `serial: None`. Vendor/model strings here are
    // placeholders, not real EDID data.
    const PRE_SERIAL_FIELD_OUTPUTS_RON: &str = r#"(
    config: {
        [
            (
                connector: "DP-3",
                make: "Example Display Co",
                model: "EX2740Q",
            ),
            (
                connector: "eDP-1",
                make: "Example Panel Corp",
                model: "0x0000",
            ),
        ]: [
            (
                mode: ((2560, 1440), Some(59951)),
                vrr: r#false,
                scale: 1.0,
                transform: Normal,
                position: (1920, 0),
                enabled: r#true,
                max_bpc: Some(12),
                xwayland_primary: false,
            ),
            (
                mode: ((1920, 1080), Some(144000)),
                vrr: r#false,
                scale: 1.0,
                transform: Normal,
                position: (0, 0),
                enabled: r#true,
                xwayland_primary: true,
            ),
        ],
    },
)"#;

    fn scratch_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "cosmic-comp-config-test-{}-{}-{}",
            std::process::id(),
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn pre_serial_field_outputs_ron_still_loads() {
        let path = scratch_path("pre-serial.ron");
        std::fs::write(&path, PRE_SERIAL_FIELD_OUTPUTS_RON).unwrap();

        let config = load_outputs(Some(&path));
        std::fs::remove_file(&path).unwrap();

        assert_eq!(config.config.len(), 1, "the one saved layout must survive");
        let (infos, outputs) = config.config.iter().next().unwrap();
        assert_eq!(infos.len(), 2);
        assert!(
            infos.iter().all(|info| info.serial.is_none()),
            "entries written before the `serial` field existed must deserialize to `None`, not fail/reset: {:?}",
            infos
        );
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn serial_round_trips_through_ron() {
        let path = scratch_path("with-serial.ron");
        let mut config = OutputsConfig {
            config: HashMap::new(),
        };
        config.config.insert(
            vec![OutputInfo {
                connector: "DP-3".into(),
                make: "Example Display Co".into(),
                model: "EX2740Q".into(),
                serial: Some("EXAMPLE-SERIAL-0001".into()),
            }],
            vec![OutputConfig::default()],
        );
        std::fs::write(&path, ron::ser::to_string_pretty(&config, Default::default()).unwrap())
            .unwrap();

        let loaded = load_outputs(Some(&path));
        std::fs::remove_file(&path).unwrap();

        let (infos, _) = loaded.config.iter().next().unwrap();
        assert_eq!(infos[0].serial.as_deref(), Some("EXAMPLE-SERIAL-0001"));
    }

    fn info(connector: &str, make: &str, model: &str, serial: Option<&str>) -> OutputInfo {
        OutputInfo {
            connector: connector.into(),
            make: make.into(),
            model: model.into(),
            serial: serial.map(Into::into),
        }
    }

    // A marker `OutputConfig` distinguishable by its `position`, so tests can
    // confirm which stored config a given current output actually matched,
    // not just that *some* match was found.
    fn marker_config(x: u32, y: u32) -> OutputConfig {
        OutputConfig {
            position: (x, y),
            ..OutputConfig::default()
        }
    }

    #[test]
    fn find_output_configs_exact_match() {
        let key = vec![info("DP-3", "Vendor", "Model", Some("SER-A"))];
        let configs = vec![marker_config(100, 0)];
        let mut map = HashMap::new();
        map.insert(key.clone(), configs.clone());

        assert_eq!(find_output_configs(&map, &key), Some(configs));
    }

    #[test]
    fn find_output_configs_matches_across_connector_rename_via_serial() {
        // Saved under the connector names from a previous boot.
        let stored_key = vec![
            info("DP-3", "Vendor", "Model", Some("SER-A")),
            info("eDP-1", "Panel", "Panel", None),
        ];
        let stored_configs = vec![marker_config(100, 0), marker_config(0, 0)];
        let mut map = HashMap::new();
        map.insert(stored_key, stored_configs);

        // Same two physical displays, but the external one renumbered from
        // DP-3 to DP-5 (e.g. a Thunderbolt dock replug) - same serial, same
        // make/model, different connector name.
        let current = vec![
            info("DP-5", "Vendor", "Model", Some("SER-A")),
            info("eDP-1", "Panel", "Panel", None),
        ];

        let result = find_output_configs(&map, &current).expect("must fall back to serial match");
        assert_eq!(result[0], marker_config(100, 0), "DP-5/SER-A must get DP-3's old config");
        assert_eq!(result[1], marker_config(0, 0));
    }

    #[test]
    fn find_output_configs_realigns_permuted_order() {
        // Same scenario as above, but the current (live) output order is the
        // opposite of how it was stored - this is the case a naive positional
        // zip between the stored Vec and the current Vec would get wrong.
        let stored_key = vec![
            info("DP-3", "Vendor", "Model", Some("SER-A")),
            info("eDP-1", "Panel", "Panel", None),
        ];
        let stored_configs = vec![marker_config(100, 0), marker_config(0, 0)];
        let mut map = HashMap::new();
        map.insert(stored_key, stored_configs);

        let current = vec![
            info("eDP-1", "Panel", "Panel", None),
            info("DP-9", "Vendor", "Model", Some("SER-A")),
        ];

        let result = find_output_configs(&map, &current).expect("must fall back to serial match");
        assert_eq!(
            result[0],
            marker_config(0, 0),
            "result[0] must align to current[0] (eDP-1), not stored[0]"
        );
        assert_eq!(
            result[1],
            marker_config(100, 0),
            "result[1] must align to current[1] (DP-9/SER-A), not stored[1]"
        );
    }

    #[test]
    fn find_output_configs_falls_back_when_no_serial_on_either_side() {
        // Neither side has a serial (e.g. displays with no EDID serial
        // descriptor) - must still match by connector+make+model exactly
        // like before this field existed.
        let stored_key = vec![info("eDP-1", "Panel", "Panel", None)];
        let stored_configs = vec![marker_config(0, 0)];
        let mut map = HashMap::new();
        map.insert(stored_key, stored_configs);

        let current = vec![info("eDP-1", "Panel", "Panel", None)];
        assert_eq!(find_output_configs(&map, &current), Some(vec![marker_config(0, 0)]));
    }

    #[test]
    fn find_output_configs_ignores_ambiguous_shared_serial() {
        // Two live outputs sharing an identical (manufacturer-default
        // placeholder) serial - a real, observed case, not hypothetical: two
        // units of the same monitor model can report the same non-unique
        // numeric EDID serial. Trusting it would risk matching the wrong
        // physical unit's saved config to the wrong connector.
        let stored_key = vec![info("DP-3", "Vendor", "Model", Some("SHARED"))];
        let stored_configs = vec![marker_config(100, 0)];
        let mut map = HashMap::new();
        map.insert(stored_key, stored_configs);

        // Two current outputs, both reporting "SHARED", on different
        // connectors than the stored entry - serial must not be trusted
        // here (ambiguous on the current side), and since connector/make/
        // model don't match the stored entry either, there must be no match
        // at all rather than an arbitrary/wrong one.
        let current = vec![
            info("DP-5", "Vendor", "Model", Some("SHARED")),
            info("DP-6", "Vendor", "Model", Some("SHARED")),
        ];
        assert_eq!(find_output_configs(&map, &current), None);
    }

    #[test]
    fn find_output_configs_ambiguous_serial_still_matches_via_fallback_when_unchanged() {
        // Same ambiguous-serial situation, but nothing about the topology
        // actually changed (same connector/make/model as when it was
        // saved) - must still match via the make/model/connector fallback,
        // exactly like before this field existed, despite the serial being
        // unusable.
        let stored_key = vec![
            info("DP-3", "Vendor", "Model", Some("SHARED")),
            info("DP-4", "Vendor", "Model", Some("SHARED")),
        ];
        let stored_configs = vec![marker_config(100, 0), marker_config(200, 0)];
        let mut map = HashMap::new();
        map.insert(stored_key, stored_configs);

        let current = vec![
            info("DP-3", "Vendor", "Model", Some("SHARED")),
            info("DP-4", "Vendor", "Model", Some("SHARED")),
        ];
        assert_eq!(
            find_output_configs(&map, &current),
            Some(vec![marker_config(100, 0), marker_config(200, 0)])
        );
    }

    #[test]
    fn find_output_configs_no_match_for_different_output_count() {
        let stored_key = vec![info("DP-3", "Vendor", "Model", Some("SER-A"))];
        let mut map = HashMap::new();
        map.insert(stored_key, vec![marker_config(0, 0)]);

        let current = vec![
            info("DP-3", "Vendor", "Model", Some("SER-A")),
            info("eDP-1", "Panel", "Panel", None),
        ];
        assert_eq!(find_output_configs(&map, &current), None);
    }
}
