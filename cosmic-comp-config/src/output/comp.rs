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
}
