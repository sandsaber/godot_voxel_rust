//! Forest-wide `meta.vxrm` for [`super::RegionFilesStream`].
//!
//! Ports the JSON sidecar in `VoxelStreamRegionFiles` (`META_FILE_NAME`).
//! It locks `block_size_po2`, `region_size_po2`, `sector_size`, and the eight
//! channel depths so every region file in the directory shares one format.

use crate::math::Vector3i;
use crate::storage::voxel_buffer::{DEFAULT_CHANNEL_DEPTH, MAX_CHANNELS};
use crate::storage::{ChannelDepth, VoxelBuffer, VoxelFormat};
use crate::streams::region::format::RegionFormat;
use std::path::Path;

/// File name written next to `lod*` trees. Matches C++ `META_FILE_NAME`.
pub const META_FILE_NAME: &str = "meta.vxrm";
/// Latest forest-meta version. Matches C++ `FORMAT_VERSION` for the forest.
pub const META_FORMAT_VERSION: u8 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForestMetaError {
    Io(String),
    Invalid(String),
}

impl std::fmt::Display for ForestMetaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(message) | Self::Invalid(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ForestMetaError {}

/// Stream-wide format stored in `meta.vxrm`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionForestMeta {
    pub version: u8,
    pub block_size_po2: u8,
    pub region_size_po2: u8,
    pub sector_size: u32,
    pub channel_depths: [ChannelDepth; MAX_CHANNELS],
}

impl Default for RegionForestMeta {
    fn default() -> Self {
        Self {
            version: META_FORMAT_VERSION,
            block_size_po2: 4,
            region_size_po2: 4,
            sector_size: 512,
            channel_depths: VoxelFormat::new().depths,
        }
    }
}

impl RegionForestMeta {
    pub fn from_settings(block_size_po2: u8, region_size: i32, sector_size: u32) -> Self {
        Self {
            block_size_po2: block_size_po2.clamp(1, 8),
            region_size_po2: region_size_po2(region_size),
            sector_size: sector_size.max(1),
            ..Self::default()
        }
    }

    pub fn capture_channel_depths(&mut self, buffer: &VoxelBuffer) {
        for (index, depth) in self.channel_depths.iter_mut().enumerate() {
            *depth = buffer.channel_depth(index);
        }
    }

    pub fn apply_channel_depths(&self, buffer: &mut VoxelBuffer) {
        for (index, &depth) in self.channel_depths.iter().enumerate() {
            buffer.set_channel_depth(index, depth);
        }
    }

    pub fn matches_buffer(&self, buffer: &VoxelBuffer) -> bool {
        let expected = 1i32 << self.block_size_po2;
        if buffer.size() != Vector3i::splat(expected) {
            return false;
        }
        (0..MAX_CHANNELS).all(|index| buffer.channel_depth(index) == self.channel_depths[index])
    }

    pub fn region_size_blocks(&self) -> i32 {
        1i32 << self.region_size_po2
    }

    pub fn to_region_format(&self) -> RegionFormat {
        RegionFormat {
            block_size_po2: self.block_size_po2,
            region_size: Vector3i::splat(self.region_size_blocks()),
            channel_depths: self.channel_depths,
            sector_size: self.sector_size,
            palette: None,
        }
    }

    pub fn to_json(&self) -> String {
        let depths: Vec<String> = self
            .channel_depths
            .iter()
            .map(|depth| (*depth as u8).to_string())
            .collect();
        format!(
            "{{\n\t\"version\": {},\n\t\"block_size_po2\": {},\n\t\"region_size_po2\": {},\n\t\"sector_size\": {},\n\t\"channel_depths\": [\n\t\t{}\n\t]\n}}\n",
            self.version,
            self.block_size_po2,
            self.region_size_po2,
            self.sector_size,
            depths.join(",\n\t\t")
        )
    }

    pub fn from_json(src: &str) -> Result<Self, ForestMetaError> {
        let version = json_u8(src, "version")?;
        let block_size_po2 = json_u8(src, "block_size_po2")?;
        let region_size_po2 = json_u8(src, "region_size_po2")?;
        let sector_size = json_u32(src, "sector_size")?;
        let depths = json_u8_array(src, "channel_depths")?;
        if depths.len() != MAX_CHANNELS {
            return Err(ForestMetaError::Invalid(format!(
                "channel_depths must have {MAX_CHANNELS} entries"
            )));
        }
        let mut channel_depths = [DEFAULT_CHANNEL_DEPTH; MAX_CHANNELS];
        for (index, raw) in depths.iter().copied().enumerate() {
            channel_depths[index] = depth_from_u8(raw)?;
        }
        let meta = Self {
            version,
            block_size_po2,
            region_size_po2,
            sector_size,
            channel_depths,
        };
        meta.validate()?;
        Ok(meta)
    }

    pub fn validate(&self) -> Result<(), ForestMetaError> {
        if !(1..=8).contains(&self.block_size_po2) {
            return Err(ForestMetaError::Invalid(format!(
                "invalid block_size_po2 {}",
                self.block_size_po2
            )));
        }
        if !(1..=8).contains(&self.region_size_po2) {
            return Err(ForestMetaError::Invalid(format!(
                "invalid region_size_po2 {}",
                self.region_size_po2
            )));
        }
        if self.sector_size == 0 {
            return Err(ForestMetaError::Invalid(
                "sector_size must be positive".into(),
            ));
        }
        Ok(())
    }

    pub fn load(directory: &Path) -> Result<Option<Self>, ForestMetaError> {
        let path = directory.join(META_FILE_NAME);
        match std::fs::read_to_string(&path) {
            Ok(text) => Ok(Some(Self::from_json(&text)?)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(ForestMetaError::Io(format!(
                "read {}: {error}",
                path.display()
            ))),
        }
    }

    pub fn save(&self, directory: &Path) -> Result<(), ForestMetaError> {
        self.validate()?;
        std::fs::create_dir_all(directory).map_err(|error| {
            ForestMetaError::Io(format!("create {}: {error}", directory.display()))
        })?;
        let path = directory.join(META_FILE_NAME);
        // Write-through-rename so a concurrent reader (or a crash) can never
        // observe a truncated file: readers treat an unparsable meta.vxrm as
        // a sticky corrupt-forest error. Each writer uses its own temporary
        // name — concurrent first-savers would otherwise race one tmp path
        // (one renames it away, the rest fail with ENOENT).
        static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = directory.join(format!("{META_FILE_NAME}.{unique}.tmp"));
        std::fs::write(&tmp, self.to_json())
            .map_err(|error| ForestMetaError::Io(format!("write {}: {error}", tmp.display())))?;
        std::fs::rename(&tmp, &path).map_err(|error| {
            ForestMetaError::Io(format!(
                "rename {} -> {}: {error}",
                tmp.display(),
                path.display()
            ))
        })
    }
}

pub fn region_size_po2(region_size: i32) -> u8 {
    let region_size = region_size.max(1) as u32;
    if region_size.is_power_of_two() {
        region_size.ilog2() as u8
    } else {
        region_size.next_power_of_two().ilog2() as u8
    }
    .clamp(0, 8)
}

fn depth_from_u8(raw: u8) -> Result<ChannelDepth, ForestMetaError> {
    match raw {
        0 => Ok(ChannelDepth::Bit8),
        1 => Ok(ChannelDepth::Bit16),
        2 => Ok(ChannelDepth::Bit32),
        3 => Ok(ChannelDepth::Bit64),
        other => Err(ForestMetaError::Invalid(format!(
            "invalid channel depth {other}"
        ))),
    }
}

fn json_after_key<'a>(src: &'a str, key: &str) -> Result<&'a str, ForestMetaError> {
    let needle = format!("\"{key}\"");
    let Some(index) = src.find(&needle) else {
        return Err(ForestMetaError::Invalid(format!("missing {key}")));
    };
    let rest = src[index + needle.len()..].trim_start();
    rest.strip_prefix(':')
        .map(str::trim_start)
        .ok_or_else(|| ForestMetaError::Invalid(format!("missing ':' after {key}")))
}

fn json_u8(src: &str, key: &str) -> Result<u8, ForestMetaError> {
    let value = json_u32(src, key)?;
    u8::try_from(value)
        .map_err(|_| ForestMetaError::Invalid(format!("{key} {value} is out of the u8 range")))
}

fn json_u32(src: &str, key: &str) -> Result<u32, ForestMetaError> {
    let rest = json_after_key(src, key)?;
    let digits = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    digits
        .parse()
        .map_err(|_| ForestMetaError::Invalid(format!("invalid number for {key}")))
}

fn json_u8_array(src: &str, key: &str) -> Result<Vec<u8>, ForestMetaError> {
    let rest = json_after_key(src, key)?;
    let rest = rest
        .strip_prefix('[')
        .ok_or_else(|| ForestMetaError::Invalid(format!("{key} is not an array")))?;
    let end = rest
        .find(']')
        .ok_or_else(|| ForestMetaError::Invalid(format!("unterminated {key} array")))?;
    let mut values = Vec::new();
    for token in rest[..end].split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let value: u8 = token
            .parse()
            .map_err(|_| ForestMetaError::Invalid(format!("invalid {key} entry")))?;
        values.push(value);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ChannelId;

    #[test]
    fn json_round_trips_default_meta() {
        let meta = RegionForestMeta::default();
        let parsed = RegionForestMeta::from_json(&meta.to_json()).unwrap();
        assert_eq!(parsed, meta);
    }

    #[test]
    fn json_accepts_compact_cpp_style() {
        let src = r#"{"version":3,"block_size_po2":4,"region_size_po2":4,"sector_size":512,"channel_depths":[1,1,0,1,1,0,0,0]}"#;
        let meta = RegionForestMeta::from_json(src).unwrap();
        assert_eq!(meta.block_size_po2, 4);
        assert_eq!(
            meta.channel_depths[ChannelId::Sdf.index()],
            ChannelDepth::Bit16
        );
        assert_eq!(
            meta.channel_depths[ChannelId::Color.index()],
            ChannelDepth::Bit8
        );
    }
}
