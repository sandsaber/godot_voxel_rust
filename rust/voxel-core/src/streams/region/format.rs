//! Pure-data types for the region-file format.
//!
//! Ported from the `RegionFormat` / `RegionBlockInfo` structs in
//! `streams/region/region_file.h`. These describe the on-disk layout; the
//! actual read/write logic lives in [`super::region_file`].

use crate::math::{Color8, Vector3i};
use crate::storage::voxel_buffer::MAX_CHANNELS;
use crate::storage::{ChannelDepth, VoxelBuffer};

/// File extension for region files. Matches `RegionFormat::FILE_EXTENSION`.
pub const FILE_EXTENSION: &str = "vxr";

/// Latest on-disk version, written by [`super::region_file::RegionFile`].
pub const FORMAT_VERSION: u8 = 3;

/// ASCII magic at the start of every region file. Matches `FORMAT_REGION_MAGIC`.
pub const MAGIC: &[u8; 4] = b"VXR_";

/// Magic (4 bytes) + version (1 byte).
pub const MAGIC_AND_VERSION_SIZE: usize = 5;

/// `block_size_po2(1) + region_size(3) + sector_size(2) + palette_flag(1)` = 7,
/// plus 8 channel-depth bytes.
pub const FIXED_HEADER_DATA_SIZE: usize = 7 + MAX_CHANNELS;

/// Bytes occupied by the palette when present: 256 entries × 4 bytes.
pub const PALETTE_SIZE_IN_BYTES: usize = 256 * 4;

/// Maximum number of blocks along any region axis. Stored as one byte on disk,
/// so the practical limit is 255. Matches `RegionFormat::MAX_BLOCKS_ACROSS`.
pub const MAX_BLOCKS_ACROSS: u32 = 255;

/// 24-bit sector index cap. Matches `RegionBlockInfo::MAX_SECTOR_INDEX`.
pub const MAX_SECTOR_INDEX: u32 = 0xffffff;
/// 8-bit sector count cap. Matches `RegionBlockInfo::MAX_SECTOR_COUNT`.
pub const MAX_SECTOR_COUNT: u32 = 0xff;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionFormatError {
    InvalidRegionAxis {
        axis: &'static str,
        value: i32,
    },
    InvalidBlockSizePo2(u8),
    InvalidSectorSize(u32),
    ByteCountOverflow,
    SectorCountOverflow {
        sectors_per_block: u64,
    },
    SectorIndexOverflow {
        max_potential_sectors: u64,
    },
    HeaderSizeOverflow,
    RegionBlockInfoOverflow {
        field: &'static str,
        value: u32,
        max: u32,
    },
}

impl std::fmt::Display for RegionFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRegionAxis { axis, value } => {
                write!(f, "invalid region {axis} axis {value}")
            }
            Self::InvalidBlockSizePo2(v) => write!(f, "invalid block_size_po2 {v}"),
            Self::InvalidSectorSize(v) => write!(f, "invalid sector_size {v}"),
            Self::ByteCountOverflow => write!(f, "region byte count overflow"),
            Self::SectorCountOverflow { sectors_per_block } => {
                write!(
                    f,
                    "sectors per block {sectors_per_block} exceeds {MAX_SECTOR_COUNT}"
                )
            }
            Self::SectorIndexOverflow {
                max_potential_sectors,
            } => {
                write!(
                    f,
                    "potential sectors {max_potential_sectors} exceeds {MAX_SECTOR_INDEX}"
                )
            }
            Self::HeaderSizeOverflow => write!(f, "region header size overflow"),
            Self::RegionBlockInfoOverflow { field, value, max } => {
                write!(f, "region block info {field} {value} exceeds {max}")
            }
        }
    }
}

impl std::error::Error for RegionFormatError {}

/// Describes the voxel format of a region file. Ported from `RegionFormat`.
///
/// All blocks in a region share this format; it is written once in the header
/// and verified on every `save_block`.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionFormat {
    /// Voxels per cubic block, as a power of two (e.g. 4 → 16³ blocks).
    pub block_size_po2: u8,
    /// Number of blocks along each axis.
    pub region_size: Vector3i,
    /// Per-channel bit depth (fixed at 8 channels).
    pub channel_depths: [ChannelDepth; MAX_CHANNELS],
    /// Block data is stored at offsets that are multiples of this size.
    pub sector_size: u32,
    /// Optional 256-entry color palette.
    pub palette: Option<[Color8; 256]>,
}

impl Default for RegionFormat {
    fn default() -> Self {
        // Matches the C++ RegionFile() ctor defaults.
        Self {
            block_size_po2: 4, // 16³ blocks
            region_size: Vector3i::new(16, 16, 16),
            channel_depths: [ChannelDepth::Bit8; MAX_CHANNELS],
            sector_size: 512,
            palette: None,
        }
    }
}

impl RegionFormat {
    /// Whether `self` is a valid, serializable format. Ported from `validate`.
    pub fn validate(&self) -> bool {
        self.validate_result().is_ok()
    }

    pub fn validate_result(&self) -> Result<(), RegionFormatError> {
        for (axis, value) in [
            ("x", self.region_size.x),
            ("y", self.region_size.y),
            ("z", self.region_size.z),
        ] {
            if value <= 0 || value as u32 >= MAX_BLOCKS_ACROSS {
                return Err(RegionFormatError::InvalidRegionAxis { axis, value });
            }
        }
        if self.block_size_po2 == 0 {
            return Err(RegionFormatError::InvalidBlockSizePo2(self.block_size_po2));
        }
        if self.sector_size == 0 || self.sector_size > u16::MAX as u32 {
            return Err(RegionFormatError::InvalidSectorSize(self.sector_size));
        }

        // Worst-case: every channel fully allocated at max depth.
        let shift = 3u32
            .checked_mul(self.block_size_po2 as u32)
            .ok_or(RegionFormatError::ByteCountOverflow)?;
        let voxels_per_block = 1u64
            .checked_shl(shift)
            .ok_or(RegionFormatError::ByteCountOverflow)?;
        let mut bytes_per_block = 0u64;
        for d in &self.channel_depths {
            let channel_bytes = (d.bit_count() / 8) as u64;
            let bytes = channel_bytes
                .checked_mul(voxels_per_block)
                .ok_or(RegionFormatError::ByteCountOverflow)?;
            bytes_per_block = bytes_per_block
                .checked_add(bytes)
                .ok_or(RegionFormatError::ByteCountOverflow)?;
        }
        let sectors_per_block = bytes_per_block.div_ceil(self.sector_size as u64);
        if sectors_per_block > MAX_SECTOR_COUNT as u64 {
            return Err(RegionFormatError::SectorCountOverflow { sectors_per_block });
        }
        let max_potential_sectors = (self.block_count_checked()? as u64)
            .checked_mul(sectors_per_block)
            .ok_or(RegionFormatError::SectorIndexOverflow {
                max_potential_sectors: u64::MAX,
            })?;
        if max_potential_sectors > MAX_SECTOR_INDEX as u64 {
            return Err(RegionFormatError::SectorIndexOverflow {
                max_potential_sectors,
            });
        }
        let _ = self.header_size_v3_checked()?;
        Ok(())
    }

    /// Whether `block` matches this region's format (size + per-channel depth).
    /// Ported from `verify_block`.
    pub fn verify_block(&self, block: &VoxelBuffer) -> bool {
        let expected_size = Vector3i::splat(1i32 << self.block_size_po2);
        if block.size() != expected_size {
            return false;
        }
        for (i, &expected_depth) in self.channel_depths.iter().enumerate() {
            if block.channel_depth(i) != expected_depth {
                return false;
            }
        }
        true
    }

    /// Byte offset where block data begins (i.e. header end). Matches
    /// `get_header_size_v3`.
    pub fn header_size_v3(&self) -> usize {
        self.header_size_v3_checked()
            .expect("RegionFormat::header_size_v3 requires a valid header size")
    }

    pub fn block_count_checked(&self) -> Result<usize, RegionFormatError> {
        let x = usize::try_from(self.region_size.x).map_err(|_| {
            RegionFormatError::InvalidRegionAxis {
                axis: "x",
                value: self.region_size.x,
            }
        })?;
        let y = usize::try_from(self.region_size.y).map_err(|_| {
            RegionFormatError::InvalidRegionAxis {
                axis: "y",
                value: self.region_size.y,
            }
        })?;
        let z = usize::try_from(self.region_size.z).map_err(|_| {
            RegionFormatError::InvalidRegionAxis {
                axis: "z",
                value: self.region_size.z,
            }
        })?;
        x.checked_mul(y)
            .and_then(|v| v.checked_mul(z))
            .ok_or(RegionFormatError::HeaderSizeOverflow)
    }

    pub fn header_size_v3_checked(&self) -> Result<usize, RegionFormatError> {
        let palette_bytes = if self.palette.is_some() {
            PALETTE_SIZE_IN_BYTES
        } else {
            0
        };
        MAGIC_AND_VERSION_SIZE
            .checked_add(FIXED_HEADER_DATA_SIZE)
            .and_then(|v| v.checked_add(palette_bytes))
            .and_then(|v| {
                self.block_count_checked()
                    .ok()
                    .and_then(|count| count.checked_mul(std::mem::size_of::<RegionBlockInfo>()))
                    .and_then(|lut| v.checked_add(lut))
            })
            .ok_or(RegionFormatError::HeaderSizeOverflow)
    }
}

/// Location and size of one block within the data area. Ported from
/// `RegionBlockInfo`.
///
/// Packed into a single `u32`: bits 31..8 = sector_index (24 bits), bits 7..0
/// = sector_count (8 bits). `data == 0` means the block slot is empty.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct RegionBlockInfo {
    pub data: u32,
}

impl RegionBlockInfo {
    /// An empty/unallocated block slot.
    pub const EMPTY: Self = Self { data: 0 };

    /// Build from sector index + count.
    pub fn new(sector_index: u32, sector_count: u32) -> Self {
        Self::try_new(sector_index, sector_count).expect("validated region block info")
    }

    pub fn try_new(sector_index: u32, sector_count: u32) -> Result<Self, RegionFormatError> {
        if sector_index > MAX_SECTOR_INDEX {
            return Err(RegionFormatError::RegionBlockInfoOverflow {
                field: "sector_index",
                value: sector_index,
                max: MAX_SECTOR_INDEX,
            });
        }
        if sector_count > MAX_SECTOR_COUNT {
            return Err(RegionFormatError::RegionBlockInfoOverflow {
                field: "sector_count",
                value: sector_count,
                max: MAX_SECTOR_COUNT,
            });
        }
        Ok(Self {
            data: (sector_index << 8) | sector_count,
        })
    }

    /// `get_sector_index` — offset into the data area, in sectors.
    #[inline]
    pub fn sector_index(self) -> u32 {
        self.data >> 8
    }

    /// `set_sector_index`.
    #[inline]
    pub fn set_sector_index(&mut self, i: u32) {
        self.try_set_sector_index(i)
            .expect("validated region block info sector index");
    }

    pub fn try_set_sector_index(&mut self, i: u32) -> Result<(), RegionFormatError> {
        if i > MAX_SECTOR_INDEX {
            return Err(RegionFormatError::RegionBlockInfoOverflow {
                field: "sector_index",
                value: i,
                max: MAX_SECTOR_INDEX,
            });
        }
        self.data = (i << 8) | (self.data & 0xff);
        Ok(())
    }

    /// `get_sector_count` — how many consecutive sectors the block occupies.
    #[inline]
    pub fn sector_count(self) -> u32 {
        self.data & 0xff
    }

    /// `set_sector_count`.
    #[inline]
    pub fn set_sector_count(&mut self, c: u32) {
        self.try_set_sector_count(c)
            .expect("validated region block info sector count");
    }

    pub fn try_set_sector_count(&mut self, c: u32) -> Result<(), RegionFormatError> {
        if c > MAX_SECTOR_COUNT {
            return Err(RegionFormatError::RegionBlockInfoOverflow {
                field: "sector_count",
                value: c,
                max: MAX_SECTOR_COUNT,
            });
        }
        self.data = c | (self.data & 0xffffff00);
        Ok(())
    }

    /// Whether this slot is allocated.
    #[inline]
    pub fn is_present(self) -> bool {
        self.data != 0
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn block_info_pack_unpack_round_trips() {
        let bi = RegionBlockInfo::new(0x123456, 0x78);
        assert_eq!(bi.sector_index(), 0x123456);
        assert_eq!(bi.sector_count(), 0x78);
        assert!(bi.is_present());
    }

    #[test]
    fn block_info_empty_is_zero() {
        assert_eq!(RegionBlockInfo::EMPTY.data, 0);
        assert!(!RegionBlockInfo::EMPTY.is_present());
    }

    #[test]
    fn block_info_setter_preserves_other_field() {
        let mut bi = RegionBlockInfo::new(100, 5);
        bi.set_sector_index(200);
        assert_eq!(bi.sector_index(), 200);
        assert_eq!(bi.sector_count(), 5);
        bi.set_sector_count(10);
        assert_eq!(bi.sector_index(), 200);
        assert_eq!(bi.sector_count(), 10);
    }

    #[test]
    fn default_format_is_valid() {
        assert!(RegionFormat::default().validate());
    }

    #[test]
    fn format_rejects_zero_block_size_po2() {
        let mut f = RegionFormat::default();
        f.block_size_po2 = 0;
        assert!(!f.validate());
    }

    #[test]
    fn format_rejects_oversized_region() {
        let mut f = RegionFormat::default();
        f.region_size = Vector3i::new(255, 16, 16); // >= MAX_BLOCKS_ACROSS
        assert!(!f.validate());
    }

    #[test]
    fn format_rejects_negative_region_axis() {
        let mut f = RegionFormat::default();
        f.region_size = Vector3i::new(-1, 16, 16);
        assert!(!f.validate());
    }

    #[test]
    fn format_rejects_zero_region_axis() {
        let mut f = RegionFormat::default();
        f.region_size = Vector3i::new(0, 16, 16);

        assert!(f.validate_result().is_err());
        assert!(!f.validate());
    }

    #[test]
    fn block_info_try_new_rejects_overflow_without_masking() {
        assert!(RegionBlockInfo::try_new(MAX_SECTOR_INDEX + 1, 1).is_err());
        assert!(RegionBlockInfo::try_new(1, MAX_SECTOR_COUNT + 1).is_err());
    }

    #[test]
    fn block_info_try_setters_reject_overflow_without_changing_value() {
        let mut info = RegionBlockInfo::new(7, 8);

        assert!(info.try_set_sector_index(MAX_SECTOR_INDEX + 1).is_err());
        assert_eq!(info.sector_index(), 7);
        assert!(info.try_set_sector_count(MAX_SECTOR_COUNT + 1).is_err());
        assert_eq!(info.sector_count(), 8);
    }

    #[test]
    fn header_size_v3_matches_expected_layout() {
        let f = RegionFormat {
            block_size_po2: 4,
            region_size: Vector3i::new(16, 16, 16),
            channel_depths: [ChannelDepth::Bit8; MAX_CHANNELS],
            sector_size: 512,
            palette: None,
        };
        // magic(4) + version(1) + block_size_po2(1) + region_size(3) +
        // channel_depths(8) + sector_size(2) + palette_flag(1) + LUT(16³×4)
        let lut = 16 * 16 * 16 * 4;
        assert_eq!(f.header_size_v3(), 5 + 1 + 3 + 8 + 2 + 1 + lut);
    }

    #[test]
    fn header_size_includes_palette_when_present() {
        let mut f = RegionFormat::default();
        let without = f.header_size_v3();
        f.palette = Some([Color8::new(0, 0, 0, 0); 256]);
        let with = f.header_size_v3();
        assert_eq!(with - without, PALETTE_SIZE_IN_BYTES);
    }
}
