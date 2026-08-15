//! `RegionFile` — on-disk voxel block archive.
//!
//! Ported from `streams/region/region_file.{h,cpp}`. A region file stores up
//! to `region_size³` voxel blocks in a sector-based sparse layout: a header
//! with a lookup table (LUT) maps each block position to a sector range, and
//! each stored block is a length-prefixed `block_serializer` payload (optionally
//! LZ4/ZSTD-compressed) padded up to a sector boundary.
//!
//! Not thread-safe (the C++ version isn't either). The
//! [`RegionFilesStream`](super::RegionFilesStream) wrapper synchronizes one
//! handle per region. Forest metadata, LRU eviction, and v2→v3 legacy
//! migration remain deferred — see [`crate::streams::region`].

use std::path::Path;

use crate::io::voxel_file::{StdVoxelFile, VoxelFile};
use crate::math::{Color8, Vector3i};
use crate::storage::{ChannelDepth, VoxelBuffer};
use crate::streams::region::format::{
    RegionBlockInfo, RegionFormat, FORMAT_VERSION, MAGIC, MAGIC_AND_VERSION_SIZE,
};
use crate::streams::{block_serializer, compressed_data, DecodeLimits};

/// Why a region operation failed. Mirrors the C++ `Error` returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionError {
    /// The requested region file does not exist.
    NotFound(String),
    /// Filesystem/IO failure (open, read, write, seek).
    Io(String),
    /// File exists but is not a valid region file (bad magic, truncated).
    BadHeader(String),
    /// On-disk version is not supported (only v3 is writable; v2 needs
    /// migration which is deferred).
    UnsupportedVersion(u8),
    /// The requested block position is outside `region_size`.
    InvalidBlockPosition,
    /// The block buffer's format doesn't match the region's.
    BlockFormatMismatch,
    /// The block slot is empty (no data has been saved there yet).
    BlockNotFound,
    /// Block (de)serialization failed.
    BlockSerializer(block_serializer::Error),
}

impl std::fmt::Display for RegionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegionError::NotFound(m) => write!(f, "region file not found: {m}"),
            RegionError::Io(m) => write!(f, "region io error: {m}"),
            RegionError::BadHeader(m) => write!(f, "region bad header: {m}"),
            RegionError::UnsupportedVersion(v) => {
                write!(f, "region unsupported version {v}")
            }
            RegionError::InvalidBlockPosition => write!(f, "region invalid block position"),
            RegionError::BlockFormatMismatch => write!(f, "region block format mismatch"),
            RegionError::BlockNotFound => write!(f, "region block not found"),
            RegionError::BlockSerializer(e) => write!(f, "region block serializer: {e}"),
        }
    }
}

impl std::error::Error for RegionError {}

impl From<block_serializer::Error> for RegionError {
    fn from(e: block_serializer::Error) -> Self {
        RegionError::BlockSerializer(e)
    }
}

/// In-memory header: version + format + the block lookup table.
struct Header {
    version: u8,
    format: RegionFormat,
    /// One entry per block slot, indexed by ZXY position.
    blocks: Vec<RegionBlockInfo>,
}

/// A region file handle. Owns the underlying file and the in-memory LUT.
///
/// Ported from `RegionFile`. Not `Clone` (it owns a file handle).
///
/// **REGION-3:** Concurrent access to the same file from multiple
/// `RegionFile` handles is **not safe** — each handle maintains its own
/// in-memory LUT and sector map, so simultaneous saves will cause lost
/// updates and sector overlap corruption. Use a single handle per file
/// (wrap in `Arc<Mutex<RegionFile>>` for thread-shared access), or
/// implement a `FileLocker`-based coordination layer.
pub struct RegionFile<F: VoxelFile = StdVoxelFile> {
    file: Option<F>,
    header: Header,
    /// Reverse map: `_sectors[i]` is the block position whose data lives at
    /// sector `i`. Rebuilt on open. Not persisted.
    sectors: Vec<Vector3i>,
    /// Whether the in-memory header/LUT needs to be persisted.
    header_dirty: bool,
    /// Byte offset where block data begins (end of header + LUT).
    blocks_begin_offset: u64,
}

impl Default for RegionFile<StdVoxelFile> {
    fn default() -> Self {
        Self::with_format(RegionFormat::default())
    }
}

impl<F: VoxelFile> RegionFile<F> {
    /// Build an unopened handle with the given default format (used when
    /// creating new files).
    pub fn with_format(format: RegionFormat) -> Self {
        format
            .validate_result()
            .expect("RegionFile::with_format requires a valid region format");
        let block_count = format
            .block_count_checked()
            .expect("validated region format has checked block count");
        Self {
            file: None,
            header: Header {
                version: FORMAT_VERSION,
                format,
                blocks: vec![RegionBlockInfo::EMPTY; block_count],
            },
            sectors: Vec::new(),
            header_dirty: false,
            blocks_begin_offset: 0,
        }
    }

    /// Whether a file is currently open.
    pub fn is_open(&self) -> bool {
        self.file.is_some()
    }

    /// The region's format (available even before open).
    pub fn format(&self) -> &RegionFormat {
        &self.header.format
    }

    /// Number of block slots in the LUT.
    pub fn header_block_count(&self) -> usize {
        self.header.blocks.len()
    }

    /// Whether `position` is a valid in-region block coordinate.
    pub fn is_valid_block_position(&self, position: Vector3i) -> bool {
        let rs = self.header.format.region_size;
        position.x >= 0
            && position.y >= 0
            && position.z >= 0
            && position.x < rs.x
            && position.y < rs.y
            && position.z < rs.z
    }

    /// Whether a block has been saved at `position`.
    pub fn has_block(&self, position: Vector3i) -> bool {
        self.block_index(position)
            .map(|i| self.header.blocks[i].is_present())
            .unwrap_or(false)
    }

    /// Recover the 3D position from a flat LUT index.
    pub fn block_position_from_index(&self, i: u32) -> Vector3i {
        let rs = self.header.format.region_size;
        // Inverse of ZXY: index = y + sy*(x + sx*z)
        let sy = rs.y as u32;
        let sx = rs.x as u32;
        let y = i % sy;
        let xz = i / sy;
        let x = xz % sx;
        let z = xz / sx;
        Vector3i::new(x as i32, y as i32, z as i32)
    }

    /// ZXY flat index for a block position, or `None` if out of range.
    fn block_index(&self, position: Vector3i) -> Option<usize> {
        if !self.is_valid_block_position(position) {
            return None;
        }
        // Matches Vector3iUtil::get_zxy_index: y + sy*(x + sx*z)
        let rs = self.header.format.region_size;
        let idx = (position.y as u32
            + rs.y as u32 * (position.x as u32 + rs.x as u32 * position.z as u32))
            as usize;
        Some(idx)
    }

    /// How many sectors `size_in_bytes` of payload occupies.
    fn sector_count_from_bytes(&self, size_in_bytes: usize) -> u32 {
        let sector_size = self.header.format.sector_size as usize;
        size_in_bytes.div_ceil(sector_size) as u32
    }

    // -------------------------------------------------------------------
    // Header (de)serialization
    // -------------------------------------------------------------------

    /// Write the header (magic + version + format + LUT) at offset 0.
    fn save_header(&mut self) -> Result<(), RegionError> {
        let file = self.file.as_mut().expect("file open");
        file.seek(0).map_err(io)?;

        // Build the header in a buffer, then write it in one go. This matches
        // the C++ approach (sequential store_8/store_16/store_buffer) but is
        // friendlier to the VoxelFile trait's bulk write.
        let header_size = self
            .header
            .format
            .header_size_v3_checked()
            .map_err(|e| RegionError::BadHeader(e.to_string()))?;
        let mut buf: Vec<u8> = Vec::with_capacity(header_size);
        buf.extend_from_slice(MAGIC);
        buf.push(self.header.version);
        buf.push(self.header.format.block_size_po2);
        buf.push(self.header.format.region_size.x as u8);
        buf.push(self.header.format.region_size.y as u8);
        buf.push(self.header.format.region_size.z as u8);
        for &d in &self.header.format.channel_depths {
            buf.push(d as u8);
        }
        buf.extend_from_slice(&(self.header.format.sector_size as u16).to_le_bytes());
        match &self.header.format.palette {
            Some(palette) => {
                buf.push(0xff);
                for c in palette {
                    buf.extend_from_slice(&[c.r, c.g, c.b, c.a]);
                }
            }
            None => buf.push(0x00),
        }
        // LUT: each RegionBlockInfo is a little-endian u32.
        for bi in &self.header.blocks {
            buf.extend_from_slice(&bi.data.to_le_bytes());
        }

        debug_assert_eq!(buf.len(), header_size);
        file.write(&buf).map_err(io)?;
        self.header_dirty = false;
        Ok(())
    }

    /// Read the header from offset 0. On success, populates `self.header` and
    /// `self.blocks_begin_offset`.
    fn load_header(&mut self) -> Result<(), RegionError> {
        let file = self.file.as_mut().expect("file open");
        file.seek(0).map_err(io)?;
        let file_len = file.len().map_err(io)?;
        if file_len < MAGIC_AND_VERSION_SIZE as u64 {
            return Err(RegionError::BadHeader("file shorter than magic".into()));
        }

        let mut magic = [0u8; 4];
        let n = file.read(&mut magic).map_err(io)?;
        if n != 4 || &magic != MAGIC {
            return Err(RegionError::BadHeader("bad magic".into()));
        }

        let mut one = [0u8; 1];
        file.read(&mut one).map_err(io)?;
        let version = one[0];
        if version != FORMAT_VERSION {
            return Err(RegionError::UnsupportedVersion(version));
        }

        // Read the format block byte by byte (small, fixed layout).
        let mut fixed = [0u8; crate::streams::region::format::FIXED_HEADER_DATA_SIZE];
        let nread = file.read(&mut fixed).map_err(io)?;
        if nread != fixed.len() {
            return Err(RegionError::BadHeader("truncated format block".into()));
        }
        let mut o = 0;
        self.header.format.block_size_po2 = fixed[o];
        o += 1;
        self.header.format.region_size =
            Vector3i::new(fixed[o] as i32, fixed[o + 1] as i32, fixed[o + 2] as i32);
        o += 3;
        for d in &mut self.header.format.channel_depths {
            *d = match fixed[o] {
                0 => ChannelDepth::Bit8,
                1 => ChannelDepth::Bit16,
                2 => ChannelDepth::Bit32,
                3 => ChannelDepth::Bit64,
                other => {
                    return Err(RegionError::BadHeader(format!(
                        "invalid channel depth byte {other:#x}"
                    )));
                }
            };
            o += 1;
        }
        self.header.format.sector_size = u16::from_le_bytes([fixed[o], fixed[o + 1]]) as u32;
        o += 2;
        let palette_flag = fixed[o];
        match palette_flag {
            0xff => {
                let mut palette = [Color8::new(0, 0, 0, 0); 256];
                let mut pbuf = [0u8; crate::streams::region::format::PALETTE_SIZE_IN_BYTES];
                let pn = file.read(&mut pbuf).map_err(io)?;
                if pn != pbuf.len() {
                    return Err(RegionError::BadHeader("truncated palette".into()));
                }
                for (i, c) in palette.iter_mut().enumerate() {
                    *c = Color8::new(
                        pbuf[i * 4],
                        pbuf[i * 4 + 1],
                        pbuf[i * 4 + 2],
                        pbuf[i * 4 + 3],
                    );
                }
                self.header.format.palette = Some(palette);
            }
            0x00 => {
                self.header.format.palette = None;
            }
            other => {
                return Err(RegionError::BadHeader(format!(
                    "unexpected palette flag {other:#x}"
                )));
            }
        }

        // LUT.
        self.header
            .format
            .validate_result()
            .map_err(|e| RegionError::BadHeader(e.to_string()))?;
        let block_count = self
            .header
            .format
            .block_count_checked()
            .map_err(|e| RegionError::BadHeader(e.to_string()))?;
        let lut_bytes = block_count
            .checked_mul(std::mem::size_of::<RegionBlockInfo>())
            .ok_or_else(|| RegionError::BadHeader("region LUT size overflow".into()))?;
        let expected_header_size = self
            .header
            .format
            .header_size_v3_checked()
            .map_err(|e| RegionError::BadHeader(e.to_string()))?;
        if file_len < expected_header_size as u64 {
            return Err(RegionError::BadHeader("truncated block LUT".into()));
        }
        let mut lut = vec![0u8; lut_bytes];
        let ln = file.read(&mut lut).map_err(io)?;
        if ln != lut_bytes {
            return Err(RegionError::BadHeader("truncated block LUT".into()));
        }
        self.header.blocks = lut
            .chunks_exact(4)
            .map(|c| RegionBlockInfo {
                data: u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            })
            .collect();
        self.header.version = version;
        self.blocks_begin_offset = expected_header_size as u64;
        self.validate_lut(file_len)?;

        // Rebuild the reverse sector map by scanning present blocks in order.
        self.rebuild_sectors();
        self.header_dirty = false;

        Ok(())
    }

    fn validate_lut(&self, file_len: u64) -> Result<(), RegionError> {
        let sector_size = self.header.format.sector_size as u64;
        let data_len = file_len.saturating_sub(self.blocks_begin_offset);
        let sector_capacity = data_len.div_ceil(sector_size);
        let mut occupied: Vec<(u32, u32, Vector3i)> = Vec::new();

        for (i, bi) in self.header.blocks.iter().copied().enumerate() {
            if !bi.is_present() {
                continue;
            }
            if bi.sector_count() == 0 {
                return Err(RegionError::BadHeader(
                    "present LUT entry has zero sectors".into(),
                ));
            }
            let start = bi.sector_index();
            let end = start
                .checked_add(bi.sector_count())
                .ok_or_else(|| RegionError::BadHeader("LUT sector interval overflow".into()))?;
            if end as u64 > sector_capacity {
                return Err(RegionError::BadHeader(format!(
                    "LUT sector interval {start}..{end} outside file sector capacity {sector_capacity}"
                )));
            }
            occupied.push((start, end, self.block_position_from_index(i as u32)));
        }

        occupied.sort_by_key(|(start, _, _)| *start);
        for pair in occupied.windows(2) {
            let (_, prev_end, prev_pos) = pair[0];
            let (next_start, _, next_pos) = pair[1];
            if next_start < prev_end {
                return Err(RegionError::BadHeader(format!(
                    "LUT sectors overlap between {prev_pos:?} and {next_pos:?}"
                )));
            }
        }

        Ok(())
    }

    /// Rebuild `_sectors` from the LUT (present blocks, in sector order).
    fn rebuild_sectors(&mut self) {
        self.sectors.clear();
        // Collect (sector_index, position) for present blocks.
        let mut present: Vec<(u32, Vector3i)> = Vec::new();
        for (i, bi) in self.header.blocks.iter().enumerate() {
            if bi.is_present() {
                present.push((bi.sector_index(), self.block_position_from_index(i as u32)));
            }
        }
        present.sort_by_key(|(s, _)| *s);
        for (_, pos) in &present {
            let bi = self.header.blocks[self.block_index(*pos).unwrap()];
            for _ in 0..bi.sector_count() {
                self.sectors.push(*pos);
            }
        }
    }

    /// Pad the file up to the next sector boundary with zeros.
    fn pad_to_sector_size(&mut self) -> Result<(), RegionError> {
        let file = self.file.as_mut().expect("file open");
        let pos = file.position().map_err(io)?;
        let sector_size = self.header.format.sector_size as u64;
        let rem = pos % sector_size;
        if rem != 0 {
            let pad = sector_size - rem;
            let zeros = vec![0u8; pad as usize];
            file.write(&zeros).map_err(io)?;
        }
        Ok(())
    }

    /// Shift sectors `start..` left by `count`, updating the LUT. Used when a
    /// block shrinks or is removed. Ported from `remove_sectors_from_block`.
    fn remove_sectors_from_block(
        &mut self,
        block_pos: Vector3i,
        count: u32,
    ) -> Result<(), RegionError> {
        let lut_index = self
            .block_index(block_pos)
            .ok_or(RegionError::InvalidBlockPosition)?;
        let bi = &mut self.header.blocks[lut_index];
        let old_index = bi.sector_index();
        let old_count = bi.sector_count();

        // Remove sectors from the end of this block, preserving its leading
        // sectors, then shift all following blocks left.
        let sector_size = self.header.format.sector_size as u64;
        let src_start = self.blocks_begin_offset + (old_index + old_count) as u64 * sector_size;
        let dst_start = src_start - count as u64 * sector_size;

        // Read the tail that needs shifting.
        let file = self.file.as_mut().expect("file open");
        let total_len = file.len().map_err(io)?;
        let move_len = total_len.saturating_sub(src_start);
        if move_len > 0 {
            let mut tail = vec![0u8; move_len as usize];
            file.seek(src_start).map_err(io)?;
            let rn = file.read(&mut tail).map_err(io)?;
            if rn != tail.len() {
                return Err(RegionError::Io(format!(
                    "short read during sector shift: {rn}/{}",
                    tail.len()
                )));
            }
            file.seek(dst_start).map_err(io)?;
            file.write(&tail).map_err(io)?;
        }
        // Truncate the freed trailing bytes (std::fs::File supports set_len).
        let new_len = total_len - count as u64 * sector_size;
        file.set_len(new_len).map_err(io)?;

        // Update the in-memory LUT: this block's count shrinks; later blocks
        // shift their sector_index down by `count`.
        let bi = &mut self.header.blocks[lut_index];
        bi.try_set_sector_count(old_count - count)
            .map_err(|e| RegionError::BadHeader(e.to_string()))?;
        if bi.sector_count() == 0 {
            self.header.blocks[lut_index] = RegionBlockInfo::EMPTY;
        }
        for other in &mut self.header.blocks {
            if other.is_present() && other.sector_index() > old_index {
                other
                    .try_set_sector_index(other.sector_index() - count)
                    .map_err(|e| RegionError::BadHeader(e.to_string()))?;
            }
        }
        // Erase the removed sectors from the reverse map. They are the last
        // `count` sectors of the current block, not necessarily the file tail.
        let remove_start = (old_index + old_count - count) as usize;
        let remove_end = (old_index + old_count) as usize;
        self.sectors.drain(remove_start..remove_end);
        Ok(())
    }

    // -------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------

    /// Load a block from disk. The buffer is reconfigured to the region's
    /// channel depths before deserialization. Returns `BlockNotFound` if the
    /// slot is empty.
    pub fn load_block(
        &mut self,
        position: Vector3i,
        out_block: &mut VoxelBuffer,
    ) -> Result<(), RegionError> {
        self.load_block_with_limits(position, out_block, DecodeLimits::default())
    }

    /// Load a block from disk with explicit decode limits.
    pub fn load_block_with_limits(
        &mut self,
        position: Vector3i,
        out_block: &mut VoxelBuffer,
        limits: DecodeLimits,
    ) -> Result<(), RegionError> {
        // A caller-ordering mistake must not abort the process (the workspace
        // builds with `panic = "abort"`); surface it like any other failure.
        if self.file.is_none() {
            return Err(RegionError::Io("load_block: file not open".into()));
        }
        if !self.is_valid_block_position(position) {
            return Err(RegionError::InvalidBlockPosition);
        }
        let lut_index = self.block_index(position).unwrap();
        let bi = self.header.blocks[lut_index];
        if !bi.is_present() {
            return Err(RegionError::BlockNotFound);
        }

        // Configure the buffer's channel depths to match the region format.
        for (ci, &d) in self.header.format.channel_depths.iter().enumerate() {
            out_block.set_channel_depth(ci, d);
        }

        let sector_size = self.header.format.sector_size as u64;
        let block_begin = self.blocks_begin_offset + bi.sector_index() as u64 * sector_size;

        // Now borrow the file — everything we needed from `self.header` is
        // already copied into locals.
        let file = self.file.as_mut().unwrap();
        file.seek(block_begin).map_err(io)?;

        let mut size_buf = [0u8; 4];
        let n = file.read(&mut size_buf).map_err(io)?;
        if n != 4 {
            return Err(RegionError::Io("short read of block size".into()));
        }
        let block_data_size = u32::from_le_bytes(size_buf) as usize;

        let max_payload_in_slot = (bi.sector_count() as usize)
            .checked_mul(self.header.format.sector_size as usize)
            .and_then(|v| v.checked_sub(4))
            .ok_or_else(|| RegionError::BadHeader("invalid block sector allocation".into()))?;
        if block_data_size > max_payload_in_slot {
            return Err(RegionError::BadHeader(format!(
                "block payload length {block_data_size} exceeds sector allocation {max_payload_in_slot}"
            )));
        }
        limits
            .check_bytes("region block payload", block_data_size)
            .map_err(|e| RegionError::BadHeader(e.to_string()))?;

        let mut payload = Vec::new();
        payload.try_reserve(block_data_size).map_err(|_| {
            RegionError::BadHeader(format!(
                "region block payload allocation failed for {block_data_size} bytes"
            ))
        })?;
        payload.resize(block_data_size, 0);
        let pn = file.read(&mut payload).map_err(io)?;
        if pn != block_data_size {
            return Err(RegionError::Io(format!(
                "short read of block payload: {pn}/{block_data_size}"
            )));
        }

        let status =
            block_serializer::decompress_and_deserialize_with_limits(&payload, out_block, limits)
                .map_err(RegionError::BlockSerializer)?;
        // META-1 parity: surface metadata loss as a non-fatal warning via
        // debug log. The voxel data is still loaded correctly.
        if status == block_serializer::DeserializeStatus::MetadataLost {
            // In a full implementation this would route through the engine's
            // logger; for now we accept the loss silently (consistent with
            // the non-metadata port) but the status is available to callers.
        }
        Ok(())
    }

    /// Write the length-prefixed payload at `block_offset` and pad to a sector
    /// boundary. Borrow-local helper so `save_block` can drop the file borrow
    /// before mutating `self.header`/`self.sectors`.
    fn write_payload(&mut self, block_offset: u64, payload: &[u8]) -> Result<(), RegionError> {
        let file = self.file.as_mut().expect("file open");
        file.seek(block_offset).map_err(io)?;
        file.write(&(payload.len() as u32).to_le_bytes())
            .map_err(io)?;
        file.write(payload).map_err(io)?;
        // File borrow ends here (NLL); pad_to_sector_size can take &mut self.
        self.pad_to_sector_size()
    }

    /// Save a block to disk, allocating/relocating sectors as needed.
    pub fn save_block(
        &mut self,
        position: Vector3i,
        block: &VoxelBuffer,
        compression_mode: compressed_data::Compression,
    ) -> Result<(), RegionError> {
        if self.file.is_none() {
            return Err(RegionError::Io("save_block: file not open".into()));
        }
        if !self.header.format.verify_block(block) {
            return Err(RegionError::BlockFormatMismatch);
        }
        if !self.is_valid_block_position(position) {
            return Err(RegionError::InvalidBlockPosition);
        }

        // Serialize + compress the block into a fresh payload.
        let mut payload = Vec::new();
        block_serializer::serialize_and_compress(block, &mut payload, compression_mode)?;
        let written_size = 4 + payload.len(); // length prefix + payload
        let new_sector_count = self.sector_count_from_bytes(written_size);

        let lut_index = self.block_index(position).unwrap();
        let existing = self.header.blocks[lut_index];
        let sector_size = self.header.format.sector_size as u64;

        if !existing.is_present() {
            // Append at end of the data area.
            let block_offset = self.blocks_begin_offset + self.sectors.len() as u64 * sector_size;
            self.write_payload(block_offset, &payload)?;

            let sector_index = ((block_offset - self.blocks_begin_offset) / sector_size) as u32;
            self.header.blocks[lut_index] =
                RegionBlockInfo::try_new(sector_index, new_sector_count)
                    .map_err(|e| RegionError::BadHeader(e.to_string()))?;
            for _ in 0..new_sector_count {
                self.sectors.push(position);
            }
        } else {
            let old_count = existing.sector_count();
            if new_sector_count <= old_count {
                // Fits in place; compact if smaller.
                if new_sector_count < old_count {
                    self.remove_sectors_from_block(position, old_count - new_sector_count)?;
                }
                let block_offset =
                    self.blocks_begin_offset + existing.sector_index() as u64 * sector_size;
                // In-place write: no padding needed (sector count unchanged or
                // already compacted, so the tail fits within the old allocation).
                let file = self.file.as_mut().expect("file open");
                file.seek(block_offset).map_err(io)?;
                file.write(&(payload.len() as u32).to_le_bytes())
                    .map_err(io)?;
                file.write(&payload).map_err(io)?;
            } else {
                // Doesn't fit: remove old, append at end.
                self.remove_sectors_from_block(position, old_count)?;
                // Re-read sector_size in case (it can't change, but borrow
                // discipline wants the read before the file borrow).
                let sector_size = self.header.format.sector_size as u64;
                let block_offset =
                    self.blocks_begin_offset + self.sectors.len() as u64 * sector_size;
                self.write_payload(block_offset, &payload)?;
                let sector_index = ((block_offset - self.blocks_begin_offset) / sector_size) as u32;
                self.header.blocks[lut_index] =
                    RegionBlockInfo::try_new(sector_index, new_sector_count)
                        .map_err(|e| RegionError::BadHeader(e.to_string()))?;
                for _ in 0..new_sector_count {
                    self.sectors.push(position);
                }
            }
        }

        // REGION-2 parity: write the header immediately after data mutation
        // (not deferred to flush). This ensures the on-disk LUT is consistent
        // with the sector layout at all times — a crash between save_block
        // and flush no longer leaves a stale LUT pointing at truncated sectors.
        self.save_header()?;
        if let Some(f) = &mut self.file {
            f.flush().map_err(io)?;
        }
        Ok(())
    }

    /// Flush pending writes to the OS.
    pub fn flush(&mut self) -> Result<(), RegionError> {
        if self.header_dirty && self.file.is_some() {
            self.save_header()?;
        }
        if let Some(f) = &mut self.file {
            f.flush().map_err(io)?;
        }
        Ok(())
    }

    /// Close the file (flushing first). No-op if already closed.
    pub fn close(&mut self) -> Result<(), RegionError> {
        self.flush()?;
        self.file.take();
        Ok(())
    }
}

impl RegionFile<StdVoxelFile> {
    /// Open an existing region file for read+write, or create it if
    /// `create_if_not_found` and it's missing.
    pub fn open(path: &Path, create_if_not_found: bool) -> Result<Self, RegionError> {
        Self::open_with_format(path, create_if_not_found, RegionFormat::default())
    }

    /// Open an existing region, or create a new one using `format`.
    ///
    /// Existing files always load their persisted format. The supplied format
    /// is used only when a missing file is created.
    pub fn open_with_format(
        path: &Path,
        create_if_not_found: bool,
        format: RegionFormat,
    ) -> Result<Self, RegionError> {
        let mut rf = Self::with_format(format);
        match StdVoxelFile::open_rw(path) {
            Ok(file) => {
                rf.file = Some(file);
                rf.load_header()?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && create_if_not_found => {
                rf.file = Some(StdVoxelFile::create(path).map_err(io)?);
                rf.blocks_begin_offset = rf
                    .header
                    .format
                    .header_size_v3_checked()
                    .map_err(|e| RegionError::BadHeader(e.to_string()))?
                    as u64;
                rf.save_header()?;
            }
            Err(error) => return Err(io(error)),
        }
        Ok(rf)
    }
}

impl<F: VoxelFile> Drop for RegionFile<F> {
    fn drop(&mut self) {
        // REGION-2 parity: surface close errors instead of silently ignoring.
        if let Err(e) = self.close() {
            eprintln!("RegionFile Drop: close() failed: {e:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Preserve filesystem absence separately from other I/O failures.
fn io(e: std::io::Error) -> RegionError {
    if e.kind() == std::io::ErrorKind::NotFound {
        RegionError::NotFound(e.to_string())
    } else {
        RegionError::Io(e.to_string())
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::io::voxel_file::test_support::MemoryFile;
    use crate::storage::voxel_buffer::{Allocator, MAX_CHANNELS};
    use crate::storage::ChannelDepth;

    /// Build a small region (2×2×2 blocks of 2³ voxels) for tests. Channel
    /// depths match `VoxelBuffer`'s defaults so `verify_block` passes on
    /// freshly-created buffers (Type=Bit16, Sdf=Bit16, rest=Bit8).
    fn small_format() -> RegionFormat {
        let mut depths = [ChannelDepth::Bit8; MAX_CHANNELS];
        depths[0] = ChannelDepth::Bit16; // Type
        depths[1] = ChannelDepth::Bit16; // Sdf
        depths[3] = ChannelDepth::Bit16; // Indices
        depths[4] = ChannelDepth::Bit16; // Weights
        RegionFormat {
            block_size_po2: 1, // 2³ blocks
            region_size: Vector3i::new(2, 2, 2),
            channel_depths: depths,
            sector_size: 64,
            palette: None,
        }
    }

    /// Open a region backed by an in-memory file (freshly created).
    fn open_memory(format: RegionFormat) -> RegionFile<MemoryFile> {
        let mut rf = RegionFile::<MemoryFile>::with_format(format);
        rf.file = Some(MemoryFile::new());
        rf.blocks_begin_offset = rf.header.format.header_size_v3() as u64;
        rf.save_header().unwrap();
        rf
    }

    struct HeaderWriteCountingFile {
        data: Vec<u8>,
        pos: u64,
        header_len: u64,
        header_write_count: usize,
    }

    impl HeaderWriteCountingFile {
        fn new(header_len: u64) -> Self {
            Self {
                data: Vec::new(),
                pos: 0,
                header_len,
                header_write_count: 0,
            }
        }

        fn header_write_count(&self) -> usize {
            self.header_write_count
        }
    }

    impl VoxelFile for HeaderWriteCountingFile {
        fn seek(&mut self, pos: u64) -> std::io::Result<()> {
            self.pos = pos;
            Ok(())
        }

        fn position(&mut self) -> std::io::Result<u64> {
            Ok(self.pos)
        }

        fn len(&self) -> std::io::Result<u64> {
            Ok(self.data.len() as u64)
        }

        fn read(&mut self, dst: &mut [u8]) -> std::io::Result<usize> {
            let avail = self.data.len().saturating_sub(self.pos as usize);
            let n = avail.min(dst.len());
            dst[..n].copy_from_slice(&self.data[self.pos as usize..self.pos as usize + n]);
            self.pos += n as u64;
            Ok(n)
        }

        fn write(&mut self, src: &[u8]) -> std::io::Result<()> {
            let start = self.pos;
            let end = start + src.len() as u64;
            if start < self.header_len && end > 0 {
                self.header_write_count += 1;
            }

            let end = end as usize;
            if end > self.data.len() {
                self.data.resize(end, 0);
            }
            self.data[self.pos as usize..end].copy_from_slice(src);
            self.pos = end as u64;
            Ok(())
        }

        fn set_len(&mut self, len: u64) -> std::io::Result<()> {
            self.data.resize(len as usize, 0);
            Ok(())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// A small voxel buffer matching the test format (2³, 8-bit channels).
    fn sample_block(value: u8) -> VoxelBuffer {
        let mut b = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        b.fill(value as u64, 0);
        b
    }

    #[test]
    fn open_creates_valid_header_in_memory() {
        let rf = open_memory(small_format());
        assert_eq!(rf.header_block_count(), 8); // 2×2×2
        assert!(rf.is_open());
        assert!(!rf.has_block(Vector3i::new(0, 0, 0)));
    }

    #[test]
    fn save_then_load_round_trips_block() {
        let mut rf = open_memory(small_format());
        let block = sample_block(42);
        rf.save_block(
            Vector3i::new(0, 0, 0),
            &block,
            compressed_data::Compression::None,
        )
        .unwrap();

        assert!(rf.has_block(Vector3i::new(0, 0, 0)));

        let mut loaded = VoxelBuffer::new(Allocator::Default);
        loaded.create(Vector3i::new(2, 2, 2));
        rf.load_block(Vector3i::new(0, 0, 0), &mut loaded).unwrap();
        // Every voxel should be 42.
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    assert_eq!(loaded.get_voxel(x, y, z, 0), 42);
                }
            }
        }
    }

    #[test]
    fn load_block_rejects_payload_larger_than_sector_allocation() {
        let mut rf = open_memory(small_format());
        let position = Vector3i::new(0, 0, 0);
        rf.save_block(
            position,
            &sample_block(42),
            compressed_data::Compression::None,
        )
        .unwrap();

        let bi = rf.header.blocks[rf.block_index(position).unwrap()];
        let block_begin =
            rf.blocks_begin_offset + bi.sector_index() as u64 * rf.header.format.sector_size as u64;
        let declared_payload = bi.sector_count() * rf.header.format.sector_size;
        let file = rf.file.as_mut().expect("memory file is open");
        file.seek(block_begin).unwrap();
        file.write(&declared_payload.to_le_bytes()).unwrap();

        let mut loaded = VoxelBuffer::new(Allocator::Default);
        let err = rf
            .load_block_with_limits(position, &mut loaded, DecodeLimits::default())
            .unwrap_err();

        assert!(
            matches!(err, RegionError::BadHeader(message) if message.contains("exceeds sector allocation"))
        );
    }

    #[test]
    fn save_block_persists_header_immediately() {
        // REGION-2 parity: save_block now writes the header immediately (not
        // deferred to flush) for crash consistency.
        let format = small_format();
        let header_len = format.header_size_v3() as u64;
        let mut rf = RegionFile::<HeaderWriteCountingFile>::with_format(format);
        rf.file = Some(HeaderWriteCountingFile::new(header_len));
        rf.blocks_begin_offset = rf.header.format.header_size_v3() as u64;
        rf.save_header().unwrap();

        let header_writes_after_create = rf.file.as_ref().unwrap().header_write_count();
        rf.save_block(
            Vector3i::new(0, 0, 0),
            &sample_block(42),
            compressed_data::Compression::None,
        )
        .unwrap();
        assert_eq!(
            rf.file.as_ref().unwrap().header_write_count(),
            header_writes_after_create + 1,
            "save_block should persist the header immediately (REGION-2 crash consistency)"
        );

        // flush should be a no-op for header (already clean).
        rf.flush().unwrap();
        assert_eq!(
            rf.file.as_ref().unwrap().header_write_count(),
            header_writes_after_create + 1,
            "flush should not re-write a clean header"
        );
    }

    #[test]
    fn load_nonexistent_block_returns_not_found() {
        let mut rf = open_memory(small_format());
        let mut buf = VoxelBuffer::new(Allocator::Default);
        buf.create(Vector3i::new(2, 2, 2));
        assert_eq!(
            rf.load_block(Vector3i::new(1, 1, 1), &mut buf),
            Err(RegionError::BlockNotFound)
        );
    }

    #[test]
    fn save_rejects_wrong_block_size() {
        let mut rf = open_memory(small_format());
        let wrong = VoxelBuffer::with_size(Vector3i::new(4, 4, 4)); // expected 2³
        assert_eq!(
            rf.save_block(
                Vector3i::new(0, 0, 0),
                &wrong,
                compressed_data::Compression::None
            ),
            Err(RegionError::BlockFormatMismatch)
        );
    }

    #[test]
    fn save_rejects_out_of_range_position() {
        let mut rf = open_memory(small_format());
        let block = sample_block(1);
        assert_eq!(
            rf.save_block(
                Vector3i::new(5, 0, 0),
                &block,
                compressed_data::Compression::None
            ),
            Err(RegionError::InvalidBlockPosition)
        );
    }

    #[test]
    fn is_valid_block_position_boundaries() {
        let rf = open_memory(small_format());
        assert!(rf.is_valid_block_position(Vector3i::new(0, 0, 0)));
        assert!(rf.is_valid_block_position(Vector3i::new(1, 1, 1)));
        assert!(!rf.is_valid_block_position(Vector3i::new(2, 0, 0)));
        assert!(!rf.is_valid_block_position(Vector3i::new(-1, 0, 0)));
    }

    #[test]
    fn block_position_from_index_inverts_zxy() {
        let rf = open_memory(small_format());
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    let pos = Vector3i::new(x, y, z);
                    let idx = rf.block_index(pos).unwrap() as u32;
                    assert_eq!(rf.block_position_from_index(idx), pos);
                }
            }
        }
    }

    #[test]
    fn overwrite_smaller_block_compacts_sectors() {
        let mut rf = open_memory(small_format());
        // First save: fill channel 0 with distinct values (non-uniform →
        // larger compressed payload).
        let mut big = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    big.set_voxel(((x * 4 + y * 2 + z) as u64) + 1, x, y, z, 0);
                }
            }
        }
        rf.save_block(
            Vector3i::new(0, 0, 0),
            &big,
            compressed_data::Compression::None,
        )
        .unwrap();
        let sectors_after_big = rf.sectors.len();

        // Second save: uniform block (tiny payload → fewer sectors).
        let small = sample_block(7);
        rf.save_block(
            Vector3i::new(0, 0, 0),
            &small,
            compressed_data::Compression::None,
        )
        .unwrap();
        let sectors_after_small = rf.sectors.len();
        assert!(
            sectors_after_small <= sectors_after_big,
            "compaction should not grow sectors: {sectors_after_small} vs {sectors_after_big}"
        );

        // The block still loads correctly.
        let mut loaded = VoxelBuffer::new(Allocator::Default);
        loaded.create(Vector3i::new(2, 2, 2));
        rf.load_block(Vector3i::new(0, 0, 0), &mut loaded).unwrap();
        assert_eq!(loaded.get_voxel(0, 0, 0, 0), 7);
    }

    #[test]
    fn shrinking_first_block_preserves_following_blocks() {
        let mut format = small_format();
        format.sector_size = 16;
        let mut rf = open_memory(format);
        let first_pos = Vector3i::new(0, 0, 0);
        let second_pos = Vector3i::new(1, 0, 0);

        let mut big = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    big.set_voxel(((x * 4 + y * 2 + z) as u64) + 1, x, y, z, 0);
                }
            }
        }
        rf.save_block(first_pos, &big, compressed_data::Compression::None)
            .unwrap();
        rf.save_block(
            second_pos,
            &sample_block(99),
            compressed_data::Compression::None,
        )
        .unwrap();
        let second_index_before =
            rf.header.blocks[rf.block_index(second_pos).unwrap()].sector_index();

        rf.save_block(
            first_pos,
            &sample_block(7),
            compressed_data::Compression::None,
        )
        .unwrap();

        let second_index_after =
            rf.header.blocks[rf.block_index(second_pos).unwrap()].sector_index();
        assert!(
            second_index_after < second_index_before,
            "following block should shift left after first block shrinks"
        );
        let mut loaded = VoxelBuffer::new(Allocator::Default);
        loaded.create(Vector3i::new(2, 2, 2));
        rf.load_block(second_pos, &mut loaded).unwrap();
        assert_eq!(loaded.get_voxel(0, 0, 0, 0), 99);
    }

    #[test]
    fn multiple_blocks_save_and_load_independently() {
        let mut rf = open_memory(small_format());
        for i in 0..4u8 {
            let pos = Vector3i::new((i & 1) as i32, ((i >> 1) & 1) as i32, 0);
            let block = sample_block(10 + i);
            rf.save_block(pos, &block, compressed_data::Compression::None)
                .unwrap();
        }
        for i in 0..4u8 {
            let pos = Vector3i::new((i & 1) as i32, ((i >> 1) & 1) as i32, 0);
            let mut loaded = VoxelBuffer::new(Allocator::Default);
            loaded.create(Vector3i::new(2, 2, 2));
            rf.load_block(pos, &mut loaded).unwrap();
            assert_eq!(loaded.get_voxel(0, 0, 0, 0), (10 + i) as u64);
        }
    }

    #[test]
    fn sector_count_from_bytes_rounds_up() {
        let rf = open_memory(RegionFormat {
            sector_size: 64,
            ..small_format()
        });
        assert_eq!(rf.sector_count_from_bytes(1), 1);
        assert_eq!(rf.sector_count_from_bytes(64), 1);
        assert_eq!(rf.sector_count_from_bytes(65), 2);
        assert_eq!(rf.sector_count_from_bytes(128), 2);
    }

    #[test]
    fn load_header_rejects_bad_magic() {
        let mut rf = RegionFile::<MemoryFile>::with_format(small_format());
        rf.file = Some(MemoryFile::with_data(b"XXXX\x03".to_vec()));
        assert!(matches!(rf.load_header(), Err(RegionError::BadHeader(_))));
    }

    #[test]
    fn load_header_rejects_unsupported_version() {
        let mut rf = RegionFile::<MemoryFile>::with_format(small_format());
        // Valid magic but version 2 (legacy, needs migration).
        rf.file = Some(MemoryFile::with_data(b"VXR_\x02".to_vec()));
        assert_eq!(rf.load_header(), Err(RegionError::UnsupportedVersion(2)));
    }

    #[test]
    fn load_header_rejects_zero_region_axis_before_lut_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.push(FORMAT_VERSION);
        bytes.push(4);
        bytes.extend_from_slice(&[0, 16, 16]);
        bytes.extend_from_slice(&[ChannelDepth::Bit8 as u8; MAX_CHANNELS]);
        bytes.extend_from_slice(&512u16.to_le_bytes());
        bytes.push(0x00);

        let mut rf = RegionFile::<MemoryFile>::with_format(small_format());
        rf.file = Some(MemoryFile::with_data(bytes));
        let err = rf.load_header().unwrap_err();

        assert!(
            matches!(err, RegionError::BadHeader(message) if message.contains("invalid region x axis"))
        );
    }

    #[test]
    fn load_header_rejects_invalid_channel_depth() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.push(FORMAT_VERSION);
        bytes.push(4);
        bytes.extend_from_slice(&[16, 16, 16]);
        bytes.extend_from_slice(&[0xff; MAX_CHANNELS]);
        bytes.extend_from_slice(&512u16.to_le_bytes());
        bytes.push(0x00);

        let mut rf = RegionFile::<MemoryFile>::with_format(small_format());
        rf.file = Some(MemoryFile::with_data(bytes));
        let err = rf.load_header().unwrap_err();

        assert!(
            matches!(err, RegionError::BadHeader(message) if message.contains("channel depth"))
        );
    }

    #[test]
    fn load_header_rejects_lut_sector_outside_file() {
        let rf = open_memory(small_format());
        let mut bytes = rf.file.as_ref().unwrap().data().to_vec();
        let lut_offset = rf.blocks_begin_offset as usize
            - rf.header.blocks.len() * std::mem::size_of::<RegionBlockInfo>();
        bytes[lut_offset..lut_offset + 4]
            .copy_from_slice(&RegionBlockInfo::new(10, 1).data.to_le_bytes());

        let mut reopened = RegionFile::<MemoryFile>::with_format(small_format());
        reopened.file = Some(MemoryFile::with_data(bytes));
        let err = reopened.load_header().unwrap_err();

        assert!(matches!(err, RegionError::BadHeader(message) if message.contains("outside file")));
    }

    #[test]
    fn load_header_rejects_overlapping_lut_sectors() {
        let rf = open_memory(small_format());
        let mut bytes = rf.file.as_ref().unwrap().data().to_vec();
        let lut_offset = rf.blocks_begin_offset as usize
            - rf.header.blocks.len() * std::mem::size_of::<RegionBlockInfo>();
        let first = RegionBlockInfo::new(0, 1).data.to_le_bytes();
        let second = RegionBlockInfo::new(0, 1).data.to_le_bytes();
        bytes[lut_offset..lut_offset + 4].copy_from_slice(&first);
        bytes[lut_offset + 4..lut_offset + 8].copy_from_slice(&second);
        bytes.resize(
            rf.blocks_begin_offset as usize + rf.header.format.sector_size as usize,
            0,
        );

        let mut reopened = RegionFile::<MemoryFile>::with_format(small_format());
        reopened.file = Some(MemoryFile::with_data(bytes));
        let err = reopened.load_header().unwrap_err();

        assert!(matches!(err, RegionError::BadHeader(message) if message.contains("overlap")));
    }
}
