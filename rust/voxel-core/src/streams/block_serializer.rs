//! `streams::block_serializer` — `VoxelBuffer` ↔ bytes.
//!
//! Ported from `streams/voxel_block_serializer.{h,cpp}`. Serializes a
//! [`VoxelBuffer`] into a compact on-disk byte stream (version 4), with optional
//! LZ4/ZSTD compression via [`crate::streams::compressed_data`].
//!
//! # Wire format (version 4)
//! ```text
//! u8  format_version (= 4)
//! u16 size.x   u16 size.y   u16 size.z
//! for each of 8 channels:
//!     u8  fmt  // low nibble: compression, high nibble: depth
//!     // Compression::None    -> raw voxel bytes (volume * depth_bytes)
//!     // Compression::Uniform -> single raw voxel (depth_bytes)
//! [u32 metadata_size + metadata bytes]   // only when metadata_size > 0
//! u32 trailing_magic (= 0x900df00d)
//! ```
//!
//! # Metadata (narrow R7 slice)
//! The metadata section persists the buffer-wide (block) entry first, then one
//! entry per voxel with local `u16 x, y, z` coordinates — the same layout as
//! the upstream C++ serializer. Entry values are type-tagged bytes; tags `0`
//! (empty) and `1` (`u64`) match the C++ `VoxelMetadata` types exactly, so
//! `MetadataValue::Nil` / `Int` are byte-identical to what C++ writes and reads.
//! Values with no C++ equivalent (`Float`, `Text`, `Bytes`) use app-specific
//! tags starting at 40 (C++ reserves `[32, 40)` for its own custom types, e.g.
//! 32 = Godot `Variant`). A section containing such foreign entries — or any
//! undecodable content — is skipped without failing the load: voxel data must
//! always survive a metadata problem, matching the upstream behavior of
//! ignoring `deserialize_metadata` failures. Direct [`deserialize`] surfaces
//! the loss as [`Error::MetadataSkipped`] after loading voxel data;
//! [`decompress_and_deserialize_with_limits`] reports it as
//! [`DeserializeStatus::MetadataLost`]. Voxel entries are written in sorted
//! position order so output is deterministic.
//!
//! Legacy version-2/3 migration needs the full Godot Variant codec and is
//! deferred (ROADMAP "R7 wide").

use crate::io::serialization::{MemoryReader, MemoryWriter};
use crate::math::Vector3i;
use crate::storage::voxel_buffer::{Compression, MAX_CHANNELS, MAX_SIZE};
use crate::storage::{ChannelDepth, MetadataValue, VoxelBuffer};
use crate::streams::compressed_data;
use crate::streams::decode_limits::{DecodeLimitError, DecodeLimits};

/// Latest on-disk version, written by [`serialize`].
pub const BLOCK_FORMAT_VERSION: u8 = 4;

/// Trailing sanity-check word. `0x900df00d` ("good food"). Matches C++.
pub const BLOCK_TRAILING_MAGIC: u32 = 0x900df00d;
const BLOCK_TRAILING_MAGIC_SIZE: usize = 4;

// Metadata entry type tags. `0` and `1` are the upstream C++ `VoxelMetadata`
// types (TYPE_EMPTY / TYPE_U64) so Nil/Int entries are byte-identical on both
// sides. C++ reserves [32, 40) for its custom types (32 = Godot Variant); our
// extensions start at TYPE_APP_SPECIFIC_BEGIN (40), matching the C++ constant.
const METADATA_TYPE_EMPTY: u8 = 0;
const METADATA_TYPE_U64: u8 = 1;
const METADATA_TYPE_CUSTOM_BEGIN: u8 = 32;
const METADATA_TYPE_APP_SPECIFIC_BEGIN: u8 = 40;
const METADATA_TYPE_F64: u8 = METADATA_TYPE_APP_SPECIFIC_BEGIN;
const METADATA_TYPE_TEXT: u8 = METADATA_TYPE_APP_SPECIFIC_BEGIN + 1;
const METADATA_TYPE_BYTES: u8 = METADATA_TYPE_APP_SPECIFIC_BEGIN + 2;

/// Why (de)serialization failed. Mirrors the `false` returns / `ERR_FAIL_COND`
/// paths in the C++.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Tried to serialize a buffer with zero volume (C++ refuses this).
    EmptyBuffer,
    /// A buffer dimension exceeds `MAX_SIZE` (won't fit a `u16`).
    SizeOverflow,
    /// Reader ran out of bytes mid-field.
    UnexpectedEof,
    /// Trailing `0x900df00d` mismatch — the stream is corrupt or truncated.
    BadTrailingMagic { expected: u32, found: u32 },
    /// Tag/version/compression/depth byte outside the valid range.
    InvalidFormat(String),
    /// Unsupported on-disk version (v2/v3 migration needs the Godot Variant
    /// codec, which is not yet ported).
    UnsupportedVersion(u8),
    /// The stream declared a metadata section whose contents this build cannot
    /// decode (foreign C++ custom/Variant entries, or a corrupt section). The
    /// voxel data itself is still loaded.
    MetadataSkipped,
    /// Compression envelope failure (LZ4/ZSTD).
    Compress(compressed_data::Error),
    /// Declared decoded size exceeded caller-provided limits or allocation failed.
    Limit(DecodeLimitError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::EmptyBuffer => write!(f, "block_serializer: cannot serialize empty buffer"),
            Error::SizeOverflow => write!(f, "block_serializer: buffer dimension exceeds MAX_SIZE"),
            Error::UnexpectedEof => write!(f, "block_serializer: unexpected end of stream"),
            Error::BadTrailingMagic { expected, found } => write!(
                f,
                "block_serializer: bad trailing magic (expected {expected:#x}, found {found:#x})"
            ),
            Error::InvalidFormat(m) => write!(f, "block_serializer: invalid format ({m})"),
            Error::UnsupportedVersion(v) => {
                write!(f, "block_serializer: unsupported version {v} (legacy migration needs Godot Variant codec)")
            }
            Error::MetadataSkipped => write!(
                f,
                "block_serializer: metadata section present but skipped (foreign or undecodable entries)"
            ),
            Error::Compress(e) => write!(f, "block_serializer: compression error: {e}"),
            Error::Limit(e) => write!(f, "block_serializer: decode limit: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<compressed_data::Error> for Error {
    fn from(e: compressed_data::Error) -> Self {
        Error::Compress(e)
    }
}

/// Pack a channel's compression (low nibble) and depth (high nibble) into one
/// byte — the on-disk `fmt` field. Matches the C++ layout.
#[inline]
fn pack_format(compression: Compression, depth: ChannelDepth) -> u8 {
    (compression as u8) | ((depth as u8) << 4)
}

/// Unpack the `fmt` byte into (compression, depth). Returns `Err(bad_nibble)`
/// if either nibble is out of range — the nibble value is returned so the
/// caller can include it in an error message.
fn unpack_format(fmt: u8) -> Result<(Compression, ChannelDepth), u8> {
    let compression = match fmt & 0x0f {
        0 => Compression::None,
        1 => Compression::Uniform,
        other => return Err(other),
    };
    let depth = match (fmt >> 4) & 0x0f {
        0 => ChannelDepth::Bit8,
        1 => ChannelDepth::Bit16,
        2 => ChannelDepth::Bit32,
        3 => ChannelDepth::Bit64,
        other => return Err(other),
    };
    Ok((compression, depth))
}

// ---------------------------------------------------------------------------
// Serialize
// ---------------------------------------------------------------------------

/// Serialize `buffer` into `dst`, clearing it first. Returns the number of
/// bytes written. Ported from `BlockSerializer::serialize` (version 4,
/// including the metadata section — omitted entirely when the buffer carries
/// no metadata).
pub fn serialize(buffer: &VoxelBuffer, dst: &mut Vec<u8>) -> Result<usize, Error> {
    dst.clear();

    let size = buffer.size();
    if size.volume_u64() == 0 {
        return Err(Error::EmptyBuffer);
    }
    if size.x as u32 > MAX_SIZE || size.y as u32 > MAX_SIZE || size.z as u32 > MAX_SIZE {
        return Err(Error::SizeOverflow);
    }

    {
        let mut w = MemoryWriter::little(dst);
        w.store_8(BLOCK_FORMAT_VERSION);
        w.store_16(size.x as u16);
        w.store_16(size.y as u16);
        w.store_16(size.z as u16);

        for ci in 0..MAX_CHANNELS {
            let compression = buffer.channel_compression(ci);
            let depth = buffer.channel_depth(ci);
            w.store_8(pack_format(compression, depth));

            match compression {
                Compression::None => {
                    let bytes = buffer.channel_bytes(ci);
                    w.store_buffer(bytes);
                }
                Compression::Uniform => {
                    // C++ reads the voxel at (0,0,0); for a uniform channel
                    // every voxel equals the default value.
                    let v = buffer.channel_default(ci);
                    store_raw_by_depth(&mut w, v, depth);
                }
            }
        }
        // Metadata section: the buffer-wide entry followed by one entry per
        // voxel, see the module docs. Like C++, nothing is written — not even
        // a size of 0 — when the buffer carries no metadata, which keeps the
        // output byte-compatible with pre-metadata saves.
        let block_metadata = buffer.block_metadata().clone();
        let mut entries: Vec<(Vector3i, MetadataValue)> = Vec::new();
        buffer.for_each_voxel_metadata(|position, value| entries.push((position, value.clone())));
        if !entries.is_empty() || !block_metadata.is_nil() {
            for &(position, _) in &entries {
                // Negative or out-of-size positions cannot be stored in u16
                // coordinates; refuse rather than wrap around.
                if position.x < 0
                    || position.y < 0
                    || position.z < 0
                    || position.x >= size.x
                    || position.y >= size.y
                    || position.z >= size.z
                {
                    return Err(Error::InvalidFormat(format!(
                        "voxel metadata position {:?} outside buffer of size {:?}",
                        position, size
                    )));
                }
            }
            // Sorted positions keep the output deterministic (the C++ FlatMap
            // iterates in key order).
            entries.sort_unstable_by_key(|&(position, _)| (position.x, position.y, position.z));
            let mut section = Vec::new();
            {
                let mut mw = MemoryWriter::little(&mut section);
                store_metadata_value(&mut mw, &block_metadata)?;
                for (position, value) in &entries {
                    mw.store_16(position.x as u16);
                    mw.store_16(position.y as u16);
                    mw.store_16(position.z as u16);
                    store_metadata_value(&mut mw, value)?;
                }
            }
            let section_len = u32::try_from(section.len()).map_err(|_| {
                Error::InvalidFormat("metadata section exceeds u32 length prefix".to_string())
            })?;
            w.store_32(section_len);
            w.store_buffer(&section);
        }
        w.store_32(BLOCK_TRAILING_MAGIC);
    }

    Ok(dst.len())
}

/// Write a single raw voxel value using the width implied by `depth`.
/// Mirrors the `switch (depth)` blocks in `serialize`.
fn store_raw_by_depth(w: &mut MemoryWriter<'_, Vec<u8>>, v: u64, depth: ChannelDepth) {
    match depth {
        ChannelDepth::Bit8 => w.store_8(v as u8),
        ChannelDepth::Bit16 => w.store_16(v as u16),
        ChannelDepth::Bit32 => w.store_32(v as u32),
        ChannelDepth::Bit64 => w.store_64(v),
    }
}

/// Append one type-tagged [`MetadataValue`] to the metadata section. Tags 0/1
/// are the C++ `VoxelMetadata` types; the rest are app-specific, see module
/// docs.
fn store_metadata_value(
    w: &mut MemoryWriter<'_, Vec<u8>>,
    value: &MetadataValue,
) -> Result<(), Error> {
    match value {
        MetadataValue::Nil => w.store_8(METADATA_TYPE_EMPTY),
        MetadataValue::Int(v) => {
            w.store_8(METADATA_TYPE_U64);
            w.store_64(*v as u64);
        }
        MetadataValue::Float(v) => {
            w.store_8(METADATA_TYPE_F64);
            w.store_64(v.to_bits());
        }
        MetadataValue::Text(s) => {
            w.store_8(METADATA_TYPE_TEXT);
            store_length_prefixed(w, s.as_bytes())?;
        }
        MetadataValue::Bytes(b) => {
            w.store_8(METADATA_TYPE_BYTES);
            store_length_prefixed(w, b)?;
        }
    }
    Ok(())
}

fn store_length_prefixed(w: &mut MemoryWriter<'_, Vec<u8>>, data: &[u8]) -> Result<(), Error> {
    let len = u32::try_from(data.len()).map_err(|_| {
        Error::InvalidFormat("metadata payload exceeds u32 length prefix".to_string())
    })?;
    w.store_32(len);
    w.store_buffer(data);
    Ok(())
}

/// Read a single raw voxel value with the width implied by `depth`.
fn read_raw_by_depth(r: &mut MemoryReader<'_>, depth: ChannelDepth) -> Option<u64> {
    match depth {
        ChannelDepth::Bit8 => r.try_get_8().map(|v| v as u64),
        ChannelDepth::Bit16 => r.try_get_16().map(|v| v as u64),
        ChannelDepth::Bit32 => r.try_get_32().map(|v| v as u64),
        ChannelDepth::Bit64 => r.try_get_64(),
    }
}

// ---------------------------------------------------------------------------
// Deserialize
// ---------------------------------------------------------------------------

/// Deserialize `src` into `buffer`, re-creating it. If a version-4 metadata
/// section is present but cannot be decoded in full (foreign C++ entries,
/// corruption), voxel data is loaded and [`Error::MetadataSkipped`] is
/// returned so the caller can decide whether to treat it as a warning. Legacy
/// v2/v3 migration is deferred — see the module docs.
pub fn deserialize(src: &[u8], buffer: &mut VoxelBuffer) -> Result<(), Error> {
    deserialize_with_limits(src, buffer, DecodeLimits::default())
}

/// Deserialize `src` into `buffer` with explicit allocation limits.
pub fn deserialize_with_limits(
    src: &[u8],
    buffer: &mut VoxelBuffer,
    limits: DecodeLimits,
) -> Result<(), Error> {
    // Quick corruption check: the last 4 bytes must be the trailing magic.
    if src.len() < BLOCK_TRAILING_MAGIC_SIZE {
        return Err(Error::UnexpectedEof);
    }
    let tail_start = src.len() - BLOCK_TRAILING_MAGIC_SIZE;
    let magic = u32::from_le_bytes([
        src[tail_start],
        src[tail_start + 1],
        src[tail_start + 2],
        src[tail_start + 3],
    ]);
    if magic != BLOCK_TRAILING_MAGIC {
        return Err(Error::BadTrailingMagic {
            expected: BLOCK_TRAILING_MAGIC,
            found: magic,
        });
    }

    let mut r = MemoryReader::little(src);
    let version = r.try_get_8().ok_or(Error::UnexpectedEof)?;
    if version != BLOCK_FORMAT_VERSION {
        // v2/v3 migration needs the Godot Variant metadata codec — deferred.
        return Err(Error::UnsupportedVersion(version));
    }

    let size_x = r.try_get_16().ok_or(Error::UnexpectedEof)? as i32;
    let size_y = r.try_get_16().ok_or(Error::UnexpectedEof)? as i32;
    let size_z = r.try_get_16().ok_or(Error::UnexpectedEof)? as i32;
    let size = crate::math::Vector3i::new(size_x, size_y, size_z);
    let voxel_count = size.volume_u64();
    limits
        .check_block_voxels(voxel_count)
        .map_err(Error::Limit)?;
    let voxel_count_usize = usize::try_from(voxel_count)
        .map_err(|_| Error::InvalidFormat("block voxel count overflows usize".to_string()))?;
    let worst_case_bytes = voxel_count_usize
        .checked_mul(MAX_CHANNELS)
        .and_then(|v| v.checked_mul(std::mem::size_of::<u64>()))
        .ok_or_else(|| Error::InvalidFormat("block byte count overflow".to_string()))?;
    limits
        .check_bytes("block voxel bytes", worst_case_bytes)
        .map_err(Error::Limit)?;
    buffer.create(size);

    for ci in 0..MAX_CHANNELS {
        let fmt = r.try_get_8().ok_or(Error::UnexpectedEof)?;
        let (compression, depth) = unpack_format(fmt).map_err(|bad| {
            Error::InvalidFormat(format!(
                "channel {ci}: bad fmt byte {fmt:#x} (bad nibble {bad})"
            ))
        })?;
        buffer.set_channel_depth(ci, depth);

        match compression {
            Compression::None => {
                // Decompress (uniform → allocated) so we can write voxel bytes.
                buffer.decompress_channel(ci);
                let dst_bytes = buffer.channel_bytes_mut(ci);
                let expected = dst_bytes.len();
                let src_slice = r.try_take(expected).ok_or(Error::UnexpectedEof)?;
                dst_bytes.copy_from_slice(src_slice);
            }
            Compression::Uniform => {
                let v = read_raw_by_depth(&mut r, depth).ok_or(Error::UnexpectedEof)?;
                buffer.clear_channel(ci, v);
            }
        }
    }

    // Anything between the channels and the trailing magic must be a metadata
    // section encoded as `[u32 size][size bytes]` (same envelope as C++).
    // Envelope corruption is a hard error, but content problems must never
    // fail the load: C++ ignores `deserialize_metadata` failures, so voxel
    // data survives and the loss is surfaced via `Error::MetadataSkipped`.
    let remaining_before_magic =
        (src.len() - BLOCK_TRAILING_MAGIC_SIZE).saturating_sub(r.position());
    if remaining_before_magic > 0 {
        if remaining_before_magic < 4 {
            return Err(Error::UnexpectedEof);
        }
        let metadata_pos = r.position();
        let metadata_size = u32::from_le_bytes([
            src[metadata_pos],
            src[metadata_pos + 1],
            src[metadata_pos + 2],
            src[metadata_pos + 3],
        ]) as usize;
        let expected_metadata_section_len = 4usize
            .checked_add(metadata_size)
            .ok_or_else(|| Error::InvalidFormat("metadata section size overflow".to_string()))?;
        if expected_metadata_section_len != remaining_before_magic {
            return Err(Error::InvalidFormat(format!(
                "metadata section length mismatch (declared {metadata_size}, remaining {})",
                remaining_before_magic - 4
            )));
        }
        if metadata_size > 0 {
            // `r` still points at the size field; slice the section bytes
            // directly so the reader shift by the u32 never matters.
            let section = src
                .get(metadata_pos + 4..metadata_pos + 4 + metadata_size)
                .ok_or(Error::UnexpectedEof)?;
            limits
                .check_bytes("block metadata section", metadata_size)
                .map_err(Error::Limit)?;
            if decode_metadata_section(section, size, limits, buffer).is_err() {
                return Err(Error::MetadataSkipped);
            }
        }
    }

    Ok(())
}

/// Why a metadata section could not be decoded. Every variant maps to
/// [`Error::MetadataSkipped`] — the distinction only exists for debugging.
#[derive(Debug)]
enum MetadataDecodeError {
    UnexpectedEof,
    /// An entry uses a C++ custom tag (e.g. 32 = Godot Variant) or another
    /// tag this build does not know; its payload length is unknowable, so the
    /// rest of the section cannot be parsed either.
    ForeignEntry,
    /// A decodable entry holds invalid contents (bad UTF-8, position outside
    /// the buffer…).
    Corrupt,
}

/// Decode a metadata section (the `[size bytes]` part) into `buffer`. Only
/// commits on full success so a skipped section never leaves half-populated
/// metadata behind.
fn decode_metadata_section(
    src: &[u8],
    buffer_size: Vector3i,
    limits: DecodeLimits,
    buffer: &mut VoxelBuffer,
) -> Result<(), MetadataDecodeError> {
    let mut r = MemoryReader::little(src);
    let block_metadata = read_metadata_value(&mut r, limits)?;
    let mut entries: Vec<(Vector3i, MetadataValue)> = Vec::new();
    while r.position() < src.len() {
        let x = i32::from(r.try_get_16().ok_or(MetadataDecodeError::UnexpectedEof)?);
        let y = i32::from(r.try_get_16().ok_or(MetadataDecodeError::UnexpectedEof)?);
        let z = i32::from(r.try_get_16().ok_or(MetadataDecodeError::UnexpectedEof)?);
        let value = read_metadata_value(&mut r, limits)?;
        // u16 coordinates are non-negative by construction; only the upper
        // bound can fail.
        if x >= buffer_size.x || y >= buffer_size.y || z >= buffer_size.z {
            return Err(MetadataDecodeError::Corrupt);
        }
        entries.push((Vector3i::new(x, y, z), value));
    }
    buffer.set_block_metadata(block_metadata);
    buffer.clear_all_voxel_metadata();
    for (position, value) in entries {
        buffer.set_voxel_metadata(position, value);
    }
    Ok(())
}

fn read_metadata_value(
    r: &mut MemoryReader<'_>,
    limits: DecodeLimits,
) -> Result<MetadataValue, MetadataDecodeError> {
    let tag = r.try_get_8().ok_or(MetadataDecodeError::UnexpectedEof)?;
    match tag {
        METADATA_TYPE_EMPTY => Ok(MetadataValue::Nil),
        METADATA_TYPE_U64 => Ok(MetadataValue::Int(
            r.try_get_64().ok_or(MetadataDecodeError::UnexpectedEof)? as i64,
        )),
        METADATA_TYPE_F64 => Ok(MetadataValue::Float(f64::from_bits(
            r.try_get_64().ok_or(MetadataDecodeError::UnexpectedEof)?,
        ))),
        METADATA_TYPE_TEXT => {
            let len = r.try_get_32().ok_or(MetadataDecodeError::UnexpectedEof)? as usize;
            // A hostile Text length must trip the string budget, not just the
            // section budget (defaults differ: 4 KiB vs 64 MiB).
            limits
                .check_string_bytes(len)
                .map_err(|_| MetadataDecodeError::Corrupt)?;
            let bytes = r
                .try_take(len)
                .map(<[u8]>::to_vec)
                .ok_or(MetadataDecodeError::UnexpectedEof)?;
            String::from_utf8(bytes)
                .map(MetadataValue::Text)
                .map_err(|_| MetadataDecodeError::Corrupt)
        }
        METADATA_TYPE_BYTES => {
            let len = r.try_get_32().ok_or(MetadataDecodeError::UnexpectedEof)? as usize;
            r.try_take(len)
                .map(<[u8]>::to_vec)
                .map(MetadataValue::Bytes)
                .ok_or(MetadataDecodeError::UnexpectedEof)
        }
        other if other >= METADATA_TYPE_CUSTOM_BEGIN => Err(MetadataDecodeError::ForeignEntry),
        _ => Err(MetadataDecodeError::Corrupt),
    }
}

// ---------------------------------------------------------------------------
// Compressed wrappers
// ---------------------------------------------------------------------------

/// Serialize `buffer`, then compress the result. Ported from
/// `BlockSerializer::serialize_and_compress`.
pub fn serialize_and_compress(
    buffer: &VoxelBuffer,
    dst: &mut Vec<u8>,
    compression_mode: compressed_data::Compression,
) -> Result<usize, Error> {
    let mut raw = Vec::new();
    serialize(buffer, &mut raw)?;
    compressed_data::compress(&raw, dst, compression_mode)?;
    Ok(dst.len())
}

/// Decompress `src`, then deserialize. Ported from
/// `BlockSerializer::decompress_and_deserialize`.
/// Outcome of a decompress+deserialize operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeserializeStatus {
    /// Voxel data loaded successfully, no metadata present or all decoded.
    Complete,
    /// Voxel data loaded successfully, but block/per-voxel metadata was
    /// present and could not be decoded in full (foreign C++ custom entries or
    /// a corrupt section). The caller must decide whether to accept the lossy
    /// load or reject it.
    MetadataLost,
}

pub fn decompress_and_deserialize(src: &[u8], buffer: &mut VoxelBuffer) -> Result<(), Error> {
    let status = decompress_and_deserialize_with_limits(src, buffer, DecodeLimits::default())?;
    // Default: accept metadata loss silently for backward compatibility with
    // existing callers that use the non-limits wrapper. Callers that care
    // should use decompress_and_deserialize_with_limits directly.
    let _ = status;
    Ok(())
}

/// Decompress `src`, then deserialize with explicit allocation limits.
/// Returns a [`DeserializeStatus`] so the caller can detect metadata loss
/// (META-1 parity: no silent Ok(()) when metadata was present).
pub fn decompress_and_deserialize_with_limits(
    src: &[u8],
    buffer: &mut VoxelBuffer,
    limits: DecodeLimits,
) -> Result<DeserializeStatus, Error> {
    let mut raw = Vec::new();
    compressed_data::decompress_with_limits(src, &mut raw, limits)?;
    match deserialize_with_limits(&raw, buffer, limits) {
        Ok(()) => Ok(DeserializeStatus::Complete),
        Err(Error::MetadataSkipped) => Ok(DeserializeStatus::MetadataLost),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vector3i;
    use crate::storage::voxel_buffer::Allocator;

    /// Build a small buffer with a few distinct voxel values in channel 0
    /// (so it stays non-uniform) and a uniform value in channel 1.
    fn sample_buffer() -> VoxelBuffer {
        let mut b = VoxelBuffer::with_size(Vector3i::new(4, 2, 3));
        // Non-uniform channel 0.
        for z in 0..3 {
            for y in 0..2 {
                for x in 0..4 {
                    b.set_voxel(((x + y * 4 + z * 8) as u64) & 0xff, x, y, z, 0);
                }
            }
        }
        // Uniform channel 1 (default Compression::Uniform).
        b.clear_channel(1, 42);
        b
    }

    fn append_metadata_section(bytes: &mut Vec<u8>, metadata: &[u8]) {
        let magic = bytes.split_off(bytes.len() - BLOCK_TRAILING_MAGIC_SIZE);
        bytes.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
        bytes.extend_from_slice(metadata);
        bytes.extend_from_slice(&magic);
    }

    fn header_only_block(size: Vector3i) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(BLOCK_FORMAT_VERSION);
        bytes.extend_from_slice(&(size.x as u16).to_le_bytes());
        bytes.extend_from_slice(&(size.y as u16).to_le_bytes());
        bytes.extend_from_slice(&(size.z as u16).to_le_bytes());
        bytes.extend_from_slice(&BLOCK_TRAILING_MAGIC.to_le_bytes());
        bytes
    }

    #[test]
    fn deserialize_rejects_block_voxel_count_over_limit_before_create() {
        let bytes = header_only_block(Vector3i::new(8, 8, 8));
        let limits = crate::streams::DecodeLimits {
            max_block_voxels: 16,
            ..crate::streams::DecodeLimits::default()
        };
        let mut dst = VoxelBuffer::new(Allocator::Default);

        let err = deserialize_with_limits(&bytes, &mut dst, limits).unwrap_err();

        assert!(matches!(err, Error::Limit(_)));
        assert_eq!(dst.size(), Vector3i::zero());
    }

    #[test]
    fn serialize_round_trips_structure() {
        let src = sample_buffer();
        let mut bytes = Vec::new();
        let n = serialize(&src, &mut bytes).unwrap();
        assert!(n > 0);

        let mut dst = VoxelBuffer::new(Allocator::Default);
        deserialize(&bytes, &mut dst).unwrap();

        assert_eq!(dst.size(), Vector3i::new(4, 2, 3));
        for z in 0..3 {
            for y in 0..2 {
                for x in 0..4 {
                    assert_eq!(
                        dst.get_voxel(x, y, z, 0),
                        src.get_voxel(x, y, z, 0),
                        "ch0 at ({x},{y},{z})"
                    );
                }
            }
        }
        // Channel 1 uniform value round-trips.
        assert_eq!(dst.get_voxel(0, 0, 0, 1), 42);
    }

    #[test]
    fn serialize_writes_version_and_trailing_magic() {
        let mut bytes = Vec::new();
        serialize(&sample_buffer(), &mut bytes).unwrap();
        assert_eq!(bytes[0], BLOCK_FORMAT_VERSION);
        let n = bytes.len();
        let magic = u32::from_le_bytes([bytes[n - 4], bytes[n - 3], bytes[n - 2], bytes[n - 1]]);
        assert_eq!(magic, BLOCK_TRAILING_MAGIC);
    }

    #[test]
    fn serialize_rejects_empty_buffer() {
        let empty = VoxelBuffer::new(Allocator::Default); // size (0,0,0)
        let mut bytes = Vec::new();
        assert_eq!(serialize(&empty, &mut bytes), Err(Error::EmptyBuffer));
    }

    #[test]
    fn deserialize_rejects_bad_trailing_magic() {
        let mut bytes = Vec::new();
        serialize(&sample_buffer(), &mut bytes).unwrap();
        // Corrupt the trailing magic.
        let n = bytes.len();
        bytes[n - 1] ^= 0xff;
        let mut dst = VoxelBuffer::new(Allocator::Default);
        match deserialize(&bytes, &mut dst) {
            Err(Error::BadTrailingMagic { .. }) => {}
            other => panic!("expected BadTrailingMagic, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_rejects_unsupported_version() {
        // Build a stream that begins with version 2.
        let mut bytes = Vec::new();
        bytes.push(2u8);
        // Pad to at least the trailing-magic length so the early magic check
        // passes; place a valid-looking magic at the end.
        bytes.extend_from_slice(&[0u8; 3]);
        bytes.extend_from_slice(&BLOCK_TRAILING_MAGIC.to_le_bytes());
        let mut dst = VoxelBuffer::new(Allocator::Default);
        assert_eq!(
            deserialize(&bytes, &mut dst),
            Err(Error::UnsupportedVersion(2))
        );
    }

    #[test]
    fn pack_unpack_format_round_trips() {
        for compression in [Compression::None, Compression::Uniform] {
            for depth in [
                ChannelDepth::Bit8,
                ChannelDepth::Bit16,
                ChannelDepth::Bit32,
                ChannelDepth::Bit64,
            ] {
                let fmt = pack_format(compression, depth);
                let (c, d) = unpack_format(fmt).unwrap();
                assert_eq!(c, compression);
                assert_eq!(d, depth);
            }
        }
    }

    #[test]
    fn unpack_format_rejects_out_of_range_nibbles() {
        assert!(unpack_format(0x02).is_err()); // bad compression
        assert!(unpack_format(0x40).is_err()); // bad depth
        assert!(unpack_format(0xff).is_err());
    }

    #[test]
    fn uniform_channel_serializes_as_single_value() {
        let mut src = VoxelBuffer::with_size(Vector3i::new(8, 8, 8));
        src.clear_channel(0, 123); // uniform
        let mut bytes = Vec::new();
        serialize(&src, &mut bytes).unwrap();

        let mut dst = VoxelBuffer::new(Allocator::Default);
        deserialize(&bytes, &mut dst).unwrap();
        // Every voxel in channel 0 is 123.
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    assert_eq!(dst.get_voxel(x, y, z, 0), 123);
                }
            }
        }
        assert_eq!(dst.channel_compression(0), Compression::Uniform);
    }

    #[test]
    fn depth_16bit_channel_round_trips() {
        let mut src = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        src.set_channel_depth(0, ChannelDepth::Bit16);
        src.decompress_channel(0);
        // Write distinct 16-bit values.
        src.set_voxel(0x1234, 0, 0, 0, 0);
        src.set_voxel(0xabcd, 1, 0, 0, 0);
        let mut bytes = Vec::new();
        serialize(&src, &mut bytes).unwrap();

        let mut dst = VoxelBuffer::new(Allocator::Default);
        deserialize(&bytes, &mut dst).unwrap();
        assert_eq!(dst.channel_depth(0), ChannelDepth::Bit16);
        assert_eq!(dst.get_voxel(0, 0, 0, 0), 0x1234);
        assert_eq!(dst.get_voxel(1, 0, 0, 0), 0xabcd);
    }

    #[test]
    fn serialize_and_compress_round_trips_with_lz4() {
        let src = sample_buffer();
        let mut compressed = Vec::new();
        serialize_and_compress(&src, &mut compressed, compressed_data::Compression::Lz4).unwrap();

        let mut dst = VoxelBuffer::new(Allocator::Default);
        decompress_and_deserialize(&compressed, &mut dst).unwrap();
        assert_eq!(dst.size(), Vector3i::new(4, 2, 3));
        for z in 0..3 {
            for y in 0..2 {
                for x in 0..4 {
                    assert_eq!(dst.get_voxel(x, y, z, 0), src.get_voxel(x, y, z, 0));
                }
            }
        }
    }

    #[test]
    fn compressed_round_trip_with_none_compression() {
        let src = sample_buffer();
        let mut wrapped = Vec::new();
        serialize_and_compress(&src, &mut wrapped, compressed_data::Compression::None).unwrap();

        let mut dst = VoxelBuffer::new(Allocator::Default);
        decompress_and_deserialize(&wrapped, &mut dst).unwrap();
        assert_eq!(dst.size(), src.size());
    }

    #[test]
    fn direct_deserialize_reports_metadata_after_loading_voxels() {
        let src = sample_buffer();
        let mut bytes = Vec::new();
        serialize(&src, &mut bytes).unwrap();
        append_metadata_section(&mut bytes, b"metadata");

        let mut dst = VoxelBuffer::new(Allocator::Default);
        assert_eq!(deserialize(&bytes, &mut dst), Err(Error::MetadataSkipped));
        assert_eq!(dst.size(), src.size());
        assert_eq!(dst.get_voxel(3, 1, 2, 0), src.get_voxel(3, 1, 2, 0));
    }

    #[test]
    fn deserialize_rejects_trailing_junk_before_magic() {
        let mut bytes = Vec::new();
        serialize(&sample_buffer(), &mut bytes).unwrap();
        let magic = bytes.split_off(bytes.len() - BLOCK_TRAILING_MAGIC_SIZE);
        bytes.push(0xff);
        bytes.extend_from_slice(&magic);

        let mut dst = VoxelBuffer::new(Allocator::Default);
        assert_eq!(deserialize(&bytes, &mut dst), Err(Error::UnexpectedEof));
    }

    #[test]
    fn deserialize_rejects_metadata_size_mismatch() {
        let mut bytes = Vec::new();
        serialize(&sample_buffer(), &mut bytes).unwrap();
        let magic = bytes.split_off(bytes.len() - BLOCK_TRAILING_MAGIC_SIZE);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(b"xy");
        bytes.extend_from_slice(&magic);

        let mut dst = VoxelBuffer::new(Allocator::Default);
        match deserialize(&bytes, &mut dst) {
            Err(Error::InvalidFormat(message)) => {
                assert!(message.contains("metadata section length mismatch"));
            }
            other => panic!("expected metadata size mismatch, got {other:?}"),
        }
    }

    #[test]
    fn compressed_wrapper_rejects_malformed_metadata_envelope() {
        let mut raw = Vec::new();
        serialize(&sample_buffer(), &mut raw).unwrap();
        let magic = raw.split_off(raw.len() - BLOCK_TRAILING_MAGIC_SIZE);
        raw.push(0xff);
        raw.extend_from_slice(&magic);

        let mut wrapped = Vec::new();
        compressed_data::compress(&raw, &mut wrapped, compressed_data::Compression::None).unwrap();

        let mut dst = VoxelBuffer::new(Allocator::Default);
        assert_eq!(
            decompress_and_deserialize(&wrapped, &mut dst),
            Err(Error::UnexpectedEof)
        );
    }

    // ------------------------------------------------------------------
    // Metadata section (R7 narrow)
    // ------------------------------------------------------------------

    /// A 2³ buffer with every channel forced uniform with a distinct value,
    /// so its serialized bytes are fully hand-computable.
    fn uniform_block() -> VoxelBuffer {
        let mut b = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        for ci in 0..MAX_CHANNELS {
            b.clear_channel(ci, ci as u64);
        }
        b
    }

    fn metadata_entries(buffer: &VoxelBuffer) -> Vec<(Vector3i, MetadataValue)> {
        let mut entries = Vec::new();
        buffer.for_each_voxel_metadata(|position, value| entries.push((position, value.clone())));
        entries.sort_unstable_by_key(|&(p, _)| (p.x, p.y, p.z));
        entries
    }

    #[test]
    fn serialize_omits_metadata_section_entirely_when_empty() {
        // Channel defaults are uniform with per-channel depths, so the pinned
        // layout is: version, 3×u16 size, per channel one fmt byte plus one
        // default value of the channel's depth width, then the magic — and
        // nothing else. No `[u32 0]` may appear before the magic: C++ omits
        // the section entirely when there is no metadata, and so must we.
        let buffer = uniform_block();
        let mut bytes = Vec::new();
        serialize(&buffer, &mut bytes).unwrap();

        assert_eq!(bytes[0], BLOCK_FORMAT_VERSION);
        let width = |ci: usize| match buffer.channel_depth(ci) {
            ChannelDepth::Bit8 => 1,
            ChannelDepth::Bit16 => 2,
            ChannelDepth::Bit32 => 4,
            ChannelDepth::Bit64 => 8,
        };
        let mut expected_len = 1 + 3 * 2; // version + sizes
        for ci in 0..MAX_CHANNELS {
            expected_len += 1 + width(ci); // fmt byte + uniform default
        }
        expected_len += BLOCK_TRAILING_MAGIC_SIZE;
        assert_eq!(
            bytes.len(),
            expected_len,
            "no metadata section may be written when the buffer has none"
        );
        assert_eq!(
            &bytes[bytes.len() - 4..],
            &BLOCK_TRAILING_MAGIC.to_le_bytes()
        );

        let mut dst = VoxelBuffer::new(Allocator::Default);
        deserialize(&bytes, &mut dst).unwrap();
        assert!(!dst.has_voxel_metadata());
        assert!(dst.block_metadata().is_nil());
    }

    #[test]
    fn metadata_section_round_trips_every_value_kind() {
        let mut src = sample_buffer();
        src.set_block_metadata(MetadataValue::Text("block-meta".into()));
        src.set_voxel_metadata(Vector3i::new(0, 0, 0), MetadataValue::Int(-42));
        src.set_voxel_metadata(Vector3i::new(1, 0, 0), MetadataValue::Float(2.5));
        src.set_voxel_metadata(Vector3i::new(2, 0, 0), MetadataValue::Text("hello".into()));
        src.set_voxel_metadata(
            Vector3i::new(3, 0, 0),
            MetadataValue::Bytes(vec![0, 1, 2, 0xff]),
        );
        src.set_voxel_metadata(Vector3i::new(0, 1, 0), MetadataValue::Int(i64::MIN));
        src.set_voxel_metadata(Vector3i::new(1, 1, 0), MetadataValue::Int(i64::MAX));

        let mut bytes = Vec::new();
        serialize(&src, &mut bytes).unwrap();

        // Pre-populate the destination with stale metadata: a successful
        // deserialize must replace it, not merge with it.
        let mut dst = VoxelBuffer::with_size(Vector3i::new(1, 1, 1));
        dst.set_block_metadata(MetadataValue::Int(1234));
        dst.set_voxel_metadata(Vector3i::zero(), MetadataValue::Int(1234));

        deserialize(&bytes, &mut dst).unwrap();
        assert_eq!(
            *dst.block_metadata(),
            MetadataValue::Text("block-meta".into())
        );
        assert_eq!(metadata_entries(&dst), metadata_entries(&src));
        assert_eq!(dst.get_voxel(3, 1, 2, 0), src.get_voxel(3, 1, 2, 0));
    }

    #[test]
    fn block_only_metadata_still_writes_a_section() {
        let mut src = uniform_block();
        src.set_block_metadata(MetadataValue::Int(7));

        let mut bytes = Vec::new();
        serialize(&src, &mut bytes).unwrap();
        // Identical buffer without metadata + u32 section length + tag + u64.
        let mut base = Vec::new();
        serialize(&uniform_block(), &mut base).unwrap();
        assert_eq!(bytes.len(), base.len() + 4 + 1 + 8);

        let mut dst = VoxelBuffer::new(Allocator::Default);
        deserialize(&bytes, &mut dst).unwrap();
        assert_eq!(*dst.block_metadata(), MetadataValue::Int(7));
        assert!(!dst.has_voxel_metadata());
    }

    #[test]
    fn nil_and_int_entries_match_cpp_bytes_exactly() {
        // C++ writes: block entry (tag + payload) first, then per-voxel
        // `u16 x, u16 y, u16 z` + entry, all little-endian, TYPE_EMPTY = 0 and
        // TYPE_U64 = 1. Our writer must be byte-identical for Nil/Int values.
        let mut with_metadata = uniform_block();
        with_metadata.set_block_metadata(MetadataValue::Int(5));
        with_metadata.set_voxel_metadata(Vector3i::new(1, 1, 1), MetadataValue::Int(7));

        let mut ours = Vec::new();
        serialize(&with_metadata, &mut ours).unwrap();

        let mut cpp_style = Vec::new();
        serialize(&uniform_block(), &mut cpp_style).unwrap();
        let magic = cpp_style.split_off(cpp_style.len() - BLOCK_TRAILING_MAGIC_SIZE);
        let mut section = Vec::new();
        section.push(1u8); // TYPE_U64 block metadata
        section.extend_from_slice(&5u64.to_le_bytes());
        section.extend_from_slice(&1u16.to_le_bytes());
        section.extend_from_slice(&1u16.to_le_bytes());
        section.extend_from_slice(&1u16.to_le_bytes());
        section.push(1u8); // TYPE_U64 voxel metadata
        section.extend_from_slice(&7u64.to_le_bytes());
        cpp_style.extend_from_slice(&(section.len() as u32).to_le_bytes());
        cpp_style.extend_from_slice(&section);
        cpp_style.extend_from_slice(&magic);

        assert_eq!(ours, cpp_style);
    }

    #[test]
    fn cpp_style_section_with_empty_block_entry_decodes() {
        // The exact shape C++ produces for "no block metadata, one u64 voxel
        // entry": leading TYPE_EMPTY byte, then the voxel entry.
        let mut bytes = Vec::new();
        serialize(&uniform_block(), &mut bytes).unwrap();
        let mut section = Vec::new();
        section.push(0u8); // TYPE_EMPTY block metadata
        section.extend_from_slice(&1u16.to_le_bytes());
        section.extend_from_slice(&0u16.to_le_bytes());
        section.extend_from_slice(&1u16.to_le_bytes());
        section.push(1u8); // TYPE_U64
        section.extend_from_slice(&9u64.to_le_bytes());
        append_metadata_section(&mut bytes, &section);

        let mut dst = VoxelBuffer::new(Allocator::Default);
        deserialize(&bytes, &mut dst).unwrap();
        assert!(dst.block_metadata().is_nil());
        assert_eq!(
            dst.voxel_metadata(Vector3i::new(1, 0, 1)),
            Some(&MetadataValue::Int(9))
        );
    }

    #[test]
    fn foreign_custom_entries_are_skipped_not_fatal() {
        // Tag 32 is the C++ METADATA_TYPE_VARIANT: undecodable here, and its
        // payload length is unknowable, so nothing after it can be parsed.
        let mut bytes = Vec::new();
        serialize(&uniform_block(), &mut bytes).unwrap();
        append_metadata_section(&mut bytes, &[32, 0xaa, 0xbb]);

        let mut dst = VoxelBuffer::new(Allocator::Default);
        assert_eq!(deserialize(&bytes, &mut dst), Err(Error::MetadataSkipped));
        // Voxel data still loaded, no partial metadata committed.
        assert_eq!(dst.size(), Vector3i::new(2, 2, 2));
        assert!(dst.block_metadata().is_nil());
        assert!(!dst.has_voxel_metadata());
    }

    #[test]
    fn foreign_entry_after_decodable_prefix_leaves_no_partial_metadata() {
        let mut bytes = Vec::new();
        serialize(&uniform_block(), &mut bytes).unwrap();
        let mut section = Vec::new();
        section.push(1u8); // decodable block metadata…
        section.extend_from_slice(&3u64.to_le_bytes());
        section.extend_from_slice(&0u16.to_le_bytes());
        section.extend_from_slice(&0u16.to_le_bytes());
        section.extend_from_slice(&0u16.to_le_bytes());
        section.push(40u8); // …then a foreign app-specific tag (f64 without payload)
        append_metadata_section(&mut bytes, &section);

        let mut dst = VoxelBuffer::new(Allocator::Default);
        assert_eq!(deserialize(&bytes, &mut dst), Err(Error::MetadataSkipped));
        assert!(dst.block_metadata().is_nil());
        assert!(!dst.has_voxel_metadata());
    }

    #[test]
    fn compressed_wrapper_reports_metadata_lost_for_foreign_entries() {
        let mut raw = Vec::new();
        serialize(&uniform_block(), &mut raw).unwrap();
        append_metadata_section(&mut raw, &[32, 0x01]);

        let mut wrapped = Vec::new();
        compressed_data::compress(&raw, &mut wrapped, compressed_data::Compression::Lz4).unwrap();

        let mut dst = VoxelBuffer::new(Allocator::Default);
        assert_eq!(
            decompress_and_deserialize_with_limits(
                &wrapped,
                &mut dst,
                crate::streams::DecodeLimits::default()
            )
            .unwrap(),
            DeserializeStatus::MetadataLost
        );
        assert_eq!(dst.size(), Vector3i::new(2, 2, 2));
    }

    #[test]
    fn corrupt_metadata_sections_are_skipped_not_fatal() {
        let sections: Vec<&[u8]> = vec![
            &[1u8][..],                          // truncated u64 payload
            &[7u8][..],                          // reserved/invalid tag [2, 32)
            &[41u8, 2, 0, 0, 0, 0xff, 0xfe][..], // text with invalid UTF-8
            &[41u8, 100, 0, 0, 0, b'h'][..],     // length overruns the section
        ];
        for section in sections {
            let mut bytes = Vec::new();
            serialize(&uniform_block(), &mut bytes).unwrap();
            append_metadata_section(&mut bytes, section);

            let mut dst = VoxelBuffer::new(Allocator::Default);
            assert_eq!(
                deserialize(&bytes, &mut dst),
                Err(Error::MetadataSkipped),
                "section {section:?} should be skipped"
            );
            assert!(!dst.has_voxel_metadata());
        }
    }

    #[test]
    fn voxel_metadata_position_outside_buffer_is_skipped_on_decode() {
        let mut bytes = Vec::new();
        serialize(&uniform_block(), &mut bytes).unwrap();
        let mut section = Vec::new();
        section.push(0u8);
        section.extend_from_slice(&5u16.to_le_bytes()); // x=5 in a 2³ buffer
        section.extend_from_slice(&0u16.to_le_bytes());
        section.extend_from_slice(&0u16.to_le_bytes());
        section.push(1u8);
        section.extend_from_slice(&1u64.to_le_bytes());
        append_metadata_section(&mut bytes, &section);

        let mut dst = VoxelBuffer::new(Allocator::Default);
        assert_eq!(deserialize(&bytes, &mut dst), Err(Error::MetadataSkipped));
        assert_eq!(dst.size(), Vector3i::new(2, 2, 2));
    }

    #[test]
    fn serialize_rejects_voxel_metadata_position_outside_buffer() {
        let mut src = uniform_block();
        src.set_voxel_metadata(Vector3i::new(5, 0, 0), MetadataValue::Int(1));
        let mut bytes = Vec::new();
        match serialize(&src, &mut bytes) {
            Err(Error::InvalidFormat(message)) => {
                assert!(message.contains("outside buffer"), "got: {message}");
            }
            other => panic!("expected InvalidFormat, got {other:?}"),
        }
    }

    #[test]
    fn metadata_section_respects_decode_limits() {
        let mut src = uniform_block();
        src.set_block_metadata(MetadataValue::Int(1));
        let mut bytes = Vec::new();
        serialize(&src, &mut bytes).unwrap();

        // The 9-byte section exceeds a 4-byte total budget → hard limit error.
        let limits = crate::streams::DecodeLimits {
            max_bytes: 4,
            ..crate::streams::DecodeLimits::default()
        };
        let mut dst = VoxelBuffer::new(Allocator::Default);
        assert!(matches!(
            deserialize_with_limits(&bytes, &mut dst, limits),
            Err(Error::Limit(_))
        ));

        // A text entry longer than the string budget is skipped (content
        // problem), not fatal.
        let mut src = sample_buffer();
        src.set_voxel_metadata(Vector3i::zero(), MetadataValue::Text("x".repeat(64)));
        let mut bytes = Vec::new();
        serialize(&src, &mut bytes).unwrap();
        let limits = crate::streams::DecodeLimits {
            max_string_bytes: 16,
            ..crate::streams::DecodeLimits::default()
        };
        let mut dst = VoxelBuffer::new(Allocator::Default);
        assert_eq!(
            deserialize_with_limits(&bytes, &mut dst, limits),
            Err(Error::MetadataSkipped)
        );
        assert_eq!(dst.size(), src.size());
    }
}
