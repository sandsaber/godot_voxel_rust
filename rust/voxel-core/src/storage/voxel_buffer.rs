//! Dense multi-channel voxel storage.
//!
//! Ported from `storage/voxel_buffer.{h,cpp}`. Up to 8 channels of variable
//! bit-depth (8/16/32/64-bit), two compression modes (`NONE` — fully allocated,
//! `UNIFORM` — single default value, no allocation), ZXY memory layout
//! (`index = y + sy*(x + sx*z)`, Y innermost). The C++ class has two allocators
//! (DEFAULT = malloc, POOL = `VoxelMemoryPool`); in Rust both map to `Vec<u8>`,
//! and the pool is opt-in via [`VoxelMemoryPool`].
//!
//! ## `QUANTIZED_SDF_*` constants
//!
//! `voxel_buffer.h` references `constants::QUANTIZED_SDF_{8,16}_BITS_SCALE[_INV]`,
//! which scale SDF floats into the `[-1,1]` snorm range before 8/16-bit
//! quantization. The C++ constants intentionally give 8-bit SDF about
//! `[-10, 10]` range and 16-bit SDF about `[-500, 500]` range.

use super::depth::ChannelDepth;
use super::funcs;
use super::voxel_memory_pool::VoxelMemoryPool;
use crate::math::Vector3i;
use std::collections::HashMap;
use std::sync::Arc;

/// Number of channels. Matches `MAX_CHANNELS`. Indexed by [`ChannelId`].
pub const MAX_CHANNELS: usize = 8;
/// Mask selecting all channels. Matches `ALL_CHANNELS_MASK`.
pub const ALL_CHANNELS_MASK: u8 = 0xff;
/// Maximum size along any axis. Matches `MAX_SIZE`.
pub const MAX_SIZE: u32 = 65535;

/// SDF quantization scale for 8-bit channels. Matches `QUANTIZED_SDF_8_BITS_SCALE`.
/// `raw = snorm_to_s8(sdf * SCALE)`;
/// `sdf = s8_to_snorm(raw) * SCALE_INV`.
pub const QUANTIZED_SDF_8_BITS_SCALE: f32 = 0.1;
/// Inverse of [`QUANTIZED_SDF_8_BITS_SCALE`].
pub const QUANTIZED_SDF_8_BITS_SCALE_INV: f32 = 1.0 / QUANTIZED_SDF_8_BITS_SCALE;
/// SDF quantization scale for 16-bit channels. Matches `QUANTIZED_SDF_16_BITS_SCALE`.
pub const QUANTIZED_SDF_16_BITS_SCALE: f32 = 0.002;
/// Inverse of [`QUANTIZED_SDF_16_BITS_SCALE`].
pub const QUANTIZED_SDF_16_BITS_SCALE_INV: f32 = 1.0 / QUANTIZED_SDF_16_BITS_SCALE;

/// Matches `constants::SDF_FAR_OUTSIDE`.
pub const SDF_FAR_OUTSIDE: f32 = 100.0;
/// Matches `constants::SDF_FAR_INSIDE`.
pub const SDF_FAR_INSIDE: f32 = -100.0;

/// In-memory voxel or block metadata. Lives on [`VoxelBuffer`] and survives
/// copy/paste/edit transactions. Persists through the v4 block serializer
/// metadata section (ROADMAP R7 narrow); foreign Godot Variant payloads need
/// the wide R7 codec and remain unsupported.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MetadataValue {
    /// Empty / cleared entry. Matches a Godot `nil` Variant.
    #[default]
    Nil,
    /// Signed integer payload.
    Int(i64),
    /// Floating-point payload.
    Float(f64),
    /// UTF-8 text payload.
    Text(String),
    /// Opaque byte payload.
    Bytes(Vec<u8>),
}

impl MetadataValue {
    /// True when this value is the empty sentinel.
    #[inline]
    pub fn is_nil(&self) -> bool {
        matches!(self, Self::Nil)
    }
}

/// Matches `mixel4::encode_indices_to_packed_u16(0, 1, 2, 3)`.
pub const MIXEL4_DEFAULT_INDICES: u64 = 0x3210;
/// Matches `mixel4::encode_weights_to_packed_u16_lossy(255, 0, 0, 0)`.
pub const MIXEL4_DEFAULT_WEIGHTS: u64 = 0x000f;

/// Identifies a channel within a [`VoxelBuffer`]. Matches `ChannelId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ChannelId {
    /// Block type / material id.
    Type = 0,
    /// Signed distance field (for smooth meshing).
    Sdf = 1,
    /// Per-voxel color.
    Color = 2,
    /// Material indices (4-way blend).
    Indices = 3,
    /// Material blend weights.
    Weights = 4,
    /// Free-form data channel 5.
    Data5 = 5,
    /// Free-form data channel 6.
    Data6 = 6,
    /// Free-form data channel 7.
    Data7 = 7,
}

impl ChannelId {
    /// Human-readable name. Matches `get_channel_name`.
    pub fn name(self) -> &'static str {
        match self {
            ChannelId::Type => "type",
            ChannelId::Sdf => "sdf",
            ChannelId::Color => "color",
            ChannelId::Indices => "indices",
            ChannelId::Weights => "weights",
            ChannelId::Data5 => "data5",
            ChannelId::Data6 => "data6",
            ChannelId::Data7 => "data7",
        }
    }

    /// Convert to the channel index `0..MAX_CHANNELS`.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// How a channel's voxels are stored. Matches `Compression`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Compression {
    /// Fully allocated array.
    None = 0,
    /// Single uniform default value; no array allocated.
    Uniform = 1,
}

/// Allocator strategy. Matches `Allocator`. In Rust both use `Vec<u8>`; `Pool`
/// routes fresh allocations through an optional [`VoxelMemoryPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Allocator {
    /// `malloc`-backed (the default).
    Default = 0,
    /// [`VoxelMemoryPool`] recycling (faster for many same-sized buffers).
    Pool = 1,
}

impl ChannelDepth {
    /// Number of bytes per voxel for this depth. Matches `get_depth_byte_count`.
    #[inline]
    pub fn byte_count(self) -> u32 {
        1u32 << (self as u32)
    }

    /// Number of bits per voxel. Matches `get_depth_bit_count`.
    #[inline]
    pub fn bit_count(self) -> u32 {
        self.byte_count() << 3
    }
}

/// Default depth constants matching the C++ `DEFAULT_*_CHANNEL_DEPTH`.
pub const DEFAULT_CHANNEL_DEPTH: ChannelDepth = ChannelDepth::Bit8;
pub const DEFAULT_TYPE_CHANNEL_DEPTH: ChannelDepth = ChannelDepth::Bit16;
pub const DEFAULT_SDF_CHANNEL_DEPTH: ChannelDepth = ChannelDepth::Bit16;
pub const DEFAULT_INDICES_CHANNEL_DEPTH: ChannelDepth = ChannelDepth::Bit16;
pub const DEFAULT_WEIGHTS_CHANNEL_DEPTH: ChannelDepth = ChannelDepth::Bit16;

/// Get the default raw (integer) value for a channel at the given depth.
/// Matches `VoxelBuffer::get_default_raw_value`.
pub fn get_default_raw_value(channel: ChannelId, depth: ChannelDepth) -> u64 {
    match channel {
        ChannelId::Type => 0,
        ChannelId::Sdf => get_default_sdf_raw_value(depth),
        ChannelId::Indices => get_default_indices_raw_value(depth),
        ChannelId::Weights => MIXEL4_DEFAULT_WEIGHTS,
        ChannelId::Color | ChannelId::Data5 | ChannelId::Data6 | ChannelId::Data7 => 0,
    }
}

/// Default SDF raw value at `depth`: far outside/air. Matches
/// `get_default_sdf_raw_value`.
pub fn get_default_sdf_raw_value(depth: ChannelDepth) -> u64 {
    match depth {
        ChannelDepth::Bit8 => funcs::snorm_to_s8(1.0) as u8 as u64,
        ChannelDepth::Bit16 => funcs::snorm_to_s16(1.0) as u16 as u64,
        ChannelDepth::Bit32 => f32::to_bits(SDF_FAR_OUTSIDE) as u64,
        ChannelDepth::Bit64 => f64::to_bits(SDF_FAR_OUTSIDE as f64),
    }
}

/// Default SDF float value at `depth`. Matches the decoded C++ default.
pub fn get_default_sdf_value(depth: ChannelDepth) -> f32 {
    raw_voxel_to_real(get_default_sdf_raw_value(depth), depth)
}

/// Default indices raw value at `depth`: material slots 0,1,2,3. Matches
/// `get_default_indices_raw_value`.
pub fn get_default_indices_raw_value(_depth: ChannelDepth) -> u64 {
    MIXEL4_DEFAULT_INDICES
}

/// Convert a float to a raw (integer) voxel value at `depth`. Matches
/// `real_to_raw_voxel`. 8/16-bit quantize to snorm × SDF scale; 32/64-bit store
/// the float bits directly.
pub fn real_to_raw_voxel(value: f32, depth: ChannelDepth) -> u64 {
    match depth {
        ChannelDepth::Bit8 => funcs::snorm_to_s8(value * QUANTIZED_SDF_8_BITS_SCALE) as u8 as u64,
        ChannelDepth::Bit16 => {
            funcs::snorm_to_s16(value * QUANTIZED_SDF_16_BITS_SCALE) as u16 as u64
        }
        ChannelDepth::Bit32 => f32::to_bits(value) as u64,
        ChannelDepth::Bit64 => f64::to_bits(value as f64),
    }
}

/// Convert a raw (integer) voxel value at `depth` back to float. Matches
/// `raw_voxel_to_real`. 8/16-bit expand from snorm × SDF scale.
pub fn raw_voxel_to_real(value: u64, depth: ChannelDepth) -> f32 {
    match depth {
        ChannelDepth::Bit8 => {
            funcs::s8_to_snorm(value as u8 as i8) * QUANTIZED_SDF_8_BITS_SCALE_INV
        }
        ChannelDepth::Bit16 => {
            funcs::s16_to_snorm(value as u16 as i16) * QUANTIZED_SDF_16_BITS_SCALE_INV
        }
        ChannelDepth::Bit32 => f32::from_bits(value as u32),
        ChannelDepth::Bit64 => f64::from_bits(value) as f32,
    }
}

/// Typed per-channel voxel storage, selected by [`ChannelDepth`].
///
/// Replaces the previous `Vec<u8>` raw-byte backing so that depth-dispatch
/// happens once per channel (a single `match`) instead of on every voxel, and
/// the hot loops index a typed slice directly (`data[i]`) instead of
/// re-encoding/decoding little-endian bytes per voxel. This is the structural
/// fix D7 (audit §9.6-D7): it removes the alignment question that blocked
/// typed SDF sampling (B1) and Cubes/Blocky zero-copy (B5), and makes
/// `depth`-dispatch natural.
///
/// Values are stored in the same little-endian ZXY layout the previous
/// `Vec<u8>` held, so [`ChannelData::as_bytes`] reproduces the exact wire
/// format the block serializer depends on (`bytemuck::cast_slice` over a LE
/// `Vec<u{16,32,64}>` is byte-identical to the old `Vec<u8>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelData {
    /// `ChannelDepth::Bit8` storage (unsigned 8-bit voxels).
    U8(Vec<u8>),
    /// `ChannelDepth::Bit16` storage (unsigned 16-bit voxels, LE).
    U16(Vec<u16>),
    /// `ChannelDepth::Bit32` storage (unsigned 32-bit voxels, LE).
    U32(Vec<u32>),
    /// `ChannelDepth::Bit64` storage (unsigned 64-bit voxels, LE).
    U64(Vec<u64>),
}

impl Default for ChannelData {
    fn default() -> Self {
        Self::U8(Vec::new())
    }
}

impl ChannelData {
    fn try_clone(&self) -> Result<Self, std::collections::TryReserveError> {
        macro_rules! clone_vec {
            ($source:expr, $variant:ident) => {{
                let mut copy = Vec::new();
                copy.try_reserve_exact($source.len())?;
                copy.extend_from_slice($source);
                Self::$variant(copy)
            }};
        }
        Ok(match self {
            Self::U8(values) => clone_vec!(values, U8),
            Self::U16(values) => clone_vec!(values, U16),
            Self::U32(values) => clone_vec!(values, U32),
            Self::U64(values) => clone_vec!(values, U64),
        })
    }

    /// Allocate a typed buffer of `len` voxels for `depth`, zero-initialised.
    /// Used when decompressing a uniform channel into a full array.
    #[inline]
    pub fn new_for_depth(depth: ChannelDepth, len: usize) -> Self {
        match depth {
            ChannelDepth::Bit8 => Self::U8(vec![0u8; len]),
            ChannelDepth::Bit16 => Self::U16(vec![0u16; len]),
            ChannelDepth::Bit32 => Self::U32(vec![0u32; len]),
            ChannelDepth::Bit64 => Self::U64(vec![0u64; len]),
        }
    }

    /// Number of voxels (not bytes) currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::U64(v) => v.len(),
        }
    }

    /// True if no voxels are stored (the uniform/compressed case).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The channel depth this storage variant corresponds to.
    #[inline]
    pub fn depth(&self) -> ChannelDepth {
        match self {
            Self::U8(_) => ChannelDepth::Bit8,
            Self::U16(_) => ChannelDepth::Bit16,
            Self::U32(_) => ChannelDepth::Bit32,
            Self::U64(_) => ChannelDepth::Bit64,
        }
    }

    /// Read the voxel at flat ZXY index `i` as a zero-extended `u64`.
    #[inline]
    pub fn get_u64(&self, i: usize) -> u64 {
        match self {
            Self::U8(v) => v[i] as u64,
            Self::U16(v) => v[i] as u64,
            Self::U32(v) => v[i] as u64,
            Self::U64(v) => v[i],
        }
    }

    /// Write a voxel value at flat ZXY index `i`, truncating to the storage
    /// width. Callers must ensure the variant matches the channel depth.
    #[inline]
    pub fn set_u64(&mut self, i: usize, value: u64) {
        match self {
            Self::U8(v) => v[i] = value as u8,
            Self::U16(v) => v[i] = value as u16,
            Self::U32(v) => v[i] = value as u32,
            Self::U64(v) => v[i] = value,
        }
    }

    /// Fill every voxel with `value` truncated to the storage width.
    #[inline]
    pub fn fill_u64(&mut self, value: u64) {
        match self {
            Self::U8(v) => v.fill(value as u8),
            Self::U16(v) => v.fill(value as u16),
            Self::U32(v) => v.fill(value as u32),
            Self::U64(v) => v.fill(value),
        }
    }

    /// View the whole typed buffer as raw little-endian bytes. Used by the
    /// block serializer to read/write channel payloads without interpreting
    /// per-voxel width. Byte-identical to the previous `Vec<u8>` storage
    /// because all targets are LE and `bytemuck::cast_slice` only reinterprets.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::U8(v) => v,
            Self::U16(v) => bytemuck::cast_slice(v),
            Self::U32(v) => bytemuck::cast_slice(v),
            Self::U64(v) => bytemuck::cast_slice(v),
        }
    }

    /// Mutable raw-byte view of the whole typed buffer. See [`Self::as_bytes`].
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::U8(v) => v,
            Self::U16(v) => bytemuck::cast_slice_mut(v),
            Self::U32(v) => bytemuck::cast_slice_mut(v),
            Self::U64(v) => bytemuck::cast_slice_mut(v),
        }
    }
}

/// One channel's storage. Matches `VoxelBuffer::Channel`. Either a uniform
/// default value (`Compression::Uniform`, no allocation) or a fully-allocated
/// typed voxel array (`Compression::None`) whose variant matches [`Channel::depth`].
#[derive(Debug)]
pub struct Channel {
    /// Allocated voxel data. Present only when `compression == None`.
    /// ZXY layout, element count = volume; variant tracks `depth`.
    pub data: ChannelData,
    /// Default value when uniform (encoded; use [`raw_voxel_to_real`] to decode).
    pub defval: u64,
    pub depth: ChannelDepth,
    pub compression: Compression,
    /// Allocated bytes (= volume * depth.byte_count()) when `None`.
    pub size_in_bytes: u32,
}

impl Default for Channel {
    fn default() -> Self {
        Self {
            data: ChannelData::default(),
            defval: 0,
            depth: DEFAULT_CHANNEL_DEPTH,
            compression: Compression::Uniform,
            size_in_bytes: 0,
        }
    }
}

impl Channel {
    /// Bytes needed to store a buffer of `size` at `depth`. Matches
    /// `get_size_in_bytes_for_volume`.
    pub fn size_in_bytes_for_volume(size: Vector3i, depth: ChannelDepth) -> usize {
        (size.x as usize) * (size.y as usize) * (size.z as usize) * depth.byte_count() as usize
    }
}

/// Dense multi-channel voxel buffer. The main Phase-3 storage type, replacing
/// the pilot's single-channel [`super::buffer::DenseVoxelBuffer`] for general use.
///
/// Owned storage. Channels start in `Compression::Uniform` and allocate on first
/// write. When the `pool` allocator is chosen, fresh allocations route through
/// the shared [`VoxelMemoryPool`] (if one is attached).
pub struct VoxelBuffer {
    size: Vector3i,
    channels: [Channel; MAX_CHANNELS],
    allocator: Allocator,
    /// Optional pool used when `allocator == Pool`.
    pool: Option<Arc<VoxelMemoryPool>>,
    /// Opaque metadata attached to the whole buffer (C++ block metadata).
    block_metadata: MetadataValue,
    /// Sparse per-voxel metadata keyed by local buffer coordinates.
    voxel_metadata: HashMap<Vector3i, MetadataValue>,
}

impl std::fmt::Debug for VoxelBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoxelBuffer")
            .field("size", &self.size)
            .field("allocator", &self.allocator)
            .field("pool_attached", &self.pool.is_some())
            .finish()
    }
}

impl VoxelBuffer {
    /// Create an empty (zero-size) buffer with the given allocator. Matches the
    /// C++ `VoxelBuffer(Allocator)` ctor. Call [`create`](Self::create) to size it.
    pub fn new(allocator: Allocator) -> Self {
        Self {
            size: Vector3i::zero(),
            channels: std::array::from_fn(default_channel_for_index),
            allocator,
            pool: None,
            block_metadata: MetadataValue::Nil,
            voxel_metadata: HashMap::new(),
        }
    }

    /// Create with a `Default` allocator and the given size; channels use the
    /// engine's default per-channel depths and uniform defaults. Convenience
    /// over [`new`](Self::new) + [`create`](Self::create).
    pub fn with_size(size: Vector3i) -> Self {
        let mut vb = Self::new(Allocator::Default);
        vb.create(size);
        vb
    }

    /// Attach a memory pool for `Allocator::Pool`. Has no effect for `Default`.
    pub fn with_pool(mut self, pool: Arc<VoxelMemoryPool>) -> Self {
        self.pool = Some(pool);
        self
    }

    /// (Re)allocate to `size` voxels. Every channel is reset to uniform at the
    /// default value for its current depth. Matches the C++ behavior when
    /// `new_format` is null: channel depths are preserved unless a caller
    /// explicitly applies a [`VoxelFormat`](crate::storage::VoxelFormat).
    pub fn create(&mut self, size: Vector3i) {
        debug_assert!(size.x >= 0 && size.y >= 0 && size.z >= 0);
        debug_assert!(
            (size.x as u32) <= MAX_SIZE
                && (size.y as u32) <= MAX_SIZE
                && (size.z as u32) <= MAX_SIZE
        );
        self.size = size;
        for (i, ch) in self.channels.iter_mut().enumerate() {
            free_channel_data(self.allocator, self.pool.as_ref(), ch);
            ch.compression = Compression::Uniform;
            ch.defval = get_default_raw_value(channel_id_from_index(i).unwrap(), ch.depth);
        }
        self.block_metadata = MetadataValue::Nil;
        self.voxel_metadata.clear();
    }

    /// Size in voxels.
    #[inline]
    pub fn size(&self) -> Vector3i {
        self.size
    }

    /// Number of allocated channels.
    #[inline]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Which allocator this buffer uses.
    #[inline]
    pub fn allocator(&self) -> Allocator {
        self.allocator
    }

    /// Depth of a channel.
    #[inline]
    pub fn channel_depth(&self, channel_index: usize) -> ChannelDepth {
        self.channels[channel_index].depth
    }

    /// Set the depth of a channel. Matches `set_channel_depth`: changing an
    /// allocated channel resets it to a uniform default because existing bytes
    /// no longer match the new element width.
    pub fn set_channel_depth(&mut self, channel_index: usize, depth: ChannelDepth) {
        if self.channels[channel_index].depth == depth {
            return;
        }
        let channel_id = channel_id_from_index(channel_index).unwrap();
        let ch = &mut self.channels[channel_index];
        free_channel_data(self.allocator, self.pool.as_ref(), ch);
        ch.depth = depth;
        ch.defval = get_default_raw_value(channel_id, depth);
        ch.compression = Compression::Uniform;
    }

    /// Compression of a channel. Matches `get_channel_compression`.
    #[inline]
    pub fn channel_compression(&self, channel_index: usize) -> Compression {
        self.channels[channel_index].compression
    }

    /// Reset a channel to a uniform raw value, freeing its allocation.
    /// Matches `clear_channel`.
    pub fn clear_channel(&mut self, channel_index: usize, clear_value: u64) {
        let ch = &mut self.channels[channel_index];
        free_channel_data(self.allocator, self.pool.as_ref(), ch);
        ch.defval = clear_value;
        ch.compression = Compression::Uniform;
    }

    /// Reset a channel to a uniform float value. Matches `clear_channel_f`.
    pub fn clear_channel_f(&mut self, channel_index: usize, clear_value: f32) {
        let depth = self.channels[channel_index].depth;
        self.clear_channel(channel_index, real_to_raw_voxel(clear_value, depth));
    }

    /// Get a voxel as a raw `u64` at `depth` width. Matches `get_voxel`.
    #[inline]
    pub fn get_voxel(&self, x: i32, y: i32, z: i32, channel_index: usize) -> u64 {
        let ch = &self.channels[channel_index];
        if ch.compression == Compression::Uniform {
            return ch.defval;
        }
        let i = voxel_index(self.size, x as usize, y as usize, z as usize);
        ch.data.get_u64(i)
    }

    /// Set a voxel from a raw `u64`. Matches `set_voxel`. Decompresses the
    /// channel on first write into it.
    #[inline]
    pub fn set_voxel(&mut self, value: u64, x: i32, y: i32, z: i32, channel_index: usize) {
        self.decompress_channel(channel_index);
        let i = voxel_index(self.size, x as usize, y as usize, z as usize);
        self.channels[channel_index].data.set_u64(i, value);
    }

    /// Get a voxel as float. Matches `get_voxel_f`.
    #[inline]
    pub fn get_voxel_f(&self, x: i32, y: i32, z: i32, channel_index: usize) -> f32 {
        let raw = self.get_voxel(x, y, z, channel_index);
        raw_voxel_to_real(raw, self.channels[channel_index].depth)
    }

    /// Set a voxel from float. Matches `set_voxel_f`.
    #[inline]
    pub fn set_voxel_f(&mut self, value: f32, x: i32, y: i32, z: i32, channel_index: usize) {
        let depth = self.channels[channel_index].depth;
        self.set_voxel(real_to_raw_voxel(value, depth), x, y, z, channel_index);
    }

    /// Fill an entire channel with a raw value. Matches `fill`.
    pub fn fill(&mut self, value: u64, channel_index: usize) {
        // If the value equals the current uniform default, stay compressed.
        let ch = &mut self.channels[channel_index];
        if ch.compression == Compression::Uniform && ch.defval == value {
            return;
        }
        // Otherwise: become uniform with this value (the simplest faithful fill
        // — the C++ also leaves uniform channels uniform when fill == defval,
        // and only allocates for non-uniform fills via the per-voxel loop).
        free_channel_data(self.allocator, self.pool.as_ref(), ch);
        ch.defval = value;
        ch.compression = Compression::Uniform;
    }

    /// Fill a rectangular area with a raw value. Matches `fill_area`.
    pub fn fill_area(&mut self, value: u64, min: Vector3i, max: Vector3i, channel_index: usize) {
        let size = self.size;
        let Some((lo, hi)) = clipped_area(size, min, max) else {
            return;
        };
        if self.channels[channel_index].compression == Compression::Uniform
            && self.channels[channel_index].defval == value
        {
            return;
        }
        self.decompress_channel(channel_index);
        let data = &mut self.channels[channel_index].data;
        // Depth is dispatched once per channel; the inner row fill indexes a
        // typed slice directly (no per-voxel byte encode/decode).
        match data {
            ChannelData::U8(v) => {
                let val = value as u8;
                for z in lo.z..hi.z {
                    for x in lo.x..hi.x {
                        let row_index = voxel_index(size, x as usize, lo.y as usize, z as usize);
                        let row_len = (hi.y - lo.y) as usize;
                        v[row_index..row_index + row_len].fill(val);
                    }
                }
            }
            ChannelData::U16(v) => {
                let val = value as u16;
                for z in lo.z..hi.z {
                    for x in lo.x..hi.x {
                        let row_index = voxel_index(size, x as usize, lo.y as usize, z as usize);
                        let row_len = (hi.y - lo.y) as usize;
                        v[row_index..row_index + row_len].fill(val);
                    }
                }
            }
            ChannelData::U32(v) => {
                let val = value as u32;
                for z in lo.z..hi.z {
                    for x in lo.x..hi.x {
                        let row_index = voxel_index(size, x as usize, lo.y as usize, z as usize);
                        let row_len = (hi.y - lo.y) as usize;
                        v[row_index..row_index + row_len].fill(val);
                    }
                }
            }
            ChannelData::U64(v) => {
                for z in lo.z..hi.z {
                    for x in lo.x..hi.x {
                        let row_index = voxel_index(size, x as usize, lo.y as usize, z as usize);
                        let row_len = (hi.y - lo.y) as usize;
                        v[row_index..row_index + row_len].fill(value);
                    }
                }
            }
        }
    }

    /// Apply `action` to every voxel in a local area of one channel.
    ///
    /// This is the safe Rust counterpart of C++ `write_box_template`: the
    /// channel is decompressed once, its depth is dispatched once, and the
    /// inner loop mutates contiguous ZXY rows by flat index.
    pub(crate) fn read_write_area<F>(
        &mut self,
        min: Vector3i,
        max: Vector3i,
        channel_index: usize,
        mut action: F,
    ) where
        F: FnMut(Vector3i, u64) -> u64,
    {
        let size = self.size;
        let Some((lo, hi)) = clipped_area(size, min, max) else {
            return;
        };
        self.decompress_channel(channel_index);
        let data = &mut self.channels[channel_index].data;
        // Depth dispatched once per channel; the inner loop indexes a typed slice
        // directly (no per-voxel from_le_bytes/to_le_bytes).
        match data {
            ChannelData::U8(v) => {
                for_each_index_and_pos(size, lo, hi, |i, pos| {
                    let current = v[i] as u64;
                    v[i] = action(pos, current) as u8;
                });
            }
            ChannelData::U16(v) => {
                for_each_index_and_pos(size, lo, hi, |i, pos| {
                    let current = v[i] as u64;
                    v[i] = action(pos, current) as u16;
                });
            }
            ChannelData::U32(v) => {
                for_each_index_and_pos(size, lo, hi, |i, pos| {
                    let current = v[i] as u64;
                    v[i] = action(pos, current) as u32;
                });
            }
            ChannelData::U64(v) => {
                for_each_index_and_pos(size, lo, hi, |i, pos| {
                    let current = v[i];
                    v[i] = action(pos, current);
                });
            }
        }
    }

    /// Like [`read_write_area`](Self::read_write_area), while also reading a
    /// second channel at the same voxel index before the write. Used by masked
    /// paste so destination-mask checks use the original mask channel value
    /// without redispatching the written channel depth for every voxel.
    pub(crate) fn read_write_area_with_channel<F>(
        &mut self,
        min: Vector3i,
        max: Vector3i,
        write_channel_index: usize,
        read_channel_index: usize,
        mut action: F,
    ) where
        F: FnMut(Vector3i, u64, u64) -> u64,
    {
        if write_channel_index == read_channel_index {
            self.read_write_area(min, max, write_channel_index, |pos, current| {
                action(pos, current, current)
            });
            return;
        }

        let size = self.size;
        let Some((lo, hi)) = clipped_area(size, min, max) else {
            return;
        };
        self.decompress_channel(write_channel_index);
        let (write_channel, read_channel) =
            channel_pair_mut(&mut self.channels, write_channel_index, read_channel_index);
        let read_defval = read_channel.defval;
        let read_is_uniform = read_channel.compression == Compression::Uniform;
        let read_data = &read_channel.data;
        let write_data = &mut write_channel.data;

        match write_data {
            ChannelData::U8(w) => {
                for_each_index_and_pos(size, lo, hi, |i, pos| {
                    let current = w[i] as u64;
                    let read_value = read_channel_value(read_data, i, read_defval, read_is_uniform);
                    w[i] = action(pos, current, read_value) as u8;
                });
            }
            ChannelData::U16(w) => {
                for_each_index_and_pos(size, lo, hi, |i, pos| {
                    let current = w[i] as u64;
                    let read_value = read_channel_value(read_data, i, read_defval, read_is_uniform);
                    w[i] = action(pos, current, read_value) as u16;
                });
            }
            ChannelData::U32(w) => {
                for_each_index_and_pos(size, lo, hi, |i, pos| {
                    let current = w[i] as u64;
                    let read_value = read_channel_value(read_data, i, read_defval, read_is_uniform);
                    w[i] = action(pos, current, read_value) as u32;
                });
            }
            ChannelData::U64(w) => {
                for_each_index_and_pos(size, lo, hi, |i, pos| {
                    let current = w[i];
                    let read_value = read_channel_value(read_data, i, read_defval, read_is_uniform);
                    w[i] = action(pos, current, read_value);
                });
            }
        }
    }

    /// True if a channel is uniform (all voxels equal its default). Matches
    /// `is_uniform`. Compressed channels are uniform by definition.
    pub fn is_uniform(&self, channel_index: usize) -> bool {
        let ch = &self.channels[channel_index];
        if ch.compression == Compression::Uniform {
            return true;
        }
        match &ch.data {
            ChannelData::U8(v) => v.iter().all(|&x| x == v[0]),
            ChannelData::U16(v) => v.iter().all(|&x| x == v[0]),
            ChannelData::U32(v) => v.iter().all(|&x| x == v[0]),
            ChannelData::U64(v) => v.iter().all(|&x| x == v[0]),
        }
    }

    /// Decompress a channel (allocate and fill with its default). No-op if
    /// already `NONE`. Matches `decompress_channel`.
    pub fn decompress_channel(&mut self, channel_index: usize) {
        // Snapshot the channel's immutable fields before the mutable borrow, to
        // avoid holding `&mut channel` across the allocation.
        let (compression, depth, defval) = {
            let ch = &self.channels[channel_index];
            (ch.compression, ch.depth, ch.defval)
        };
        if compression == Compression::None {
            return;
        }
        let volume = Channel::size_in_bytes_for_volume(self.size, depth);
        let voxel_count = (self.size.volume_u64()) as usize;
        let mut data = self.alloc_typed(depth, voxel_count);
        data.fill_u64(defval);
        let ch = &mut self.channels[channel_index];
        ch.data = data;
        ch.size_in_bytes = volume as u32;
        ch.compression = Compression::None;
    }

    /// Compress any channel whose voxels are all equal into a uniform default.
    /// Matches `compress_uniform_channels`.
    pub fn compress_uniform_channels(&mut self) {
        for ci in 0..MAX_CHANNELS {
            if self.channels[ci].compression == Compression::None && self.is_uniform(ci) {
                let defval = self.channels[ci].data.get_u64(0);
                let ch = &mut self.channels[ci];
                free_channel_data(self.allocator, self.pool.as_ref(), ch);
                ch.defval = defval;
                ch.compression = Compression::Uniform;
            }
        }
    }

    /// Mirror a channel along the given axis (0=X, 1=Y, 2=Z).
    /// Matches upstream `mirror(channel, axis)`.
    pub fn mirror(&mut self, channel_index: usize, axis: usize) {
        let axis = axis.min(2);
        if self.channels[channel_index].compression == Compression::Uniform {
            return; // uniform channels are already mirrored
        }
        self.decompress_channel(channel_index);
        let size = self.size;
        let data = &mut self.channels[channel_index].data;
        match axis {
            0 => {
                // Mirror X: swap (x,y,z) <-> (sx-1-x,y,z)
                for z in 0..size.z {
                    for y in 0..size.y {
                        for x in 0..size.x / 2 {
                            let i_a = voxel_index(size, x as usize, y as usize, z as usize);
                            let i_b = voxel_index(
                                size,
                                (size.x - 1 - x) as usize,
                                y as usize,
                                z as usize,
                            );
                            let a = data.get_u64(i_a);
                            let b = data.get_u64(i_b);
                            data.set_u64(i_a, b);
                            data.set_u64(i_b, a);
                        }
                    }
                }
            }
            1 => {
                for z in 0..size.z {
                    for y in 0..size.y / 2 {
                        for x in 0..size.x {
                            let i_a = voxel_index(size, x as usize, y as usize, z as usize);
                            let i_b = voxel_index(
                                size,
                                x as usize,
                                (size.y - 1 - y) as usize,
                                z as usize,
                            );
                            let a = data.get_u64(i_a);
                            let b = data.get_u64(i_b);
                            data.set_u64(i_a, b);
                            data.set_u64(i_b, a);
                        }
                    }
                }
            }
            _ => {
                for z in 0..size.z / 2 {
                    for y in 0..size.y {
                        for x in 0..size.x {
                            let i_a = voxel_index(size, x as usize, y as usize, z as usize);
                            let i_b = voxel_index(
                                size,
                                x as usize,
                                y as usize,
                                (size.z - 1 - z) as usize,
                            );
                            let a = data.get_u64(i_a);
                            let b = data.get_u64(i_b);
                            data.set_u64(i_a, b);
                            data.set_u64(i_b, a);
                        }
                    }
                }
            }
        }
    }

    /// Rotate a channel 90° around the given axis (0=X, 1=Y, 2=Z), clockwise.
    /// Matches upstream `rotate_90(channel, axis, clockwise)`.
    pub fn rotate_90(&mut self, channel_index: usize, axis: usize, clockwise: bool) {
        let axis = axis.min(2);
        if self.channels[channel_index].compression == Compression::Uniform {
            return;
        }
        self.decompress_channel(channel_index);
        let size = self.size;
        let data = &mut self.channels[channel_index].data;
        let old_values: Vec<u64> = (0..(size.x as usize * size.y as usize * size.z as usize))
            .map(|i| data.get_u64(i))
            .collect();
        match axis {
            0 => {
                // Rotate in the YZ plane
                for z in 0..size.z {
                    for y in 0..size.y {
                        let (ny, nz) = if clockwise {
                            (size.z - 1 - z, y)
                        } else {
                            (z, size.y - 1 - y)
                        };
                        for x in 0..size.x {
                            let old_i = voxel_index(size, x as usize, y as usize, z as usize);
                            let new_i = voxel_index(size, x as usize, ny as usize, nz as usize);
                            data.set_u64(new_i, old_values[old_i]);
                        }
                    }
                }
            }
            1 => {
                // Rotate in the XZ plane
                for z in 0..size.z {
                    for x in 0..size.x {
                        let (nx, nz) = if clockwise {
                            (size.z - 1 - z, x)
                        } else {
                            (z, size.x - 1 - x)
                        };
                        for y in 0..size.y {
                            let old_i = voxel_index(size, x as usize, y as usize, z as usize);
                            let new_i = voxel_index(size, nx as usize, y as usize, nz as usize);
                            data.set_u64(new_i, old_values[old_i]);
                        }
                    }
                }
            }
            _ => {
                // Rotate in the XY plane
                for y in 0..size.y {
                    for x in 0..size.x {
                        let (nx, ny) = if clockwise {
                            (size.y - 1 - y, x)
                        } else {
                            (y, size.x - 1 - x)
                        };
                        for z in 0..size.z {
                            let old_i = voxel_index(size, x as usize, y as usize, z as usize);
                            let new_i = voxel_index(size, nx as usize, ny as usize, z as usize);
                            data.set_u64(new_i, old_values[old_i]);
                        }
                    }
                }
            }
        }
    }

    /// Buffer-level op: add another buffer's values to this one (per-channel).
    /// Matches upstream `op_add_buffer_f`.
    pub fn op_add_buffer_f(&mut self, other: &VoxelBuffer, channel_index: usize) {
        if self.size != other.size {
            return;
        }
        self.decompress_channel(channel_index);
        let data = &mut self.channels[channel_index].data;
        let depth = self.channels[channel_index].depth;
        let (sx, sy, sz) = (
            self.size.x as usize,
            self.size.y as usize,
            self.size.z as usize,
        );
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let i = voxel_index(self.size, x, y, z);
                    let a = raw_voxel_to_real(data.get_u64(i), depth);
                    let b = other.get_voxel_f(x as i32, y as i32, z as i32, channel_index);
                    data.set_u64(i, real_to_raw_voxel(a + b, depth));
                }
            }
        }
    }

    /// Buffer-level op: subtract another buffer's values from this one.
    pub fn op_sub_buffer_f(&mut self, other: &VoxelBuffer, channel_index: usize) {
        if self.size != other.size {
            return;
        }
        self.decompress_channel(channel_index);
        let data = &mut self.channels[channel_index].data;
        let depth = self.channels[channel_index].depth;
        let (sx, sy, sz) = (
            self.size.x as usize,
            self.size.y as usize,
            self.size.z as usize,
        );
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let i = voxel_index(self.size, x, y, z);
                    let a = raw_voxel_to_real(data.get_u64(i), depth);
                    let b = other.get_voxel_f(x as i32, y as i32, z as i32, channel_index);
                    data.set_u64(i, real_to_raw_voxel(a - b, depth));
                }
            }
        }
    }

    /// Buffer-level op: element-wise minimum.
    pub fn op_min_buffer_f(&mut self, other: &VoxelBuffer, channel_index: usize) {
        if self.size != other.size {
            return;
        }
        self.decompress_channel(channel_index);
        let data = &mut self.channels[channel_index].data;
        let depth = self.channels[channel_index].depth;
        let (sx, sy, sz) = (
            self.size.x as usize,
            self.size.y as usize,
            self.size.z as usize,
        );
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let i = voxel_index(self.size, x, y, z);
                    let a = raw_voxel_to_real(data.get_u64(i), depth);
                    let b = other.get_voxel_f(x as i32, y as i32, z as i32, channel_index);
                    data.set_u64(i, real_to_raw_voxel(a.min(b), depth));
                }
            }
        }
    }

    /// Buffer-level op: element-wise maximum.
    pub fn op_max_buffer_f(&mut self, other: &VoxelBuffer, channel_index: usize) {
        if self.size != other.size {
            return;
        }
        self.decompress_channel(channel_index);
        let data = &mut self.channels[channel_index].data;
        let depth = self.channels[channel_index].depth;
        let (sx, sy, sz) = (
            self.size.x as usize,
            self.size.y as usize,
            self.size.z as usize,
        );
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let i = voxel_index(self.size, x, y, z);
                    let a = raw_voxel_to_real(data.get_u64(i), depth);
                    let b = other.get_voxel_f(x as i32, y as i32, z as i32, channel_index);
                    data.set_u64(i, real_to_raw_voxel(a.max(b), depth));
                }
            }
        }
    }

    /// Paste another buffer into this one at the given offset.
    /// Matches upstream `paste(src, src_min, dst_min, channel_mask)`.
    pub fn paste(
        &mut self,
        src: &VoxelBuffer,
        src_min: Vector3i,
        dst_min: Vector3i,
        channel_mask: u8,
    ) {
        let dst_size = self.size;
        let src_size = src.size;
        for ci in 0..MAX_CHANNELS {
            if channel_mask & (1 << ci) == 0 {
                continue;
            }
            self.decompress_channel(ci);
            for dz in 0..src_size.z {
                let sz = dz + src_min.z;
                let dz_dst = dz + dst_min.z - src_min.z;
                if dz_dst < 0 || dz_dst >= dst_size.z {
                    continue;
                }
                for dy in 0..src_size.y {
                    let sy = dy + src_min.y;
                    let dy_dst = dy + dst_min.y - src_min.y;
                    if dy_dst < 0 || dy_dst >= dst_size.y {
                        continue;
                    }
                    for dx in 0..src_size.x {
                        let sx = dx + src_min.x;
                        let dx_dst = dx + dst_min.x - src_min.x;
                        if dx_dst < 0 || dx_dst >= dst_size.x {
                            continue;
                        }
                        let val = src.get_voxel(sx, sy, sz, ci);
                        self.set_voxel(val, dx_dst, dy_dst, dz_dst, ci);
                    }
                }
            }
        }
        self.copy_voxel_metadata_from(src, src_min, dst_min);
    }

    /// Raw bytes of a channel (decompressed). Matches `get_channel_as_bytes`.
    /// Decompresses if needed. Returns a LE byte view over the typed storage
    /// (see [`ChannelData::as_bytes_mut`]) — wire-format-stable.
    pub fn channel_bytes_mut(&mut self, channel_index: usize) -> &mut [u8] {
        self.decompress_channel(channel_index);
        self.channels[channel_index].data.as_bytes_mut()
    }

    /// Raw bytes of a channel (read-only). If compressed, returns an empty slice
    /// — callers should use `defval` in that case. For a guaranteed-materialized
    /// view, call `decompress_channel` first. Returns a LE byte view over the
    /// typed storage (see [`ChannelData::as_bytes`]) — wire-format-stable.
    pub fn channel_bytes(&self, channel_index: usize) -> &[u8] {
        self.channels[channel_index].data.as_bytes()
    }

    /// Typed voxel data of a channel (read-only). For a `Compression::Uniform`
    /// channel the returned [`ChannelData`] is empty — callers should use
    /// [`channel_default`](Self::channel_default) in that case. For a
    /// guaranteed-materialized view, call [`decompress_channel`](Self::decompress_channel) first.
    ///
    /// This is the depth-dispatch entry point for hot loops that want to index
    /// a typed slice directly (one `match` on the variant, then `slice[i]`)
    /// instead of paying per-voxel depth dispatch through `get_voxel`.
    pub fn channel_data(&self, channel_index: usize) -> &ChannelData {
        &self.channels[channel_index].data
    }

    /// The uniform default value of a channel.
    pub fn channel_default(&self, channel_index: usize) -> u64 {
        self.channels[channel_index].defval
    }

    /// Copy an entire channel from `other`. Matches `copy_channel_from`.
    pub fn copy_channel_from(&mut self, other: &VoxelBuffer, channel_index: usize) {
        assert_eq!(
            self.size, other.size,
            "copy_channel_from requires equal buffer sizes"
        );
        let src = &other.channels[channel_index];
        assert_eq!(
            self.channels[channel_index].depth, src.depth,
            "copy_channel_from requires equal channel depths"
        );

        if src.compression == Compression::None {
            // Clone the typed storage directly (same depth ⇒ same variant).
            let data = src.data.clone();
            let dst = &mut self.channels[channel_index];
            free_channel_data(self.allocator, self.pool.as_ref(), dst);
            dst.defval = src.defval;
            dst.compression = Compression::None;
            dst.size_in_bytes = src.size_in_bytes;
            dst.data = data;
            return;
        }

        let dst = &mut self.channels[channel_index];
        free_channel_data(self.allocator, self.pool.as_ref(), dst);
        dst.defval = src.defval;
        dst.compression = Compression::Uniform;
    }

    /// Copy all voxel data, channel depths, and in-memory metadata into `dst`.
    pub fn copy_to(&self, dst: &mut VoxelBuffer) {
        dst.create(self.size);
        for ci in 0..MAX_CHANNELS {
            dst.set_channel_depth(ci, self.channels[ci].depth);
        }
        dst.copy_channels_from(self);
        dst.block_metadata = self.block_metadata.clone();
        dst.voxel_metadata.clone_from(&self.voxel_metadata);
    }

    pub fn copy_to_owned(&self) -> VoxelBuffer {
        let mut dst = VoxelBuffer::new(self.allocator);
        if let Some(pool) = &self.pool {
            dst = dst.with_pool(pool.clone());
        }
        self.copy_to(&mut dst);
        dst
    }

    /// Fallible deep copy used by transactional persistence preparation.
    /// Every dense channel reserves its exact typed element count before any
    /// source bytes are copied, so allocation failure remains a recoverable
    /// pre-publication error instead of aborting the process.
    pub(crate) fn try_copy_to_owned(
        &self,
    ) -> Result<VoxelBuffer, std::collections::TryReserveError> {
        let mut dst = VoxelBuffer::new(self.allocator);
        dst.size = self.size;
        dst.pool = self.pool.clone();
        for (source, target) in self.channels.iter().zip(dst.channels.iter_mut()) {
            target.data = source.data.try_clone()?;
            target.defval = source.defval;
            target.depth = source.depth;
            target.compression = source.compression;
            target.size_in_bytes = source.size_in_bytes;
        }
        dst.block_metadata = self.block_metadata.clone();
        dst.voxel_metadata.clone_from(&self.voxel_metadata);
        Ok(dst)
    }

    /// Copy a rectangular area of one channel from `other`. Matches the C++
    /// `copy_channel_from(other, src_min, src_max, dst_min, channel)` overload.
    pub fn copy_channel_from_area(
        &mut self,
        other: &VoxelBuffer,
        mut src_min: Vector3i,
        mut src_max: Vector3i,
        mut dst_min: Vector3i,
        channel_index: usize,
    ) {
        let src = &other.channels[channel_index];
        assert_eq!(
            self.channels[channel_index].depth, src.depth,
            "copy_channel_from_area requires equal channel depths"
        );

        Vector3i::sort_min_max(&mut src_min, &mut src_max);
        funcs::clip_copy_region(
            &mut src_min,
            &mut src_max,
            other.size,
            &mut dst_min,
            self.size,
        );
        let area_size = src_max - src_min;
        if area_size.x <= 0 || area_size.y <= 0 || area_size.z <= 0 {
            return;
        }

        if src.compression == Compression::None {
            if self.channels[channel_index].compression == Compression::Uniform {
                self.decompress_channel(channel_index);
            }
            // Typed row-by-row copy: same depth ⇒ same variant. The helper
            // dispatches once per channel instead of per-voxel.
            copy_channel_region_typed(
                &mut self.channels[channel_index].data,
                self.size,
                dst_min,
                &src.data,
                other.size,
                src_min,
                src_max,
            );
            return;
        }

        if self.channels[channel_index].compression == Compression::Uniform
            && self.channels[channel_index].defval == src.defval
        {
            return;
        }

        self.fill_area(src.defval, dst_min, dst_min + area_size, channel_index);
    }

    /// Nearest-neighbor 2:1 downscale of all channels from a region of `self`
    /// into a region of `dst`. Matches `VoxelBuffer::downscale_to`.
    ///
    /// For each destination voxel `dst_pos`, the source voxel sampled is
    /// `src_min + ((dst_pos - dst_min) << 1)`. Channels that are uniform on
    /// both ends with equal defaults are skipped (no allocation, no writes).
    /// This is the mip-map kernel used by [`crate::storage::VoxelData`] to
    /// cascade edits up the LOD chain.
    pub fn downscale_to(
        &self,
        dst: &mut VoxelBuffer,
        mut src_min: Vector3i,
        mut src_max: Vector3i,
        mut dst_min: Vector3i,
    ) {
        // Clamp source region into this buffer.
        src_min = src_min.clamp(Vector3i::zero(), self.size - Vector3i::splat(1));
        src_max = src_max.clamp(Vector3i::zero(), self.size);

        let dst_max_raw = dst_min + ((src_max - src_min) >> 1);

        // Clamp destination region into `dst`.
        dst_min = dst_min.clamp(Vector3i::zero(), dst.size - Vector3i::splat(1));
        let dst_max = dst_max_raw.clamp(Vector3i::zero(), dst.size);

        for channel_index in 0..MAX_CHANNELS {
            let src_compression = self.channel_compression(channel_index);
            let dst_compression = dst.channel_compression(channel_index);
            let src_defval = self.channel_default(channel_index);
            let dst_defval = dst.channel_default(channel_index);

            // If both channels carry the same uniform default there is nothing
            // to do — the destination already matches. Matches the C++ fast path.
            if src_compression == Compression::Uniform
                && dst_compression == Compression::Uniform
                && src_defval == dst_defval
            {
                continue;
            }

            // ZXY iteration matches the C++ loop order so downscaled buffers
            // remain byte-comparable with the reference implementation.
            dst.read_write_area(dst_min, dst_max, channel_index, |dst_pos, _dst_value| {
                let src_pos = src_min + ((dst_pos - dst_min) << 1);
                // Source bounds were clamped above; verify defensively.
                debug_assert!(src_pos.x >= 0 && src_pos.y >= 0 && src_pos.z >= 0);
                debug_assert!(src_pos.x < self.size.x);
                debug_assert!(src_pos.y < self.size.y);
                debug_assert!(src_pos.z < self.size.z);

                if src_compression == Compression::Uniform {
                    src_defval
                } else {
                    self.get_voxel(src_pos.x, src_pos.y, src_pos.z, channel_index)
                }
            });
        }
    }

    /// Copy all channels from `other`. Matches `copy_channels_from`.
    pub fn copy_channels_from(&mut self, other: &VoxelBuffer) {
        for ci in 0..MAX_CHANNELS {
            self.copy_channel_from(other, ci);
        }
    }

    /// Replace the buffer-wide (block) metadata entry.
    pub fn set_block_metadata(&mut self, metadata: MetadataValue) {
        self.block_metadata = metadata;
    }

    /// Buffer-wide metadata. `Nil` when none is set.
    pub fn block_metadata(&self) -> &MetadataValue {
        &self.block_metadata
    }

    /// Set or clear metadata for one local voxel. `Nil` removes the entry.
    pub fn set_voxel_metadata(&mut self, position: Vector3i, metadata: MetadataValue) {
        if metadata.is_nil() {
            self.voxel_metadata.remove(&position);
            return;
        }
        self.voxel_metadata.insert(position, metadata);
    }

    /// Metadata for one local voxel, if any.
    pub fn voxel_metadata(&self, position: Vector3i) -> Option<&MetadataValue> {
        self.voxel_metadata.get(&position)
    }

    /// Remove metadata for one local voxel.
    pub fn clear_voxel_metadata(&mut self, position: Vector3i) {
        self.voxel_metadata.remove(&position);
    }

    /// Drop every per-voxel metadata entry. Block metadata is left untouched.
    pub fn clear_all_voxel_metadata(&mut self) {
        self.voxel_metadata.clear();
    }

    /// True when at least one voxel carries metadata.
    pub fn has_voxel_metadata(&self) -> bool {
        !self.voxel_metadata.is_empty()
    }

    /// Visit every per-voxel metadata entry.
    pub fn for_each_voxel_metadata(&self, mut f: impl FnMut(Vector3i, &MetadataValue)) {
        for (position, value) in &self.voxel_metadata {
            f(*position, value);
        }
    }

    /// Visit metadata whose local position is in `[min, max)` (max exclusive).
    pub fn for_each_voxel_metadata_in_area(
        &self,
        min: Vector3i,
        max: Vector3i,
        mut f: impl FnMut(Vector3i, &MetadataValue),
    ) {
        let (lo, hi) = sorted_half_open_box(min, max);
        for (position, value) in &self.voxel_metadata {
            if point_in_half_open_box(*position, lo, hi) {
                f(*position, value);
            }
        }
    }

    /// Next metadata position in `[min, max)` strictly after `start` in ZXY
    /// order. Used by the GDScript iterator helper.
    pub fn next_voxel_metadata_pos_in_area(
        &self,
        min: Vector3i,
        max: Vector3i,
        start: Vector3i,
    ) -> Option<Vector3i> {
        let (lo, hi) = sorted_half_open_box(min, max);
        let mut best: Option<Vector3i> = None;
        for position in self.voxel_metadata.keys().copied() {
            if !point_in_half_open_box(position, lo, hi) {
                continue;
            }
            if !metadata_pos_after(position, start) {
                continue;
            }
            if best.is_none_or(|current| metadata_pos_after(current, position)) {
                best = Some(position);
            }
        }
        best
    }

    /// Copy overlapping per-voxel metadata from `src` using the same
    /// `src_min`/`dst_min` convention as [`paste`](Self::paste).
    pub fn copy_voxel_metadata_from(
        &mut self,
        src: &VoxelBuffer,
        src_min: Vector3i,
        dst_min: Vector3i,
    ) {
        if src.voxel_metadata.is_empty() {
            return;
        }
        let src_size = src.size;
        let dst_size = self.size;
        for (src_pos, value) in &src.voxel_metadata {
            if src_pos.x < src_min.x || src_pos.y < src_min.y || src_pos.z < src_min.z {
                continue;
            }
            if src_pos.x >= src_size.x || src_pos.y >= src_size.y || src_pos.z >= src_size.z {
                continue;
            }
            let dst_pos = Vector3i::new(
                dst_min.x + (src_pos.x - src_min.x),
                dst_min.y + (src_pos.y - src_min.y),
                dst_min.z + (src_pos.z - src_min.z),
            );
            if dst_pos.x < 0 || dst_pos.y < 0 || dst_pos.z < 0 {
                continue;
            }
            if dst_pos.x >= dst_size.x || dst_pos.y >= dst_size.y || dst_pos.z >= dst_size.z {
                continue;
            }
            self.voxel_metadata.insert(dst_pos, value.clone());
        }
    }

    // ---- internal helpers ----

    /// Allocate a typed voxel buffer of `voxel_count` elements for `depth`.
    ///
    /// Pool recycling for typed storage is deferred (the byte-oriented
    /// `VoxelMemoryPool` is test-only in the Rust port today — no production
    /// path selects `Allocator::Pool`). The `Allocator::Pool` API surface is
    /// kept for C++ parity; when a production caller wires one up (likely the
    /// Phase 5 FFI bridge), `VoxelMemoryPool` should grow per-depth typed
    /// buckets. See audit §9.6-D7 / §11.2 M1.B.
    fn alloc_typed(&self, depth: ChannelDepth, voxel_count: usize) -> ChannelData {
        let _ = (self.allocator, &self.pool);
        ChannelData::new_for_depth(depth, voxel_count)
    }
}

/// Free a channel's data. Pool recycling for typed storage is deferred (see
/// [`VoxelBuffer::alloc_typed`]); today this just drops the typed buffer. Free
/// function (not a method) to avoid borrow conflicts when called while holding
/// `&mut self.channels[i]`.
fn free_channel_data(allocator: Allocator, pool: Option<&Arc<VoxelMemoryPool>>, ch: &mut Channel) {
    let _ = (allocator, pool);
    if ch.data.is_empty() {
        ch.size_in_bytes = 0;
        return;
    }
    ch.data = ChannelData::default();
    ch.size_in_bytes = 0;
}

impl Drop for VoxelBuffer {
    fn drop(&mut self) {
        // Pool recycling for typed storage is deferred (see `alloc_typed`);
        // typed buffers are dropped normally. Kept for parity with the future
        // pool integration.
        let _ = (&self.allocator, &self.pool);
        for ch in &mut self.channels {
            ch.data = ChannelData::default();
        }
    }
}

// ---- free helpers ----

fn clipped_area(size: Vector3i, min: Vector3i, max: Vector3i) -> Option<(Vector3i, Vector3i)> {
    let mut lo = min;
    let mut hi = max;
    Vector3i::sort_min_max(&mut lo, &mut hi);
    lo.x = crate::math::funcs::clamp(lo.x, 0, size.x);
    lo.y = crate::math::funcs::clamp(lo.y, 0, size.y);
    lo.z = crate::math::funcs::clamp(lo.z, 0, size.z);
    hi.x = crate::math::funcs::clamp(hi.x, 0, size.x);
    hi.y = crate::math::funcs::clamp(hi.y, 0, size.y);
    hi.z = crate::math::funcs::clamp(hi.z, 0, size.z);
    if hi.x <= lo.x || hi.y <= lo.y || hi.z <= lo.z {
        return None;
    }
    Some((lo, hi))
}

fn sorted_half_open_box(min: Vector3i, max: Vector3i) -> (Vector3i, Vector3i) {
    let mut lo = min;
    let mut hi = max;
    Vector3i::sort_min_max(&mut lo, &mut hi);
    (lo, hi)
}

fn point_in_half_open_box(position: Vector3i, lo: Vector3i, hi: Vector3i) -> bool {
    position.x >= lo.x
        && position.y >= lo.y
        && position.z >= lo.z
        && position.x < hi.x
        && position.y < hi.y
        && position.z < hi.z
}

fn metadata_pos_after(position: Vector3i, start: Vector3i) -> bool {
    (position.z, position.x, position.y) > (start.z, start.x, start.y)
}

#[inline]
fn for_each_index_and_pos<F>(size: Vector3i, min: Vector3i, max: Vector3i, mut f: F)
where
    F: FnMut(usize, Vector3i),
{
    for z in min.z..max.z {
        for x in min.x..max.x {
            let row_start = voxel_index(size, x as usize, min.y as usize, z as usize);
            for (i, y) in (row_start..).zip(min.y..max.y) {
                f(i, Vector3i::new(x, y, z));
            }
        }
    }
}

fn channel_pair_mut(
    channels: &mut [Channel; MAX_CHANNELS],
    write_channel_index: usize,
    read_channel_index: usize,
) -> (&mut Channel, &Channel) {
    debug_assert_ne!(write_channel_index, read_channel_index);
    if write_channel_index < read_channel_index {
        let (left, right) = channels.split_at_mut(read_channel_index);
        (&mut left[write_channel_index], &right[0])
    } else {
        let (left, right) = channels.split_at_mut(write_channel_index);
        (&mut right[0], &left[read_channel_index])
    }
}

/// Copy a rectangular ZXY sub-region from `src` to `dst`, both typed voxel
/// buffers of the same depth (variant). The variant is dispatched once; the
/// inner copy is row-by-row over a typed slice (`item_size` is implicit —
/// `T: Copy`). Replaces the old `copy_3d_region_zxy(&[u8], item_size)` call
/// from `copy_channel_from_area`.
///
/// `dst_min`/`src_min`/`src_max` are assumed already clipped to bounds by the
/// caller (via `funcs::clip_copy_region`).
fn copy_channel_region_typed(
    dst: &mut ChannelData,
    dst_size: Vector3i,
    dst_min: Vector3i,
    src: &ChannelData,
    src_size: Vector3i,
    src_min: Vector3i,
    src_max: Vector3i,
) {
    fn copy_rows<T: Copy>(
        dst: &mut [T],
        dst_size: Vector3i,
        dst_min: Vector3i,
        src: &[T],
        src_size: Vector3i,
        src_min: Vector3i,
        src_max: Vector3i,
    ) {
        let area_size = src_max - src_min;
        let dst_row_off = dst_size.y as usize;
        let src_row_off = src_size.y as usize;
        let row_len = area_size.y as usize;
        for z in 0..area_size.z {
            let mut src_ri = voxel_index(
                src_size,
                src_min.x as usize,
                src_min.y as usize,
                (src_min.z + z) as usize,
            );
            let mut dst_ri = voxel_index(
                dst_size,
                dst_min.x as usize,
                dst_min.y as usize,
                (dst_min.z + z) as usize,
            );
            for _x in 0..area_size.x {
                dst[dst_ri..dst_ri + row_len].copy_from_slice(&src[src_ri..src_ri + row_len]);
                src_ri += src_row_off;
                dst_ri += dst_row_off;
            }
        }
    }
    match (dst, src) {
        (ChannelData::U8(d), ChannelData::U8(s)) => {
            copy_rows(d, dst_size, dst_min, s, src_size, src_min, src_max)
        }
        (ChannelData::U16(d), ChannelData::U16(s)) => {
            copy_rows(d, dst_size, dst_min, s, src_size, src_min, src_max)
        }
        (ChannelData::U32(d), ChannelData::U32(s)) => {
            copy_rows(d, dst_size, dst_min, s, src_size, src_min, src_max)
        }
        (ChannelData::U64(d), ChannelData::U64(s)) => {
            copy_rows(d, dst_size, dst_min, s, src_size, src_min, src_max)
        }
        _ => debug_assert!(
            false,
            "copy_channel_region_typed: src/dst ChannelData variants differ"
        ),
    }
}

#[inline]
fn read_channel_value(data: &ChannelData, i: usize, defval: u64, is_uniform: bool) -> u64 {
    if is_uniform {
        defval
    } else {
        data.get_u64(i)
    }
}

/// Index into a flat ZXY channel. Matches `get_index(x,y,z) = y + sy*(x + sx*z)`.
#[inline]
pub fn voxel_index(size: Vector3i, x: usize, y: usize, z: usize) -> usize {
    debug_assert!(x < size.x as usize && y < size.y as usize && z < size.z as usize);
    y + (size.y as usize) * (x + (size.x as usize) * z)
}

/// Default depth for a channel at linear index `i` (matches DEFAULT_*_CHANNEL_DEPTH).
fn default_depth_for_channel_index(i: usize) -> ChannelDepth {
    match i {
        0 => DEFAULT_TYPE_CHANNEL_DEPTH,
        1 => DEFAULT_SDF_CHANNEL_DEPTH,
        3 => DEFAULT_INDICES_CHANNEL_DEPTH,
        4 => DEFAULT_WEIGHTS_CHANNEL_DEPTH,
        _ => DEFAULT_CHANNEL_DEPTH,
    }
}

fn default_channel_for_index(i: usize) -> Channel {
    let depth = default_depth_for_channel_index(i);
    Channel {
        depth,
        defval: get_default_raw_value(channel_id_from_index(i).unwrap(), depth),
        ..Default::default()
    }
}

/// Recover a `ChannelId` from a linear index, or `None` if out of range.
pub fn channel_id_from_index(i: usize) -> Option<ChannelId> {
    match i {
        0 => Some(ChannelId::Type),
        1 => Some(ChannelId::Sdf),
        2 => Some(ChannelId::Color),
        3 => Some(ChannelId::Indices),
        4 => Some(ChannelId::Weights),
        5 => Some(ChannelId::Data5),
        6 => Some(ChannelId::Data6),
        7 => Some(ChannelId::Data7),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_sets_defaults() {
        let vb = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        assert_eq!(vb.size(), Vector3i::new(4, 4, 4));
        // All channels uniform by default.
        for ci in 0..MAX_CHANNELS {
            assert_eq!(vb.channel_compression(ci), Compression::Uniform);
        }
        // Type channel at 16-bit, SDF at 16-bit, color at 8-bit.
        assert_eq!(
            vb.channel_depth(ChannelId::Type.index()),
            ChannelDepth::Bit16
        );
        assert_eq!(
            vb.channel_depth(ChannelId::Sdf.index()),
            ChannelDepth::Bit16
        );
        assert_eq!(
            vb.channel_depth(ChannelId::Color.index()),
            ChannelDepth::Bit8
        );
        assert_eq!(
            vb.channel_default(ChannelId::Indices.index()),
            0x3210,
            "C++ mixel4 default indices encode slots 0,1,2,3"
        );
        assert_eq!(
            vb.channel_default(ChannelId::Weights.index()),
            0x000f,
            "C++ mixel4 default weights encode full weight in slot 0"
        );
    }

    #[test]
    fn new_initializes_channel_defaults_before_create() {
        let vb = VoxelBuffer::new(Allocator::Default);
        assert_eq!(vb.size(), Vector3i::zero());
        assert_eq!(
            vb.channel_depth(ChannelId::Type.index()),
            ChannelDepth::Bit16
        );
        assert_eq!(
            vb.channel_depth(ChannelId::Sdf.index()),
            ChannelDepth::Bit16
        );
        assert_eq!(
            vb.channel_default(ChannelId::Sdf.index()),
            funcs::snorm_to_s16(1.0) as u16 as u64
        );
        assert_eq!(
            vb.channel_default(ChannelId::Indices.index()),
            MIXEL4_DEFAULT_INDICES
        );
        assert_eq!(
            vb.channel_default(ChannelId::Weights.index()),
            MIXEL4_DEFAULT_WEIGHTS
        );
    }

    #[test]
    fn create_preserves_existing_channel_depths() {
        let mut vb = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        vb.set_channel_depth(ChannelId::Sdf.index(), ChannelDepth::Bit32);
        vb.set_channel_depth(ChannelId::Color.index(), ChannelDepth::Bit16);
        vb.set_voxel_f(-0.25, 0, 0, 0, ChannelId::Sdf.index());

        vb.create(Vector3i::new(3, 3, 3));

        assert_eq!(vb.size(), Vector3i::new(3, 3, 3));
        assert_eq!(
            vb.channel_depth(ChannelId::Sdf.index()),
            ChannelDepth::Bit32
        );
        assert_eq!(
            vb.channel_depth(ChannelId::Color.index()),
            ChannelDepth::Bit16
        );
        assert_eq!(
            vb.channel_compression(ChannelId::Sdf.index()),
            Compression::Uniform
        );
        assert_eq!(
            vb.get_voxel_f(0, 0, 0, ChannelId::Sdf.index()),
            get_default_sdf_value(ChannelDepth::Bit32)
        );
    }

    #[test]
    fn uniform_get_returns_default() {
        let vb = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        // SDF channel is 16-bit by default; C++ defaults it to max positive
        // snorm, i.e. "far outside"/air, not solid.
        assert_eq!(
            vb.channel_default(ChannelId::Sdf.index()),
            funcs::snorm_to_s16(1.0) as u16 as u64
        );
        // Decoded through C++ QUANTIZED_SDF_16_BITS_SCALE_INV (500.0).
        let f = vb.get_voxel_f(0, 0, 0, ChannelId::Sdf.index());
        assert!(
            (f - 500.0).abs() < 1e-3,
            "SDF default decoded to {f}, want ~500.0"
        );
    }

    #[test]
    fn sdf_quantization_constants_match_cpp_ranges() {
        assert_eq!(QUANTIZED_SDF_8_BITS_SCALE, 0.1);
        assert_eq!(QUANTIZED_SDF_8_BITS_SCALE_INV, 10.0);
        assert_eq!(QUANTIZED_SDF_16_BITS_SCALE, 0.002);
        assert!((QUANTIZED_SDF_16_BITS_SCALE_INV - 500.0).abs() < 1e-3);

        assert_eq!(real_to_raw_voxel(10.0, ChannelDepth::Bit8), 127);
        assert_eq!(real_to_raw_voxel(500.0, ChannelDepth::Bit16), 32767);
        assert!((raw_voxel_to_real(127, ChannelDepth::Bit8) - 10.0).abs() < 1e-6);
        assert!((raw_voxel_to_real(32767, ChannelDepth::Bit16) - 500.0).abs() < 1e-3);
    }

    #[test]
    fn set_voxel_decompresses() {
        let mut vb = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        vb.set_voxel(42, 0, 0, 0, ChannelId::Type.index());
        assert_eq!(
            vb.channel_compression(ChannelId::Type.index()),
            Compression::None
        );
        assert_eq!(vb.get_voxel(0, 0, 0, ChannelId::Type.index()), 42);
        // Other voxels retain the (now-materialized) default of 0.
        assert_eq!(vb.get_voxel(1, 1, 1, ChannelId::Type.index()), 0);
    }

    #[test]
    fn set_channel_depth_resets_materialized_channel_storage() {
        let mut vb = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        vb.set_voxel(42, 0, 0, 0, ChannelId::Type.index());
        assert_eq!(
            vb.channel_compression(ChannelId::Type.index()),
            Compression::None
        );

        vb.set_channel_depth(ChannelId::Type.index(), ChannelDepth::Bit32);

        assert_eq!(
            vb.channel_compression(ChannelId::Type.index()),
            Compression::Uniform
        );
        assert!(vb.channel_bytes(ChannelId::Type.index()).is_empty());
        assert_eq!(
            vb.channel_depth(ChannelId::Type.index()),
            ChannelDepth::Bit32
        );
        vb.set_voxel(0x1122_3344, 1, 1, 1, ChannelId::Type.index());
        assert_eq!(vb.get_voxel(1, 1, 1, ChannelId::Type.index()), 0x1122_3344);
    }

    #[test]
    fn sdf_float_roundtrip_16bit() {
        let mut vb = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        // SDF channel is 16-bit by default.
        vb.set_voxel_f(0.5, 0, 0, 0, ChannelId::Sdf.index());
        let back = vb.get_voxel_f(0, 0, 0, ChannelId::Sdf.index());
        assert!((back - 0.5).abs() < 0.02, "got {back}");
    }

    #[test]
    fn sdf_float_roundtrip_32bit() {
        let mut vb = VoxelBuffer::with_size(Vector3i::new(1, 1, 1));
        // Force 32-bit on the SDF channel.
        vb.channels[ChannelId::Sdf.index()].depth = ChannelDepth::Bit32;
        vb.set_voxel_f(1.25, 0, 0, 0, ChannelId::Sdf.index());
        assert_eq!(vb.get_voxel_f(0, 0, 0, ChannelId::Sdf.index()), 1.25);
    }

    #[test]
    fn fill_makes_uniform() {
        let mut vb = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        vb.set_voxel(1, 0, 0, 0, ChannelId::Type.index()); // decompress
        vb.fill(7, ChannelId::Type.index());
        assert_eq!(
            vb.channel_compression(ChannelId::Type.index()),
            Compression::Uniform
        );
        assert_eq!(vb.channel_default(ChannelId::Type.index()), 7);
        assert_eq!(vb.get_voxel(2, 2, 2, ChannelId::Type.index()), 7);
    }

    #[test]
    fn fill_area_writes_subregion() {
        let mut vb = VoxelBuffer::with_size(Vector3i::new(3, 3, 3));
        vb.fill_area(
            9,
            Vector3i::new(1, 1, 1),
            Vector3i::new(2, 2, 2),
            ChannelId::Type.index(),
        );
        assert_eq!(vb.get_voxel(1, 1, 1, ChannelId::Type.index()), 9);
        assert_eq!(vb.get_voxel(0, 0, 0, ChannelId::Type.index()), 0);
    }

    #[test]
    fn is_uniform_after_uniform_fill() {
        let mut vb = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        vb.set_voxel(5, 0, 0, 0, ChannelId::Type.index());
        assert!(!vb.is_uniform(ChannelId::Type.index()));
        vb.fill(5, ChannelId::Type.index());
        assert!(vb.is_uniform(ChannelId::Type.index()));
    }

    #[test]
    fn compress_uniform_channels() {
        let mut vb = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        // Materialize then fill uniformly.
        vb.set_voxel(3, 0, 0, 0, ChannelId::Type.index());
        vb.fill(3, ChannelId::Type.index());
        // After fill it's already uniform; materialize first to test compress.
        vb.decompress_channel(ChannelId::Type.index());
        assert_eq!(
            vb.channel_compression(ChannelId::Type.index()),
            Compression::None
        );
        vb.compress_uniform_channels();
        assert_eq!(
            vb.channel_compression(ChannelId::Type.index()),
            Compression::Uniform
        );
        assert_eq!(vb.channel_default(ChannelId::Type.index()), 3);
    }

    #[test]
    fn copy_channel_from_clones() {
        let mut a = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        a.set_voxel(11, 0, 0, 0, ChannelId::Type.index());
        let mut b = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        b.copy_channel_from(&a, ChannelId::Type.index());
        assert_eq!(b.get_voxel(0, 0, 0, ChannelId::Type.index()), 11);
    }

    // Pool recycling for typed channel storage (`ChannelData`) is deferred —
    // see audit §9.6-D7 / §11.2 M1.B and `VoxelBuffer::alloc_typed`. The byte-
    // oriented `VoxelMemoryPool` no longer intercepts typed allocations, so
    // these three `Allocator::Pool` round-trip tests would under-count. They
    // are re-enabled once `VoxelMemoryPool` grows per-depth typed buckets (or a
    // production caller wires a pool up, likely in Phase 5).
    #[test]
    #[ignore = "pool recycling for typed ChannelData storage deferred (D7)"]
    fn copy_channel_from_allocates_through_destination_pool() {
        let mut src = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        src.set_voxel(11, 0, 0, 0, ChannelId::Type.index());
        let pool = Arc::new(VoxelMemoryPool::new());
        let mut dst = VoxelBuffer::new(Allocator::Pool).with_pool(pool.clone());
        dst.create(src.size());

        dst.copy_channel_from(&src, ChannelId::Type.index());

        assert_eq!(pool.used_blocks(), 1);
        assert_eq!(dst.get_voxel(0, 0, 0, ChannelId::Type.index()), 11);

        dst.clear_channel(ChannelId::Type.index(), 0);
        assert_eq!(pool.used_blocks(), 0);
    }

    #[test]
    fn copy_channel_from_area_copies_materialized_region() {
        let channel = ChannelId::Type.index();
        let mut src = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        for z in 0..4 {
            for x in 0..4 {
                for y in 0..4 {
                    src.set_voxel((1 + y + 10 * x + 100 * z) as u64, x, y, z, channel);
                }
            }
        }
        let mut dst = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        dst.fill(999, channel);

        dst.copy_channel_from_area(
            &src,
            Vector3i::new(1, 1, 1),
            Vector3i::new(3, 3, 3),
            Vector3i::zero(),
            channel,
        );

        for z in 0..4 {
            for x in 0..4 {
                for y in 0..4 {
                    let expected = if x < 2 && y < 2 && z < 2 {
                        src.get_voxel(x + 1, y + 1, z + 1, channel)
                    } else {
                        999
                    };
                    assert_eq!(dst.get_voxel(x, y, z, channel), expected);
                }
            }
        }
    }

    #[test]
    fn copy_channel_from_area_uniform_source_overwrites_materialized_region() {
        let channel = ChannelId::Type.index();
        let src = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        let mut dst = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        dst.set_voxel(42, 1, 1, 1, channel);
        dst.set_voxel(43, 2, 2, 2, channel);

        dst.copy_channel_from_area(
            &src,
            Vector3i::zero(),
            Vector3i::new(2, 2, 2),
            Vector3i::new(1, 1, 1),
            channel,
        );

        assert_eq!(dst.get_voxel(1, 1, 1, channel), 0);
        assert_eq!(dst.get_voxel(2, 2, 2, channel), 0);
    }

    #[test]
    #[should_panic(expected = "requires equal buffer sizes")]
    fn copy_channel_from_rejects_size_mismatch() {
        let src = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        let mut dst = VoxelBuffer::with_size(Vector3i::new(1, 1, 1));
        dst.copy_channel_from(&src, ChannelId::Type.index());
    }

    #[test]
    fn voxel_metadata_round_trips_and_clears() {
        let mut buf = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        buf.set_block_metadata(MetadataValue::Text("chunk".into()));
        buf.set_voxel_metadata(Vector3i::new(1, 2, 3), MetadataValue::Int(42));
        buf.set_voxel_metadata(Vector3i::new(0, 0, 1), MetadataValue::Float(1.5));
        assert_eq!(buf.block_metadata(), &MetadataValue::Text("chunk".into()));
        assert_eq!(
            buf.voxel_metadata(Vector3i::new(1, 2, 3)),
            Some(&MetadataValue::Int(42))
        );
        buf.set_voxel_metadata(Vector3i::new(1, 2, 3), MetadataValue::Nil);
        assert!(buf.voxel_metadata(Vector3i::new(1, 2, 3)).is_none());
        assert!(buf.has_voxel_metadata());
        buf.clear_all_voxel_metadata();
        assert!(!buf.has_voxel_metadata());
        assert_eq!(buf.block_metadata(), &MetadataValue::Text("chunk".into()));
    }

    #[test]
    fn metadata_survives_copy_to_and_is_cleared_by_create() {
        let mut src = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        src.set_block_metadata(MetadataValue::Int(7));
        src.set_voxel_metadata(Vector3i::new(1, 0, 0), MetadataValue::Bytes(vec![1, 2]));
        let dst = src.copy_to_owned();
        assert_eq!(dst.block_metadata(), &MetadataValue::Int(7));
        assert_eq!(
            dst.voxel_metadata(Vector3i::new(1, 0, 0)),
            Some(&MetadataValue::Bytes(vec![1, 2]))
        );
        src.create(Vector3i::new(2, 2, 2));
        assert!(src.block_metadata().is_nil());
        assert!(!src.has_voxel_metadata());
    }

    #[test]
    fn paste_copies_overlapping_voxel_metadata() {
        let mut src = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        src.set_voxel(3, 0, 0, 0, ChannelId::Type.index());
        src.set_voxel_metadata(Vector3i::new(0, 0, 0), MetadataValue::Int(11));
        src.set_voxel_metadata(Vector3i::new(1, 1, 1), MetadataValue::Int(22));
        let mut dst = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        dst.paste(
            &src,
            Vector3i::zero(),
            Vector3i::new(1, 1, 1),
            1 << ChannelId::Type.index(),
        );
        assert_eq!(
            dst.voxel_metadata(Vector3i::new(1, 1, 1)),
            Some(&MetadataValue::Int(11))
        );
        assert_eq!(
            dst.voxel_metadata(Vector3i::new(2, 2, 2)),
            Some(&MetadataValue::Int(22))
        );
        assert!(dst.voxel_metadata(Vector3i::zero()).is_none());
    }

    #[test]
    fn next_voxel_metadata_pos_walks_zxy_after_start() {
        let mut buf = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        buf.set_voxel_metadata(Vector3i::new(1, 0, 0), MetadataValue::Int(1));
        buf.set_voxel_metadata(Vector3i::new(0, 1, 0), MetadataValue::Int(2));
        buf.set_voxel_metadata(Vector3i::new(0, 0, 1), MetadataValue::Int(3));
        let min = Vector3i::zero();
        let max = Vector3i::splat(4);
        let first = buf
            .next_voxel_metadata_pos_in_area(min, max, Vector3i::new(i32::MIN, i32::MIN, i32::MIN))
            .expect("first");
        let second = buf
            .next_voxel_metadata_pos_in_area(min, max, first)
            .expect("second");
        let third = buf
            .next_voxel_metadata_pos_in_area(min, max, second)
            .expect("third");
        assert!(buf
            .next_voxel_metadata_pos_in_area(min, max, third)
            .is_none());
        assert_eq!(first, Vector3i::new(0, 1, 0));
        assert_eq!(second, Vector3i::new(1, 0, 0));
        assert_eq!(third, Vector3i::new(0, 0, 1));
    }

    #[test]
    fn depth_byte_count() {
        assert_eq!(ChannelDepth::Bit8.byte_count(), 1);
        assert_eq!(ChannelDepth::Bit16.byte_count(), 2);
        assert_eq!(ChannelDepth::Bit32.byte_count(), 4);
        assert_eq!(ChannelDepth::Bit64.byte_count(), 8);
    }

    #[test]
    #[ignore = "pool recycling for typed ChannelData storage deferred (D7)"]
    fn pool_allocator_round_trip() {
        let pool = Arc::new(VoxelMemoryPool::new());
        {
            let mut vb = VoxelBuffer::new(Allocator::Pool).with_pool(pool.clone());
            vb.create(Vector3i::new(4, 4, 4));
            vb.set_voxel(1, 0, 0, 0, ChannelId::Type.index());
            assert_eq!(vb.get_voxel(0, 0, 0, ChannelId::Type.index()), 1);
            // Drop returns the allocation to the pool.
        }
        // Pool should have some memory after the drop (used_blocks back to 0).
        assert_eq!(pool.used_blocks(), 0);
    }

    #[test]
    #[ignore = "pool recycling for typed ChannelData storage deferred (D7)"]
    fn create_recycles_existing_pooled_channel_data() {
        let pool = Arc::new(VoxelMemoryPool::new());
        let mut vb = VoxelBuffer::new(Allocator::Pool).with_pool(pool.clone());
        vb.create(Vector3i::new(4, 4, 4));
        vb.set_voxel(1, 0, 0, 0, ChannelId::Type.index());
        assert_eq!(pool.used_blocks(), 1);

        vb.create(Vector3i::new(2, 2, 2));

        assert_eq!(
            pool.used_blocks(),
            0,
            "create() must return materialized pooled channels before resetting"
        );
    }

    #[test]
    fn channel_id_names() {
        assert_eq!(ChannelId::Type.name(), "type");
        assert_eq!(ChannelId::Sdf.name(), "sdf");
        assert_eq!(ChannelId::Data7.name(), "data7");
    }

    #[test]
    fn real_to_raw_32bit_is_bit_cast() {
        assert_eq!(
            real_to_raw_voxel(1.5, ChannelDepth::Bit32),
            f32::to_bits(1.5) as u64
        );
        assert_eq!(
            raw_voxel_to_real(f32::to_bits(1.5) as u64, ChannelDepth::Bit32),
            1.5
        );
    }

    #[test]
    fn downscale_to_samples_nearest_neighbor_2_to_1() {
        // Build a 4×4×4 source where each voxel carries its ZXY index in the
        // Type channel, so we can verify exactly which source voxel each dst
        // cell sampled.
        let channel = ChannelId::Type.index();
        let mut src = VoxelBuffer::with_size(Vector3i::splat(4));
        for z in 0..4 {
            for x in 0..4 {
                for y in 0..4 {
                    let v = (z * 16 + x * 4 + y) as u64;
                    src.set_voxel(v, x, y, z, channel);
                }
            }
        }

        let mut dst = VoxelBuffer::with_size(Vector3i::splat(2));
        src.downscale_to(
            &mut dst,
            Vector3i::zero(),
            Vector3i::splat(4),
            Vector3i::zero(),
        );

        for z in 0..2 {
            for x in 0..2 {
                for y in 0..2 {
                    let expected = ((z * 2) * 16 + (x * 2) * 4 + (y * 2)) as u64;
                    assert_eq!(dst.get_voxel(x, y, z, channel), expected);
                }
            }
        }
    }

    #[test]
    fn downscale_to_skips_uniform_channels_with_matching_default() {
        let channel = ChannelId::Type.index();
        let mut src = VoxelBuffer::with_size(Vector3i::splat(4));
        src.fill(7, channel);
        // SDF stays at its default far-outside sentinel on both ends.

        let mut dst = VoxelBuffer::with_size(Vector3i::splat(2));
        src.downscale_to(
            &mut dst,
            Vector3i::zero(),
            Vector3i::splat(4),
            Vector3i::zero(),
        );

        // Type channel was uniform-7, dst was uniform-0 → materialized to 7.
        assert_eq!(dst.get_voxel(0, 0, 0, channel), 7);
        // SDF channel was uniform on both ends with equal defaults → untouched,
        // stays uniform (no allocation).
        assert_eq!(
            dst.channel_compression(ChannelId::Sdf.index()),
            Compression::Uniform
        );
    }

    #[test]
    fn downscale_to_clamps_oversized_source_region_into_dst_bounds() {
        // Source region extends past the source buffer; the implementation
        // clamps it to the available 4³ region before sampling. The dst min
        // stays at the origin so the whole dst buffer is filled.
        let channel = ChannelId::Type.index();
        let mut src = VoxelBuffer::with_size(Vector3i::splat(4));
        src.fill(3, channel);
        let mut dst = VoxelBuffer::with_size(Vector3i::splat(2));

        src.downscale_to(
            &mut dst,
            Vector3i::zero(),
            Vector3i::splat(99),
            Vector3i::zero(),
        );

        for z in 0..2 {
            for x in 0..2 {
                for y in 0..2 {
                    assert_eq!(dst.get_voxel(x, y, z, channel), 3);
                }
            }
        }
    }

    #[test]
    fn downscale_to_into_destination_subregion_uses_offset_mapping() {
        // Writing into a non-zero dst_min still maps back to the correct
        // source voxel via `src_min + ((dst_pos - dst_min) << 1)`.
        let channel = ChannelId::Type.index();
        let mut src = VoxelBuffer::with_size(Vector3i::splat(4));
        src.fill(5, channel);
        // Materialize a single marker voxel.
        src.set_voxel(42, 2, 0, 0, channel);

        let mut dst = VoxelBuffer::with_size(Vector3i::splat(4));
        // Downscale the 4³ source into the (1..3)³ region of an 4³ dst buffer.
        src.downscale_to(
            &mut dst,
            Vector3i::zero(),
            Vector3i::splat(4),
            Vector3i::new(1, 1, 1),
        );

        // dst(1,1,1) samples src(0,0,0) = 5; dst(2,*,*) samples src(2,*,*) so
        // dst(2,1,1) = src(2,0,0) = 42.
        assert_eq!(dst.get_voxel(1, 1, 1, channel), 5);
        assert_eq!(dst.get_voxel(2, 1, 1, channel), 42);
    }

    #[test]
    fn voxel_buffer_hot_paths_use_depth_hoisted_helpers() {
        let source = include_str!("voxel_buffer.rs");
        for name in ["get_voxel", "set_voxel", "get_voxel_f", "set_voxel_f"] {
            let marker = ["#[inline]\n    pub fn ", name].concat();
            assert!(
                source.contains(&marker),
                "{name} should stay inline on the hot voxel path"
            );
        }

        let row_index_marker = ["fn for_each", "_index_and_pos"].concat();
        assert!(
            source.contains(&row_index_marker),
            "fill/downscale helpers should compute the ZXY row base outside the inner Y loop"
        );

        let read_write_marker = ["read_write", "_area"].concat();
        assert!(
            source.contains(&read_write_marker),
            "downscale/write-box style loops should dispatch channel depth once before iterating voxels"
        );

        let old_downscale_write = [
            "dst",
            ".set_voxel(value, dst_pos.x, dst_pos.y, dst_pos.z, channel_index)",
        ]
        .concat();
        assert!(
            !source.contains(&old_downscale_write),
            "downscale_to should write through the depth-hoisted area helper"
        );
    }
}
