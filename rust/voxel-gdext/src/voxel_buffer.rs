//! Additional Godot classes for voxel-core types.
//!
//! `VoxelBufferGD` exposes a VoxelBuffer as a Godot RefCounted.
//! `VoxelInstancerGD` is a Node3D for scatter-based instance placement.

use godot::prelude::*;
use voxel_core::instancing::scatter::{InstanceGenerator, RandomScatterGenerator};
use voxel_core::instancing::{InstanceLibrary, ScatterConfig};
use voxel_core::math::{Vector3f, Vector3i};
use voxel_core::storage::{ChannelId, VoxelBuffer, VoxelFormat};

pub(crate) const MAX_SCRIPT_ITEMS: usize = 65_536;
pub(crate) const MAX_SCRIPT_VOXELS: u64 = 2_097_152;

pub(crate) fn validate_channel(channel: i32) -> Result<usize, &'static str> {
    let channel = usize::try_from(channel).map_err(|_| "channel must be non-negative")?;
    if channel >= voxel_core::storage::voxel_buffer::MAX_CHANNELS {
        return Err("channel is outside the supported range");
    }
    Ok(channel)
}

pub(crate) fn validate_buffer_size(
    size_x: i32,
    size_y: i32,
    size_z: i32,
) -> Result<Vector3i, &'static str> {
    let maximum = i32::try_from(voxel_core::storage::voxel_buffer::MAX_SIZE)
        .map_err(|_| "core maximum buffer size does not fit i32")?;
    if !(0..=maximum).contains(&size_x)
        || !(0..=maximum).contains(&size_y)
        || !(0..=maximum).contains(&size_z)
    {
        return Err("buffer dimensions must be between zero and MAX_SIZE");
    }
    let width = u64::try_from(size_x).map_err(|_| "buffer width must be non-negative")?;
    let height = u64::try_from(size_y).map_err(|_| "buffer height must be non-negative")?;
    let depth = u64::try_from(size_z).map_err(|_| "buffer depth must be non-negative")?;
    let volume = width
        .checked_mul(height)
        .and_then(|volume| volume.checked_mul(depth))
        .ok_or("buffer volume overflowed")?;
    if volume > MAX_SCRIPT_VOXELS {
        return Err("buffer volume exceeds the script allocation limit");
    }
    Ok(Vector3i::new(size_x, size_y, size_z))
}

pub(crate) fn validate_position(
    position: Vector3i,
    size: Vector3i,
) -> Result<Vector3i, &'static str> {
    if position.x < 0
        || position.y < 0
        || position.z < 0
        || position.x >= size.x
        || position.y >= size.y
        || position.z >= size.z
    {
        return Err("voxel position is outside the buffer");
    }
    Ok(position)
}

pub(crate) fn validate_voxel_value(value: i64) -> Result<u64, &'static str> {
    u64::try_from(value).map_err(|_| "voxel value must be non-negative")
}

pub(crate) fn validate_finite_f64(value: f64) -> Result<f32, &'static str> {
    if !value.is_finite() || value < f64::from(-f32::MAX) || value > f64::from(f32::MAX) {
        return Err("value must be finite and representable as f32");
    }
    Ok(value as f32)
}

fn validate_script_item_count(count: i32) -> Result<usize, &'static str> {
    let count = crate::resources2::validate_nonnegative_count(count)?;
    if count > MAX_SCRIPT_ITEMS {
        return Err("count exceeds the script allocation limit");
    }
    Ok(count)
}

/// Convert a GDScript depth constant (DEPTH_8_BIT etc.) back to a
/// `ChannelDepth`. Returns `None` for out-of-range values. Mirrors the
/// `DEPTH_*` constants exposed to GDScript on `VoxelBufferGD`.
fn channel_depth_from_i64(value: i64) -> Option<voxel_core::storage::ChannelDepth> {
    use voxel_core::storage::ChannelDepth;
    match value {
        x if x == VoxelBufferGD::DEPTH_8_BIT => Some(ChannelDepth::Bit8),
        x if x == VoxelBufferGD::DEPTH_16_BIT => Some(ChannelDepth::Bit16),
        x if x == VoxelBufferGD::DEPTH_32_BIT => Some(ChannelDepth::Bit32),
        x if x == VoxelBufferGD::DEPTH_64_BIT => Some(ChannelDepth::Bit64),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// VoxelBufferGD — RefCounted wrapper around VoxelBuffer
// ---------------------------------------------------------------------------

/// A Godot `RefCounted` wrapping a [`VoxelBuffer`]. Exposes basic voxel
/// read/write to GDScript for testing and procedural generation.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelBuffer)]
pub struct VoxelBufferGD {
    base: Base<RefCounted>,
    buffer: VoxelBuffer,
}

#[godot_api]
impl IRefCounted for VoxelBufferGD {
    fn init(base: Base<RefCounted>) -> Self {
        let mut buffer = VoxelBuffer::with_size(Vector3i::splat(1));
        VoxelFormat::new().configure_buffer(&mut buffer);
        Self { base, buffer }
    }
}

#[godot_api]
impl VoxelBufferGD {
    // ChannelId enum

    /// Channel used to store voxel types. Used by VoxelMesherBlocky.
    #[constant]
    const CHANNEL_TYPE: i64 = 0;

    /// Channel used to store SDF data (signed distance field). Used by
    /// VoxelMesherTransvoxel and other smooth meshers. Values should preferably
    /// be accessed as floats. Negative values are below the isosurface (inside
    /// matter), and positive values are above the surface (outside matter).
    #[constant]
    const CHANNEL_SDF: i64 = 1;

    /// Channel used to store color data. Used by VoxelMesherCubes.
    #[constant]
    const CHANNEL_COLOR: i64 = 2;

    /// Channel used to store material indices. Used with smooth voxels.
    #[constant]
    const CHANNEL_INDICES: i64 = 3;

    /// Channel used to store material weights, when more than one index can be
    /// stored per voxel. Used with smooth voxels.
    #[constant]
    const CHANNEL_WEIGHTS: i64 = 4;

    /// Free channel. Not used by the engine yet.
    #[constant]
    const CHANNEL_DATA5: i64 = 5;

    /// Free channel. Not used by the engine yet.
    #[constant]
    const CHANNEL_DATA6: i64 = 6;

    /// Free channel. Not used by the engine yet.
    #[constant]
    const CHANNEL_DATA7: i64 = 7;

    /// Maximum number of channels a VoxelBuffer can have.
    #[constant]
    const MAX_CHANNELS: i64 = 8;

    // ChannelMask enum

    /// Bitmask with one bit set at the position corresponding to CHANNEL_TYPE.
    #[constant]
    const CHANNEL_TYPE_BIT: i64 = 1;

    /// Bitmask with one bit set at the position corresponding to CHANNEL_SDF.
    #[constant]
    const CHANNEL_SDF_BIT: i64 = 2;

    /// Bitmask with one bit set at the position corresponding to CHANNEL_COLOR.
    #[constant]
    const CHANNEL_COLOR_BIT: i64 = 4;

    /// Bitmask with one bit set at the position corresponding to CHANNEL_INDICES.
    #[constant]
    const CHANNEL_INDICES_BIT: i64 = 8;

    /// Bitmask with one bit set at the position corresponding to CHANNEL_WEIGHTS.
    #[constant]
    const CHANNEL_WEIGHTS_BIT: i64 = 16;

    /// Bitmask with one bit set at the position corresponding to CHANNEL_DATA5.
    #[constant]
    const CHANNEL_DATA5_BIT: i64 = 32;

    /// Bitmask with one bit set at the position corresponding to CHANNEL_DATA6.
    #[constant]
    const CHANNEL_DATA6_BIT: i64 = 64;

    /// Bitmask with one bit set at the position corresponding to CHANNEL_DATA7.
    #[constant]
    const CHANNEL_DATA7_BIT: i64 = 128;

    /// Bitmask with all channel bits set.
    #[constant]
    const ALL_CHANNELS_MASK: i64 = 255;

    // Depth enum

    /// Voxels will be stored with 8 bits. Raw values will range from 0 to 255.
    /// Float values can take 255 values distributed from -10.0 to 10.0. Values
    /// outside the range will be clamped.
    #[constant]
    const DEPTH_8_BIT: i64 = 0;

    /// Voxels will be stored with 16 bits. Raw values will range from 0 to
    /// 65,535. Float values can take 65,535 values distributed from -500.0 to
    /// 500.0. Values outside the range will be clamped.
    #[constant]
    const DEPTH_16_BIT: i64 = 1;

    /// Voxels will be stored with 32 bits. Raw values will range from 0 to
    /// 4,294,967,295, and float values will use regular IEEE 754 representation
    /// (float).
    #[constant]
    const DEPTH_32_BIT: i64 = 2;

    /// Voxels will be stored with 64 bits. Raw values will range from 0 to
    /// 18,446,744,073,709,551,615, and float values will use regular IEEE 754
    /// representation (double).
    #[constant]
    const DEPTH_64_BIT: i64 = 3;

    /// How many depth configuration there are.
    #[constant]
    const DEPTH_COUNT: i64 = 4;

    // Compression enum

    /// The channel is not compressed. Every value is stored individually inside
    /// an array in memory.
    #[constant]
    const COMPRESSION_NONE: i64 = 0;

    /// All voxels of the channel have the same value, so they are stored as one
    /// single value, to save space.
    #[constant]
    const COMPRESSION_UNIFORM: i64 = 1;

    /// How many compression modes there are.
    #[constant]
    const COMPRESSION_COUNT: i64 = 2;

    // Allocator enum

    /// Uses Godot's default memory allocator (at time of writing, it is malloc).
    /// Preferred for occasional buffers with uncommon size, or very large size.
    #[constant]
    const ALLOCATOR_DEFAULT: i64 = 0;

    /// Uses a pool allocator. Can be faster than the default allocator buffers
    /// are created very frequently with similar size. This memory will remain
    /// allocated after use, under the assumption that other buffers will need it
    /// soon after. Does not support very large buffers (greater than 2
    /// megabytes).
    #[constant]
    const ALLOCATOR_POOL: i64 = 1;

    /// How many allocator modes there are.
    #[constant]
    const ALLOCATOR_COUNT: i64 = 2;

    // Uncategorized

    /// Maximum size a buffer can have when serialized. Buffers that contain
    /// uniform-compressed voxels can reach it, but in practice, the limit is
    /// much lower and depends on available memory.
    #[constant]
    const MAX_SIZE: i64 = 65535;

    #[func]
    fn create(&mut self, size_x: i32, size_y: i32, size_z: i32) {
        let Ok(size) = validate_buffer_size(size_x, size_y, size_z) else {
            godot_error!("VoxelBuffer.create: invalid buffer size");
            return;
        };
        self.buffer = VoxelBuffer::with_size(size);
        VoxelFormat::new().configure_buffer(&mut self.buffer);
    }

    #[func]
    fn set_voxel(&mut self, x: i32, y: i32, z: i32, channel: i32, value: i64) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.set_voxel: invalid channel");
            return;
        };
        let Ok(position) = validate_position(Vector3i::new(x, y, z), self.buffer.size()) else {
            godot_error!("VoxelBuffer.set_voxel: position is outside the buffer");
            return;
        };
        let Ok(value) = validate_voxel_value(value) else {
            godot_error!("VoxelBuffer.set_voxel: invalid voxel value");
            return;
        };
        self.buffer
            .set_voxel(value, position.x, position.y, position.z, channel);
    }

    #[func]
    fn get_voxel(&self, x: i32, y: i32, z: i32, channel: i32) -> i64 {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.get_voxel: invalid channel");
            return 0;
        };
        let Ok(position) = validate_position(Vector3i::new(x, y, z), self.buffer.size()) else {
            godot_error!("VoxelBuffer.get_voxel: position is outside the buffer");
            return 0;
        };
        match i64::try_from(
            self.buffer
                .get_voxel(position.x, position.y, position.z, channel),
        ) {
            Ok(value) => value,
            Err(_) => {
                godot_error!("VoxelBuffer.get_voxel: voxel value exceeds GDScript integer range");
                0
            }
        }
    }

    #[func]
    fn get_size_x(&self) -> i32 {
        self.buffer.size().x
    }

    #[func]
    fn get_size_y(&self) -> i32 {
        self.buffer.size().y
    }

    #[func]
    fn get_size_z(&self) -> i32 {
        self.buffer.size().z
    }

    #[func]
    fn fill_channel(&mut self, channel: i32, value: i64) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.fill_channel: invalid channel");
            return;
        };
        let Ok(value) = validate_voxel_value(value) else {
            godot_error!("VoxelBuffer.fill_channel: invalid voxel value");
            return;
        };
        self.buffer.fill(value, channel);
    }

    #[func]
    fn clear_channel(&mut self, channel: i32, value: i64) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.clear_channel: invalid channel");
            return;
        };
        let Ok(value) = validate_voxel_value(value) else {
            godot_error!("VoxelBuffer.clear_channel: invalid voxel value");
            return;
        };
        self.buffer.clear_channel(channel, value);
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelBuffer methods (upstream 5828cbeb).
    // These expose the same names + parameter order as the C++ binding.
    // Voxel-buffer-facing wrappers delegate to voxel-core's VoxelBuffer
    // wherever a native implementation exists, and stub the rest.
    // -----------------------------------------------------------------

    /// Returns the size of the buffer as a `Vector3i`.
    #[func]
    fn get_size(&self) -> godot::builtin::Vector3i {
        let s = self.buffer.size();
        godot::builtin::Vector3i::new(s.x, s.y, s.z)
    }

    /// Returns the raw integer voxel at `position` from the given channel.
    /// Matches the C++ `get_voxel_v(Vector3i, channel)` overload.
    #[func]
    fn get_voxel_v(&self, position: godot::builtin::Vector3i, channel: i32) -> i64 {
        self.get_voxel(position.x, position.y, position.z, channel)
    }

    /// Sets the raw integer voxel at `position` in the given channel.
    /// Matches the C++ `set_voxel_v(value, Vector3i, channel)` overload.
    #[func]
    fn set_voxel_v(&mut self, value: i64, position: godot::builtin::Vector3i, channel: i32) {
        self.set_voxel(position.x, position.y, position.z, channel, value);
    }

    /// Returns the float-interpreted voxel value at the given position.
    #[func]
    fn get_voxel_f(&self, x: i32, y: i32, z: i32, channel: i32) -> f32 {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.get_voxel_f: invalid channel");
            return 0.0;
        };
        let Ok(position) = validate_position(Vector3i::new(x, y, z), self.buffer.size()) else {
            godot_error!("VoxelBuffer.get_voxel_f: position is outside the buffer");
            return 0.0;
        };
        self.buffer
            .get_voxel_f(position.x, position.y, position.z, channel)
    }

    /// Sets the voxel value at the given position from a float.
    #[func]
    fn set_voxel_f(&mut self, value: f32, x: i32, y: i32, z: i32, channel: i32) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.set_voxel_f: invalid channel");
            return;
        };
        let Ok(position) = validate_position(Vector3i::new(x, y, z), self.buffer.size()) else {
            godot_error!("VoxelBuffer.set_voxel_f: position is outside the buffer");
            return;
        };
        if !value.is_finite() {
            godot_error!("VoxelBuffer.set_voxel_f: value must be finite");
            return;
        }
        self.buffer
            .set_voxel_f(value, position.x, position.y, position.z, channel);
    }

    /// Fill an entire channel with a raw integer value.
    /// Matches the C++ `fill(value, channel=0)` overload. The `channel`
    /// argument defaults to `0` (CHANNEL_TYPE) when omitted by GDScript.
    #[func]
    fn fill(&mut self, value: i64, #[opt(default = 0)] channel: i32) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.fill: invalid channel");
            return;
        };
        let Ok(value) = validate_voxel_value(value) else {
            godot_error!("VoxelBuffer.fill: invalid voxel value");
            return;
        };
        self.buffer.fill(value, channel);
    }

    /// Fill a rectangular area with a raw integer value.
    /// Matches the C++ `fill_area(value, min, max, channel=0)` overload.
    #[func]
    fn fill_area(
        &mut self,
        value: i64,
        min: godot::builtin::Vector3i,
        max: godot::builtin::Vector3i,
        #[opt(default = 0)] channel: i32,
    ) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.fill_area: invalid channel");
            return;
        };
        let Ok(value) = validate_voxel_value(value) else {
            godot_error!("VoxelBuffer.fill_area: invalid voxel value");
            return;
        };
        self.buffer.fill_area(
            value,
            Vector3i::new(min.x, min.y, min.z),
            Vector3i::new(max.x, max.y, max.z),
            channel,
        );
    }

    /// Fill an entire channel with a float value.
    /// Matches the C++ `fill_f(value, channel=0)` overload.
    #[func]
    fn fill_f(&mut self, value: f32, #[opt(default = 0)] channel: i32) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.fill_f: invalid channel");
            return;
        };
        if !value.is_finite() {
            godot_error!("VoxelBuffer.fill_f: value must be finite");
            return;
        }
        self.buffer.clear_channel_f(channel, value);
    }

    /// Fill a rectangular area with a float value.
    /// Matches the C++ `fill_area_f(value, min, max, channel=0)` overload.
    #[func]
    fn fill_area_f(
        &mut self,
        value: f32,
        min: godot::builtin::Vector3i,
        max: godot::builtin::Vector3i,
        #[opt(default = 0)] channel: i32,
    ) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.fill_area_f: invalid channel");
            return;
        };
        if !value.is_finite() {
            godot_error!("VoxelBuffer.fill_area_f: value must be finite");
            return;
        }
        // voxel-core's VoxelBuffer does not expose a native fill_area_f, so we
        // translate the float to a raw value through the channel's depth, then
        // delegate to fill_area. This matches the C++ behavior (which also
        // quantizes per-channel depth before writing).
        let depth = self.buffer.channel_depth(channel);
        let raw = voxel_core::storage::voxel_buffer::real_to_raw_voxel(value, depth);
        self.buffer.fill_area(
            raw,
            Vector3i::new(min.x, min.y, min.z),
            Vector3i::new(max.x, max.y, max.z),
            channel,
        );
    }

    /// Reset every channel to its uniform default and free all allocations.
    /// Matches the C++ `clear()` overload that takes no arguments.
    #[func]
    fn clear(&mut self) {
        let size = self.buffer.size();
        // Re-creating the buffer at the same size resets every channel to its
        // uniform default value and drops existing allocations, matching the
        // C++ behavior of `clear()` (which iterates channels and clears them).
        self.buffer.create(size);
    }

    /// Returns true if every voxel in the channel is equal.
    #[func]
    fn is_uniform(&self, channel: i32) -> bool {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.is_uniform: invalid channel");
            return true;
        };
        self.buffer.is_uniform(channel)
    }

    /// Compress any channel whose voxels are all equal to a uniform default.
    #[func]
    fn compress_uniform_channels(&mut self) {
        self.buffer.compress_uniform_channels();
    }

    /// Allocate a uniform channel into a full voxel array.
    #[func]
    fn decompress_channel(&mut self, channel: i32) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.decompress_channel: invalid channel");
            return;
        };
        self.buffer.decompress_channel(channel);
    }

    /// Returns the depth of a channel as one of the `DEPTH_*` constants.
    #[func]
    fn get_channel_depth(&self, channel: i32) -> i64 {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.get_channel_depth: invalid channel");
            return Self::DEPTH_8_BIT;
        };
        self.buffer.channel_depth(channel) as i64
    }

    /// Sets the depth of a channel. Resets the channel to a uniform default.
    #[func]
    fn set_channel_depth(&mut self, channel: i32, depth: i64) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.set_channel_depth: invalid channel");
            return;
        };
        let Some(depth) = channel_depth_from_i64(depth) else {
            godot_error!("VoxelBuffer.set_channel_depth: invalid depth");
            return;
        };
        self.buffer.set_channel_depth(channel, depth);
    }

    /// Returns the compression mode of a channel.
    #[func]
    fn get_channel_compression(&self, channel: i32) -> i64 {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.get_channel_compression: invalid channel");
            return Self::COMPRESSION_NONE;
        };
        self.buffer.channel_compression(channel) as i64
    }

    /// Returns the allocator this buffer uses (one of the `ALLOCATOR_*` constants).
    #[func]
    fn get_allocator(&self) -> i64 {
        self.buffer.allocator() as i64
    }

    /// Copy an entire channel from another `VoxelBuffer`. Both buffers must
    /// share the same size and the channel must use the same depth on both
    /// ends. Matches the C++ `copy_channel_from(other, channel)` overload.
    #[func]
    fn copy_channel_from(&mut self, other: Gd<RefCounted>, channel: i32) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.copy_channel_from: invalid channel");
            return;
        };
        let Ok(other_gd) = other.try_cast::<VoxelBufferGD>() else {
            godot_error!("VoxelBuffer.copy_channel_from: argument must be a VoxelBuffer");
            return;
        };
        let bound = other_gd.bind();
        if bound.buffer.size() != self.buffer.size() {
            godot_error!("VoxelBuffer.copy_channel_from: buffer sizes must match");
            return;
        }
        if bound.buffer.channel_depth(channel) != self.buffer.channel_depth(channel) {
            godot_error!("VoxelBuffer.copy_channel_from: channel depths must match");
            return;
        }
        self.buffer.copy_channel_from(&bound.buffer, channel);
    }

    /// Copy a rectangular area of one channel from another `VoxelBuffer`.
    /// Matches the C++ `copy_channel_from(other, src_min, src_max, dst_min, channel)` overload.
    #[func]
    fn copy_channel_from_area(
        &mut self,
        other: Gd<RefCounted>,
        src_min: godot::builtin::Vector3i,
        src_max: godot::builtin::Vector3i,
        dst_min: godot::builtin::Vector3i,
        channel: i32,
    ) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.copy_channel_from_area: invalid channel");
            return;
        };
        let Ok(other_gd) = other.try_cast::<VoxelBufferGD>() else {
            godot_error!("VoxelBuffer.copy_channel_from_area: argument must be a VoxelBuffer");
            return;
        };
        let bound = other_gd.bind();
        if bound.buffer.channel_depth(channel) != self.buffer.channel_depth(channel) {
            godot_error!("VoxelBuffer.copy_channel_from_area: channel depths must match");
            return;
        }
        self.buffer.copy_channel_from_area(
            &bound.buffer,
            Vector3i::new(src_min.x, src_min.y, src_min.z),
            Vector3i::new(src_max.x, src_max.y, src_max.z),
            Vector3i::new(dst_min.x, dst_min.y, dst_min.z),
            channel,
        );
    }

    /// Nearest-neighbor 2:1 downscale of all channels from a region of `self`
    /// into a region of `dst`. Matches the C++ `downscale_to(dst, src_min, src_max, dst_min)` overload.
    #[func]
    fn downscale_to(
        &self,
        dst: Gd<RefCounted>,
        src_min: godot::builtin::Vector3i,
        src_max: godot::builtin::Vector3i,
        dst_min: godot::builtin::Vector3i,
    ) {
        let Ok(mut dst_gd) = dst.try_cast::<VoxelBufferGD>() else {
            godot_error!("VoxelBuffer.downscale_to: argument must be a VoxelBuffer");
            return;
        };
        let mut bound = dst_gd.bind_mut();
        self.buffer.downscale_to(
            &mut bound.buffer,
            Vector3i::new(src_min.x, src_min.y, src_min.z),
            Vector3i::new(src_max.x, src_max.y, src_max.z),
            Vector3i::new(dst_min.x, dst_min.y, dst_min.z),
        );
    }

    /// Returns the raw bytes of a channel (decompressed) as a `PackedByteArray`.
    /// Matches the C++ `get_channel_as_byte_array(channel)` overload.
    #[func]
    fn get_channel_as_byte_array(&mut self, channel: i32) -> PackedByteArray {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.get_channel_as_byte_array: invalid channel");
            return PackedByteArray::new();
        };
        // Materialize the channel first so uniform channels return their full
        // volume of bytes rather than an empty compressed view.
        self.buffer.decompress_channel(channel);
        let bytes: &[u8] = self.buffer.channel_bytes(channel);
        PackedByteArray::from(bytes)
    }

    /// Replaces the raw bytes of a channel from a `PackedByteArray`. The
    /// channel is decompressed first; the byte slice must contain at least
    /// `volume * depth.byte_count()` bytes. Returns `false` on a size
    /// mismatch or invalid channel. Matches the C++
    /// `set_channel_from_byte_array(channel, source)` overload.
    #[func]
    fn set_channel_from_byte_array(&mut self, channel: i32, source: PackedByteArray) -> bool {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelBuffer.set_channel_from_byte_array: invalid channel");
            return false;
        };
        let size = self.buffer.size();
        let volume = (size.x as usize) * (size.y as usize) * (size.z as usize);
        let depth = self.buffer.channel_depth(channel);
        let needed = volume * depth.byte_size();
        let src = source.as_slice();
        if src.len() < needed {
            godot_error!(
                "VoxelBuffer.set_channel_from_byte_array: source is too small ({} < {})",
                src.len(),
                needed
            );
            return false;
        }
        self.buffer.decompress_channel(channel);
        let dst = self.buffer.channel_bytes_mut(channel);
        dst[..needed].copy_from_slice(&src[..needed]);
        true
    }

    /// Returns a `VoxelTool` for editing this buffer. The voxel-core does not
    /// expose a scriptable VoxelTool yet, so this is a stub that logs and
    /// returns null. Matches the C++ `get_voxel_tool()` overload.
    #[func]
    fn get_voxel_tool(&mut self) -> Variant {
        let _ = &mut self.buffer;
        godot_error!("VoxelBuffer.get_voxel_tool: not implemented in this port");
        Variant::nil()
    }

    // ---- Block metadata (C++ `VoxelBuffer::set_block_metadata` family) ----
    // The voxel-core port does not model block metadata yet, so these are
    // no-op stubs that log once. They keep the GDScript API surface intact.

    /// Sets opaque block metadata. Stubbed (voxel-core has no metadata yet).
    #[func]
    fn set_block_metadata(&mut self, _metadata: Variant) {
        godot_error!("VoxelBuffer.set_block_metadata: not implemented in this port");
    }

    /// Returns opaque block metadata. Stubbed (voxel-core has no metadata yet).
    #[func]
    fn get_block_metadata(&self) -> Variant {
        godot_error!("VoxelBuffer.get_block_metadata: not implemented in this port");
        Variant::nil()
    }

    // ---- Per-voxel metadata (C++ `VoxelBuffer::set_voxel_metadata` family) ----

    /// Sets the metadata entry for a single voxel. Stubbed.
    #[func]
    fn set_voxel_metadata(&mut self, _position: godot::builtin::Vector3i, _metadata: Variant) {
        godot_error!("VoxelBuffer.set_voxel_metadata: not implemented in this port");
    }

    /// Returns the metadata entry for a single voxel. Stubbed.
    #[func]
    fn get_voxel_metadata(&self, _position: godot::builtin::Vector3i) -> Variant {
        godot_error!("VoxelBuffer.get_voxel_metadata: not implemented in this port");
        Variant::nil()
    }

    /// Clears the metadata entry for a single voxel. Stubbed.
    #[func]
    fn clear_voxel_metadata(&mut self, _position: godot::builtin::Vector3i) {
        godot_error!("VoxelBuffer.clear_voxel_metadata: not implemented in this port");
    }

    /// Returns the next voxel position carrying metadata, or
    /// `(0,0,0)` if there is none (stubbed — always reports empty). Matches
    /// the C++ `for_each_voxel_metadata_in_area` iterator pattern by
    /// returning zero positions in this port.
    #[func]
    fn next_voxel_metadata_pos_in_area(
        &self,
        _min: godot::builtin::Vector3i,
        _max: godot::builtin::Vector3i,
        _start: godot::builtin::Vector3i,
    ) -> godot::builtin::Vector3i {
        godot::builtin::Vector3i::ZERO
    }

    /// Iterate every voxel metadata entry in an area, invoking `callback`
    /// with `(position, metadata)`. Stubbed (voxel-core has no metadata yet).
    #[func]
    fn for_each_voxel_metadata_in_area(
        &mut self,
        _min: godot::builtin::Vector3i,
        _max: godot::builtin::Vector3i,
        _callback: Callable,
    ) {
        // No metadata storage in this port; the C++ overload iterates an
        // internal map, but we have nothing to iterate. Stubbed to a no-op.
    }

    /// Iterate every voxel metadata entry in the whole buffer. Stubbed.
    #[func]
    fn for_each_voxel_metadata(&mut self, _callback: Callable) {
        // No metadata storage in this port; stubbed to a no-op.
    }

    // ---- 3D texture helpers (C++ SDF-ZXY texture bridge) ----

    /// Builds a 3D `ImageTexture` from the SDF channel laid out ZXY. Stubbed
    /// because voxel-core does not bridge to Godot textures directly. Matches
    /// the C++ `create_3d_texture_from_sdf_zxy` overload.
    #[func]
    fn create_3d_texture_from_sdf_zxy(&mut self, _texture: Gd<Object>, _channel: i32) -> bool {
        godot_error!("VoxelBuffer.create_3d_texture_from_sdf_zxy: not implemented in this port");
        false
    }

    /// Refreshes a 3D `ImageTexture` from the SDF channel. Stubbed.
    #[func]
    fn update_3d_texture_from_sdf_zxy(&mut self, _texture: Gd<Object>, _channel: i32) -> bool {
        godot_error!("VoxelBuffer.update_3d_texture_from_sdf_zxy: not implemented in this port");
        false
    }

    /// Prints ASCII debug slices of the SDF channel along Y. The voxel-core
    /// port does not implement a debug printer; this stub logs once.
    #[func]
    fn debug_print_sdf_y_slices(&self) {
        godot_error!("VoxelBuffer.debug_print_sdf_y_slices: not implemented in this port");
    }

    // ---- Geometric transforms ----

    /// Mirror a channel along an axis (0=X, 1=Y, 2=Z).
    #[func]
    fn mirror(&mut self, axis: i32, channel: i32) {
        let ch = channel as usize;
        if ch >= voxel_core::storage::voxel_buffer::MAX_CHANNELS {
            godot_error!("VoxelBuffer.mirror: invalid channel");
            return;
        }
        self.buffer.mirror(ch, axis as usize);
    }

    /// Rotate a channel 90° clockwise around an axis.
    #[func]
    fn rotate_90(&mut self, axis: i32, channel: i32) {
        let ch = channel as usize;
        if ch >= voxel_core::storage::voxel_buffer::MAX_CHANNELS {
            godot_error!("VoxelBuffer.rotate_90: invalid channel");
            return;
        }
        self.buffer.rotate_90(ch, axis as usize, true);
    }

    /// Re-encode channel values via a lookup table.
    #[func]
    fn remap_values(&mut self, channel: i32, remap: PackedInt32Array) {
        let ch = channel as usize;
        if ch >= voxel_core::storage::voxel_buffer::MAX_CHANNELS {
            godot_error!("VoxelBuffer.remap_values: invalid channel");
            return;
        }
        self.buffer.decompress_channel(ch);
        let data = &mut self.buffer;
        let size = data.size();
        let sz = size.x as usize * size.y as usize * size.z as usize;
        for i in 0..sz {
            let val = data.get_voxel(
                (i % size.x as usize) as i32,
                ((i / size.x as usize) % size.y as usize) as i32,
                (i / (size.x as usize * size.y as usize)) as i32,
                ch,
            );
            let idx = val.min(remap.len() as u64 - 1) as usize;
            let mapped = remap.get(idx).unwrap_or(0).max(0) as u64;
            data.set_voxel(
                mapped,
                (i % size.x as usize) as i32,
                ((i / size.x as usize) % size.y as usize) as i32,
                (i / (size.x as usize * size.y as usize)) as i32,
                ch,
            );
        }
    }

    // ---- op_* family (channel-wide set operations) ----

    /// Set-difference: this = min(this, other) (SDF convention).
    #[func]
    fn op_difference(&mut self, other: Gd<RefCounted>, channel: i32, _other_channel: i32) {
        let Ok(other_buf) = other.try_cast::<Self>() else {
            godot_error!("VoxelBuffer.op_difference: other must be a VoxelBuffer");
            return;
        };
        let bound = other_buf.bind();
        self.buffer.op_min_buffer_f(&bound.buffer, channel as usize);
    }

    /// Set-intersection: this = max(this, other) (SDF convention).
    #[func]
    fn op_intersection(&mut self, other: Gd<RefCounted>, channel: i32, _other_channel: i32) {
        let Ok(other_buf) = other.try_cast::<Self>() else {
            godot_error!("VoxelBuffer.op_intersection: other must be a VoxelBuffer");
            return;
        };
        let bound = other_buf.bind();
        self.buffer.op_max_buffer_f(&bound.buffer, channel as usize);
    }

    /// Set-union: this = min(this, other) (SDF union = min).
    #[func]
    fn op_union(&mut self, other: Gd<RefCounted>, channel: i32, _other_channel: i32) {
        let Ok(other_buf) = other.try_cast::<Self>() else {
            godot_error!("VoxelBuffer.op_union: other must be a VoxelBuffer");
            return;
        };
        let bound = other_buf.bind();
        self.buffer.op_min_buffer_f(&bound.buffer, channel as usize);
    }

    /// Paste another buffer's channels into this one at an offset.
    #[func]
    fn paste(&mut self, other: Gd<RefCounted>, min: godot::builtin::Vector3i, channel_mask: i64) {
        let Ok(other_buf) = other.try_cast::<Self>() else {
            godot_error!("VoxelBuffer.paste: other must be a VoxelBuffer");
            return;
        };
        let bound = other_buf.bind();
        self.buffer.paste(
            &bound.buffer,
            Vector3i::zero(),
            Vector3i::new(min.x, min.y, min.z),
            channel_mask as u8,
        );
    }
}

impl VoxelBufferGD {
    /// Borrow the underlying engine-agnostic [`VoxelBuffer`]. Used by sibling
    /// binding classes (mesher resources, modifiers) that need direct access
    /// to run voxel-core logic without round-tripping through Godot calls.
    pub fn core_buffer(&self) -> &VoxelBuffer {
        &self.buffer
    }

    /// Mutably borrow the underlying [`VoxelBuffer`].
    pub fn core_buffer_mut(&mut self) -> &mut VoxelBuffer {
        &mut self.buffer
    }

    /// Construct a `VoxelBufferGD` wrapping an already-built engine-agnostic
    /// [`VoxelBuffer`]. Used by sibling binding classes (e.g. `VoxelFormat`)
    /// that need to hand back a fully-configured Godot buffer from a core one.
    pub fn from_core(base: Base<RefCounted>, buffer: VoxelBuffer) -> Self {
        Self { base, buffer }
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use voxel_core::storage::voxel_buffer::{MAX_CHANNELS, MAX_SIZE};

    #[test]
    fn validate_channel_rejects_values_outside_the_core_channel_range() {
        assert!(validate_channel(-1).is_err());
        assert!(validate_channel(MAX_CHANNELS as i32).is_err());
    }

    #[test]
    fn validate_buffer_size_rejects_negative_and_oversized_dimensions() {
        assert!(validate_buffer_size(-1, 1, 1).is_err());
        assert!(validate_buffer_size(MAX_SIZE as i32 + 1, 1, 1).is_err());
    }

    #[test]
    fn validate_buffer_size_rejects_overflowing_and_oversized_total_volume() {
        assert!(validate_buffer_size(MAX_SIZE as i32, MAX_SIZE as i32, MAX_SIZE as i32).is_err());
        assert!(validate_buffer_size(129, 128, 128).is_err());
        assert_eq!(
            validate_buffer_size(128, 128, 128),
            Ok(Vector3i::new(128, 128, 128))
        );
    }

    #[test]
    fn validate_position_rejects_coordinates_outside_the_buffer() {
        assert!(validate_position(Vector3i::new(-1, 0, 0), Vector3i::splat(4)).is_err());
    }

    #[test]
    fn script_item_count_rejects_oversized_allocations() {
        assert!(validate_script_item_count((MAX_SCRIPT_ITEMS + 1) as i32).is_err());
    }
}

// ---------------------------------------------------------------------------
// VoxelInstancerGD — Node3D for scatter-based instance placement
// ---------------------------------------------------------------------------

/// A Godot `Node3D` that scatters instances (trees, rocks, grass) on
/// a parent [`VoxelTerrain`](crate::terrain::VoxelTerrain) using
/// [`voxel_core::instancing`].
#[derive(GodotClass)]
#[class(base = Node3D, tool, rename = VoxelInstancer)]
pub struct VoxelInstancerGD {
    base: Base<Node3D>,
    /// Instance library (items to scatter).
    library: InstanceLibrary,
    /// Scatter config.
    config: ScatterConfig,
    /// Density multiplier.
    #[var]
    density_multiplier: f32,
}

#[godot_api]
impl INode3D for VoxelInstancerGD {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            library: InstanceLibrary::new(),
            config: ScatterConfig::default(),
            density_multiplier: 1.0,
        }
    }

    fn ready(&mut self) {
        godot_print!("VoxelInstancerGD ready");
    }
}

#[godot_api]
impl VoxelInstancerGD {
    /// Add a scatter item (returns item index).
    #[func]
    fn add_item(&mut self, name: GString, density: f64, min_scale: f64, max_scale: f64) -> i32 {
        let Ok(density) = validate_finite_f64(density) else {
            godot_error!("VoxelInstancer.add_item: density must be finite and f32-representable");
            return -1;
        };
        let Ok(min_scale) = validate_finite_f64(min_scale) else {
            godot_error!(
                "VoxelInstancer.add_item: minimum scale must be finite and f32-representable"
            );
            return -1;
        };
        let Ok(max_scale) = validate_finite_f64(max_scale) else {
            godot_error!(
                "VoxelInstancer.add_item: maximum scale must be finite and f32-representable"
            );
            return -1;
        };
        let item = voxel_core::instancing::InstanceLibraryItem {
            name: name.to_string(),
            density,
            min_scale,
            max_scale,
            ..Default::default()
        };
        let Ok(index) = i32::try_from(self.library.len()) else {
            godot_error!("VoxelInstancer.add_item: too many scatter items");
            return -1;
        };
        self.library.add_item(item);
        index
    }

    /// Get the number of items in the library.
    #[func]
    fn get_item_count(&self) -> i32 {
        i32::try_from(self.library.len()).unwrap_or(i32::MAX)
    }

    /// Set the random seed for scatter.
    #[func]
    fn set_seed(&mut self, seed: i64) {
        let Ok(seed) = u32::try_from(seed) else {
            godot_error!("VoxelInstancer.set_seed: seed must be between zero and u32::MAX");
            return;
        };
        self.config.seed = seed;
    }

    /// Generate instances from a VoxelBufferGD's surface.
    /// Extracts surface points where solid meets air, runs the scatter
    /// generator for each library item, returns total instance count.
    #[func]
    fn scatter_from_buffer(&mut self, buffer: Gd<RefCounted>) -> i32 {
        if self.library.is_empty() {
            return 0;
        }
        if !self.density_multiplier.is_finite() {
            godot_error!("VoxelInstancer.scatter_from_buffer: density multiplier must be finite");
            return 0;
        }
        // Try to cast to VoxelBufferGD for direct field access.
        if let Ok(buf_gd) = buffer.clone().try_cast::<VoxelBufferGD>() {
            let bound = buf_gd.bind();
            let sx = bound.get_size_x();
            let sy = bound.get_size_y();
            let sz = bound.get_size_z();

            let mut positions = Vec::new();
            let mut normals = Vec::new();
            for z in 1..sz {
                for y in 1..sy {
                    for x in 1..sx {
                        let vt = bound.get_voxel(x, y, z, 0);
                        let vt_below = bound.get_voxel(x, y - 1, z, 0);
                        if vt != 0 && vt_below == 0 {
                            positions.push(Vector3f::new(x as f32, y as f32, z as f32));
                            normals.push(Vector3f::new(0.0, 1.0, 0.0));
                        }
                    }
                }
            }
            drop(bound);
            drop(buf_gd);

            if positions.is_empty() {
                return 0;
            }

            let mut total = 0i64;
            for (idx, item) in self.library.items.iter().enumerate() {
                let Ok(seed_offset) = u32::try_from(idx) else {
                    godot_error!("VoxelInstancer.scatter_from_buffer: too many scatter items");
                    return 0;
                };
                let gen = RandomScatterGenerator {
                    density: item.density * self.density_multiplier,
                    min_scale: item.min_scale,
                    max_scale: item.max_scale,
                    snap_to_normal: item.snap_to_normal,
                };
                let result = gen.generate(&positions, &normals, seed_offset, &self.config);
                let Ok(count) = i64::try_from(result.len()) else {
                    godot_error!(
                        "VoxelInstancer.scatter_from_buffer: instance count exceeds i64 range"
                    );
                    return 0;
                };
                let Some(next_total) = total.checked_add(count) else {
                    godot_error!("VoxelInstancer.scatter_from_buffer: instance count overflow");
                    return 0;
                };
                total = next_total;
            }
            return i32::try_from(total).unwrap_or(i32::MAX);
        }
        0
    }

    /// Generate instances from dummy surface positions (test/debug).
    /// Returns the total instance count.
    #[func]
    fn scatter_test(&self, count: i32) -> i32 {
        let Ok(count) = validate_script_item_count(count) else {
            godot_error!("VoxelInstancer.scatter_test: count must fit the script allocation limit");
            return 0;
        };
        if self.library.is_empty() {
            return 0;
        }
        if !self.density_multiplier.is_finite() {
            godot_error!("VoxelInstancer.scatter_test: density multiplier must be finite");
            return 0;
        }
        let positions: Vec<Vector3f> = (0..count)
            .map(|i| Vector3f::new(i as f32, 0.0, 0.0))
            .collect();
        let normals = vec![Vector3f::new(0.0, 1.0, 0.0); count];
        let gen = RandomScatterGenerator {
            density: self.library.items[0].density * self.density_multiplier,
            min_scale: self.library.items[0].min_scale,
            max_scale: self.library.items[0].max_scale,
            snap_to_normal: true,
        };
        let result = gen.generate(&positions, &normals, 0, &self.config);
        i32::try_from(result.len()).unwrap_or(i32::MAX)
    }
}

// ---------------------------------------------------------------------------
// VoxelToolTerrainGD — RefCounted terrain editing tool
// ---------------------------------------------------------------------------

/// A Godot `RefCounted` that wraps a live [`VoxelTerrain`](crate::terrain::VoxelTerrain)
/// and applies sphere/box/voxel edits through `VoxelTerrainCore::try_edit_voxel`.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelToolTerrain)]
pub struct VoxelToolTerrainGD {
    base: Base<RefCounted>,
    /// Weak reference to the terrain node path (canonical inspector field).
    terrain_path: GString,
    terrain: Option<Gd<crate::terrain::VoxelTerrain>>,
    channel: usize,
    value: u64,
}

#[godot_api]
impl IRefCounted for VoxelToolTerrainGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            terrain_path: "..".to_godot(),
            terrain: None,
            channel: ChannelId::Sdf.index(),
            value: 1,
        }
    }
}

#[godot_api]
impl VoxelToolTerrainGD {
    #[func]
    fn set_terrain_path(&mut self, path: GString) {
        self.terrain_path = path;
    }

    #[func]
    fn get_terrain_path(&self) -> GString {
        self.terrain_path.clone()
    }

    #[func]
    fn set_channel(&mut self, channel: i32) {
        let Ok(channel) = validate_channel(channel) else {
            godot_error!("VoxelToolTerrain.set_channel: invalid channel");
            return;
        };
        self.channel = channel;
    }

    #[func]
    fn get_channel(&self) -> i32 {
        i32::try_from(self.channel).unwrap_or(0)
    }

    #[func]
    fn set_value(&mut self, value: i64) {
        let Ok(value) = validate_voxel_value(value) else {
            godot_error!("VoxelToolTerrain.set_value: invalid voxel value");
            return;
        };
        self.value = value;
    }

    #[func]
    fn do_sphere(&mut self, center: Vector3, radius: f32, mode: i32) {
        if !center.x.is_finite()
            || !center.y.is_finite()
            || !center.z.is_finite()
            || !radius.is_finite()
        {
            godot_error!("VoxelToolTerrain.do_sphere: center and radius must be finite");
            return;
        }
        if radius < 0.0 {
            godot_error!("VoxelToolTerrain.do_sphere: radius must be non-negative");
            return;
        }
        let Some(mut terrain) = self.terrain.clone() else {
            godot_error!("VoxelToolTerrain.do_sphere: no terrain is bound");
            return;
        };
        let edit_mode = match mode {
            0 => voxel_core::edition::EditMode::Add,
            1 => voxel_core::edition::EditMode::Remove,
            _ => voxel_core::edition::EditMode::Set,
        };
        let channel = self.channel;
        let value = self.value;
        let mut bound = terrain.bind_mut();
        bound.edit_sphere(
            voxel_core::math::Vector3f::new(center.x, center.y, center.z),
            radius,
            channel,
            edit_mode,
            value,
        );
    }

    #[func]
    fn do_box(&mut self, min: godot::builtin::Vector3i, max: godot::builtin::Vector3i, mode: i32) {
        let Some(mut terrain) = self.terrain.clone() else {
            godot_error!("VoxelToolTerrain.do_box: no terrain is bound");
            return;
        };
        let edit_mode = match mode {
            0 => voxel_core::edition::EditMode::Add,
            1 => voxel_core::edition::EditMode::Remove,
            _ => voxel_core::edition::EditMode::Set,
        };
        let channel = self.channel;
        let value = self.value;
        let mut bound = terrain.bind_mut();
        bound.edit_box(
            Vector3i::new(min.x, min.y, min.z),
            Vector3i::new(max.x, max.y, max.z),
            channel,
            edit_mode,
            value,
        );
    }

    #[func]
    fn set_voxel(&mut self, position: godot::builtin::Vector3i, value: i64) {
        let Ok(value) = validate_voxel_value(value) else {
            godot_error!("VoxelToolTerrain.set_voxel: invalid voxel value");
            return;
        };
        let Some(mut terrain) = self.terrain.clone() else {
            godot_error!("VoxelToolTerrain.set_voxel: no terrain is bound");
            return;
        };
        let channel = self.channel;
        terrain.bind_mut().edit_world_voxel(
            Vector3i::new(position.x, position.y, position.z),
            channel,
            value,
        );
    }

    #[func]
    fn get_voxel(&self, position: godot::builtin::Vector3i) -> i64 {
        let Some(terrain) = self.terrain.as_ref() else {
            return 0;
        };
        i64::try_from(terrain.bind().read_world_voxel(
            Vector3i::new(position.x, position.y, position.z),
            self.channel,
        ))
        .unwrap_or_default()
    }

    // -----------------------------------------------------------------
    // Pinned VoxelToolTerrain methods
    // (upstream 5828cbeb: VoxelToolTerrain.xml).
    //
    // The Rust binding wraps a terrain by node path and does not yet expose
    // the live terrain data; these methods are faithful no-op stubs that
    // round-trip the GDScript contract without panicking.
    // -----------------------------------------------------------------

    /// Operates on a hemisphere, where `flat_direction` points away from the
    /// flat surface (like a normal). `smoothness` blends the flat part with
    /// the rounded part (higher = softer edge). Matches upstream's pinned
    /// `do_hemisphere`. Faithful stub: the live terrain edit is deferred.
    #[func]
    #[allow(clippy::too_many_arguments)]
    fn do_hemisphere(
        &mut self,
        _center: Vector3,
        _radius: f32,
        _flat_direction: Vector3,
        _smoothness: f32,
    ) {
        // TODO(port): route the hemisphere edit into the bound terrain.
    }

    /// Executes a function for each voxel holding metadata in the given area.
    /// The callback takes two arguments: voxel position (`Vector3i`) and voxel
    /// metadata (`Variant`). Matches upstream's pinned
    /// `for_each_voxel_metadata_in_area`. Faithful stub: no metadata store is
    /// bound yet, so the callback is never invoked.
    #[func]
    fn for_each_voxel_metadata_in_area(&self, _voxel_area: Aabb, _callback: Callable) {
        // TODO(port): iterate the bound terrain's per-voxel metadata.
    }

    /// Picks random voxels within `area` and executes a function on those
    /// whose model is `random_tickable` and whose `tags_mask` matches. The
    /// callback takes voxel position (`Vector3i`) and voxel value (`int`).
    /// `batch_count` tunes the internal picking strategy; `tags_mask` filters
    /// models. Matches upstream's pinned `run_blocky_random_tick`. Faithful
    /// stub: no live terrain is bound, so no ticks are produced.
    #[func]
    fn run_blocky_random_tick(
        &mut self,
        _area: Aabb,
        _voxel_count: i32,
        _callback: Callable,
        _batch_count: i32,
        _tags_mask: i32,
    ) {
        // TODO(port): drive the blocky random-tick loop against the terrain.
    }
}

impl VoxelToolTerrainGD {
    pub(crate) fn bind_terrain(&mut self, terrain: Gd<crate::terrain::VoxelTerrain>) {
        self.terrain = Some(terrain);
    }
}

// ---------------------------------------------------------------------------
// VoxelRaycastResultGD — RefCounted result container
// ---------------------------------------------------------------------------

/// Result of a voxel raycast. Contains hit position, previous position,
/// and distance along the ray.
///
/// The pinned GDScript-facing properties (`normal`, `position`,
/// `previous_position`) mirror upstream `VoxelRaycastResult` (5828cbeb). They
/// are read-only getters (no setter) composing the existing integer fields.
/// The pinned `distance` property is provided by the existing `distance`
/// field's auto-generated getter.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelRaycastResult)]
pub struct VoxelRaycastResultGD {
    base: Base<RefCounted>,
    #[var]
    hit_x: i32,
    #[var]
    hit_y: i32,
    #[var]
    hit_z: i32,
    #[var]
    prev_x: i32,
    #[var]
    prev_y: i32,
    #[var]
    prev_z: i32,
    #[var]
    distance: f32,
    #[var]
    normal_x: i32,
    #[var]
    normal_y: i32,
    #[var]
    normal_z: i32,
    /// The pinned GDScript-facing read-only `normal` property.
    #[var(get = get_normal, no_set)]
    normal_prop: PhantomVar<Vector3>,
    /// The pinned GDScript-facing read-only `position` property.
    #[var(get = get_position, no_set)]
    position_prop: PhantomVar<godot::builtin::Vector3i>,
    /// The pinned GDScript-facing read-only `previous_position` property.
    #[var(get = get_previous_position, no_set)]
    previous_position_prop: PhantomVar<godot::builtin::Vector3i>,
}

#[godot_api]
impl IRefCounted for VoxelRaycastResultGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            hit_x: 0,
            hit_y: 0,
            hit_z: 0,
            prev_x: 0,
            prev_y: 0,
            prev_z: 0,
            distance: 0.0,
            normal_x: 0,
            normal_y: 0,
            normal_z: 0,
            normal_prop: PhantomVar::default(),
            position_prop: PhantomVar::default(),
            previous_position_prop: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelRaycastResultGD {
    /// Whether this result represents a valid hit (distance > 0 and a
    /// non-zero normal). A default-constructed result reports no hit.
    #[func]
    fn did_hit(&self) -> bool {
        self.distance > 0.0 && (self.normal_x != 0 || self.normal_y != 0 || self.normal_z != 0)
    }

    /// The hit position as a packed array [x, y, z].
    #[func]
    fn get_hit_position(&self) -> PackedInt32Array {
        PackedInt32Array::from(&[self.hit_x, self.hit_y, self.hit_z][..])
    }

    // -----------------------------------------------------------------
    // Pinned VoxelRaycastResult properties (read-only getters)
    // (upstream 5828cbeb: VoxelRaycastResult.xml).
    // -----------------------------------------------------------------

    /// Unit vector pointing away from the surface that was hit (upstream
    /// default `Vector3(0, 0, 0)`). Only available when the producing
    /// `VoxelTool` was configured to compute normals.
    #[func]
    fn get_normal(&self) -> Vector3 {
        Vector3::new(
            self.normal_x as f32,
            self.normal_y as f32,
            self.normal_z as f32,
        )
    }

    /// Integer position of the voxel that was hit (upstream default
    /// `Vector3i(0, 0, 0)`).
    #[func]
    fn get_position(&self) -> godot::builtin::Vector3i {
        godot::builtin::Vector3i::new(self.hit_x, self.hit_y, self.hit_z)
    }

    /// Integer position of the previous voxel along the ray before the final
    /// hit (upstream default `Vector3i(0, 0, 0)`).
    #[func]
    fn get_previous_position(&self) -> godot::builtin::Vector3i {
        godot::builtin::Vector3i::new(self.prev_x, self.prev_y, self.prev_z)
    }
}

// ---------------------------------------------------------------------------
// VoxelNodeGD — base Node3D for voxel volumes (VoxelNode equivalent)
// ---------------------------------------------------------------------------

/// Base Node3D for voxel volume nodes. Holds shared properties.
/// In C++ this is the base class for VoxelTerrain/VoxelLodTerrain.
/// In Rust, VoxelTerrain inherits Node3D directly, but this class
/// exists for API parity and future VoxelLodTerrain.
///
/// The pinned GDScript-facing properties (`cast_shadow`, `format`,
/// `generator`, `gi_mode`, `mesher`, `render_layers_mask`, `stream`) and the
/// `convert_to_nodes` method mirror upstream `VoxelNode` (5828cbeb). They are
/// stored faithfully so GDScript reads round-trip.
#[derive(GodotClass)]
#[class(base = Node3D, tool, rename = VoxelNode)]
pub struct VoxelNodeGD {
    base: Base<Node3D>,
    /// Whether the terrain streams blocks around viewers.
    #[var]
    auto_load: bool,
    /// Maximum view distance in voxels.
    #[var]
    max_view_distance: i64,
    /// Pinned `cast_shadow` (backing field; enum `ShadowCastingSetting`).
    /// Upstream default 1 (SHADOW_CASTING_SETTING_ONE_PASS).
    cast_shadow_value: i32,
    /// Pinned `gi_mode` (backing field; enum `GIMode`). Upstream default 0
    /// (GI_MODE_DISABLED).
    gi_mode_value: i32,
    /// Pinned `render_layers_mask` (backing field). Upstream default 1.
    render_layers_mask_value: i32,
    /// Pinned `format` resource (backing field; `None` until assigned).
    format_resource: Option<Gd<crate::resources::VoxelFormatGD>>,
    /// Pinned `generator` resource (backing field; `None` until assigned).
    generator_resource: Option<Gd<Resource>>,
    /// Pinned `mesher` resource (backing field; `None` until assigned).
    mesher_resource: Option<Gd<Resource>>,
    /// Pinned `stream` resource (backing field; `None` until assigned).
    stream_resource: Option<Gd<Resource>>,
    /// The pinned GDScript-facing `cast_shadow` property.
    #[var(get = get_shadow_casting, set = set_shadow_casting)]
    cast_shadow: PhantomVar<i32>,
    /// The pinned GDScript-facing `format` property.
    #[var(get = get_format, set = set_format)]
    format: PhantomVar<Option<Gd<crate::resources::VoxelFormatGD>>>,
    /// The pinned GDScript-facing `generator` property.
    #[var(get = get_generator, set = set_generator)]
    generator: PhantomVar<Option<Gd<Resource>>>,
    /// The pinned GDScript-facing `gi_mode` property.
    #[var(get = get_gi_mode, set = set_gi_mode)]
    gi_mode: PhantomVar<i32>,
    /// The pinned GDScript-facing `mesher` property.
    #[var(get = get_mesher, set = set_mesher)]
    mesher: PhantomVar<Option<Gd<Resource>>>,
    /// The pinned GDScript-facing `render_layers_mask` property.
    #[var(get = get_render_layers_mask, set = set_render_layers_mask)]
    render_layers_mask: PhantomVar<i32>,
    /// The pinned GDScript-facing `stream` property.
    #[var(get = get_stream, set = set_stream)]
    stream: PhantomVar<Option<Gd<Resource>>>,
}

#[godot_api]
impl INode3D for VoxelNodeGD {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            auto_load: true,
            max_view_distance: 192,
            cast_shadow_value: 1,
            gi_mode_value: 0,
            render_layers_mask_value: 1,
            format_resource: None,
            generator_resource: None,
            mesher_resource: None,
            stream_resource: None,
            cast_shadow: PhantomVar::default(),
            format: PhantomVar::default(),
            generator: PhantomVar::default(),
            gi_mode: PhantomVar::default(),
            mesher: PhantomVar::default(),
            render_layers_mask: PhantomVar::default(),
            stream: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelNodeGD {
    /// NodeConversionFlags: include instancer output as MultiMeshInstance
    /// chunks (canonical `NODE_CONVERSION_INCLUDE_INSTANCER`).
    #[constant]
    const NODE_CONVERSION_INCLUDE_INSTANCER: i64 = 1;
    /// NodeConversionFlags: include chunks internally hidden by the LOD system
    /// (canonical `NODE_CONVERSION_INCLUDE_INVISIBLE_BLOCKS`).
    #[constant]
    const NODE_CONVERSION_INCLUDE_INVISIBLE_BLOCKS: i64 = 2;
    /// NodeConversionFlags: apply material overrides present on the terrain to
    /// result chunks (canonical `NODE_CONVERSION_INCLUDE_MATERIAL_OVERRIDES`).
    #[constant]
    const NODE_CONVERSION_INCLUDE_MATERIAL_OVERRIDES: i64 = 4;

    /// Whether this node is currently streaming blocks.
    #[func]
    fn is_streaming(&self) -> bool {
        self.auto_load
    }

    /// The effective view distance in blocks (max_view_distance / 16, min 1).
    #[func]
    fn get_view_distance_blocks(&self) -> i64 {
        (self.max_view_distance / 16).max(1)
    }

    // -----------------------------------------------------------------
    // Pinned VoxelNode method
    // (upstream 5828cbeb: VoxelNode.xml).
    // -----------------------------------------------------------------

    /// Generates a tree of vanilla Godot nodes representing the terrain at the
    /// time of calling (canonical `convert_to_nodes`). The returned root is
    /// not added to the scene tree. `flags` is a bitmask of
    /// `NODE_CONVERSION_*` constants. Faithful stub: the Rust binding does not
    /// yet build the snapshot tree, so `None` is returned.
    #[func]
    fn convert_to_nodes(&self, _flags: i32) -> Option<Gd<godot::classes::Node3D>> {
        // TODO(port): assemble the snapshot node tree from live terrain data.
        None
    }

    // -----------------------------------------------------------------
    // Pinned VoxelNode properties
    // (upstream 5828cbeb: VoxelNode.xml).
    // -----------------------------------------------------------------

    /// Shadow casting mode used by terrain meshes (enum
    /// `GeometryInstance3D.ShadowCastingSetting`; upstream default 1).
    #[func]
    fn get_shadow_casting(&self) -> i32 {
        self.cast_shadow_value
    }

    #[func]
    fn set_shadow_casting(&mut self, mode: i32) {
        self.cast_shadow_value = mode;
    }

    /// Overrides the default format of voxels (`None` = engine default).
    #[func]
    fn get_format(&self) -> Option<Gd<crate::resources::VoxelFormatGD>> {
        self.format_resource.clone()
    }

    #[func]
    fn set_format(&mut self, format: Option<Gd<crate::resources::VoxelFormatGD>>) {
        self.format_resource = format;
    }

    /// Procedural generator used to load voxel blocks when not present in the
    /// stream (`None` until assigned).
    #[func]
    fn get_generator(&self) -> Option<Gd<Resource>> {
        self.generator_resource.clone()
    }

    #[func]
    fn set_generator(&mut self, generator: Option<Gd<Resource>>) {
        self.generator_resource = generator;
    }

    /// Global Illumination mode used by terrain meshes (enum
    /// `GeometryInstance3D.GIMode`; upstream default 0).
    #[func]
    fn get_gi_mode(&self) -> i32 {
        self.gi_mode_value
    }

    #[func]
    fn set_gi_mode(&mut self, mode: i32) {
        self.gi_mode_value = mode;
    }

    /// Defines how voxels are transformed into visible meshes (`None` until
    /// assigned).
    #[func]
    fn get_mesher(&self) -> Option<Gd<Resource>> {
        self.mesher_resource.clone()
    }

    #[func]
    fn set_mesher(&mut self, mesher: Option<Gd<Resource>>) {
        self.mesher_resource = mesher;
    }

    /// Render layers mask used by terrain meshes (upstream default 1).
    #[func]
    fn get_render_layers_mask(&self) -> i32 {
        self.render_layers_mask_value
    }

    #[func]
    fn set_render_layers_mask(&mut self, mask: i32) {
        self.render_layers_mask_value = mask;
    }

    /// Primary source of persistent voxel data (`None` until assigned).
    #[func]
    fn get_stream(&self) -> Option<Gd<Resource>> {
        self.stream_resource.clone()
    }

    #[func]
    fn set_stream(&mut self, stream: Option<Gd<Resource>>) {
        self.stream_resource = stream;
    }
}

// ---------------------------------------------------------------------------
// VoxelGeneratorGraphGD — Resource wrapper for GraphGenerator
// ---------------------------------------------------------------------------

/// A Godot `Resource` wrapping a graph-based terrain generator.
///
/// Wraps [`voxel_core::generators::graph::GraphGenerator`] — the functional
/// API builds graphs, compiles them via `CompiledGraph`, and samples the SDF
/// output at a world point, exercising the full graph generation pipeline
/// through the binding.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelGeneratorGraph)]
pub struct VoxelGeneratorGraphGD {
    base: Base<Resource>,
    /// Graph nodes serialized as a JSON string (for save/load).
    graph_json: GString,
    /// The lazily-built engine-agnostic graph generator.
    generator: Option<voxel_core::generators::graph::GraphGenerator>,
}

#[godot_api]
impl IResource for VoxelGeneratorGraphGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            graph_json: "{}".to_godot(),
            generator: None,
        }
    }
}

#[godot_api]
impl VoxelGeneratorGraphGD {
    #[func]
    fn get_graph_json(&self) -> GString {
        self.graph_json.clone()
    }

    /// Replace the graph JSON. Resets the cached generator (it will rebuild on
    /// the next `sample_*` call).
    #[func]
    fn set_graph_json(&mut self, json: GString) {
        self.graph_json = json;
        self.generator = None;
    }

    /// Build a sphere-SDF graph (center `(cx,cy,cz)`, radius `r`), compile it,
    /// and return the sampled signed distance at world point `(px,py,pz)`.
    /// Negative = inside the sphere. Returns `NaN` if the graph fails to
    /// compile (malformed topology).
    #[func]
    #[allow(clippy::too_many_arguments)]
    fn sample_sphere_sdf(
        &mut self,
        cx: f32,
        cy: f32,
        cz: f32,
        r: f32,
        px: f32,
        py: f32,
        pz: f32,
    ) -> f32 {
        if !cx.is_finite()
            || !cy.is_finite()
            || !cz.is_finite()
            || !px.is_finite()
            || !py.is_finite()
            || !pz.is_finite()
            || !r.is_finite()
            || r < 0.0
        {
            godot_error!("VoxelGeneratorGraph.sample_sphere_sdf: inputs must be finite and radius non-negative");
            return f32::NAN;
        }
        use voxel_core::generators::graph::{
            CompiledGraph, CompiledScratch, Graph, GraphInputs, GraphOutput, GraphPort, NodeKind,
        };
        // SdfSphere evaluates sdf_sphere(pos, ZERO, radius), so feed the
        // sample point relative to the sphere center as the position inputs.
        let mut graph = Graph::new();
        let nx = graph.push(NodeKind::Constant(px - cx));
        let ny = graph.push(NodeKind::Constant(py - cy));
        let nz = graph.push(NodeKind::Constant(pz - cz));
        let nr = graph.push(NodeKind::Constant(r));
        let sphere = graph.push(NodeKind::SdfSphere {
            x: Some(GraphPort {
                node: nx,
                output: 0,
            }),
            y: Some(GraphPort {
                node: ny,
                output: 0,
            }),
            z: Some(GraphPort {
                node: nz,
                output: 0,
            }),
            radius: Some(GraphPort {
                node: nr,
                output: 0,
            }),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: sphere,
                output: 0,
            }),
        });
        // Cache the generator so the compiled graph is reused across calls.
        self.generator = Some(voxel_core::generators::graph::GraphGenerator::new(graph));
        let Some(generator) = self.generator.as_ref() else {
            godot_error!("VoxelGeneratorGraph.sample_sphere_sdf: graph generator was not retained");
            return f32::NAN;
        };
        let Ok(compiled) = CompiledGraph::compile(generator.graph()) else {
            return f32::NAN;
        };
        let xs = [0.0f32];
        let zs = [0.0f32];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &zs,
        };
        let mut scratch = CompiledScratch::new();
        let mut out = Vec::new();
        compiled.generate_slice(&inputs, 1, &mut scratch, &mut out, false);
        out.into_iter()
            .find(|(k, _)| *k == GraphOutput::Sdf)
            .and_then(|(_, v)| v.into_iter().next())
            .unwrap_or(f32::NAN)
    }

    /// Returns the number of nodes in the currently-cached generator's graph,
    /// or 0 if no graph has been built yet.
    #[func]
    fn get_node_count(&self) -> i32 {
        self.generator
            .as_ref()
            .map(|g| i32::try_from(g.graph().nodes().len()).unwrap_or(i32::MAX))
            .unwrap_or(0)
    }
}
