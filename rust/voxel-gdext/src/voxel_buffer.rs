//! Additional Godot classes for voxel-core types.
//!
//! `VoxelBufferGD` exposes a VoxelBuffer as a Godot RefCounted.
//! `VoxelInstancerGD` is a Node3D for scatter-based instance placement.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use godot::classes::multi_mesh::TransformFormat;
use godot::classes::{BoxMesh, MultiMesh, MultiMeshInstance3D, Node3D, PackedScene};
use godot::prelude::*;
use voxel_core::instancing::scatter::{InstanceGenerator, RandomScatterGenerator};
use voxel_core::instancing::{BlockInstanceData, InstanceLibrary, InstanceMeshType, ScatterConfig};
use voxel_core::math::{Vector3f, Vector3i};
use voxel_core::storage::{ChannelId, MetadataValue, VoxelBuffer, VoxelFormat};

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

pub(crate) fn metadata_from_variant(value: &Variant) -> Result<MetadataValue, &'static str> {
    if value.is_nil() {
        return Ok(MetadataValue::Nil);
    }
    if let Ok(v) = value.try_to::<i64>() {
        return Ok(MetadataValue::Int(v));
    }
    if let Ok(v) = value.try_to::<f64>() {
        return Ok(MetadataValue::Float(v));
    }
    if let Ok(v) = value.try_to::<GString>() {
        return Ok(MetadataValue::Text(v.to_string()));
    }
    if let Ok(v) = value.try_to::<PackedByteArray>() {
        return Ok(MetadataValue::Bytes(v.as_slice().to_vec()));
    }
    // Wide (R7) types: bool, vectors, rect/plane/quaternion/AABB/color,
    // Dictionary, Array, and packed arrays map onto the Variant arm.
    if let Some(wide) = variant_wire_from_godot(value) {
        return Ok(MetadataValue::Variant(wide));
    }
    Err("metadata must be nil, int, float, String, PackedByteArray, bool, Vector2/3/4, Rect2, Plane, Quaternion, AABB, Color, Dictionary, Array, or a packed array")
}

pub(crate) fn metadata_to_variant(value: &MetadataValue) -> Variant {
    match value {
        MetadataValue::Nil => Variant::nil(),
        MetadataValue::Int(v) => v.to_variant(),
        MetadataValue::Float(v) => v.to_variant(),
        MetadataValue::Text(v) => GString::from(v.as_str()).to_variant(),
        MetadataValue::Bytes(v) => PackedByteArray::from(v.as_slice()).to_variant(),
        MetadataValue::Variant(v) => variant_wire_to_godot(v),
    }
}

fn variant_wire_to_godot(value: &voxel_core::streams::variant_wire::VariantWireValue) -> Variant {
    use godot::builtin::{Aabb as GAabb, Plane as GPlane, Rect2};
    use voxel_core::streams::variant_wire::VariantWireValue as V;
    match value {
        V::Nil => Variant::nil(),
        V::Bool(b) => b.to_variant(),
        V::Int(i) => i.to_variant(),
        V::Float(f) => f.to_variant(),
        V::Text(s) => GString::from(s.as_str()).to_variant(),
        V::Vector2(v) => godot::builtin::Vector2::new(v[0] as f32, v[1] as f32).to_variant(),
        V::Vector2i(v) => godot::builtin::Vector2i::new(v[0], v[1]).to_variant(),
        V::Rect2(v) => {
            Rect2::from_components(v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32).to_variant()
        }
        V::Rect2i(v) => {
            godot::builtin::Rect2i::from_components(v[0], v[1], v[2], v[3]).to_variant()
        }
        V::Vector3(v) => {
            godot::builtin::Vector3::new(v[0] as f32, v[1] as f32, v[2] as f32).to_variant()
        }
        V::Vector3i(v) => godot::builtin::Vector3i::new(v[0], v[1], v[2]).to_variant(),
        V::Vector4(v) => {
            godot::builtin::Vector4::new(v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32)
                .to_variant()
        }
        V::Vector4i(v) => godot::builtin::Vector4i::new(v[0], v[1], v[2], v[3]).to_variant(),
        V::Plane(v) => GPlane::new(
            godot::builtin::Vector3::new(v[0] as f32, v[1] as f32, v[2] as f32),
            v[3] as f32,
        )
        .to_variant(),
        V::Quaternion(v) => {
            godot::builtin::Quaternion::new(v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32)
                .to_variant()
        }
        V::Aabb(v) => GAabb::new(
            godot::builtin::Vector3::new(v[0] as f32, v[1] as f32, v[2] as f32),
            godot::builtin::Vector3::new(v[3] as f32, v[4] as f32, v[5] as f32),
        )
        .to_variant(),
        V::Color(v) => godot::builtin::Color::from_rgba(v[0], v[1], v[2], v[3]).to_variant(),
        V::Array(items) => {
            let mut arr = VarArray::new();
            for item in items {
                arr.push(&variant_wire_to_godot(item));
            }
            arr.to_variant()
        }
        V::Dictionary(pairs) => {
            let mut dict = VarDictionary::new();
            for (key, value) in pairs {
                let key = variant_wire_to_godot(key);
                let value = variant_wire_to_godot(value);
                dict.set(&key, &value);
            }
            dict.to_variant()
        }
        V::ByteArray(bytes) => PackedByteArray::from(bytes.as_slice()).to_variant(),
        V::Int32Array(items) => PackedInt32Array::from(items.as_slice()).to_variant(),
        V::Int64Array(items) => PackedInt64Array::from(items.as_slice()).to_variant(),
        V::Float32Array(items) => PackedFloat32Array::from(items.as_slice()).to_variant(),
        V::Float64Array(items) => PackedFloat64Array::from(items.as_slice()).to_variant(),
        V::StringArray(items) => {
            PackedStringArray::from_iter(items.iter().map(|s| GString::from(s.as_str())))
                .to_variant()
        }
        V::Vector2Array(items) => PackedVector2Array::from_iter(
            items
                .iter()
                .map(|v| godot::builtin::Vector2::new(v[0] as f32, v[1] as f32)),
        )
        .to_variant(),
        V::Vector3Array(items) => PackedVector3Array::from_iter(
            items
                .iter()
                .map(|v| godot::builtin::Vector3::new(v[0] as f32, v[1] as f32, v[2] as f32)),
        )
        .to_variant(),
        V::ColorArray(items) => PackedColorArray::from_iter(
            items
                .iter()
                .map(|c| godot::builtin::Color::from_rgba(c[0], c[1], c[2], c[3])),
        )
        .to_variant(),
    }
}

/// Convert a Godot Variant into the wide wire representation. Returns None
/// for types outside the supported metadata subset.
fn variant_wire_from_godot(
    value: &Variant,
) -> Option<voxel_core::streams::variant_wire::VariantWireValue> {
    variant_wire_from_godot_depth(value, 0)
}

/// Max nesting accepted from GDScript — matches the wire decoder's
/// DecodeLimits::max_variant_depth default.
const VARIANT_CONVERSION_MAX_DEPTH: u32 = 64;

fn variant_wire_from_godot_depth(
    value: &Variant,
    depth: u32,
) -> Option<voxel_core::streams::variant_wire::VariantWireValue> {
    if depth > VARIANT_CONVERSION_MAX_DEPTH {
        return None; // cyclic or absurdly nested — reject whole-value
    }
    use voxel_core::streams::variant_wire::VariantWireValue as V;
    // Scalars must map too: the canonical C++ metadata dictionary is
    // string keys with int/float/string/null values.
    if value.is_nil() {
        return Some(V::Nil);
    }
    if let Ok(v) = value.try_to::<i64>() {
        return Some(V::Int(v));
    }
    if let Ok(v) = value.try_to::<f64>() {
        return Some(V::Float(v));
    }
    if let Ok(v) = value.try_to::<GString>() {
        return Some(V::Text(v.to_string()));
    }
    if let Ok(b) = value.try_to::<bool>() {
        return Some(V::Bool(b));
    }
    if let Ok(v) = value.try_to::<godot::builtin::Vector2>() {
        return Some(V::Vector2([v.x as f64, v.y as f64]));
    }
    if let Ok(v) = value.try_to::<godot::builtin::Vector2i>() {
        return Some(V::Vector2i([v.x, v.y]));
    }
    if let Ok(v) = value.try_to::<godot::builtin::Rect2>() {
        return Some(V::Rect2([
            v.position.x as f64,
            v.position.y as f64,
            v.size.x as f64,
            v.size.y as f64,
        ]));
    }
    if let Ok(v) = value.try_to::<godot::builtin::Rect2i>() {
        return Some(V::Rect2i([v.position.x, v.position.y, v.size.x, v.size.y]));
    }
    if let Ok(v) = value.try_to::<godot::builtin::Vector3>() {
        return Some(V::Vector3([v.x as f64, v.y as f64, v.z as f64]));
    }
    if let Ok(v) = value.try_to::<godot::builtin::Vector3i>() {
        return Some(V::Vector3i([v.x, v.y, v.z]));
    }
    if let Ok(v) = value.try_to::<godot::builtin::Vector4>() {
        return Some(V::Vector4([v.x as f64, v.y as f64, v.z as f64, v.w as f64]));
    }
    if let Ok(v) = value.try_to::<godot::builtin::Vector4i>() {
        return Some(V::Vector4i([v.x, v.y, v.z, v.w]));
    }
    if let Ok(v) = value.try_to::<godot::builtin::Plane>() {
        return Some(V::Plane([
            v.normal.x as f64,
            v.normal.y as f64,
            v.normal.z as f64,
            v.d as f64,
        ]));
    }
    if let Ok(v) = value.try_to::<godot::builtin::Quaternion>() {
        return Some(V::Quaternion([
            v.x as f64, v.y as f64, v.z as f64, v.w as f64,
        ]));
    }
    if let Ok(v) = value.try_to::<godot::builtin::Aabb>() {
        return Some(V::Aabb([
            v.position.x as f64,
            v.position.y as f64,
            v.position.z as f64,
            v.size.x as f64,
            v.size.y as f64,
            v.size.z as f64,
        ]));
    }
    if let Ok(v) = value.try_to::<godot::builtin::Color>() {
        return Some(V::Color([v.r, v.g, v.b, v.a]));
    }
    if let Ok(dict) = value.try_to::<VarDictionary>() {
        let mut pairs = Vec::new();
        for (key, value) in dict.iter_shared() {
            // Nested keys: wide types only; unsupported key types reject the
            // whole conversion (None) rather than half-converting.
            let key = variant_wire_from_godot_depth(&key, depth + 1)?;
            let value = variant_wire_from_godot_depth(&value, depth + 1)?;
            pairs.push((key, value));
        }
        return Some(V::Dictionary(pairs));
    }
    if let Ok(arr) = value.try_to::<VarArray>() {
        let mut items = Vec::new();
        for item in arr.iter_shared() {
            items.push(variant_wire_from_godot_depth(&item, depth + 1)?);
        }
        return Some(V::Array(items));
    }
    if let Ok(v) = value.try_to::<PackedInt32Array>() {
        return Some(V::Int32Array(v.as_slice().to_vec()));
    }
    if let Ok(v) = value.try_to::<PackedInt64Array>() {
        return Some(V::Int64Array(v.as_slice().to_vec()));
    }
    if let Ok(v) = value.try_to::<PackedFloat32Array>() {
        return Some(V::Float32Array(v.as_slice().to_vec()));
    }
    if let Ok(v) = value.try_to::<PackedFloat64Array>() {
        return Some(V::Float64Array(v.as_slice().to_vec()));
    }
    if let Ok(v) = value.try_to::<PackedStringArray>() {
        return Some(V::StringArray(
            v.as_slice().iter().map(|s| s.to_string()).collect(),
        ));
    }
    if let Ok(v) = value.try_to::<PackedVector2Array>() {
        return Some(V::Vector2Array(
            v.as_slice()
                .iter()
                .map(|v| [v.x as f64, v.y as f64])
                .collect(),
        ));
    }
    if let Ok(v) = value.try_to::<PackedVector3Array>() {
        return Some(V::Vector3Array(
            v.as_slice()
                .iter()
                .map(|v| [v.x as f64, v.y as f64, v.z as f64])
                .collect(),
        ));
    }
    if let Ok(v) = value.try_to::<PackedColorArray>() {
        return Some(V::ColorArray(
            v.as_slice().iter().map(|c| [c.r, c.g, c.b, c.a]).collect(),
        ));
    }
    None
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
    // Persisted through the v4 block serializer metadata section (R7 narrow).
    // Supported payloads: nil/int/float/String/PackedByteArray. C++ Variant
    // (Dictionary/Object) payloads need the wide R7 codec.

    /// Sets opaque block metadata on this buffer.
    #[func]
    fn set_block_metadata(&mut self, metadata: Variant) {
        match metadata_from_variant(&metadata) {
            Ok(value) => self.buffer.set_block_metadata(value),
            Err(error) => godot_error!("VoxelBuffer.set_block_metadata: {error}"),
        }
    }

    /// Returns opaque block metadata, or `nil` when none is set.
    #[func]
    fn get_block_metadata(&self) -> Variant {
        metadata_to_variant(self.buffer.block_metadata())
    }

    // ---- Per-voxel metadata (C++ `VoxelBuffer::set_voxel_metadata` family) ----

    /// Sets the metadata entry for a single voxel. `nil` clears the entry.
    #[func]
    fn set_voxel_metadata(&mut self, position: godot::builtin::Vector3i, metadata: Variant) {
        let pos = Vector3i::new(position.x, position.y, position.z);
        let Ok(pos) = validate_position(pos, self.buffer.size()) else {
            godot_error!("VoxelBuffer.set_voxel_metadata: position is outside the buffer");
            return;
        };
        match metadata_from_variant(&metadata) {
            Ok(value) => self.buffer.set_voxel_metadata(pos, value),
            Err(error) => godot_error!("VoxelBuffer.set_voxel_metadata: {error}"),
        }
    }

    /// Returns the metadata entry for a single voxel, or `nil` when none.
    #[func]
    fn get_voxel_metadata(&self, position: godot::builtin::Vector3i) -> Variant {
        let pos = Vector3i::new(position.x, position.y, position.z);
        match self.buffer.voxel_metadata(pos) {
            Some(value) => metadata_to_variant(value),
            None => Variant::nil(),
        }
    }

    /// Clears the metadata entry for a single voxel.
    #[func]
    fn clear_voxel_metadata(&mut self, position: godot::builtin::Vector3i) {
        let size = self.buffer.size();
        let pos = Vector3i::new(position.x, position.y, position.z);
        if validate_position(pos, size).is_err() {
            godot_error!("VoxelBuffer.clear_voxel_metadata: position is outside the buffer");
            return;
        }
        self.buffer.clear_voxel_metadata(pos);
    }

    /// Returns the next voxel position carrying metadata in `[min, max)`
    /// after `start` (ZXY order), or `(0,0,0)` if there is none. Pass
    /// `Vector3i(INT32_MIN, …)` to obtain the first entry, including one at
    /// the origin.
    #[func]
    fn next_voxel_metadata_pos_in_area(
        &self,
        min: godot::builtin::Vector3i,
        max: godot::builtin::Vector3i,
        start: godot::builtin::Vector3i,
    ) -> godot::builtin::Vector3i {
        match self.buffer.next_voxel_metadata_pos_in_area(
            Vector3i::new(min.x, min.y, min.z),
            Vector3i::new(max.x, max.y, max.z),
            Vector3i::new(start.x, start.y, start.z),
        ) {
            Some(pos) => godot::builtin::Vector3i::new(pos.x, pos.y, pos.z),
            None => godot::builtin::Vector3i::ZERO,
        }
    }

    /// Iterate every voxel metadata entry in `[min, max)`, invoking `callback`
    /// with `(position, metadata)`.
    #[func]
    fn for_each_voxel_metadata_in_area(
        &self,
        min: godot::builtin::Vector3i,
        max: godot::builtin::Vector3i,
        callback: Callable,
    ) {
        if !callback.is_valid() {
            godot_error!("VoxelBuffer.for_each_voxel_metadata_in_area: callback is invalid");
            return;
        }
        self.buffer.for_each_voxel_metadata_in_area(
            Vector3i::new(min.x, min.y, min.z),
            Vector3i::new(max.x, max.y, max.z),
            |pos, value| {
                let gpos = godot::builtin::Vector3i::new(pos.x, pos.y, pos.z);
                callback.call(&[gpos.to_variant(), metadata_to_variant(value)]);
            },
        );
    }

    /// Iterate every voxel metadata entry in the whole buffer.
    #[func]
    fn for_each_voxel_metadata(&self, callback: Callable) {
        if !callback.is_valid() {
            godot_error!("VoxelBuffer.for_each_voxel_metadata: callback is invalid");
            return;
        }
        self.buffer.for_each_voxel_metadata(|pos, value| {
            let gpos = godot::builtin::Vector3i::new(pos.x, pos.y, pos.z);
            callback.call(&[gpos.to_variant(), metadata_to_variant(value)]);
        });
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
    use std::f32::consts::FRAC_1_SQRT_2;
    use voxel_core::instancing::InstanceLibraryItem;
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

    #[test]
    fn instance_transform_applies_rotation_scale_and_position() {
        // Identity rotation, scale 2, at (1, 2, 3).
        let identity = BlockInstanceData {
            position: Vector3f::new(1.0, 2.0, 3.0),
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: 2.0,
            item_index: 0,
        };
        let t = instance_transform(&identity);
        assert_eq!(t.origin, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(t.basis.col_a(), Vector3::new(2.0, 0.0, 0.0));
        assert_eq!(t.basis.col_b(), Vector3::new(0.0, 2.0, 0.0));
        assert_eq!(t.basis.col_c(), Vector3::new(0.0, 0.0, 2.0));

        // 90-degree yaw around +Y: quaternion (0, sqrt(2)/2, 0, sqrt(2)/2).
        let yawed = BlockInstanceData {
            position: Vector3f::zero(),
            rotation: [0.0, FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2],
            scale: 1.0,
            item_index: 0,
        };
        let t = instance_transform(&yawed);
        let close = |a: Vector3, b: Vector3| {
            (a.x - b.x).abs() < 1e-6 && (a.y - b.y).abs() < 1e-6 && (a.z - b.z).abs() < 1e-6
        };
        assert!(close(t.basis.col_a(), Vector3::new(0.0, 0.0, -1.0)));
        assert!(close(t.basis.col_c(), Vector3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn item_is_scene_routes_only_scene_typed_items() {
        let mut library = InstanceLibrary::new();
        library.add_item(InstanceLibraryItem::default());
        library.add_item(InstanceLibraryItem {
            mesh_type: InstanceMeshType::Scene,
            ..Default::default()
        });
        assert!(!item_is_scene(&library, 0));
        assert!(item_is_scene(&library, 1));
        assert!(!item_is_scene(&library, 2), "out of range is never a scene");
    }

    #[test]
    fn bucket_instances_by_item_splits_and_drops_out_of_range() {
        let make = |item_index: u32| BlockInstanceData {
            position: Vector3f::new(item_index as f32, 0.0, 0.0),
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: 1.0,
            item_index,
        };
        let buckets = bucket_instances_by_item(&[make(0), make(1), make(0), make(9)], 2);
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0].len(), 2);
        assert_eq!(buckets[1].len(), 1);
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
    /// Optional per-item mesh used when uploading MultiMeshes.
    item_meshes: Vec<Option<Gd<godot::classes::Mesh>>>,
    /// Optional per-item scene; items with one are scene-typed and spawn
    /// real `Node3D`s instead of a MultiMesh (R5 scene instancing).
    item_scenes: Vec<Option<Gd<PackedScene>>>,
    /// Live MultiMeshInstance3D children created by the last one-shot scatter.
    uploaded_instances: Vec<Gd<MultiMeshInstance3D>>,
    /// Live scene-instance `Node3D` children created by the last one-shot
    /// scatter.
    uploaded_scene_nodes: Vec<Gd<Node3D>>,
    /// Resident per-block instance data, streamed with terrain paging.
    instance_blocks: voxel_core::instancing::InstanceBlockMap,
    /// Nodes keyed by streamed block `(x, y, z, lod)`.
    streamed_nodes: HashMap<(i32, i32, i32, u8), Vec<StreamedInstanceNode>>,
    /// Density multiplier.
    #[var]
    density_multiplier: f32,
    /// Whether the no-surface-points warning has fired (one-shot).
    warned_no_surface: bool,
    /// Last parent `mesh_revision` seen. When it advances, terrain edits
    /// remeshed existing blocks (same keys), so every instance block is
    /// dropped and rescattered — otherwise instances would float over dug
    /// ground or stay buried under raised terrain.
    mesh_revision_seen: i64,
}

/// A node spawned for one streamed instance block: one `MultiMeshInstance3D`
/// per MultiMesh-typed item, or one `Node3D` per instance for scene-typed
/// items.
enum StreamedInstanceNode {
    MultiMesh(Gd<MultiMeshInstance3D>),
    Scene(Gd<Node3D>),
}

#[godot_api]
impl INode3D for VoxelInstancerGD {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            library: InstanceLibrary::new(),
            config: ScatterConfig::default(),
            item_meshes: Vec::new(),
            item_scenes: Vec::new(),
            uploaded_instances: Vec::new(),
            uploaded_scene_nodes: Vec::new(),
            instance_blocks: voxel_core::instancing::InstanceBlockMap::new(),
            streamed_nodes: HashMap::new(),
            density_multiplier: 1.0,
            warned_no_surface: false,
            mesh_revision_seen: -1,
        }
    }

    fn ready(&mut self) {
        godot_print!("VoxelInstancerGD ready");
        self.base_mut().set_process(true);
    }

    fn process(&mut self, _delta: f64) {
        // Editor hint: no streaming work unless the scene is running (the
        // parent terrain is likewise gated by `run_stream_in_editor`).
        if godot::classes::Engine::singleton().is_editor_hint() {
            return;
        }
        let _ = self.sync_stream();
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
        if density > 1.0 {
            godot_warn!(
                "VoxelInstancer.add_item: density is a probability per surface cell in 0..=1 (upstream uses per-square-meter density); clamping {density}"
            );
        }
        let item = voxel_core::instancing::InstanceLibraryItem {
            name: name.to_string(),
            density: density.min(1.0),
            min_scale,
            max_scale,
            ..Default::default()
        };
        let Ok(index) = i32::try_from(self.library.len()) else {
            godot_error!("VoxelInstancer.add_item: too many scatter items");
            return -1;
        };
        self.library.add_item(item);
        self.item_meshes.push(None);
        self.item_scenes.push(None);
        index
    }

    /// Assign the mesh used when this item is uploaded as a MultiMesh.
    /// Assigning a mesh switches the item back to the MultiMesh type and
    /// clears any previously assigned scene.
    #[func]
    fn set_item_mesh(&mut self, index: i32, mesh: Gd<godot::classes::Mesh>) {
        let Ok(index) = usize::try_from(index) else {
            godot_error!("VoxelInstancer.set_item_mesh: index must be non-negative");
            return;
        };
        if index >= self.item_meshes.len() {
            godot_error!("VoxelInstancer.set_item_mesh: index is out of range");
            return;
        }
        self.item_meshes[index] = Some(mesh);
        self.item_scenes[index] = None;
        if let Some(item) = self.library.items.get_mut(index) {
            item.mesh_type = InstanceMeshType::MultiMesh;
        }
    }

    /// Assign the scene instantiated per instance of this item. Items with a
    /// scene are scene-typed: every scattered instance spawns a real `Node3D`
    /// (the scene's root, placed at the instance transform) instead of being
    /// packed into a MultiMesh. Assigning a scene clears any previously
    /// assigned mesh. Rejected (with one error) when the scene cannot be
    /// instantiated or its root is not a `Node3D`.
    #[func]
    fn set_item_scene(&mut self, index: i32, scene: Gd<PackedScene>) {
        let Ok(index) = usize::try_from(index) else {
            godot_error!("VoxelInstancer.set_item_scene: index must be non-negative");
            return;
        };
        if index >= self.item_scenes.len() {
            godot_error!("VoxelInstancer.set_item_scene: index is out of range");
            return;
        }
        if !scene_root_is_node3d(&scene) {
            godot_error!(
                "VoxelInstancer.set_item_scene: scene root must be a Node3D; assignment rejected"
            );
            return;
        }
        self.item_scenes[index] = Some(scene);
        self.item_meshes[index] = None;
        if let Some(item) = self.library.items.get_mut(index) {
            item.mesh_type = InstanceMeshType::Scene;
        }
    }

    /// The scene assigned to an item, or `null` when the item has no scene
    /// (MultiMesh-typed items never have one — assigning a mesh clears it).
    #[func]
    fn get_item_scene(&self, index: i32) -> Option<Gd<PackedScene>> {
        let Ok(index) = usize::try_from(index) else {
            godot_error!("VoxelInstancer.get_item_scene: index must be non-negative");
            return None;
        };
        let scene = self.item_scenes.get(index).cloned().flatten();
        if scene.is_none() && index >= self.item_scenes.len() {
            godot_error!("VoxelInstancer.get_item_scene: index is out of range");
        }
        scene
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
    /// The result replaces this node's children: MultiMesh items upload one
    /// `MultiMeshInstance3D` each, scene items spawn one `Node3D` per
    /// instance.
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
            let (positions, normals) = voxel_core::instancing::extract_surface_points(
                bound.core_buffer(),
                Vector3f::zero(),
                0,
            );
            drop(bound);
            drop(buf_gd);

            if positions.is_empty() {
                // The contract says the result replaces this node's children:
                // an emptied surface must clear the previous scatter, not
                // keep it rendering.
                self.clear_uploads();
                return 0;
            }

            let mut total = 0i64;
            let mut by_item: Vec<Vec<voxel_core::instancing::scatter::BlockInstanceData>> =
                Vec::new();
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
                by_item.push(result);
            }
            self.upload_scatter_instances(&by_item);
            return i32::try_from(total).unwrap_or(i32::MAX);
        }
        godot_error!("VoxelInstancer.scatter_from_buffer: argument must be a VoxelBuffer");
        0
    }

    /// Generate instances from dummy surface positions (test/debug).
    /// Returns the total instance count.
    #[func]
    fn scatter_test(&mut self, count: i32) -> i32 {
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
        self.upload_scatter_instances(std::slice::from_ref(&result));
        i32::try_from(result.len()).unwrap_or(i32::MAX)
    }

    /// Stream instances for every currently paged mesh block of a parent
    /// `VoxelTerrain` / `VoxelLodTerrain`. Called from `_process`; also
    /// available as a one-shot from GDScript.
    #[func]
    fn sync_stream(&mut self) -> i32 {
        if self.library.is_empty() || !self.density_multiplier.is_finite() {
            return i32::try_from(self.instance_blocks.instance_count()).unwrap_or(i32::MAX);
        }
        let Some(locations) = self.parent_mesh_locations() else {
            return i32::try_from(self.instance_blocks.instance_count()).unwrap_or(i32::MAX);
        };
        // Rescatter on terrain edits: remeshes keep the same mesh-block
        // keys, so the wanted-set diff below alone would never notice.
        let revision = self.parent_mesh_revision();
        if revision != self.mesh_revision_seen {
            if self.mesh_revision_seen >= 0 && !self.instance_blocks.is_empty() {
                let stale: Vec<(Vector3i, u8)> = self.instance_blocks.keys().collect();
                for (pos, lod) in stale {
                    self.unload_instance_block(pos, lod);
                }
            }
            self.mesh_revision_seen = revision;
        }
        let block_size = self.parent_block_size();
        let wanted: HashSet<(i32, i32, i32, u8)> = locations.iter().copied().collect();
        // Deterministic order for both lists: HashMap iteration would leak
        // random ordering into the scene-tree child order.
        let mut stale: Vec<(Vector3i, u8)> = self
            .instance_blocks
            .keys()
            .filter(|(pos, lod)| !wanted.contains(&(pos.x, pos.y, pos.z, *lod)))
            .collect();
        stale.sort_unstable_by_key(|(pos, lod)| (pos.x, pos.y, pos.z, *lod));
        for (pos, lod) in stale {
            self.unload_instance_block(pos, lod);
        }
        let mut ordered: Vec<(i32, i32, i32, u8)> = wanted.into_iter().collect();
        ordered.sort_unstable();
        for (x, y, z, lod) in ordered {
            if lod != 0 {
                continue;
            }
            let pos = Vector3i::new(x, y, z);
            if self.instance_blocks.contains(pos, lod) {
                continue;
            }
            self.load_instance_block(pos, lod, block_size);
        }
        i32::try_from(self.instance_blocks.instance_count()).unwrap_or(i32::MAX)
    }

    /// Number of resident streamed instance blocks.
    #[func]
    fn get_streamed_block_count(&self) -> i32 {
        i32::try_from(self.instance_blocks.len()).unwrap_or(i32::MAX)
    }

    /// Number of instances across all streamed blocks.
    #[func]
    fn get_streamed_instance_count(&self) -> i32 {
        i32::try_from(self.instance_blocks.instance_count()).unwrap_or(i32::MAX)
    }
}

impl VoxelInstancerGD {
    /// Free the children created by the last one-shot scatter. Nodes are
    /// removed from the tree before `queue_free` so same-frame re-scatters
    /// never present two generations as siblings (Godot would otherwise
    /// auto-mangle the new children's names for the rest of the frame).
    fn clear_uploads(&mut self) {
        let mut instances = std::mem::take(&mut self.uploaded_instances);
        let mut scenes = std::mem::take(&mut self.uploaded_scene_nodes);
        for mut old in instances.drain(..) {
            self.base_mut().remove_child(&old);
            old.queue_free();
        }
        for mut old in scenes.drain(..) {
            self.base_mut().remove_child(&old);
            old.queue_free();
        }
    }

    fn upload_scatter_instances(&mut self, by_item: &[Vec<BlockInstanceData>]) {
        self.clear_uploads();
        let default_mesh = BoxMesh::new_gd().upcast::<godot::classes::Mesh>();
        for (item_index, instances) in by_item.iter().enumerate() {
            if instances.is_empty() {
                continue;
            }
            if self.item_is_scene(item_index) {
                let Some(scene) = self.item_scenes.get(item_index).and_then(|s| s.clone()) else {
                    continue;
                };
                for node in instantiate_scene_nodes(
                    &scene,
                    &format!("scatter_scene_{item_index}"),
                    instances,
                ) {
                    self.base_mut().add_child(&node);
                    self.uploaded_scene_nodes.push(node);
                }
                continue;
            }
            let mesh = self
                .item_meshes
                .get(item_index)
                .and_then(|mesh| mesh.clone())
                .unwrap_or_else(|| default_mesh.clone());
            let mut multimesh = MultiMesh::new_gd();
            multimesh.set_transform_format(TransformFormat::TRANSFORM_3D);
            let Ok(count) = i32::try_from(instances.len()) else {
                continue;
            };
            multimesh.set_mesh(&mesh);
            multimesh.set_instance_count(count);
            for (i, instance) in instances.iter().enumerate() {
                let Ok(index) = i32::try_from(i) else {
                    break;
                };
                multimesh.set_instance_transform(index, instance_transform(instance));
            }
            let mut node = MultiMeshInstance3D::new_alloc();
            node.set_multimesh(&multimesh);
            node.set_name(&format!("scatter_item_{item_index}"));
            self.base_mut().add_child(&node);
            self.uploaded_instances.push(node);
        }
    }

    fn parent_mesh_locations(&self) -> Option<Vec<(i32, i32, i32, u8)>> {
        let parent = self.base().get_parent()?;
        if let Ok(terrain) = parent.clone().try_cast::<crate::terrain::VoxelTerrain>() {
            return Some(unpack_mesh_locations(
                terrain.bind().get_mesh_block_locations(),
            ));
        }
        if let Ok(terrain) = parent.try_cast::<crate::resources2::VoxelLodTerrainGD>() {
            return Some(unpack_mesh_locations(
                terrain.bind().get_mesh_block_locations(),
            ));
        }
        None
    }

    fn parent_mesh_revision(&self) -> i64 {
        let Some(parent) = self.base().get_parent() else {
            return 0;
        };
        if let Ok(terrain) = parent.clone().try_cast::<crate::terrain::VoxelTerrain>() {
            return terrain.bind().get_mesh_revision();
        }
        if let Ok(terrain) = parent.try_cast::<crate::resources2::VoxelLodTerrainGD>() {
            return terrain.bind().get_mesh_revision();
        }
        0
    }

    fn parent_block_size(&self) -> i32 {
        let Some(parent) = self.base().get_parent() else {
            return 16;
        };
        if let Ok(terrain) = parent.clone().try_cast::<crate::terrain::VoxelTerrain>() {
            return terrain.bind().get_mesh_block_size().max(1);
        }
        if let Ok(terrain) = parent.try_cast::<crate::resources2::VoxelLodTerrainGD>() {
            return terrain.bind().get_mesh_block_size().max(1);
        }
        16
    }

    fn parent_surface_points(
        &self,
        position: Vector3i,
        block_size: i32,
    ) -> (Vec<Vector3f>, Vec<Vector3f>) {
        let Some(parent) = self.base().get_parent() else {
            return (Vec::new(), Vec::new());
        };
        if let Ok(terrain) = parent.clone().try_cast::<crate::terrain::VoxelTerrain>() {
            return terrain
                .bind()
                .surface_points_for_block(position, block_size);
        }
        if let Ok(terrain) = parent.try_cast::<crate::resources2::VoxelLodTerrainGD>() {
            return terrain
                .bind()
                .surface_points_for_block(position, block_size);
        }
        (Vec::new(), Vec::new())
    }

    fn load_instance_block(&mut self, position: Vector3i, lod_index: u8, block_size: i32) {
        let (positions, normals) = self.parent_surface_points(position, block_size);
        if positions.is_empty() && !self.warned_no_surface {
            self.warned_no_surface = true;
            godot_warn!(
                "VoxelInstancer: a mesh block produced zero surface points. Surface extraction \
                 reads the TYPE channel and looks for air cells resting on solid ground; \
                 SDF-only terrain (nothing in the TYPE channel) yields no instances."
            );
        }
        let instances = voxel_core::instancing::scatter_block_instances(
            &self.library,
            &self.config,
            self.density_multiplier,
            &positions,
            &normals,
        );
        let by_item = bucket_instances_by_item(&instances, self.library.len());
        let nodes = self.upload_block_instances(position, lod_index, &by_item);
        self.streamed_nodes
            .insert((position.x, position.y, position.z, lod_index), nodes);
        self.instance_blocks
            .upsert(voxel_core::instancing::InstanceBlock::new(
                position, lod_index, instances,
            ));
    }

    fn unload_instance_block(&mut self, position: Vector3i, lod_index: u8) {
        self.instance_blocks.remove(position, lod_index);
        if let Some(nodes) = self
            .streamed_nodes
            .remove(&(position.x, position.y, position.z, lod_index))
        {
            for node in nodes {
                match node {
                    StreamedInstanceNode::MultiMesh(mut old) => {
                        self.base_mut().remove_child(&old);
                        old.queue_free();
                    }
                    StreamedInstanceNode::Scene(mut old) => {
                        self.base_mut().remove_child(&old);
                        old.queue_free();
                    }
                }
            }
        }
    }

    fn item_is_scene(&self, item_index: usize) -> bool {
        item_is_scene(&self.library, item_index)
    }

    fn upload_block_instances(
        &mut self,
        position: Vector3i,
        lod_index: u8,
        by_item: &[Vec<BlockInstanceData>],
    ) -> Vec<StreamedInstanceNode> {
        let default_mesh = BoxMesh::new_gd().upcast::<godot::classes::Mesh>();
        let mut nodes = Vec::new();
        for (item_index, instances) in by_item.iter().enumerate() {
            if instances.is_empty() {
                continue;
            }
            if self.item_is_scene(item_index) {
                let Some(scene) = self.item_scenes.get(item_index).and_then(|s| s.clone()) else {
                    // A scene-typed item without a scene has nothing to
                    // instantiate; skip it like upstream skips empty items.
                    continue;
                };
                let prefix = format!(
                    "scene_lod{lod_index}_{}_{}_{}_{item_index}",
                    position.x, position.y, position.z
                );
                for node in instantiate_scene_nodes(&scene, &prefix, instances) {
                    self.base_mut().add_child(&node);
                    nodes.push(StreamedInstanceNode::Scene(node));
                }
                continue;
            }
            let mesh = self
                .item_meshes
                .get(item_index)
                .and_then(|mesh| mesh.clone())
                .unwrap_or_else(|| default_mesh.clone());
            let mut multimesh = MultiMesh::new_gd();
            multimesh.set_transform_format(TransformFormat::TRANSFORM_3D);
            let Ok(count) = i32::try_from(instances.len()) else {
                continue;
            };
            multimesh.set_mesh(&mesh);
            multimesh.set_instance_count(count);
            for (i, instance) in instances.iter().enumerate() {
                let Ok(index) = i32::try_from(i) else {
                    break;
                };
                multimesh.set_instance_transform(index, instance_transform(instance));
            }
            let mut node = MultiMeshInstance3D::new_alloc();
            node.set_multimesh(&multimesh);
            node.set_name(&format!(
                "inst_lod{lod_index}_{}_{}_{}_{item_index}",
                position.x, position.y, position.z
            ));
            self.base_mut().add_child(&node);
            nodes.push(StreamedInstanceNode::MultiMesh(node));
        }
        nodes
    }
}

/// Whether the library item at `item_index` is scene-typed. Out-of-range
/// indices are never scene items.
fn item_is_scene(library: &InstanceLibrary, item_index: usize) -> bool {
    library
        .items
        .get(item_index)
        .is_some_and(|item| matches!(item.mesh_type, InstanceMeshType::Scene))
}

/// Split scattered instances into one bucket per library item, dropping
/// instances whose item index is out of range.
fn bucket_instances_by_item(
    instances: &[BlockInstanceData],
    library_len: usize,
) -> Vec<Vec<BlockInstanceData>> {
    let mut by_item = vec![Vec::new(); library_len];
    for instance in instances {
        if let Some(slot) = by_item.get_mut(instance.item_index as usize) {
            slot.push(*instance);
        }
    }
    by_item
}

/// Whether `scene` can back a scene-typed item: it instantiates and its root
/// is a `Node3D`. The probe instance is never parented, so it is freed
/// explicitly.
fn scene_root_is_node3d(scene: &Gd<PackedScene>) -> bool {
    let Some(probe) = scene.instantiate() else {
        return false;
    };
    match probe.try_cast::<Node3D>() {
        Ok(root) => {
            root.free();
            true
        }
        Err(orphan) => {
            orphan.free();
            false
        }
    }
}

/// Build the Godot transform of one scattered instance: quaternion rotation,
/// uniform scale, world-space position.
fn instance_transform(instance: &BlockInstanceData) -> Transform3D {
    let [qx, qy, qz, qw] = instance.rotation;
    let quat = Quaternion::new(qx, qy, qz, qw);
    let basis = Basis::from_quaternion(quat).scaled(Vector3::splat(instance.scale));
    Transform3D::new(
        basis,
        Vector3::new(
            instance.position.x,
            instance.position.y,
            instance.position.z,
        ),
    )
}

/// Instantiate `scene` once per instance data entry, naming each root
/// `{prefix}_{i}`. Scenes whose root is not a `Node3D` are reported and
/// skipped, matching upstream's requirement that instance roots are 3D.
fn instantiate_scene_nodes(
    scene: &Gd<PackedScene>,
    prefix: &str,
    instances: &[BlockInstanceData],
) -> Vec<Gd<Node3D>> {
    let mut nodes = Vec::new();
    for (i, instance) in instances.iter().enumerate() {
        let Some(instantiated) = scene.instantiate() else {
            godot_error!("VoxelInstancer: scene failed to instantiate; instance {i} skipped");
            continue;
        };
        // The skipped subtree was never parented, so it must be freed
        // explicitly — dropping a manually-managed `Gd` alone would leak it.
        let mut node = match instantiated.try_cast::<Node3D>() {
            Ok(node) => node,
            Err(orphan) => {
                godot_error!(
                    "VoxelInstancer: scene root of item must be a Node3D; instance {i} skipped"
                );
                orphan.free();
                continue;
            }
        };
        node.set_transform(instance_transform(instance));
        node.set_name(&format!("{prefix}_{i}"));
        nodes.push(node);
    }
    nodes
}

fn unpack_mesh_locations(packed: PackedInt32Array) -> Vec<(i32, i32, i32, u8)> {
    let slice = packed.as_slice();
    let mut out = Vec::new();
    let mut index = 0;
    while index + 3 < slice.len() {
        let lod = u8::try_from(slice[index + 3].max(0)).unwrap_or(0);
        out.push((slice[index], slice[index + 1], slice[index + 2], lod));
        index += 4;
    }
    out
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
    lod_terrain: Option<Gd<crate::resources2::VoxelLodTerrainGD>>,
    channel: usize,
    value: u64,
    /// Seed for random-tick draws (see `run_blocky_random_tick`).
    random_seed: u32,
}

#[godot_api]
impl IRefCounted for VoxelToolTerrainGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            terrain_path: "..".to_godot(),
            terrain: None,
            lod_terrain: None,
            channel: ChannelId::Sdf.index(),
            value: 1,
            random_seed: 0,
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

    /// Add mode for `do_sphere`/`do_box` (solidify in SDF terms).
    #[constant]
    const MODE_ADD: i64 = 0;
    /// Remove mode for `do_sphere`/`do_box`.
    #[constant]
    const MODE_REMOVE: i64 = 1;

    /// Value written by Set-mode operations (blocky-style channels).
    #[func]
    fn get_value(&self) -> i64 {
        i64::try_from(self.value).unwrap_or(0)
    }

    /// Seed for `run_blocky_random_tick` draws. Same seed + same candidates
    /// produce the same ticks; different seeds cover different subsets.
    #[func]
    fn set_seed(&mut self, seed: i64) {
        let Ok(seed) = u32::try_from(seed) else {
            godot_error!("VoxelToolTerrain.set_seed: seed must be between zero and u32::MAX");
            return;
        };
        self.random_seed = seed;
    }

    #[func]
    fn get_seed(&self) -> i64 {
        i64::from(self.random_seed)
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
        let edit_mode = match mode {
            0 => voxel_core::edition::EditMode::Add,
            1 => voxel_core::edition::EditMode::Remove,
            _ => voxel_core::edition::EditMode::Set,
        };
        let channel = self.channel;
        let value = self.value;
        let core_center = voxel_core::math::Vector3f::new(center.x, center.y, center.z);
        if let Some(mut terrain) = self.terrain.clone() {
            terrain
                .bind_mut()
                .edit_sphere(core_center, radius, channel, edit_mode, value);
            return;
        }
        if let Some(mut terrain) = self.lod_terrain.clone() {
            terrain
                .bind_mut()
                .edit_sphere(core_center, radius, channel, edit_mode, value);
            return;
        }
        godot_error!("VoxelToolTerrain.do_sphere: no terrain is bound");
    }

    #[func]
    fn do_box(&mut self, min: godot::builtin::Vector3i, max: godot::builtin::Vector3i, mode: i32) {
        let edit_mode = match mode {
            0 => voxel_core::edition::EditMode::Add,
            1 => voxel_core::edition::EditMode::Remove,
            _ => voxel_core::edition::EditMode::Set,
        };
        let channel = self.channel;
        let value = self.value;
        let core_min = Vector3i::new(min.x, min.y, min.z);
        let core_max = Vector3i::new(max.x, max.y, max.z);
        if let Some(mut terrain) = self.terrain.clone() {
            terrain
                .bind_mut()
                .edit_box(core_min, core_max, channel, edit_mode, value);
            return;
        }
        if let Some(mut terrain) = self.lod_terrain.clone() {
            terrain
                .bind_mut()
                .edit_box(core_min, core_max, channel, edit_mode, value);
            return;
        }
        godot_error!("VoxelToolTerrain.do_box: no terrain is bound");
    }

    #[func]
    fn set_voxel(&mut self, position: godot::builtin::Vector3i, value: i64) {
        let Ok(value) = validate_voxel_value(value) else {
            godot_error!("VoxelToolTerrain.set_voxel: invalid voxel value");
            return;
        };
        let pos = Vector3i::new(position.x, position.y, position.z);
        let channel = self.channel;
        if let Some(mut terrain) = self.terrain.clone() {
            terrain.bind_mut().edit_world_voxel(pos, channel, value);
            return;
        }
        if let Some(mut terrain) = self.lod_terrain.clone() {
            terrain.bind_mut().edit_world_voxel(pos, channel, value);
            return;
        }
        godot_error!("VoxelToolTerrain.set_voxel: no terrain is bound");
    }

    #[func]
    fn get_voxel(&self, position: godot::builtin::Vector3i) -> i64 {
        let pos = Vector3i::new(position.x, position.y, position.z);
        if let Some(terrain) = self.terrain.as_ref() {
            return i64::try_from(terrain.bind().read_world_voxel(pos, self.channel))
                .unwrap_or_default();
        }
        if let Some(terrain) = self.lod_terrain.as_ref() {
            return i64::try_from(terrain.bind().read_world_voxel(pos, self.channel))
                .unwrap_or_default();
        }
        0
    }

    // -----------------------------------------------------------------
    // Pinned VoxelToolTerrain methods
    // (upstream 5828cbeb: VoxelToolTerrain.xml).
    //
    // The Rust binding wraps a live terrain core and edits it directly
    // (sphere/box/hemisphere/smooth/paste, per-voxel metadata, blocky
    // random-tick). Each brush runs as one storage transaction per
    // overlapping data block.
    // -----------------------------------------------------------------

    /// Operates on a hemisphere, where `flat_direction` points away from the
    /// flat surface (like a normal). `smoothness` blends the flat part with
    /// the rounded part (higher = softer edge).
    #[func]
    #[allow(clippy::too_many_arguments)]
    fn do_hemisphere(
        &mut self,
        center: Vector3,
        radius: f32,
        flat_direction: Vector3,
        smoothness: f32,
        mode: i32,
    ) {
        if !center.x.is_finite()
            || !center.y.is_finite()
            || !center.z.is_finite()
            || !radius.is_finite()
            || radius < 0.0
            || !flat_direction.x.is_finite()
            || !flat_direction.y.is_finite()
            || !flat_direction.z.is_finite()
            || !smoothness.is_finite()
            || smoothness < 0.0
        {
            godot_error!(
                "VoxelToolTerrain.do_hemisphere: center, radius, direction and smoothness must be finite; radius and smoothness non-negative"
            );
            return;
        }
        let core_center = voxel_core::math::Vector3f::new(center.x, center.y, center.z);
        let core_dir =
            voxel_core::math::Vector3f::new(flat_direction.x, flat_direction.y, flat_direction.z);
        let channel = self.channel;
        let value = self.value;
        let edit_mode = match mode {
            1 => voxel_core::edition::EditMode::Remove,
            _ => voxel_core::edition::EditMode::Add,
        };
        if let Some(mut terrain) = self.terrain.clone() {
            terrain.bind_mut().edit_hemisphere(
                core_center,
                radius,
                core_dir,
                smoothness,
                channel,
                edit_mode,
                value,
            );
            return;
        }
        if let Some(mut terrain) = self.lod_terrain.clone() {
            terrain.bind_mut().edit_hemisphere(
                core_center,
                radius,
                core_dir,
                smoothness,
                channel,
                edit_mode,
                value,
            );
            return;
        }
        godot_error!("VoxelToolTerrain.do_hemisphere: no terrain is bound");
    }

    /// Smooth the SDF channel inside a sphere of influence using a box blur.
    #[func]
    fn do_smooth(&mut self, center: Vector3, radius: f32, blur_radius: i32) {
        if !center.x.is_finite()
            || !center.y.is_finite()
            || !center.z.is_finite()
            || !radius.is_finite()
            || radius < 0.0
            || blur_radius < 0
        {
            godot_error!(
                "VoxelToolTerrain.do_smooth: center and radius must be finite; radius and blur_radius non-negative"
            );
            return;
        }
        let core_center = voxel_core::math::Vector3f::new(center.x, center.y, center.z);
        let channel = self.channel;
        if let Some(mut terrain) = self.terrain.clone() {
            terrain
                .bind_mut()
                .edit_smooth(core_center, radius, blur_radius, channel);
            return;
        }
        if let Some(mut terrain) = self.lod_terrain.clone() {
            terrain
                .bind_mut()
                .edit_smooth(core_center, radius, blur_radius, channel);
            return;
        }
        godot_error!("VoxelToolTerrain.do_smooth: no terrain is bound");
    }

    /// Set per-voxel metadata at a world-space position. `nil` clears it.
    #[func]
    fn set_voxel_metadata(&mut self, position: godot::builtin::Vector3i, metadata: Variant) {
        let pos = Vector3i::new(position.x, position.y, position.z);
        let value = match metadata_from_variant(&metadata) {
            Ok(value) => value,
            Err(error) => {
                godot_error!("VoxelToolTerrain.set_voxel_metadata: {error}");
                return;
            }
        };
        let stored = if value.is_nil() { None } else { Some(value) };
        if let Some(mut terrain) = self.terrain.clone() {
            terrain.bind_mut().edit_world_voxel_metadata(pos, stored);
            return;
        }
        if let Some(mut terrain) = self.lod_terrain.clone() {
            terrain.bind_mut().edit_world_voxel_metadata(pos, stored);
            return;
        }
        godot_error!("VoxelToolTerrain.set_voxel_metadata: no terrain is bound");
    }

    /// Get per-voxel metadata at a world-space position, or `nil`.
    #[func]
    fn get_voxel_metadata(&self, position: godot::builtin::Vector3i) -> Variant {
        let pos = Vector3i::new(position.x, position.y, position.z);
        let value = if let Some(terrain) = self.terrain.as_ref() {
            terrain.bind().read_world_voxel_metadata(pos)
        } else if let Some(terrain) = self.lod_terrain.as_ref() {
            terrain.bind().read_world_voxel_metadata(pos)
        } else {
            return Variant::nil();
        };
        match value {
            Some(metadata) => metadata_to_variant(&metadata),
            None => Variant::nil(),
        }
    }

    /// Executes a function for each voxel holding metadata in the given area.
    /// The callback takes two arguments: voxel position (`Vector3i`) and voxel
    /// metadata (`Variant`).
    #[func]
    fn for_each_voxel_metadata_in_area(&self, voxel_area: Aabb, callback: Callable) {
        if !callback.is_valid() {
            godot_error!("VoxelToolTerrain.for_each_voxel_metadata_in_area: callback is invalid");
            return;
        }
        if !voxel_area.position.x.is_finite()
            || !voxel_area.position.y.is_finite()
            || !voxel_area.position.z.is_finite()
            || !voxel_area.size.x.is_finite()
            || !voxel_area.size.y.is_finite()
            || !voxel_area.size.z.is_finite()
        {
            godot_error!("VoxelToolTerrain.for_each_voxel_metadata_in_area: area must be finite");
            return;
        }
        let min = Vector3i::new(
            voxel_area.position.x.floor() as i32,
            voxel_area.position.y.floor() as i32,
            voxel_area.position.z.floor() as i32,
        );
        let max = Vector3i::new(
            (voxel_area.position.x + voxel_area.size.x).ceil() as i32,
            (voxel_area.position.y + voxel_area.size.y).ceil() as i32,
            (voxel_area.position.z + voxel_area.size.z).ceil() as i32,
        );
        if let Some(terrain) = self.terrain.as_ref() {
            terrain
                .bind()
                .for_each_world_voxel_metadata(min, max, |pos, value| {
                    let gpos = godot::builtin::Vector3i::new(pos.x, pos.y, pos.z);
                    callback.call(&[gpos.to_variant(), metadata_to_variant(value)]);
                });
            return;
        }
        if let Some(terrain) = self.lod_terrain.as_ref() {
            terrain
                .bind()
                .for_each_world_voxel_metadata(min, max, |pos, value| {
                    let gpos = godot::builtin::Vector3i::new(pos.x, pos.y, pos.z);
                    callback.call(&[gpos.to_variant(), metadata_to_variant(value)]);
                });
            return;
        }
        godot_error!("VoxelToolTerrain.for_each_voxel_metadata_in_area: no terrain is bound");
    }

    /// Paste a `VoxelBuffer` into the bound terrain so `src(0,0,0)` lands at
    /// `origin`. `channel_mask` is a bitset of channels to copy.
    #[func]
    fn do_paste(
        &mut self,
        origin: godot::builtin::Vector3i,
        source: Gd<RefCounted>,
        channel_mask: i64,
    ) {
        let Ok(buffer) = source.try_cast::<VoxelBufferGD>() else {
            godot_error!("VoxelToolTerrain.do_paste: source must be a VoxelBuffer");
            return;
        };
        let mask = u8::try_from(channel_mask).unwrap_or(u8::MAX);
        let pos = Vector3i::new(origin.x, origin.y, origin.z);
        let bound = buffer.bind();
        if let Some(mut terrain) = self.terrain.clone() {
            terrain
                .bind_mut()
                .paste_buffer(pos, bound.core_buffer(), mask);
            return;
        }
        if let Some(mut terrain) = self.lod_terrain.clone() {
            terrain
                .bind_mut()
                .paste_buffer(pos, bound.core_buffer(), mask);
            return;
        }
        godot_error!("VoxelToolTerrain.do_paste: no terrain is bound");
    }

    /// Picks voxels within `area` and calls `callback(position, value)` on a
    /// strided subset. `voxel_count` is the maximum number of callbacks;
    /// `batch_count` controls the stride. When a `VoxelMesherBlocky` library
    /// is attached, only `random_tickable` models whose `tags_mask` intersects
    /// `tags_mask` are candidates (`tags_mask == 0` means any tag). Without a
    /// library, non-zero voxels on the tool channel are candidates and a
    /// non-zero `tags_mask` filters by `(value & tags_mask) != 0`.
    #[func]
    fn run_blocky_random_tick(
        &mut self,
        area: Aabb,
        voxel_count: i32,
        callback: Callable,
        batch_count: i32,
        tags_mask: i32,
    ) {
        if !callback.is_valid() {
            godot_error!("VoxelToolTerrain.run_blocky_random_tick: callback is invalid");
            return;
        }
        let Ok(limit) = usize::try_from(voxel_count.max(0)) else {
            return;
        };
        let limit = limit.min(MAX_SCRIPT_ITEMS);
        if limit == 0 {
            return;
        }
        let batch = usize::try_from(batch_count.max(1)).unwrap_or(1);
        let min = Vector3i::new(
            area.position.x.floor() as i32,
            area.position.y.floor() as i32,
            area.position.z.floor() as i32,
        );
        let max = Vector3i::new(
            (area.position.x + area.size.x).ceil() as i32 - 1,
            (area.position.y + area.size.y).ceil() as i32 - 1,
            (area.position.z + area.size.z).ceil() as i32 - 1,
        );
        if !area.position.x.is_finite()
            || !area.position.y.is_finite()
            || !area.position.z.is_finite()
            || !area.size.x.is_finite()
            || !area.size.y.is_finite()
            || !area.size.z.is_finite()
        {
            godot_error!("VoxelToolTerrain.run_blocky_random_tick: area must be finite");
            return;
        }
        // Bound the scan, not just the results: the box is iterated per
        // voxel on the main thread, so a giant AABB must be rejected up
        // front instead of hanging the frame.
        let span = |lo: i64, hi: i64| (hi - lo + 1).max(0);
        let volume = span(min.x as i64, max.x as i64)
            .saturating_mul(span(min.y as i64, max.y as i64))
            .saturating_mul(span(min.z as i64, max.z as i64));
        if volume > MAX_SCRIPT_VOXELS as i64 {
            godot_error!(
                "VoxelToolTerrain.run_blocky_random_tick: area volume {volume} exceeds the scan budget {MAX_SCRIPT_VOXELS}"
            );
            return;
        }
        let channel = self.channel;
        let mask = tags_mask as u32;
        // The candidate filter runs during the scan: collecting any non-zero
        // voxel first and filtering afterwards let dense untickable material
        // (e.g. stone) fill the scan cap and starve tickable voxels later in
        // ZYX order.
        let is_candidate = |library: Option<&voxel_core::meshers::blocky::BakedLibrary>,
                            value: u64| {
            voxel_core::edition::ops::voxel_is_random_tick_candidate(value, mask, library)
        };
        let candidates = if let Some(terrain) = self.terrain.as_ref() {
            let bound = terrain.bind();
            let library = bound.blocky_library();
            bound.collect_voxels_in_box(min, max, channel, MAX_SCRIPT_ITEMS, |value| {
                is_candidate(library.as_ref(), value)
            })
        } else if let Some(terrain) = self.lod_terrain.as_ref() {
            let bound = terrain.bind();
            let library = bound.blocky_library();
            bound.collect_voxels_in_box(min, max, channel, MAX_SCRIPT_ITEMS, |value| {
                is_candidate(library.as_ref(), value)
            })
        } else {
            godot_error!("VoxelToolTerrain.run_blocky_random_tick: no terrain is bound");
            return;
        };
        if candidates.is_empty() {
            return;
        }
        // Draw uniformly random candidates per call (upstream semantics):
        // a fixed stride would re-tick the same positions every call and
        // permanently starve the rest. `batch` spreads the draws across
        // invocations for statistical coverage; `voxel_count` caps total
        // callbacks this call. Deterministic under a fixed seed.
        let mut rng = voxel_core::instancing::scatter::SimpleRng::new(self.random_seed);
        let draws = batch.min(limit).min(candidates.len());
        let mut picked: Vec<usize> = (0..candidates.len()).collect();
        for i in 0..draws {
            let j = i + (rng.next_u32() as usize) % (picked.len() - i);
            picked.swap(i, j);
            let (pos, value) = &candidates[picked[i]];
            let gpos = godot::builtin::Vector3i::new(pos.x, pos.y, pos.z);
            let gval = i64::try_from(*value).unwrap_or(0);
            callback.call(&[gpos.to_variant(), gval.to_variant()]);
        }
    }
}

impl VoxelToolTerrainGD {
    pub(crate) fn bind_terrain(&mut self, terrain: Gd<crate::terrain::VoxelTerrain>) {
        self.terrain = Some(terrain);
        self.lod_terrain = None;
    }

    pub(crate) fn bind_lod_terrain(&mut self, terrain: Gd<crate::resources2::VoxelLodTerrainGD>) {
        self.lod_terrain = Some(terrain);
        self.terrain = None;
    }
}

// ---------------------------------------------------------------------------
// VoxelRaycastResultGD — RefCounted result container
// ---------------------------------------------------------------------------

/// Result of a voxel raycast. Contains hit position, previous position,
/// and distance along the ray.
///
/// The pinned GDScript-facing properties (`distance`, `normal`, `position`,
/// `previous_position`) mirror upstream `VoxelRaycastResult` (5828cbeb). They
/// are read-only getters (no setter) composing the integer fields.
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
    distance: f32,
    #[var]
    normal_x: i32,
    #[var]
    normal_y: i32,
    #[var]
    normal_z: i32,
    /// The pinned GDScript-facing read-only `distance` property.
    #[var(get = get_distance, no_set)]
    distance_prop: PhantomVar<f32>,
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

/// Pack a core raycast hit into the pinned `VoxelRaycastResult` member tuple
/// `(distance, normal, position, previous_position)`. Producers (the
/// `VoxelTool.raycast` binding, staged later) fill a result's fields from this
/// tuple; the field mapping here is the engine-free contract the tests pin
/// down.
#[allow(dead_code)] // producers land in a later stage; tests exercise it now
pub(crate) fn raycast_result_from_hit(
    hit: &voxel_core::edition::raycast::VoxelRaycastHit,
) -> (
    f32,
    Vector3,
    godot::builtin::Vector3i,
    godot::builtin::Vector3i,
) {
    (
        hit.distance,
        Vector3::new(
            hit.normal.x as f32,
            hit.normal.y as f32,
            hit.normal.z as f32,
        ),
        godot::builtin::Vector3i::new(hit.position.x, hit.position.y, hit.position.z),
        godot::builtin::Vector3i::new(
            hit.previous_position.x,
            hit.previous_position.y,
            hit.previous_position.z,
        ),
    )
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
            distance_prop: PhantomVar::default(),
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

    /// Distance between the origin of the ray and the surface of the cube
    /// representing the hit voxel (upstream default `0.0`).
    #[func]
    fn get_distance(&self) -> f32 {
        self.distance
    }

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

/// Engine-free tests for the pinned `VoxelRaycastResult` member mapping.
/// `raycast_result_from_hit` is the producer-side contract; it only touches
/// voxel-core hits and Godot builtin values, so no Godot runtime is needed.
#[cfg(test)]
mod raycast_result_tests {
    use super::raycast_result_from_hit;
    use godot::builtin::Vector3i as GdVector3i;
    use godot::prelude::Vector3;
    use voxel_core::edition::raycast::voxel_raycast;
    use voxel_core::edition::raycast::VoxelRaycastHit;
    use voxel_core::math::{Vector3f, Vector3i};

    #[test]
    fn raycast_result_defaults_match_pinned_members() {
        // A freshly-initialized result carries all-zero fields, so packing a
        // zero hit must yield exactly the pinned upstream member defaults:
        // distance 0.0, normal Vector3(0,0,0), position Vector3i(0,0,0),
        // previous_position Vector3i(0,0,0).
        let zero_hit = VoxelRaycastHit {
            position: Vector3i::new(0, 0, 0),
            previous_position: Vector3i::new(0, 0, 0),
            distance: 0.0,
            normal: Vector3i::new(0, 0, 0),
        };
        let (distance, normal, position, previous_position) = raycast_result_from_hit(&zero_hit);
        assert_eq!(distance, 0.0);
        assert_eq!(normal, Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(position, GdVector3i::new(0, 0, 0));
        assert_eq!(previous_position, GdVector3i::new(0, 0, 0));
    }

    #[test]
    fn raycast_result_packs_core_hit() {
        // Mirrors the core test raycast.rs#raycast_hits_solid_voxel_along_x:
        // from (0.5, 0.5, 0.5) along +X, the ray hits voxel (3, 0, 0) after
        // passing through (2, 0, 0), entering through its -X face.
        let hit = voxel_raycast(
            Vector3f::new(0.5, 0.5, 0.5),
            Vector3f::new(1.0, 0.0, 0.0),
            100.0,
            |state| state.position == Vector3i::new(3, 0, 0),
        )
        .expect("ray along +X must reach voxel (3,0,0)");
        let (distance, normal, position, previous_position) = raycast_result_from_hit(&hit);
        assert!(
            distance > 0.0,
            "hit distance must be positive (got {distance})"
        );
        assert_eq!(normal, Vector3::new(-1.0, 0.0, 0.0));
        assert_eq!(position, GdVector3i::new(3, 0, 0));
        assert_eq!(previous_position, GdVector3i::new(2, 0, 0));
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
    /// The graph under construction. Assigned generators compile this graph.
    graph: voxel_core::generators::graph::Graph,
    /// Cached compiled generator matching `graph`.
    generator: Option<voxel_core::generators::graph::GraphGenerator>,
    // -----------------------------------------------------------------
    // Pinned VoxelGeneratorGraph members (upstream VoxelGeneratorGraph.xml).
    // Backing fields + validated setters; `use_xz_caching` is wired into the
    // core generator, the rest are stored faithfully (see setter comments).
    // -----------------------------------------------------------------
    /// Pinned `debug_block_clipping` (stored; upstream inverts clipped blocks
    /// for visualization — the Rust core does not implement block clipping
    /// inversion yet, so this currently steers nothing).
    debug_block_clipping_value: bool,
    /// Pinned `sdf_clip_threshold` (stored; core culling is interval-based and
    /// always-on, so the threshold does not gate anything yet).
    sdf_clip_threshold_value: f32,
    /// Pinned `subdivision_size` (stored; the core does not run range analysis
    /// on block subdivisions yet).
    subdivision_size_value: i64,
    /// Pinned `texture_mode` (stored; texture outputs are not produced by the
    /// Rust graph runtime yet).
    texture_mode_value: i64,
    /// Pinned `use_optimized_execution_map` (stored; the Rust runtime has no
    /// per-area node-skipping execution map yet).
    use_optimized_execution_map_value: bool,
    /// Pinned `use_subdivision` (stored; gates `subdivision_size`, which the
    /// core does not consume yet).
    use_subdivision_value: bool,
    /// Pinned `use_xz_caching` — wired to the core generator's XZ-prefix
    /// cache (see `create_core_generator`).
    use_xz_caching_value: bool,
    /// Pinned `debug_block_clipping` GDScript property.
    #[var(get = is_debug_clipped_blocks, set = set_debug_clipped_blocks)]
    debug_block_clipping: PhantomVar<bool>,
    /// Pinned `sdf_clip_threshold` GDScript property.
    #[var(get = get_sdf_clip_threshold, set = set_sdf_clip_threshold)]
    sdf_clip_threshold: PhantomVar<f32>,
    /// Pinned `subdivision_size` GDScript property.
    #[var(get = get_subdivision_size, set = set_subdivision_size)]
    subdivision_size: PhantomVar<i64>,
    /// Pinned `texture_mode` GDScript property (enum `TextureMode`).
    #[var(get = get_texture_mode, set = set_texture_mode)]
    texture_mode: PhantomVar<i64>,
    /// Pinned `use_optimized_execution_map` GDScript property.
    #[var(get = is_using_optimized_execution_map, set = set_use_optimized_execution_map)]
    use_optimized_execution_map: PhantomVar<bool>,
    /// Pinned `use_subdivision` GDScript property.
    #[var(get = is_using_subdivision, set = set_use_subdivision)]
    use_subdivision: PhantomVar<bool>,
    /// Pinned `use_xz_caching` GDScript property.
    #[var(get = is_using_xz_caching, set = set_use_xz_caching)]
    use_xz_caching: PhantomVar<bool>,
}

#[godot_api]
impl IResource for VoxelGeneratorGraphGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            graph_json: "{}".to_godot(),
            graph: voxel_core::generators::graph::Graph::new(),
            generator: None,
            // Pinned upstream defaults (VoxelGeneratorGraph.xml).
            debug_block_clipping_value: false,
            sdf_clip_threshold_value: 1.5,
            subdivision_size_value: 16,
            texture_mode_value: 0,
            use_optimized_execution_map_value: true,
            use_subdivision_value: true,
            use_xz_caching_value: true,
            debug_block_clipping: PhantomVar::default(),
            sdf_clip_threshold: PhantomVar::default(),
            subdivision_size: PhantomVar::default(),
            texture_mode: PhantomVar::default(),
            use_optimized_execution_map: PhantomVar::default(),
            use_subdivision: PhantomVar::default(),
            use_xz_caching: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelGeneratorGraphGD {
    // Pinned `TextureMode` enum constants (upstream VoxelGeneratorGraph.xml).
    const TEXTURE_MODE_MIXEL4: i64 = 0;
    const TEXTURE_MODE_SINGLE: i64 = 1;

    /// Emitted when a graph node is renamed (upstream `node_name_changed`).
    ///
    /// Not emitted yet: node names are not editable through this binding —
    /// they arrive with the `VoxelGraphFunction` rework in a later stage.
    #[signal]
    fn node_name_changed(node_id: i64);

    /// Canonical `clear`: erase all nodes and connections from the graph
    /// (alias of [`Self::clear_graph`]).
    #[func]
    fn clear(&mut self) {
        self.clear_graph();
    }

    /// Canonical `compile`: compile the graph and return a report dictionary.
    ///
    /// Layout: `{"success": bool, "node_id": int, "message": String}`.
    /// `node_id` is `-1` on success or when the error is not about a
    /// particular node; otherwise it is the id of the offending node
    /// (a dangling input port reference, or any node for a cycle).
    #[func]
    fn compile(&self) -> VarDictionary {
        use voxel_core::generators::graph::CompiledGraph;
        let mut result = VarDictionary::new();
        match CompiledGraph::compile(&self.graph) {
            Ok(_) => {
                result.set("success", true);
                result.set("node_id", -1_i64);
                result.set("message", &GString::new());
            }
            Err(error) => {
                let (node_id, message) = graph_compile_error_report(&error);
                result.set("success", false);
                result.set("node_id", node_id);
                result.set("message", &message.to_godot());
            }
        }
        result
    }

    /// Generates a block of voxels within the specified world area (pinned
    /// `generate_block`, upstream `VoxelGenerator.xml`). Same pattern as the
    /// other concrete generators: build the core generator, run it into the
    /// buffer, truncate the float origin via the shared helper.
    #[func]
    fn generate_block(
        &self,
        mut out_buffer: Gd<crate::voxel_buffer::VoxelBufferGD>,
        origin_in_voxels: Vector3,
        lod: i32,
    ) {
        let generator = self.create_core_generator();
        crate::generators::generate_core_block_into_gd(
            generator.as_ref(),
            &mut out_buffer,
            origin_in_voxels,
            lod,
        );
    }

    /// Canonical `debug_analyze_range`: estimate the SDF output range over the
    /// axis-aligned box with corners `min_pos`/`max_pos`. Returns
    /// `Vector2(min, max)`; `Vector2(NAN, NAN)` (with an error log) on invalid
    /// input or a graph that does not compile.
    ///
    /// Note: this is interval analysis over the compiled graph — upstream's
    /// variant inspects its optimized execution map instead. For a debug
    /// method the interval estimate is the honest equivalent.
    #[func]
    fn debug_analyze_range(&self, min_pos: Vector3, max_pos: Vector3) -> Vector2 {
        match graph_debug_analyze_range(
            &self.graph,
            [min_pos.x, min_pos.y, min_pos.z],
            [max_pos.x, max_pos.y, max_pos.z],
        ) {
            Ok((min, max)) => Vector2::new(min, max),
            Err(message) => {
                godot_error!("VoxelGeneratorGraph.debug_analyze_range: {message}");
                Vector2::new(f32::NAN, f32::NAN)
            }
        }
    }

    /// Canonical `raycast_sdf_approx`: march the ray from `ray_origin` to
    /// `ray_end` in steps of `stride`, sampling the compiled SDF. Returns the
    /// distance from the origin at the first sample where the SDF is negative,
    /// or `-1.0` when the ray never enters matter (or the input is invalid /
    /// the graph does not compile). The result is accurate up to `stride`.
    #[func]
    fn raycast_sdf_approx(&self, ray_origin: Vector3, ray_end: Vector3, stride: f32) -> f32 {
        match graph_raycast_sdf_approx(
            &self.graph,
            [ray_origin.x, ray_origin.y, ray_origin.z],
            [ray_end.x, ray_end.y, ray_end.z],
            stride,
        ) {
            Ok(Some(distance)) => distance,
            Ok(None) => -1.0,
            Err(message) => {
                godot_error!("VoxelGeneratorGraph.raycast_sdf_approx: {message}");
                -1.0
            }
        }
    }

    /// Canonical `debug_load_waves_preset`: replace the graph with the
    /// built-in sin-waves terrain preset
    /// (`sin(0.1 * x) + cos(0.1 * z)` → SDF).
    #[func]
    fn debug_load_waves_preset(&mut self) {
        self.graph = waves_preset_graph();
        self.generator = None;
    }

    /// Canonical `debug_measure_microseconds_per_voxel`: indicative timing of
    /// the compiled graph. With `use_singular_queries = false` it times whole
    /// Y-slice generation; with `true` it times the same volume as one query
    /// per voxel. Returns microseconds per voxel, or `-1.0` if the graph has
    /// no SDF output or does not compile.
    #[func]
    fn debug_measure_microseconds_per_voxel(&self, use_singular_queries: bool) -> f32 {
        graph_measure_microseconds_per_voxel(&self.graph, use_singular_queries)
    }

    /// Canonical `generate_image_from_sdf`: sample the compiled SDF on the
    /// plane spanned by the transform's X/Y axes (span `size` in world units,
    /// centered on the transform origin) and store each sample into the
    /// corresponding image pixel (samples centered on pixels). The image
    /// should have a 32-bit float format and must not be compressed. The
    /// written pixel count is bounded by the shared
    /// `MAX_GENERATED_IMAGE_PIXELS` script workload budget.
    #[func]
    fn generate_image_from_sdf(
        &self,
        mut im: Gd<godot::classes::Image>,
        transform: Transform3D,
        size: Vector2,
    ) {
        use voxel_core::generators::graph::CompiledGraph;
        let width = im.get_width();
        let height = im.get_height();
        if width <= 0 || height <= 0 {
            godot_error!("VoxelGeneratorGraph.generate_image_from_sdf: image must not be empty");
            return;
        }
        let pixel_count = i64::from(width) * i64::from(height);
        if pixel_count > crate::resources3::MAX_GENERATED_IMAGE_PIXELS {
            godot_error!(
                "VoxelGeneratorGraph.generate_image_from_sdf: pixel count exceeds the script workload limit"
            );
            return;
        }
        if !size.x.is_finite() || !size.y.is_finite() || size.x <= 0.0 || size.y <= 0.0 {
            godot_error!(
                "VoxelGeneratorGraph.generate_image_from_sdf: size components must be finite and positive"
            );
            return;
        }
        if im.is_compressed() {
            godot_error!(
                "VoxelGeneratorGraph.generate_image_from_sdf: image must not be compressed"
            );
            return;
        }
        let Ok(compiled) = CompiledGraph::compile(&self.graph) else {
            godot_error!("VoxelGeneratorGraph.generate_image_from_sdf: graph does not compile");
            return;
        };
        if !compiled.nodes().iter().any(|n| n.kind.is_output()) {
            godot_error!(
                "VoxelGeneratorGraph.generate_image_from_sdf: graph has no SDF output node"
            );
            return;
        }
        let mut scratch = voxel_core::generators::graph::CompiledScratch::new();
        for py in 0..height {
            for px in 0..width {
                // Pixel-center samples; +Y up in the plane like upstream.
                let u = ((px as f32 + 0.5) / width as f32 - 0.5) * size.x;
                let v = (0.5 - (py as f32 + 0.5) / height as f32) * size.y;
                let local = Vector3::new(u, v, 0.0);
                let world = transform.basis * local + transform.origin;
                let sdf = sample_compiled_sdf(&compiled, world.x, world.y, world.z, &mut scratch);
                im.set_pixel(px, py, Color::from_rgb(sdf, sdf, sdf));
            }
        }
    }

    // -----------------------------------------------------------------
    // Pinned VoxelGeneratorGraph members (upstream VoxelGeneratorGraph.xml).
    // -----------------------------------------------------------------

    /// Pinned `debug_block_clipping` getter. Stored faithfully; the core does
    /// not implement clipped-block inversion yet, so it steers nothing.
    #[func]
    fn is_debug_clipped_blocks(&self) -> bool {
        self.debug_block_clipping_value
    }

    #[func]
    fn set_debug_clipped_blocks(&mut self, enabled: bool) {
        self.debug_block_clipping_value = enabled;
    }

    /// Pinned `sdf_clip_threshold` getter (any finite real).
    #[func]
    fn get_sdf_clip_threshold(&self) -> f32 {
        self.sdf_clip_threshold_value
    }

    #[func]
    fn set_sdf_clip_threshold(&mut self, threshold: f32) {
        if validate_graph_finite_float(threshold).is_err() {
            godot_error!("VoxelGeneratorGraph.set_sdf_clip_threshold: value must be finite");
            return;
        }
        self.sdf_clip_threshold_value = threshold;
    }

    /// Pinned `subdivision_size` getter (positive power of two).
    #[func]
    fn get_subdivision_size(&self) -> i64 {
        self.subdivision_size_value
    }

    #[func]
    fn set_subdivision_size(&mut self, size: i64) {
        if validate_graph_subdivision_size(size).is_err() {
            godot_error!(
                "VoxelGeneratorGraph.set_subdivision_size: value must be a power of two >= 2"
            );
            return;
        }
        self.subdivision_size_value = size;
    }

    /// Pinned `texture_mode` getter (enum `TextureMode`).
    #[func]
    fn get_texture_mode(&self) -> i64 {
        self.texture_mode_value
    }

    #[func]
    fn set_texture_mode(&mut self, mode: i64) {
        if validate_graph_texture_mode(mode).is_err() {
            godot_error!(
                "VoxelGeneratorGraph.set_texture_mode: value must be TEXTURE_MODE_MIXEL4 (0) or TEXTURE_MODE_SINGLE (1)"
            );
            return;
        }
        self.texture_mode_value = mode;
    }

    /// Pinned `use_optimized_execution_map` getter. Stored faithfully; the
    /// Rust runtime has no per-area node-skipping execution map yet.
    #[func]
    fn is_using_optimized_execution_map(&self) -> bool {
        self.use_optimized_execution_map_value
    }

    #[func]
    fn set_use_optimized_execution_map(&mut self, enabled: bool) {
        self.use_optimized_execution_map_value = enabled;
    }

    /// Pinned `use_subdivision` getter. Gates `subdivision_size`, which the
    /// core does not consume yet.
    #[func]
    fn is_using_subdivision(&self) -> bool {
        self.use_subdivision_value
    }

    #[func]
    fn set_use_subdivision(&mut self, enabled: bool) {
        self.use_subdivision_value = enabled;
    }

    /// Pinned `use_xz_caching` getter. Wired to the core generator's
    /// XZ-prefix cache (see `create_core_generator`).
    #[func]
    fn is_using_xz_caching(&self) -> bool {
        self.use_xz_caching_value
    }

    #[func]
    fn set_use_xz_caching(&mut self, enabled: bool) {
        self.use_xz_caching_value = enabled;
    }

    /// Remove every node from the graph under construction.
    #[func]
    fn clear_graph(&mut self) {
        self.graph.clear();
        self.generator = None;
        self.graph_json = "{}".to_godot();
    }

    /// Append a node. Port arguments are node ids; `-1` means unconnected.
    /// Returns the new node id, or `-1` for an unknown kind.
    #[func]
    fn add_node(&mut self, kind: GString, a: i64, b: i64, c: i64, d: i64, value: f32) -> i64 {
        if !value.is_finite() {
            godot_error!("VoxelGeneratorGraph.add_node: value must be finite");
            return -1;
        }
        let Some(node_kind) = voxel_core::generators::graph::node_kind_from_spec(
            &kind.to_string(),
            a,
            b,
            c,
            d,
            value,
        ) else {
            godot_error!("VoxelGeneratorGraph.add_node: unknown kind '{kind}'");
            return -1;
        };
        let id = self.graph.push(node_kind);
        self.generator = None;
        i64::from(id)
    }

    /// Append an `Expression` node. Variables `x`/`y`/`z` bind to the given
    /// port ids (`-1` = 0.0). Returns the new node id, or `-1` on parse error.
    #[func]
    fn add_expression_node(&mut self, expression: GString, x: i64, y: i64, z: i64) -> i64 {
        match voxel_core::generators::graph::expression_node::ExpressionNode::new(
            &expression.to_string(),
            &[("x", 0), ("y", 1), ("z", 2)],
        ) {
            Ok(expr) => {
                let id = self
                    .graph
                    .push(voxel_core::generators::graph::NodeKind::Expression {
                        x: voxel_core::generators::graph::optional_graph_port(x),
                        y: voxel_core::generators::graph::optional_graph_port(y),
                        z: voxel_core::generators::graph::optional_graph_port(z),
                        expr: std::sync::Arc::new(expr),
                    });
                self.generator = None;
                i64::from(id)
            }
            Err(message) => {
                godot_error!("VoxelGeneratorGraph.add_expression_node: {message}");
                -1
            }
        }
    }

    /// Append an `Image2D` node filled with `fill`, sampled at ports `x`/`y`.
    #[func]
    fn add_image2d_node(&mut self, width: i32, height: i32, fill: f32, x: i64, y: i64) -> i64 {
        if width <= 0 || height <= 0 || width > 4096 || height > 4096 || !fill.is_finite() {
            godot_error!(
                "VoxelGeneratorGraph.add_image2d_node: width/height must be in 1..=4096 and fill finite"
            );
            return -1;
        }
        let image = voxel_core::generators::graph::image::Image2D::new_filled(
            width as u32,
            height as u32,
            fill,
        );
        let id = self
            .graph
            .push(voxel_core::generators::graph::NodeKind::Image2D {
                x: voxel_core::generators::graph::optional_graph_port(x),
                y: voxel_core::generators::graph::optional_graph_port(y),
                image: std::sync::Arc::new(image),
            });
        self.generator = None;
        i64::from(id)
    }

    /// Number of nodes in the graph under construction.
    #[func]
    fn get_graph_node_count(&self) -> i32 {
        i32::try_from(self.graph.nodes().len()).unwrap_or(i32::MAX)
    }

    /// `true` if the current graph compiles (no cycles / dangling ports).
    #[func]
    fn compile_graph(&self) -> bool {
        voxel_core::generators::graph::CompiledGraph::compile(&self.graph).is_ok()
    }

    #[func]
    fn get_graph_json(&self) -> GString {
        if self.graph.nodes().is_empty() {
            return self.graph_json.clone();
        }
        voxel_core::generators::graph::graph_to_json(&self.graph).to_godot()
    }

    /// Replace the graph JSON interchange string. A non-empty document that is
    /// not `{}` is parsed as a compact node list (`{"nodes":[...]}`). Parse
    /// failures log an error and leave the previous graph in place.
    #[func]
    fn set_graph_json(&mut self, json: GString) {
        let text = json.to_string();
        self.graph_json = json;
        let trimmed = text.trim();
        if trimmed.is_empty() || trimmed == "{}" {
            return;
        }
        match parse_graph_json(trimmed) {
            Ok(graph) => {
                self.graph = graph;
                self.generator = None;
            }
            Err(message) => {
                godot_error!("VoxelGeneratorGraph.set_graph_json: {message}");
            }
        }
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

    /// Returns the number of nodes in the graph under construction.
    #[func]
    fn get_node_count(&self) -> i32 {
        self.get_graph_node_count()
    }

    /// Compile the current graph and sample the SDF at `(x, y, z)`.
    /// Returns `NaN` if the graph is empty or does not compile.
    #[func]
    fn compile_and_sample(&self, x: f32, y: f32, z: f32) -> f32 {
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            godot_error!("VoxelGeneratorGraph.compile_and_sample: coordinates must be finite");
            return f32::NAN;
        }
        if self.graph.nodes().is_empty() {
            return f32::NAN;
        }
        use voxel_core::generators::graph::{
            CompiledGraph, CompiledScratch, GraphInputs, GraphOutput,
        };
        let Ok(compiled) = CompiledGraph::compile(&self.graph) else {
            return f32::NAN;
        };
        let xs = [x];
        let zs = [z];
        let inputs = GraphInputs { x: &xs, y, z: &zs };
        let mut scratch = CompiledScratch::new();
        let mut out = Vec::new();
        compiled.generate_slice(&inputs, 1, &mut scratch, &mut out, false);
        out.into_iter()
            .find(|(kind, _)| *kind == GraphOutput::Sdf)
            .and_then(|(_, values)| values.into_iter().next())
            .unwrap_or(f32::NAN)
    }
}

impl VoxelGeneratorGraphGD {
    /// Build the engine-agnostic generator assigned to terrain. An empty graph
    /// becomes a default origin sphere so assigning this resource never
    /// silently turns into Waves. The pinned `use_xz_caching` property gates
    /// the core generator's XZ-prefix cache.
    pub(crate) fn create_core_generator(&self) -> voxel_core::storage::SharedVoxelGenerator {
        let graph = if self.graph.nodes().is_empty() {
            default_sphere_graph()
        } else {
            self.graph.clone()
        };
        Arc::new(
            voxel_core::generators::graph::GraphGenerator::new(graph)
                .with_xz_caching(self.use_xz_caching_value),
        )
    }
}

fn default_sphere_graph() -> voxel_core::generators::graph::Graph {
    use voxel_core::generators::graph::{Graph, GraphPort, NodeKind};
    let mut graph = Graph::new();
    let nx = graph.push(NodeKind::InputX);
    let ny = graph.push(NodeKind::InputY);
    let nz = graph.push(NodeKind::InputZ);
    let nr = graph.push(NodeKind::Constant(32.0));
    let sphere = graph.push(NodeKind::SdfSphere {
        x: Some(GraphPort::new(nx)),
        y: Some(GraphPort::new(ny)),
        z: Some(GraphPort::new(nz)),
        radius: Some(GraphPort::new(nr)),
    });
    graph.push(NodeKind::OutputSdf {
        a: Some(GraphPort::new(sphere)),
    });
    graph
}

/// The built-in sin-waves terrain preset (`debug_load_waves_preset`):
/// `sdf = sin(0.1 * x) + cos(0.1 * z)`. Built from existing node kinds
/// (InputX/Z, Constant, Multiply, Sin, Cos, Add, OutputSdf) so it exercises
/// the ordinary graph pipeline. Engine-free for unit testing.
fn waves_preset_graph() -> voxel_core::generators::graph::Graph {
    use voxel_core::generators::graph::{Graph, GraphPort, NodeKind};
    const FREQUENCY: f32 = 0.1;
    let mut graph = Graph::new();
    let nx = graph.push(NodeKind::InputX);
    let nz = graph.push(NodeKind::InputZ);
    let frequency = graph.push(NodeKind::Constant(FREQUENCY));
    let scaled_x = graph.push(NodeKind::Multiply {
        a: Some(GraphPort::new(nx)),
        b: Some(GraphPort::new(frequency)),
    });
    let scaled_z = graph.push(NodeKind::Multiply {
        a: Some(GraphPort::new(nz)),
        b: Some(GraphPort::new(frequency)),
    });
    let sin_x = graph.push(NodeKind::Sin {
        a: Some(GraphPort::new(scaled_x)),
    });
    let cos_z = graph.push(NodeKind::Cos {
        a: Some(GraphPort::new(scaled_z)),
    });
    let sum = graph.push(NodeKind::Add {
        a: Some(GraphPort::new(sin_x)),
        b: Some(GraphPort::new(cos_z)),
    });
    graph.push(NodeKind::OutputSdf {
        a: Some(GraphPort::new(sum)),
    });
    graph
}

/// Map a core topology error to the pinned `compile()` report fields:
/// `(node_id, message)`. `node_id` is `-1` when the error is not about a
/// particular node (cycles). Engine-free for unit testing.
fn graph_compile_error_report(error: &voxel_core::generators::graph::TopoError) -> (i64, String) {
    use voxel_core::generators::graph::TopoError;
    match error {
        TopoError::Cycle => (
            -1,
            "the graph contains a cycle (a node depends on itself)".to_owned(),
        ),
        TopoError::DanglingPort(id) => (
            i64::from(*id),
            format!("an input references node {id}, which does not exist"),
        ),
    }
}

/// Sample the first SDF output of a compiled graph at a single world point.
/// Returns `NaN` when the graph produces no SDF output. Engine-free.
fn sample_compiled_sdf(
    compiled: &voxel_core::generators::graph::CompiledGraph,
    x: f32,
    y: f32,
    z: f32,
    scratch: &mut voxel_core::generators::graph::CompiledScratch,
) -> f32 {
    use voxel_core::generators::graph::{GraphInputs, GraphOutput};
    let xs = [x];
    let zs = [z];
    let inputs = GraphInputs { x: &xs, y, z: &zs };
    let mut out = Vec::new();
    compiled.generate_slice(&inputs, 1, scratch, &mut out, false);
    out.into_iter()
        .find(|(kind, _)| *kind == GraphOutput::Sdf)
        .and_then(|(_, values)| values.into_iter().next())
        .unwrap_or(f32::NAN)
}

/// Interval estimate of the graph's SDF output over the axis-aligned box with
/// corners `min_pos`/`max_pos` (corners may be given in any order). Returns
/// `(min, max)`; infinite bounds mean "not analysable" (hard nodes such as
/// noise fall back to a conservative full range). Engine-free for testing.
fn graph_debug_analyze_range(
    graph: &voxel_core::generators::graph::Graph,
    min_pos: [f32; 3],
    max_pos: [f32; 3],
) -> Result<(f32, f32), String> {
    use voxel_core::generators::graph::CompiledGraph;
    use voxel_core::math::Interval;
    if min_pos.iter().chain(max_pos.iter()).any(|v| !v.is_finite()) {
        return Err("box corners must be finite".to_owned());
    }
    let compiled =
        CompiledGraph::compile(graph).map_err(|error| graph_compile_error_report(&error).1)?;
    let interval = compiled.analyze_range(
        Interval::from_unordered(min_pos[0], max_pos[0]),
        Interval::from_unordered(min_pos[1], max_pos[1]),
        Interval::from_unordered(min_pos[2], max_pos[2]),
    );
    Ok((interval.min, interval.max))
}

/// Upper bound on ray-march samples in [`graph_raycast_sdf_approx`]. Keeps a
/// degenerate long-ray / tiny-stride combination from hanging the caller;
/// matching upstream exactly is not attempted.
const MAX_RAYCAST_SAMPLES: u32 = 65_536;

/// March the ray from `ray_origin` towards `ray_end` in steps of `stride`,
/// sampling the compiled SDF at each step.
///
/// * `Ok(Some(distance))` — first sample along the ray where the SDF is
///   negative; `distance` is measured from `ray_origin` and is accurate up to
///   `stride`.
/// * `Ok(None)` — the ray never entered matter (or the march hit the sample
///   cap; see [`MAX_RAYCAST_SAMPLES`]).
/// * `Err(message)` — invalid input (non-finite endpoints, zero-length ray,
///   non-positive stride) or the graph does not compile / has no SDF output.
///
/// Engine-free for unit testing.
fn graph_raycast_sdf_approx(
    graph: &voxel_core::generators::graph::Graph,
    ray_origin: [f32; 3],
    ray_end: [f32; 3],
    stride: f32,
) -> Result<Option<f32>, String> {
    use voxel_core::generators::graph::CompiledGraph;
    if ray_origin
        .iter()
        .chain(ray_end.iter())
        .any(|v| !v.is_finite())
    {
        return Err("ray endpoints must be finite".to_owned());
    }
    if !stride.is_finite() || stride <= 0.0 {
        return Err("stride must be finite and strictly positive".to_owned());
    }
    let delta = [
        ray_end[0] - ray_origin[0],
        ray_end[1] - ray_origin[1],
        ray_end[2] - ray_origin[2],
    ];
    let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    if length == 0.0 {
        return Err("ray has zero length".to_owned());
    }
    if !length.is_finite() {
        return Err("ray is too long (length is not finite)".to_owned());
    }
    let compiled =
        CompiledGraph::compile(graph).map_err(|error| graph_compile_error_report(&error).1)?;
    if !compiled.nodes().iter().any(|n| n.kind.is_output()) {
        return Err("graph has no SDF output node".to_owned());
    }
    let direction = [delta[0] / length, delta[1] / length, delta[2] / length];
    let samples = ((length / stride) + 1.0)
        .min(MAX_RAYCAST_SAMPLES as f32)
        .max(1.0) as u32;
    let mut scratch = voxel_core::generators::graph::CompiledScratch::new();
    for i in 0..samples {
        let d = (i as f32) * stride;
        if d > length {
            break;
        }
        let sdf = sample_compiled_sdf(
            &compiled,
            ray_origin[0] + direction[0] * d,
            ray_origin[1] + direction[1] * d,
            ray_origin[2] + direction[2] * d,
            &mut scratch,
        );
        if sdf < 0.0 {
            return Ok(Some(d));
        }
    }
    Ok(None)
}

/// Indicative microseconds-per-voxel timing of the compiled graph
/// (`debug_measure_microseconds_per_voxel`). With
/// `use_singular_queries = false`, times whole Y-slice generation; with
/// `true`, times one single-voxel query per voxel. Returns `-1.0` when the
/// graph does not compile or has no SDF output. Engine-free for testing.
fn graph_measure_microseconds_per_voxel(
    graph: &voxel_core::generators::graph::Graph,
    use_singular_queries: bool,
) -> f32 {
    use std::time::Instant;
    use voxel_core::generators::graph::{CompiledGraph, CompiledScratch, GraphInputs};
    let Ok(compiled) = CompiledGraph::compile(graph) else {
        return -1.0;
    };
    if !compiled.nodes().iter().any(|n| n.kind.is_output()) {
        return -1.0;
    }
    let mut scratch = CompiledScratch::new();
    let mut outputs = Vec::new();
    const SLICE_SIDE: usize = 32;
    const SLICE_VOXELS: usize = SLICE_SIDE * SLICE_SIDE;
    const SLICE_ITERATIONS: usize = 8;
    const SINGULAR_QUERIES: usize = 512;
    let xs: Vec<f32> = (0..SLICE_VOXELS).map(|i| (i % SLICE_SIDE) as f32).collect();
    let zs: Vec<f32> = (0..SLICE_VOXELS).map(|i| (i / SLICE_SIDE) as f32).collect();
    let start = Instant::now();
    let total_voxels = if use_singular_queries {
        for i in 0..SINGULAR_QUERIES {
            let x = [i as f32];
            let z = [(i / 3) as f32];
            let inputs = GraphInputs {
                x: &x,
                y: (i % 5) as f32,
                z: &z,
            };
            compiled.generate_slice(&inputs, 1, &mut scratch, &mut outputs, false);
        }
        SINGULAR_QUERIES
    } else {
        for _ in 0..SLICE_ITERATIONS {
            let inputs = GraphInputs {
                x: &xs,
                y: 0.0,
                z: &zs,
            };
            compiled.generate_slice(&inputs, SLICE_VOXELS, &mut scratch, &mut outputs, false);
        }
        SLICE_VOXELS * SLICE_ITERATIONS
    };
    let elapsed = start.elapsed();
    (elapsed.as_secs_f64() * 1.0e6 / total_voxels as f64) as f32
}

/// Finite-float validation for the canonical graph float property.
fn validate_graph_finite_float(value: f32) -> Result<f32, &'static str> {
    if !value.is_finite() {
        return Err("value must be finite");
    }
    Ok(value)
}

/// `subdivision_size` must be a positive power of two (upstream rejects other
/// values so block subdivision stays aligned).
fn validate_graph_subdivision_size(size: i64) -> Result<i64, &'static str> {
    if size < 2 || (size & (size - 1)) != 0 {
        return Err("value must be a power of two >= 2");
    }
    Ok(size)
}

/// `texture_mode` accepts exactly the two pinned `TextureMode` values.
fn validate_graph_texture_mode(mode: i64) -> Result<i64, &'static str> {
    if mode == VoxelGeneratorGraphGD::TEXTURE_MODE_MIXEL4
        || mode == VoxelGeneratorGraphGD::TEXTURE_MODE_SINGLE
    {
        return Ok(mode);
    }
    Err("value must be TEXTURE_MODE_MIXEL4 (0) or TEXTURE_MODE_SINGLE (1)")
}

/// Parse the compact interchange produced / accepted by `set_graph_json`.
/// Format: `{"nodes":[{"kind":"InputX"},{"kind":"Constant","value":10},
/// {"kind":"SdfSphere","a":0,"b":1,"c":2,"d":3},{"kind":"OutputSdf","a":4}]}`.
fn parse_graph_json(text: &str) -> Result<voxel_core::generators::graph::Graph, String> {
    use voxel_core::generators::graph::{node_kind_from_spec, Graph};
    let mut graph = Graph::new();
    let Some(nodes_start) = text.find('[') else {
        return Err("expected a JSON object with a nodes array".to_owned());
    };
    let Some(nodes_end) = text.rfind(']') else {
        return Err("nodes array is not closed".to_owned());
    };
    if nodes_end < nodes_start {
        return Err("nodes array is malformed".to_owned());
    }
    let body = &text[nodes_start + 1..nodes_end];
    for raw_object in body.split('}') {
        let object = raw_object.trim().trim_start_matches(',').trim();
        if object.is_empty() {
            continue;
        }
        let object = object.trim_start_matches('{').trim();
        let Some(kind) = json_string_field(object, "kind") else {
            return Err("each node must have a kind".to_owned());
        };
        let a = json_i64_field(object, "a").unwrap_or(-1);
        let b = json_i64_field(object, "b").unwrap_or(-1);
        let c = json_i64_field(object, "c").unwrap_or(-1);
        let d = json_i64_field(object, "d").unwrap_or(-1);
        let value = json_f32_field(object, "value").unwrap_or(0.0);
        if kind == "Expression" {
            let expr = json_string_field(object, "expr").unwrap_or_default();
            match voxel_core::generators::graph::expression_node::ExpressionNode::new(
                &expr,
                &[("x", 0), ("y", 1), ("z", 2)],
            ) {
                Ok(parsed) => {
                    graph.push(voxel_core::generators::graph::NodeKind::Expression {
                        x: voxel_core::generators::graph::optional_graph_port(a),
                        y: voxel_core::generators::graph::optional_graph_port(b),
                        z: voxel_core::generators::graph::optional_graph_port(c),
                        expr: std::sync::Arc::new(parsed),
                    });
                    continue;
                }
                Err(message) => return Err(message),
            }
        }
        let Some(node_kind) = node_kind_from_spec(&kind, a, b, c, d, value) else {
            return Err(format!("unknown node kind '{kind}'"));
        };
        graph.push(node_kind);
    }
    if graph.nodes().is_empty() {
        return Err("nodes array is empty".to_owned());
    }
    Ok(graph)
}

fn json_string_field(object: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let rest = object.split_once(&needle)?.1;
    let rest = rest.trim_start_matches(|c: char| c == ':' || c.is_whitespace());
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_owned())
}

fn json_i64_field(object: &str, key: &str) -> Option<i64> {
    json_number_token(object, key)?.parse().ok()
}

fn json_f32_field(object: &str, key: &str) -> Option<f32> {
    json_number_token(object, key)?.parse().ok()
}

fn json_number_token<'a>(object: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let rest = object.split_once(&needle)?.1;
    let rest = rest.trim_start_matches(|c: char| c == ':' || c.is_whitespace());
    let end = rest
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')
        })
        .unwrap_or(rest.len());
    let token = rest[..end].trim();
    if token.is_empty() {
        None
    } else {
        Some(token)
    }
}

#[cfg(test)]
mod graph_json_tests {
    use super::parse_graph_json;
    use voxel_core::generators::graph::NodeKind;

    #[test]
    fn parse_graph_json_reads_documented_node_list() {
        let graph = parse_graph_json(
            r#"{"nodes":[
                {"kind":"InputY"},
                {"kind":"SdfPlane","a":0},
                {"kind":"Constant","value":10},
                {"kind":"OutputSdf","a":1}
            ]}"#,
        )
        .expect("valid interchange");
        assert_eq!(graph.nodes().len(), 4);
        assert!(matches!(graph.nodes()[0].kind, NodeKind::InputY));
        assert!(matches!(graph.nodes()[2].kind, NodeKind::Constant(v) if v == 10.0));
        assert!(parse_graph_json(r#"{"nodes":[{"kind":"Nope"}]}"#).is_err());
    }
}

/// Engine-free tests for the pinned VoxelGeneratorGraph surface. All helpers
/// under test only touch voxel-core types, so no Godot runtime is needed.
#[cfg(test)]
mod graph_generator_tests {
    use super::{
        graph_compile_error_report, graph_debug_analyze_range,
        graph_measure_microseconds_per_voxel, graph_raycast_sdf_approx, sample_compiled_sdf,
        validate_graph_finite_float, validate_graph_subdivision_size, validate_graph_texture_mode,
        waves_preset_graph, VoxelGeneratorGraphGD,
    };
    use voxel_core::generators::graph::{
        CompiledGraph, CompiledScratch, Graph, GraphNode, GraphPort, NodeKind, TopoError,
    };

    /// Sphere SDF of radius `radius` centred at the origin, fed by the world
    /// coordinate inputs.
    fn sphere_graph(radius: f32) -> Graph {
        let mut graph = Graph::new();
        let nx = graph.push(NodeKind::InputX);
        let ny = graph.push(NodeKind::InputY);
        let nz = graph.push(NodeKind::InputZ);
        let nr = graph.push(NodeKind::Constant(radius));
        let sphere = graph.push(NodeKind::SdfSphere {
            x: Some(GraphPort::new(nx)),
            y: Some(GraphPort::new(ny)),
            z: Some(GraphPort::new(nz)),
            radius: Some(GraphPort::new(nr)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(sphere)),
        });
        graph
    }

    #[test]
    fn generator_graph_compile_report_maps_topology_errors_to_node_ids() {
        // A valid graph reports success through compile(), not the error map.
        assert!(CompiledGraph::compile(&sphere_graph(10.0)).is_ok());

        // Dangling port: node 0 reads from a node that does not exist (id 99).
        let mut dangling = Graph::new();
        dangling.push(NodeKind::Add {
            a: Some(GraphPort::new(99)),
            b: None,
        });
        let error = CompiledGraph::compile(&dangling).unwrap_err();
        assert_eq!(error, TopoError::DanglingPort(99));
        let (node_id, message) = graph_compile_error_report(&error);
        assert_eq!(node_id, 99, "dangling port should blame its node");
        // {id} in the message is the missing *referenced* node, so the text
        // must describe it as referenced-not-existing (node 0 is the node
        // holding the input; 99 is the one that does not exist).
        assert_eq!(message, "an input references node 99, which does not exist");

        // Cycle: nodes 1 and 2 reference each other; no single node is to
        // blame, so the pinned report uses -1.
        let mut cyclic = Graph::new();
        cyclic.add_node(GraphNode::new(
            1,
            NodeKind::Add {
                a: Some(GraphPort::new(2)),
                b: None,
            },
        ));
        cyclic.add_node(GraphNode::new(
            2,
            NodeKind::Add {
                a: Some(GraphPort::new(1)),
                b: None,
            },
        ));
        let error = CompiledGraph::compile(&cyclic).unwrap_err();
        assert_eq!(error, TopoError::Cycle);
        let (node_id, message) = graph_compile_error_report(&error);
        assert_eq!(node_id, -1, "cycle has no single offending node");
        assert!(!message.is_empty());
    }

    #[test]
    fn generator_graph_raycast_sdf_approx_hits_sphere_within_stride() {
        let graph = sphere_graph(10.0);
        // Straight through the sphere centre: the surface is at distance 20.
        let hit = graph_raycast_sdf_approx(&graph, [0.0, 0.0, -30.0], [0.0, 0.0, 30.0], 1.0)
            .expect("valid input");
        let distance = hit.expect("ray through the sphere must hit");
        assert!(
            (distance - 20.0).abs() <= 1.0,
            "hit distance {distance} should be within one stride of 20"
        );

        // Ray starting inside: immediate hit at distance 0.
        let inside = graph_raycast_sdf_approx(&graph, [0.0, 0.0, 0.0], [0.0, 0.0, 30.0], 1.0)
            .expect("valid input")
            .expect("ray starting inside must hit");
        assert_eq!(inside, 0.0);

        // Parallel ray that never enters the sphere: miss → Ok(None).
        let miss = graph_raycast_sdf_approx(&graph, [0.0, 0.0, 30.0], [0.0, 0.0, 60.0], 1.0)
            .expect("valid input");
        assert!(miss.is_none(), "ray outside the sphere must miss");

        // Invalid input: zero stride, zero-length ray, non-finite endpoints.
        assert!(
            graph_raycast_sdf_approx(&graph, [0.0, 0.0, -30.0], [0.0, 0.0, 30.0], 0.0).is_err()
        );
        assert!(
            graph_raycast_sdf_approx(&graph, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1.0).is_err(),
            "zero-length ray must be rejected"
        );
        assert!(
            graph_raycast_sdf_approx(&graph, [f32::NAN, 0.0, -30.0], [0.0, 0.0, 30.0], 1.0)
                .is_err()
        );
    }

    #[test]
    fn generator_graph_debug_analyze_range_returns_constant_interval() {
        // Constant(2) → OutputSdf: the interval is exactly (2, 2) everywhere.
        let mut constant = Graph::new();
        let two = constant.push(NodeKind::Constant(2.0));
        constant.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(two)),
        });
        let (min, max) = graph_debug_analyze_range(&constant, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
            .expect("constant graph analyses");
        assert_eq!((min, max), (2.0, 2.0));

        // Plane y = 0 straddling a box that crosses zero: min < 0 < max.
        let mut plane = Graph::new();
        let ny = plane.push(NodeKind::InputY);
        let zero = plane.push(NodeKind::Constant(0.0));
        let sdf = plane.push(NodeKind::SdfPlane {
            y: Some(GraphPort::new(ny)),
            height: Some(GraphPort::new(zero)),
        });
        plane.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(sdf)),
        });
        let (min, max) = graph_debug_analyze_range(&plane, [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
            .expect("plane graph analyses");
        assert!(min < 0.0, "plane over a straddling box must reach negative");
        assert!(max > 0.0, "plane over a straddling box must reach positive");

        // Non-finite corners are rejected.
        assert!(
            graph_debug_analyze_range(&plane, [f32::INFINITY, 0.0, 0.0], [1.0, 1.0, 1.0]).is_err()
        );
    }

    #[test]
    fn generator_graph_waves_preset_compiles_and_produces_wavy_sdf() {
        let graph = waves_preset_graph();
        let compiled = CompiledGraph::compile(&graph).expect("waves preset compiles");
        let mut scratch = CompiledScratch::new();
        // Preset: sdf = sin(0.1 * x) + cos(0.1 * z).
        let cases = [
            (0.0f32, 0.0f32, 0.0f32),
            (10.0, 0.0, 0.0),
            (0.0, 42.0, 10.0),
            (25.0, -7.0, 40.0),
        ];
        for (x, y, z) in cases {
            let expected = (0.1 * x).sin() + (0.1 * z).cos();
            let actual = sample_compiled_sdf(&compiled, x, y, z, &mut scratch);
            assert!(
                (actual - expected).abs() < 1e-5,
                "sdf at ({x},{y},{z}): {actual} vs {expected}"
            );
        }
        // The output must actually vary along X and Z (it is wavy, not flat),
        // and must be independent of Y.
        let flat = sample_compiled_sdf(&compiled, 0.0, 3.0, 0.0, &mut scratch);
        assert!((sample_compiled_sdf(&compiled, 15.0, 3.0, 0.0, &mut scratch) - flat).abs() > 0.1);
        assert!((sample_compiled_sdf(&compiled, 0.0, 3.0, 15.0, &mut scratch) - flat).abs() > 0.1);
        assert!(
            (sample_compiled_sdf(&compiled, 7.0, 3.0, 9.0, &mut scratch)
                - sample_compiled_sdf(&compiled, 7.0, -8.0, 9.0, &mut scratch))
            .abs()
                < 1e-6
        );
    }

    #[test]
    fn generator_graph_texture_mode_constants_match_pinned_values() {
        assert_eq!(VoxelGeneratorGraphGD::TEXTURE_MODE_MIXEL4, 0);
        assert_eq!(VoxelGeneratorGraphGD::TEXTURE_MODE_SINGLE, 1);
        // The texture-mode validator accepts exactly those two values.
        assert_eq!(validate_graph_texture_mode(0), Ok(0));
        assert_eq!(validate_graph_texture_mode(1), Ok(1));
        assert!(validate_graph_texture_mode(2).is_err());
        assert!(validate_graph_texture_mode(-1).is_err());
    }

    #[test]
    fn generator_graph_property_round_trips() {
        // subdivision_size: powers of two >= 2 pass, everything else fails.
        for valid in [2_i64, 4, 8, 16, 32, 64] {
            assert_eq!(validate_graph_subdivision_size(valid), Ok(valid));
        }
        for invalid in [0_i64, 1, 3, 5, 12, -8] {
            assert!(
                validate_graph_subdivision_size(invalid).is_err(),
                "{invalid} must be rejected"
            );
        }
        // sdf_clip_threshold: any finite real.
        assert_eq!(validate_graph_finite_float(1.5), Ok(1.5));
        assert_eq!(validate_graph_finite_float(-3.0), Ok(-3.0));
        assert!(validate_graph_finite_float(f32::NAN).is_err());
        assert!(validate_graph_finite_float(f32::INFINITY).is_err());

        // The timing helper returns a positive value for a usable graph and
        // -1.0 for one without an SDF output.
        let waves = waves_preset_graph();
        let slice_mode = graph_measure_microseconds_per_voxel(&waves, false);
        assert!(
            slice_mode >= 0.0 && slice_mode.is_finite(),
            "slice timing should be a finite non-negative float, got {slice_mode}"
        );
        let singular_mode = graph_measure_microseconds_per_voxel(&waves, true);
        assert!(
            singular_mode >= 0.0 && singular_mode.is_finite(),
            "singular timing should be a finite non-negative float, got {singular_mode}"
        );
        let mut no_output = Graph::new();
        no_output.push(NodeKind::InputX);
        assert_eq!(
            graph_measure_microseconds_per_voxel(&no_output, false),
            -1.0
        );
        assert_eq!(graph_measure_microseconds_per_voxel(&no_output, true), -1.0);
    }
}
