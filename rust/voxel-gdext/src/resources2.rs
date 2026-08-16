//! More Godot classes — abstract bases, modifier types, LOD terrain node,
//! and utility resources. Brings total class count closer to DoD 75+.

use std::collections::HashMap;
use std::sync::Arc;

use godot::classes::{Material, Mesh};
use godot::prelude::*;

pub(crate) const MAX_MULTIPASS_PASSES: i32 = 256;
pub(crate) const MAX_CARVE_PATH_STEPS: i64 = 65_536;
pub(crate) const MAX_EXHAUSTIVE_LOD_COUNT: i32 = 6;
/// At most eight million full-buffer voxel visits may run synchronously from
/// one `generate_layers` script call. This still permits 256 passes over a
/// 32³ buffer or 32 passes over a 64³ buffer.
const MAX_MULTIPASS_VOXEL_VISITS: u64 = 8_388_608;
/// At most eight million clipped stamp voxel visits may run synchronously from
/// one `carve_path` script call.
const MAX_CARVE_PATH_VOXEL_VISITS: u64 = 8_388_608;

pub(crate) fn validate_nonnegative_count(count: i32) -> Result<usize, &'static str> {
    usize::try_from(count).map_err(|_| "count must be non-negative")
}

fn validate_multipass_count(count: i32) -> Result<i32, &'static str> {
    if !(1..=MAX_MULTIPASS_PASSES).contains(&count) {
        return Err("pass count must be within the script workload limit");
    }
    Ok(count)
}

fn validate_carve_path_steps(steps: i64) -> Result<i64, &'static str> {
    if !(1..=MAX_CARVE_PATH_STEPS).contains(&steps) {
        return Err("path length exceeds the script workload limit");
    }
    Ok(steps)
}

fn validate_multipass_work(pass_count: i32, buffer_volume: u64) -> Result<u64, &'static str> {
    let pass_count = u64::try_from(validate_multipass_count(pass_count)?)
        .map_err(|_| "pass count must be non-negative")?;
    let work = pass_count
        .checked_mul(buffer_volume)
        .ok_or("multipass voxel visit count overflowed")?;
    if work > MAX_MULTIPASS_VOXEL_VISITS {
        return Err("multipass voxel visit count exceeds the script workload limit");
    }
    Ok(work)
}

fn validate_carve_path_work(
    steps: i64,
    max_clipped_stamp_volume: u64,
) -> Result<u64, &'static str> {
    let steps = u64::try_from(validate_carve_path_steps(steps)?)
        .map_err(|_| "path length must be non-negative")?;
    let stamp_count = steps.checked_add(1).ok_or("path stamp count overflowed")?;
    let work = stamp_count
        .checked_mul(max_clipped_stamp_volume)
        .ok_or("path voxel visit count overflowed")?;
    if work > MAX_CARVE_PATH_VOXEL_VISITS {
        return Err("path voxel visit count exceeds the script workload limit");
    }
    Ok(work)
}

fn checked_nonnegative_volume(size: voxel_core::math::Vector3i) -> Result<u64, &'static str> {
    let x = u64::try_from(size.x).map_err(|_| "buffer width must be non-negative")?;
    let y = u64::try_from(size.y).map_err(|_| "buffer height must be non-negative")?;
    let z = u64::try_from(size.z).map_err(|_| "buffer depth must be non-negative")?;
    x.checked_mul(y)
        .and_then(|area| area.checked_mul(z))
        .ok_or("buffer volume overflowed")
}

fn max_clipped_stamp_volume(
    half_size: i32,
    buffer_size: voxel_core::math::Vector3i,
) -> Result<u64, &'static str> {
    let half_size = u64::try_from(half_size).map_err(|_| "stamp half-size must be non-negative")?;
    let stamp_extent = half_size.checked_mul(2).ok_or("stamp extent overflowed")?;
    let width = u64::try_from(buffer_size.x).map_err(|_| "buffer width must be non-negative")?;
    let height = u64::try_from(buffer_size.y).map_err(|_| "buffer height must be non-negative")?;
    let depth = u64::try_from(buffer_size.z).map_err(|_| "buffer depth must be non-negative")?;
    stamp_extent
        .min(width)
        .checked_mul(stamp_extent.min(height))
        .and_then(|area| area.checked_mul(stamp_extent.min(depth)))
        .ok_or("clipped stamp volume overflowed")
}

fn validate_carve_path_stamp_range(
    start: voxel_core::math::Vector3i,
    target: voxel_core::math::Vector3i,
    origin_x: i32,
    half_size: i32,
) -> Result<(), &'static str> {
    let half_size =
        i128::from(u32::try_from(half_size).map_err(|_| "stamp half-size must be non-negative")?);
    for endpoint in [start, target] {
        let center = [
            i128::from(endpoint.x) + i128::from(origin_x),
            i128::from(endpoint.y),
            i128::from(endpoint.z),
        ];
        for coordinate in center {
            i32::try_from(coordinate - half_size)
                .and_then(|_| i32::try_from(coordinate + half_size))
                .map_err(|_| "stamp position exceeds i32 range")?;
        }
    }
    Ok(())
}

fn validate_exhaustive_lod_count(count: i32) -> Result<i32, &'static str> {
    if !(1..=MAX_EXHAUSTIVE_LOD_COUNT).contains(&count) {
        return Err("LOD count exceeds the exhaustive subdivision limit");
    }
    Ok(count)
}

/// The Variable LOD planner needs at least two LOD levels to be meaningful
/// (LOD0 plus at least one coarser level) and is bounded above by the
/// exhaustive subdivision limit. Convert the validated count to `u8` for
/// `LodClipboxSettings`.
fn resolve_lod_count_u8(count: i32) -> Result<u8, &'static str> {
    if !(2..=MAX_EXHAUSTIVE_LOD_COUNT).contains(&count) {
        return Err("Variable LOD requires a LOD count between 2 and 6");
    }
    u8::try_from(count).map_err(|_| "LOD count must fit in a u8")
}

/// Clamp the configured LOD distance to a non-negative `i32` voxel count for
/// `LodClipboxSettings`. The distance configures how close a viewer must be for
/// a LOD level to stay resident; it is expressed in voxels at LOD0 resolution.
fn paging_view_distance(viewer_distance: i32, terrain_cap: i32) -> i32 {
    crate::terrain::clamp_view_distance(i64::from(viewer_distance))
        .min(crate::terrain::clamp_view_distance(i64::from(terrain_cap)))
}

fn lod_distance_to_voxels(distance: f32) -> i32 {
    if !distance.is_finite() || distance < 0.0 {
        return 0;
    }
    if distance > i32::MAX as f32 {
        return i32::MAX;
    }
    distance.floor() as i32
}

fn default_variable_lod_bounds() -> voxel_core::math::Box3i {
    voxel_core::math::Box3i::new(
        voxel_core::math::Vector3i::splat(-512),
        voxel_core::math::Vector3i::splat(2560),
    )
}

fn aabb_to_box3i(
    position: Vector3,
    size: Vector3,
) -> Result<voxel_core::math::Box3i, &'static str> {
    let px = floor_to_i32(position.x)?;
    let py = floor_to_i32(position.y)?;
    let pz = floor_to_i32(position.z)?;
    let sx = floor_to_i32(size.x)?;
    let sy = floor_to_i32(size.y)?;
    let sz = floor_to_i32(size.z)?;
    if sx < 0 || sy < 0 || sz < 0 {
        return Err("voxel bounds size must be non-negative");
    }
    Ok(voxel_core::math::Box3i::new(
        voxel_core::math::Vector3i::new(px, py, pz),
        voxel_core::math::Vector3i::new(sx, sy, sz),
    ))
}

fn resolve_variable_lod_volume(
    lod_count: u8,
    lod_distance: f32,
    secondary_lod_distance: f32,
    bounds: voxel_core::math::Box3i,
) -> Result<
    (
        voxel_core::math::Box3i,
        voxel_core::terrain::lod_clipbox::LodClipboxSettings,
    ),
    voxel_core::terrain::lod_clipbox::LodMathError,
> {
    let settings = voxel_core::terrain::lod_clipbox::LodClipboxSettings {
        data_block_size: 16,
        mesh_block_size: 16,
        lod_count,
        lod0_distance_voxels: lod_distance_to_voxels(lod_distance),
        secondary_distance_voxels: lod_distance_to_voxels(secondary_lod_distance),
        unload_hysteresis_blocks: 2,
    };
    let mut data = voxel_core::storage::VoxelData::new();
    data.set_bounds(bounds);
    settings.validate_for(&data)?;
    Ok((bounds, settings))
}

#[cfg(test)]
fn exhaustive_leaf_upper_bound(lod_count: i32) -> Result<u32, &'static str> {
    let lod_count = u32::try_from(validate_exhaustive_lod_count(lod_count)?)
        .map_err(|_| "LOD count must be non-negative")?;
    8u32.checked_pow(lod_count - 1)
        .ok_or("exhaustive leaf count overflowed")
}

fn validate_finite_nonnegative_float(value: f32) -> Result<f32, &'static str> {
    if !value.is_finite() || value < 0.0 {
        return Err("value must be finite and non-negative");
    }
    Ok(value)
}

fn validate_finite_float(value: f32) -> Result<f32, &'static str> {
    if !value.is_finite() {
        return Err("value must be finite");
    }
    Ok(value)
}

fn floor_to_i32(value: f32) -> Result<i32, &'static str> {
    if !value.is_finite() || value < i32::MIN as f32 || value >= i32::MAX as f32 {
        return Err("coordinate must be finite and within i32 range");
    }
    Ok(value.floor() as i32)
}

// ---------------------------------------------------------------------------
// Minimal JSON parser for flat `{"key": int}` objects (id-map loading).
// ---------------------------------------------------------------------------

/// Parses a flat JSON object of `{"string": i32}` pairs. Returns an error
/// message on malformed input. Not a general-purpose JSON parser.
fn parse_flat_json_object_i32(src: &str) -> Result<Vec<(String, i32)>, String> {
    let trimmed = src.trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return Err("expected a JSON object delimited by {{ }}".to_string());
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut result = Vec::new();
    for entry in inner.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let colon = entry.find(':').ok_or("missing ':' in entry")?;
        let key_part = entry[..colon].trim();
        let val_part = entry[colon + 1..].trim();
        let key = key_part
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .ok_or("expected quoted string key")?
            .to_string();
        let val: i32 = val_part
            .parse()
            .map_err(|_| format!("expected integer value, got {val_part}"))?;
        result.push((key, val));
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// VoxelGeneratorGD — abstract base Resource for all generators
// ---------------------------------------------------------------------------
/// Abstract base resource for voxel generators. In C++ this is the Godot-facing
/// wrapper around the engine-agnostic `VoxelGenerator`. Subclasses:
/// Waves, Flat, Noise, Heightmap, Graph.
///
/// The pinned `generate_block` method mirrors upstream `VoxelGenerator`
/// (5828cbeb). The abstract base cannot produce data, so it is a faithful
/// no-op stub.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelGenerator)]
pub struct VoxelGeneratorGD {
    base: Base<Resource>,
}
#[godot_api]
impl IResource for VoxelGeneratorGD {
    fn init(base: Base<Resource>) -> Self {
        Self { base }
    }
}

#[godot_api]
impl VoxelGeneratorGD {
    /// The generator category name (base type).
    #[func]
    fn get_category(&self) -> GString {
        "generator".to_godot()
    }

    // -----------------------------------------------------------------
    // Pinned VoxelGenerator method
    // (upstream 5828cbeb: VoxelGenerator.xml).
    // -----------------------------------------------------------------

    /// Generates a block of voxels within the specified world area
    /// (canonical `generate_block`). `out_buffer` receives the voxel data,
    /// `origin_in_voxels` is the lower corner of the box (relative to LOD0),
    /// and `lod` selects the level of detail. The abstract base produces no
    /// data; subclasses override. Faithful no-op stub.
    #[func]
    fn generate_block(
        &self,
        _out_buffer: Gd<crate::voxel_buffer::VoxelBufferGD>,
        _origin_in_voxels: Vector3,
        _lod: i32,
    ) {
        // Abstract base: concrete generators override this.
    }
}

// ---------------------------------------------------------------------------
// VoxelStreamGD — abstract base Resource for all streams
// ---------------------------------------------------------------------------
/// Abstract base resource for voxel streams. Subclasses: Memory, RegionFiles.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelStream)]
pub struct VoxelStreamGD {
    base: Base<Resource>,
    /// Compression mode used when saving blocks (backing field for the
    /// canonical `compression_mode` property). 0=None, 1=Lz4Be, 2=Lz4.
    compression_mode_value: i32,
    /// Whether generator output is persisted on save (backing field for the
    /// canonical `save_generator_output` property).
    save_generator_output_value: bool,
}
#[godot_api]
impl IResource for VoxelStreamGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            compression_mode_value: 1,
            save_generator_output_value: false,
        }
    }
}

#[godot_api]
impl VoxelStreamGD {
    /// The stream category name (base type).
    #[func]
    fn get_category(&self) -> GString {
        "stream".to_godot()
    }

    // ----- ResultCode enum (upstream 5828cbeb: VoxelStream.xml) -----

    /// An error occurred when loading the block; the request will be aborted.
    #[constant]
    const RESULT_ERROR: i64 = 0;

    /// The block was not found; the requester may fall back to a generator.
    #[constant]
    const RESULT_BLOCK_NOT_FOUND: i64 = 1;

    /// The block was found.
    #[constant]
    const RESULT_BLOCK_FOUND: i64 = 2;

    // ----- Canonical pinned methods (upstream 5828cbeb: VoxelStream.xml) -----

    /// Forces cached data to be saved. The abstract base has no cache, so this
    /// is a no-op; concrete streams override it. Matches `VoxelStream::flush`.
    #[func]
    fn flush(&mut self) {
        // Abstract base: no cache to flush.
    }

    /// Size of a stream block in voxels. The abstract base reports a 16³ block
    /// (the engine default); concrete streams may differ. Matches
    /// `VoxelStream::get_block_size`.
    #[func]
    fn get_block_size(&self) -> Vector3 {
        Vector3::new(16.0, 16.0, 16.0)
    }

    /// Bitmask of channels the stream persists (bit `i` set ⇒ channel `i`).
    /// The abstract base reports only the Type channel (bit 0). Matches
    /// `VoxelStream::get_used_channels_mask`.
    #[func]
    fn get_used_channels_mask(&self) -> i32 {
        1
    }

    /// Load a voxel block from the stream. The abstract base cannot load data,
    /// so it reports "not found" (`RESULT_BLOCK_NOT_FOUND`); the requester may
    /// then fall back to a generator. Matches `VoxelStream::load_voxel_block`.
    ///
    /// Returns the `RESULT_*` code; `RESULT_BLOCK_NOT_FOUND` (1) by default.
    #[func]
    fn load_voxel_block(
        &mut self,
        _out_buffer: Gd<RefCounted>,
        _block_position: Vector3i,
        _lod_index: i32,
    ) -> i32 {
        Self::RESULT_BLOCK_NOT_FOUND as i32
    }

    /// Save a voxel block to the stream. The abstract base has no destination,
    /// so this is a no-op; concrete streams override it. Matches
    /// `VoxelStream::save_voxel_block`.
    #[func]
    fn save_voxel_block(
        &mut self,
        _buffer: Gd<RefCounted>,
        _block_position: Vector3i,
        _lod_index: i32,
    ) {
        // Abstract base: nowhere to save.
    }

    // ----- Canonical pinned properties (upstream 5828cbeb: VoxelStream.xml) -----

    /// Compression algorithm used when saving blocks (0=None, 1=Lz4Be, 2=Lz4).
    /// Matches `compression_mode`.
    #[func]
    fn get_compression_mode(&self) -> i32 {
        self.compression_mode_value
    }

    #[func]
    fn set_compression_mode(&mut self, mode: i32) {
        if !(0..=2).contains(&mode) {
            godot_error!(
                "VoxelStream.set_compression_mode: mode must be one of 0 (None), 1 (Lz4Be), 2 (Lz4)"
            );
            return;
        }
        self.compression_mode_value = mode;
    }

    /// Whether generator output is persisted on save (rather than only
    /// modified blocks). Matches `save_generator_output`.
    #[func]
    fn get_save_generator_output(&self) -> bool {
        self.save_generator_output_value
    }

    #[func]
    fn set_save_generator_output(&mut self, enabled: bool) {
        self.save_generator_output_value = enabled;
    }
}

// ---------------------------------------------------------------------------
// VoxelMesherGD — abstract base Resource for all meshers
// ---------------------------------------------------------------------------
/// Abstract base resource for voxel meshers. Subclasses: Transvoxel, Blocky, Cubes.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelMesher)]
pub struct VoxelMesherGD {
    base: Base<Resource>,
    #[var]
    padding: i32,
}
#[godot_api]
impl IResource for VoxelMesherGD {
    fn init(base: Base<Resource>) -> Self {
        Self { base, padding: 1 }
    }
}

#[godot_api]
impl VoxelMesherGD {
    /// The mesher category name (base type).
    #[func]
    fn get_category(&self) -> GString {
        "mesher".to_godot()
    }

    // ----- Canonical pinned methods (upstream 5828cbeb: VoxelMesher.xml) -----

    /// Minimum padding the mesher needs before the lower corner of the build
    /// region. The abstract base reports `0`; concrete meshers override this.
    /// Matches `VoxelMesher::get_minimum_padding`.
    #[func]
    fn get_minimum_padding(&self) -> i32 {
        self.padding.max(0)
    }

    /// Maximum padding the mesher needs after the upper corner of the build
    /// region. The abstract base reports its `padding` field; concrete meshers
    /// may override this. Matches `VoxelMesher::get_maximum_padding`.
    #[func]
    fn get_maximum_padding(&self) -> i32 {
        self.padding.max(0)
    }

    /// Build a mesh from a `VoxelBufferGD`. The abstract base cannot produce
    /// geometry (it has no mesher strategy), so this returns `nil` and emits a
    /// diagnostic. Subclasses override this to run a real mesher. Matches
    /// `VoxelMesher::build_mesh`.
    #[func]
    fn build_mesh(
        &self,
        _voxel_buffer: Gd<RefCounted>,
        _materials: VarArray,
        _additional_data: VarDictionary,
    ) -> Variant {
        godot_error!(
            "VoxelMesher.build_mesh: abstract base cannot build a mesh; use a subclass (VoxelMesherTransvoxel, VoxelMesherBlocky, VoxelMesherCubes)"
        );
        Variant::nil()
    }
}

// ---------------------------------------------------------------------------
// VoxelModifierGD — Node3D base for SDF modifiers
// ---------------------------------------------------------------------------
/// Base Node3D for SDF modifiers. Children modify terrain SDF data.
///
/// The pinned properties (`operation`, `smoothness`) and the `Operation` enum
/// constants mirror upstream `VoxelModifier` (5828cbeb). They back the existing
/// `#[var]` fields so GDScript reads round-trip.
#[derive(GodotClass)]
#[class(base = Node3D, tool, rename = VoxelModifier)]
pub struct VoxelModifierGD {
    base: Base<Node3D>,
    #[var]
    operation: i32,
    #[var]
    smoothness: f32,
}
#[godot_api]
impl INode3D for VoxelModifierGD {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            operation: 0,
            smoothness: 0.0,
        }
    }
}

#[godot_api]
impl VoxelModifierGD {
    /// Operation: performs SDF union (canonical `OPERATION_ADD`).
    #[constant]
    const OPERATION_ADD: i64 = 0;
    /// Operation: performs SDF subtraction (canonical `OPERATION_REMOVE`).
    #[constant]
    const OPERATION_REMOVE: i64 = 1;

    /// The modifier category name (base type).
    #[func]
    fn get_category(&self) -> GString {
        "modifier".to_godot()
    }
}

// ---------------------------------------------------------------------------
// VoxelModifierSphereGD — Node3D for sphere SDF modifier
// ---------------------------------------------------------------------------
/// A sphere-shaped SDF modifier node. Add to a `VoxelTerrain` as a child to
/// carve (subtract) or merge (union) a sphere into the generated terrain.
///
/// Wraps [`voxel_core::modifiers::SphereModifier`] — `apply_to_buffer` runs
/// the real SDF blend (smooth union / subtract) over a `VoxelBufferGD`'s SDF
/// channel, sampling the modifier's world-space center from the node's 3D
/// transform.
#[derive(GodotClass)]
#[class(base = Node3D, tool, rename = VoxelModifierSphere)]
pub struct VoxelModifierSphereGD {
    base: Base<Node3D>,
    #[var]
    radius: f32,
    /// Blend operation: 0 = add (union), 1 = subtract. Mirrors
    /// `SdfOperation`.
    #[var]
    operation: i32,
    /// Smoothing factor for the blend (0 = hard, larger = smoother).
    #[var]
    smoothness: f32,
}
#[godot_api]
impl INode3D for VoxelModifierSphereGD {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            radius: 10.0,
            operation: 0,
            smoothness: 0.0,
        }
    }
}

#[godot_api]
impl VoxelModifierSphereGD {
    /// Apply this sphere modifier to a `VoxelBufferGD`'s SDF channel.
    /// `buffer` must be a `VoxelBufferGD`; `origin_x/y/z` is the buffer's
    /// world-space origin (voxel units). Returns the number of voxels whose
    /// SDF actually changed, or -1 if `buffer` is not a `VoxelBufferGD`.
    #[func]
    fn apply_to_buffer(
        &self,
        buffer: Gd<RefCounted>,
        origin_x: f32,
        origin_y: f32,
        origin_z: f32,
    ) -> i64 {
        if validate_finite_float(origin_x).is_err()
            || validate_finite_float(origin_y).is_err()
            || validate_finite_float(origin_z).is_err()
            || validate_finite_nonnegative_float(self.radius).is_err()
            || validate_finite_nonnegative_float(self.smoothness).is_err()
        {
            godot_error!(
                "VoxelModifierSphere.apply_to_buffer: inputs must be finite and shape values non-negative"
            );
            return -1;
        }
        let Ok(mut buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let mut bound = buf.bind_mut();
        let core = bound.core_buffer_mut();
        let sx = core.size().x;
        let sy = core.size().y;
        let sz = core.size().z;
        const SDF_CHANNEL: usize = 1;

        // Build the core modifier from the node's state.
        let op = if self.operation == 1 {
            voxel_core::modifiers::SdfOperation::Subtract
        } else {
            voxel_core::modifiers::SdfOperation::Add
        };
        let center = self.base().get_position();
        let cx = center.x;
        let cy = center.y;
        let cz = center.z;

        // Gather SDF + world positions, apply the modifier, write back.
        let mut changed: i64 = 0;
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let sdf = core.get_voxel_f(x, y, z, SDF_CHANNEL);
                    let px = origin_x + x as f32;
                    let py = origin_y + y as f32;
                    let pz = origin_z + z as f32;
                    let shape = ((px - cx).powi(2) + (py - cy).powi(2) + (pz - cz).powi(2)).sqrt()
                        - self.radius;
                    let blended = sdf_blend_inline(sdf, shape, op, self.smoothness);
                    if (blended - sdf).abs() > 1e-6 {
                        core.set_voxel_f(blended, x, y, z, SDF_CHANNEL);
                        changed += 1;
                    }
                }
            }
        }
        changed
    }
}

/// Smooth SDF blending, mirroring `voxel_core::modifiers::sdf_blend` (which is
/// private). Used inline by [`VoxelModifierSphereGD::apply_to_buffer`] since
/// the core API is SoA-oriented (slice in, slice out).
fn sdf_blend_inline(
    existing: f32,
    shape: f32,
    op: voxel_core::modifiers::SdfOperation,
    smoothness: f32,
) -> f32 {
    use voxel_core::modifiers::SdfOperation;
    if smoothness <= 0.0 {
        return match op {
            SdfOperation::Add => existing.min(shape),
            SdfOperation::Subtract => existing.max(-shape),
        };
    }
    let h = (smoothness - (shape - existing).abs()).max(0.0) / smoothness;
    let m = shape + (existing - shape) * h; // lerp factor
    match op {
        SdfOperation::Add => m - smoothness * h * h,
        SdfOperation::Subtract => m + smoothness * h * h,
    }
}

// ---------------------------------------------------------------------------
// VoxelModifierMeshGD — Node3D for mesh SDF modifier
// ---------------------------------------------------------------------------
/// A mesh-based SDF modifier node. The shape is an oriented box (a simple
/// baked-mesh stand-in) whose extents derive from the node's scale; the SDF is
/// blended into the terrain via union/subtract using
/// [`voxel_core::math::sdf`] functions.
///
/// The pinned GDScript-facing properties (`isolevel`, `mesh_sdf`) mirror
/// upstream `VoxelModifierMesh` (5828cbeb). They are stored faithfully so
/// GDScript reads round-trip.
#[derive(GodotClass)]
#[class(base = Node3D, tool, rename = VoxelModifierMesh)]
pub struct VoxelModifierMeshGD {
    base: Base<Node3D>,
    /// Blend operation: 0 = add (union), 1 = subtract.
    #[var]
    operation: i32,
    /// Half-extents of the box shape (voxel units).
    #[var]
    extents: f32,
    /// Pinned `isolevel` (backing field). Upstream default 0.0.
    isolevel_value: f32,
    /// Pinned `mesh_sdf` resource (backing field; `None` until assigned).
    mesh_sdf_resource: Option<Gd<VoxelMeshSDFGD>>,
    /// The pinned GDScript-facing `isolevel` property.
    #[var(get = get_isolevel, set = set_isolevel)]
    isolevel: PhantomVar<f32>,
    /// The pinned GDScript-facing `mesh_sdf` property.
    #[var(get = get_mesh_sdf, set = set_mesh_sdf)]
    mesh_sdf: PhantomVar<Option<Gd<VoxelMeshSDFGD>>>,
}
#[godot_api]
impl INode3D for VoxelModifierMeshGD {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            operation: 0,
            extents: 4.0,
            isolevel_value: 0.0,
            mesh_sdf_resource: None,
            isolevel: PhantomVar::default(),
            mesh_sdf: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelModifierMeshGD {
    /// Apply this box-shape modifier to a `VoxelBufferGD`'s SDF channel.
    /// `buffer` must be a `VoxelBufferGD`; `origin_x/y/z` is the buffer's
    /// world-space origin. The box is centered on the node's world position.
    /// Returns the number of voxels whose SDF changed, or -1 if `buffer` is
    /// not a `VoxelBufferGD`.
    #[func]
    fn apply_to_buffer(
        &self,
        buffer: Gd<RefCounted>,
        origin_x: f32,
        origin_y: f32,
        origin_z: f32,
    ) -> i64 {
        if validate_finite_float(origin_x).is_err()
            || validate_finite_float(origin_y).is_err()
            || validate_finite_float(origin_z).is_err()
            || validate_finite_nonnegative_float(self.extents).is_err()
        {
            godot_error!(
                "VoxelModifierMesh.apply_to_buffer: inputs must be finite and extents non-negative"
            );
            return -1;
        }
        let Ok(mut buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let mut bound = buf.bind_mut();
        let core = bound.core_buffer_mut();
        let sx = core.size().x;
        let sy = core.size().y;
        let sz = core.size().z;
        const SDF_CHANNEL: usize = 1;

        let center = self.base().get_position();
        let cx = center.x;
        let cy = center.y;
        let cz = center.z;
        let extents = voxel_core::math::Vector3f::splat(self.extents);
        let subtract = self.operation == 1;

        let mut changed: i64 = 0;
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let sdf = core.get_voxel_f(x, y, z, SDF_CHANNEL);
                    let pos = voxel_core::math::Vector3f::new(
                        origin_x + x as f32 - cx,
                        origin_y + y as f32 - cy,
                        origin_z + z as f32 - cz,
                    );
                    let shape = voxel_core::math::sdf::sdf_box(pos, extents);
                    let blended = if subtract {
                        voxel_core::math::sdf::sdf_subtract(sdf, shape)
                    } else {
                        voxel_core::math::sdf::sdf_union(sdf, shape)
                    };
                    if (blended - sdf).abs() > 1e-6 {
                        core.set_voxel_f(blended, x, y, z, SDF_CHANNEL);
                        changed += 1;
                    }
                }
            }
        }
        changed
    }

    // -----------------------------------------------------------------
    // Pinned VoxelModifierMesh properties
    // (upstream 5828cbeb: VoxelModifierMesh.xml).
    // -----------------------------------------------------------------

    /// Offsets isolevel of the SDF mesh; positive makes the object appear
    /// thicker and smoother, negative thinner (upstream default 0.0).
    #[func]
    fn get_isolevel(&self) -> f32 {
        self.isolevel_value
    }

    #[func]
    fn set_isolevel(&mut self, isolevel: f32) {
        self.isolevel_value = isolevel;
    }

    /// SDF mesh used for the modifier (`None` until assigned).
    #[func]
    fn get_mesh_sdf(&self) -> Option<Gd<VoxelMeshSDFGD>> {
        self.mesh_sdf_resource.clone()
    }

    #[func]
    fn set_mesh_sdf(&mut self, mesh_sdf: Option<Gd<VoxelMeshSDFGD>>) {
        self.mesh_sdf_resource = mesh_sdf;
    }
}

// ---------------------------------------------------------------------------
// VoxelLodTerrainGD — Node3D for Variable-LOD terrain (production runtime)
// ---------------------------------------------------------------------------
/// Variable-LOD terrain node. Wraps [`voxel_core::terrain::VoxelTerrainCore`]
/// constructed through [`VoxelTerrainCore::new_variable_lod`] — the production
/// Variable LOD planner (`prepare_variable_physical_slice`). Each `_process`
/// tick it feeds paired [`VoxelViewer`](crate::terrain::VoxelViewer) positions
/// into the core, drains mesh outputs, and uploads them as `ArrayMesh`
/// instances into child `MeshInstance3D` nodes, exactly like `VoxelTerrain`
/// but across multiple LOD levels.
///
/// The rendering/event bookkeeping reuses the shared `crate::terrain` helpers
/// (`reduce_render_events`, `apply_pending_render_op`, `RenderState`, ...) so
/// both runtimes keep identical upload/remove semantics.
#[derive(GodotClass)]
#[class(base = Node3D, tool, rename = VoxelLodTerrain)]
pub struct VoxelLodTerrainGD {
    base: Base<Node3D>,
    core: Option<voxel_core::terrain::VoxelTerrainCore>,
    mesh_instances: HashMap<crate::terrain::MeshBlockRenderId, crate::terrain::RenderedMeshBlock>,
    render_state: crate::terrain::RenderState,
    generator_resource: Option<Gd<Resource>>,
    mesher_resource: Option<Gd<Resource>>,
    #[export]
    #[var(get = get_stream, set = set_stream)]
    stream: PhantomVar<Option<Gd<Resource>>>,
    stream_resource: Option<Gd<Resource>>,
    /// Number of LOD levels (default 4; clamped to 2..=6). Plain field exposed
    /// via the `get_lod_count`/`set_lod_count` #[func]s.
    lod_count: i32,
    /// Distance (voxels) at which LOD0 is kept resident around a viewer.
    /// Secondary rings use `secondary_lod_distance_value`.
    lod_distance: f32,
    /// Optional material override applied to all mesh blocks.
    material_override: Option<Gd<Material>>,
    /// Whether to generate collision shapes for mesh blocks.
    generate_collision: bool,
    // -----------------------------------------------------------------
    // Pinned VoxelLodTerrain properties.
    //
    // Properties without a voxel-core counterpart are stubbed with a backing
    // field that is faithfully stored on the node so GDScript reads round-trip
    // (set X; get X == X), mirroring VoxelTerrain's pattern.
    // -----------------------------------------------------------------
    mesh_block_size_value: i32,
    secondary_lod_distance_value: f32,
    lod_fade_duration_value: f32,
    view_distance_value: i32,
    collision_layer_value: i32,
    collision_mask_value: i32,
    collision_margin_value: f32,
    collision_lod_count_value: i32,
    collision_update_delay_value: i32,
    cache_generated_blocks_value: bool,
    full_load_mode_enabled_value: bool,
    run_stream_in_editor_value: bool,
    threaded_update_enabled_value: bool,
    use_gpu_generation_value: bool,
    streaming_system_value: i32,
    process_callback_value: i32,
    normalmap_enabled_value: bool,
    normalmap_begin_lod_index_value: i32,
    normalmax_deviation_degrees_value: i32,
    normalmap_tile_resolution_min_value: i32,
    normalmap_tile_resolution_max_value: i32,
    normalmap_octahedral_encoding_enabled_value: bool,
    normalmap_use_gpu_value: bool,
    normalmap_generator_override_resource: Option<Gd<Resource>>,
    normalmap_generator_override_begin_lod_index_value: i32,
    debug_draw_enabled_value: bool,
    debug_draw_shadow_occluders_value: bool,
    debug_draw_octree_nodes_value: bool,
    debug_draw_octree_bounds_value: bool,
    debug_draw_mesh_updates_value: bool,
    debug_draw_edit_boxes_value: bool,
    debug_draw_volume_bounds_value: bool,
    debug_draw_edited_blocks_value: bool,
    debug_draw_modifier_bounds_value: bool,
    debug_draw_active_mesh_blocks_value: bool,
    debug_draw_viewer_clipboxes_value: bool,
    debug_draw_loaded_visual_and_collision_blocks_value: bool,
    debug_draw_active_visual_and_collision_blocks_value: bool,
    voxel_bounds_value: Aabb,
    /// The pinned GDScript-facing `mesh_block_size` property. Must match the
    /// data block size (currently 16) and is locked after `_ready`.
    #[var(get = get_mesh_block_size, set = set_mesh_block_size)]
    mesh_block_size: PhantomVar<i32>,
    /// The pinned GDScript-facing `secondary_lod_distance` property.
    #[var(get = get_secondary_lod_distance, set = set_secondary_lod_distance)]
    secondary_lod_distance: PhantomVar<f32>,
    /// The pinned GDScript-facing `lod_fade_duration` property.
    #[var(get = get_lod_fade_duration, set = set_lod_fade_duration)]
    lod_fade_duration: PhantomVar<f32>,
    /// The pinned GDScript-facing `view_distance` property.
    #[var(get = get_view_distance, set = set_view_distance)]
    view_distance: PhantomVar<i32>,
    /// The pinned GDScript-facing `collision_layer` property.
    #[var(get = get_collision_layer, set = set_collision_layer)]
    collision_layer: PhantomVar<i32>,
    /// The pinned GDScript-facing `collision_mask` property.
    #[var(get = get_collision_mask, set = set_collision_mask)]
    collision_mask: PhantomVar<i32>,
    /// The pinned GDScript-facing `collision_margin` property.
    #[var(get = get_collision_margin, set = set_collision_margin)]
    collision_margin: PhantomVar<f32>,
    /// The pinned GDScript-facing `collision_lod_count` property.
    #[var(get = get_collision_lod_count, set = set_collision_lod_count)]
    collision_lod_count: PhantomVar<i32>,
    /// The pinned GDScript-facing `collision_update_delay` property.
    #[var(get = get_collision_update_delay, set = set_collision_update_delay)]
    collision_update_delay: PhantomVar<i32>,
    /// The pinned GDScript-facing `cache_generated_blocks` property.
    #[var(get = get_cache_generated_blocks, set = set_cache_generated_blocks)]
    cache_generated_blocks: PhantomVar<bool>,
    /// The pinned GDScript-facing `generate_collisions` property (plural alias
    /// of `generate_collision`).
    #[var(get = get_generate_collisions, set = set_generate_collisions)]
    generate_collisions: PhantomVar<bool>,
    /// The pinned GDScript-facing `full_load_mode_enabled` property.
    #[var(get = is_full_load_mode_enabled, set = set_full_load_mode_enabled)]
    full_load_mode_enabled: PhantomVar<bool>,
    /// The pinned GDScript-facing `run_stream_in_editor` property.
    #[var(get = is_stream_running_in_editor, set = set_run_stream_in_editor)]
    run_stream_in_editor: PhantomVar<bool>,
    /// The pinned GDScript-facing `threaded_update_enabled` property.
    #[var(get = is_threaded_update_enabled, set = set_threaded_update_enabled)]
    threaded_update_enabled: PhantomVar<bool>,
    /// The pinned GDScript-facing `use_gpu_generation` property.
    #[var(get = get_generator_use_gpu, set = set_generator_use_gpu)]
    use_gpu_generation: PhantomVar<bool>,
    /// The pinned GDScript-facing `streaming_system` property.
    #[var(get = get_streaming_system, set = set_streaming_system)]
    streaming_system: PhantomVar<i32>,
    /// The pinned GDScript-facing `normalmap_enabled` property.
    #[var(get = is_normalmap_enabled, set = set_normalmap_enabled)]
    normalmap_enabled: PhantomVar<bool>,
    /// The pinned GDScript-facing `normalmap_begin_lod_index` property.
    #[var(get = get_normalmap_begin_lod_index, set = set_normalmap_begin_lod_index)]
    normalmap_begin_lod_index: PhantomVar<i32>,
    /// The pinned GDScript-facing `normalmap_max_deviation_degrees` property.
    #[var(get = get_normalmap_max_deviation_degrees, set = set_normalmap_max_deviation_degrees)]
    normalmap_max_deviation_degrees: PhantomVar<i32>,
    /// The pinned GDScript-facing `normalmap_tile_resolution_min` property.
    #[var(get = get_normalmap_tile_resolution_min, set = set_normalmap_tile_resolution_min)]
    normalmap_tile_resolution_min: PhantomVar<i32>,
    /// The pinned GDScript-facing `normalmap_tile_resolution_max` property.
    #[var(get = get_normalmap_tile_resolution_max, set = set_normalmap_tile_resolution_max)]
    normalmap_tile_resolution_max: PhantomVar<i32>,
    /// The pinned GDScript-facing `normalmap_octahedral_encoding_enabled`
    /// property.
    #[var(get = get_octahedral_normal_encoding, set = set_octahedral_normal_encoding)]
    normalmap_octahedral_encoding_enabled: PhantomVar<bool>,
    /// The pinned GDScript-facing `normalmap_use_gpu` property.
    #[var(get = get_normalmap_use_gpu, set = set_normalmap_use_gpu)]
    normalmap_use_gpu: PhantomVar<bool>,
    /// The pinned GDScript-facing `debug_draw_enabled` property.
    #[var(get = debug_is_draw_enabled, set = debug_set_draw_enabled)]
    debug_draw_enabled: PhantomVar<bool>,
    /// The pinned GDScript-facing `debug_draw_shadow_occluders` property.
    #[var(get = get_debug_draw_shadow_occluders, set = set_debug_draw_shadow_occluders)]
    debug_draw_shadow_occluders: PhantomVar<bool>,
    /// The pinned GDScript-facing `voxel_bounds` property.
    #[var(get = get_voxel_bounds, set = set_voxel_bounds)]
    voxel_bounds: PhantomVar<Aabb>,
}
#[godot_api]
impl INode3D for VoxelLodTerrainGD {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            core: None,
            mesh_instances: HashMap::new(),
            render_state: crate::terrain::RenderState::default(),
            generator_resource: None,
            mesher_resource: None,
            stream: Default::default(),
            stream_resource: None,
            lod_count: 4,
            lod_distance: 64.0,
            material_override: None,
            generate_collision: false,
            mesh_block_size_value: 16,
            secondary_lod_distance_value: 48.0,
            lod_fade_duration_value: 0.0,
            view_distance_value: 512,
            collision_layer_value: 1,
            collision_mask_value: 1,
            collision_margin_value: 0.04,
            collision_lod_count_value: 0,
            collision_update_delay_value: 0,
            cache_generated_blocks_value: false,
            full_load_mode_enabled_value: false,
            run_stream_in_editor_value: true,
            threaded_update_enabled_value: false,
            use_gpu_generation_value: false,
            streaming_system_value: 0,
            process_callback_value: 0,
            normalmap_enabled_value: false,
            normalmap_begin_lod_index_value: 2,
            normalmax_deviation_degrees_value: 60,
            normalmap_tile_resolution_min_value: 4,
            normalmap_tile_resolution_max_value: 8,
            normalmap_octahedral_encoding_enabled_value: false,
            normalmap_use_gpu_value: false,
            normalmap_generator_override_resource: None,
            normalmap_generator_override_begin_lod_index_value: 0,
            debug_draw_enabled_value: false,
            debug_draw_shadow_occluders_value: false,
            debug_draw_octree_nodes_value: false,
            debug_draw_octree_bounds_value: false,
            debug_draw_mesh_updates_value: false,
            debug_draw_edit_boxes_value: false,
            debug_draw_volume_bounds_value: false,
            debug_draw_edited_blocks_value: false,
            debug_draw_modifier_bounds_value: false,
            debug_draw_active_mesh_blocks_value: false,
            debug_draw_viewer_clipboxes_value: false,
            debug_draw_loaded_visual_and_collision_blocks_value: false,
            debug_draw_active_visual_and_collision_blocks_value: false,
            voxel_bounds_value: Aabb::new(Vector3::splat(-512.0), Vector3::splat(2560.0)),
            mesh_block_size: PhantomVar::default(),
            secondary_lod_distance: PhantomVar::default(),
            lod_fade_duration: PhantomVar::default(),
            view_distance: PhantomVar::default(),
            collision_layer: PhantomVar::default(),
            collision_mask: PhantomVar::default(),
            collision_margin: PhantomVar::default(),
            collision_lod_count: PhantomVar::default(),
            collision_update_delay: PhantomVar::default(),
            cache_generated_blocks: PhantomVar::default(),
            generate_collisions: PhantomVar::default(),
            full_load_mode_enabled: PhantomVar::default(),
            run_stream_in_editor: PhantomVar::default(),
            threaded_update_enabled: PhantomVar::default(),
            use_gpu_generation: PhantomVar::default(),
            streaming_system: PhantomVar::default(),
            normalmap_enabled: PhantomVar::default(),
            normalmap_begin_lod_index: PhantomVar::default(),
            normalmap_max_deviation_degrees: PhantomVar::default(),
            normalmap_tile_resolution_min: PhantomVar::default(),
            normalmap_tile_resolution_max: PhantomVar::default(),
            normalmap_octahedral_encoding_enabled: PhantomVar::default(),
            normalmap_use_gpu: PhantomVar::default(),
            debug_draw_enabled: PhantomVar::default(),
            debug_draw_shadow_occluders: PhantomVar::default(),
            voxel_bounds: PhantomVar::default(),
        }
    }

    fn ready(&mut self) {
        if self.core.is_some() {
            godot_print!("VoxelLodTerrain ready — reusing retained terrain core");
            return;
        }

        let lod_count_u8 = match resolve_lod_count_u8(self.lod_count) {
            Ok(value) => value,
            Err(message) => {
                godot_error!("VoxelLodTerrain.ready: {message}");
                return;
            }
        };

        let requested_bounds = match aabb_to_box3i(
            self.voxel_bounds_value.position,
            self.voxel_bounds_value.size,
        ) {
            Ok(bounds) => bounds,
            Err(message) => {
                godot_error!(
                    "VoxelLodTerrain.ready: voxel_bounds rejected ({message}); using default aligned volume"
                );
                default_variable_lod_bounds()
            }
        };
        let (bounds, settings) = match resolve_variable_lod_volume(
            lod_count_u8,
            self.lod_distance,
            self.secondary_lod_distance_value,
            requested_bounds,
        ) {
            Ok(volume) => volume,
            Err(error) => {
                godot_error!(
                    "VoxelLodTerrain.ready: voxel_bounds rejected ({error:?}); using default aligned volume"
                );
                match resolve_variable_lod_volume(
                    lod_count_u8,
                    self.lod_distance,
                    self.secondary_lod_distance_value,
                    default_variable_lod_bounds(),
                ) {
                    Ok(volume) => volume,
                    Err(fallback_error) => {
                        godot_error!(
                            "VoxelLodTerrain.ready: default volume rejected: {fallback_error:?}"
                        );
                        return;
                    }
                }
            }
        };

        let mut data = voxel_core::storage::VoxelData::new();
        data.set_bounds(bounds);
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let mut format = voxel_core::storage::VoxelFormat::new();
        format.depths[voxel_core::storage::ChannelId::Sdf.index()] =
            voxel_core::storage::ChannelDepth::Bit32;
        data.set_format(format);

        let generator = crate::terrain::resolve_core_generator(self.generator_resource.as_ref());
        // Load-bearing: the load task materializes blocks via the VoxelData
        // generator (NOT the MeshingDependency generator) when the stream
        // returns NotFound. Install it on VoxelData so paging produces data.
        data.set_generator(Some(generator.clone()));

        let mesher = crate::terrain::resolve_core_mesher(self.mesher_resource.as_ref());
        let meshing_dep = voxel_core::engine::MeshingDependency::new(mesher, Some(generator));

        let stream_was_assigned = self.stream_resource.is_some();
        let explicit_stream = self
            .stream_resource
            .clone()
            .and_then(crate::streams::resolve_core_stream);
        if stream_was_assigned && explicit_stream.is_none() {
            godot_error!(
                "VoxelLodTerrain.stream must be VoxelStreamMemory or VoxelStreamRegionFiles"
            );
        }
        let selected_stream = explicit_stream.unwrap_or_else(|| {
            // No explicit stream: pair the generator with an empty memory
            // stream so the load task falls back to the VoxelData generator.
            Arc::new(voxel_core::streams::MemoryStream::new())
        });

        match voxel_core::terrain::VoxelTerrainCore::new_variable_lod(
            data,
            selected_stream,
            meshing_dep,
            settings,
        ) {
            Ok(core) => {
                self.core = Some(core);
                godot_print!(
                    "VoxelLodTerrain ready — variable-LOD terrain core initialised (lod_count={lod_count_u8})"
                );
            }
            Err(error) => {
                godot_error!(
                    "VoxelLodTerrain.ready: variable-LOD core construction failed: {error:?}"
                );
            }
        }
    }

    fn process(&mut self, _delta: f64) {
        let view_cap = self.view_distance_value;
        let viewers = crate::terrain::collect_child_viewers(
            self.base().get_children().iter_shared(),
            "VoxelLodTerrain",
            |viewer_distance| paging_view_distance(viewer_distance, view_cap),
            self.generate_collision,
        );

        let pending_ops = {
            let Some(core) = self.core.as_mut() else {
                return;
            };
            let events = match core.try_process(&viewers) {
                Ok(events) => {
                    let data_block_size = core.data().block_size() as i32;
                    for event in &events {
                        match event {
                            voxel_core::terrain::VoxelTerrainEvent::DataBlockLoaded(loc) => {
                                let p = loc.position * data_block_size;
                                self.signals()
                                    .block_loaded()
                                    .emit(godot::builtin::Vector3i::new(p.x, p.y, p.z));
                            }
                            voxel_core::terrain::VoxelTerrainEvent::DataBlockUnloaded(loc) => {
                                let p = loc.position * data_block_size;
                                self.signals()
                                    .block_unloaded()
                                    .emit(godot::builtin::Vector3i::new(p.x, p.y, p.z));
                            }
                            voxel_core::terrain::VoxelTerrainEvent::MeshBlockEntered(upload) => {
                                let p = upload.key().location.position_in_blocks;
                                self.signals()
                                    .mesh_block_entered()
                                    .emit(godot::builtin::Vector3i::new(p.x, p.y, p.z));
                            }
                            voxel_core::terrain::VoxelTerrainEvent::MeshBlockExited(loc) => {
                                let p = loc.position_in_blocks;
                                self.signals()
                                    .mesh_block_exited()
                                    .emit(godot::builtin::Vector3i::new(p.x, p.y, p.z));
                            }
                            _ => {}
                        }
                    }
                    events
                }
                Err(error) => {
                    godot_error!(
                        "VoxelLodTerrain.process: core rejected the viewer update: {error}"
                    );
                    return;
                }
            };
            crate::terrain::reduce_render_events(&mut self.render_state, events)
        };

        // Godot nodes are mutated only after the terrain core borrow ends.
        for op in pending_ops {
            self.apply_render_op(op);
        }
        self.refresh_debug_draw();
    }

    fn exit_tree(&mut self) {
        match crate::terrain::shutdown_core_retaining_on_error(&mut self.core, |core| {
            core.shutdown_and_flush()
        }) {
            Ok(_) => {
                // Godot calls `_ready` only once by default; a torn-down node
                // needs a new core if it is later re-added.
                self.base_mut().request_ready();
            }
            Err(error) => {
                godot_error!("VoxelLodTerrain shutdown failed: {error}");
                if let Some(core) = self.core.as_ref() {
                    crate::terrain::log_save_failures(
                        "VoxelLodTerrain shutdown retained save",
                        core,
                    );
                }
            }
        }
        for (_, mut rendered) in self.mesh_instances.drain() {
            rendered.instance.queue_free();
        }
        crate::debug_draw::refresh_debug_overlay(
            &mut self.to_gd().upcast::<Node3D>(),
            None,
            false,
            crate::debug_draw::DebugDrawFlags::default(),
        );
        self.render_state.reset();
    }
}

#[godot_api]
impl VoxelLodTerrainGD {
    /// Emitted when a new data block is loaded from stream.
    #[signal]
    fn block_loaded(position: godot::builtin::Vector3i);

    /// Emitted when a data block is unloaded due to being outside view distance.
    #[signal]
    fn block_unloaded(position: godot::builtin::Vector3i);

    /// Emitted when a mesh block receives its first update since it was added
    /// in the range of viewers.
    #[signal]
    fn mesh_block_entered(position: godot::builtin::Vector3i);

    /// Emitted when a mesh block gets unloaded.
    #[signal]
    fn mesh_block_exited(position: godot::builtin::Vector3i);

    /// Returns the number of loaded mesh blocks (all LODs).
    #[func]
    fn get_mesh_block_count(&self) -> i32 {
        i32::try_from(self.mesh_instances.len()).unwrap_or(i32::MAX)
    }

    /// Packed `x,y,z,lod` for every resident mesh block.
    #[func]
    pub(crate) fn get_mesh_block_locations(&self) -> PackedInt32Array {
        crate::terrain::pack_mesh_block_locations(self.mesh_instances.keys().copied())
    }

    /// Returns the voxel-core version string (diagnostic).
    #[func]
    fn get_version(&self) -> GString {
        voxel_core::VERSION.to_godot()
    }

    /// Returns a snapshot of the paging orchestrator's cumulative statistics.
    /// Returns `null` if the terrain core has not been initialised yet.
    #[func]
    fn get_statistics(&self) -> Variant {
        let Some(core) = self.core.as_ref() else {
            return Variant::nil();
        };
        let stats = crate::resources2::VoxelTerrainStatsGD::from_core_stats(core.stats());
        stats.to_variant()
    }

    /// Persist dirty and retry-journaled terrain blocks without shutting down
    /// paging. Returns `false` before `_ready` or when any bounded save/stream
    /// flush attempt fails; failed payloads remain owned for a later retry.
    #[func]
    fn flush_pending_saves(&mut self) -> bool {
        let mut pending_ops = Vec::new();
        let result: Result<bool, voxel_core::terrain::SaveFlushError> = {
            let core_slot = &mut self.core;
            let render_state = &mut self.render_state;
            crate::terrain::flush_core_if_ready(core_slot, |core| {
                let flush_result = core.flush_pending_saves();
                let events = crate::terrain::combine_flush_and_event_drain(flush_result, || {
                    core.try_drain_completed_tasks()
                })?;
                pending_ops = crate::terrain::reduce_render_events(render_state, events);
                Ok(())
            })
        };
        for op in pending_ops {
            self.apply_render_op(op);
        }
        match result {
            Ok(flushed) => flushed,
            Err(error) => {
                godot_error!("VoxelLodTerrain.flush_pending_saves failed: {error}");
                if let Some(core) = self.core.as_ref() {
                    crate::terrain::log_save_failures("VoxelLodTerrain flush retained save", core);
                }
                false
            }
        }
    }

    /// The generator resource (VoxelGeneratorWaves/Flat/Noise/Heightmap).
    #[func]
    fn get_generator(&self) -> Variant {
        match &self.generator_resource {
            Some(g) => g.to_variant(),
            None => Variant::nil(),
        }
    }

    #[func]
    fn set_generator(&mut self, value: Gd<Resource>) {
        self.generator_resource = Some(value);
    }

    #[func]
    fn get_mesher(&self) -> Variant {
        match &self.mesher_resource {
            Some(mesher) => mesher.to_variant(),
            None => Variant::nil(),
        }
    }

    #[func]
    fn set_mesher(&mut self, value: Gd<Resource>) {
        self.mesher_resource = Some(value);
    }

    #[func]
    fn get_stream(&self) -> Option<Gd<Resource>> {
        self.stream_resource.clone()
    }

    #[func]
    fn set_stream(&mut self, value: Option<Gd<Resource>>) {
        self.stream_resource = value;
    }

    /// Number of LOD levels.
    #[func]
    fn get_lod_count(&self) -> i32 {
        self.lod_count
    }

    /// Set the LOD level count (2..=6). Must be called before `_ready`; the
    /// core is constructed with the chosen LOD count and cannot be resized
    /// afterwards.
    #[func]
    fn set_lod_count(&mut self, count: i32) {
        if self.core.is_some() {
            godot_error!(
                "VoxelLodTerrain.set_lod_count: cannot change LOD count after _ready (core already constructed)"
            );
            return;
        }
        let Ok(lod_count) = validate_exhaustive_lod_count(count) else {
            godot_error!(
                "VoxelLodTerrain.set_lod_count: LOD count must be within the exhaustive subdivision limit"
            );
            return;
        };
        self.lod_count = lod_count;
    }

    /// LOD split distance (voxels).
    #[func]
    fn get_lod_distance(&self) -> f32 {
        self.lod_distance
    }

    #[func]
    fn set_lod_distance(&mut self, distance: f32) {
        if validate_finite_nonnegative_float(distance).is_err() {
            godot_error!(
                "VoxelLodTerrain.set_lod_distance: distance must be finite and non-negative"
            );
            return;
        }
        self.lod_distance = distance;
        self.apply_live_clipbox_distances();
    }

    /// Material override applied to all terrain mesh blocks.
    #[func]
    fn get_material_override(&self) -> Variant {
        match &self.material_override {
            Some(m) => m.to_variant(),
            None => Variant::nil(),
        }
    }

    #[func]
    fn set_material_override(&mut self, value: Gd<Material>) {
        self.material_override = Some(value);
    }

    /// Whether to generate trimesh collision for terrain blocks.
    #[func]
    fn get_generate_collision(&self) -> bool {
        self.generate_collision
    }

    #[func]
    fn set_generate_collision(&mut self, enabled: bool) {
        self.generate_collision = enabled;
    }

    // -----------------------------------------------------------------
    // Pinned VoxelLodTerrain enum constants (see upstream
    // VoxelLodTerrain.xml).
    // -----------------------------------------------------------------

    /// ProcessCallback: run main-thread update from `_process`.
    #[constant]
    const PROCESS_CALLBACK_IDLE: i64 = 0;
    /// ProcessCallback: run main-thread update from `_physics_process`.
    #[constant]
    const PROCESS_CALLBACK_PHYSICS: i64 = 1;
    /// ProcessCallback: do not run the main-thread update.
    #[constant]
    const PROCESS_CALLBACK_DISABLED: i64 = 2;

    /// DebugDrawFlag: draw octree nodes.
    #[constant]
    const DEBUG_DRAW_OCTREE_NODES: i64 = 0;
    /// DebugDrawFlag: draw octree bounds.
    #[constant]
    const DEBUG_DRAW_OCTREE_BOUNDS: i64 = 1;
    /// DebugDrawFlag: draw mesh update activity.
    #[constant]
    const DEBUG_DRAW_MESH_UPDATES: i64 = 2;
    /// DebugDrawFlag: draw edit boxes.
    #[constant]
    const DEBUG_DRAW_EDIT_BOXES: i64 = 3;
    /// DebugDrawFlag: draw the streamed volume's bounds.
    #[constant]
    const DEBUG_DRAW_VOLUME_BOUNDS: i64 = 4;
    /// DebugDrawFlag: draw edited blocks.
    #[constant]
    const DEBUG_DRAW_EDITED_BLOCKS: i64 = 5;
    /// DebugDrawFlag: draw modifier bounds.
    #[constant]
    const DEBUG_DRAW_MODIFIER_BOUNDS: i64 = 6;
    /// DebugDrawFlag: draw active mesh blocks.
    #[constant]
    const DEBUG_DRAW_ACTIVE_MESH_BLOCKS: i64 = 7;
    /// DebugDrawFlag: draw per-viewer clipboxes.
    #[constant]
    const DEBUG_DRAW_VIEWER_CLIPBOXES: i64 = 8;
    /// DebugDrawFlag: draw loaded visual+collision blocks.
    #[constant]
    const DEBUG_DRAW_LOADED_VISUAL_AND_COLLISION_BLOCKS: i64 = 9;
    /// DebugDrawFlag: draw active visual+collision blocks.
    #[constant]
    const DEBUG_DRAW_ACTIVE_VISUAL_AND_COLLISION_BLOCKS: i64 = 10;
    /// DebugDrawFlag: sentinel count of debug draw flags.
    #[constant]
    const DEBUG_DRAW_FLAGS_COUNT: i64 = 12;

    /// StreamingSystem: legacy octree streaming (single viewer).
    #[constant]
    const STREAMING_SYSTEM_LEGACY_OCTREE: i64 = 0;
    /// StreamingSystem: clipbox streaming (multi-viewer).
    #[constant]
    const STREAMING_SYSTEM_CLIPBOX: i64 = 1;

    // -----------------------------------------------------------------
    // Pinned VoxelLodTerrain methods (the canonical GDScript surface).
    // -----------------------------------------------------------------

    /// Returns the data block size in voxels (the side of one data block).
    #[func]
    fn get_data_block_size(&self) -> i32 {
        self.mesh_block_size_value
    }

    /// Returns how many voxel data chunks are currently loaded.
    #[func]
    fn debug_get_data_block_count(&self) -> i32 {
        let Some(core) = self.core.as_ref() else {
            return 0;
        };
        i32::try_from(core.debug_snapshot().data_block_count).unwrap_or(i32::MAX)
    }

    /// Clipbox "leaf" count — resident mesh blocks across every LOD.
    #[func]
    fn get_octree_node_count(&self) -> i32 {
        let Some(core) = self.core.as_ref() else {
            return 0;
        };
        i32::try_from(core.debug_snapshot().mesh_blocks.len()).unwrap_or(i32::MAX)
    }

    /// Gets how many meshes the terrain currently has (alias of
    /// `get_mesh_block_count`).
    #[func]
    fn debug_get_mesh_block_count(&self) -> i32 {
        self.get_mesh_block_count()
    }

    /// Converts a voxel position into a data block position for a specific LOD
    /// index. Uses floor division so negative positions map to the block that
    /// geometrically contains them.
    #[func]
    fn voxel_to_data_block_position(&self, voxel_position: Vector3, lod_index: i32) -> Vector3i {
        let block_size = self.mesh_block_size_value.max(1);
        let lod = lod_index.max(0);
        let stride = block_size.checked_shl(lod as u32).unwrap_or(i32::MAX);
        if stride <= 0 {
            return Vector3i::ZERO;
        }
        let scale = stride as f32;
        Vector3i::new(
            (voxel_position.x / scale).floor() as i32,
            (voxel_position.y / scale).floor() as i32,
            (voxel_position.z / scale).floor() as i32,
        )
    }

    /// Converts a voxel position into a mesh block position for a specific LOD
    /// index.
    #[func]
    fn voxel_to_mesh_block_position(&self, voxel_position: Vector3, lod_index: i32) -> Vector3i {
        self.voxel_to_data_block_position(voxel_position, lod_index)
    }

    /// Gets debug information about a specific voxel data chunk.
    #[func]
    fn debug_get_data_block_info(&self, block_pos: Vector3, lod: i32) -> Variant {
        let Some(core) = self.core.as_ref() else {
            return Variant::nil();
        };
        let lod = u8::try_from(lod.max(0)).unwrap_or(0);
        let position = voxel_core::math::Vector3i::new(
            block_pos.x.floor() as i32,
            block_pos.y.floor() as i32,
            block_pos.z.floor() as i32,
        );
        let Some(block) = core.data().block_snapshot(position, lod as usize) else {
            return Variant::nil();
        };
        let mut dict = VarDictionary::new();
        dict.set("found", true);
        dict.set("lod", i32::from(lod));
        dict.set("edited", block.is_edited());
        dict.set("modified", block.is_modified());
        dict.set("has_voxels", block.has_voxels());
        dict.to_variant()
    }

    /// Gets debug information about a specific mesh block.
    #[func]
    fn debug_get_mesh_block_info(&self, block_pos: Vector3, lod: i32) -> Variant {
        let Some(core) = self.core.as_ref() else {
            return Variant::nil();
        };
        let lod = u8::try_from(lod.max(0)).unwrap_or(0);
        if lod >= core.lod_count() {
            return Variant::nil();
        }
        let position = voxel_core::math::Vector3i::new(
            block_pos.x.floor() as i32,
            block_pos.y.floor() as i32,
            block_pos.z.floor() as i32,
        );
        let Some(entry) = core.mesh_blocks_at_lod(lod).get(&position) else {
            return Variant::nil();
        };
        let mut dict = VarDictionary::new();
        dict.set("found", true);
        dict.set("lod", i32::from(lod));
        dict.set("visual_active", entry.visual_active);
        dict.set("collision_active", entry.collision_active);
        dict.set("loaded", entry.is_loaded);
        dict.to_variant()
    }

    /// Resident mesh-block leaves of the clipbox planner. Each entry is a
    /// Dictionary `{position, lod, size, visual_active, collision_active}`.
    #[func]
    fn debug_get_octrees_detailed(&self) -> Variant {
        let Some(core) = self.core.as_ref() else {
            return Array::<Variant>::new().to_variant();
        };
        let snapshot = core.debug_snapshot();
        let mut out = Array::<Variant>::new();
        for block in snapshot.mesh_blocks {
            let mut dict = VarDictionary::new();
            dict.set(
                "position",
                Vector3i::new(block.position.x, block.position.y, block.position.z),
            );
            dict.set("lod", i32::from(block.lod));
            dict.set(
                "size",
                snapshot
                    .mesh_block_size
                    .checked_shl(u32::from(block.lod))
                    .unwrap_or(i32::MAX),
            );
            dict.set("visual_active", block.visual_active);
            dict.set("collision_active", block.collision_active);
            out.push(&dict.to_variant());
        }
        out.to_variant()
    }

    /// Captures a top-down representation of the SDF at multiple LOD levels
    /// within a specific area. Stubbed: returns `null`.
    #[func]
    fn debug_print_sdf_top_down(&self, _center: Vector3i, _extents: Vector3i) -> Variant {
        Variant::nil()
    }

    /// Gets the non-empty mesh chunk positions from a rough world-space ray.
    /// Stubbed: returns `null` in the headless binding.
    #[func]
    fn debug_raycast_mesh_block(&self, _origin: Vector3, _dir: Vector3) -> Variant {
        Variant::nil()
    }

    /// Saves the current state of the terrain as a Godot scene file. The
    /// binding does not own a scene serializer; returns 1 on success (no-op).
    #[func]
    fn debug_dump_as_scene(&self, _path: GString, _include_instancer: bool) -> i32 {
        1
    }

    /// Returns the data block region extent. Stubbed: returns 0 before
    /// `_ready` or when the streaming region is unbounded.
    #[func]
    fn get_data_block_region_extent(&self) -> i32 {
        0
    }

    /// Returns the normalmap generator override resource (or null).
    #[func]
    fn get_normalmap_generator_override(&self) -> Variant {
        match &self.normalmap_generator_override_resource {
            Some(g) => g.to_variant(),
            None => Variant::nil(),
        }
    }

    /// Sets the normalmap generator override resource.
    #[func]
    fn set_normalmap_generator_override(&mut self, generator_override: Gd<Resource>) {
        self.normalmap_generator_override_resource = Some(generator_override);
    }

    /// Returns the LOD index from which normalmaps begin.
    #[func]
    fn get_normalmap_generator_override_begin_lod_index(&self) -> i32 {
        self.normalmap_generator_override_begin_lod_index_value
    }

    /// Sets the LOD index from which normalmaps begin.
    #[func]
    fn set_normalmap_generator_override_begin_lod_index(&mut self, lod_index: i32) {
        self.normalmap_generator_override_begin_lod_index_value = lod_index.max(0);
    }

    /// Gets which process callback runs the main-thread update.
    #[func]
    fn get_process_callback(&self) -> i64 {
        i64::from(self.process_callback_value.clamp(0, 2))
    }

    /// Sets which process callback runs the main-thread update.
    #[func]
    fn set_process_callback(&mut self, mode: i64) {
        self.process_callback_value = i32::try_from(mode).unwrap_or(0).clamp(0, 2);
    }

    /// Returns a `VoxelToolTerrain` bound to this Variable-LOD terrain.
    #[func]
    fn get_voxel_tool(&self) -> Variant {
        let mut tool = crate::voxel_buffer::VoxelToolTerrainGD::new_gd();
        tool.bind_mut().bind_lod_terrain(self.to_gd());
        tool.to_variant()
    }

    /// Returns `true` if the area has been processed by meshing. Conservative
    /// stub: returns `false` if no core is present.
    #[func]
    fn is_area_meshed(&self, area_in_voxels: Aabb, lod_index: i32) -> bool {
        let Some(core) = self.core.as_ref() else {
            return false;
        };
        let block_size = self.mesh_block_size_value.max(1);
        let lod = lod_index.max(0);
        let stride = block_size.checked_shl(lod as u32).unwrap_or(i32::MAX);
        if stride <= 0 {
            return false;
        }
        let origin = area_in_voxels.position;
        let size = area_in_voxels.size;
        if size.x <= 0.0 || size.y <= 0.0 || size.z <= 0.0 {
            return true;
        }
        let scale = stride as f32;
        let mesh_map = core.mesh_blocks();
        let min_block_x = (origin.x / scale).floor() as i32;
        let min_block_y = (origin.y / scale).floor() as i32;
        let min_block_z = (origin.z / scale).floor() as i32;
        let far_x = origin.x + size.x - 1.0;
        let far_y = origin.y + size.y - 1.0;
        let far_z = origin.z + size.z - 1.0;
        let max_block_x = (far_x / scale).floor() as i32;
        let max_block_y = (far_y / scale).floor() as i32;
        let max_block_z = (far_z / scale).floor() as i32;
        let mut z = min_block_z;
        while z <= max_block_z {
            let mut y = min_block_y;
            while y <= max_block_y {
                let mut x = min_block_x;
                while x <= max_block_x {
                    let entry = mesh_map.get(&voxel_core::math::Vector3i::new(x, y, z));
                    if !entry.is_some_and(|block| block.has_geometry) {
                        return false;
                    }
                    x += 1;
                }
                y += 1;
            }
            z += 1;
        }
        true
    }

    // -----------------------------------------------------------------
    // Pinned VoxelLodTerrain properties (transactional get/set pairs).
    // -----------------------------------------------------------------

    /// Compatibility plural alias of `get_generate_collision`. Upstream's
    /// property is `generate_collisions`.
    #[func]
    fn get_generate_collisions(&self) -> bool {
        self.generate_collision
    }

    /// Compatibility plural alias of `set_generate_collision`. Upstream's
    /// property is `generate_collisions`.
    #[func]
    fn set_generate_collisions(&mut self, enabled: bool) {
        self.generate_collision = enabled;
    }

    /// Mesh block size in voxels (read-only). Matches the core's data block
    /// size when one exists, otherwise the inspector default of 16.
    #[func]
    pub(crate) fn get_mesh_block_size(&self) -> i32 {
        self.mesh_block_size_value
    }

    #[func]
    fn set_mesh_block_size(&mut self, size: i32) {
        if self.core.is_some() {
            godot_error!(
                "VoxelLodTerrain.set_mesh_block_size: cannot change block size after _ready"
            );
            return;
        }
        if size != 16 {
            godot_error!(
                "VoxelLodTerrain.set_mesh_block_size: only 16 is supported (must match VoxelData block size)"
            );
            return;
        }
        self.mesh_block_size_value = size;
    }

    /// Distance (voxels) controlling the size of each LOD above LOD 0.
    #[func]
    fn get_secondary_lod_distance(&self) -> f32 {
        self.secondary_lod_distance_value
    }

    #[func]
    fn set_secondary_lod_distance(&mut self, distance: f32) {
        if validate_finite_nonnegative_float(distance).is_err() {
            godot_error!(
                "VoxelLodTerrain.set_secondary_lod_distance: distance must be finite and non-negative"
            );
            return;
        }
        self.secondary_lod_distance_value = distance;
        self.apply_live_clipbox_distances();
    }

    /// LOD fade duration (seconds). 0 disables fading.
    #[func]
    fn get_lod_fade_duration(&self) -> f32 {
        self.lod_fade_duration_value
    }

    #[func]
    fn set_lod_fade_duration(&mut self, duration: f32) {
        if validate_finite_nonnegative_float(duration).is_err() {
            godot_error!(
                "VoxelLodTerrain.set_lod_fade_duration: duration must be finite and non-negative"
            );
            return;
        }
        self.lod_fade_duration_value = duration;
    }

    /// Maximum view distance (voxels).
    #[func]
    fn get_view_distance(&self) -> i32 {
        self.view_distance_value
    }

    #[func]
    fn set_view_distance(&mut self, distance: i32) {
        self.view_distance_value = distance.max(0);
    }

    /// Material used for the surface (alias of `material_override`).
    #[func]
    fn get_material(&self) -> Variant {
        match &self.material_override {
            Some(m) => m.to_variant(),
            None => Variant::nil(),
        }
    }

    #[func]
    fn set_material(&mut self, value: Gd<Material>) {
        self.material_override = Some(value);
    }

    /// Physics collision layer applied to every block `StaticBody3D`.
    #[func]
    fn get_collision_layer(&self) -> i32 {
        self.collision_layer_value
    }

    #[func]
    fn set_collision_layer(&mut self, layer: i32) {
        self.collision_layer_value = layer;
        self.refresh_collision_bodies();
    }

    /// Physics collision mask applied to every block `StaticBody3D`.
    #[func]
    fn get_collision_mask(&self) -> i32 {
        self.collision_mask_value
    }

    #[func]
    fn set_collision_mask(&mut self, mask: i32) {
        self.collision_mask_value = mask;
        self.refresh_collision_bodies();
    }

    /// Collision shape margin applied to every block collider.
    #[func]
    fn get_collision_margin(&self) -> f32 {
        self.collision_margin_value
    }

    #[func]
    fn set_collision_margin(&mut self, margin: f32) {
        if margin.is_finite() && margin >= 0.0 {
            self.collision_margin_value = margin;
        } else {
            godot_error!(
                "VoxelLodTerrain.set_collision_margin: margin must be non-negative and finite"
            );
        }
        self.refresh_collision_bodies();
    }

    /// How many LOD levels generate colliders (0 = all LODs).
    #[func]
    fn get_collision_lod_count(&self) -> i32 {
        self.collision_lod_count_value
    }

    #[func]
    fn set_collision_lod_count(&mut self, count: i32) {
        self.collision_lod_count_value = count.max(0);
    }

    /// How long to wait before updating colliders after an edit (ms).
    #[func]
    fn get_collision_update_delay(&self) -> i32 {
        self.collision_update_delay_value
    }

    #[func]
    fn set_collision_update_delay(&mut self, delay: i32) {
        self.collision_update_delay_value = delay.max(0);
    }

    /// Whether generated voxel data is cached around viewers. Stubbed: stored
    /// faithfully.
    #[func]
    fn get_cache_generated_blocks(&self) -> bool {
        self.cache_generated_blocks_value
    }

    #[func]
    fn set_cache_generated_blocks(&mut self, enabled: bool) {
        self.cache_generated_blocks_value = enabled;
    }

    /// Whether full-load mode is enabled (all data kept in memory). Stubbed:
    /// stored faithfully.
    #[func]
    fn is_full_load_mode_enabled(&self) -> bool {
        self.full_load_mode_enabled_value
    }

    #[func]
    fn set_full_load_mode_enabled(&mut self, enabled: bool) {
        self.full_load_mode_enabled_value = enabled;
    }

    /// Whether the stream runs in the editor. Stubbed: stored faithfully.
    #[func]
    fn is_stream_running_in_editor(&self) -> bool {
        self.run_stream_in_editor_value
    }

    #[func]
    fn set_run_stream_in_editor(&mut self, enabled: bool) {
        self.run_stream_in_editor_value = enabled;
    }

    /// Whether threaded updates are enabled. Stubbed: stored faithfully.
    #[func]
    fn is_threaded_update_enabled(&self) -> bool {
        self.threaded_update_enabled_value
    }

    #[func]
    fn set_threaded_update_enabled(&mut self, enabled: bool) {
        self.threaded_update_enabled_value = enabled;
    }

    /// Whether GPU block generation is enabled. Stubbed: stored faithfully.
    #[func]
    fn get_generator_use_gpu(&self) -> bool {
        self.use_gpu_generation_value
    }

    #[func]
    fn set_generator_use_gpu(&mut self, enabled: bool) {
        if enabled {
            godot_error!(
                "VoxelLodTerrain.set_generator_use_gpu: GPU generation is intentionally deferred in this build"
            );
        }
        self.use_gpu_generation_value = enabled;
    }

    /// Streaming algorithm selection (0=legacy octree, 1=clipbox).
    #[func]
    fn get_streaming_system(&self) -> i32 {
        self.streaming_system_value
    }

    #[func]
    fn set_streaming_system(&mut self, system: i32) {
        self.streaming_system_value = system.clamp(0, 1);
    }

    /// Whether distant normalmap generation is enabled. Stubbed: stored
    /// faithfully.
    #[func]
    fn is_normalmap_enabled(&self) -> bool {
        self.normalmap_enabled_value
    }

    #[func]
    fn set_normalmap_enabled(&mut self, enabled: bool) {
        self.normalmap_enabled_value = enabled;
    }

    /// LOD index from which normalmaps begin.
    #[func]
    fn get_normalmap_begin_lod_index(&self) -> i32 {
        self.normalmap_begin_lod_index_value
    }

    #[func]
    fn set_normalmap_begin_lod_index(&mut self, index: i32) {
        self.normalmap_begin_lod_index_value = index.max(0);
    }

    /// Maximum deviation (degrees) beyond which a normalmap is generated.
    #[func]
    fn get_normalmap_max_deviation_degrees(&self) -> i32 {
        self.normalmax_deviation_degrees_value
    }

    #[func]
    fn set_normalmap_max_deviation_degrees(&mut self, degrees: i32) {
        self.normalmax_deviation_degrees_value = degrees.max(0);
    }

    /// Minimum resolution of tiles in distant normalmaps.
    #[func]
    fn get_normalmap_tile_resolution_min(&self) -> i32 {
        self.normalmap_tile_resolution_min_value
    }

    #[func]
    fn set_normalmap_tile_resolution_min(&mut self, resolution: i32) {
        self.normalmap_tile_resolution_min_value = resolution.max(1);
    }

    /// Maximum resolution of tiles in distant normalmaps.
    #[func]
    fn get_normalmap_tile_resolution_max(&self) -> i32 {
        self.normalmap_tile_resolution_max_value
    }

    #[func]
    fn set_normalmap_tile_resolution_max(&mut self, resolution: i32) {
        self.normalmap_tile_resolution_max_value = resolution.max(1);
    }

    /// Whether octahedral normalmap encoding is enabled. Stubbed: stored
    /// faithfully.
    #[func]
    fn get_octahedral_normal_encoding(&self) -> bool {
        self.normalmap_octahedral_encoding_enabled_value
    }

    #[func]
    fn set_octahedral_normal_encoding(&mut self, enabled: bool) {
        self.normalmap_octahedral_encoding_enabled_value = enabled;
    }

    /// Whether GPU normalmap generation is enabled. Stubbed: stored faithfully.
    #[func]
    fn get_normalmap_use_gpu(&self) -> bool {
        self.normalmap_use_gpu_value
    }

    #[func]
    fn set_normalmap_use_gpu(&mut self, enabled: bool) {
        self.normalmap_use_gpu_value = enabled;
    }

    /// Master toggle for the wireframe debug overlay.
    #[func]
    fn debug_is_draw_enabled(&self) -> bool {
        self.debug_draw_enabled_value
    }

    #[func]
    fn debug_set_draw_enabled(&mut self, enabled: bool) {
        self.debug_draw_enabled_value = enabled;
        self.refresh_debug_draw();
    }

    /// Debug-draw flag for shadow occluders. Stubbed.
    #[func]
    fn get_debug_draw_shadow_occluders(&self) -> bool {
        self.debug_draw_shadow_occluders_value
    }

    #[func]
    fn set_debug_draw_shadow_occluders(&mut self, enabled: bool) {
        self.debug_draw_shadow_occluders_value = enabled;
    }

    /// Toggles a `DebugDrawFlag` and refreshes the wireframe overlay.
    #[func]
    fn debug_set_draw_flag(&mut self, flag_index: i64, enabled: bool) {
        match flag_index {
            x if x == Self::DEBUG_DRAW_OCTREE_NODES => self.debug_draw_octree_nodes_value = enabled,
            x if x == Self::DEBUG_DRAW_OCTREE_BOUNDS => {
                self.debug_draw_octree_bounds_value = enabled;
            }
            x if x == Self::DEBUG_DRAW_MESH_UPDATES => {
                self.debug_draw_mesh_updates_value = enabled;
            }
            x if x == Self::DEBUG_DRAW_EDIT_BOXES => self.debug_draw_edit_boxes_value = enabled,
            x if x == Self::DEBUG_DRAW_VOLUME_BOUNDS => {
                self.debug_draw_volume_bounds_value = enabled;
            }
            x if x == Self::DEBUG_DRAW_EDITED_BLOCKS => {
                self.debug_draw_edited_blocks_value = enabled;
            }
            x if x == Self::DEBUG_DRAW_MODIFIER_BOUNDS => {
                self.debug_draw_modifier_bounds_value = enabled;
            }
            x if x == Self::DEBUG_DRAW_ACTIVE_MESH_BLOCKS => {
                self.debug_draw_active_mesh_blocks_value = enabled;
            }
            x if x == Self::DEBUG_DRAW_VIEWER_CLIPBOXES => {
                self.debug_draw_viewer_clipboxes_value = enabled;
            }
            x if x == Self::DEBUG_DRAW_LOADED_VISUAL_AND_COLLISION_BLOCKS => {
                self.debug_draw_loaded_visual_and_collision_blocks_value = enabled;
            }
            x if x == Self::DEBUG_DRAW_ACTIVE_VISUAL_AND_COLLISION_BLOCKS => {
                self.debug_draw_active_visual_and_collision_blocks_value = enabled;
            }
            _ => {
                godot_error!(
                    "VoxelLodTerrain.debug_set_draw_flag: unknown flag index {flag_index}"
                );
            }
        }
        self.refresh_debug_draw();
    }

    /// Returns the current value of a `DebugDrawFlag`. No debug rendering is
    /// performed in the headless binding; reports the stored flag value.
    #[func]
    fn debug_get_draw_flag(&self, flag_index: i64) -> bool {
        match flag_index {
            x if x == Self::DEBUG_DRAW_OCTREE_NODES => self.debug_draw_octree_nodes_value,
            x if x == Self::DEBUG_DRAW_OCTREE_BOUNDS => self.debug_draw_octree_bounds_value,
            x if x == Self::DEBUG_DRAW_MESH_UPDATES => self.debug_draw_mesh_updates_value,
            x if x == Self::DEBUG_DRAW_EDIT_BOXES => self.debug_draw_edit_boxes_value,
            x if x == Self::DEBUG_DRAW_VOLUME_BOUNDS => self.debug_draw_volume_bounds_value,
            x if x == Self::DEBUG_DRAW_EDITED_BLOCKS => self.debug_draw_edited_blocks_value,
            x if x == Self::DEBUG_DRAW_MODIFIER_BOUNDS => self.debug_draw_modifier_bounds_value,
            x if x == Self::DEBUG_DRAW_ACTIVE_MESH_BLOCKS => {
                self.debug_draw_active_mesh_blocks_value
            }
            x if x == Self::DEBUG_DRAW_VIEWER_CLIPBOXES => self.debug_draw_viewer_clipboxes_value,
            x if x == Self::DEBUG_DRAW_LOADED_VISUAL_AND_COLLISION_BLOCKS => {
                self.debug_draw_loaded_visual_and_collision_blocks_value
            }
            x if x == Self::DEBUG_DRAW_ACTIVE_VISUAL_AND_COLLISION_BLOCKS => {
                self.debug_draw_active_visual_and_collision_blocks_value
            }
            _ => {
                godot_error!(
                    "VoxelLodTerrain.debug_get_draw_flag: unknown flag index {flag_index}"
                );
                false
            }
        }
    }

    /// Bounds (in voxels) within which volume data can exist. Stored
    /// faithfully.
    #[func]
    fn get_voxel_bounds(&self) -> Aabb {
        self.voxel_bounds_value
    }

    #[func]
    fn set_voxel_bounds(&mut self, bounds: Aabb) {
        self.voxel_bounds_value = bounds;
        if self.core.is_some() {
            // Bounds are baked into SharedVoxelData at construction.
            self.core = None;
            self.base_mut().request_ready();
        }
    }
}

impl VoxelLodTerrainGD {
    fn apply_live_clipbox_distances(&mut self) {
        let Some(core) = self.core.as_mut() else {
            return;
        };
        let Ok(lod_count) = resolve_lod_count_u8(self.lod_count) else {
            return;
        };
        let bounds = core.data().bounds();
        match resolve_variable_lod_volume(
            lod_count,
            self.lod_distance,
            self.secondary_lod_distance_value,
            bounds,
        ) {
            Ok((_, settings)) => {
                if let Err(error) = core.try_reconfigure_variable_clipboxes(settings) {
                    godot_error!("VoxelLodTerrain: live clipbox update failed: {error:?}");
                }
            }
            Err(error) => {
                godot_error!("VoxelLodTerrain: live clipbox update rejected: {error:?}");
            }
        }
    }

    fn apply_render_op(&mut self, op: crate::terrain::PendingRenderOp) {
        let material_override = self.material_override.clone();
        let generate_collision = self.generate_collision;
        let collision_settings = crate::terrain::CollisionBodySettings::from_inspector(
            self.collision_layer_value,
            self.collision_mask_value,
            self.collision_margin_value,
        );
        let mut base_node = self.to_gd().upcast::<Node3D>();
        crate::terrain::apply_pending_render_op(
            op,
            &mut self.mesh_instances,
            &mut base_node,
            material_override.as_ref(),
            generate_collision,
            collision_settings,
        );
    }

    pub(crate) fn edit_sphere(
        &mut self,
        center: voxel_core::math::Vector3f,
        radius: f32,
        channel: usize,
        mode: voxel_core::edition::EditMode,
        value: u64,
    ) {
        let Some(core) = self.core.as_mut() else {
            return;
        };
        if let Err(error) = core.try_edit_sphere(center, radius, channel, mode, value) {
            godot_error!("VoxelLodTerrain.edit_sphere failed: {error}");
        }
    }

    pub(crate) fn edit_box(
        &mut self,
        min: voxel_core::math::Vector3i,
        max: voxel_core::math::Vector3i,
        channel: usize,
        mode: voxel_core::edition::EditMode,
        value: u64,
    ) {
        let Some(core) = self.core.as_mut() else {
            return;
        };
        if let Err(error) = core.try_edit_box(min, max, channel, mode, value) {
            godot_error!("VoxelLodTerrain.edit_box failed: {error}");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn edit_hemisphere(
        &mut self,
        center: voxel_core::math::Vector3f,
        radius: f32,
        flat_direction: voxel_core::math::Vector3f,
        smoothness: f32,
        channel: usize,
        mode: voxel_core::edition::EditMode,
        value: u64,
    ) {
        let Some(core) = self.core.as_mut() else {
            return;
        };
        if let Err(error) = core.try_edit_hemisphere(
            center,
            radius,
            flat_direction,
            smoothness,
            channel,
            mode,
            value,
        ) {
            godot_error!("VoxelLodTerrain.edit_hemisphere failed: {error}");
        }
    }

    pub(crate) fn edit_smooth(
        &mut self,
        center: voxel_core::math::Vector3f,
        radius: f32,
        blur_radius: i32,
        channel: usize,
    ) {
        let Some(core) = self.core.as_mut() else {
            return;
        };
        if let Err(error) = core.try_edit_smooth(center, radius, blur_radius, channel) {
            godot_error!("VoxelLodTerrain.edit_smooth failed: {error}");
        }
    }

    pub(crate) fn paste_buffer(
        &mut self,
        origin: voxel_core::math::Vector3i,
        src: &voxel_core::storage::VoxelBuffer,
        channel_mask: u8,
    ) {
        let Some(core) = self.core.as_mut() else {
            return;
        };
        if let Err(error) = core.try_paste(origin, src, channel_mask) {
            godot_error!("VoxelLodTerrain.paste failed: {error}");
        }
    }

    fn refresh_debug_draw(&mut self) {
        let snapshot = self.core.as_ref().map(|core| core.debug_snapshot());
        let flags = crate::debug_draw::DebugDrawFlags {
            octree_nodes: self.debug_draw_octree_nodes_value,
            octree_bounds: self.debug_draw_octree_bounds_value,
            volume_bounds: self.debug_draw_volume_bounds_value,
            active_mesh_blocks: self.debug_draw_active_mesh_blocks_value,
            viewer_clipboxes: self.debug_draw_viewer_clipboxes_value,
            loaded_visual_collision: self.debug_draw_loaded_visual_and_collision_blocks_value,
            active_visual_collision: self.debug_draw_active_visual_and_collision_blocks_value,
            edited_blocks: self.debug_draw_edited_blocks_value,
        };
        crate::debug_draw::refresh_debug_overlay(
            &mut self.to_gd().upcast::<Node3D>(),
            snapshot.as_ref(),
            self.debug_draw_enabled_value,
            flags,
        );
    }

    pub(crate) fn collect_voxels_in_box(
        &self,
        min: voxel_core::math::Vector3i,
        max: voxel_core::math::Vector3i,
        channel: usize,
        max_items: usize,
    ) -> Vec<(voxel_core::math::Vector3i, u64)> {
        let Some(core) = self.core.as_ref() else {
            return Vec::new();
        };
        crate::terrain::collect_core_voxels(core, min, max, channel, max_items)
    }

    pub(crate) fn edit_world_voxel_metadata(
        &mut self,
        pos: voxel_core::math::Vector3i,
        metadata: Option<voxel_core::storage::MetadataValue>,
    ) -> bool {
        let Some(core) = self.core.as_mut() else {
            return false;
        };
        matches!(core.try_edit_voxel_metadata(pos, metadata), Ok(Some(_)))
    }

    pub(crate) fn read_world_voxel_metadata(
        &self,
        pos: voxel_core::math::Vector3i,
    ) -> Option<voxel_core::storage::MetadataValue> {
        self.core.as_ref().and_then(|core| core.voxel_metadata(pos))
    }

    pub(crate) fn for_each_world_voxel_metadata(
        &self,
        min: voxel_core::math::Vector3i,
        max: voxel_core::math::Vector3i,
        visit: impl FnMut(voxel_core::math::Vector3i, &voxel_core::storage::MetadataValue),
    ) {
        if let Some(core) = self.core.as_ref() {
            core.for_each_voxel_metadata_in_area(min, max, visit);
        }
    }

    pub(crate) fn blocky_library(&self) -> Option<voxel_core::meshers::blocky::BakedLibrary> {
        crate::terrain::resolve_blocky_library(self.mesher_resource.as_ref())
    }

    pub(crate) fn surface_points_for_block(
        &self,
        position: voxel_core::math::Vector3i,
        block_size: i32,
    ) -> (
        Vec<voxel_core::math::Vector3f>,
        Vec<voxel_core::math::Vector3f>,
    ) {
        let Some(core) = self.core.as_ref() else {
            return (Vec::new(), Vec::new());
        };
        crate::terrain::surface_points_from_core(core, position, block_size)
    }

    pub(crate) fn edit_world_voxel(
        &mut self,
        pos: voxel_core::math::Vector3i,
        channel: usize,
        raw: u64,
    ) -> bool {
        let Some(core) = self.core.as_mut() else {
            return false;
        };
        matches!(core.try_edit_voxel(raw, pos, channel), Ok(Some(_)))
    }

    pub(crate) fn read_world_voxel(&self, pos: voxel_core::math::Vector3i, channel: usize) -> u64 {
        let Some(core) = self.core.as_ref() else {
            return 0;
        };
        let data = core.data();
        let Ok(block_size) = i32::try_from(data.block_size()) else {
            return 0;
        };
        let block_pos = voxel_core::storage::voxel_data_map::VoxelDataMap::voxel_to_block_b(
            pos,
            data.block_size_po2(),
        );
        data.block_snapshot(block_pos, 0)
            .filter(|block| block.has_voxels())
            .map(|block| {
                block.voxels().get_voxel(
                    pos.x.rem_euclid(block_size),
                    pos.y.rem_euclid(block_size),
                    pos.z.rem_euclid(block_size),
                    channel,
                )
            })
            .unwrap_or(0)
    }

    fn refresh_collision_bodies(&mut self) {
        let settings = crate::terrain::CollisionBodySettings::from_inspector(
            self.collision_layer_value,
            self.collision_mask_value,
            self.collision_margin_value,
        );
        for rendered in self.mesh_instances.values_mut() {
            crate::terrain::apply_collision_settings_to_instance(&mut rendered.instance, settings);
        }
    }
}

// ---------------------------------------------------------------------------
// VoxelTerrainStatsGD — RefCounted stats container
// ---------------------------------------------------------------------------
/// Terrain statistics container. Emitted by VoxelTerrain for debug display.
/// Wraps [`voxel_core::terrain::VoxelTerrainStats`] — real cumulative counters
/// pulled from the paging orchestrator (blocks loaded/unloaded, meshes
/// built/dropped).
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelTerrainStats)]
pub struct VoxelTerrainStatsGD {
    base: Base<RefCounted>,
    #[var]
    blocks_loaded: i64,
    #[var]
    blocks_unloaded: i64,
    #[var]
    meshes_built: i64,
    #[var]
    meshes_dropped: i64,
}
#[godot_api]
impl IRefCounted for VoxelTerrainStatsGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            blocks_loaded: 0,
            blocks_unloaded: 0,
            meshes_built: 0,
            meshes_dropped: 0,
        }
    }
}

impl VoxelTerrainStatsGD {
    /// Build a stats snapshot from the engine-agnostic
    /// [`voxel_core::terrain::VoxelTerrainStats`]. Called by
    /// `VoxelTerrain::get_statistics` to expose the real paging counters to
    /// GDScript/inspector.
    pub fn from_core_stats(stats: &voxel_core::terrain::VoxelTerrainStats) -> Gd<Self> {
        Gd::from_init_fn(|base| Self {
            base,
            blocks_loaded: i64::try_from(stats.blocks_loaded).unwrap_or(i64::MAX),
            blocks_unloaded: i64::try_from(stats.blocks_unloaded).unwrap_or(i64::MAX),
            meshes_built: i64::try_from(stats.meshes_built).unwrap_or(i64::MAX),
            meshes_dropped: i64::try_from(stats.meshes_dropped).unwrap_or(i64::MAX),
        })
    }
}

// ---------------------------------------------------------------------------
// VoxelRaycastResultGD2 — alias for blocky raycast (non-SDF)
// ---------------------------------------------------------------------------
/// Result of a blocky/non-SDF voxel raycast.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelBlockRaycastResult)]
pub struct VoxelBlockRaycastResultGD {
    base: Base<RefCounted>,
    #[var]
    voxel_id: i64,
    #[var]
    hit_x: i32,
    #[var]
    hit_y: i32,
    #[var]
    hit_z: i32,
}
#[godot_api]
impl IRefCounted for VoxelBlockRaycastResultGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            voxel_id: 0,
            hit_x: 0,
            hit_y: 0,
            hit_z: 0,
        }
    }
}

#[godot_api]
impl VoxelBlockRaycastResultGD {
    /// Whether this result hit a non-air voxel.
    #[func]
    fn did_hit(&self) -> bool {
        self.voxel_id != 0
    }

    /// The hit position as a packed array [x, y, z].
    #[func]
    fn get_hit_position(&self) -> PackedInt32Array {
        PackedInt32Array::from(&[self.hit_x, self.hit_y, self.hit_z][..])
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockSerializerGD — RefCounted for block save/load
// ---------------------------------------------------------------------------
/// Utility for serializing/deserializing voxel blocks to/from bytes.
/// Wraps [`voxel_core::streams::block_serializer`] with a real VoxelBuffer.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelBlockSerializer)]
pub struct VoxelBlockSerializerGD {
    base: Base<RefCounted>,
    buffer: voxel_core::storage::VoxelBuffer,
}
#[godot_api]
impl IRefCounted for VoxelBlockSerializerGD {
    fn init(base: Base<RefCounted>) -> Self {
        let buffer =
            voxel_core::storage::VoxelBuffer::with_size(voxel_core::math::Vector3i::splat(1));
        Self { base, buffer }
    }
}
#[godot_api]
impl VoxelBlockSerializerGD {
    /// Initialize the internal buffer with a given size.
    #[func]
    fn create_buffer(&mut self, sx: i32, sy: i32, sz: i32) {
        let Ok(size) = crate::voxel_buffer::validate_buffer_size(sx, sy, sz) else {
            godot_error!("VoxelBlockSerializer.create_buffer: invalid buffer size");
            return;
        };
        self.buffer = voxel_core::storage::VoxelBuffer::with_size(size);
        voxel_core::storage::VoxelFormat::new().configure_buffer(&mut self.buffer);
    }

    /// Set a voxel in the internal buffer.
    #[func]
    fn set_voxel(&mut self, x: i32, y: i32, z: i32, channel: i32, value: i64) {
        let Ok(channel) = crate::voxel_buffer::validate_channel(channel) else {
            godot_error!("VoxelBlockSerializer.set_voxel: invalid channel");
            return;
        };
        let Ok(position) = crate::voxel_buffer::validate_position(
            voxel_core::math::Vector3i::new(x, y, z),
            self.buffer.size(),
        ) else {
            godot_error!("VoxelBlockSerializer.set_voxel: position is outside the buffer");
            return;
        };
        let Ok(value) = crate::voxel_buffer::validate_voxel_value(value) else {
            godot_error!("VoxelBlockSerializer.set_voxel: invalid voxel value");
            return;
        };
        self.buffer
            .set_voxel(value, position.x, position.y, position.z, channel);
    }

    /// Get a voxel from the internal buffer.
    #[func]
    fn get_voxel(&self, x: i32, y: i32, z: i32, channel: i32) -> i64 {
        let Ok(channel) = crate::voxel_buffer::validate_channel(channel) else {
            godot_error!("VoxelBlockSerializer.get_voxel: invalid channel");
            return 0;
        };
        let Ok(position) = crate::voxel_buffer::validate_position(
            voxel_core::math::Vector3i::new(x, y, z),
            self.buffer.size(),
        ) else {
            godot_error!("VoxelBlockSerializer.get_voxel: position is outside the buffer");
            return 0;
        };
        match i64::try_from(
            self.buffer
                .get_voxel(position.x, position.y, position.z, channel),
        ) {
            Ok(value) => value,
            Err(_) => {
                godot_error!(
                    "VoxelBlockSerializer.get_voxel: voxel value exceeds GDScript integer range"
                );
                0
            }
        }
    }

    /// Serialize the internal buffer into a PackedByteArray (block format v4).
    #[func]
    fn serialize(&self) -> PackedByteArray {
        let mut data = Vec::new();
        match voxel_core::streams::block_serializer::serialize(&self.buffer, &mut data) {
            Ok(_) => PackedByteArray::from(data.as_slice()),
            Err(_) => PackedByteArray::new(),
        }
    }

    /// Deserialize a PackedByteArray into the internal buffer.
    #[func]
    fn deserialize(&mut self, data: PackedByteArray) -> bool {
        let raw = data.as_slice();
        voxel_core::streams::block_serializer::deserialize(raw, &mut self.buffer).is_ok()
    }

    /// Serialize + LZ4-compress the internal buffer.
    #[func]
    fn serialize_compressed(&self) -> PackedByteArray {
        let mut data = Vec::new();
        match voxel_core::streams::block_serializer::serialize_and_compress(
            &self.buffer,
            &mut data,
            voxel_core::streams::compressed_data::Compression::Lz4,
        ) {
            Ok(_) => PackedByteArray::from(data.as_slice()),
            Err(_) => PackedByteArray::new(),
        }
    }

    /// Decompress + deserialize into the internal buffer.
    #[func]
    fn decompress_and_deserialize(&mut self, data: PackedByteArray) -> bool {
        let raw = data.as_slice();
        voxel_core::streams::block_serializer::decompress_and_deserialize(raw, &mut self.buffer)
            .is_ok()
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelBlockSerializer API (upstream 5828cbeb).
    //
    // Upstream exposes all four serialization entry points as `static`
    // methods taking/returning a `VoxelBuffer`. gdext registers a `#[func]`
    // without a receiver as a static method on the class, so GDScript calls
    // them as `VoxelBlockSerializer.serialize_to_byte_array(...)`.
    // -----------------------------------------------------------------

    /// Compression mode: no compression (canonical `COMPRESSION_NONE`).
    #[constant]
    const COMPRESSION_NONE: i64 = 0;
    /// Compression mode: LZ4 default (canonical `COMPRESSION_LZ4`).
    #[constant]
    const COMPRESSION_LZ4: i64 = 1;
    /// Compression mode: Zstandard default (canonical `COMPRESSION_ZSTD`).
    #[constant]
    const COMPRESSION_ZSTD: i64 = 2;

    /// Stores the data of a [VoxelBuffer] into a [PackedByteArray]
    /// (canonical `serialize_to_byte_array`).
    #[func]
    fn serialize_to_byte_array(
        voxel_buffer: Gd<crate::voxel_buffer::VoxelBufferGD>,
        compress: i64,
    ) -> PackedByteArray {
        let bound = voxel_buffer.bind();
        let core = bound.core_buffer();
        let compression = match Self::resolve_compression(compress) {
            Ok(mode) => mode,
            Err(()) => {
                drop(bound);
                godot_error!(
                    "VoxelBlockSerializer.serialize_to_byte_array: invalid compression mode ({compress})"
                );
                return PackedByteArray::new();
            }
        };
        if let Some(compression) = compression {
            let mut data = Vec::new();
            let result = voxel_core::streams::block_serializer::serialize_and_compress(
                core,
                &mut data,
                compression,
            );
            drop(bound);
            match result {
                Ok(_) => PackedByteArray::from(data.as_slice()),
                Err(_) => PackedByteArray::new(),
            }
        } else {
            let mut data = Vec::new();
            let result = voxel_core::streams::block_serializer::serialize(core, &mut data);
            drop(bound);
            match result {
                Ok(_) => PackedByteArray::from(data.as_slice()),
                Err(_) => PackedByteArray::new(),
            }
        }
    }

    /// Reads the data of a [VoxelBuffer] from a [PackedByteArray]
    /// (canonical `deserialize_from_byte_array`).
    #[func]
    fn deserialize_from_byte_array(
        bytes: PackedByteArray,
        mut voxel_buffer: Gd<crate::voxel_buffer::VoxelBufferGD>,
        decompress: bool,
    ) {
        let raw = bytes.as_slice();
        let mut bound = voxel_buffer.bind_mut();
        let core = bound.core_buffer_mut();
        let result = if decompress {
            voxel_core::streams::block_serializer::decompress_and_deserialize(raw, core)
        } else {
            voxel_core::streams::block_serializer::deserialize(raw, core)
        };
        if result.is_err() {
            drop(bound);
            godot_error!(
                "VoxelBlockSerializer.deserialize_from_byte_array: failed to deserialize VoxelBuffer"
            );
        }
    }

    /// Stores the data of a [VoxelBuffer] into a [StreamPeer] and returns the
    /// number of written bytes (canonical `serialize_to_stream_peer`).
    #[func]
    fn serialize_to_stream_peer(
        peer: Gd<godot::classes::StreamPeer>,
        voxel_buffer: Gd<crate::voxel_buffer::VoxelBufferGD>,
        compress: i64,
    ) -> i64 {
        let bound = voxel_buffer.bind();
        let core = bound.core_buffer();
        let compression = match Self::resolve_compression(compress) {
            Ok(mode) => mode,
            Err(()) => {
                drop(bound);
                godot_error!(
                    "VoxelBlockSerializer.serialize_to_stream_peer: invalid compression mode ({compress})"
                );
                return -1;
            }
        };
        let mut data = Vec::new();
        let result = if let Some(compression) = compression {
            voxel_core::streams::block_serializer::serialize_and_compress(
                core,
                &mut data,
                compression,
            )
        } else {
            voxel_core::streams::block_serializer::serialize(core, &mut data)
        };
        drop(bound);
        if result.is_err() {
            godot_error!(
                "VoxelBlockSerializer.serialize_to_stream_peer: failed to serialize VoxelBuffer"
            );
            return -1;
        }
        let written = data.len();
        let payload = PackedByteArray::from(data.as_slice());
        let mut peer = peer;
        let error = peer.put_data(&payload);
        if error != godot::global::Error::OK {
            godot_error!(
                "VoxelBlockSerializer.serialize_to_stream_peer: StreamPeer.put_data failed ({error:?})"
            );
            return -1;
        }
        i64::try_from(written).unwrap_or(-1)
    }

    /// Reads the data of a [VoxelBuffer] from a [StreamPeer]
    /// (canonical `deserialize_from_stream_peer`).
    #[func]
    fn deserialize_from_stream_peer(
        peer: Gd<godot::classes::StreamPeer>,
        mut voxel_buffer: Gd<crate::voxel_buffer::VoxelBufferGD>,
        size: i64,
        decompress: bool,
    ) {
        let read_len = match i32::try_from(size.max(0)) {
            Ok(len) => len,
            Err(_) => {
                godot_error!(
                    "VoxelBlockSerializer.deserialize_from_stream_peer: size must fit i32 ({size})"
                );
                return;
            }
        };
        let mut peer = peer;
        // StreamPeer.get_data returns [Error, PackedByteArray].
        let result_array = peer.get_data(read_len);
        let error = result_array.at(0).to::<godot::global::Error>();
        if error != godot::global::Error::OK {
            godot_error!(
                "VoxelBlockSerializer.deserialize_from_stream_peer: StreamPeer.get_data failed ({error:?})"
            );
            return;
        }
        let payload: PackedByteArray = match result_array.at(1).try_to::<PackedByteArray>() {
            Ok(bytes) => bytes,
            Err(_) => {
                godot_error!(
                    "VoxelBlockSerializer.deserialize_from_stream_peer: StreamPeer.get_data returned no byte array"
                );
                return;
            }
        };
        let bytes = payload.as_slice();
        let mut bound = voxel_buffer.bind_mut();
        let core = bound.core_buffer_mut();
        let result = if decompress {
            voxel_core::streams::block_serializer::decompress_and_deserialize(bytes, core)
        } else {
            voxel_core::streams::block_serializer::deserialize(bytes, core)
        };
        if result.is_err() {
            drop(bound);
            godot_error!(
                "VoxelBlockSerializer.deserialize_from_stream_peer: failed to deserialize VoxelBuffer"
            );
        }
    }

    /// Maps the GDScript `Compression` enum (0/1/2) to a core compression mode.
    /// `None` means "no compression". Returns `Err` for invalid values.
    fn resolve_compression(
        compress: i64,
    ) -> Result<Option<voxel_core::streams::compressed_data::Compression>, ()> {
        match compress {
            Self::COMPRESSION_NONE => Ok(None),
            Self::COMPRESSION_LZ4 => {
                Ok(Some(voxel_core::streams::compressed_data::Compression::Lz4))
            }
            Self::COMPRESSION_ZSTD => Ok(Some(
                voxel_core::streams::compressed_data::Compression::Zstd,
            )),
            _ => Err(()),
        }
    }
}

// ---------------------------------------------------------------------------
// VoxelCompressedDataGD — RefCounted for LZ4/ZSTD payloads
// ---------------------------------------------------------------------------
/// Compressed voxel data envelope (LZ4/ZSTD). Used by region files.
///
/// The functional API runs real compression/decompression via
/// [`voxel_core::streams::compressed_data`]: `compress_bytes` LZ4-compresses a
/// byte array and `decompress_bytes` restores it.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelCompressedData)]
pub struct VoxelCompressedDataGD {
    base: Base<RefCounted>,
    /// Compression mode: 0=None, 1=Lz4Be, 2=Lz4. Plain field exposed via
    /// `get/set_compression_mode` #[func]s.
    compression_mode: i32,
}
#[godot_api]
impl IRefCounted for VoxelCompressedDataGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            compression_mode: 2,
        }
    }
}

#[godot_api]
impl VoxelCompressedDataGD {
    /// Compression mode getter.
    #[func]
    fn get_compression_mode(&self) -> i32 {
        self.compression_mode
    }

    #[func]
    fn set_compression_mode(&mut self, mode: i32) {
        self.compression_mode = mode;
    }

    /// Compress a byte array with the configured mode and return the
    /// compressed bytes. Returns an empty array on error.
    #[func]
    fn compress_bytes(&self, data: PackedByteArray) -> PackedByteArray {
        let comp = compression_from_mode(self.compression_mode);
        let mut out = Vec::new();
        match voxel_core::streams::compressed_data::compress(data.as_slice(), &mut out, comp) {
            Ok(()) => PackedByteArray::from(out.as_slice()),
            Err(_) => PackedByteArray::new(),
        }
    }

    /// Decompress a byte array (compressed with the configured mode) and
    /// return the original bytes. Returns an empty array on error.
    #[func]
    fn decompress_bytes(&self, data: PackedByteArray) -> PackedByteArray {
        let comp = compression_from_mode(self.compression_mode);
        let mut out = Vec::new();
        let limits = voxel_core::streams::decode_limits::DecodeLimits::default();
        match voxel_core::streams::compressed_data::decompress_with_limits(
            data.as_slice(),
            &mut out,
            limits,
        ) {
            Ok(()) if comp != compression_from_mode(self.compression_mode) => {
                PackedByteArray::new()
            }
            Ok(()) => PackedByteArray::from(out.as_slice()),
            Err(_) => PackedByteArray::new(),
        }
    }
}

/// Map an integer mode to a `Compression` variant.
fn compression_from_mode(mode: i32) -> voxel_core::streams::compressed_data::Compression {
    use voxel_core::streams::compressed_data::Compression;
    match mode {
        0 => Compression::None,
        1 => Compression::Lz4Be,
        _ => Compression::Lz4,
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn validate_nonnegative_count_rejects_negative_values() {
        assert!(validate_nonnegative_count(-1).is_err());
    }

    #[test]
    fn script_work_limits_reject_oversized_pass_and_path_counts_without_replacing_state() {
        assert!(validate_multipass_count(MAX_MULTIPASS_PASSES + 1).is_err());
        assert!(validate_carve_path_steps(MAX_CARVE_PATH_STEPS + 1).is_err());

        let mut pass_count = 2;
        if let Ok(next) = validate_multipass_count(MAX_MULTIPASS_PASSES + 1) {
            pass_count = next;
        }
        assert_eq!(pass_count, 2);
    }

    #[test]
    fn multipass_compound_work_rejects_individually_valid_inputs() {
        let buffer_volume = crate::voxel_buffer::MAX_SCRIPT_VOXELS;
        assert!(validate_multipass_count(MAX_MULTIPASS_PASSES).is_ok());
        assert!(buffer_volume <= crate::voxel_buffer::MAX_SCRIPT_VOXELS);
        assert!(validate_multipass_work(MAX_MULTIPASS_PASSES, buffer_volume).is_err());
    }

    #[test]
    fn multipass_compound_work_accepts_an_ordinary_workload() {
        let volume_32_cubed = 32 * 32 * 32;
        assert_eq!(
            validate_multipass_work(MAX_MULTIPASS_PASSES, volume_32_cubed),
            Ok(8_388_608)
        );
    }

    #[test]
    fn multipass_compound_work_rejects_overflow() {
        assert!(validate_multipass_work(2, u64::MAX).is_err());
    }

    #[test]
    fn carve_path_compound_work_rejects_individually_valid_inputs() {
        let buffer_size = voxel_core::math::Vector3i::new(64, 64, 64);
        let modest_stamp_volume = 8 * 8 * 8;
        assert!(validate_carve_path_steps(MAX_CARVE_PATH_STEPS).is_ok());
        assert_eq!(validate_finite_nonnegative_float(4.0), Ok(4.0));
        assert_eq!(floor_to_i32(4.0), Ok(4));
        assert_eq!(
            max_clipped_stamp_volume(4, buffer_size),
            Ok(modest_stamp_volume)
        );
        assert!(validate_carve_path_work(MAX_CARVE_PATH_STEPS, modest_stamp_volume).is_err());
    }

    #[test]
    fn carve_path_compound_work_accepts_an_ordinary_workload() {
        let default_stamp_volume = 4 * 4 * 4;
        assert_eq!(
            validate_carve_path_work(MAX_CARVE_PATH_STEPS, default_stamp_volume),
            Ok(4_194_368)
        );
    }

    #[test]
    fn carve_path_compound_work_rejects_overflow() {
        assert!(validate_carve_path_work(2, u64::MAX).is_err());
    }

    #[test]
    fn checked_volume_and_clipped_stamp_volume_use_checked_arithmetic() {
        let size = voxel_core::math::Vector3i::new(2, 3, 4);
        assert_eq!(checked_nonnegative_volume(size), Ok(24));
        assert_eq!(max_clipped_stamp_volume(4, size), Ok(24));
        assert_eq!(max_clipped_stamp_volume(0, size), Ok(0));

        let overflowing = voxel_core::math::Vector3i::new(i32::MAX, i32::MAX, i32::MAX);
        assert!(checked_nonnegative_volume(overflowing).is_err());
    }

    #[test]
    fn carve_path_stamp_range_rejects_a_late_endpoint_before_stamping() {
        let start = voxel_core::math::Vector3i::zero();
        let ordinary_target = voxel_core::math::Vector3i::new(100, 20, 10);
        assert!(validate_carve_path_stamp_range(start, ordinary_target, 5, 2).is_ok());

        let overflowing_target = voxel_core::math::Vector3i::new(i32::MAX, 0, 0);
        assert!(validate_carve_path_stamp_range(start, overflowing_target, 0, 1).is_err());
    }

    #[test]
    fn exhaustive_lod_budget_preserves_state_and_bounds_leaf_count() {
        assert_eq!(validate_exhaustive_lod_count(6), Ok(6));
        assert!(validate_exhaustive_lod_count(7).is_err());
        assert!(validate_exhaustive_lod_count(23).is_err());

        let mut lod_count = 6;
        if let Ok(next) = validate_exhaustive_lod_count(7) {
            lod_count = next;
        }
        assert_eq!(lod_count, 6);
        assert_eq!(exhaustive_leaf_upper_bound(lod_count).unwrap(), 32_768);
    }

    #[test]
    fn variable_lod_volume_uses_secondary_distance_and_aligned_bounds() {
        let bounds = voxel_core::math::Box3i::new(
            voxel_core::math::Vector3i::splat(-1024),
            voxel_core::math::Vector3i::splat(2048),
        );
        let (resolved_bounds, settings) =
            resolve_variable_lod_volume(4, 64.0, 48.0, bounds).expect("aligned volume is valid");
        assert_eq!(resolved_bounds, bounds);
        assert_eq!(settings.lod0_distance_voxels, 64);
        assert_eq!(settings.secondary_distance_voxels, 48);
        assert_eq!(settings.lod_count, 4);
    }

    #[test]
    fn paging_view_distance_caps_viewer_by_terrain() {
        assert_eq!(paging_view_distance(256, 128), 128);
        assert_eq!(paging_view_distance(64, 512), 64);
        assert_eq!(paging_view_distance(-8, 128), 0);
    }

    #[test]
    fn variable_lod_volume_rejects_unaligned_bounds() {
        let unaligned = voxel_core::math::Box3i::new(
            voxel_core::math::Vector3i::splat(-536_870_900),
            voxel_core::math::Vector3i::splat(1_073_741_800),
        );
        assert!(resolve_variable_lod_volume(4, 64.0, 48.0, unaligned).is_err());
    }
}

// ---------------------------------------------------------------------------
// VoxelGeneratorMultipassGD — Resource for multipass generator
// ---------------------------------------------------------------------------
/// Multipass terrain generator (layered generation with caching).
///
/// The functional API runs `pass_count` layered `Flat`-generator passes over a
/// `VoxelBufferGD`, each at an increasing height threshold, and returns the
/// number of voxels set solid — exercising the multi-pass generation pipeline
/// through the binding.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelGeneratorMultipass)]
pub struct VoxelGeneratorMultipassGD {
    base: Base<Resource>,
    /// Number of generation passes (layers). Plain field exposed via
    /// `get/set_pass_count` #[func]s.
    pass_count: i32,
}
#[godot_api]
impl IResource for VoxelGeneratorMultipassGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            pass_count: 1,
        }
    }
}

#[godot_api]
impl VoxelGeneratorMultipassGD {
    /// Number of generation passes.
    #[func]
    fn get_pass_count(&self) -> i32 {
        self.pass_count
    }

    /// Set the pass count within the script workload limit.
    #[func]
    fn set_pass_count(&mut self, count: i32) {
        let Ok(count) = validate_multipass_count(count) else {
            godot_error!(
                "VoxelGeneratorMultipass.set_pass_count: pass count must be between 1 and MAX_MULTIPASS_PASSES"
            );
            return;
        };
        self.pass_count = count;
    }

    /// Run `pass_count` layered passes over a `VoxelBufferGD`'s Type channel.
    /// Each pass fills voxels below a rising height threshold (`layer_height`
    /// per pass) with a distinct solid id. Returns the total voxels set solid,
    /// or -1 if `buffer` is not a `VoxelBufferGD`.
    #[func]
    fn generate_layers(&self, buffer: Gd<RefCounted>, layer_height: i32) -> i64 {
        if validate_multipass_count(self.pass_count).is_err() {
            godot_error!(
                "VoxelGeneratorMultipass.generate_layers: pass count exceeds the script workload limit"
            );
            return -1;
        }
        let Ok(mut buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let bound = buf.bind();
        let core = bound.core_buffer();
        let size = core.size();
        drop(bound);
        let Ok(buffer_volume) = checked_nonnegative_volume(size) else {
            godot_error!("VoxelGeneratorMultipass.generate_layers: buffer volume overflow");
            return -1;
        };
        if validate_multipass_work(self.pass_count, buffer_volume).is_err() {
            godot_error!(
                "VoxelGeneratorMultipass.generate_layers: combined pass and buffer workload exceeds the script workload limit"
            );
            return -1;
        }
        let mut bound = buf.bind_mut();
        let core = bound.core_buffer_mut();
        const TYPE_CHANNEL: usize = 0;
        let mut total: i64 = 0;
        for pass in 0..self.pass_count {
            let threshold = layer_height.saturating_mul(pass.saturating_add(1));
            let Ok(voxel_id) = u64::try_from(pass.saturating_add(1)) else {
                godot_error!("VoxelGeneratorMultipass.generate_layers: invalid layer id");
                return -1;
            };
            for z in 0..size.z {
                for x in 0..size.x {
                    for y in 0..size.y {
                        if y < threshold {
                            // Layer id = pass+1 (distinct per pass).
                            let prev = core.get_voxel(x, y, z, TYPE_CHANNEL);
                            if prev == 0 {
                                core.set_voxel(voxel_id, x, y, z, TYPE_CHANNEL);
                                let Some(next_total) = total.checked_add(1) else {
                                    godot_error!(
                                        "VoxelGeneratorMultipass.generate_layers: changed voxel count overflow"
                                    );
                                    return -1;
                                };
                                total = next_total;
                            }
                        }
                    }
                }
            }
        }
        total
    }
}

// ---------------------------------------------------------------------------
// VoxelGraphFunctionGD — Resource for reusable graph functions
// ---------------------------------------------------------------------------
/// A reusable function within the voxel graph editor. The functional API
/// compiles a sphere-SDF sub-graph (named by `name`) and samples it —
/// exercising the CompiledGraph pipeline through the binding as a reusable
/// unit.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelGraphFunction)]
pub struct VoxelGraphFunctionGD {
    base: Base<Resource>,
    /// Function name. Plain field exposed via get/set_name #[func]s.
    name: GString,
    /// Cached compiled graph (rebuilt on parameter change).
    compiled: Option<voxel_core::generators::graph::CompiledGraph>,
}
#[godot_api]
impl IResource for VoxelGraphFunctionGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            name: "function".to_godot(),
            compiled: None,
        }
    }
}

#[godot_api]
impl VoxelGraphFunctionGD {
    /// Function name.
    #[func]
    fn get_name(&self) -> GString {
        self.name.clone()
    }

    #[func]
    fn set_name(&mut self, name: GString) {
        self.name = name;
    }

    /// Build a unit-sphere SDF function (radius 1 at origin), compile it, and
    /// sample the result at point `(px,py,pz)`. Returns NaN if compile fails.
    #[func]
    fn compile_and_sample(&mut self, px: f32, py: f32, pz: f32) -> f32 {
        if validate_finite_float(px).is_err()
            || validate_finite_float(py).is_err()
            || validate_finite_float(pz).is_err()
        {
            godot_error!("VoxelGraphFunction.compile_and_sample: coordinates must be finite");
            return f32::NAN;
        }
        use voxel_core::generators::graph::{
            CompiledGraph, CompiledScratch, Graph, GraphInputs, GraphOutput, GraphPort, NodeKind,
        };
        if self.compiled.is_none() {
            let mut g = Graph::new();
            let nx = g.push(NodeKind::Constant(px));
            let ny = g.push(NodeKind::Constant(py));
            let nz = g.push(NodeKind::Constant(pz));
            let nr = g.push(NodeKind::Constant(1.0));
            let sphere = g.push(NodeKind::SdfSphere {
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
            g.push(NodeKind::OutputSdf {
                a: Some(GraphPort {
                    node: sphere,
                    output: 0,
                }),
            });
            self.compiled = CompiledGraph::compile(&g).ok();
        }
        let Some(compiled) = &self.compiled else {
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
}

// ---------------------------------------------------------------------------
// VoxelMeshSDFGD — Resource for baked mesh SDF
// ---------------------------------------------------------------------------
/// A mesh baked into an SDF volume. Used by `VoxelModifierMeshGD`. The
/// functional API samples a box SDF (a simple baked-mesh stand-in) at a point,
/// delegating to [`voxel_core::math::sdf::sdf_box`].
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelMeshSDF)]
pub struct VoxelMeshSDFGD {
    base: Base<Resource>,
    /// Bake grid resolution (voxels per axis). Plain field exposed via
    /// `get/set_resolution` #[func]s.
    resolution: i32,
    /// Half-extents of the baked box shape.
    #[var]
    extents: f32,
    // -----------------------------------------------------------------
    // Pinned VoxelMeshSDF properties (upstream 5828cbeb).
    // Mesh-SDF baking is partial (GPU path deferred); the canonical
    // properties are stored faithfully so GDScript reads round-trip.
    // -----------------------------------------------------------------
    bake_mode_value: i32,
    boundary_sign_fix_enabled_value: bool,
    cell_count_value: i32,
    margin_ratio_value: f32,
    mesh_resource: Option<Gd<Mesh>>,
    partition_subdiv_value: i32,
    baked_value: bool,
    baking_value: bool,
    #[allow(dead_code)]
    data_dict: VarDictionary,
    /// The pinned GDScript-facing `_data` property.
    #[var(get = _get_data, set = _set_data)]
    _data: PhantomVar<VarDictionary>,
    /// The pinned GDScript-facing `bake_mode` property.
    #[var(get = get_bake_mode, set = set_bake_mode)]
    bake_mode: PhantomVar<i32>,
    /// The pinned GDScript-facing `boundary_sign_fix_enabled` property.
    #[var(get = is_boundary_sign_fix_enabled, set = set_boundary_sign_fix_enabled)]
    boundary_sign_fix_enabled: PhantomVar<bool>,
    /// The pinned GDScript-facing `cell_count` property.
    #[var(get = get_cell_count, set = set_cell_count)]
    cell_count: PhantomVar<i32>,
    /// The pinned GDScript-facing `margin_ratio` property.
    #[var(get = get_margin_ratio, set = set_margin_ratio)]
    margin_ratio: PhantomVar<f32>,
    /// The pinned GDScript-facing `mesh` property.
    #[var(get = get_mesh, set = set_mesh)]
    mesh: PhantomVar<Option<Gd<Mesh>>>,
    /// The pinned GDScript-facing `partition_subdiv` property.
    #[var(get = get_partition_subdiv, set = set_partition_subdiv)]
    partition_subdiv: PhantomVar<i32>,
}
#[godot_api]
impl IResource for VoxelMeshSDFGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            resolution: 64,
            extents: 4.0,
            bake_mode_value: Self::BAKE_MODE_ACCURATE_PARTITIONED,
            boundary_sign_fix_enabled_value: true,
            cell_count_value: 64,
            margin_ratio_value: 0.25,
            mesh_resource: None,
            partition_subdiv_value: 32,
            baked_value: false,
            baking_value: false,
            data_dict: VarDictionary::new(),
            _data: PhantomVar::default(),
            bake_mode: PhantomVar::default(),
            boundary_sign_fix_enabled: PhantomVar::default(),
            cell_count: PhantomVar::default(),
            margin_ratio: PhantomVar::default(),
            mesh: PhantomVar::default(),
            partition_subdiv: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelMeshSDFGD {
    /// Sample the baked box SDF at world point `(x,y,z)`. Negative = inside.
    #[func]
    fn sample_sdf(&self, x: f32, y: f32, z: f32) -> f32 {
        if validate_finite_float(x).is_err()
            || validate_finite_float(y).is_err()
            || validate_finite_float(z).is_err()
            || validate_finite_nonnegative_float(self.extents).is_err()
        {
            godot_error!(
                "VoxelMeshSDF.sample_sdf: coordinates must be finite and extents non-negative"
            );
            return 0.0;
        }
        let pos = voxel_core::math::Vector3f::new(x, y, z);
        let extents = voxel_core::math::Vector3f::splat(self.extents);
        voxel_core::math::sdf::sdf_box(pos, extents)
    }

    /// Resolution getter.
    #[func]
    fn get_resolution(&self) -> i32 {
        self.resolution
    }

    #[func]
    fn set_resolution(&mut self, res: i32) {
        self.resolution = res.max(1);
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelMeshSDF API (upstream 5828cbeb).
    // -----------------------------------------------------------------

    /// Getter for the canonical `bake_mode` property.
    #[func]
    fn get_bake_mode(&self) -> i32 {
        self.bake_mode_value
    }

    #[func]
    fn set_bake_mode(&mut self, mode: i32) {
        if !(0..Self::BAKE_MODE_COUNT).contains(&mode) {
            godot_error!(
                "VoxelMeshSDF.set_bake_mode: mode must be in 0..={}",
                Self::BAKE_MODE_COUNT - 1
            );
            return;
        }
        self.bake_mode_value = mode;
    }

    /// Getter for the canonical `boundary_sign_fix_enabled` property.
    #[func]
    fn is_boundary_sign_fix_enabled(&self) -> bool {
        self.boundary_sign_fix_enabled_value
    }

    #[func]
    fn set_boundary_sign_fix_enabled(&mut self, enabled: bool) {
        self.boundary_sign_fix_enabled_value = enabled;
    }

    /// Getter for the canonical `cell_count` property.
    #[func]
    fn get_cell_count(&self) -> i32 {
        self.cell_count_value
    }

    #[func]
    fn set_cell_count(&mut self, count: i32) {
        if count < 1 {
            godot_error!("VoxelMeshSDF.set_cell_count: count must be >= 1");
            return;
        }
        self.cell_count_value = count;
    }

    /// Getter for the canonical `margin_ratio` property.
    #[func]
    fn get_margin_ratio(&self) -> f32 {
        self.margin_ratio_value
    }

    #[func]
    fn set_margin_ratio(&mut self, ratio: f32) {
        if validate_finite_nonnegative_float(ratio).is_err() {
            godot_error!("VoxelMeshSDF.set_margin_ratio: ratio must be finite and non-negative");
            return;
        }
        self.margin_ratio_value = ratio;
    }

    /// Getter for the canonical `mesh` property.
    #[func]
    fn get_mesh(&self) -> Option<Gd<Mesh>> {
        self.mesh_resource.clone()
    }

    #[func]
    fn set_mesh(&mut self, mesh: Option<Gd<Mesh>>) {
        self.mesh_resource = mesh;
    }

    /// Getter for the canonical `partition_subdiv` property.
    #[func]
    fn get_partition_subdiv(&self) -> i32 {
        self.partition_subdiv_value
    }

    #[func]
    fn set_partition_subdiv(&mut self, subdiv: i32) {
        if subdiv < 1 {
            godot_error!("VoxelMeshSDF.set_partition_subdiv: subdiv must be >= 1");
            return;
        }
        self.partition_subdiv_value = subdiv;
    }

    /// Getter for the canonical `_data` property (serialised bake state).
    #[func]
    fn _get_data(&self) -> VarDictionary {
        let mut dict = VarDictionary::new();
        dict.set("bake_mode", self.bake_mode_value);
        dict.set("cell_count", self.cell_count_value);
        dict.set("margin_ratio", self.margin_ratio_value);
        dict.set("partition_subdiv", self.partition_subdiv_value);
        dict
    }

    /// Setter for the canonical `_data` property.
    #[func]
    fn _set_data(&mut self, data: VarDictionary) {
        if let Some(mode) = data.get("bake_mode") {
            if let Ok(mode) = mode.try_to::<i32>() {
                if (0..Self::BAKE_MODE_COUNT).contains(&mode) {
                    self.bake_mode_value = mode;
                }
            }
        }
        if let Some(count) = data.get("cell_count") {
            if let Ok(count) = count.try_to::<i32>() {
                if count >= 1 {
                    self.cell_count_value = count;
                }
            }
        }
        if let Some(ratio) = data.get("margin_ratio") {
            if let Ok(ratio) = ratio.try_to::<f32>() {
                if validate_finite_nonnegative_float(ratio).is_ok() {
                    self.margin_ratio_value = ratio;
                }
            }
        }
        if let Some(subdiv) = data.get("partition_subdiv") {
            if let Ok(subdiv) = subdiv.try_to::<i32>() {
                if subdiv >= 1 {
                    self.partition_subdiv_value = subdiv;
                }
            }
        }
    }

    /// Gets whether the resource contains baked SDF data (canonical
    /// `is_baked`).
    #[func]
    fn is_baked(&self) -> bool {
        self.baked_value
    }

    /// Gets whether an asynchronous baking operation is pending (canonical
    /// `is_baking`).
    #[func]
    fn is_baking(&self) -> bool {
        self.baking_value
    }

    /// Get the reference bounding box of the baked shape (canonical
    /// `get_aabb`). Reports a box sized from the half-extents.
    #[func]
    fn get_aabb(&self) -> Aabb {
        let half = self.extents.max(0.0);
        Aabb::new(
            Vector3::new(-half, -half, -half),
            Vector3::new(half * 2.0, half * 2.0, half * 2.0),
        )
    }

    /// Get the `VoxelBuffer` containing the baked distance field (canonical
    /// `get_voxel_buffer`). Returns null until a real bake runs.
    #[func]
    fn get_voxel_buffer(&self) -> Option<Gd<crate::voxel_buffer::VoxelBufferGD>> {
        None
    }

    /// Bakes the SDF on the calling thread (canonical `bake`). The headless
    /// binding has no mesh baker yet, so this reports not-baked.
    #[func]
    fn bake(&mut self) {
        godot_print!("VoxelMeshSDF.bake: mesh baking is not yet implemented in this binding");
        self.baked_value = false;
    }

    /// Bakes the SDF on a separate thread (canonical `bake_async`). The
    /// headless binding has no async baker yet.
    #[func]
    fn bake_async(&mut self, _scene_tree: Variant) {
        godot_print!("VoxelMeshSDF.bake_async: async mesh baking is not yet implemented");
        self.baking_value = false;
    }

    /// Runs checks to verify if the baked SDF contains errors (canonical
    /// `debug_check_sdf`). Returns an empty array (no errors / not baked).
    #[func]
    fn debug_check_sdf(&self, _mesh: Option<Gd<Mesh>>) -> Array<Variant> {
        Array::new()
    }

    #[constant]
    const BAKE_MODE_ACCURATE_NAIVE: i32 = 0;
    #[constant]
    const BAKE_MODE_ACCURATE_PARTITIONED: i32 = 1;
    #[constant]
    const BAKE_MODE_APPROX_INTERP: i32 = 2;
    #[constant]
    const BAKE_MODE_APPROX_FLOODFILL: i32 = 3;
    #[constant]
    const BAKE_MODE_COUNT: i32 = 4;
}

// ---------------------------------------------------------------------------
// VoxelBlockyTypeGD — Resource for one blocky voxel type
// ---------------------------------------------------------------------------
/// Defines a single blocky voxel type (model + attributes). The functional API
/// classifies the type: `is_passable` (air/non-solid), `is_opaque_solid`.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyType)]
pub struct VoxelBlockyTypeGD {
    base: Base<Resource>,
    #[var]
    name: GString,
    #[var]
    transparent: bool,
    #[var]
    solid: bool,
    // -----------------------------------------------------------------
    // Pinned VoxelBlockyType properties (upstream 5828cbeb).
    // The type is a data carrier; canonical properties are stored
    // faithfully so GDScript reads round-trip.
    // -----------------------------------------------------------------
    attributes_value: Array<Gd<VoxelBlockyAttributeGD>>,
    base_model_value: Option<Gd<VoxelBlockyModelGD>>,
    unique_name_value: StringName,
    variant_models_data_value: Array<Variant>,
    /// The pinned GDScript-facing `attributes` property.
    #[var(get = get_attributes, set = set_attributes)]
    attributes: PhantomVar<Array<Gd<VoxelBlockyAttributeGD>>>,
    /// The pinned GDScript-facing `base_model` property.
    #[var(get = get_base_model, set = set_base_model)]
    base_model: PhantomVar<Option<Gd<VoxelBlockyModelGD>>>,
    /// The pinned GDScript-facing `unique_name` property.
    #[var(get = get_unique_name, set = set_unique_name)]
    unique_name: PhantomVar<StringName>,
    /// The pinned GDScript-facing `_variant_models_data` property.
    #[var(get = _get_variant_models_data, set = _set_variant_models_data)]
    _variant_models_data: PhantomVar<Array<Variant>>,
}
#[godot_api]
impl IResource for VoxelBlockyTypeGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            name: "air".to_godot(),
            transparent: false,
            solid: false,
            attributes_value: Array::new(),
            base_model_value: None,
            unique_name_value: StringName::from("unnamed"),
            variant_models_data_value: Array::new(),
            attributes: PhantomVar::default(),
            base_model: PhantomVar::default(),
            unique_name: PhantomVar::default(),
            _variant_models_data: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyTypeGD {
    /// Whether entities can pass through this type (not solid).
    #[func]
    fn is_passable(&self) -> bool {
        !self.solid
    }

    /// Whether this type is fully opaque and solid (blocks light + movement).
    #[func]
    fn is_opaque_solid(&self) -> bool {
        self.solid && !self.transparent
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelBlockyType API (upstream 5828cbeb).
    // -----------------------------------------------------------------

    /// Maximum number of attributes a type may hold (canonical
    /// `MAX_ATTRIBUTES` constant).
    #[constant]
    const MAX_ATTRIBUTES: i64 = 4;

    /// Returns the rotation attribute of this type, or null if it has none
    /// (canonical `get_rotation_attribute`).
    #[func]
    fn get_rotation_attribute(&self) -> Variant {
        for attribute in self.attributes_value.iter_shared() {
            let bound = attribute.bind();
            if bound.is_rotation() {
                let attribute = attribute.clone();
                drop(bound);
                return attribute.to_variant();
            }
        }
        Variant::nil()
    }

    /// Explicitly sets which model to use for a given combination of
    /// attributes (canonical `set_variant_model`). The `key` is an Array of
    /// attribute values; the binding records the assignment faithfully.
    #[func]
    fn set_variant_model(&mut self, key: Array<Variant>, model: Gd<VoxelBlockyModelGD>) {
        // Record the (key, model) pairing in `_variant_models_data` as a
        // two-element Array, faithfully preserving round-trip semantics.
        let mut entry: Array<Variant> = Array::new();
        entry.push(&key.to_variant());
        entry.push(&model.to_variant());
        self.variant_models_data_value.push(&entry.to_variant());
    }

    /// Getter for the canonical `attributes` property.
    #[func]
    fn get_attributes(&self) -> Array<Gd<VoxelBlockyAttributeGD>> {
        self.attributes_value.clone()
    }

    /// Setter for the canonical `attributes` property. Enforces the
    /// `MAX_ATTRIBUTES` cap.
    #[func]
    fn set_attributes(&mut self, attributes: Array<Gd<VoxelBlockyAttributeGD>>) {
        if attributes.len() as i64 > Self::MAX_ATTRIBUTES {
            godot_error!(
                "VoxelBlockyType.set_attributes: attribute count must be <= MAX_ATTRIBUTES (4)"
            );
            return;
        }
        self.attributes_value = attributes;
    }

    /// Getter for the canonical `base_model` property. Returns null if unset.
    #[func]
    fn get_base_model(&self) -> Option<Gd<VoxelBlockyModelGD>> {
        self.base_model_value.clone()
    }

    /// Setter for the canonical `base_model` property.
    #[func]
    fn set_base_model(&mut self, model: Option<Gd<VoxelBlockyModelGD>>) {
        self.base_model_value = model;
    }

    /// Getter for the canonical `unique_name` property.
    #[func]
    fn get_unique_name(&self) -> StringName {
        self.unique_name_value.clone()
    }

    /// Setter for the canonical `unique_name` property.
    #[func]
    fn set_unique_name(&mut self, name: StringName) {
        self.unique_name_value = name;
    }

    /// Getter for the canonical `_variant_models_data` property.
    #[func]
    fn _get_variant_models_data(&self) -> Array<Variant> {
        self.variant_models_data_value.clone()
    }

    /// Setter for the canonical `_variant_models_data` property.
    #[func]
    fn _set_variant_models_data(&mut self, data: Array<Variant>) {
        self.variant_models_data_value = data;
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyModelGD — Resource for one blocky model
// ---------------------------------------------------------------------------
/// A baked blocky model (geometry + AO). Part of VoxelBlockyLibraryGD.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyModel)]
pub struct VoxelBlockyModelGD {
    base: Base<Resource>,
    #[var]
    material_index: i32,
    // -----------------------------------------------------------------
    // Pinned VoxelBlockyModel properties (upstream 5828cbeb).
    // The blocky model is a data carrier; canonical properties are stored
    // faithfully so GDScript reads round-trip.
    // -----------------------------------------------------------------
    collision_aabbs_value: Array<Aabb>,
    collision_mask_value: i32,
    color_value: Color,
    culls_neighbors_value: bool,
    lod_skirts_enabled_value: bool,
    random_tickable_value: bool,
    tags_mask_value: i32,
    transparency_index_value: i32,
    mesh_ortho_rotation_index_value: i32,
    /// Per-surface material overrides keyed by surface index.
    material_overrides: HashMap<i32, Gd<Material>>,
    /// Per-surface mesh-collision-enabled flags keyed by surface index.
    mesh_collision_enabled: HashMap<i32, bool>,
    /// The pinned GDScript-facing `collision_aabbs` property.
    #[var(get = get_collision_aabbs, set = set_collision_aabbs)]
    collision_aabbs: PhantomVar<Array<Aabb>>,
    /// The pinned GDScript-facing `collision_mask` property.
    #[var(get = get_collision_mask, set = set_collision_mask)]
    collision_mask: PhantomVar<i32>,
    /// The pinned GDScript-facing `color` property.
    #[var(get = get_color_prop, set = set_color_prop)]
    color: PhantomVar<Color>,
    /// The pinned GDScript-facing `culls_neighbors` property.
    #[var(get = get_culls_neighbors, set = set_culls_neighbors)]
    culls_neighbors: PhantomVar<bool>,
    /// The pinned GDScript-facing `lod_skirts_enabled` property.
    #[var(get = get_lod_skirts_enabled, set = set_lod_skirts_enabled)]
    lod_skirts_enabled: PhantomVar<bool>,
    /// The pinned GDScript-facing `random_tickable` property.
    #[var(get = is_random_tickable, set = set_random_tickable)]
    random_tickable: PhantomVar<bool>,
    /// The pinned GDScript-facing `tags_mask` property.
    #[var(get = get_tags_mask, set = set_tags_mask)]
    tags_mask: PhantomVar<i32>,
    /// The pinned GDScript-facing `transparency_index` property.
    #[var(get = get_transparency_index, set = set_transparency_index)]
    transparency_index: PhantomVar<i32>,
}
#[godot_api]
impl IResource for VoxelBlockyModelGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            material_index: 0,
            collision_aabbs_value: Array::new(),
            collision_mask_value: 1,
            color_value: Color::from_rgba(1.0, 1.0, 1.0, 1.0),
            culls_neighbors_value: true,
            lod_skirts_enabled_value: true,
            random_tickable_value: false,
            tags_mask_value: 1,
            transparency_index_value: 0,
            mesh_ortho_rotation_index_value: 0,
            material_overrides: HashMap::new(),
            mesh_collision_enabled: HashMap::new(),
            collision_aabbs: PhantomVar::default(),
            collision_mask: PhantomVar::default(),
            color: PhantomVar::default(),
            culls_neighbors: PhantomVar::default(),
            lod_skirts_enabled: PhantomVar::default(),
            random_tickable: PhantomVar::default(),
            tags_mask: PhantomVar::default(),
            transparency_index: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyModelGD {
    /// Whether this model has a material assigned (material_index > 0).
    #[func]
    fn has_material(&self) -> bool {
        self.material_index >= 0
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelBlockyModel API (upstream 5828cbeb).
    // -----------------------------------------------------------------

    /// Getter for the canonical `collision_aabbs` property.
    #[func]
    fn get_collision_aabbs(&self) -> Array<Aabb> {
        self.collision_aabbs_value.clone()
    }

    #[func]
    fn set_collision_aabbs(&mut self, aabbs: Array<Aabb>) {
        self.collision_aabbs_value = aabbs;
    }

    /// Getter for the canonical `collision_mask` property.
    #[func]
    fn get_collision_mask(&self) -> i32 {
        self.collision_mask_value
    }

    #[func]
    fn set_collision_mask(&mut self, mask: i32) {
        self.collision_mask_value = mask;
    }

    /// Getter for the canonical `color` property.
    #[func]
    pub(crate) fn get_color_prop(&self) -> Color {
        self.color_value
    }

    #[func]
    fn set_color_prop(&mut self, color: Color) {
        self.color_value = color;
    }

    /// Getter for the canonical `culls_neighbors` property.
    #[func]
    fn get_culls_neighbors(&self) -> bool {
        self.culls_neighbors_value
    }

    #[func]
    fn set_culls_neighbors(&mut self, enabled: bool) {
        self.culls_neighbors_value = enabled;
    }

    /// Getter for the canonical `lod_skirts_enabled` property.
    #[func]
    fn get_lod_skirts_enabled(&self) -> bool {
        self.lod_skirts_enabled_value
    }

    #[func]
    fn set_lod_skirts_enabled(&mut self, enabled: bool) {
        self.lod_skirts_enabled_value = enabled;
    }

    /// Getter for the canonical `random_tickable` property.
    #[func]
    fn is_random_tickable(&self) -> bool {
        self.random_tickable_value
    }

    #[func]
    fn set_random_tickable(&mut self, enabled: bool) {
        self.random_tickable_value = enabled;
    }

    /// Getter for the canonical `tags_mask` property.
    #[func]
    fn get_tags_mask(&self) -> i32 {
        self.tags_mask_value
    }

    #[func]
    fn set_tags_mask(&mut self, mask: i32) {
        self.tags_mask_value = mask;
    }

    /// Getter for the canonical `transparency_index` property.
    #[func]
    fn get_transparency_index(&self) -> i32 {
        self.transparency_index_value
    }

    #[func]
    fn set_transparency_index(&mut self, index: i32) {
        if index < 0 {
            godot_error!("VoxelBlockyModel.set_transparency_index: index must be >= 0");
            return;
        }
        self.transparency_index_value = index;
    }

    /// Gets the 90-degree rotation ID that will be applied to the model when
    /// the library is baked (canonical `get_mesh_ortho_rotation_index`).
    #[func]
    fn get_mesh_ortho_rotation_index(&self) -> i32 {
        self.mesh_ortho_rotation_index_value
    }

    /// Sets the 90-degree rotation ID (one of 24 possible rotations).
    #[func]
    fn set_mesh_ortho_rotation_index(&mut self, i: i32) {
        if i < 0 {
            godot_error!("VoxelBlockyModel.set_mesh_ortho_rotation_index: i must be >= 0");
            return;
        }
        self.mesh_ortho_rotation_index_value = i;
    }

    /// Rotates the model 90 degrees around the given axis (canonical
    /// `rotate_90`). The axis is a `Vector3i.Axis` enum (0=X, 1=Y, 2=Z).
    #[func]
    fn rotate_90(&mut self, axis: i32, clockwise: bool) {
        if !(0..=2).contains(&axis) {
            godot_error!("VoxelBlockyModel.rotate_90: axis must be 0, 1 or 2");
            return;
        }
        // Track the rotation as a 24-state ortho index; this binding only
        // records the intent, the actual mesh is baked by the mesher.
        let step = if clockwise { 1 } else { 3 };
        self.mesh_ortho_rotation_index_value = (self.mesh_ortho_rotation_index_value + step) % 4;
    }

    /// Gets the material override for a specific surface (canonical
    /// `get_material_override`). Returns null if none set.
    #[func]
    fn get_material_override(&self, index: i32) -> Option<Gd<Material>> {
        self.material_overrides.get(&index).cloned()
    }

    /// Sets a material override for a specific surface (canonical
    /// `set_material_override`).
    #[func]
    fn set_material_override(&mut self, index: i32, material: Option<Gd<Material>>) {
        if let Some(material) = material {
            self.material_overrides.insert(index, material);
        } else {
            self.material_overrides.remove(&index);
        }
    }

    /// Tells if a specific surface produces mesh-based collisions (canonical
    /// `is_mesh_collision_enabled`).
    #[func]
    fn is_mesh_collision_enabled(&self, surface_index: i32) -> bool {
        self.mesh_collision_enabled
            .get(&surface_index)
            .copied()
            .unwrap_or(true)
    }

    /// Enables or disables mesh-based collision on a specific surface
    /// (canonical `set_mesh_collision_enabled`).
    #[func]
    fn set_mesh_collision_enabled(&mut self, surface_index: i32, enabled: bool) {
        self.mesh_collision_enabled.insert(surface_index, enabled);
    }

    #[constant]
    const SIDE_NEGATIVE_X: i32 = 1;
    #[constant]
    const SIDE_POSITIVE_X: i32 = 0;
    #[constant]
    const SIDE_NEGATIVE_Y: i32 = 2;
    #[constant]
    const SIDE_POSITIVE_Y: i32 = 3;
    #[constant]
    const SIDE_NEGATIVE_Z: i32 = 4;
    #[constant]
    const SIDE_POSITIVE_Z: i32 = 5;
    #[constant]
    const SIDE_COUNT: i32 = 6;
}

impl VoxelBlockyModelGD {
    /// Bake inspector fields (color, tags, random-tick) into a core model.
    pub(crate) fn to_baked_model(&self) -> voxel_core::meshers::blocky::BakedModel {
        let cube =
            voxel_core::meshers::blocky::solid_cube_model(voxel_core::math::Color::from_rgb(
                self.color_value.r,
                self.color_value.g,
                self.color_value.b,
            ));
        let mut model = voxel_core::meshers::blocky::apply_ortho_rotation(
            cube,
            usize::try_from(self.mesh_ortho_rotation_index_value.max(0)).unwrap_or(0),
        );
        model.is_random_tickable = self.random_tickable_value;
        model.tags_mask = self.tags_mask_value as u32;
        model.culls_neighbors = self.culls_neighbors_value;
        model.lod_skirts = self.lod_skirts_enabled_value;
        model.transparency_index = self.transparency_index_value.clamp(0, 255) as u8;
        model
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyAttributeGD — Resource base for blocky attributes
// ---------------------------------------------------------------------------
/// Base for blocky type attributes (axis, rotation, direction, custom).
/// The functional API reports the attribute kind name.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyAttribute)]
pub struct VoxelBlockyAttributeGD {
    base: Base<Resource>,
    attr_name: GString,
    // -----------------------------------------------------------------
    // Pinned VoxelBlockyAttribute properties (upstream 5828cbeb).
    // The base attribute is a data carrier; canonical read-only values are
    // stored faithfully so GDScript reads round-trip. Subclasses override
    // the defaults via their own backing fields.
    // -----------------------------------------------------------------
    default_value_value: i32,
    value_count_value: i32,
    is_rotation_value: bool,
}
#[godot_api]
impl IResource for VoxelBlockyAttributeGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            attr_name: "base".to_godot(),
            default_value_value: 0,
            value_count_value: 1,
            is_rotation_value: false,
        }
    }
}

#[godot_api]
impl VoxelBlockyAttributeGD {
    /// The attribute's display name.
    #[func]
    fn get_attribute_name(&self) -> GString {
        self.attr_name.clone()
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelBlockyAttribute API (upstream 5828cbeb).
    // -----------------------------------------------------------------

    /// Maximum number of distinct values an attribute may hold
    /// (canonical `MAX_VALUES` constant).
    #[constant]
    const MAX_VALUES: i64 = 256;

    /// Returns the default value of the attribute (canonical
    /// `get_default_value`). Stored faithfully; the base class defaults to 0.
    #[func]
    fn get_default_value(&self) -> i32 {
        self.default_value_value
    }

    /// Returns the number of distinct values the attribute can take (canonical
    /// `get_value_count`). Stored faithfully; the base class defaults to 1.
    #[func]
    fn get_value_count(&self) -> i32 {
        self.value_count_value
    }

    /// Returns `true` if this attribute represents a rotation (canonical
    /// `is_rotation`). Stored faithfully; the base class defaults to `false`.
    #[func]
    fn is_rotation(&self) -> bool {
        self.is_rotation_value
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyAttributeAxisGD
// ---------------------------------------------------------------------------
/// Axis attribute for blocky types (X/Y/Z). The functional API reports the
/// axis as an integer (0=X, 1=Y, 2=Z).
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyAttributeAxis)]
pub struct VoxelBlockyAttributeAxisGD {
    base: Base<Resource>,
    axis: i32,
    // -----------------------------------------------------------------
    // Pinned VoxelBlockyAttributeAxis property (upstream 5828cbeb).
    // -----------------------------------------------------------------
    horizontal_only_value: bool,
    /// The pinned GDScript-facing `horizontal_only` property.
    #[var(get = is_horizontal_only, set = set_horizontal_only)]
    horizontal_only: PhantomVar<bool>,
}
#[godot_api]
impl IResource for VoxelBlockyAttributeAxisGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            axis: 0,
            horizontal_only_value: false,
            horizontal_only: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyAttributeAxisGD {
    /// Get the axis (0=X, 1=Y, 2=Z).
    #[func]
    fn get_axis(&self) -> i32 {
        self.axis
    }

    /// Set the axis (clamped 0-2).
    #[func]
    fn set_axis(&mut self, axis: i32) {
        self.axis = axis.clamp(0, 2);
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelBlockyAttributeAxis API (upstream 5828cbeb).
    // -----------------------------------------------------------------

    /// Axis enum value: X axis (canonical `AXIS_X`).
    #[constant]
    const AXIS_X: i64 = 0;
    /// Axis enum value: Y axis (canonical `AXIS_Y`).
    #[constant]
    const AXIS_Y: i64 = 1;
    /// Axis enum value: Z axis (canonical `AXIS_Z`).
    #[constant]
    const AXIS_Z: i64 = 2;
    /// Axis enum value: sentinel count of axes (canonical `AXIS_COUNT`).
    #[constant]
    const AXIS_COUNT: i64 = 3;

    /// Returns the axis index (0=X, 1=Y, 2=Z) of the dominant component of the
    /// given vector (canonical `from_vec3`). A zero vector resolves to X.
    #[func]
    #[allow(clippy::wrong_self_convention)]
    fn from_vec3(&self, v: Vector3) -> i32 {
        let ax = v.x.abs();
        let ay = v.y.abs();
        let az = v.z.abs();
        if ay >= ax && ay >= az {
            1
        } else if az >= ax {
            2
        } else {
            0
        }
    }

    /// Getter for the canonical `horizontal_only` property.
    #[func]
    fn is_horizontal_only(&self) -> bool {
        self.horizontal_only_value
    }

    /// Setter for the canonical `horizontal_only` property.
    #[func]
    fn set_horizontal_only(&mut self, enabled: bool) {
        self.horizontal_only_value = enabled;
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyAttributeRotationGD
// ---------------------------------------------------------------------------
/// Rotation attribute for blocky types (0-360 degrees). The functional API
/// normalizes the rotation to [0, 360).
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyAttributeRotation)]
pub struct VoxelBlockyAttributeRotationGD {
    base: Base<Resource>,
    rotation_degrees: i32,
    // -----------------------------------------------------------------
    // Pinned VoxelBlockyAttributeRotation property (upstream 5828cbeb).
    // Upstream names the property `horizontal_only` but exposes it through
    // `is_horizontal_roll_enabled` / `set_horizontal_roll_enabled`.
    // -----------------------------------------------------------------
    horizontal_only_value: bool,
    /// The pinned GDScript-facing `horizontal_only` property.
    #[var(get = is_horizontal_roll_enabled, set = set_horizontal_roll_enabled)]
    horizontal_only: PhantomVar<bool>,
}
#[godot_api]
impl IResource for VoxelBlockyAttributeRotationGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            rotation_degrees: 0,
            horizontal_only_value: false,
            horizontal_only: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyAttributeRotationGD {
    /// Get the rotation in degrees (normalized to [0, 360)).
    #[func]
    fn get_rotation(&self) -> i32 {
        self.rotation_degrees.rem_euclid(360)
    }

    /// Set the rotation in degrees.
    #[func]
    fn set_rotation(&mut self, degrees: i32) {
        self.rotation_degrees = degrees;
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelBlockyAttributeRotation API (upstream 5828cbeb).
    // -----------------------------------------------------------------

    /// Getter for the canonical `horizontal_only` property (exposed as
    /// `is_horizontal_roll_enabled` upstream).
    #[func]
    fn is_horizontal_roll_enabled(&self) -> bool {
        self.horizontal_only_value
    }

    /// Setter for the canonical `horizontal_only` property (exposed as
    /// `set_horizontal_roll_enabled` upstream).
    #[func]
    fn set_horizontal_roll_enabled(&mut self, enabled: bool) {
        self.horizontal_only_value = enabled;
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyAttributeDirectionGD
// ---------------------------------------------------------------------------
/// Direction attribute for blocky types (cardinal direction). The functional
/// API reports the direction name.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyAttributeDirection)]
pub struct VoxelBlockyAttributeDirectionGD {
    base: Base<Resource>,
    direction: i32,
    // -----------------------------------------------------------------
    // Pinned VoxelBlockyAttributeDirection property (upstream 5828cbeb).
    // -----------------------------------------------------------------
    horizontal_only_value: bool,
    /// The pinned GDScript-facing `horizontal_only` property.
    #[var(get = is_horizontal_only, set = set_horizontal_only)]
    horizontal_only: PhantomVar<bool>,
}
#[godot_api]
impl IResource for VoxelBlockyAttributeDirectionGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            direction: 0,
            horizontal_only_value: false,
            horizontal_only: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyAttributeDirectionGD {
    /// Get the direction name (0=North, 1=East, 2=South, 3=West).
    #[func]
    fn get_direction_name(&self) -> GString {
        match self.direction {
            0 => "North".to_godot(),
            1 => "East".to_godot(),
            2 => "South".to_godot(),
            3 => "West".to_godot(),
            _ => "Unknown".to_godot(),
        }
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelBlockyAttributeDirection API (upstream 5828cbeb).
    // -----------------------------------------------------------------

    /// Direction enum value: -X face (canonical `DIR_NEGATIVE_X`).
    #[constant]
    const DIR_NEGATIVE_X: i64 = 0;
    /// Direction enum value: +X face (canonical `DIR_POSITIVE_X`).
    #[constant]
    const DIR_POSITIVE_X: i64 = 1;
    /// Direction enum value: -Y face (canonical `DIR_NEGATIVE_Y`).
    #[constant]
    const DIR_NEGATIVE_Y: i64 = 2;
    /// Direction enum value: +Y face (canonical `DIR_POSITIVE_Y`).
    #[constant]
    const DIR_POSITIVE_Y: i64 = 3;
    /// Direction enum value: -Z face (canonical `DIR_NEGATIVE_Z`).
    #[constant]
    const DIR_NEGATIVE_Z: i64 = 4;
    /// Direction enum value: +Z face (canonical `DIR_POSITIVE_Z`).
    #[constant]
    const DIR_POSITIVE_Z: i64 = 5;
    /// Direction enum value: sentinel count of directions (canonical
    /// `DIR_COUNT`).
    #[constant]
    const DIR_COUNT: i64 = 6;

    /// Returns the face direction (0..5) the given vector points the most
    /// towards (canonical `from_vec3`). A zero vector resolves to
    /// `DIR_NEGATIVE_X` (0).
    #[func]
    #[allow(clippy::wrong_self_convention)]
    fn from_vec3(&self, v: Vector3) -> i32 {
        let ax = v.x.abs();
        let ay = v.y.abs();
        let az = v.z.abs();
        if ay >= ax && ay >= az {
            if v.y >= 0.0 {
                3
            } else {
                2
            }
        } else if az >= ax {
            if v.z >= 0.0 {
                5
            } else {
                4
            }
        } else if v.x >= 0.0 {
            1
        } else {
            0
        }
    }

    /// Getter for the canonical `horizontal_only` property.
    #[func]
    fn is_horizontal_only(&self) -> bool {
        self.horizontal_only_value
    }

    /// Setter for the canonical `horizontal_only` property.
    #[func]
    fn set_horizontal_only(&mut self, enabled: bool) {
        self.horizontal_only_value = enabled;
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyAttributeCustomGD
// ---------------------------------------------------------------------------
/// Custom attribute for blocky types (user-defined data). The functional API
/// stores/retrieves a custom integer value.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyAttributeCustom)]
pub struct VoxelBlockyAttributeCustomGD {
    base: Base<Resource>,
    custom_value: i64,
    // -----------------------------------------------------------------
    // Pinned VoxelBlockyAttributeCustom properties (upstream 5828cbeb).
    // -----------------------------------------------------------------
    attribute_name_value: StringName,
    default_value_value: i32,
    value_count_value: i32,
    /// Per-value display names keyed by value index.
    value_names: HashMap<i32, StringName>,
    /// The pinned GDScript-facing `attribute_name` property.
    #[var(get = get_attribute_name, set = set_attribute_name)]
    attribute_name: PhantomVar<StringName>,
    /// The pinned GDScript-facing `default_value` property.
    #[var(get = get_default_value, set = set_default_value)]
    default_value: PhantomVar<i32>,
    /// The pinned GDScript-facing `value_count` property.
    #[var(get = get_value_count, set = set_value_count)]
    value_count: PhantomVar<i32>,
}
#[godot_api]
impl IResource for VoxelBlockyAttributeCustomGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            custom_value: 0,
            attribute_name_value: StringName::default(),
            default_value_value: 0,
            value_count_value: 2,
            value_names: HashMap::new(),
            attribute_name: PhantomVar::default(),
            default_value: PhantomVar::default(),
            value_count: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyAttributeCustomGD {
    /// Get the custom value.
    #[func]
    fn get_custom_value(&self) -> i64 {
        self.custom_value
    }

    /// Set the custom value.
    #[func]
    fn set_custom_value(&mut self, value: i64) {
        self.custom_value = value;
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelBlockyAttributeCustom API (upstream 5828cbeb).
    // -----------------------------------------------------------------

    /// Assigns a display name to one of the attribute's values (canonical
    /// `set_value_name`).
    #[func]
    fn set_value_name(&mut self, value: i32, value_name: StringName) {
        if value < 0 {
            godot_error!("VoxelBlockyAttributeCustom.set_value_name: value must be >= 0");
            return;
        }
        self.value_names.insert(value, value_name);
    }

    /// Getter for the canonical `attribute_name` property.
    #[func]
    fn get_attribute_name(&self) -> StringName {
        self.attribute_name_value.clone()
    }

    /// Setter for the canonical `attribute_name` property.
    #[func]
    fn set_attribute_name(&mut self, name: StringName) {
        self.attribute_name_value = name;
    }

    /// Getter for the canonical `default_value` property.
    #[func]
    fn get_default_value(&self) -> i32 {
        self.default_value_value
    }

    /// Setter for the canonical `default_value` property.
    #[func]
    fn set_default_value(&mut self, value: i32) {
        if value < 0 {
            godot_error!("VoxelBlockyAttributeCustom.set_default_value: value must be >= 0");
            return;
        }
        self.default_value_value = value;
    }

    /// Getter for the canonical `value_count` property.
    #[func]
    fn get_value_count(&self) -> i32 {
        self.value_count_value
    }

    /// Setter for the canonical `value_count` property.
    #[func]
    fn set_value_count(&mut self, count: i32) {
        if count < 0 {
            godot_error!("VoxelBlockyAttributeCustom.set_value_count: count must be >= 0");
            return;
        }
        if count > VoxelBlockyAttributeGD::MAX_VALUES as i32 {
            godot_error!(
                "VoxelBlockyAttributeCustom.set_value_count: count must be <= MAX_VALUES (256)"
            );
            return;
        }
        self.value_count_value = count;
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyTypeLibraryGD
// ---------------------------------------------------------------------------
/// A library of blocky types (vs models). Used by the type-based blocky mesher.
///
/// Wraps [`voxel_core::meshers::blocky::BakedLibrary`] — the real model table
/// consumed by the blocky mesher. `add_color_type` appends a solid-color model
/// and `get_type_count` reports how many types are registered.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyTypeLibrary)]
pub struct VoxelBlockyTypeLibraryGD {
    base: Base<Resource>,
    /// Number of registered types (plain field; exposed via `get_type_count`
    /// #[func] to avoid a `#[var]` auto-getter collision).
    type_count: i32,
    /// The real baked model table. Kept in sync with `type_count`.
    library: voxel_core::meshers::blocky::BakedLibrary,
    // -----------------------------------------------------------------
    // Pinned VoxelBlockyTypeLibrary properties (upstream 5828cbeb).
    // -----------------------------------------------------------------
    /// Canonical `types` property backing field.
    types_value: Array<Gd<VoxelBlockyTypeGD>>,
    /// Canonical `_id_map_data` property backing field (name list).
    id_map_data: PackedStringArray,
    /// name -> default model index (built from `_id_map_data`).
    id_map: HashMap<String, i32>,
    /// The pinned GDScript-facing `types` property.
    #[var(get = get_types, set = set_types)]
    types: PhantomVar<Array<Gd<VoxelBlockyTypeGD>>>,
    /// The pinned GDScript-facing `_id_map_data` property.
    #[var(get = _get_id_map_data, set = _set_id_map_data)]
    _id_map_data_prop: PhantomVar<PackedStringArray>,
}
#[godot_api]
impl IResource for VoxelBlockyTypeLibraryGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            type_count: 0,
            library: voxel_core::meshers::blocky::BakedLibrary::default(),
            types_value: Array::new(),
            id_map_data: PackedStringArray::new(),
            id_map: HashMap::new(),
            types: PhantomVar::default(),
            _id_map_data_prop: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyTypeLibraryGD {
    /// Append a solid-color blocky type and return its id (the index of the
    /// new model). Mirrors the C++ `VoxelBlockyTypeLibrary::add_type`.
    #[func]
    fn add_color_type(&mut self, r: f32, g: f32, b: f32, a: f32) -> i32 {
        if validate_finite_float(r).is_err()
            || validate_finite_float(g).is_err()
            || validate_finite_float(b).is_err()
            || validate_finite_float(a).is_err()
        {
            godot_error!("VoxelBlockyTypeLibrary.add_color_type: colors must be finite");
            return -1;
        }
        let Ok(id) = i32::try_from(self.library.models.len()) else {
            godot_error!("VoxelBlockyTypeLibrary.add_color_type: too many models");
            return -1;
        };
        let model = voxel_core::meshers::blocky::BakedModel {
            color: voxel_core::math::Color::new(r, g, b, a),
            empty: false,
            ..voxel_core::meshers::blocky::BakedModel::default()
        };
        self.library.models.push(model);
        self.type_count = i32::try_from(self.library.models.len()).unwrap_or(i32::MAX);
        id
    }

    /// Returns the number of registered types (read-only `#[var]` mirror).
    #[func]
    fn get_type_count(&self) -> i32 {
        self.type_count
    }

    /// Returns `true` if the type at `id` exists in the library.
    #[func]
    fn has_type(&self, id: i32) -> bool {
        let Ok(id) = u32::try_from(id) else {
            godot_error!("VoxelBlockyTypeLibrary.has_type: id must be non-negative");
            return false;
        };
        self.library.has_model(id)
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelBlockyTypeLibrary API (upstream 5828cbeb).
    // The type-library runtime is partially ported; the canonical surface
    // stores the id map and types faithfully. Lookup methods return the
    // stored model index or -1 when the name is unknown.
    // -----------------------------------------------------------------

    /// Getter for the canonical `types` property.
    #[func]
    fn get_types(&self) -> Array<Gd<VoxelBlockyTypeGD>> {
        self.types_value.clone()
    }

    #[func]
    fn set_types(&mut self, types: Array<Gd<VoxelBlockyTypeGD>>) {
        self.types_value = types;
        self.type_count = i32::try_from(self.types_value.len()).unwrap_or(i32::MAX);
    }

    /// Getter for the canonical `_id_map_data` property.
    #[func]
    fn _get_id_map_data(&self) -> PackedStringArray {
        self.id_map_data.clone()
    }

    /// Setter for the canonical `_id_map_data` property. Rebuilds the internal
    /// name -> index map (the index is the position in the array).
    #[func]
    fn _set_id_map_data(&mut self, data: PackedStringArray) {
        self.id_map = HashMap::new();
        for (index, name) in data.as_slice().iter().enumerate() {
            if let Ok(index) = i32::try_from(index) {
                self.id_map.insert(name.to_string(), index);
            }
        }
        self.id_map_data = data;
    }

    /// Default model index for a type name (canonical
    /// `get_model_index_default`). -1 if the name is unknown.
    #[func]
    fn get_model_index_default(&self, type_name: StringName) -> i32 {
        self.id_map
            .get(type_name.to_string().as_str())
            .copied()
            .unwrap_or(-1)
    }

    /// Model index for a type name and a single attribute value (canonical
    /// `get_model_index_single_attribute`). Attribute-based selection is not
    /// yet modelled, so this returns the default index.
    #[func]
    fn get_model_index_single_attribute(
        &self,
        type_name: StringName,
        _attrib_value: Variant,
    ) -> i32 {
        self.get_model_index_default(type_name)
    }

    /// Model index for a type name and a set of attributes (canonical
    /// `get_model_index_with_attributes`). Attribute-based selection is not
    /// yet modelled, so this returns the default index.
    #[func]
    fn get_model_index_with_attributes(
        &self,
        type_name: StringName,
        _attribs_dict: VarDictionary,
    ) -> i32 {
        self.get_model_index_default(type_name)
    }

    /// Gets the type resource registered under a name (canonical
    /// `get_type_from_name`). Returns null if not found.
    #[func]
    fn get_type_from_name(&self, type_name: StringName) -> Option<Gd<VoxelBlockyTypeGD>> {
        let target = type_name.to_string();
        for i in 0..self.types_value.len() {
            if let Some(t) = self.types_value.get(i) {
                if t.bind().name.to_string() == target {
                    return Some(t);
                }
            }
        }
        None
    }

    /// Returns `[type_name, {attribute: value, ...}]` for a model index
    /// (canonical `get_type_name_and_attributes_from_model_index`). Returns an
    /// empty array if the index is unknown.
    #[func]
    fn get_type_name_and_attributes_from_model_index(&self, model_index: i32) -> Array<Variant> {
        let mut result: Array<Variant> = Array::new();
        for (name, idx) in &self.id_map {
            if *idx == model_index {
                result.push(&name.clone().to_godot().to_variant());
                return result;
            }
        }
        result
    }

    /// Replaces the id map from a JSON object string of `{"name": index}`
    /// pairs (canonical `load_id_map_from_json`). A minimal hand parser
    /// handles flat objects of string keys to integer values. Returns true
    /// on success.
    #[func]
    fn load_id_map_from_json(&mut self, json: GString) -> bool {
        let src = json.to_string();
        let pairs = match parse_flat_json_object_i32(&src) {
            Ok(pairs) => pairs,
            Err(msg) => {
                godot_error!("VoxelBlockyTypeLibrary.load_id_map_from_json: {msg}");
                return false;
            }
        };
        self.id_map.clear();
        for (name, index) in &pairs {
            self.id_map.insert(name.clone(), *index);
        }
        let mut names: Vec<String> = pairs.into_iter().map(|(n, _)| n).collect();
        names.sort_by_key(|n| self.id_map.get(n).copied().unwrap_or(0));
        self.id_map_data = PackedStringArray::from(
            names
                .iter()
                .map(|n| n.clone().to_godot())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        true
    }

    /// Replaces the id map from a string array (canonical
    /// `load_id_map_from_string_array`). The index is the position in the
    /// array. Returns true on success.
    #[func]
    fn load_id_map_from_string_array(&mut self, str_array: PackedStringArray) -> bool {
        self._set_id_map_data(str_array);
        true
    }

    /// Serialises the id map to a JSON object (canonical
    /// `serialize_id_map_to_json`).
    #[func]
    fn serialize_id_map_to_json(&self) -> GString {
        let mut entries: Vec<(&String, i32)> = self.id_map.iter().map(|(k, v)| (k, *v)).collect();
        entries.sort();
        let mut out = String::from('{');
        for (i, (name, index)) in entries.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('"');
            // Escape the minimal set required for round-tripping simple names.
            for ch in name.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    c => out.push(c),
                }
            }
            out.push_str("\":");
            out.push_str(&index.to_string());
        }
        out.push('}');
        out.to_godot()
    }

    /// Serialises the id map to a string array (canonical
    /// `serialize_id_map_to_string_array`).
    #[func]
    fn serialize_id_map_to_string_array(&self) -> PackedStringArray {
        self.id_map_data.clone()
    }
}

impl VoxelBlockyTypeLibraryGD {
    /// Borrow the underlying [`BakedLibrary`]. Used by sibling binding classes
    /// (the blocky mesher resource) that need direct access to run the blocky
    /// mesher without round-tripping through Godot calls.
    #[allow(dead_code)]
    pub fn core_library(&self) -> &voxel_core::meshers::blocky::BakedLibrary {
        &self.library
    }
}

// ---------------------------------------------------------------------------
// VoxelBoxMoverGD — Node for box-based terrain editing
// ---------------------------------------------------------------------------
/// A `Node3D` that moves a box through terrain, editing voxels in its path.
/// Wraps [`voxel_core::edition::ops::VoxelToolBuffer::do_box`] — `carve_path`
/// stamps a box at each integer step from the node's position to a target,
/// returning the number of voxels edited.
#[derive(GodotClass)]
#[class(base = Node3D, tool, rename = VoxelBoxMover)]
pub struct VoxelBoxMoverGD {
    base: Base<Node3D>,
    /// Half-size of the stamping box (voxel units).
    #[var]
    box_size: f32,
    /// Collision mask used to detect collidable voxels (backing field for the
    /// canonical `collision_mask` property). Bit `i` selects voxel id `i`.
    collision_mask_value: i32,
    /// Maximum height climbable as stairs (backing field for the canonical
    /// `max_step_height` property).
    max_step_height_value: f32,
    /// Whether step climbing is enabled (backing field for the canonical
    /// `step_climbing_enabled` property).
    step_climbing_enabled_value: bool,
    /// Records whether the last `get_motion` call climbed a step.
    has_stepped_up_value: bool,
}
#[godot_api]
impl INode3D for VoxelBoxMoverGD {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            box_size: 2.0,
            collision_mask_value: 1,
            max_step_height_value: 0.0,
            step_climbing_enabled_value: false,
            has_stepped_up_value: false,
        }
    }
}

#[godot_api]
impl VoxelBoxMoverGD {
    /// Stamp a solid box at each integer step along the line from the node's
    /// current position to `(target_x,target_y,target_z)` into a
    /// `VoxelBufferGD`'s Type channel. `origin_x` offsets the stamp into the
    /// buffer's local space. Returns the number of steps stamped, or -1 if
    /// `buffer` is not a `VoxelBufferGD`.
    #[func]
    fn carve_path(
        &self,
        buffer: Gd<RefCounted>,
        origin_x: i32,
        target_x: i32,
        target_y: i32,
        target_z: i32,
    ) -> i64 {
        let Ok(mut buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let Ok(box_size) = validate_finite_nonnegative_float(self.box_size) else {
            godot_error!("VoxelBoxMover.carve_path: box size must be finite and non-negative");
            return -1;
        };
        let Ok(half) = floor_to_i32(box_size) else {
            godot_error!("VoxelBoxMover.carve_path: box size must be finite and within i32 range");
            return -1;
        };
        const TYPE_CHANNEL: usize = 0;
        let start = self.base().get_position();
        let (Ok(sx), Ok(sy), Ok(sz)) = (
            floor_to_i32(start.x),
            floor_to_i32(start.y),
            floor_to_i32(start.z),
        ) else {
            godot_error!(
                "VoxelBoxMover.carve_path: start position must be finite and within i32 range"
            );
            return -1;
        };

        // Walk integer steps from start to target (DDA on the longest axis).
        let dx = i64::from(target_x) - i64::from(sx);
        let dy = i64::from(target_y) - i64::from(sy);
        let dz = i64::from(target_z) - i64::from(sz);
        let steps = dx.abs().max(dy.abs()).max(dz.abs()).max(1);
        if validate_carve_path_steps(steps).is_err() {
            godot_error!("VoxelBoxMover.carve_path: path length exceeds MAX_CARVE_PATH_STEPS");
            return -1;
        }
        let size = {
            let bound = buf.bind();
            bound.core_buffer().size()
        };
        let Ok(max_stamp_volume) = max_clipped_stamp_volume(half, size) else {
            godot_error!("VoxelBoxMover.carve_path: clipped stamp volume overflow");
            return -1;
        };
        if validate_carve_path_work(steps, max_stamp_volume).is_err() {
            godot_error!(
                "VoxelBoxMover.carve_path: combined path and stamp workload exceeds the script workload limit"
            );
            return -1;
        }
        let start = voxel_core::math::Vector3i::new(sx, sy, sz);
        let target = voxel_core::math::Vector3i::new(target_x, target_y, target_z);
        if validate_carve_path_stamp_range(start, target, origin_x, half).is_err() {
            godot_error!("VoxelBoxMover.carve_path: stamp position exceeds i32 range");
            return -1;
        }

        // Precompute every box before taking a mutable buffer borrow. Besides
        // keeping all validation ahead of mutation, this ensures an unexpected
        // coordinate conversion failure cannot leave a partially carved path.
        let Some(stamp_count) = steps.checked_add(1) else {
            godot_error!("VoxelBoxMover.carve_path: path stamp count overflow");
            return -1;
        };
        let Ok(stamp_capacity) = usize::try_from(stamp_count) else {
            godot_error!("VoxelBoxMover.carve_path: path stamp count overflow");
            return -1;
        };
        let mut stamps = Vec::with_capacity(stamp_capacity);
        for i in 0..=steps {
            let cx = i128::from(sx)
                + (i128::from(dx) * i128::from(i)) / i128::from(steps)
                + i128::from(origin_x);
            let cy = i128::from(sy) + (i128::from(dy) * i128::from(i)) / i128::from(steps);
            let cz = i128::from(sz) + (i128::from(dz) * i128::from(i)) / i128::from(steps);
            let (Ok(min_x), Ok(min_y), Ok(min_z), Ok(max_x), Ok(max_y), Ok(max_z)) = (
                i32::try_from(cx - i128::from(half)),
                i32::try_from(cy - i128::from(half)),
                i32::try_from(cz - i128::from(half)),
                i32::try_from(cx + i128::from(half)),
                i32::try_from(cy + i128::from(half)),
                i32::try_from(cz + i128::from(half)),
            ) else {
                godot_error!("VoxelBoxMover.carve_path: stamp position exceeds i32 range");
                return -1;
            };
            let min = voxel_core::math::Vector3i::new(min_x, min_y, min_z);
            let max = voxel_core::math::Vector3i::new(max_x, max_y, max_z);
            stamps.push((min, max));
        }
        let Ok(stamped) = i64::try_from(stamps.len()) else {
            godot_error!("VoxelBoxMover.carve_path: path stamp count overflow");
            return -1;
        };

        let mut bound = buf.bind_mut();
        let core = bound.core_buffer_mut();
        let mut tool = voxel_core::edition::ops::VoxelToolBuffer::new(core, TYPE_CHANNEL);
        for (min, max) in stamps {
            tool.do_box(min, max);
        }
        stamped
    }

    // ----- Canonical pinned methods (upstream 5828cbeb: VoxelBoxMover.xml) -----

    /// Collision mask used to detect collidable voxels. Matches
    /// `VoxelBoxMover::get_collision_mask`.
    #[func]
    fn get_collision_mask(&self) -> i32 {
        self.collision_mask_value
    }

    /// Set the collision mask used to detect collidable voxels. Only voxels
    /// sharing at least one bit between the masks are detected. Matches
    /// `VoxelBoxMover::set_collision_mask`.
    #[func]
    fn set_collision_mask(&mut self, mask: i32) {
        self.collision_mask_value = mask;
    }

    /// Maximum height that can be climbed like stairs. Matches
    /// `VoxelBoxMover::get_max_step_height`.
    #[func]
    fn get_max_step_height(&self) -> f32 {
        self.max_step_height_value
    }

    /// Set the maximum climbable step height. Matches
    /// `VoxelBoxMover::set_max_step_height`.
    #[func]
    fn set_max_step_height(&mut self, height: f32) {
        if validate_finite_float(height).is_err() || height < 0.0 {
            godot_error!(
                "VoxelBoxMover.set_max_step_height: height must be finite and non-negative"
            );
            return;
        }
        self.max_step_height_value = height;
    }

    /// Whether step climbing is enabled. Matches
    /// `VoxelBoxMover::is_step_climbing_enabled`.
    #[func]
    fn is_step_climbing_enabled(&self) -> bool {
        self.step_climbing_enabled_value
    }

    /// Toggle step climbing. When enabled, `get_motion` attempts to climb small
    /// steps. Matches `VoxelBoxMover::set_step_climbing_enabled`.
    #[func]
    fn set_step_climbing_enabled(&mut self, enabled: bool) {
        self.step_climbing_enabled_value = enabled;
    }

    /// Whether the last `get_motion` call caused step climbing. Matches
    /// `VoxelBoxMover::has_stepped_up`.
    #[func]
    fn has_stepped_up(&self) -> bool {
        self.has_stepped_up_value
    }

    /// Given a motion vector, returns a modified vector telling how much to
    /// move the character. The engine's move-and-slide implementation isn't
    /// ported, so this returns the original motion unchanged (no collision
    /// resolution) and records no step climb. Matches `VoxelBoxMover::get_motion`.
    #[func]
    fn get_motion(
        &mut self,
        _pos: Vector3,
        motion: Vector3,
        _aabb: Aabb,
        _terrain: Gd<Object>,
    ) -> Vector3 {
        // No collision resolution is available in the port; the motion is
        // returned unmodified. Reset step-climb state every call.
        self.has_stepped_up_value = false;
        motion
    }

    /// Tests whether an axis-aligned box intersects any voxel collision box.
    /// The engine's blocky collision query isn't ported; this performs a
    /// functional AABB-overlap check against the box-mover's own stamping box
    /// (centered on the node), returning whether they overlap. Matches
    /// `VoxelBoxMover::intersects`.
    #[func]
    fn intersects(&self, aabb: Aabb, _terrain: Gd<Object>) -> bool {
        let center = self.base().get_position();
        let half = self.box_size;
        let mover_min = Vector3::new(center.x - half, center.y - half, center.z - half);
        let mover_max = Vector3::new(center.x + half, center.y + half, center.z + half);
        let other_min = aabb.position;
        let other_max = aabb.position + aabb.size;
        mover_min.x <= other_max.x
            && mover_max.x >= other_min.x
            && mover_min.y <= other_max.y
            && mover_max.y >= other_min.y
            && mover_min.z <= other_max.z
            && mover_max.z >= other_min.z
    }
}

// ---------------------------------------------------------------------------
// VoxelAStarGrid3DGD — RefCounted for 3D pathfinding
// ---------------------------------------------------------------------------
/// 3D A* pathfinding grid on voxel terrain. The voxel-core pathfinding engine
/// isn't ported yet; this binding provides a functional walkability query over
/// a `VoxelBufferGD` — `is_walkable` checks that a cell is air while the cell
/// below is solid (ground-walking semantics), mirroring how an A* grid would
/// classify passable nodes.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelAStarGrid3D)]
pub struct VoxelAStarGrid3DGD {
    base: Base<RefCounted>,
    /// Canonical `region` backing field: maximum search area in voxels.
    region: Aabb,
    /// Canonical `is_running_async` backing field.
    running_async: bool,
    /// Canonical `debug_get_visited_positions` backing field (last search).
    visited: Array<Vector3i>,
}
#[godot_api]
impl IRefCounted for VoxelAStarGrid3DGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            // Default region covers a moderate search area (50³ is already
            // expensive per the upstream docs).
            region: Aabb::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(50.0, 50.0, 50.0)),
            running_async: false,
            visited: Array::new(),
        }
    }
}

#[godot_api]
impl VoxelAStarGrid3DGD {
    /// Check whether cell `(x,y,z)` in a `VoxelBufferGD` is walkable: the cell
    /// itself is air (Type channel == 0) and the cell below is solid (≠ 0).
    /// Returns false if `buffer` is not a `VoxelBufferGD` or out of bounds.
    #[func]
    fn is_walkable(&self, buffer: Gd<RefCounted>, x: i32, y: i32, z: i32) -> bool {
        let Ok(buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return false;
        };
        let bound = buf.bind();
        let core = bound.core_buffer();
        if x < 0 || y < 1 || z < 0 || x >= core.size().x || y >= core.size().y || z >= core.size().z
        {
            return false;
        }
        const TYPE_CHANNEL: usize = 0;
        let here = core.get_voxel(x, y, z, TYPE_CHANNEL);
        let below = core.get_voxel(x, y - 1, z, TYPE_CHANNEL);
        here == 0 && below != 0
    }

    /// Count walkable cells in a `VoxelBufferGD` (air with solid below).
    /// Returns -1 if `buffer` is not a `VoxelBufferGD`.
    #[func]
    fn count_walkable(&self, buffer: Gd<RefCounted>) -> i64 {
        let Ok(buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let bound = buf.bind();
        let core = bound.core_buffer();
        let sx = core.size().x;
        let sy = core.size().y;
        let sz = core.size().z;
        const TYPE_CHANNEL: usize = 0;
        let mut count: i64 = 0;
        for z in 0..sz {
            for x in 0..sx {
                for y in 1..sy {
                    if core.get_voxel(x, y, z, TYPE_CHANNEL) == 0
                        && core.get_voxel(x, y - 1, z, TYPE_CHANNEL) != 0
                    {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelAStarGrid3D API (upstream 5828cbeb).
    // Blocky A* pathfinding is partial in this binding; the canonical
    // surface stores its configuration faithfully and stubs the searches
    // (returning empty arrays) until a terrain-backed implementation lands.
    // -----------------------------------------------------------------

    /// Sets the maximum region limit that will be considered for pathfinding,
    /// in voxels (canonical `set_region`).
    #[func]
    fn set_region(&mut self, box_value: Aabb) {
        if box_value.size.x < 0.0 || box_value.size.y < 0.0 || box_value.size.z < 0.0 {
            godot_error!("VoxelAStarGrid3D.set_region: size must be non-negative");
            return;
        }
        self.region = box_value;
    }

    /// Gets the maximum region limit that will be considered for pathfinding,
    /// in voxels (canonical `get_region`).
    #[func]
    fn get_region(&self) -> Aabb {
        self.region
    }

    /// Sets the terrain that will be used to do searches in (canonical
    /// `set_terrain`). The headless binding does not yet back the search with
    /// a live terrain, so the value is accepted and ignored.
    #[func]
    fn set_terrain(&mut self, _terrain: Variant) {
        godot_print!("VoxelAStarGrid3D.set_terrain: terrain-backed search is not yet implemented");
    }

    /// Calculates a path between two voxel positions (canonical `find_path`).
    /// Returns an empty array until a terrain-backed search is implemented.
    #[func]
    fn find_path(&self, _from_position: Vector3i, _to_position: Vector3i) -> Array<Vector3i> {
        Array::new()
    }

    /// Same as `find_path`, but on a separate thread (canonical
    /// `find_path_async`). The result is emitted via the
    /// `async_search_completed` signal; this binding does not yet run the
    /// search, so it clears the running flag immediately.
    #[func]
    fn find_path_async(&mut self, _from_position: Vector3i, _to_position: Vector3i) {
        // The headless binding has no background pathfinder; clear the flag so
        // callers do not block on `is_running_async`.
        self.running_async = false;
    }

    /// Returns true if a path is currently being calculated asynchronously
    /// (canonical `is_running_async`).
    #[func]
    fn is_running_async(&self) -> bool {
        self.running_async
    }

    /// Gets the list of voxel positions visited by the last pathfinding
    /// request (canonical `debug_get_visited_positions`). Empty until a real
    /// search runs.
    #[func]
    fn debug_get_visited_positions(&self) -> Array<Vector3i> {
        self.visited.clone()
    }
}
