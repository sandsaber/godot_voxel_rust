//! Shape operations and SDF blending for single-buffer voxel editing.
//!
//! Ports `DoShapeSingleBuffer` and SDF shape/operation types from
//! `edition/funcs.h`. All operations work on a single [`VoxelBuffer`]
//! and are engine-agnostic (no Godot dependency).

use crate::math::{Vector3f, Vector3i};
use crate::meshers::blocky::BakedLibrary;
use crate::storage::{ChannelDepth, ChannelId, VoxelBuffer};

/// Edit mode (add/remove/set). Matches C++ `Mode` in `funcs.h:492`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditMode {
    Add,
    Remove,
    Set,
}

/// SDF blend mode for combining the shape's SDF with the existing voxel.
/// Matches C++ SDF operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdfBlendMode {
    /// `min(existing, shape_sdf)` — union (grow).
    Union,
    /// `max(existing, -shape_sdf)` — subtract (carve).
    Subtract,
    /// Replace the voxel value entirely.
    Set,
}

/// A voxel tool that edits a single [`VoxelBuffer`]. Engine-agnostic
/// equivalent of C++ `VoxelToolBuffer`.
pub struct VoxelToolBuffer<'a> {
    buffer: &'a mut VoxelBuffer,
    channel: usize,
    mode: EditMode,
    /// Value written for blocky `Set` mode (raw u64).
    value: u64,
}

impl<'a> VoxelToolBuffer<'a> {
    pub fn new(buffer: &'a mut VoxelBuffer, channel: usize) -> Self {
        Self {
            buffer,
            channel,
            mode: EditMode::Set,
            value: 1,
        }
    }

    pub fn with_mode(mut self, mode: EditMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_value(mut self, value: u64) -> Self {
        self.value = value;
        self
    }

    /// Edit a sphere at `center` with `radius`. SDF channels use smooth
    /// blending; blocky channels use value set/add/remove.
    pub fn do_sphere(&mut self, center: Vector3f, radius: f32) {
        do_sphere(
            self.buffer,
            self.channel,
            self.mode,
            self.value,
            center,
            radius,
        );
    }

    /// Edit an axis-aligned box from `min` to `max`.
    pub fn do_box(&mut self, min: Vector3i, max: Vector3i) {
        do_box(self.buffer, self.channel, self.mode, self.value, min, max);
    }

    /// Edit a hemisphere. `flat_direction` is the outward normal of the flat face.
    pub fn do_hemisphere(
        &mut self,
        center: Vector3f,
        radius: f32,
        flat_direction: Vector3f,
        smoothness: f32,
    ) {
        do_hemisphere(
            self.buffer,
            self.channel,
            self.mode,
            self.value,
            center,
            radius,
            flat_direction,
            smoothness,
        );
    }

    /// Smooth the SDF channel inside a sphere of influence.
    pub fn do_smooth(&mut self, center: Vector3f, radius: f32, blur_radius: i32) {
        do_smooth(self.buffer, self.channel, center, radius, blur_radius);
    }

    /// Set a single voxel at integer position.
    pub fn set_voxel(&mut self, pos: Vector3i, value: u64) {
        if pos.x < 0 || pos.y < 0 || pos.z < 0 {
            return;
        }
        let size = self.buffer.size();
        if pos.x >= size.x || pos.y >= size.y || pos.z >= size.z {
            return;
        }
        self.buffer
            .set_voxel(value, pos.x, pos.y, pos.z, self.channel);
    }

    /// Read-only access to the underlying buffer. Used by the Godot binding
    /// to inspect edit results (e.g. count solid voxels after a carve).
    pub fn buffer(&self) -> &VoxelBuffer {
        self.buffer
    }
}

/// Apply a sphere edit to a VoxelBuffer's channel.
pub fn do_sphere(
    buffer: &mut VoxelBuffer,
    channel: usize,
    mode: EditMode,
    value: u64,
    center: Vector3f,
    radius: f32,
) {
    let depth = buffer.channel_depth(channel);
    let is_sdf = channel == ChannelId::Sdf.index();
    let size = buffer.size();

    // Compute the integer bounding box of the sphere.
    let min = Vector3i::new(
        (center.x - radius).floor() as i32,
        (center.y - radius).floor() as i32,
        (center.z - radius).floor() as i32,
    )
    .max_element(Vector3i::zero());
    let max = Vector3i::new(
        (center.x + radius).ceil() as i32,
        (center.y + radius).ceil() as i32,
        (center.z + radius).ceil() as i32,
    )
    .min_element(size);
    if min.x >= max.x || min.y >= max.y || min.z >= max.z {
        return;
    }

    let r2 = radius * radius;
    buffer.decompress_channel(channel);

    for z in min.z..max.z {
        for y in min.y..max.y {
            for x in min.x..max.x {
                let dx = x as f32 + 0.5 - center.x;
                let dy = y as f32 + 0.5 - center.y;
                let dz = z as f32 + 0.5 - center.z;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let inside = dist_sq <= r2;

                if !inside {
                    continue;
                }

                if is_sdf && depth != ChannelDepth::Bit8 {
                    // Smooth SDF blending. SDF convention: negative = inside
                    // (solid), positive = outside (air). Shape SDF is
                    // `dist - radius` (negative inside the sphere).
                    let sdf = dist_sq.sqrt() - radius;
                    let existing = buffer.get_voxel_f(x, y, z, channel);
                    let blended = blend_sdf(existing, sdf, mode);
                    buffer.set_voxel_f(blended, x, y, z, channel);
                } else {
                    // Blocky value mode.
                    match mode {
                        EditMode::Add => {
                            let cur = buffer.get_voxel(x, y, z, channel);
                            if cur == 0 {
                                buffer.set_voxel(value, x, y, z, channel);
                            }
                        }
                        EditMode::Remove => {
                            buffer.set_voxel(0, x, y, z, channel);
                        }
                        EditMode::Set => {
                            buffer.set_voxel(value, x, y, z, channel);
                        }
                    }
                }
            }
        }
    }
}

/// Apply a box edit to a VoxelBuffer's channel.
pub fn do_box(
    buffer: &mut VoxelBuffer,
    channel: usize,
    mode: EditMode,
    value: u64,
    min: Vector3i,
    max: Vector3i,
) {
    let depth = buffer.channel_depth(channel);
    let is_sdf = channel == ChannelId::Sdf.index();
    let size = buffer.size();
    let lo = min.max_element(Vector3i::zero());
    let hi = max.min_element(size);
    if lo.x >= hi.x || lo.y >= hi.y || lo.z >= hi.z {
        return;
    }

    buffer.decompress_channel(channel);

    if is_sdf && depth != ChannelDepth::Bit8 {
        for z in lo.z..hi.z {
            for y in lo.y..hi.y {
                for x in lo.x..hi.x {
                    let pos = Vector3i::new(x, y, z);
                    let sdf = sdf_box(pos, lo, hi);
                    let existing = buffer.get_voxel_f(x, y, z, channel);
                    let blended = blend_sdf(existing, sdf, mode);
                    buffer.set_voxel_f(blended, x, y, z, channel);
                }
            }
        }
    } else {
        for z in lo.z..hi.z {
            for y in lo.y..hi.y {
                for x in lo.x..hi.x {
                    match mode {
                        EditMode::Add => {
                            let cur = buffer.get_voxel(x, y, z, channel);
                            if cur == 0 {
                                buffer.set_voxel(value, x, y, z, channel);
                            }
                        }
                        EditMode::Remove => {
                            buffer.set_voxel(0, x, y, z, channel);
                        }
                        EditMode::Set => {
                            buffer.set_voxel(value, x, y, z, channel);
                        }
                    }
                }
            }
        }
    }
}

/// Signed distance of a hemisphere: the intersection of a sphere and the
/// half-space opposite `flat_direction` (the outward normal of the flat face).
/// `smoothness` > 0 rounds the crease with a polynomial smooth intersection.
pub fn hemisphere_sdf(
    point: Vector3f,
    center: Vector3f,
    radius: f32,
    flat_direction: Vector3f,
    smoothness: f32,
) -> f32 {
    let dx = point.x - center.x;
    let dy = point.y - center.y;
    let dz = point.z - center.z;
    let sphere = (dx * dx + dy * dy + dz * dz).sqrt() - radius;
    let len_sq = flat_direction.x * flat_direction.x
        + flat_direction.y * flat_direction.y
        + flat_direction.z * flat_direction.z;
    let (nx, ny, nz) = if len_sq > 1e-16 {
        let inv = len_sq.sqrt().recip();
        (
            flat_direction.x * inv,
            flat_direction.y * inv,
            flat_direction.z * inv,
        )
    } else {
        (0.0, 1.0, 0.0)
    };
    let plane = dx * nx + dy * ny + dz * nz;
    smooth_intersection(sphere, plane, smoothness.max(0.0))
}

fn smooth_union(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b * (1.0 - h) + a * h - k * h * (1.0 - h)
}

fn smooth_intersection(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.max(b);
    }
    -smooth_union(-a, -b, k)
}

/// Apply a hemisphere edit to a VoxelBuffer's channel.
#[allow(clippy::too_many_arguments)]
pub fn do_hemisphere(
    buffer: &mut VoxelBuffer,
    channel: usize,
    mode: EditMode,
    value: u64,
    center: Vector3f,
    radius: f32,
    flat_direction: Vector3f,
    smoothness: f32,
) {
    let depth = buffer.channel_depth(channel);
    let is_sdf = channel == ChannelId::Sdf.index();
    let size = buffer.size();
    let pad = radius + smoothness.max(0.0);
    let min = Vector3i::new(
        (center.x - pad).floor() as i32,
        (center.y - pad).floor() as i32,
        (center.z - pad).floor() as i32,
    )
    .max_element(Vector3i::zero());
    let max = Vector3i::new(
        (center.x + pad).ceil() as i32,
        (center.y + pad).ceil() as i32,
        (center.z + pad).ceil() as i32,
    )
    .min_element(size);
    if min.x >= max.x || min.y >= max.y || min.z >= max.z {
        return;
    }

    buffer.decompress_channel(channel);

    for z in min.z..max.z {
        for y in min.y..max.y {
            for x in min.x..max.x {
                let point = Vector3f::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                let sdf = hemisphere_sdf(point, center, radius, flat_direction, smoothness);
                if sdf > 0.0 {
                    continue;
                }
                if is_sdf && depth != ChannelDepth::Bit8 {
                    let existing = buffer.get_voxel_f(x, y, z, channel);
                    let blended = blend_sdf(existing, sdf, mode);
                    buffer.set_voxel_f(blended, x, y, z, channel);
                } else {
                    match mode {
                        EditMode::Add => {
                            let cur = buffer.get_voxel(x, y, z, channel);
                            if cur == 0 {
                                buffer.set_voxel(value, x, y, z, channel);
                            }
                        }
                        EditMode::Remove => {
                            buffer.set_voxel(0, x, y, z, channel);
                        }
                        EditMode::Set => {
                            buffer.set_voxel(value, x, y, z, channel);
                        }
                    }
                }
            }
        }
    }
}

/// Smooth the SDF channel inside a sphere of influence using [`box_blur`].
pub fn do_smooth(
    buffer: &mut VoxelBuffer,
    channel: usize,
    center: Vector3f,
    radius: f32,
    blur_radius: i32,
) {
    if channel != ChannelId::Sdf.index() || !radius.is_finite() || radius < 0.0 {
        return;
    }
    let mut src = VoxelBuffer::with_size(buffer.size());
    src.set_channel_depth(channel, buffer.channel_depth(channel));
    src.copy_channel_from(buffer, channel);
    box_blur(&src, buffer, blur_radius.max(0), center, radius);
}

/// Blend the shape SDF with the existing voxel SDF value.
pub fn blend_sdf(existing: f32, shape_sdf: f32, mode: EditMode) -> f32 {
    match mode {
        EditMode::Add => existing.min(shape_sdf),
        EditMode::Remove => existing.max(-shape_sdf),
        EditMode::Set => shape_sdf,
    }
}

/// For box SDF: compute the signed distance from point to the box boundary.
/// Negative = inside the box (solid), positive = outside (air).
fn sdf_box(pos: Vector3i, lo: Vector3i, hi: Vector3i) -> f32 {
    let cx = pos.x as f32 + 0.5;
    let cy = pos.y as f32 + 0.5;
    let cz = pos.z as f32 + 0.5;
    // Box half-extents from center.
    let hx = (hi.x - lo.x) as f32 * 0.5;
    let hy = (hi.y - lo.y) as f32 * 0.5;
    let hz = (hi.z - lo.z) as f32 * 0.5;
    let bcx = lo.x as f32 + hx;
    let bcy = lo.y as f32 + hy;
    let bcz = lo.z as f32 + hz;
    // Distance from center, positive outside the half-extent.
    let dx = (cx - bcx).abs() - hx;
    let dy = (cy - bcy).abs() - hy;
    let dz = (cz - bcz).abs() - hz;
    // SDF: max(dx,dy,dz) when outside, min(dx,dy,dz) when inside.
    let outside = dx.max(dy).max(dz).max(0.0);
    let inside = dx.max(dy).max(dz).min(0.0);
    outside + inside
}

trait Vector3iExt {
    fn min_element(self, other: Self) -> Self;
    fn max_element(self, other: Self) -> Self;
}

impl Vector3iExt for Vector3i {
    fn min_element(self, other: Self) -> Self {
        Vector3i::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }
    fn max_element(self, other: Self) -> Self {
        Vector3i::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }
}

/// Box blur of the SDF channel. Averages each voxel's SDF value over a cube
/// of side `(2*radius+1)` centered on it, but only within a sphere of
/// influence centered at `sphere_center` with `sphere_radius`. Voxels outside
/// the sphere are left unchanged. This is the simple reference implementation
/// matching `ops::box_blur_slow_ref` from `edition/funcs.h`.
///
/// The destination buffer must have the same size and SDF channel depth as
/// the source.
pub fn box_blur(
    src: &VoxelBuffer,
    dst: &mut VoxelBuffer,
    radius: i32,
    sphere_center: Vector3f,
    sphere_radius: f32,
) {
    let channel = ChannelId::Sdf.index();
    let size = src.size();
    debug_assert_eq!(dst.size(), size);

    let r2 = sphere_radius * sphere_radius;

    for z in 0..size.z {
        for y in 0..size.y {
            for x in 0..size.x {
                // Check sphere of influence.
                let dx = x as f32 - sphere_center.x;
                let dy = y as f32 - sphere_center.y;
                let dz = z as f32 - sphere_center.z;
                if dx * dx + dy * dy + dz * dz > r2 {
                    // Outside sphere → copy source value unchanged.
                    dst.set_voxel_f(src.get_voxel_f(x, y, z, channel), x, y, z, channel);
                    continue;
                }

                // Average over the cube.
                let lo_x = (x - radius).max(0);
                let hi_x = (x + radius + 1).min(size.x);
                let lo_y = (y - radius).max(0);
                let hi_y = (y + radius + 1).min(size.y);
                let lo_z = (z - radius).max(0);
                let hi_z = (z + radius + 1).min(size.z);

                let mut sum = 0.0f64;
                let mut count = 0u32;
                for bz in lo_z..hi_z {
                    for by in lo_y..hi_y {
                        for bx in lo_x..hi_x {
                            sum += src.get_voxel_f(bx, by, bz, channel) as f64;
                            count += 1;
                        }
                    }
                }

                let avg = if count > 0 {
                    (sum / count as f64) as f32
                } else {
                    src.get_voxel_f(x, y, z, channel)
                };
                dst.set_voxel_f(avg, x, y, z, channel);
            }
        }
    }
}

/// Whether a voxel id should participate in blocky random-tick.
///
/// When a baked library is present the model must be `is_random_tickable`,
/// and `tags_mask` (when non-zero) must intersect the model's tag bits.
/// Without a library, non-zero voxels are candidates; a non-zero `tags_mask`
/// then filters by `(voxel_id as u32) & tags_mask`.
pub fn voxel_is_random_tick_candidate(
    voxel: u64,
    tags_mask: u32,
    library: Option<&BakedLibrary>,
) -> bool {
    if voxel == 0 {
        return false;
    }
    if let Some(library) = library {
        let Some(model) = library.models.get(voxel as usize) else {
            return false;
        };
        if !model.is_random_tickable {
            return false;
        }
        return tags_mask == 0 || (model.tags_mask & tags_mask) != 0;
    }
    tags_mask == 0 || ((voxel as u32) & tags_mask) != 0
}

/// Run blocky random tick: iterate over random tickable voxels within a box
/// and invoke `callback` for each one selected. Returns the number of
/// callbacks invoked. Matches `ops::run_blocky_random_tick` semantics:
/// each voxel with `is_random_tickable` set in the baked model at that index
/// is a candidate; we randomly select up to `batch_count` per iteration.
pub fn run_blocky_random_tick<F: FnMut(Vector3i)>(
    buf: &VoxelBuffer,
    voxel_box: crate::math::Box3i,
    tickable_id: u64,
    channel: usize,
    batch_count: usize,
    seed: u32,
    callback: F,
) {
    let mut callback = callback;
    let size = buf.size();

    // Collect candidate positions.
    let mut candidates = Vec::new();
    for pos in voxel_box.iter_cells_zxy() {
        if pos.x < 0 || pos.y < 0 || pos.z < 0 {
            continue;
        }
        if pos.x >= size.x || pos.y >= size.y || pos.z >= size.z {
            continue;
        }
        let voxel = buf.get_voxel(pos.x, pos.y, pos.z, channel);
        if voxel == tickable_id {
            candidates.push(pos);
        }
    }

    if candidates.is_empty() {
        return;
    }

    // Draw uniformly random candidates per call (upstream semantics: a
    // fixed-stride subset would re-tick the same positions forever and
    // permanently starve the rest). Deterministic under a fixed seed.
    let mut rng = crate::instancing::scatter::SimpleRng::new(seed);
    let draws = batch_count.min(candidates.len());
    for _ in 0..draws {
        let index = (rng.next_u32() as usize) % candidates.len();
        callback(candidates[index]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{ChannelDepth, VoxelFormat};

    fn make_buffer(size: i32) -> VoxelBuffer {
        let mut buf = VoxelBuffer::with_size(Vector3i::splat(size));
        let mut fmt = VoxelFormat::new();
        fmt.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        fmt.configure_buffer(&mut buf);
        buf
    }

    #[test]
    fn random_tick_candidate_honors_tags_and_library() {
        assert!(!voxel_is_random_tick_candidate(0, 0, None));
        assert!(voxel_is_random_tick_candidate(3, 0, None));
        assert!(voxel_is_random_tick_candidate(3, 1, None));
        assert!(!voxel_is_random_tick_candidate(2, 1, None));

        let library = crate::meshers::blocky::BakedLibrary {
            models: vec![
                crate::meshers::blocky::BakedModel::default(),
                crate::meshers::blocky::BakedModel {
                    is_random_tickable: true,
                    tags_mask: 0b01,
                    ..crate::meshers::blocky::BakedModel::default()
                },
                crate::meshers::blocky::BakedModel {
                    is_random_tickable: false,
                    tags_mask: 0b01,
                    ..crate::meshers::blocky::BakedModel::default()
                },
            ],
            ..crate::meshers::blocky::BakedLibrary::default()
        };
        assert!(voxel_is_random_tick_candidate(1, 0, Some(&library)));
        assert!(voxel_is_random_tick_candidate(1, 0b01, Some(&library)));
        assert!(!voxel_is_random_tick_candidate(1, 0b10, Some(&library)));
        assert!(!voxel_is_random_tick_candidate(2, 0, Some(&library)));
    }

    #[test]
    fn do_sphere_add_creates_solid_in_sdf() {
        let mut buf = make_buffer(16);
        let ch = ChannelId::Sdf.index();
        // Start fully outside (air).
        buf.clear_channel_f(ch, 100.0); // SDF_FAR_OUTSIDE

        do_sphere(
            &mut buf,
            ch,
            EditMode::Add,
            1,
            Vector3f::new(8.0, 8.0, 8.0),
            4.0,
        );

        // Center should be inside (negative SDF).
        let center = buf.get_voxel_f(8, 8, 8, ch);
        assert!(center < 0.0, "center should be inside solid, got {center}");

        // Corner should still be outside.
        let corner = buf.get_voxel_f(0, 0, 0, ch);
        assert!(corner > 0.0, "corner should remain air, got {corner}");
    }

    #[test]
    fn do_hemisphere_is_half_of_a_sphere() {
        let mut buf = make_buffer(16);
        let ch = ChannelId::Sdf.index();
        buf.clear_channel_f(ch, 100.0);
        do_hemisphere(
            &mut buf,
            ch,
            EditMode::Add,
            1,
            Vector3f::new(8.0, 8.0, 8.0),
            4.0,
            Vector3f::new(0.0, 1.0, 0.0),
            0.0,
        );
        assert!(
            buf.get_voxel_f(8, 6, 8, ch) < 0.0,
            "below the equator should be solid"
        );
        assert!(
            buf.get_voxel_f(8, 10, 8, ch) > 0.0,
            "above the flat face should stay air"
        );
    }

    #[test]
    fn do_smooth_moves_a_sharp_sdf_boundary() {
        let mut buf = make_buffer(8);
        let ch = ChannelId::Sdf.index();
        buf.clear_channel_f(ch, 1.0);
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..4 {
                    buf.set_voxel_f(-1.0, x, y, z, ch);
                }
            }
        }
        let before = buf.get_voxel_f(3, 4, 4, ch);
        do_smooth(&mut buf, ch, Vector3f::new(4.0, 4.0, 4.0), 8.0, 1);
        let after = buf.get_voxel_f(3, 4, 4, ch);
        assert!(after > before, "blur should pull the solid side toward air");
    }

    #[test]
    fn do_sphere_remove_carves_hole() {
        let mut buf = make_buffer(16);
        let ch = ChannelId::Sdf.index();
        // Start fully inside (solid).
        buf.clear_channel_f(ch, -100.0);

        do_sphere(
            &mut buf,
            ch,
            EditMode::Remove,
            1,
            Vector3f::new(8.0, 8.0, 8.0),
            4.0,
        );

        // Center should now be outside (carved).
        let center = buf.get_voxel_f(8, 8, 8, ch);
        assert!(center > 0.0, "center should be carved to air, got {center}");
    }

    #[test]
    fn do_box_set_blocky() {
        let mut buf = make_buffer(16);
        let ch = ChannelId::Type.index();
        do_box(
            &mut buf,
            ch,
            EditMode::Set,
            42,
            Vector3i::new(2, 2, 2),
            Vector3i::new(6, 6, 6),
        );

        assert_eq!(buf.get_voxel(3, 3, 3, ch), 42);
        assert_eq!(buf.get_voxel(0, 0, 0, ch), 0);
    }

    #[test]
    fn voxel_tool_buffer_set_voxel() {
        let mut buf = make_buffer(8);
        let ch = ChannelId::Type.index();
        let mut tool = VoxelToolBuffer::new(&mut buf, ch).with_value(7);
        tool.set_voxel(Vector3i::new(1, 2, 3), 7);
        assert_eq!(buf.get_voxel(1, 2, 3, ch), 7);
    }
}
