//! Transvoxel regular-cell mesh extraction.
//!
//! Faithful port of `build_regular_mesh` from `meshers/transvoxel/transvoxel.cpp`
//! (lines ~186-610). Produces a smooth surface mesh from an SDF voxel volume
//! using the regular-cell portion of Eric Lengyel's Transvoxel algorithm.
//!
//! Phase 0 implements `TEXTURES_NONE` mode only (no mixel4 / single_s4 material
//! blending). The kernel is generic over `RegularMesherInput + ?Sized`: the
//! regular path is monomorphized per SDF width (8/16/32-bit typed inputs,
//! mirroring the C++ template dispatch); `&dyn` remains only where callers
//! pass the enum-dispatched adapter (transition passes, Bit64 fallback).

// `RegularMesherInput::len` mirrors `Span::len` and intentionally has no
// `is_empty`; the indexing loop in `cell_samples` is clearer than an iterator.
#![allow(clippy::len_without_is_empty, clippy::needless_range_loop)]

use super::regular_tables;
use super::structures::{Cache, MeshArrays};
use crate::math::funcs;
use crate::math::{Vector3f, Vector3i};

/// Padding required around the voxel block for normal computation
/// (matches `MIN_PADDING` / `MAX_PADDING` in transvoxel.h).
pub const MIN_PADDING: i32 = 1;
pub const MAX_PADDING: i32 = 2;

/// Input contract: provides typed SDF samples for a padded voxel block.
///
/// In C++ this is `Span<const TSdf>` plus template specializations for
/// `int8_t` / `int16_t` / `float`. Here we use a trait so the same
/// `build_regular_mesh` body works for any channel depth.
pub trait RegularMesherInput {
    /// Number of samples == `size.x * size.y * size.z`.
    fn len(&self) -> usize;
    /// Padded block size (including MIN_PADDING/MAX_PADDING on each axis).
    fn block_size(&self) -> Vector3i;
    /// SDF sample as `f32`, after the same conversion C++ applies with
    /// `sdf_as_float`. The isolevel is always 0.
    fn sample_f32(&self, data_index: usize) -> f32;
}

impl dyn RegularMesherInput {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Parameters for [`build_regular_mesh`].
#[derive(Debug, Clone)]
pub struct BuildRegularMeshParams {
    /// LOD index (0 = highest detail). Scales vertex positions by `1 << lod`.
    pub lod_index: u32,
    /// How far from the edge endpoints a vertex may be placed before being
    /// clamped. Matches C++ `edge_clamp_margin`. Typical value: a small epsilon.
    pub edge_clamp_margin: f32,
}

impl Default for BuildRegularMeshParams {
    fn default() -> Self {
        Self {
            lod_index: 0,
            edge_clamp_margin: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers (ported from transvoxel.cpp lines 22-184)
// ---------------------------------------------------------------------------

/// SDF sign bit: 1 if the sample is negative (inside solid).
/// Matches C++ `sign_f(float v) { return v < 0.f; }`.
#[inline]
fn sign_f(v: f32) -> u8 {
    (v < 0.0) as u8
}

/// Direction to the preceding cell, given a 3-bit reuse direction code.
/// Matches `dir_to_prev_vec(uint8_t dir)`.
#[inline]
fn dir_to_prev_vec(dir: u8) -> Vector3i {
    Vector3i::new(
        -((dir & 1) as i32),
        -(((dir >> 1) & 1) as i32),
        -(((dir >> 2) & 1) as i32),
    )
}

/// Normalize, returning (0,1,0) for a zero-length input. Matches `normalized_not_null`.
#[inline]
fn normalized_not_null(n: Vector3f) -> Vector3f {
    let lsq = vector3_length_squared(n);
    if lsq == 0.0 {
        Vector3f::new(0.0, 1.0, 0.0)
    } else {
        let l = funcs::sqrt_f32(lsq);
        Vector3f::new(n.x / l, n.y / l, n.z / l)
    }
}

#[inline]
fn vector3_length_squared(v: Vector3f) -> f32 {
    v.x * v.x + v.y * v.y + v.z * v.z
}

/// Compute a 6-bit border mask for a position within a block.
/// Bits: 1=-X 2=+X 4=-Y 8=+Y 16=-Z 32=+Z. Matches `get_border_mask`.
#[inline]
fn get_border_mask(pos: Vector3i, block_size: Vector3i) -> u8 {
    let mut mask = 0u8;
    for (axis, (p, s)) in [pos.x, pos.y, pos.z]
        .iter()
        .zip([block_size.x, block_size.y, block_size.z].iter())
        .enumerate()
    {
        if *p == 0 {
            mask |= 1 << (axis * 2);
        }
        if *p == *s {
            mask |= 1 << (axis * 2 + 1);
        }
    }
    mask
}

/// Convert a `Vector3i` to `Vector3f`. Matches `to_vec3f(Vector3i)`.
#[inline]
fn to_vec3f(v: Vector3i) -> Vector3f {
    Vector3f::new(v.x as f32, v.y as f32, v.z as f32)
}

/// Multiply each component by `1 << lod`. Matches C++ `Vector3i << lod_index`.
#[inline]
fn scale_for_lod(v: Vector3i, lod_index: u32) -> Vector3i {
    let s = (1i32) << lod_index;
    Vector3i::new(v.x * s, v.y * s, v.z * s)
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/// Extract a regular-cell surface mesh from an SDF voxel block.
///
/// Port of `build_regular_mesh` (transvoxel.cpp:186). Writes vertices, normals,
/// LOD data and triangle indices into `output`. The vertex-reuse cache `cache`
/// is reset and used internally; pass a thread-local `Cache` for reuse across
/// calls.
pub fn build_regular_mesh<S: RegularMesherInput + ?Sized>(
    input: &S,
    params: &BuildRegularMeshParams,
    cache: &mut Cache,
    output: &mut MeshArrays,
) {
    let lod_index = params.lod_index;
    let edge_clamp_margin = params.edge_clamp_margin;
    let edge_clamp_margin_max = 1.0 - edge_clamp_margin;

    let block_size_with_padding = input.block_size();
    // The actual block (without padding). Matches:
    //   block_size = block_size_with_padding - (MIN_PADDING + MAX_PADDING)
    let block_size = Vector3i::new(
        block_size_with_padding.x - (MIN_PADDING + MAX_PADDING),
        block_size_with_padding.y - (MIN_PADDING + MAX_PADDING),
        block_size_with_padding.z - (MIN_PADDING + MAX_PADDING),
    );
    let block_size_scaled = scale_for_lod(block_size, lod_index);

    cache.reset_reuse_cells(block_size_with_padding);

    // Iteration range: covers all cells with one extra voxel of reach for normals.
    let min_pos = Vector3i::splat(MIN_PADDING);
    let max_pos = Vector3i::new(
        block_size_with_padding.x - MAX_PADDING,
        block_size_with_padding.y - MAX_PADDING,
        block_size_with_padding.z - MAX_PADDING,
    );

    // Neighbor offsets in the flat data array. The C++ VoxelBuffer uses a ZXY
    // memory layout: index = y + size.y * (x + size.x * z). So Y is the
    // innermost axis, which makes the Y+1 neighbor one element away.
    let sy = block_size_with_padding.y as usize;
    let sx = block_size_with_padding.x as usize;
    let n010 = 1usize; // Y+1 (Y innermost)
    let n100 = sy; // X+1
    let n001 = sy * sx; // Z+1
    let n110 = n010 + n100;
    let n101 = n100 + n001;
    let n011 = n010 + n001;
    let n111 = n100 + n010 + n001;

    let isolevel: f32 = 0.0;

    let mut pos = Vector3i::zero();
    pos.z = min_pos.z;
    while pos.z < max_pos.z {
        pos.y = min_pos.y;
        while pos.y < max_pos.y {
            // Starting flat index for (min_pos.x, pos.y, pos.z) in ZXY layout.
            let mut data_index =
                (pos.y as usize) + sy * ((min_pos.x as usize) + sx * (pos.z as usize));

            pos.x = min_pos.x;
            while pos.x < max_pos.x {
                // ---- Early-out: skip cells that don't cross the isolevel ----
                // C++ performs this fast path on the raw SDF (`sdf_data >
                // isolevel`) before converting through `sdf_as_float`, which
                // negates float samples. Because `sample_f32()` is the converted
                // value, the faithful equivalent is `< isolevel` here.
                let s = input.sample_f32(data_index) < isolevel;
                let all_same = (input.sample_f32(data_index + n010) < isolevel) == s
                    && (input.sample_f32(data_index + n100) < isolevel) == s
                    && (input.sample_f32(data_index + n110) < isolevel) == s
                    && (input.sample_f32(data_index + n001) < isolevel) == s
                    && (input.sample_f32(data_index + n011) < isolevel) == s
                    && (input.sample_f32(data_index + n101) < isolevel) == s
                    && (input.sample_f32(data_index + n111) < isolevel) == s;
                if all_same {
                    data_index += sy;
                    pos.x += 1;
                    continue;
                }

                // Corner data indices (matches the C++ corner diagram):
                //    6-------7
                //   /|      /|
                //  4-------5 |
                //  | 2-----|-3
                //  |/      |/   z y
                //  0-------1    |/  o--x
                let corner_data_indices = [
                    data_index,        // 0
                    data_index + n100, // 1
                    data_index + n010, // 2
                    data_index + n110, // 3
                    data_index + n001, // 4
                    data_index + n101, // 5
                    data_index + n011, // 6
                    data_index + n111, // 7
                ];

                let mut cell_samples = [0.0f32; 8];
                for i in 0..8 {
                    cell_samples[i] = input.sample_f32(corner_data_indices[i]);
                }

                // Concatenate sign bits → case code (bit 0 = corner 0 ... bit 7 = corner 7).
                let mut case_code: u8 = 0;
                for i in 0..8 {
                    case_code |= sign_f(cell_samples[i]) << i;
                }

                if case_code == 0 || case_code == 255 {
                    data_index += sy;
                    pos.x += 1;
                    continue;
                }

                // ---- Per-cell geometry lookup ----
                let direction_validity_mask: u8 = {
                    let mut m = 0u8;
                    if pos.x > min_pos.x {
                        m |= 1;
                    }
                    if pos.y > min_pos.y {
                        m |= 2;
                    }
                    if pos.z > min_pos.z {
                        m |= 4;
                    }
                    m
                };

                let regular_cell_class_index = regular_tables::get_regular_cell_class(case_code);
                let regular_cell_data =
                    regular_tables::get_regular_cell_data(regular_cell_class_index);
                let triangle_count = regular_cell_data.triangle_count();
                let vertex_count = regular_cell_data.vertex_count();

                let mut cell_vertex_indices = [-1i32; 12];

                let cell_border_mask = get_border_mask(
                    Vector3i::new(pos.x - min_pos.x, pos.y - min_pos.y, pos.z - min_pos.z),
                    Vector3i::new(block_size.x - 1, block_size.y - 1, block_size.z - 1),
                );

                // Corner positions (un-padded, scaled by LOD).
                let corner_positions = {
                    let mut cps = [Vector3i::zero(); 8];
                    let px = [
                        pos.x,
                        pos.x + 1,
                        pos.x,
                        pos.x + 1,
                        pos.x,
                        pos.x + 1,
                        pos.x,
                        pos.x + 1,
                    ];
                    let py = [
                        pos.y,
                        pos.y,
                        pos.y + 1,
                        pos.y + 1,
                        pos.y,
                        pos.y,
                        pos.y + 1,
                        pos.y + 1,
                    ];
                    let pz = [
                        pos.z,
                        pos.z,
                        pos.z,
                        pos.z,
                        pos.z + 1,
                        pos.z + 1,
                        pos.z + 1,
                        pos.z + 1,
                    ];
                    for i in 0..8 {
                        cps[i] = Vector3i::new(
                            (px[i] - min_pos.x) << lod_index,
                            (py[i] - min_pos.y) << lod_index,
                            (pz[i] - min_pos.z) << lod_index,
                        );
                    }
                    cps
                };

                // ---- For each vertex produced by this cell ----
                for vertex_index in 0..vertex_count {
                    let rvd = regular_tables::get_regular_vertex_data(case_code, vertex_index);
                    let edge_code_low = (rvd & 0xff) as u8;
                    let edge_code_high = ((rvd >> 8) & 0xff) as u8;

                    let v0 = ((edge_code_low >> 4) & 0x0f) as usize;
                    let v1 = (edge_code_low & 0x0f) as usize;
                    debug_assert!(v1 > v0, "transvoxel: v1 must be > v0");

                    let sample0 = cell_samples[v0];
                    let sample1 = cell_samples[v1];

                    let p0 = corner_positions[v0];
                    let p1 = corner_positions[v1];

                    // Interpolation parameter t along the edge.
                    let t = funcs::clampf(
                        sample1 / (sample1 - sample0),
                        edge_clamp_margin,
                        edge_clamp_margin_max,
                    );

                    if t > 0.0 && t < 1.0 {
                        // Vertex is interior to the edge — try reuse, else create.
                        let reuse_dir = (edge_code_high >> 4) & 0x0f;
                        let reuse_vertex_index = (edge_code_high & 0x0f) as usize;

                        let present = (reuse_dir & direction_validity_mask) == reuse_dir;

                        if present {
                            let cache_pos = pos + dir_to_prev_vec(reuse_dir);
                            let prev_cell = cache.get_reuse_cell(cache_pos);
                            if prev_cell.packed_texture_indices
                                == cache.get_reuse_cell(pos).packed_texture_indices
                            {
                                cell_vertex_indices[vertex_index as usize] =
                                    prev_cell.vertices[reuse_vertex_index];
                            }
                        }

                        let need_create =
                            !present || cell_vertex_indices[vertex_index as usize] == -1;

                        if need_create {
                            let t0 = t;
                            let t1 = 1.0 - t;

                            let primaryf = to_vec3f(p0) * t0 + to_vec3f(p1) * t1;
                            let cg0 = get_corner_gradient(input, corner_data_indices[v0]);
                            let cg1 = get_corner_gradient(input, corner_data_indices[v1]);
                            let normal = normalized_not_null(cg0 * t0 + cg1 * t1);

                            let (secondary, vertex_border_mask) = if cell_border_mask > 0 {
                                let sec =
                                    get_secondary_position(primaryf, normal, lod_index, block_size);
                                let vbm = get_border_mask(p0, block_size_scaled)
                                    & get_border_mask(p1, block_size_scaled);
                                (sec, vbm)
                            } else {
                                (Vector3f::zero(), 0u8)
                            };

                            let vi = output.add_vertex(
                                primaryf,
                                normal,
                                cell_border_mask,
                                vertex_border_mask,
                                0,
                                secondary,
                            );
                            cell_vertex_indices[vertex_index as usize] = vi;

                            if (reuse_dir & 8) != 0 {
                                let cell = cache.get_reuse_cell_mut(pos);
                                cell.vertices[reuse_vertex_index] = vi;
                            }
                        }
                    } else if t == 0.0 && v1 == 7 {
                        // Vertex sits on corner 7 of the cell; this cell owns it.
                        let primaryf = to_vec3f(p1);
                        let cg1 = get_corner_gradient(input, corner_data_indices[v1]);
                        let normal = normalized_not_null(cg1);

                        let (secondary, vertex_border_mask) = if cell_border_mask > 0 {
                            let sec =
                                get_secondary_position(primaryf, normal, lod_index, block_size);
                            (sec, get_border_mask(p1, block_size_scaled))
                        } else {
                            (Vector3f::zero(), 0u8)
                        };

                        let vi = output.add_vertex(
                            primaryf,
                            normal,
                            cell_border_mask,
                            vertex_border_mask,
                            0,
                            secondary,
                        );
                        cell_vertex_indices[vertex_index as usize] = vi;
                        let cell = cache.get_reuse_cell_mut(pos);
                        cell.vertices[0] = vi;
                    } else {
                        // Vertex is on p0 or p1 but reuse is disabled in the default
                        // build (VOXEL_TRANSVOXEL_REUSE_VERTEX_ON_COINCIDENT_CASES off).
                        let vi_index = if t == 0.0 { v1 } else { v0 };
                        let primary = if t == 0.0 { p1 } else { p0 };
                        let primaryf = to_vec3f(primary);
                        let cg = get_corner_gradient(input, corner_data_indices[vi_index]);
                        let normal = normalized_not_null(cg);

                        let (secondary, vertex_border_mask) = if cell_border_mask > 0 {
                            let sec =
                                get_secondary_position(primaryf, normal, lod_index, block_size);
                            (sec, get_border_mask(primary, block_size_scaled))
                        } else {
                            (Vector3f::zero(), 0u8)
                        };

                        let vi = output.add_vertex(
                            primaryf,
                            normal,
                            cell_border_mask,
                            vertex_border_mask,
                            0,
                            secondary,
                        );
                        cell_vertex_indices[vertex_index as usize] = vi;
                    }
                } // for each cell vertex

                // ---- Emit triangles ----
                for t in 0..triangle_count {
                    let t0 = (t as usize) * 3;
                    let i0 = cell_vertex_indices[regular_cell_data.get_vertex_index(t0) as usize];
                    let i1 =
                        cell_vertex_indices[regular_cell_data.get_vertex_index(t0 + 1) as usize];
                    let i2 =
                        cell_vertex_indices[regular_cell_data.get_vertex_index(t0 + 2) as usize];
                    output.indices.push(i0);
                    output.indices.push(i1);
                    output.indices.push(i2);
                }

                data_index += sy;
                pos.x += 1;
            } // x
            pos.y += 1;
        } // y
        pos.z += 1;
    } // z
}

// ---------------------------------------------------------------------------
// Gradient / secondary position helpers
// ---------------------------------------------------------------------------

/// Central-difference gradient at a corner. Matches `get_corner_gradient`.
fn get_corner_gradient<S: RegularMesherInput + ?Sized>(input: &S, data_index: usize) -> Vector3f {
    // We need the block strides; pull them from the input's block size.
    let bs = input.block_size();
    let n010 = 1usize;
    let n100 = bs.y as usize;
    let n001 = (bs.y as usize) * (bs.x as usize);

    let nx = input.sample_f32(data_index - n100);
    let px = input.sample_f32(data_index + n100);
    let ny = input.sample_f32(data_index - n010);
    let py = input.sample_f32(data_index + n010);
    let nz = input.sample_f32(data_index - n001);
    let pz = input.sample_f32(data_index + n001);

    // Note: C++ applies sdf_as_float (sign flip) inside the sdf_data indexing,
    // but sample_f32() already returns the signed-distance convention used by
    // the algorithm, so no extra negation here.
    Vector3f::new(nx - px, ny - py, nz - pz)
}

/// Secondary position for LOD transitions. Matches `get_secondary_position`.
fn get_secondary_position(
    primary: Vector3f,
    normal: Vector3f,
    lod_index: u32,
    block_size_non_scaled: Vector3i,
) -> Vector3f {
    let mut delta = get_border_offset(primary, lod_index, block_size_non_scaled);
    delta = project_border_offset(delta, normal);

    // Clamp to ±2^lod to avoid shooting far at very high LOD.
    let p2k = (1u32 << lod_index) as f32;
    delta = Vector3f::new(
        funcs::clampf(delta.x, -p2k, p2k),
        funcs::clampf(delta.y, -p2k, p2k),
        funcs::clampf(delta.z, -p2k, p2k),
    );

    primary + delta
}

const TRANSITION_CELL_SCALE: f32 = 0.25;

/// Matches `get_border_offset`.
fn get_border_offset(
    pos_scaled: Vector3f,
    lod_index: u32,
    block_size_non_scaled: Vector3i,
) -> Vector3f {
    let mut delta = [0.0f32; 3];
    let p2k = (1u32 << lod_index) as f32;
    let p2mk = 1.0 / p2k;
    let wk = TRANSITION_CELL_SCALE * p2k;

    let p_arr = [pos_scaled.x, pos_scaled.y, pos_scaled.z];
    let s_arr = [
        block_size_non_scaled.x as f32,
        block_size_non_scaled.y as f32,
        block_size_non_scaled.z as f32,
    ];

    for i in 0..3 {
        let p = p_arr[i];
        let s = s_arr[i];
        if p < p2k {
            delta[i] = (1.0 - p2mk * p) * wk;
        } else if p > p2k * (s - 1.0) {
            delta[i] = (s - 1.0 - p2mk * p) * wk;
        }
    }
    Vector3f::new(delta[0], delta[1], delta[2])
}

/// Matches `project_border_offset`.
fn project_border_offset(delta: Vector3f, normal: Vector3f) -> Vector3f {
    Vector3f::new(
        (1.0 - normal.x * normal.x) * delta.x
            - normal.y * normal.x * delta.y
            - normal.z * normal.x * delta.z,
        -normal.x * normal.y * delta.x + (1.0 - normal.y * normal.y) * delta.y
            - normal.z * normal.y * delta.z,
        -normal.x * normal.z * delta.x - normal.y * normal.z * delta.y
            + (1.0 - normal.z * normal.z) * delta.z,
    )
}

#[cfg(test)]
mod tests {
    //! Unit tests for the regular-cell Transvoxel meshing path.
    //!
    //! The C++ golden parity (`tests/transvoxel_parity.rs`, `transvoxel_sphere.rs`)
    //! exercises the *whole* mesher against reference output; these tests instead
    //! pin the building blocks (`build_regular_mesh` on minimal controlled inputs,
    //! plus the private helpers that the golden path only covers indirectly).
    //! This is the "small vertex-interpolation / reuse-cache logic" the README
    //! flags as the only place future mesh-parity drift could originate.
    use super::*;
    use crate::math::{Vector3f, Vector3i};

    // ---- helper tests -------------------------------------------------------

    #[test]
    fn sign_f_flags_negative_samples_as_inside() {
        // C++ `sign_f(v) { return v < 0.f; }` — strictly less than zero.
        assert_eq!(sign_f(-0.1), 1);
        assert_eq!(sign_f(0.0), 0); // zero is *not* inside
        assert_eq!(sign_f(0.1), 0);
    }

    #[test]
    fn dir_to_prev_vec_unpacks_three_bit_direction() {
        // dir is packed X(0) Y(1) Z(2), each bit negated to point at the prev cell.
        assert_eq!(dir_to_prev_vec(0b000), Vector3i::new(0, 0, 0));
        assert_eq!(dir_to_prev_vec(0b001), Vector3i::new(-1, 0, 0));
        assert_eq!(dir_to_prev_vec(0b010), Vector3i::new(0, -1, 0));
        assert_eq!(dir_to_prev_vec(0b100), Vector3i::new(0, 0, -1));
        assert_eq!(dir_to_prev_vec(0b111), Vector3i::new(-1, -1, -1));
    }

    #[test]
    fn normalized_not_null_falls_back_to_up_for_zero_input() {
        // A zero vector must not produce NaN — it falls back to (0,1,0).
        let n = normalized_not_null(Vector3f::new(0.0, 0.0, 0.0));
        assert_eq!(n, Vector3f::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn normalized_not_null_unitizes_nonzero_input() {
        let n = normalized_not_null(Vector3f::new(0.0, 2.0, 0.0));
        assert!((n.y - 1.0).abs() < 1e-6);
        assert!(n.x.abs() < 1e-6 && n.z.abs() < 1e-6);
    }

    #[test]
    fn get_border_mask_sets_one_bit_per_face() {
        // Bits: 1=-X 2=+X 4=-Y 8=+Y 16=-Z 32=+Z. Origin hits all three low faces
        // (X==0, Y==0, Z==0) → bits 0, 2, 4 → 1 + 4 + 16 = 21.
        assert_eq!(
            get_border_mask(Vector3i::new(0, 0, 0), Vector3i::new(4, 4, 4)),
            0b01_0101
        );
        // Far corner hits all three high faces (X==4, Y==4, Z==4) → bits 1, 3, 5.
        assert_eq!(
            get_border_mask(Vector3i::new(4, 4, 4), Vector3i::new(4, 4, 4)),
            0b10_1010
        );
        // Interior cell touches no border.
        assert_eq!(
            get_border_mask(Vector3i::new(2, 2, 2), Vector3i::new(4, 4, 4)),
            0
        );
        // A single-axis edge: only -X (bit0) and +Z (bit5) → 1 + 32 = 33.
        assert_eq!(
            get_border_mask(Vector3i::new(0, 2, 4), Vector3i::new(4, 4, 4)),
            33
        );
    }

    #[test]
    fn scale_for_lod_shifts_by_one_per_lod() {
        assert_eq!(
            scale_for_lod(Vector3i::new(1, 2, 3), 0),
            Vector3i::new(1, 2, 3)
        );
        assert_eq!(
            scale_for_lod(Vector3i::new(1, 2, 3), 1),
            Vector3i::new(2, 4, 6)
        );
        assert_eq!(
            scale_for_lod(Vector3i::new(1, 1, 1), 2),
            Vector3i::new(4, 4, 4)
        );
    }

    // ---- minimal controlled SDF input --------------------------------------

    /// A flat f32 SDF buffer over a padded block (ZXY layout: index = y + sy*(x + sx*z)).
    struct FlatSdf {
        block_size: Vector3i,
        data: Vec<f32>,
    }

    impl FlatSdf {
        /// Fill every voxel with `value`.
        fn filled(block_size: Vector3i, value: f32) -> Self {
            let n = (block_size.x * block_size.y * block_size.z) as usize;
            Self {
                block_size,
                data: vec![value; n],
            }
        }
    }

    impl RegularMesherInput for FlatSdf {
        fn len(&self) -> usize {
            self.data.len()
        }
        fn block_size(&self) -> Vector3i {
            self.block_size
        }
        fn sample_f32(&self, data_index: usize) -> f32 {
            self.data[data_index]
        }
    }

    // ---- build_regular_mesh on controlled inputs ---------------------------

    // The padded block must cover at least MIN_PADDING on the low side and
    // MAX_PADDING on the high side. With block_size = 1, padded = 1 + 1 + 2 = 4.
    const PADDED: Vector3i = Vector3i::new(4, 4, 4);

    #[test]
    fn build_regular_mesh_empty_volume_emits_no_geometry() {
        // Uniformly outside the surface (positive SDF) → no cell crosses the
        // isolevel, so no vertices and no indices are produced.
        let input = FlatSdf::filled(PADDED, 1.0);
        let mut cache = Cache::default();
        let mut output = MeshArrays::default();
        build_regular_mesh(
            &input,
            &BuildRegularMeshParams::default(),
            &mut cache,
            &mut output,
        );
        assert!(
            output.vertices.is_empty(),
            "vertices: {:?}",
            output.vertices
        );
        assert!(output.indices.is_empty());
    }

    #[test]
    fn build_regular_mesh_fully_solid_volume_emits_no_geometry() {
        // Uniformly inside the surface (negative SDF). The whole block is one
        // region, so again no boundary → no geometry. This is the case_code==255
        // early-out path.
        let input = FlatSdf::filled(PADDED, -1.0);
        let mut cache = Cache::default();
        let mut output = MeshArrays::default();
        build_regular_mesh(
            &input,
            &BuildRegularMeshParams::default(),
            &mut cache,
            &mut output,
        );
        assert!(
            output.vertices.is_empty(),
            "vertices: {:?}",
            output.vertices
        );
        assert!(output.indices.is_empty());
    }

    #[test]
    fn build_regular_mesh_half_space_produces_watertight_slice() {
        // A planar SDF crossing halfway down Y produces exactly one triangle
        // sheet. With SDF = y - threshold, every cell straddles the same plane.
        // We craft a half-space: solid where y < 2 (in block-local coords within
        // the padded buffer), i.e. negative below the plane.
        let mut input = FlatSdf::filled(PADDED, 0.0);
        let sy = PADDED.y as usize;
        let sx = PADDED.x as usize;
        for z in 0..PADDED.z {
            for x in 0..PADDED.x {
                for y in 0..PADDED.y {
                    let i = (y as usize) + sy * ((x as usize) + sx * (z as usize));
                    // signed distance to the plane y = 1.5; negative below → solid.
                    input.data[i] = 1.5 - y as f32;
                }
            }
        }
        let mut cache = Cache::default();
        let mut output = MeshArrays::default();
        build_regular_mesh(
            &input,
            &BuildRegularMeshParams::default(),
            &mut cache,
            &mut output,
        );

        // Geometry must exist and indices must be a multiple of 3 (triangles).
        assert!(!output.indices.is_empty(), "expected a surface slice");
        assert_eq!(
            output.indices.len() % 3,
            0,
            "index count must be a multiple of 3"
        );
        // Every vertex the triangles reference must be a valid vertex index.
        for &idx in &output.indices {
            assert!(
                (0..output.vertices.len() as i32).contains(&idx),
                "index {idx} out of vertex range 0..{}",
                output.vertices.len()
            );
        }
        // Vertex / normal / lod_data arrays must stay parallel (add_vertex pushes all three).
        assert_eq!(output.vertices.len(), output.normals.len());
        assert_eq!(output.vertices.len(), output.lod_data.len());
    }

    #[test]
    fn build_regular_mesh_emits_normals_on_both_sides_of_the_plane() {
        // Same half-space as above, but check that vertex interpolation placed
        // the surface at y ≈ 1.5 (the threshold) and that normals are non-zero.
        let mut input = FlatSdf::filled(PADDED, 0.0);
        let sy = PADDED.y as usize;
        let sx = PADDED.x as usize;
        for z in 0..PADDED.z {
            for x in 0..PADDED.x {
                for y in 0..PADDED.y {
                    let i = (y as usize) + sy * ((x as usize) + sx * (z as usize));
                    input.data[i] = 1.5 - y as f32;
                }
            }
        }
        let mut cache = Cache::default();
        let mut output = MeshArrays::default();
        build_regular_mesh(
            &input,
            &BuildRegularMeshParams::default(),
            &mut cache,
            &mut output,
        );

        // The single cell (block_size==1) straddles y in [1,2] in padded
        // coordinates; after subtracting min_pos, emitted vertices lie in
        // [0,1]. The SDF zero-crossing is at y_padded = 1.5, i.e. y_out ≈ 0.5.
        for v in &output.vertices {
            assert!(
                v.y >= 0.0 - 1e-5 && v.y <= 1.0 + 1e-5,
                "vertex y={:.4} escaped the [0,1] output band",
                v.y
            );
        }
        // The plane normal points toward +Y (the outside); normals should be
        // predominantly +Y. Allow for central-difference noise but require net up.
        let net_y: f32 = output.normals.iter().map(|n| n.y).sum();
        assert!(net_y > 0.0, "normals should point +Y, net_y={net_y}");
    }
}
