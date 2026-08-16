//! Bake a triangle mesh into a [`BakedModel`]: ortho-rotate, then split
//! triangles onto cube sides vs the interior.
//!
//! Ports `blocky::bake_mesh_geometry` from
//! `meshers/blocky/voxel_blocky_model_mesh.cpp`. Side classification uses the
//! same `side_vertex_tolerance` rule: a triangle belongs to a face when all
//! three vertices lie on that face of the unit cube `[0,1]³`.

use crate::constants::cube_tables::{Side, SIDE_NORMALS};
use crate::math::ortho_basis::{ORTHOGONAL_BASIS_IDENTITY_INDEX, ORTHO_BASES};
use crate::math::{Color, Vector2f, Vector3f};
use crate::meshers::blocky::baked_library::{BakedModel, ModelSurface, SideSurface, MAX_SURFACES};
use std::collections::HashMap;

/// Indexed triangle mesh used as input to [`bake_mesh_model`].
#[derive(Debug, Clone, Default)]
pub struct MeshGeometry {
    pub positions: Vec<Vector3f>,
    pub normals: Vec<Vector3f>,
    pub uvs: Vec<Vector2f>,
    pub indices: Vec<i32>,
    /// Optional 4-float tangents per vertex. Empty means "no tangents".
    pub tangents: Vec<f32>,
}

impl MeshGeometry {
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty() || self.indices.is_empty()
    }
}

/// Default `side_vertex_tolerance` (upstream `VoxelBlockyModelMesh`).
pub const DEFAULT_SIDE_VERTEX_TOLERANCE: f32 = 0.001;

/// Bake `geometry` into a model: apply `ortho_rotation` (0..=23) about the
/// cube center, then send each triangle to a side surface or the interior.
pub fn bake_mesh_model(
    geometry: &MeshGeometry,
    ortho_rotation: usize,
    side_vertex_tolerance: f32,
    cutout_sides_enabled: bool,
) -> BakedModel {
    let mut model = BakedModel {
        empty: geometry.positions.is_empty(),
        cutout_sides_enabled,
        culls_neighbors: true,
        contributes_to_ao: true,
        color: Color::new(1.0, 1.0, 1.0, 1.0),
        ..BakedModel::default()
    };
    if geometry.is_empty() {
        return model;
    }
    model.model.surface_count = 1;
    model.model.surfaces[0].collision_enabled = true;

    let mut positions = geometry.positions.clone();
    let mut normals = if geometry.normals.len() == positions.len() {
        geometry.normals.clone()
    } else {
        vec![Vector3f::new(0.0, 1.0, 0.0); positions.len()]
    };
    let uvs = if geometry.uvs.len() == positions.len() {
        geometry.uvs.clone()
    } else {
        vec![Vector2f::new(0.0, 0.0); positions.len()]
    };
    let mut tangents = if geometry.tangents.len() == positions.len() * 4 {
        geometry.tangents.clone()
    } else {
        Vec::new()
    };

    apply_ortho_rotation_to_arrays(&mut positions, &mut normals, &mut tangents, ortho_rotation);

    let tolerance = side_vertex_tolerance.max(0.0);
    let mut added_side: [HashMap<i32, i32>; Side::COUNT] = std::array::from_fn(|_| HashMap::new());
    let mut added_interior: HashMap<i32, i32> = HashMap::new();

    let index_chunks = geometry.indices.len() / 3;
    for tri in 0..index_chunks {
        let src = [
            geometry.indices[tri * 3],
            geometry.indices[tri * 3 + 1],
            geometry.indices[tri * 3 + 2],
        ];
        if src.iter().any(|&i| i < 0 || i as usize >= positions.len()) {
            continue;
        }
        let tri_pos = [
            positions[src[0] as usize],
            positions[src[1] as usize],
            positions[src[2] as usize],
        ];
        if let Some(side) = triangle_side(tri_pos[0], tri_pos[1], tri_pos[2], tolerance) {
            append_triangle_to_side(
                &mut model.model.sides_surfaces[side as usize][0],
                &mut added_side[side as usize],
                src,
                &tri_pos,
                &uvs,
                &tangents,
            );
        } else {
            append_triangle_to_interior(
                &mut model.model.surfaces[0],
                &mut added_interior,
                src,
                &tri_pos,
                &normals,
                &uvs,
                &tangents,
            );
        }
    }

    model.empty = model.model.surfaces[0].indices.is_empty()
        && (0..Side::COUNT).all(|side| model.model.sides_surfaces[side][0].indices.is_empty());
    model
}

/// Rotate an already-baked model's geometry by an ortho index. Identity is a
/// no-op. Used for cube models that already have side quads.
pub fn apply_ortho_rotation(model: BakedModel, ortho_rotation: usize) -> BakedModel {
    if ortho_rotation == ORTHOGONAL_BASIS_IDENTITY_INDEX || ortho_rotation >= ORTHO_BASES.len() {
        return model;
    }
    let geometry = mesh_geometry_from_model(&model);
    if geometry.is_empty() {
        return model;
    }
    let mut baked = bake_mesh_model(
        &geometry,
        ortho_rotation,
        DEFAULT_SIDE_VERTEX_TOLERANCE,
        model.cutout_sides_enabled,
    );
    baked.color = model.color;
    baked.culls_neighbors = model.culls_neighbors;
    baked.contributes_to_ao = model.contributes_to_ao;
    baked.is_transparent = model.is_transparent;
    baked.transparency_index = model.transparency_index;
    baked.is_random_tickable = model.is_random_tickable;
    baked.tags_mask = model.tags_mask;
    baked.empty = model.empty;
    baked.lod_skirts = model.lod_skirts;
    baked.box_collision_mask = model.box_collision_mask;
    baked.box_collision_aabbs = model.box_collision_aabbs;
    baked.fluid_index = model.fluid_index;
    baked.fluid_level = model.fluid_level;
    baked
}

/// Which cube face a vertex sits on (bit `side`). A vertex can lie on an edge
/// or corner and therefore set multiple bits.
pub fn vertex_sides_mask(pos: Vector3f, tolerance: f32) -> u8 {
    let mut mask = 0u8;
    if (pos.x - 0.0).abs() <= tolerance {
        mask |= 1 << Side::Right as u8;
    }
    if (pos.x - 1.0).abs() <= tolerance {
        mask |= 1 << Side::Left as u8;
    }
    if (pos.y - 0.0).abs() <= tolerance {
        mask |= 1 << Side::Bottom as u8;
    }
    if (pos.y - 1.0).abs() <= tolerance {
        mask |= 1 << Side::Top as u8;
    }
    if (pos.z - 0.0).abs() <= tolerance {
        mask |= 1 << Side::Back as u8;
    }
    if (pos.z - 1.0).abs() <= tolerance {
        mask |= 1 << Side::Front as u8;
    }
    mask
}

/// Side of the unit cube that contains all three vertices, if any.
pub fn triangle_side(a: Vector3f, b: Vector3f, c: Vector3f, tolerance: f32) -> Option<u8> {
    let mask = vertex_sides_mask(a, tolerance)
        & vertex_sides_mask(b, tolerance)
        & vertex_sides_mask(c, tolerance);
    if mask == 0 {
        return None;
    }
    (0..Side::COUNT as u8).find(|&side| mask == (1 << side))
}

fn apply_ortho_rotation_to_arrays(
    positions: &mut [Vector3f],
    normals: &mut [Vector3f],
    tangents: &mut [f32],
    ortho_rotation: usize,
) {
    if ortho_rotation == ORTHOGONAL_BASIS_IDENTITY_INDEX || ortho_rotation >= ORTHO_BASES.len() {
        return;
    }
    let basis = ORTHO_BASES[ortho_rotation];
    let half = Vector3f::splat(0.5);
    for p in positions.iter_mut() {
        *p = basis.xform_f(*p - half) + half;
    }
    for n in normals.iter_mut() {
        *n = basis.xform_f(*n);
    }
    if tangents.len() == normals.len() * 4 {
        for ti in 0..normals.len() {
            let i0 = ti * 4;
            let t = Vector3f::new(tangents[i0], tangents[i0 + 1], tangents[i0 + 2]);
            let t = basis.xform_f(t);
            tangents[i0] = t.x;
            tangents[i0 + 1] = t.y;
            tangents[i0 + 2] = t.z;
        }
    }
}

fn append_triangle_to_side(
    side: &mut SideSurface,
    added: &mut HashMap<i32, i32>,
    src: [i32; 3],
    tri_pos: &[Vector3f; 3],
    uvs: &[Vector2f],
    tangents: &[f32],
) {
    for (j, src_index) in src.iter().copied().enumerate() {
        if let Some(&dst) = added.get(&src_index) {
            side.indices.push(dst);
            continue;
        }
        let dst = side.positions.len() as i32;
        side.indices.push(dst);
        side.positions.push(tri_pos[j]);
        side.uvs
            .push(uvs.get(src_index as usize).copied().unwrap_or_default());
        if tangents.len() >= (src_index as usize + 1) * 4 {
            let ti = src_index as usize * 4;
            side.tangents.extend_from_slice(&tangents[ti..ti + 4]);
        }
        added.insert(src_index, dst);
    }
}

fn append_triangle_to_interior(
    surface: &mut ModelSurface,
    added: &mut HashMap<i32, i32>,
    src: [i32; 3],
    tri_pos: &[Vector3f; 3],
    normals: &[Vector3f],
    uvs: &[Vector2f],
    tangents: &[f32],
) {
    for (j, src_index) in src.iter().copied().enumerate() {
        if let Some(&dst) = added.get(&src_index) {
            surface.indices.push(dst);
            continue;
        }
        let dst = surface.positions.len() as i32;
        surface.indices.push(dst);
        surface.positions.push(tri_pos[j]);
        surface.normals.push(
            normals
                .get(src_index as usize)
                .copied()
                .unwrap_or(Vector3f::new(0.0, 1.0, 0.0)),
        );
        surface
            .uvs
            .push(uvs.get(src_index as usize).copied().unwrap_or_default());
        if tangents.len() >= (src_index as usize + 1) * 4 {
            let ti = src_index as usize * 4;
            surface.tangents.extend_from_slice(&tangents[ti..ti + 4]);
        }
        added.insert(src_index, dst);
    }
}

fn mesh_geometry_from_model(model: &crate::meshers::blocky::BakedModel) -> MeshGeometry {
    let mut geometry = MeshGeometry::default();
    let surface_count = (model.model.surface_count as usize).min(MAX_SURFACES);
    for surface_index in 0..surface_count {
        let surface = &model.model.surfaces[surface_index];
        let base = geometry.positions.len() as i32;
        for i in 0..surface.positions.len() {
            let normal = surface
                .normals
                .get(i)
                .copied()
                .unwrap_or(Vector3f::new(0.0, 1.0, 0.0));
            let uv = surface.uvs.get(i).copied().unwrap_or_default();
            geometry.positions.push(surface.positions[i]);
            geometry.normals.push(normal);
            geometry.uvs.push(uv);
        }
        for &index in &surface.indices {
            geometry.indices.push(base + index);
        }
        for (side, side_normal) in SIDE_NORMALS.iter().enumerate() {
            let side_surface = &model.model.sides_surfaces[side][surface_index];
            if side_surface.indices.is_empty() {
                continue;
            }
            let normal = Vector3f::new(
                side_normal.x as f32,
                side_normal.y as f32,
                side_normal.z as f32,
            );
            let base = geometry.positions.len() as i32;
            for i in 0..side_surface.positions.len() {
                let uv = side_surface.uvs.get(i).copied().unwrap_or_default();
                geometry.positions.push(side_surface.positions[i]);
                geometry.normals.push(normal);
                geometry.uvs.push(uv);
            }
            for &index in &side_surface.indices {
                geometry.indices.push(base + index);
            }
        }
    }
    geometry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::cube_tables::Side;
    use crate::math::ortho_basis::OrthoRotationId;
    use crate::meshers::blocky::{bake_library, solid_cube_model, BakedLibrary};

    fn quad_on_positive_x() -> MeshGeometry {
        // Two triangles covering x=1, y/z in [0,1].
        MeshGeometry {
            positions: vec![
                Vector3f::new(1.0, 0.0, 0.0),
                Vector3f::new(1.0, 1.0, 0.0),
                Vector3f::new(1.0, 1.0, 1.0),
                Vector3f::new(1.0, 0.0, 1.0),
            ],
            normals: vec![Vector3f::new(1.0, 0.0, 0.0); 4],
            uvs: vec![
                Vector2f::new(0.0, 0.0),
                Vector2f::new(1.0, 0.0),
                Vector2f::new(1.0, 1.0),
                Vector2f::new(0.0, 1.0),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            tangents: Vec::new(),
        }
    }

    #[test]
    fn face_triangle_lands_on_the_matching_side() {
        let baked = bake_mesh_model(
            &quad_on_positive_x(),
            0,
            DEFAULT_SIDE_VERTEX_TOLERANCE,
            false,
        );
        assert!(!baked.model.sides_surfaces[Side::Left as usize][0]
            .indices
            .is_empty());
        assert!(baked.model.surfaces[0].indices.is_empty());
        for side in 0..Side::COUNT {
            if side == Side::Left as usize {
                continue;
            }
            assert!(baked.model.sides_surfaces[side][0].indices.is_empty());
        }
    }

    #[test]
    fn y180_moves_positive_x_face_to_negative_x() {
        let baked = bake_mesh_model(
            &quad_on_positive_x(),
            OrthoRotationId::Y180 as usize,
            DEFAULT_SIDE_VERTEX_TOLERANCE,
            false,
        );
        assert!(!baked.model.sides_surfaces[Side::Right as usize][0]
            .indices
            .is_empty());
        assert!(baked.model.sides_surfaces[Side::Left as usize][0]
            .indices
            .is_empty());
        for p in &baked.model.sides_surfaces[Side::Right as usize][0].positions {
            assert!((p.x - 0.0).abs() <= DEFAULT_SIDE_VERTEX_TOLERANCE);
        }
    }

    #[test]
    fn interior_triangle_stays_off_sides() {
        let geometry = MeshGeometry {
            positions: vec![
                Vector3f::new(0.2, 0.2, 0.2),
                Vector3f::new(0.8, 0.2, 0.2),
                Vector3f::new(0.5, 0.8, 0.5),
            ],
            normals: vec![Vector3f::new(0.0, 0.0, 1.0); 3],
            uvs: vec![Vector2f::new(0.0, 0.0); 3],
            indices: vec![0, 1, 2],
            tangents: Vec::new(),
        };
        let baked = bake_mesh_model(&geometry, 0, DEFAULT_SIDE_VERTEX_TOLERANCE, true);
        assert_eq!(baked.model.surfaces[0].indices.len(), 3);
        assert!(baked.cutout_sides_enabled);
        for side in 0..Side::COUNT {
            assert!(baked.model.sides_surfaces[side][0].indices.is_empty());
        }
    }

    #[test]
    fn rotated_cube_still_has_six_full_sides() {
        let cube = apply_ortho_rotation(
            solid_cube_model(Color::new(1.0, 1.0, 1.0, 1.0)),
            OrthoRotationId::Y90 as usize,
        );
        let mut lib = BakedLibrary {
            models: vec![BakedModel::default(), cube],
            ..BakedLibrary::default()
        };
        bake_library(&mut lib);
        assert_eq!(lib.models[1].model.empty_sides_mask, 0);
        assert_eq!(lib.models[1].model.full_sides_mask, 0b0011_1111);
    }
}
