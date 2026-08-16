//! Plain-data model library consumed by the blocky mesher.
//!
//! Ported from `meshers/blocky/blocky_baked_library.h`. These structs are the
//! "baked" representation of voxel models: geometry arrays, side-culling
//! hints, and collision boxes, all plain data with no Godot dependency. The
//! mesher reads them directly on the hot path; the Godot `Resource` / editor
//! layer that *produces* them is deferred to Phase 5.
//!
//! The C++ header pulls in Godot `Color` and `AABB` transitively; here `Color`
//! is the already-ported [`crate::math::Color`] and `AABB` is a local
//! [`Aabb`] struct (`{min, max}`).

use crate::constants::cube_tables::Side;
use crate::math::{Color, Vector2f, Vector3f};
use std::collections::HashMap;

/// Maximum number of models in a library. Matches `MAX_MODELS`.
pub const MAX_MODELS: usize = 65536;
/// Maximum number of baked fluids. Matches `MAX_FLUIDS`.
pub const MAX_FLUIDS: usize = 256;
/// Maximum number of materials. Matches `MAX_MATERIALS`.
pub const MAX_MATERIALS: usize = 65536;
/// Maximum surfaces per model (opaque + transparent). Matches `MAX_SURFACES`.
pub const MAX_SURFACES: usize = 2;

/// Convention: model index 0 means "air" (no model). Matches `AIR_ID`.
pub const AIR_ID: u16 = 0;
/// Sentinel for "no fluid". Matches `NULL_FLUID_INDEX`.
pub const NULL_FLUID_INDEX: u8 = 255;

/// Axis-aligned bounding box, a plain-data stand-in for Godot's `AABB`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Aabb {
    pub min: Vector3f,
    pub max: Vector3f,
}

impl Aabb {
    pub fn new(min: Vector3f, max: Vector3f) -> Self {
        Self { min, max }
    }
}

/// A growable bitset. Stand-in for `DynamicBitset`; the blocky bake pass uses
/// it to store the `pattern_a × pattern_b` occlusion matrix.
#[derive(Debug, Clone, Default)]
pub struct DynamicBitset {
    bits: Vec<u64>,
    len: usize,
}

impl DynamicBitset {
    /// Create an empty bitset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of bits stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no bits are set.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Resize to `len` bits, zeroing any new bits.
    pub fn resize(&mut self, len: usize) {
        let words = len.div_ceil(64);
        self.bits.resize(words, 0);
        self.len = len;
    }

    /// Get bit `i`. Out-of-range reads return `false`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        if i >= self.len {
            return false;
        }
        (self.bits[i / 64] >> (i % 64)) & 1 != 0
    }

    /// Set bit `i` to `v`. Grows the bitset if necessary.
    #[inline]
    pub fn set(&mut self, i: usize, v: bool) {
        if i >= self.len {
            self.resize(i + 1);
        }
        if v {
            self.bits[i / 64] |= 1 << (i % 64);
        } else {
            self.bits[i / 64] &= !(1 << (i % 64));
        }
    }

    /// Set bit `i` to `true`. Does NOT grow (caller must `resize` first).
    #[inline]
    pub fn set_unchecked(&mut self, i: usize, v: bool) {
        debug_assert!(i < self.len);
        if v {
            self.bits[i / 64] |= 1 << (i % 64);
        } else {
            self.bits[i / 64] &= !(1 << (i % 64));
        }
    }
}

/// Geometry for one side of a baked model (positions + uvs + indices +
/// tangents). Normals are implicit (the side's normal). Matches
/// `BakedModel::SideSurface`.
#[derive(Debug, Clone, Default)]
pub struct SideSurface {
    pub positions: Vec<Vector3f>,
    pub uvs: Vec<Vector2f>,
    pub indices: Vec<i32>,
    pub tangents: Vec<f32>,
}

impl SideSurface {
    pub fn clear(&mut self) {
        self.positions.clear();
        self.uvs.clear();
        self.indices.clear();
        self.tangents.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Geometry for the "inside" of a model (not on a face). Matches
/// `BakedModel::Surface`.
#[derive(Debug, Clone)]
pub struct ModelSurface {
    pub positions: Vec<Vector3f>,
    pub normals: Vec<Vector3f>,
    pub uvs: Vec<Vector2f>,
    pub indices: Vec<i32>,
    pub tangents: Vec<f32>,
    pub material_id: u32,
    pub collision_enabled: bool,
}

impl Default for ModelSurface {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            tangents: Vec::new(),
            material_id: 0,
            collision_enabled: true,
        }
    }
}

impl ModelSurface {
    pub fn clear(&mut self) {
        self.positions.clear();
        self.normals.clear();
        self.uvs.clear();
        self.indices.clear();
        self.tangents.clear();
    }
}

/// A baked cube model: up to [`MAX_SURFACES`] inside surfaces plus per-side
/// geometry. Matches `BakedModel::Model`.
#[derive(Debug, Clone)]
pub struct BakedModelMesh {
    pub surfaces: [ModelSurface; MAX_SURFACES],
    /// `[side][surface_index]` → side geometry.
    pub sides_surfaces: [[SideSurface; MAX_SURFACES]; Side::COUNT],
    pub surface_count: u32,
    /// Bitmask of sides that have no geometry. Bit `i` = `Side as u8`.
    pub empty_sides_mask: u8,
    /// Bitmask of sides that fully cover the face (a full quad).
    pub full_sides_mask: u8,
    /// Per-side pattern index used by the side-culling matrix.
    pub side_pattern_indices: [u32; Side::COUNT],
}

impl Default for BakedModelMesh {
    fn default() -> Self {
        Self {
            surfaces: Default::default(),
            sides_surfaces: std::array::from_fn(|_| Default::default()),
            surface_count: 0,
            empty_sides_mask: 0,
            full_sides_mask: 0,
            side_pattern_indices: [0; Side::COUNT],
        }
    }
}

impl BakedModelMesh {
    pub fn clear(&mut self) {
        for s in &mut self.surfaces {
            s.clear();
        }
        for side in &mut self.sides_surfaces {
            for ss in side {
                ss.clear();
            }
        }
    }
}

/// One entry in the blocky library: a mesh plus metadata. Matches
/// `BakedModel`.
#[derive(Debug, Clone)]
pub struct BakedModel {
    pub model: BakedModelMesh,
    pub color: Color,
    pub transparency_index: u8,
    pub culls_neighbors: bool,
    pub contributes_to_ao: bool,
    pub empty: bool,
    pub is_random_tickable: bool,
    pub is_transparent: bool,
    pub cutout_sides_enabled: bool,
    pub fluid_index: u8,
    pub fluid_level: u8,
    pub lod_skirts: bool,
    pub box_collision_mask: u32,
    pub tags_mask: u32,
    pub box_collision_aabbs: Vec<Aabb>,
    /// Pre-computed cutout side geometry: `(side, neighbor_shape_id)` → the
    /// side surfaces to render when a neighbor of that shape partially occludes
    /// `side`. Populated by the bake pass. Matches
    /// `BakedModel::cutout_side_surfaces`.
    pub cutout_side_surfaces: HashMap<(u8, u32), Vec<SideSurface>>,
}

impl Default for BakedModel {
    fn default() -> Self {
        Self {
            model: BakedModelMesh::default(),
            color: Color::new(1.0, 1.0, 1.0, 1.0),
            transparency_index: 0,
            culls_neighbors: true,
            contributes_to_ao: true,
            empty: true,
            is_random_tickable: false,
            is_transparent: false,
            cutout_sides_enabled: false,
            fluid_index: NULL_FLUID_INDEX,
            fluid_level: 0,
            lod_skirts: false,
            box_collision_mask: 0,
            tags_mask: 0,
            box_collision_aabbs: Vec::new(),
            cutout_side_surfaces: HashMap::new(),
        }
    }
}

impl BakedModel {
    pub fn clear(&mut self) {
        self.model.clear();
        self.empty = true;
    }
}

/// Surface data for a baked fluid (one set per side). Matches
/// `BakedFluid::Surface`.
#[derive(Debug, Clone, Default)]
pub struct FluidSurface {
    pub positions: Vec<Vector3f>,
    pub indices: Vec<i32>,
    pub tangents: Vec<f32>,
}

impl FluidSurface {
    pub fn clear(&mut self) {
        self.positions.clear();
        self.indices.clear();
        self.tangents.clear();
    }
}

/// A baked fluid model. Matches `BakedFluid`.
#[derive(Debug, Clone)]
pub struct BakedFluid {
    pub side_surfaces: [FluidSurface; Side::COUNT],
    pub material_id: u32,
    pub max_level: u8,
    pub dip_when_flowing_down: bool,
}

/// Top height of a full fluid block (matches `BakedFluid::TOP_HEIGHT`).
pub const FLUID_TOP_HEIGHT: f32 = 0.9375;
/// Bottom height of a fluid block (matches `BakedFluid::BOTTOM_HEIGHT`).
pub const FLUID_BOTTOM_HEIGHT: f32 = 0.0625;

impl Default for BakedFluid {
    fn default() -> Self {
        Self {
            side_surfaces: std::array::from_fn(|_| FluidSurface::default()),
            material_id: 0,
            max_level: 1,
            dip_when_flowing_down: false,
        }
    }
}

/// The baked library: models, fluids, and the side-pattern occlusion matrix.
/// Matches `BakedLibrary`.
#[derive(Debug, Clone, Default)]
pub struct BakedLibrary {
    /// `pattern_a + pattern_b * side_pattern_count` → does A occlude B?
    pub side_pattern_culling: DynamicBitset,
    pub side_pattern_count: u32,
    pub models: Vec<BakedModel>,
    pub fluids: Vec<BakedFluid>,
    pub indexed_materials_count: u32,
}

/// One full unit-cube face: four corners and two triangles, winding from
/// [`SIDE_QUAD_TRIANGLES`](crate::constants::cube_tables::SIDE_QUAD_TRIANGLES).
pub fn full_cube_side_surface(side: usize) -> SideSurface {
    use crate::constants::cube_tables::{CORNER_POSITION, SIDE_CORNERS, SIDE_QUAD_TRIANGLES};
    use crate::math::Vector2f;

    let corners = SIDE_CORNERS[side];
    SideSurface {
        positions: corners.iter().map(|&c| CORNER_POSITION[c]).collect(),
        uvs: vec![
            Vector2f::new(0.0, 0.0),
            Vector2f::new(1.0, 0.0),
            Vector2f::new(1.0, 1.0),
            Vector2f::new(0.0, 1.0),
        ],
        indices: SIDE_QUAD_TRIANGLES[side].to_vec(),
        tangents: Vec::new(),
    }
}

/// Opaque full cube used as a default blocky model (index ≥ 1; 0 stays air).
pub fn solid_cube_model(color: Color) -> BakedModel {
    let mut cube = BakedModel {
        color,
        empty: false,
        culls_neighbors: true,
        contributes_to_ao: true,
        ..BakedModel::default()
    };
    cube.model.surface_count = 1;
    cube.model.surfaces[0].collision_enabled = true;
    for side in 0..Side::COUNT {
        cube.model.sides_surfaces[side][0] = full_cube_side_surface(side);
    }
    cube
}

impl BakedLibrary {
    /// Air at index 0 plus one solid cube at index 1. Call
    /// [`crate::meshers::blocky::bake_library`] before meshing.
    pub fn with_air_and_solid_cube(color: Color) -> Self {
        Self {
            models: vec![BakedModel::default(), solid_cube_model(color)],
            ..Self::default()
        }
    }

    /// Whether model `i` exists.
    pub fn has_model(&self, i: u32) -> bool {
        (i as usize) < self.models.len()
    }

    /// `get_side_pattern_occlusion` — whether pattern A occludes pattern B.
    #[inline]
    pub fn get_side_pattern_occlusion(&self, pattern_a: u32, pattern_b: u32) -> bool {
        debug_assert!(pattern_a < self.side_pattern_count);
        debug_assert!(pattern_b < self.side_pattern_count);
        self.side_pattern_culling
            .get((pattern_a + pattern_b * self.side_pattern_count) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_bitset_set_get_round_trips() {
        let mut bs = DynamicBitset::new();
        bs.resize(100);
        bs.set(0, true);
        bs.set(50, true);
        bs.set(99, true);
        assert!(bs.get(0));
        assert!(bs.get(50));
        assert!(bs.get(99));
        assert!(!bs.get(1));
        assert!(!bs.get(100)); // out of range
    }

    #[test]
    fn dynamic_bitset_set_grows() {
        let mut bs = DynamicBitset::new();
        bs.set(200, true);
        assert!(bs.get(200));
        assert_eq!(bs.len(), 201);
    }

    #[test]
    fn baked_model_defaults_to_empty_white() {
        let m = BakedModel::default();
        assert!(m.empty);
        assert_eq!(m.color, Color::new(1.0, 1.0, 1.0, 1.0));
        assert_eq!(m.fluid_index, NULL_FLUID_INDEX);
        assert!(m.culls_neighbors);
        assert!(m.contributes_to_ao);
    }

    #[test]
    fn air_and_solid_cube_library_is_meshable_after_bake() {
        let cube = solid_cube_model(Color::from_rgb(1.0, 0.0, 0.0));
        assert!(!cube.empty);
        assert_eq!(cube.model.surface_count, 1);
        assert!(!cube.model.sides_surfaces[0][0].is_empty());

        let mut lib = BakedLibrary::with_air_and_solid_cube(Color::from_rgb(0.4, 0.4, 0.4));
        assert!(lib.models[0].empty);
        assert!(!lib.models[1].empty);
        crate::meshers::blocky::bake_library(&mut lib);
        assert!(lib.side_pattern_count > 0);
    }

    #[test]
    fn baked_model_clear_resets_empty() {
        let mut m = BakedModel {
            empty: false,
            model: BakedModelMesh {
                surface_count: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        m.clear();
        assert!(m.empty);
    }

    #[test]
    fn baked_library_has_model_respects_size() {
        let lib = BakedLibrary {
            models: vec![BakedModel::default(), BakedModel::default()],
            ..Default::default()
        };
        assert!(lib.has_model(0));
        assert!(lib.has_model(1));
        assert!(!lib.has_model(2));
    }

    #[test]
    fn side_surface_is_empty_when_no_indices() {
        let ss = SideSurface::default();
        assert!(ss.is_empty());
    }

    #[test]
    fn aabb_new_stores_min_max() {
        let a = Aabb::new(Vector3f::new(0.0, 0.0, 0.0), Vector3f::new(1.0, 1.0, 1.0));
        assert_eq!(a.min, Vector3f::new(0.0, 0.0, 0.0));
        assert_eq!(a.max, Vector3f::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn side_constants_match_cube_tables() {
        assert_eq!(Side::COUNT, 6);
        assert_eq!(MAX_SURFACES, 2);
    }
}
