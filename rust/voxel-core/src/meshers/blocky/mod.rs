//! `meshers::blocky` — Minecraft-style blocky voxel mesher.
//!
//! Ported from `meshers/blocky/`. Produces meshes from a voxel-type channel
//! using a [`baked_library::BakedLibrary`] of pre-baked models. Unlike the
//! cubes mesher (which synthesizes one quad per face boundary), the blocky
//! mesher looks up each voxel's model in the library and emits its baked
//! geometry, applying neighbor-based face culling and ambient occlusion.
//!
//! ## Current
//! - [`baked_library`] — `BakedModel`/`BakedLibrary`/`BakedFluid` plain-data
//!   structs (no Godot dependency).
//! - [`bake`] — side-culling matrix generation + cutout-surface baking
//!   ([`bake::bake_library`]).
//! - [`mesher`] — [`mesher::generate_mesh`] core algorithm (face culling + AO).
//! - [`lod_skirts`] — [`lod_skirts::append_skirts`] LOD seam-skirt appending.
//! - [`shadow_occluders`] — [`shadow_occluders::generate_shadow_occluders`]
//!   shadow occluder geometry generation.
//!
//! ## Deferred (Phase 5)
//! - The Godot `Resource` / `Ref<Material>` / editor layer (`VoxelMesherBlocky`,
//!   `VoxelBlockyLibraryBase`, `VoxelBlockyModel*`).
//! - `TintSampler` integration for `lod_skirts` (needs `VoxelBuffer` channels).

pub mod bake;
pub mod baked_library;
pub mod lod_skirts;
pub mod mesh_geometry;
pub mod mesher;
pub mod shadow_occluders;

pub use bake::bake_library;
pub use baked_library::{
    full_cube_side_surface, solid_cube_model, Aabb, BakedFluid, BakedLibrary, BakedModel,
    BakedModelMesh, DynamicBitset, FluidSurface, ModelSurface, SideSurface, AIR_ID,
    FLUID_BOTTOM_HEIGHT, FLUID_TOP_HEIGHT, MAX_FLUIDS, MAX_MATERIALS, MAX_MODELS, MAX_SURFACES,
    NULL_FLUID_INDEX,
};
pub use lod_skirts::append_skirts;
pub use mesh_geometry::{
    apply_ortho_rotation, bake_mesh_model, MeshGeometry, DEFAULT_SIDE_VERTEX_TOLERANCE,
};
pub use mesher::{generate_mesh, BlockyArrays};
pub use shadow_occluders::{
    generate_occluders_geometry, generate_shadow_occluders, ShadowOccluderArrays,
};
