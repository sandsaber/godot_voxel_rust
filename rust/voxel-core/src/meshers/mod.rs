//! Meshers — convert voxel data into renderable triangle meshes.
//!
//! Ported from `meshers/`. Currently includes the transvoxel (smooth SDF)
//! mesher, the cubes (blocky) mesher, the blocky (model-library) mesher
//! data layer, and the [`voxel_mesher`] trait that unifies them for the
//! terrain meshing pipeline.

pub mod blocky;
pub mod builtin;
pub mod cubes;
pub mod mesh_block_task;
pub mod transvoxel;
pub mod voxel_mesher;

pub use builtin::{BlockyMesher, CubesMesher, TransvoxelMesher};
pub use mesh_block_task::{
    gather_voxels_cpu, BlockMeshOutput, MeshBlockKey, MeshBlockLocation, MeshBlockTask,
    MeshBlockTaskOutput, MeshBlockTaskParams, MeshBuildFeatures, MeshUploadSnapshot, PayloadState,
};
pub use voxel_mesher::{
    CollisionSurface, MeshArraysPool, MesherInput, MesherOutput, SharedVoxelMesher, Surface,
    SurfaceArrays, VoxelMesher,
};
