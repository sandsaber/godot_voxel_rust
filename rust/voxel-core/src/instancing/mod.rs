//! Voxel instancing — scatter objects (trees, rocks, grass) on terrain surfaces.
//!
//! Ports the engine-agnostic half of `terrain/instancing/`. The Godot-facing
//! `VoxelInstancer` Node3D wrapper lives in `voxel-gdext`.
//!
//! ## Status
//! Instance blocks stream with terrain paging: generate per data block, drop
//! when the block leaves the viewer box.

pub mod block;
pub mod library;
pub mod scatter;

pub use block::{extract_surface_points, scatter_block_instances, InstanceBlock, InstanceBlockMap};
pub use library::{InstanceLibrary, InstanceLibraryItem, InstanceMeshType};
pub use scatter::{BlockInstanceData, InstanceGenerator, ScatterConfig};
