//! Voxel edition tools — engine-agnostic shape operations and SDF blending.
//!
//! Ports the engine-agnostic half of `edition/funcs.h` (shapes, SDF operations,
//! `DoShapeSingleBuffer`). The Godot-facing `VoxelTool` / `VoxelToolBuffer`
//! wrappers live in `voxel-gdext`; this module provides the pure-Rust core.
//!
//! ## Status
//! MVP: `SdfSphere`, `SdfAxisAlignedBox`, SDF blend modes, and
//! `do_shape_single_buffer` for editing a single `VoxelBuffer`.

pub mod ops;
pub mod raycast;

pub use ops::{blend_sdf, do_box, do_sphere, EditMode, SdfBlendMode, VoxelToolBuffer};
pub use raycast::{voxel_raycast, VoxelRaycastHit, VoxelRaycastState};
