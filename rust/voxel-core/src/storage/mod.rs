//! Voxel storage primitives ported from `storage/`.
//!
//! Phase 3 includes the full engine-agnostic `VoxelBuffer` stack: dense storage,
//! channel compression, memory-pool allocation, format configuration and storage
//! helpers. Godot-facing resources and terrain streaming live in later phases.

pub mod buffer;
pub mod depth;
pub mod funcs;
pub mod voxel_buffer;
pub mod voxel_data;
pub mod voxel_data_block;
pub mod voxel_data_map;
pub mod voxel_format;
pub mod voxel_memory_pool;

pub use buffer::{DenseVoxelBuffer, VoxelBufferRead};
pub use depth::ChannelDepth;
pub use voxel_buffer::{Allocator, Channel, ChannelData, ChannelId, Compression, VoxelBuffer};
pub use voxel_data::{
    BlockLocation, BlockToSave, SharedVoxelData, SharedVoxelDataLodReadGuard,
    SharedVoxelDataMutationError, SharedVoxelGenerator, SharedVoxelStream, VoxelData,
    VoxelDataKeyRevision, VoxelDataLodResizeError,
};
pub use voxel_data_block::{Viewers, VoxelDataBlock};
pub use voxel_data_map::VoxelDataMap;
pub use voxel_format::{DepthRange, VoxelFormat};
pub use voxel_memory_pool::VoxelMemoryPool;
