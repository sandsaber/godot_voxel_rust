//! `streams` — serialization formats for voxel block persistence.
//!
//! Each submodule handles one of godot_voxel's internal on-disk / in-memory
//! block representations. These are the formats the terrain layer round-trips
//! through its block cache; they are **not** public asset formats like the
//! `.vox` parser in [`crate::format::vox`].
//!
//! ## Current
//! - [`block_serializer`] — `VoxelBuffer` ↔ bytes (v4 format), with optional
//!   LZ4/ZSTD compression. Depends on [`compressed_data`].
//! - [`compressed_data`] — LZ4/ZSTD compression envelope used by the block
//!   serializer. LZ4 is pure-Rust (`lz4_flex`); ZSTD is behind a feature.
//! - [`instance_data`] — lossy-compressed per-block instance transforms
//!   (instanced grass / detail).
//! - [`region`] — region-file archive format (`.vxr`): sector-based sparse
//!   block storage with header LUT, built on [`block_serializer`].
//! - [`stream_cache`] — in-memory `(position, lod)` → `VoxelBuffer` cache
//!   (`BlockCache`), ported from `voxel_stream_cache`. Single-threaded; the
//!   C++ per-LoD `RWLock` is omitted (Phase 4).
//! - [`block_data_output`] — load/save task result payload mirroring the
//!   engine's `BlockDataOutput` shape.
//! - [`load_block_data_task`] / [`save_block_data_task`] — voxel-only threaded
//!   stream I/O tasks used by the Phase 4 terrain streamer.
//! - [`voxel_stream`] — engine-agnostic base stream contract ported from
//!   `streams/voxel_stream`.
//! - [`stream_memory`] — fake in-memory `VoxelStream` for tests (`MemoryStream`),
//!   ported from `voxel_stream_memory`.

pub mod block_data_output;
pub mod block_serializer;
pub mod compressed_data;
pub mod decode_limits;
pub(crate) mod flush_voxel_stream_task;
pub mod instance_data;
pub mod load_block_data_task;
mod persistence_task;
pub mod region;
pub mod save_block_data_task;
pub mod stream_cache;
pub mod stream_memory;
pub mod variant_wire;
pub mod voxel_stream;

pub use block_data_output::{BlockDataOutput, BlockDataOutputKind};
pub use decode_limits::{DecodeLimitError, DecodeLimits};
pub use load_block_data_task::{
    BlockGenerationRequest, BlockGenerationTaskFactory, BlockGenerationTaskResult,
    LoadBlockDataParams, LoadBlockDataTask,
};
pub use persistence_task::{
    FlushTaskTerminal, PersistenceAcknowledgement, PersistenceIoPhase, SaveTaskTerminal,
};
pub use region::RegionFilesStream;
pub use save_block_data_task::SaveBlockDataTask;
pub use stream_cache::BlockCache;
pub use stream_memory::MemoryStream;
pub use voxel_stream::{
    LoadResult, SaveMode, StreamResult, VoxelLoadQuery, VoxelSaveQuery, VoxelStream,
    VoxelStreamError,
};
