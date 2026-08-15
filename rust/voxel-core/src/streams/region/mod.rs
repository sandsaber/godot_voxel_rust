//! `streams::region` — godot_voxel's region-file archive format.
//!
//! Ported from `streams/region/region_file.{h,cpp}`. A region file (`*.vxr`)
//! stores up to `region_size³` voxel blocks in a sector-based sparse layout.
//! Each block is a length-prefixed [`crate::streams::block_serializer`] payload
//! (optionally LZ4/ZSTD-compressed) padded to a sector boundary; a header LUT
//! maps block positions to sector ranges.
//!
//! ## Current
//! - [`format`] — `RegionFormat`, `RegionBlockInfo`, on-disk constants.
//! - [`region_file`] — [`region_file::RegionFile`] with header save/load,
//!   sector allocation, `load_block`/`save_block`.
//! - [`region_files_stream`] — thread-safe, LOD-aware filesystem stream with
//!   one synchronized handle per open region.
//!
//! ## Deferred
//! - **Forest metadata and tools**: meta.vxrm JSON, LRU eviction,
//!   `convert_files` — C++ surfaces tied to Godot `Resource`/`JSON`.
//! - **v2→v3 legacy migration**: needs `FileAccess::insert_bytes` (grow-file-
//!   in-place); only relevant for reading old saves.
//! - **Cross-process file locking** (`file_utils.h`).

pub mod format;
pub mod region_file;
pub mod region_files_stream;

pub use format::{RegionBlockInfo, RegionFormat, FILE_EXTENSION, FORMAT_VERSION, MAGIC};
pub use region_file::{RegionError, RegionFile};
pub use region_files_stream::RegionFilesStream;
