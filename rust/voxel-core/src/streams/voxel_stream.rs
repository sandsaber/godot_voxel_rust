//! Engine-agnostic voxel stream contract.
//!
//! Ported from `streams/voxel_stream.{h,cpp}` without Godot `Resource`
//! bindings. `RESULT_ERROR` maps to [`Err`], while a missing block remains a
//! normal [`LoadResult::NotFound`] so callers can fall back to generation.

use crate::constants::voxel_constants::{
    DEFAULT_BLOCK_SIZE_PO2, DEFAULT_MAX_SUPPORTED_BLOCK_COORDINATE,
    DEFAULT_MIN_SUPPORTED_BLOCK_COORDINATE,
};
use crate::math::{Box3i, Vector3i};
use crate::storage::VoxelBuffer;

/// Outcome of a voxel-block load attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadResult {
    /// The block was found and copied into the caller's buffer. Corresponds to
    /// `RESULT_BLOCK_FOUND`.
    Found,
    /// No block is stored at the queried `(position, lod)`. Corresponds to
    /// `RESULT_BLOCK_NOT_FOUND`.
    NotFound,
}

/// Persistence capability reported by a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaveMode {
    /// The stream cannot persist blocks at all.
    #[default]
    None,
    /// Blocks persist for the lifetime of the process.
    Memory,
    /// Blocks persist to a filesystem-backed store.
    Filesystem,
}

/// Stream-level failures. A missing block is intentionally not an error; use
/// [`LoadResult::NotFound`] for that case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoxelStreamError {
    InvalidLod { lod: u8, max_lod: u8 },
    InvalidBlockPosition { position: Vector3i },
    BlockFormatMismatch(String),
    UnsupportedOperation { operation: &'static str },
    Io(String),
    CorruptData(String),
}

impl std::fmt::Display for VoxelStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLod { lod, max_lod } => {
                write!(f, "invalid LOD {lod}, expected less than {max_lod}")
            }
            Self::InvalidBlockPosition { position } => {
                write!(f, "invalid block position {position:?}")
            }
            Self::BlockFormatMismatch(detail) => write!(f, "block format mismatch: {detail}"),
            Self::UnsupportedOperation { operation } => {
                write!(f, "unsupported stream operation: {operation}")
            }
            Self::Io(message) => write!(f, "stream I/O error: {message}"),
            Self::CorruptData(message) => write!(f, "corrupt stream data: {message}"),
        }
    }
}

impl std::error::Error for VoxelStreamError {}

pub type StreamResult<T> = Result<T, VoxelStreamError>;

/// Single voxel block load query.
#[derive(Debug)]
pub struct VoxelLoadQuery<'a> {
    pub voxel_buffer: &'a mut VoxelBuffer,
    pub position_in_blocks: Vector3i,
    pub lod_index: u8,
}

impl<'a> VoxelLoadQuery<'a> {
    pub fn new(
        voxel_buffer: &'a mut VoxelBuffer,
        position_in_blocks: Vector3i,
        lod_index: u8,
    ) -> Self {
        Self {
            voxel_buffer,
            position_in_blocks,
            lod_index,
        }
    }
}

/// Single voxel block save query.
#[derive(Debug)]
pub struct VoxelSaveQuery<'a> {
    pub voxel_buffer: &'a VoxelBuffer,
    pub position_in_blocks: Vector3i,
    pub lod_index: u8,
}

impl<'a> VoxelSaveQuery<'a> {
    pub fn new(voxel_buffer: &'a VoxelBuffer, position_in_blocks: Vector3i, lod_index: u8) -> Self {
        Self {
            voxel_buffer,
            position_in_blocks,
            lod_index,
        }
    }
}

/// Source/sink of paged voxel blocks.
pub trait VoxelStream: Send + Sync {
    fn load_voxel_block(&self, _query: VoxelLoadQuery<'_>) -> StreamResult<LoadResult> {
        Ok(LoadResult::NotFound)
    }

    fn save_voxel_block(&self, _query: VoxelSaveQuery<'_>) -> StreamResult<()> {
        Ok(())
    }

    fn load_voxel_blocks(
        &self,
        queries: &mut [VoxelLoadQuery<'_>],
    ) -> Vec<StreamResult<LoadResult>> {
        queries
            .iter_mut()
            .map(|query| {
                self.load_voxel_block(VoxelLoadQuery {
                    voxel_buffer: &mut *query.voxel_buffer,
                    position_in_blocks: query.position_in_blocks,
                    lod_index: query.lod_index,
                })
            })
            .collect()
    }

    fn save_voxel_blocks(&self, queries: &[VoxelSaveQuery<'_>]) -> Vec<StreamResult<()>> {
        queries
            .iter()
            .map(|query| {
                self.save_voxel_block(VoxelSaveQuery {
                    voxel_buffer: query.voxel_buffer,
                    position_in_blocks: query.position_in_blocks,
                    lod_index: query.lod_index,
                })
            })
            .collect()
    }

    fn get_used_channels_mask(&self) -> u8 {
        0
    }

    fn get_block_size_po2(&self) -> u8 {
        DEFAULT_BLOCK_SIZE_PO2
    }

    fn get_lod_count(&self) -> u8 {
        1
    }

    fn get_supported_save_mode(&self) -> SaveMode {
        SaveMode::None
    }

    fn get_supported_block_range(&self) -> Box3i {
        Box3i::from_min_max(
            Vector3i::splat(DEFAULT_MIN_SUPPORTED_BLOCK_COORDINATE),
            Vector3i::splat(DEFAULT_MAX_SUPPORTED_BLOCK_COORDINATE),
        )
    }

    fn flush(&self) -> StreamResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{LoadResult, SaveMode, VoxelLoadQuery, VoxelStream};
    use crate::constants::voxel_constants::{
        DEFAULT_BLOCK_SIZE_PO2, DEFAULT_MAX_SUPPORTED_BLOCK_COORDINATE,
        DEFAULT_MIN_SUPPORTED_BLOCK_COORDINATE,
    };
    use crate::math::{Box3i, Vector3i};
    use crate::storage::VoxelBuffer;

    struct EmptyStream;

    impl VoxelStream for EmptyStream {}

    #[test]
    fn default_load_returns_not_found() {
        let stream = EmptyStream;
        let mut buffer = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));

        let result = stream
            .load_voxel_block(VoxelLoadQuery::new(&mut buffer, Vector3i::new(1, 2, 3), 0))
            .unwrap();

        assert_eq!(result, LoadResult::NotFound);
    }

    #[test]
    fn default_stream_metadata_matches_cpp_defaults() {
        let stream = EmptyStream;

        assert_eq!(stream.get_supported_save_mode(), SaveMode::None);
        assert_eq!(stream.get_used_channels_mask(), 0);
        assert_eq!(stream.get_block_size_po2(), DEFAULT_BLOCK_SIZE_PO2);
        assert_eq!(stream.get_lod_count(), 1);
        assert_eq!(
            stream.get_supported_block_range(),
            Box3i::from_min_max(
                Vector3i::splat(DEFAULT_MIN_SUPPORTED_BLOCK_COORDINATE),
                Vector3i::splat(DEFAULT_MAX_SUPPORTED_BLOCK_COORDINATE),
            )
        );
        stream.flush().unwrap();
    }
}
