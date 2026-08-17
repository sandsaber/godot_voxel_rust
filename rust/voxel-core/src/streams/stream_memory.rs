//! `streams::stream_memory` — supported in-memory voxel stream backend.
//!
//! Ported from `streams/voxel_stream_memory.{h,cpp}`. This runtime backend
//! stores blocks in a map keyed on `(position, lod)` and round-trips them
//! through `save_block` / `load_block` without touching the filesystem. It is
//! suitable for transient terrain data that only needs to live for the current
//! process.
//!
//! ## What changed from C++
//!
//! The C++ `VoxelStreamMemory` inherits `VoxelStream` (a Godot `Resource`,
//! which drags in `ClassDB`, `GDCLASS`, `_bind_methods`, per-LoD maps, an
//! `artificial_save_latency_usec` knob and instance blocks). The Rust port keeps
//! the engine-agnostic data storage and implements the Phase 4
//! [`VoxelStream`](super::voxel_stream::VoxelStream) trait.
//!
//! The C++ storage is `FixedArray<Lod, MAX_LOD>` with one
//! `StdUnorderedMap<Vector3i, VoxelChunk>` per LoD, each `VoxelChunk` wrapping
//! a `VoxelBuffer`. We collapse that to a flat
//! [`HashMap`]`<(Vector3i, u8), VoxelBuffer>` keyed on `(position, lod)`, which
//! is the same shape the sibling [`BlockCache`](crate::streams::BlockCache)
//! uses; the per-LoD split only existed to shard the `Mutex`.

use super::voxel_stream::{
    LoadResult, SaveMode, StreamResult, VoxelLoadQuery, VoxelSaveQuery, VoxelStream,
};
use crate::constants::voxel_constants::MAX_LOD;
use crate::math::Vector3i;
use crate::storage::voxel_buffer::ALL_CHANNELS_MASK;
use crate::storage::VoxelBuffer;
use std::collections::HashMap;
use std::sync::{
    PoisonError, RwLock as StdRwLock, RwLockReadGuard as StdRwLockReadGuard,
    RwLockWriteGuard as StdRwLockWriteGuard,
};

/// Supported in-memory [`VoxelStream`] backend that stores block copies in a
/// `HashMap` without touching the filesystem.
///
/// The stream owns every stored [`VoxelBuffer`]. [`MemoryStream::save_block`]
/// copies the supplied buffer in (matching the C++ `copy_to`), and
/// [`MemoryStream::load_block`] hands back a fresh copy on a hit — the C++
/// memory stream likewise retains ownership and the caller's buffer is
/// populated by copy.
#[derive(Debug, Default)]
pub struct MemoryStream {
    blocks: StdRwLock<HashMap<(Vector3i, u8), VoxelBuffer>>,
}

impl MemoryStream {
    /// Empty stream. Matches a default-constructed `VoxelStreamMemory`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of blocks currently stored. The C++ class has no exact
    /// equivalent (it exposes `load_all_blocks` instead), but the count is the
    /// natural invariant for tests.
    pub fn len(&self) -> usize {
        self.read_blocks().len()
    }

    /// Whether the stream holds any blocks.
    pub fn is_empty(&self) -> bool {
        self.read_blocks().is_empty()
    }

    /// Whether the stream reports the given persistence mode. Always
    /// [`SaveMode::Memory`] for [`MemoryStream`]; provided so a generic test
    /// harness can branch on it the way it would on `VoxelStream::supports_*`.
    pub fn get_supported_save_mode(&self) -> SaveMode {
        SaveMode::Memory
    }

    /// Store a copy of `voxels` at `(position, lod)`, overwriting any prior
    /// block at that key. Ported from `save_voxel_blocks` (single-block path).
    ///
    /// The C++ path copies via `voxel_buffer.copy_to(dst.voxels, true)`; we do
    /// the same, cloning the source buffer into the map so the caller keeps
    /// its own copy. Saving a block whose size is empty is silently ignored,
    /// matching the "you can't have meaningfully saved nothing" invariant the
    /// cache enforces too.
    pub fn save_block(&self, position: Vector3i, lod: u8, voxels: &VoxelBuffer) {
        if voxels.size().is_empty_size() {
            return;
        }
        let mut entry = VoxelBuffer::new(voxels.allocator());
        voxels.copy_to(&mut entry);
        self.write_blocks().insert((position, lod), entry);
    }

    /// Load the block at `(position, lod)` into `out_voxels`. Ported from
    /// `load_voxel_blocks` (single-block path).
    ///
    /// Returns [`LoadResult::Found`] and copies the stored buffer into
    /// `out_voxels` (resized to match) on a hit; [`LoadResult::NotFound`] on a
    /// miss, leaving `out_voxels` untouched — exactly as the C++ code leaves
    /// `q.voxel_buffer` untouched when setting `RESULT_BLOCK_NOT_FOUND`.
    pub fn load_block(
        &self,
        position: Vector3i,
        lod: u8,
        out_voxels: &mut VoxelBuffer,
    ) -> LoadResult {
        let blocks = self.read_blocks();
        let Some(stored) = blocks.get(&(position, lod)) else {
            return LoadResult::NotFound;
        };
        stored.copy_to(out_voxels);
        LoadResult::Found
    }

    /// Remove a stored block. Useful when a test wants to model a block being
    /// deleted between a save and a subsequent load. No direct C++ counterpart
    /// (the memory stream never erases), but trivially faithful to the
    /// underlying map storage.
    pub fn remove(&self, position: Vector3i, lod: u8) -> bool {
        self.write_blocks().remove(&(position, lod)).is_some()
    }

    /// Drop every stored block.
    pub fn clear(&self) {
        self.write_blocks().clear();
    }

    fn read_blocks(&self) -> StdRwLockReadGuard<'_, HashMap<(Vector3i, u8), VoxelBuffer>> {
        self.blocks.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write_blocks(&self) -> StdRwLockWriteGuard<'_, HashMap<(Vector3i, u8), VoxelBuffer>> {
        self.blocks.write().unwrap_or_else(PoisonError::into_inner)
    }
}

impl VoxelStream for MemoryStream {
    fn load_voxel_block(&self, query: VoxelLoadQuery<'_>) -> StreamResult<LoadResult> {
        Ok(self.load_block(
            query.position_in_blocks,
            query.lod_index,
            query.voxel_buffer,
        ))
    }

    fn save_voxel_block(&self, query: VoxelSaveQuery<'_>) -> StreamResult<()> {
        self.save_block(
            query.position_in_blocks,
            query.lod_index,
            query.voxel_buffer,
        );
        Ok(())
    }

    fn get_used_channels_mask(&self) -> u8 {
        ALL_CHANNELS_MASK
    }

    fn get_lod_count(&self) -> u8 {
        MAX_LOD as u8
    }

    fn get_supported_save_mode(&self) -> SaveMode {
        SaveMode::Memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vector3i;
    use crate::storage::{Allocator, ChannelDepth, ChannelId, VoxelBuffer};
    use crate::streams::{VoxelLoadQuery, VoxelSaveQuery, VoxelStream};

    /// Build a small non-uniform block: a 2³ buffer with `value` written to one
    /// voxel in the Type channel, so a copy is observably different from a
    /// freshly-created uniform-default buffer.
    fn sample_block(value: u64) -> VoxelBuffer {
        let mut b = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        b.set_voxel(value, 1, 0, 1, ChannelId::Type.index());
        b
    }

    /// Read the Type-channel voxel back, exercising the (de)compress path so
    /// stored copies stay byte-identical with the original.
    fn type_voxel(buf: &VoxelBuffer, x: i32, y: i32, z: i32) -> u64 {
        buf.get_voxel(x, y, z, ChannelId::Type.index())
    }

    #[test]
    fn supported_save_mode_is_memory() {
        let stream = MemoryStream::new();
        assert_eq!(stream.get_supported_save_mode(), SaveMode::Memory);
    }

    #[test]
    fn save_then_load_round_trips_block_data() {
        let stream = MemoryStream::new();
        let pos = Vector3i::new(5, -3, 1);
        let stored = sample_block(123);

        stream.save_block(pos, 0, &stored);
        assert_eq!(stream.len(), 1);

        let mut loaded = VoxelBuffer::new(Allocator::Default);
        assert_eq!(stream.load_block(pos, 0, &mut loaded), LoadResult::Found);
        assert_eq!(loaded.size(), Vector3i::new(2, 2, 2));
        assert_eq!(type_voxel(&loaded, 1, 0, 1), 123);
        // Untouched voxels keep the (now-materialized) default of 0.
        assert_eq!(type_voxel(&loaded, 0, 0, 0), 0);
    }

    #[test]
    fn save_then_load_preserves_block_and_voxel_metadata() {
        let stream = MemoryStream::new();
        let pos = Vector3i::new(1, 2, 3);
        let mut stored = sample_block(9);
        stored.set_block_metadata(crate::storage::MetadataValue::Text("meta".into()));
        stored.set_voxel_metadata(
            Vector3i::new(1, 0, 1),
            crate::storage::MetadataValue::Int(4),
        );

        stream.save_block(pos, 0, &stored);

        let mut loaded = VoxelBuffer::new(Allocator::Default);
        assert_eq!(stream.load_block(pos, 0, &mut loaded), LoadResult::Found);
        assert_eq!(
            *loaded.block_metadata(),
            crate::storage::MetadataValue::Text("meta".into())
        );
        assert_eq!(
            loaded.voxel_metadata(Vector3i::new(1, 0, 1)),
            Some(&crate::storage::MetadataValue::Int(4))
        );
    }

    #[test]
    fn load_returns_not_found_on_miss_and_leaves_buffer_untouched() {
        let stream = MemoryStream::new();
        let mut out = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        out.set_voxel(7, 0, 0, 0, ChannelId::Type.index());

        assert_eq!(
            stream.load_block(Vector3i::new(0, 0, 0), 0, &mut out),
            LoadResult::NotFound
        );
        // Miss must not touch the caller's buffer.
        assert_eq!(type_voxel(&out, 0, 0, 0), 7);
    }

    #[test]
    fn save_overwrites_existing_block_at_same_key() {
        let stream = MemoryStream::new();
        let pos = Vector3i::new(2, 2, 2);

        stream.save_block(pos, 0, &sample_block(1));
        stream.save_block(pos, 0, &sample_block(2));
        assert_eq!(stream.len(), 1, "overwrite must not grow the entry count");

        let mut loaded = VoxelBuffer::new(Allocator::Default);
        assert_eq!(stream.load_block(pos, 0, &mut loaded), LoadResult::Found);
        assert_eq!(type_voxel(&loaded, 1, 0, 1), 2);
    }

    #[test]
    fn save_ignores_empty_size_buffer() {
        let stream = MemoryStream::new();
        let empty = VoxelBuffer::new(Allocator::Default); // size (0,0,0)
        stream.save_block(Vector3i::new(0, 0, 0), 0, &empty);
        assert!(stream.is_empty(), "empty-size buffer must not be stored");
    }

    #[test]
    fn keys_distinct_on_position_and_lod() {
        let stream = MemoryStream::new();
        let pos = Vector3i::new(1, 1, 1);

        stream.save_block(pos, 0, &sample_block(10));
        stream.save_block(pos, 2, &sample_block(20));
        assert_eq!(stream.len(), 2);

        let mut a = VoxelBuffer::new(Allocator::Default);
        let mut b = VoxelBuffer::new(Allocator::Default);
        assert_eq!(stream.load_block(pos, 0, &mut a), LoadResult::Found);
        assert_eq!(stream.load_block(pos, 2, &mut b), LoadResult::Found);
        assert_eq!(type_voxel(&a, 1, 0, 1), 10);
        assert_eq!(type_voxel(&b, 1, 0, 1), 20);
    }

    #[test]
    fn round_trips_custom_channel_depths() {
        let stream = MemoryStream::new();
        let pos = Vector3i::new(-4, 5, 6);
        let mut stored = sample_block(0x1234_5678);
        stored.set_channel_depth(ChannelId::Type.index(), ChannelDepth::Bit32);
        stored.set_voxel(0x1234_5678, 1, 0, 1, ChannelId::Type.index());

        stream.save_block(pos, 0, &stored);

        let mut loaded = VoxelBuffer::new(Allocator::Default);
        assert_eq!(stream.load_block(pos, 0, &mut loaded), LoadResult::Found);
        assert_eq!(
            loaded.channel_depth(ChannelId::Type.index()),
            ChannelDepth::Bit32
        );
        assert_eq!(type_voxel(&loaded, 1, 0, 1), 0x1234_5678);
    }

    #[test]
    fn implements_voxel_stream_contract() {
        let stream = MemoryStream::new();
        let pos = Vector3i::new(8, 0, -2);
        let stored = sample_block(77);
        stream
            .save_voxel_block(VoxelSaveQuery::new(&stored, pos, 0))
            .unwrap();

        let mut loaded = VoxelBuffer::new(Allocator::Default);
        let result = stream
            .load_voxel_block(VoxelLoadQuery::new(&mut loaded, pos, 0))
            .unwrap();

        assert_eq!(result, LoadResult::Found);
        assert_eq!(type_voxel(&loaded, 1, 0, 1), 77);
        assert_eq!(stream.get_supported_save_mode(), SaveMode::Memory);
        assert_eq!(stream.get_used_channels_mask(), ALL_CHANNELS_MASK);
        assert_eq!(stream.get_lod_count(), MAX_LOD as u8);
    }
}
