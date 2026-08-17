//! `streams::stream_cache` — in-memory block cache for voxel streams.
//!
//! Ported from `streams/voxel_stream_cache.{h,cpp}`. A simple in-memory
//! database mapping `(block_position, lod)` → [`VoxelBuffer`] so callers can
//! cache recently-loaded blocks and re-serve them without hitting the
//! underlying stream, or buffer up saves and flush them later in batch.
//!
//! ## What changed from C++
//!
//! The C++ `VoxelStreamCache` is laid out as `FixedArray<Lod, MAX_LOD>` where
//! each `Lod` carries its own `RWLock` + `StdUnorderedMap<Vector3i, Block>`. We
//! collapse all of that into a single flat [`HashMap`]`<(Vector3i, u8),
//! VoxelBuffer>` keyed on `(position, lod)`, as the rest of the Rust port is
//! single-threaded (Phase 4) and the per-lod split was only there to shard the
//! lock. We also omit the instancer branch (`VOXEL_ENABLE_INSTANCER`):
//! instance-block caching is tracked separately and not part of this port.
//!
//! The C++ `Block` carried extra flags (`has_voxels`, `voxels_deleted`) that
//! only mattered for the (still unimplemented in C++) "erase" use case. In Rust
//! presence in the map *is* the "has voxels" signal; `remove` is the erase
//! path, so the flags are unnecessary.

use crate::math::Vector3i;
use crate::storage::VoxelBuffer;
use std::collections::HashMap;

/// In-memory cache of voxel blocks, keyed on `(block_position, lod)`.
///
/// Ported from `VoxelStreamCache`. Single-threaded; the C++ `RWLock` per LoD
/// is omitted (Phase 4 will reintroduce locking at a higher level).
///
/// The cache owns its [`VoxelBuffer`]s. [`BlockCache::set`] copies the supplied
/// buffer into the cache (matching the C++ `copy_to` on insert), and
/// [`BlockCache::get`] returns a fresh copy — mirroring the C++ contract that
/// the cache retains ownership and the caller gets an independent buffer.
#[derive(Debug, Default)]
pub struct BlockCache {
    blocks: HashMap<(Vector3i, u8), VoxelBuffer>,
}

impl BlockCache {
    /// Empty cache. Matches the C++ default-constructed `VoxelStreamCache`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cached blocks. Matches `get_indicative_block_count`.
    ///
    /// "Indicative" because in C++ the count lumps voxels and instances
    /// together; here it is just the entry count.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether the cache holds no blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Whether a block at `(position, lod)` is cached. Matches the implicit
    /// "found" check the C++ reader performs before touching `Block::has_voxels`.
    pub fn contains(&self, position: Vector3i, lod: u8) -> bool {
        self.blocks.contains_key(&(position, lod))
    }

    /// Copy a cached block into `out_voxels`. Ported from `load_voxel_block`.
    ///
    /// Returns `true` when a cached block was found and copied (the C++
    /// `RESULT_BLOCK_FOUND` path), `false` on a miss (`RESULT_BLOCK_NOT_FOUND`).
    /// On a hit the cached buffer is deep-copied into `out_voxels`, resized to
    /// match, so the caller receives an independent snapshot — exactly as the
    /// C++ `block.voxels.copy_to(out_voxels, true)` does.
    pub fn get(&self, position: Vector3i, lod: u8, out_voxels: &mut VoxelBuffer) -> bool {
        let Some(cached) = self.blocks.get(&(position, lod)) else {
            return false;
        };
        cached.copy_to(out_voxels);
        true
    }

    /// Store (a copy of) `voxels` into the cache at `(position, lod)`. Ported
    /// from `save_voxel_block`.
    ///
    /// The C++ path takes ownership via `move_to` on overwrite and `copy_to`
    /// on first insert; in Rust we always clone into the map so the caller
    /// keeps its buffer (matching the "copy on insert" half of the C++
    /// contract, which is the half external callers depend on).
    ///
    /// Like the C++ `ZN_ASSERT_RETURN_MSG` this is a no-op when `voxels` has
    /// an empty size — saving a zero-volume buffer is a bug, not a cache state.
    pub fn set(&mut self, position: Vector3i, lod: u8, voxels: &VoxelBuffer) {
        if voxels.size().is_empty_size() {
            // Mirrors the C++ assert: an empty-size buffer is unexpected. We
            // silently bail rather than cache a degenerate entry.
            return;
        }
        let mut entry = VoxelBuffer::new(voxels.allocator());
        voxels.copy_to(&mut entry);
        self.blocks.insert((position, lod), entry);
    }

    /// Drop a cached block. Returns `true` if a block was present and removed.
    /// This is the erase path the C++ `Block::voxels_deleted` flag was
    /// scaffolding for.
    pub fn remove(&mut self, position: Vector3i, lod: u8) -> bool {
        self.blocks.remove(&(position, lod)).is_some()
    }

    /// Drop every cached block. Ported from the per-lod `blocks.clear()` inside
    /// the C++ `flush` template (without invoking a save callback).
    pub fn clear(&mut self) {
        self.blocks.clear();
    }

    /// Drain the cache, invoking `save_func` once per cached block with its
    /// `(position, lod, voxels)` triple, then leave the cache empty. Ported
    /// from the C++ `flush<F>` template, minus the per-lod lock.
    ///
    /// Blocks are handed out by reference in iteration order; the C++ map is
    /// unordered so this matches the C++ "no ordering guarantee" behavior.
    pub fn flush<F>(&mut self, mut save_func: F)
    where
        F: FnMut(Vector3i, u8, &VoxelBuffer),
    {
        for ((position, lod), voxels) in self.blocks.drain() {
            save_func(position, lod, &voxels);
        }
    }
}

/// Read a single voxel back through the cache's public API in tests, to avoid
/// reaching into private `VoxelBuffer` internals from the test module. Kept
/// in the binary only behind `#[cfg(test)]`.
#[cfg(test)]
fn sample_voxel(buf: &VoxelBuffer, x: i32, y: i32, z: i32) -> u64 {
    // Type channel is index 0; reading a voxel exercises the full
    // (de)compress round-trip so cached copies stay byte-identical.
    buf.get_voxel(x, y, z, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vector3i;
    use crate::storage::{Allocator, ChannelId, VoxelBuffer};

    /// Build a small non-uniform block: a 2³ buffer with `value` in the Type
    /// channel. Non-uniform so a copy is observably different from a freshly
    /// created (uniform-default) buffer.
    fn sample_block(value: u64) -> VoxelBuffer {
        let mut b = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        b.set_voxel(value, 1, 1, 1, ChannelId::Type.index());
        b
    }

    #[test]
    fn set_then_get_round_trips_voxel_data() {
        let mut cache = BlockCache::new();
        let pos = Vector3i::new(3, -1, 7);
        let stored = sample_block(42);

        cache.set(pos, 0, &stored);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(pos, 0));

        let mut out = VoxelBuffer::new(Allocator::Default);
        assert!(cache.get(pos, 0, &mut out));
        assert_eq!(out.size(), Vector3i::new(2, 2, 2));
        assert_eq!(sample_voxel(&out, 1, 1, 1), 42);
    }

    #[test]
    fn set_then_get_preserves_block_and_voxel_metadata() {
        let mut cache = BlockCache::new();
        let pos = Vector3i::new(0, 0, 0);
        let mut stored = sample_block(5);
        stored.set_block_metadata(crate::storage::MetadataValue::Int(3));
        stored.set_voxel_metadata(
            Vector3i::new(1, 1, 1),
            crate::storage::MetadataValue::Text("cached".into()),
        );

        cache.set(pos, 0, &stored);
        let mut out = VoxelBuffer::new(Allocator::Default);
        assert!(cache.get(pos, 0, &mut out));
        assert_eq!(*out.block_metadata(), crate::storage::MetadataValue::Int(3));
        assert_eq!(
            out.voxel_metadata(Vector3i::new(1, 1, 1)),
            Some(&crate::storage::MetadataValue::Text("cached".into()))
        );
    }

    #[test]
    fn get_returns_false_on_miss_and_leaves_buffer_untouched() {
        let cache = BlockCache::new();
        let mut out = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        out.set_voxel(9, 0, 0, 0, ChannelId::Type.index());

        assert!(!cache.get(Vector3i::new(0, 0, 0), 2, &mut out));
        // Miss must not touch the caller's buffer.
        assert_eq!(sample_voxel(&out, 0, 0, 0), 9);
    }

    #[test]
    fn set_overwrites_existing_entry() {
        let mut cache = BlockCache::new();
        let pos = Vector3i::new(1, 2, 3);

        cache.set(pos, 0, &sample_block(1));
        cache.set(pos, 0, &sample_block(2));
        // Overwrite must not grow the entry count.
        assert_eq!(cache.len(), 1);

        let mut out = VoxelBuffer::new(Allocator::Default);
        assert!(cache.get(pos, 0, &mut out));
        assert_eq!(sample_voxel(&out, 1, 1, 1), 2);
    }

    #[test]
    fn set_ignores_empty_size_buffer() {
        let mut cache = BlockCache::new();
        let empty = VoxelBuffer::new(Allocator::Default); // size (0,0,0)
        cache.set(Vector3i::new(0, 0, 0), 0, &empty);
        assert_eq!(cache.len(), 0, "empty-size buffer must not be cached");
    }

    #[test]
    fn keys_on_both_position_and_lod() {
        let mut cache = BlockCache::new();
        let pos = Vector3i::new(4, 5, 6);

        cache.set(pos, 0, &sample_block(10));
        cache.set(pos, 1, &sample_block(20));
        assert_eq!(cache.len(), 2);

        let mut lod0 = VoxelBuffer::new(Allocator::Default);
        let mut lod1 = VoxelBuffer::new(Allocator::Default);
        assert!(cache.get(pos, 0, &mut lod0));
        assert!(cache.get(pos, 1, &mut lod1));
        assert_eq!(sample_voxel(&lod0, 1, 1, 1), 10);
        assert_eq!(sample_voxel(&lod1, 1, 1, 1), 20);
    }

    #[test]
    fn remove_drops_entry_and_returns_whether_present() {
        let mut cache = BlockCache::new();
        let pos = Vector3i::new(-2, 8, 0);
        cache.set(pos, 3, &sample_block(7));

        assert!(cache.remove(pos, 3));
        assert!(!cache.contains(pos, 3));
        assert_eq!(cache.len(), 0);
        // Removing a missing entry is a no-op returning false.
        assert!(!cache.remove(pos, 3));
    }

    #[test]
    fn clear_empties_the_cache() {
        let mut cache = BlockCache::new();
        cache.set(Vector3i::new(0, 0, 0), 0, &sample_block(1));
        cache.set(Vector3i::new(1, 1, 1), 0, &sample_block(2));
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
        assert!(!cache.contains(Vector3i::new(0, 0, 0), 0));
    }

    #[test]
    fn flush_invokes_callback_for_each_block_then_empties() {
        let mut cache = BlockCache::new();
        cache.set(Vector3i::new(0, 0, 0), 0, &sample_block(11));
        cache.set(Vector3i::new(1, 0, 0), 1, &sample_block(22));

        let mut seen = 0;
        cache.flush(|_pos, _lod, voxels| {
            seen += 1;
            // The handed-out buffer must be readable (not a degenerate copy).
            assert_eq!(voxels.size(), Vector3i::new(2, 2, 2));
        });
        assert_eq!(seen, 2);
        assert!(cache.is_empty(), "flush must drain the cache");
    }
}
