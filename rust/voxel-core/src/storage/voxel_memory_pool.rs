//! Power-of-two byte-block pool for `VoxelBuffer` allocations.
//!
//! Ported from `storage/voxel_memory_pool.{h,cpp}`. The C++ pool hands out raw
//! `uint8_t*` blocks and recycles them by power-of-two size (up to 2²⁰ = 1 MiB);
//! larger allocations fall back to `malloc`. The scenario: most `VoxelBuffer`s
//! use the same power-of-two size, so recycling avoids realloc churn.
//!
//! This Rust port keeps the *semantics* (21 power-of-two buckets, thread-safe
//! recycle, fallback for oversized) but is idiomatic and safe: it pools owned
//! `Vec<u8>` blocks rather than raw pointers, so there is no manual free or
//! leak surface. Callers get a `Vec<u8>` back; "recycling" returns it to the
//! bucket for the next `allocate` of the same power-of-two to reuse.

use std::sync::Mutex;

/// Maximum power-of-two block size the pool handles: 2²⁰ = 1,048,576 bytes.
/// Allocations larger than this bypass the pool (allocated fresh each time).
/// Matches the C++ `FixedArray<Pool, 21>` (indices 0..=20).
pub const MAX_POT_EXPONENT: usize = 20;
/// Highest supported (pooled) allocation size, in bytes.
pub const HIGHEST_SUPPORTED_SIZE: usize = 1 << MAX_POT_EXPONENT;
/// Number of power-of-two buckets (exponents 0..=MAX_POT_EXPONENT inclusive).
const POT_COUNT: usize = MAX_POT_EXPONENT + 1;

/// Returns the power-of-two exponent `i` such that `2^i` is the smallest power
/// of two `>= size`. Used to pick the bucket. Matches `get_pool_index_from_size`.
///
/// Panics (debug) if `size` is 0 or exceeds `HIGHEST_SUPPORTED_SIZE`.
#[inline]
fn pool_index_from_size(size: usize) -> usize {
    debug_assert!(size != 0, "VoxelMemoryPool: zero-size allocation");
    debug_assert!(
        size <= HIGHEST_SUPPORTED_SIZE,
        "VoxelMemoryPool: size {size} exceeds pooled max {HIGHEST_SUPPORTED_SIZE}"
    );
    // Smallest k such that (1 << k) >= size == position of the highest set bit,
    // rounded up. Equivalent to the C++ get_next_power_of_two_32 + shift.
    let pot = crate::math::funcs::get_next_power_of_two_32(size as u32);
    crate::math::funcs::get_shift_from_power_of_two_32(pot) as usize
}

/// A thread-safe pool of recyclable byte buffers, bucketed by power-of-two size.
///
/// Mirrors `VoxelMemoryPool`. Not a singleton in Rust (the engine uses one
/// process-global instance; here callers own a `VoxelMemoryPool` and may wrap it
/// in a `lazy_static`/`OnceLock` if they want singleton behaviour).
pub struct VoxelMemoryPool {
    /// One bucket per power-of-two exponent. `buckets[i]` recycles blocks of
    /// `1 << i` bytes.
    buckets: [Mutex<Vec<Vec<u8>>>; POT_COUNT],
    /// Live (allocated, not yet recycled) block count — matches `_used_blocks`.
    used_blocks: std::sync::atomic::AtomicU32,
    /// Bytes currently handed out — matches `_used_memory`.
    used_memory: std::sync::atomic::AtomicU64,
    /// Total bytes managed (used + idle in the pool) — matches `_total_memory`.
    total_memory: std::sync::atomic::AtomicU64,
}

impl Default for VoxelMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelMemoryPool {
    pub fn new() -> Self {
        // Each bucket starts empty. An array of length 21 of `Mutex<Vec<_>>`
        // can't derive `Default`, so use an inline const block (stable since
        // Rust 1.79) to repeat the const `Mutex::new(Vec::new())` initializer.
        Self {
            buckets: [const { Mutex::new(Vec::new()) }; POT_COUNT],
            used_blocks: std::sync::atomic::AtomicU32::new(0),
            used_memory: std::sync::atomic::AtomicU64::new(0),
            total_memory: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Allocate `size` bytes. If a recycled block of the matching power-of-two
    /// is available it is reused; otherwise a fresh `Vec` is allocated. The
    /// returned Vec has `len() == capacity() == next_pow2(size)` (the bucket
    /// size), which is `>= size`. Matches `allocate`.
    ///
    /// `size == 0` returns an empty Vec (the C++ version asserts and returns
    /// null; in Rust an empty Vec is the natural empty allocation).
    pub fn allocate(&self, size: usize) -> Vec<u8> {
        use std::sync::atomic::Ordering;
        if size == 0 {
            return Vec::new();
        }
        let block = if size > HIGHEST_SUPPORTED_SIZE {
            // Oversized: bypass the pool (matches the C++ fallback path).
            self.total_memory.fetch_add(size as u64, Ordering::Relaxed);
            vec![0u8; size]
        } else {
            let i = pool_index_from_size(size);
            let capacity = 1usize << i;
            // Try to reuse a recycled block. We account for memory by the
            // *capacity* actually occupied (the bucket size), which is stable
            // across allocate/recycle — the C++ accounts by the caller-requested
            // `size`, but since Rust recycles an owned Vec we lose that original
            // size, so capacity is the consistent metric.
            // Recover from mutex poisoning (a panic in another thread while
            // holding the bucket lock) rather than propagating it: a poisoned
            // pool bucket still contains valid recyclable blocks. Matches the
            // pattern used across the engine (voxel_data, terrain_core, ...).
            let reused = {
                self.buckets[i]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .pop()
            };
            match reused {
                Some(block) => block,
                None => {
                    self.total_memory
                        .fetch_add(capacity as u64, Ordering::Relaxed);
                    vec![0u8; capacity]
                }
            }
        };
        // Account by the bytes actually occupied (block.len() == capacity here).
        let occupied = block.len() as u64;
        self.used_blocks.fetch_add(1, Ordering::Relaxed);
        self.used_memory.fetch_add(occupied, Ordering::Relaxed);
        block
    }

    /// Return a previously-`allocate`d block to the pool for reuse. Matches
    /// `recycle`. The block's capacity must match a power-of-two bucket
    /// (debug-asserted); oversized blocks are simply dropped (they were not
    /// pooled to begin with).
    pub fn recycle(&self, block: Vec<u8>) {
        use std::sync::atomic::Ordering;
        if block.is_empty() {
            // Mirrors the C++ `block == nullptr && size == 0` early return.
            return;
        }
        let occupied = block.len() as u64;
        let capacity = block.capacity().max(block.len());
        self.used_blocks.fetch_sub(1, Ordering::Relaxed);
        self.used_memory.fetch_sub(occupied, Ordering::Relaxed);
        if capacity > HIGHEST_SUPPORTED_SIZE {
            // Oversized: was never pooled; freeing it. Account against
            // total_memory (mirrors the C++ ZN_FREE + _total_memory -= size path).
            self.total_memory.fetch_sub(occupied, Ordering::Relaxed);
            drop(block);
        } else {
            debug_assert!(
                crate::math::funcs::is_power_of_two(capacity),
                "recycled block capacity {capacity} is not a power of two"
            );
            let i = pool_index_from_size(capacity);
            // Only recycle if the capacity matches the bucket size exactly;
            // otherwise (e.g. Vec grew beyond its bucket) drop it to avoid
            // mismatched capacities confusing the pool.
            if capacity == 1usize << i {
                self.buckets[i]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(block);
            } else {
                // Doesn't match a bucket; free it.
                self.total_memory.fetch_sub(occupied, Ordering::Relaxed);
                drop(block);
            }
        }
    }

    /// Drop all recycled (idle) blocks, freeing their memory. Blocks currently
    /// handed out are unaffected. Matches `clear_unused_blocks`.
    pub fn clear_unused_blocks(&self) {
        use std::sync::atomic::Ordering;
        for (i, bucket) in self.buckets.iter().enumerate() {
            let mut b = bucket.lock().unwrap_or_else(|e| e.into_inner());
            let n = b.len();
            self.total_memory
                .fetch_sub(((1usize << i) * n) as u64, Ordering::Relaxed);
            b.clear();
        }
    }

    /// Live (allocated, not recycled) block count. Matches `debug_get_used_blocks`.
    #[inline]
    pub fn used_blocks(&self) -> u32 {
        self.used_blocks.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Bytes currently handed out. Matches `debug_get_used_memory`.
    #[inline]
    pub fn used_memory(&self) -> u64 {
        self.used_memory.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Total bytes managed (used + idle). Matches `debug_get_total_memory`.
    #[inline]
    pub fn total_memory(&self) -> u64 {
        self.total_memory.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl VoxelMemoryPool {
    /// Reset all counters and drop every pooled block. Matches `clear`. Used in
    /// tests; in the engine this runs at module shutdown.
    pub fn clear(&self) {
        use std::sync::atomic::Ordering;
        for bucket in self.buckets.iter() {
            bucket.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
        self.used_blocks.store(0, Ordering::Relaxed);
        self.used_memory.store(0, Ordering::Relaxed);
        self.total_memory.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_index_from_size_boundaries() {
        assert_eq!(pool_index_from_size(1), 0);
        assert_eq!(pool_index_from_size(2), 1);
        assert_eq!(pool_index_from_size(3), 2); // next pot of 3 is 4 = 2^2
        assert_eq!(pool_index_from_size(4), 2);
        assert_eq!(pool_index_from_size(5), 3); // next pot of 5 is 8
        assert_eq!(pool_index_from_size(1024), 10);
        assert_eq!(
            pool_index_from_size(HIGHEST_SUPPORTED_SIZE),
            MAX_POT_EXPONENT
        );
    }

    #[test]
    fn allocate_returns_at_least_size() {
        let pool = VoxelMemoryPool::new();
        for &size in &[1usize, 3, 7, 100, 1024, 100_000] {
            let b = pool.allocate(size);
            assert!(b.len() >= size, "len {} < requested {size}", b.len());
        }
    }

    #[test]
    fn allocate_zero_is_empty() {
        let pool = VoxelMemoryPool::new();
        assert!(pool.allocate(0).is_empty());
        assert_eq!(pool.used_blocks(), 0);
    }

    #[test]
    fn recycle_then_allocate_reuses() {
        let pool = VoxelMemoryPool::new();
        let b = pool.allocate(100);
        assert_eq!(pool.used_blocks(), 1);
        pool.recycle(b);
        assert_eq!(pool.used_blocks(), 0);
        // Re-allocating the same size should reuse the recycled block, not grow
        // total_memory (which only increases on fresh allocations).
        let total_before = pool.total_memory();
        let b2 = pool.allocate(100);
        let total_after = pool.total_memory();
        assert_eq!(
            total_after, total_before,
            "reuse should not increase total_memory"
        );
        assert_eq!(pool.used_blocks(), 1);
        pool.recycle(b2);
    }

    #[test]
    fn oversized_bypasses_pool() {
        let pool = VoxelMemoryPool::new();
        let huge = HIGHEST_SUPPORTED_SIZE * 2;
        let b = pool.allocate(huge);
        assert_eq!(b.len(), huge);
        assert_eq!(pool.used_memory(), huge as u64);
        // Recycling an oversized block just frees it (no bucket holds it).
        pool.recycle(b);
        assert_eq!(pool.used_blocks(), 0);
        assert_eq!(pool.used_memory(), 0);
    }

    #[test]
    fn clear_unused_drops_idle_blocks() {
        let pool = VoxelMemoryPool::new();
        let b1 = pool.allocate(64);
        let b2 = pool.allocate(64);
        pool.recycle(b1);
        pool.recycle(b2);
        // Two blocks idle in the bucket.
        assert!(pool.total_memory() > 0);
        pool.clear_unused_blocks();
        // After clearing, total_memory reflects only live blocks (none here).
        assert_eq!(pool.used_blocks(), 0);
    }

    #[test]
    fn used_memory_tracks_live_only() {
        let pool = VoxelMemoryPool::new();
        // Memory is accounted by occupied capacity (the bucket size), which is
        // the next power of two >= the request. Use an exact power of two so
        // len == request == accounted.
        let b = pool.allocate(1024);
        assert_eq!(pool.used_memory(), 1024);
        pool.recycle(b);
        assert_eq!(pool.used_memory(), 0);
    }
}
