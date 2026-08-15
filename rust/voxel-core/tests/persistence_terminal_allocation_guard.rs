use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use voxel_core::engine::StreamingDependency;
use voxel_core::math::Vector3i;
use voxel_core::storage::VoxelBuffer;
use voxel_core::streams::{MemoryStream, SaveBlockDataTask, VoxelStream};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: all operations are forwarded to `System` with the exact pointer and
// layout supplied by the caller; atomics only observe successful allocations.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplied a valid allocation layout.
        let allocation = unsafe { System.alloc(layout) };
        if COUNTING.load(Ordering::Relaxed) && !allocation.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        allocation
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer/layout pair came from this allocator's System.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplied a valid allocation layout.
        let allocation = unsafe { System.alloc_zeroed(layout) };
        if COUNTING.load(Ordering::Relaxed) && !allocation.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        allocation
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the pointer/layout pair and size are forwarded unchanged.
        let allocation = unsafe { System.realloc(pointer, layout, new_size) };
        if COUNTING.load(Ordering::Relaxed) && !allocation.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        allocation
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn persistence_terminal_take_allocates_nothing() {
    let stream: Arc<dyn VoxelStream> = Arc::new(MemoryStream::new());
    let mut task = SaveBlockDataTask::new_voxels_with_generation_at_revision(
        Vector3i::new(1, 2, 3),
        0,
        VoxelBuffer::with_size(Vector3i::splat(2)),
        StreamingDependency::new(stream),
        None,
        0,
        77,
    );

    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let terminal = task.take_terminal();
    COUNTING.store(false, Ordering::SeqCst);

    assert!(terminal.is_some());
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
}
