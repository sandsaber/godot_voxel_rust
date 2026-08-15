use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use voxel_core::math::{Box3i, Vector3i};
use voxel_core::terrain::clipbox_coordinator::{
    ClipboxCoordinator, ClipboxViewerUpdate, MeshDemand,
};
use voxel_core::terrain::lod_clipbox::LodClipboxSettings;

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every allocation operation is forwarded to `System` with the exact
// pointer/layout contract received from the caller; the wrapper only observes
// successful returned pointers through non-panicking atomics.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding the caller-provided valid allocation layout.
        let allocation = unsafe { System.alloc(layout) };
        if COUNTING.load(Ordering::Relaxed) && !allocation.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        allocation
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer/layout pair came from this allocator's `System`
        // delegation and is forwarded unchanged.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding the caller-provided valid allocation layout.
        let allocation = unsafe { System.alloc_zeroed(layout) };
        if COUNTING.load(Ordering::Relaxed) && !allocation.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        allocation
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the original pointer/layout pair and requested size are
        // forwarded unchanged to the allocator that created the allocation.
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
fn exact_identical_coordinator_update_allocates_nothing() {
    let settings = LodClipboxSettings {
        data_block_size: 16,
        mesh_block_size: 16,
        lod_count: 3,
        lod0_distance_voxels: 48,
        secondary_distance_voxels: 48,
        unload_hysteresis_blocks: 2,
    };
    let bounds = Box3i::new(Vector3i::splat(-2048), Vector3i::splat(4096));
    let viewers = [
        ClipboxViewerUpdate {
            id: 2,
            position_voxels: Vector3i::new(-81, 25, 7),
            view_distance_voxels: Vector3i::splat(160),
            demand: MeshDemand {
                visuals: false,
                collisions: true,
            },
        },
        ClipboxViewerUpdate {
            id: 1,
            position_voxels: Vector3i::new(17, -9, 33),
            view_distance_voxels: Vector3i::splat(192),
            demand: MeshDemand {
                visuals: true,
                collisions: true,
            },
        },
    ];
    let mut coordinator = ClipboxCoordinator::new(settings, bounds).unwrap();
    coordinator.update_viewers(&viewers).unwrap();

    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let delta = coordinator.update_viewers(&viewers).unwrap();
    COUNTING.store(false, Ordering::SeqCst);

    assert!(delta.changes.is_empty());
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
}
