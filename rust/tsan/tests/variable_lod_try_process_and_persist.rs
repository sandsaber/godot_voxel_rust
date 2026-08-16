//! TSan: production Variable-LOD `try_process` + checkpoint/flush.
//!
//! The owner thread pumps the public terrain API. Concurrency comes from the
//! internal `ThreadedTaskRunner` created by `new_variable_lod` plus a reader
//! thread that only uses cloneable read-only handles.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use voxel_core::engine::MeshingDependency;
use voxel_core::generators::simple::Flat;
use voxel_core::math::{Box3i, Vector3i};
use voxel_core::meshers::TransvoxelMesher;
use voxel_core::storage::{ChannelId, VoxelBuffer, VoxelData};
use voxel_core::streams::{LoadResult, MemoryStream};
use voxel_core::terrain::lod_clipbox::LodClipboxSettings;
use voxel_core::terrain::{MeshDemand, ViewerUpdate, VoxelTerrainCore, VoxelTerrainEvent};

fn make_variable_terrain(stream: Arc<MemoryStream>) -> VoxelTerrainCore {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-256), Vector3i::splat(512)));
    let generator: Arc<Flat> = Arc::new(Flat::default());
    data.set_generator(Some(generator.clone()));
    let settings = LodClipboxSettings {
        data_block_size: 16,
        mesh_block_size: 16,
        lod_count: 3,
        lod0_distance_voxels: 16,
        secondary_distance_voxels: 16,
        unload_hysteresis_blocks: 2,
    };
    let dep = MeshingDependency::new(Arc::new(TransvoxelMesher::new()), Some(generator));
    VoxelTerrainCore::new_variable_lod(data, stream, dep, settings).expect("construct")
}

fn viewer(id: u32, position: Vector3i) -> ViewerUpdate {
    ViewerUpdate {
        id,
        world_position_voxels: position,
        horizontal_view_distance_voxels: 48,
        vertical_view_distance_voxels: 48,
        demand: MeshDemand {
            visuals: true,
            collisions: false,
        },
    }
}

fn pump_until_resident(core: &mut VoxelTerrainCore, viewers: &[ViewerUpdate]) {
    for _ in 0..16 {
        for _ in 0..16 {
            let _ = core.try_process(viewers).unwrap();
            if core.data().block_snapshot(Vector3i::zero(), 0).is_some() {
                return;
            }
        }
        core.wait_for_pending_tasks();
    }
    panic!("block never became resident");
}

#[test]
fn variable_lod_try_process_overlaps_workers_without_races() {
    let stream = Arc::new(MemoryStream::new());
    let mut core = make_variable_terrain(stream);
    let view = core.data();
    let snapshots = Arc::new(AtomicUsize::new(0));

    thread::scope(|scope| {
        let snapshots = snapshots.clone();
        scope.spawn(move || {
            for _ in 0..256 {
                let _ = view.block_snapshot(Vector3i::zero(), 0);
                let _ = view.lod_count();
                snapshots.fetch_add(1, Ordering::SeqCst);
                thread::yield_now();
            }
        });

        let origin = [viewer(1, Vector3i::zero())];
        pump_until_resident(&mut core, &origin);
        for _ in 0..8 {
            let _ = core.try_process(&origin).unwrap();
        }
        let moved = [viewer(1, Vector3i::new(96, 0, 0))];
        for _ in 0..8 {
            let _ = core.try_process(&moved).unwrap();
            core.wait_for_pending_tasks();
        }
        for _ in 0..16 {
            let events = core.try_process(&[]).unwrap();
            if events
                .iter()
                .any(|e| matches!(e, VoxelTerrainEvent::DataBlockUnloaded(_)))
            {
                break;
            }
            core.wait_for_pending_tasks();
        }
    });

    assert!(snapshots.load(Ordering::SeqCst) > 0);
}

#[test]
fn variable_lod_flush_checkpoint_under_worker_overlap() {
    let stream = Arc::new(MemoryStream::new());
    let core = Arc::new(Mutex::new(make_variable_terrain(stream.clone())));
    let channel = ChannelId::Type.index();
    let origin = [viewer(1, Vector3i::zero())];

    {
        let mut guard = core.lock().expect("core lock");
        pump_until_resident(&mut guard, &origin);
        let mut edited = false;
        for _ in 0..32 {
            if guard
                .try_edit_voxel(61, Vector3i::new(1, 1, 1), channel)
                .unwrap()
                .is_some()
            {
                edited = true;
                break;
            }
            let _ = guard.try_process(&origin).unwrap();
            guard.wait_for_pending_tasks();
        }
        assert!(edited, "resident LOD0 voxel must accept an edit");
    }

    thread::scope(|scope| {
        let core_reader = core.clone();
        let stream_reader = stream.clone();
        scope.spawn(move || {
            for _ in 0..64 {
                let _ = stream_reader.len();
                let mut buf = VoxelBuffer::with_size(Vector3i::splat(16));
                let _ = stream_reader.load_block(Vector3i::zero(), 0, &mut buf);
                let _ = core_reader.lock().expect("core lock").pending_task_count();
                thread::yield_now();
            }
        });

        let mut guard = core.lock().expect("core lock");
        guard.flush_pending_saves().unwrap();
        for _ in 0..8 {
            let _ = guard.try_process(&origin).unwrap();
        }
        assert!(guard.last_save_checkpoint_error().is_none());
        assert!(guard.last_save_failures().is_empty());
        guard.shutdown_and_flush().unwrap();
    });

    let mut loaded = VoxelBuffer::with_size(Vector3i::splat(16));
    assert_eq!(
        stream.load_block(Vector3i::zero(), 0, &mut loaded),
        LoadResult::Found
    );
    assert_eq!(loaded.get_voxel(1, 1, 1, channel), 61);
}
