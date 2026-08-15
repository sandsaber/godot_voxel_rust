use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use voxel_core::engine::MeshingDependency;
use voxel_core::math::{Box3i, Vector3i};
use voxel_core::meshers::{
    MeshBlockKey, MeshBlockLocation, MeshBlockTask, MeshBlockTaskParams, MesherInput, MesherOutput,
    VoxelMesher,
};
use voxel_core::storage::{
    ChannelId, SharedVoxelData, VoxelBuffer, VoxelData, VoxelDataBlock, VoxelFormat,
};
use voxel_core::tasks::{ScheduledTask, TaskLane, ThreadedTaskRunner};

const MESH_TASKS: usize = 96;
const EDIT_OPS: usize = 128;
const LOAD_OPS: usize = 64;

struct StressMesher {
    builds: Arc<AtomicUsize>,
}

impl VoxelMesher for StressMesher {
    fn build(&self, _output: &mut MesherOutput, input: &MesherInput<'_>) {
        assert!(input.voxels.size().x > 0);
        self.builds.fetch_add(1, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(1));
    }

    fn minimum_padding(&self) -> u32 {
        1
    }

    fn maximum_padding(&self) -> u32 {
        1
    }

    fn used_channels_mask(&self) -> u32 {
        1 << ChannelId::Type.index()
    }
}

#[test]
fn threaded_edit_load_mesh_stress_keeps_data_and_region_locks_consistent() {
    let mut voxel_data = VoxelData::new();
    voxel_data.set_bounds(Box3i::new(Vector3i::splat(-512), Vector3i::splat(2048)));
    voxel_data.set_streaming_enabled(false);
    voxel_data.set_full_load_completed(true);

    let data = Arc::new(SharedVoxelData::new(voxel_data));
    let builds = Arc::new(AtomicUsize::new(0));
    let mesher = Arc::new(StressMesher {
        builds: builds.clone(),
    });
    let meshing_dependency = MeshingDependency::new(mesher, None);
    let mut runner = ThreadedTaskRunner::new(6);
    let start = Arc::new(Barrier::new(3));
    let edit_successes = Arc::new(AtomicUsize::new(0));
    let load_successes = Arc::new(AtomicUsize::new(0));

    thread::scope(|scope| {
        let edit_data = data.clone();
        let edit_start = start.clone();
        let edit_successes = edit_successes.clone();
        scope.spawn(move || {
            edit_start.wait();
            for i in 0..EDIT_OPS {
                let pos = Vector3i::new((i % 32) as i32 - 16, 1, (i / 32) as i32 - 2);
                if edit_data
                    .try_edit_voxel_checked(i as u64, pos, ChannelId::Type.index())
                    .unwrap_or(false)
                {
                    edit_successes.fetch_add(1, Ordering::SeqCst);
                }
                thread::yield_now();
            }
        });

        let load_data = data.clone();
        let load_start = start.clone();
        let load_successes = load_successes.clone();
        scope.spawn(move || {
            load_start.wait();
            let block_size = load_data.block_size() as i32;
            for i in 0..LOAD_OPS {
                let block_pos = Vector3i::new(64 + i as i32, 0, 8);
                let mut voxels = VoxelBuffer::with_size(Vector3i::splat(block_size));
                VoxelFormat::new().configure_buffer(&mut voxels);
                voxels.fill((i % 255) as u64, ChannelId::Type.index());

                if load_data
                    .try_set_block(block_pos, VoxelDataBlock::with_voxels(voxels, 0))
                    .unwrap()
                {
                    load_successes.fetch_add(1, Ordering::SeqCst);
                }
                thread::yield_now();
            }
        });

        start.wait();

        let tasks = (0..MESH_TASKS).map(|i| {
            let pos = Vector3i::new((i % 8) as i32 - 4, 0, (i / 8) as i32 - 6);
            ScheduledTask::new(
                Box::new(MeshBlockTask::new(MeshBlockTaskParams {
                    key: MeshBlockKey {
                        location: MeshBlockLocation::new(pos, 0),
                        revision: 0,
                    },
                    data: data.clone(),
                    meshing_dependency: meshing_dependency.clone(),
                    collision_hint: false,
                    lod_hint: false,
                    mesh_arrays_pool: None,
                })),
                TaskLane::Parallel,
            )
        });
        runner.enqueue_many(tasks);
        runner.wait_for_all_tasks();
    });

    let mut completed = std::collections::VecDeque::new();
    runner.try_drain_completed_into(&mut completed).unwrap();
    let mut mesh_outputs = 0;
    for task in &mut completed {
        let Some(task) = task.task_any_mut().downcast_mut::<MeshBlockTask>() else {
            continue;
        };
        let output = task.take_output().expect("mesh task should expose output");
        assert!(!output.dropped());
        mesh_outputs += 1;
    }

    assert_eq!(mesh_outputs, MESH_TASKS);
    assert_eq!(builds.load(Ordering::SeqCst), MESH_TASKS);
    assert_eq!(edit_successes.load(Ordering::SeqCst), EDIT_OPS);
    assert_eq!(load_successes.load(Ordering::SeqCst), LOAD_OPS);
    assert_eq!(data.locked_region_count(0), 0);
    assert!(data.block_count() >= LOAD_OPS);
}
