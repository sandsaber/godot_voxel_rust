use std::sync::Arc;
use voxel_core::engine::MeshingDependency;
use voxel_core::generators::simple::Flat;
use voxel_core::math::{Box3i, Vector3i};
use voxel_core::meshers::TransvoxelMesher;
use voxel_core::storage::{ChannelId, VoxelBuffer, VoxelData};
use voxel_core::streams::{LoadResult, MemoryStream, VoxelLoadQuery, VoxelStream};
use voxel_core::terrain::lod_clipbox::LodClipboxSettings;
use voxel_core::terrain::{MeshDemand, ViewerUpdate, VoxelTerrainCore};

fn core() -> (VoxelTerrainCore, Arc<MemoryStream>) {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-128), Vector3i::splat(256)));
    data.set_generator(Some(Arc::new(Flat::default())));
    let stream = Arc::new(MemoryStream::new());
    let core = VoxelTerrainCore::new_variable_lod(
        data,
        stream.clone(),
        MeshingDependency::new(Arc::new(TransvoxelMesher::new()), None),
        LodClipboxSettings {
            data_block_size: 16,
            mesh_block_size: 16,
            lod_count: 3,
            lod0_distance_voxels: 16,
            secondary_distance_voxels: 16,
            unload_hysteresis_blocks: 0,
        },
    )
    .unwrap();
    (core, stream)
}

fn viewer() -> [ViewerUpdate; 1] {
    [ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: 64,
        vertical_view_distance_voxels: 64,
        demand: MeshDemand {
            visuals: true,
            collisions: false,
        },
    }]
}

fn settle(core: &mut VoxelTerrainCore, viewers: &[ViewerUpdate]) {
    // Waiting joins the worker batch; the following tick consumes its outputs
    // and may start the next stage. A final stationary tick proves convergence.
    for _ in 0..12 {
        core.try_process(viewers).unwrap();
        core.wait_for_pending_tasks();
    }
    assert_eq!(core.pending_task_count(), 0);
    for lod in 0..core.lod_count() {
        for entry in core.mesh_blocks_at_lod(lod).values() {
            if entry.needs_visual() || entry.needs_collision() {
                assert_eq!(entry.applied_revision, entry.requested_revision);
                assert!(!entry.is_in_update_list);
            }
        }
    }
}

#[test]
fn live_distance_changes_preserve_residency_edits_and_future_paging() {
    let (mut core, stream) = core();
    let viewers = viewer();
    settle(&mut core, &viewers);
    let original_count = core.data().block_positions(1).len();
    core.try_edit_voxel(77, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .unwrap();
    let mut settings = core.variable_lod_settings().unwrap();
    settings.secondary_distance_voxels = 32;
    core.try_reconfigure_variable_clipboxes(settings).unwrap();
    settle(&mut core, &viewers);
    assert!(core.data().block_positions(1).len() > original_count);
    assert!(core
        .debug_snapshot()
        .mesh_blocks
        .iter()
        .any(|block| block.visual_active));
    settings.secondary_distance_voxels = 16;
    core.try_reconfigure_variable_clipboxes(settings).unwrap();
    settle(&mut core, &viewers);
    assert_eq!(core.data().block_positions(1).len(), original_count);
    assert_eq!(
        core.data()
            .block_snapshot(Vector3i::zero(), 0)
            .unwrap()
            .voxels()
            .get_voxel(0, 0, 0, ChannelId::Type.index()),
        77
    );
    settle(&mut core, &[]);
    core.flush_pending_saves().unwrap();
    assert!(core.data().block_positions(0).is_empty());
    let mut saved = VoxelBuffer::with_size(Vector3i::splat(16));
    assert_eq!(
        stream
            .load_voxel_block(VoxelLoadQuery::new(&mut saved, Vector3i::zero(), 0))
            .unwrap(),
        LoadResult::Found
    );
    assert_eq!(saved.get_voxel(0, 0, 0, ChannelId::Type.index()), 77);
    settle(&mut core, &viewers);
    assert!(core
        .debug_snapshot()
        .mesh_blocks
        .iter()
        .any(|block| block.visual_active));
}

#[test]
fn disabled_initial_tick_can_resume_with_a_stationary_viewer() {
    let (mut core, _) = core();
    core.automatic_loading_enabled = false;
    core.try_process(&viewer()).unwrap();
    assert_eq!(core.pending_task_count(), 0);
    assert!(core.data().block_positions(0).is_empty());
    core.automatic_loading_enabled = true;
    settle(&mut core, &viewer());
    assert!(!core.data().block_positions(0).is_empty());
    assert!(core
        .debug_snapshot()
        .mesh_blocks
        .iter()
        .any(|block| block.visual_active));
}

#[test]
fn disabled_ticks_finish_loads_without_losing_queued_meshes_or_dirty_blocks() {
    let (mut core, stream) = core();
    let viewers = viewer();
    core.try_process(&viewers).unwrap();
    core.wait_for_pending_tasks();
    core.automatic_loading_enabled = false;
    core.try_process(&viewers).unwrap();
    assert!(!core.data().block_positions(0).is_empty());
    assert_eq!(core.pending_task_count(), 0);
    core.try_edit_voxel(91, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .unwrap();
    let resident_count = core.data().block_positions(0).len();
    core.try_process(&[]).unwrap();
    assert_eq!(core.data().block_positions(0).len(), resident_count);
    assert_eq!(core.pending_task_count(), 0);
    core.automatic_loading_enabled = true;
    settle(&mut core, &viewers);
    assert!(core
        .debug_snapshot()
        .mesh_blocks
        .iter()
        .any(|block| block.visual_active));
    settle(&mut core, &[]);
    core.flush_pending_saves().unwrap();
    let mut saved = VoxelBuffer::with_size(Vector3i::splat(16));
    assert_eq!(
        stream
            .load_voxel_block(VoxelLoadQuery::new(&mut saved, Vector3i::zero(), 0))
            .unwrap(),
        LoadResult::Found
    );
    assert_eq!(saved.get_voxel(0, 0, 0, ChannelId::Type.index()), 91);
}

#[test]
fn live_reconfiguration_rejects_structural_and_invalid_changes_without_mutation() {
    let (mut core, _) = core();
    settle(&mut core, &viewer());
    let original = core.variable_lod_settings().unwrap();
    for settings in [
        LodClipboxSettings {
            lod_count: 2,
            ..original
        },
        LodClipboxSettings {
            data_block_size: 32,
            mesh_block_size: 32,
            ..original
        },
        LodClipboxSettings {
            secondary_distance_voxels: -1,
            ..original
        },
    ] {
        assert!(core.try_reconfigure_variable_clipboxes(settings).is_err());
        assert_eq!(core.variable_lod_settings(), Some(original));
        settle(&mut core, &viewer());
    }
}
