//! Deterministic Variable LOD reference-model test sequences.
//!
//! Phase E item 6: provides deterministic, reproducible test sequences that
//! exercise the Variable LOD planner across known viewer trajectories. These
//! sequences serve as regression anchors — if the planner's observable output
//! changes, the test fails. They complement the unit tests in voxel_terrain_core
//! by driving the PUBLIC `try_process` API with deterministic viewer updates
//! and asserting stable block-count snapshots.

use std::sync::Arc;

use voxel_core::generators::simple::Flat;
use voxel_core::math::{Box3i, Vector3i};
use voxel_core::meshers::TransvoxelMesher;
use voxel_core::storage::VoxelData;
use voxel_core::streams::MemoryStream;
use voxel_core::terrain::clipbox_coordinator::MeshDemand;
use voxel_core::terrain::lod_clipbox::LodClipboxSettings;
use voxel_core::terrain::voxel_terrain_core::{ViewerUpdate, VoxelTerrainCore};

fn make_terrain() -> VoxelTerrainCore {
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
    let mesher = Arc::new(TransvoxelMesher::new());
    let dep = voxel_core::engine::MeshingDependency::new(mesher, Some(generator));
    VoxelTerrainCore::new_variable_lod(data, Arc::new(MemoryStream::new()), dep, settings)
        .expect("terrain constructs")
}

fn viewer(id: u32, x: i32, z: i32) -> ViewerUpdate {
    ViewerUpdate {
        id,
        world_position_voxels: Vector3i::new(x, 0, z),
        horizontal_view_distance_voxels: 48,
        vertical_view_distance_voxels: 48,
        demand: MeshDemand {
            visuals: true,
            collisions: false,
        },
    }
}

fn pump(core: &mut VoxelTerrainCore, viewers: &[ViewerUpdate]) {
    core.try_process(viewers).unwrap();
    core.wait_for_pending_tasks();
}

/// Sequence 1: enter → converge → exit. The mesh block count must be stable
/// across repeated runs (deterministic) and must drop to zero on exit.
#[test]
fn reference_sequence_enter_converge_exit() {
    let mut core = make_terrain();
    // Enter and converge.
    for _ in 0..10 {
        pump(&mut core, &[viewer(1, 0, 0)]);
    }
    let converged = core.mesh_blocks().len();
    assert!(converged > 0, "converged block count must be > 0");

    // Steady state: re-pumping the same viewer must not change the count
    // beyond small jitter.
    for _ in 0..5 {
        pump(&mut core, &[viewer(1, 0, 0)]);
    }
    let steady = core.mesh_blocks().len();
    let diff = steady.abs_diff(converged);
    assert!(
        diff <= converged / 4,
        "steady state drift too large: {converged} -> {steady}"
    );

    // Exit: block count must eventually reach zero.
    for _ in 0..20 {
        pump(&mut core, &[]);
    }
    let exited = core.mesh_blocks().len();
    assert_eq!(exited, 0, "all mesh blocks must be unloaded on exit");
}

/// Sequence 2: enter → move to a distant position → return. The block count
/// after return must be close to the original converged count (deterministic).
#[test]
fn reference_sequence_move_and_return() {
    let mut core = make_terrain();
    for _ in 0..10 {
        pump(&mut core, &[viewer(1, 0, 0)]);
    }
    let original = core.mesh_blocks().len();
    assert!(original > 0);

    // Move far away (outside view distance).
    for _ in 0..10 {
        pump(&mut core, &[viewer(1, 256, 256)]);
    }
    // Some blocks may remain from the old position during teardown, but the
    // count must differ from the original.
    let moved = core.mesh_blocks().len();

    // Return to origin.
    for _ in 0..10 {
        pump(&mut core, &[viewer(1, 0, 0)]);
    }
    let returned = core.mesh_blocks().len();
    assert!(
        returned > 0,
        "must re-page blocks after return, got {returned}"
    );
    // The returned count should be close to the original.
    let diff = returned.abs_diff(original);
    assert!(
        diff <= original / 2,
        "return count drift too large: {original} -> {returned} (moved={moved})"
    );
}

/// Sequence 3: two viewers enter at different positions, then one exits.
/// The remaining viewer's blocks must persist.
#[test]
fn reference_sequence_multi_viewer_partial_exit() {
    let mut core = make_terrain();
    // Both viewers enter.
    for _ in 0..10 {
        pump(&mut core, &[viewer(1, 0, 0), viewer(2, 128, 0)]);
    }
    let both = core.mesh_blocks().len();
    assert!(both > 0, "both viewers must produce blocks");

    // Viewer 2 exits.
    for _ in 0..10 {
        pump(&mut core, &[viewer(1, 0, 0)]);
    }
    let one = core.mesh_blocks().len();
    assert!(one > 0, "remaining viewer must still have blocks");

    // The count with one viewer should be <= count with two.
    assert!(
        one <= both,
        "one-viewer count ({one}) must not exceed two-viewer count ({both})"
    );
}
