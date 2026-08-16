//! Terrain — paging orchestrators that drive [`crate::storage::VoxelData`] +
//! [`crate::meshers::MeshBlockTask`] based on viewer positions.
//!
//! Ports the engine-agnostic core of `terrain/fixed_lod/voxel_terrain.cpp`
//! (single-LOD paging) and (later) the multi-LOD equivalent under
//! `terrain/variable_lod/`. Godot `Node3D` / `RenderingServer` glue lives in
//! the `voxel-gdext` crate.

pub mod area_finder;
pub mod replication;

pub use area_finder::{box_subtraction, AreaError, AreaId, VoxelAreaFinder, MAX_CELLS_PER_AREA};
pub mod clipbox_coordinator;
#[allow(dead_code)] // Dormant Task 6 ledger; VoxelTerrainCore adoption is the next bounded slice.
pub(super) mod coverage_hold_ledger;
pub mod lod_clipbox;
pub mod lod_octree;
pub mod variable_lod_coverage;
pub mod voxel_terrain_core;

pub use clipbox_coordinator::MeshDemand;
pub use lod_octree::{LodOctree, OctreeNodeData, OctreeUpdateActions};
pub use variable_lod_coverage::{
    AcceptedFeatureSnapshot, CoverageFeature, CoverageHoldId, CoverageHoldIntentManifest,
    CoverageHoldOwnerDelta, CoverageInput, CoverageInputKind, CoverageInvariantError,
    CoverageReconcileResult, CoverageWorkCounters, FeatureReadiness, FeatureTopologyGroup,
    RenderTopologyBatch, TopologyOperation, TransitionFace, TransitionMask, VariableLodCoverage,
};
pub use voxel_terrain_core::{
    DataRefField, DebugEditedBlock, DebugMeshBlock, IndeterminateIoResolution,
    JournalPersistenceState, MeshBlockEntry, MeshLifecycleEventDescriptor, MeshLifecycleEventKind,
    MeshOutputApplyError, MeshRefField, PairedViewer, PersistenceOperation,
    PreparedPublicationConflict, SaveFlushError, TerrainDebugSnapshot, UnsavedBlockSave,
    UnsavedBlockSaveDetails, VariableLodConstructionError, VariableLodCoverageHoldError, ViewerId,
    ViewerInputError, ViewerState, ViewerUpdate, VoxelEditOutcome, VoxelTerrainCore,
    VoxelTerrainDataView, VoxelTerrainEvent, VoxelTerrainRuntimeError, VoxelTerrainStats,
};

// Task 4/6 integration seam. These zero-runtime function-pointer contracts
// keep the sibling-only transactional API checked before the terrain core
// starts calling it, without widening visibility or suppressing dead-code
// diagnostics across the implementation.
type CoveragePreviewResult = Result<variable_lod_coverage::CoveragePreview, CoverageInvariantError>;
const _: fn(&VariableLodCoverage, &[CoverageInput]) -> CoveragePreviewResult =
    VariableLodCoverage::preview_reconcile;
const _: fn(
    &mut VariableLodCoverage,
    variable_lod_coverage::CoveragePreview,
) -> Result<CoverageReconcileResult, CoverageInvariantError> = VariableLodCoverage::apply_preview;
const _: for<'a> fn(&'a variable_lod_coverage::CoveragePreview) -> &'a CoverageReconcileResult =
    variable_lod_coverage::CoveragePreview::result;
const _: fn(&variable_lod_coverage::CoveragePreview) -> u64 =
    variable_lod_coverage::CoveragePreview::base_revision;
const _: fn(&variable_lod_coverage::CoveragePreview) -> u64 =
    variable_lod_coverage::CoveragePreview::next_revision;
const _: fn(&variable_lod_coverage::CoveragePreview) -> CoverageWorkCounters =
    variable_lod_coverage::CoveragePreview::work_counters;
const _: fn(
    crate::meshers::MeshBlockLocation,
    u8,
) -> Result<Vec<crate::meshers::MeshBlockLocation>, CoverageInvariantError> =
    variable_lod_coverage::checked_children;

#[cfg(test)]
mod coverage_preview_seam_tests {
    use super::variable_lod_coverage::{CoverageWorkCounters, VariableLodCoverage};

    #[test]
    fn sibling_can_inspect_work_counters_before_consuming_preview() {
        let coverage = VariableLodCoverage::try_new(3).unwrap();
        let preview = coverage.preview_reconcile(&[]).unwrap();
        assert_eq!(preview.work_counters(), CoverageWorkCounters::default());
        assert_eq!(preview.base_revision(), preview.next_revision());
    }
}
