//! Engine-agnostic single-LOD terrain paging orchestrator.
//!
//! Ports the engine-agnostic core of `terrain/fixed_lod/voxel_terrain.cpp`.
//! Drives [`VoxelData`] + [`MeshBlockTask`] from a set of paired viewers:
//! each [`try_process`](VoxelTerrainCore::try_process) tick diffs viewer positions
//! against the previous frame, loads/unloads data blocks via
//! [`VoxelData::view_area`] / [`VoxelData::unview_area`], requests mesh
//! updates for blocks whose neighbour data is resident, and runs the
//! pending load/mesh tasks through a [`ThreadedTaskRunner`].
//!
//! ## What is intentionally NOT here
//! - Godot `Node3D` / `RenderingServer` / `World3D` integration — that lives
//!   in the `voxel-gdext` crate (Phase 5).
//! - Instancer, multiplayer, collisions-as-separate-flag, quick-reload,
//!   save-on-unload, GPU generation, detail textures — all deferred.
//! - Multi-LOD paging (VoxelLodTerrain) — a separate orchestrator later.
//!
//! The minimum supported configuration is `mesh_block_size ==
//! data_block_size` (factor 1). The factor abstraction is preserved in the
//! helpers so a future patch can extend it without rewriting the hot path.

use super::clipbox_coordinator::{
    ClipboxCoordinator, CoordinatorError, CoordinatorStateRetirement, MeshDemand,
    ResidentDemandDelta, ValidatedCoordinatorUpdate,
};
use super::coverage_hold_ledger::{
    CoverageDataRefcountUpdate, CoverageDataResource, CoverageHoldLedger, CoverageHoldLedgerError,
    CoverageMeshRefcountUpdate, PreparedCoverageHoldPhases,
};
use super::lod_clipbox::{
    bounds_in_lod_blocks, checked_box_intersection, clipped_meshing_data_box, LodClipboxSettings,
    LodMathError,
};
use super::variable_lod_coverage::{
    CoverageFeature, CoverageInvariantError, CoverageReconcileResult, CoverageStateRetirement,
    FeatureTopologyGroup, RenderTopologyBatch, TopologyOperation, ValidatedCoveragePreview,
    VariableLodCoverage,
};
#[allow(unused_imports)] // DemandCounts is used by variable-mode tests
use super::{
    clipbox_coordinator::{ClipboxViewerUpdate, DemandCountField, DemandCounts, ResidentBlockKind},
    coverage_hold_ledger::{
        CoverageHoldResourceSnapshot, CoverageMeshResource, PreparedCoverageHoldResolution,
    },
    variable_lod_coverage::{AcceptedFeatureSnapshot, CoverageInput},
};
use crate::constants::voxel_constants::MAX_LOD;
use crate::engine::{MeshingDependency, StreamingDependency};
use crate::math::{Box3i, Vector3i};
use crate::meshers::{
    BlockMeshOutput, MeshArraysPool, MeshBlockKey, MeshBlockLocation, MeshBlockTask,
    MeshBlockTaskOutput, MeshBlockTaskParams, MeshBuildFeatures, MeshUploadSnapshot, PayloadState,
};
use crate::storage::voxel_buffer::MAX_CHANNELS;
use crate::storage::voxel_data::SharedVoxelDataTransactionPreview;
use crate::storage::voxel_data::{
    CommittedSharedVoxelDataTransaction, PreparedSharedVoxelDataTransaction,
    RemovedSharedVoxelDataBlock, RevisionedBlockToSave, SharedVoxelDataSettingsSnapshot,
    SharedVoxelDataShutdownMutationPermit, SharedVoxelDataTransactionBlockSnapshot,
    SharedVoxelDataTransactionOperation,
};
use crate::storage::{
    BlockLocation, BlockToSave, SharedVoxelData, SharedVoxelDataMutationError, VoxelBuffer,
    VoxelData, VoxelDataBlock, VoxelDataKeyRevision, VoxelDataLodResizeError,
};
use crate::streams::flush_voxel_stream_task::FlushVoxelStreamTask;
use crate::streams::save_block_data_task::{
    PreparedSaveBlockDataPayloadInstaller, PreparedSaveBlockDataTask,
};
use crate::streams::{
    BlockDataOutput, BlockDataOutputKind, FlushTaskTerminal, MemoryStream,
    PersistenceAcknowledgement, PersistenceIoPhase, SaveBlockDataTask, SaveTaskTerminal,
    StreamResult, VoxelStream, VoxelStreamError,
};
use crate::tasks::threaded_task_runner::{FilledPreparedTaskBatch, PreparedTaskBatchError};
use crate::tasks::{
    threaded_task::PreparedCompletedTaskFollowUps, CompletedTask, RequestCancellation,
    ScheduledTask, TaskCompletionStatus, TaskLane, TaskRequestTag, ThreadedTask,
    ThreadedTaskRunner,
};
use std::collections::{BTreeMap, BTreeSet};
use std::collections::{HashMap, HashSet, TryReserveError, VecDeque};
use std::sync::Arc;
#[cfg(test)]
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Condvar, Mutex,
};

/// Lightweight viewer identity (mirrors C++ `ViewerID`).
pub type ViewerId = u32;

/// Per-viewer cached state used to diff boxes between frames.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ViewerState {
    pub local_position_voxels: Vector3i,
    /// LOD-0 data box (backward compat for single-LOD code).
    pub data_box: Box3i,
    /// LOD-0 mesh box (backward compat for single-LOD code).
    pub mesh_box: Box3i,
    /// Per-LOD data boxes (index 0 = LOD 0). Empty when single-LOD.
    pub data_box_per_lod: Vec<Box3i>,
    /// Per-LOD mesh boxes. Empty when single-LOD.
    pub mesh_box_per_lod: Vec<Box3i>,
    pub horizontal_view_distance_voxels: i32,
    pub vertical_view_distance_voxels: i32,
    pub demand: MeshDemand,
}

/// A viewer the terrain is currently tracking.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PairedViewer {
    pub id: ViewerId,
    pub state: ViewerState,
    pub prev_state: ViewerState,
}

/// Rendered mesh block entry. Mirrors the per-block state of C++
/// `VoxelMeshBlockVT` minus the Godot mesh/collision resources. Carries the
/// most recent immutable upload so downstream consumers can share its exact
/// accepted identity.
#[derive(Debug, Default)]
pub struct MeshBlockEntry {
    pub position: Vector3i,
    /// Number of fixed-LOD viewers retaining this mesh block.
    pub resident_viewers: u32,
    /// Number of fixed-LOD viewers requesting a visual payload.
    pub visual_viewers: u32,
    /// Number of fixed-LOD viewers requesting a collision payload.
    pub collision_viewers: u32,
    /// Coverage-owned visual residency, independent from viewer residency.
    pub visual_coverage_holds: u32,
    /// Coverage-owned collision residency, independent from viewer residency.
    pub collision_coverage_holds: u32,
    /// Whether the last accepted snapshot makes the visual feature active.
    pub visual_active: bool,
    /// Whether the last accepted snapshot makes the collision feature active.
    pub collision_active: bool,
    /// `true` once at least one mesh result has been applied.
    pub is_loaded: bool,
    /// Revision most recently requested for this block.
    pub requested_revision: Option<u64>,
    /// Generation of the most recently issued physical task. This identity is
    /// intentionally independent from `requested_revision`: retries rebuild
    /// the same content revision under a fresh physical generation.
    request_generation: u64,
    /// Complete feature union requested by the current revision.
    requested_features: MeshBuildFeatures,
    /// Complete feature union built by the most recently accepted revision.
    applied_features: MeshBuildFeatures,
    /// Revision most recently accepted by [`VoxelTerrainCore::try_apply_mesh_output`].
    pub applied_revision: Option<u64>,
    /// `true` when the most recently applied output contains render geometry.
    pub has_geometry: bool,
    /// `true` while the block is queued in `blocks_pending_update`.
    pub is_in_update_list: bool,
    /// Consecutive cancelled/panicked physical tasks for the requested
    /// revision. A fresh revision or accepted current output resets it.
    pub terminal_retry_count: u32,
    /// Current runner request owned by this resident entry. Every fixed- and
    /// variable-LOD worker request carries this exact epoch/generation tag.
    physical_request: Option<PhysicalRequest>,
    /// Most recent complete accepted upload. Replacing this snapshot replaces
    /// readiness for every feature, including `NotBuilt` tombstones.
    accepted_upload: Option<Arc<MeshUploadSnapshot>>,
}

impl MeshBlockEntry {
    pub fn accepted_upload(&self) -> Option<&Arc<MeshUploadSnapshot>> {
        self.accepted_upload.as_ref()
    }

    pub const fn mesh_viewers(&self) -> u32 {
        self.resident_viewers
    }

    /// Overflow-free aggregate residency diagnostic.
    pub const fn resident_refcount(&self) -> u64 {
        self.resident_viewers as u64
            + self.visual_coverage_holds as u64
            + self.collision_coverage_holds as u64
    }

    pub const fn needs_visual(&self) -> bool {
        self.visual_viewers > 0 || self.visual_coverage_holds > 0
    }

    pub const fn needs_collision(&self) -> bool {
        self.collision_viewers > 0 || self.collision_coverage_holds > 0
    }

    fn ref_field_mut(&mut self, field: MeshRefField) -> &mut u32 {
        match field {
            MeshRefField::ResidentViewers => &mut self.resident_viewers,
            MeshRefField::VisualViewers => &mut self.visual_viewers,
            MeshRefField::CollisionViewers => &mut self.collision_viewers,
            MeshRefField::VisualCoverageHolds => &mut self.visual_coverage_holds,
            MeshRefField::CollisionCoverageHolds => &mut self.collision_coverage_holds,
        }
    }

    fn checked_apply_ref_delta(
        &mut self,
        location: MeshBlockLocation,
        field: MeshRefField,
        delta: i64,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        let current = *self.ref_field_mut(field);
        let next = i128::from(current) + i128::from(delta);
        if next < 0 {
            return Err(VoxelTerrainRuntimeError::MeshRefcountUnderflow { location, field });
        }
        let next = u32::try_from(next)
            .map_err(|_| VoxelTerrainRuntimeError::MeshRefcountOverflow { location, field })?;
        *self.ref_field_mut(field) = next;
        Ok(())
    }

    fn matches_physical_request(&self, tag: Option<TaskRequestTag>) -> bool {
        match &self.physical_request {
            Some(request) => request.matches(tag),
            None => tag.is_none(),
        }
    }

    fn matches_previous_epoch_request(
        &self,
        tag: Option<TaskRequestTag>,
        current_epoch: u64,
    ) -> bool {
        self.physical_request.as_ref().is_some_and(|request| {
            request.matches(tag) && request.tag.request_epoch != current_epoch
        })
    }

    fn cancel_physical_request_if_superseded_by(&self, next: Option<&PhysicalRequest>) {
        if let Some(current) = &self.physical_request {
            if next.is_none_or(|next| {
                next.tag != current.tag || !Arc::ptr_eq(&next.cancellation, &current.cancellation)
            }) {
                current.cancel();
            }
        }
    }

    fn clone_for_draft(&self) -> Self {
        Self {
            position: self.position,
            resident_viewers: self.resident_viewers,
            visual_viewers: self.visual_viewers,
            collision_viewers: self.collision_viewers,
            visual_coverage_holds: self.visual_coverage_holds,
            collision_coverage_holds: self.collision_coverage_holds,
            visual_active: self.visual_active,
            collision_active: self.collision_active,
            is_loaded: self.is_loaded,
            requested_revision: self.requested_revision,
            request_generation: self.request_generation,
            requested_features: self.requested_features,
            applied_features: self.applied_features,
            applied_revision: self.applied_revision,
            has_geometry: self.has_geometry,
            is_in_update_list: self.is_in_update_list,
            terminal_retry_count: self.terminal_retry_count,
            physical_request: self.physical_request.clone(),
            accepted_upload: self.accepted_upload.clone(),
        }
    }
}

/// Optional notifier for terrain lifecycle events. Mirrors the C++ signals
/// `block_entered` / `block_exited` / `data_block_loaded` (the Rust core
/// surfaces them as a single sink so the Godot binding can route them).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VoxelTerrainStats {
    pub blocks_loaded: u64,
    pub blocks_unloaded: u64,
    pub meshes_built: u64,
    pub meshes_dropped: u64,
}

/// Lifecycle events emitted by [`VoxelTerrainCore::try_process`]. A Godot binding
/// can drain these to fire signals; tests inspect them to verify paging.
#[derive(Debug, Clone)]
pub enum VoxelTerrainEvent {
    /// A data block finished loading (and was inserted into `VoxelData`).
    DataBlockLoaded(BlockLocation),
    /// A data block was unloaded (viewers dropped to zero / out of range).
    DataBlockUnloaded(BlockLocation),
    /// A mesh block produced geometry after being empty.
    MeshBlockEntered(Arc<MeshUploadSnapshot>),
    /// A mesh block replaced existing geometry with new geometry.
    MeshBlockUpdated(Arc<MeshUploadSnapshot>),
    /// A mesh block became empty.
    MeshBlockBecameEmpty(Arc<MeshUploadSnapshot>),
    /// A mesh block was unloaded (no more viewers).
    MeshBlockExited(MeshBlockLocation),
    /// One atomic renderer topology/activity transition.
    RenderTopologyChanged(RenderTopologyBatch),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshLifecycleEventKind {
    Entered,
    Updated,
    BecameEmpty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshLifecycleEventDescriptor {
    pub kind: MeshLifecycleEventKind,
    pub key: MeshBlockKey,
    pub features: MeshBuildFeatures,
    pub visual_state: PayloadState,
    pub collision_state: PayloadState,
}

impl VoxelTerrainEvent {
    pub fn mesh_descriptor(&self) -> Option<MeshLifecycleEventDescriptor> {
        let (kind, upload) = match self {
            Self::MeshBlockEntered(upload) => (MeshLifecycleEventKind::Entered, upload),
            Self::MeshBlockUpdated(upload) => (MeshLifecycleEventKind::Updated, upload),
            Self::MeshBlockBecameEmpty(upload) => (MeshLifecycleEventKind::BecameEmpty, upload),
            Self::DataBlockLoaded(_)
            | Self::DataBlockUnloaded(_)
            | Self::MeshBlockExited(_)
            | Self::RenderTopologyChanged(_) => return None,
        };
        Some(MeshLifecycleEventDescriptor {
            kind,
            key: upload.key(),
            features: upload.features(),
            visual_state: upload.visual_state(),
            collision_state: upload.collision_state(),
        })
    }
}

pub enum MeshOutputApplyError {
    NotAdmitted {
        error: VoxelTerrainRuntimeError,
        output: BlockMeshOutput,
    },
    Admitted {
        error: VoxelTerrainRuntimeError,
    },
}

impl std::fmt::Debug for MeshOutputApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAdmitted { error, output } => f
                .debug_struct("NotAdmitted")
                .field("error", error)
                .field("output", output)
                .finish(),
            Self::Admitted { error } => f.debug_struct("Admitted").field("error", error).finish(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaveFlushError {
    Stream(VoxelStreamError),
    DataMutation(SharedVoxelDataMutationError),
    CompletionDrain { error: VoxelTerrainRuntimeError },
    IndeterminatePersistence { operation: PersistenceOperation },
    SaveAdmission { error: VoxelTerrainRuntimeError },
    UnsavedBlocks { count: usize },
}

/// One voxel block that could not be persisted within the bounded flush
/// attempt budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsavedBlockSave {
    pub position_in_blocks: Vector3i,
    pub lod_index: u8,
    /// Number of failed save attempts recorded for the current generation.
    pub retry_count: u32,
    /// Most recent stream error for the current generation, when the stream
    /// task reached the persistence call.
    pub error: Option<VoxelStreamError>,
}

/// Exact save-journal identity and diagnostics for one voxel block that could
/// not be persisted within the bounded flush attempt budget.
///
/// This extends [`UnsavedBlockSave`] without changing that compatibility
/// structure's public literal shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsavedBlockSaveDetails {
    pub position_in_blocks: Vector3i,
    pub lod_index: u8,
    /// Storage revision of the exact voxel snapshot that failed to persist.
    pub block_revision: u64,
    /// Monotonic save-journal generation paired with `block_revision`.
    pub save_generation: u64,
    /// Number of failed save attempts recorded for the current generation.
    pub retry_count: u32,
    /// Most recent stream error for the current generation, when the stream
    /// task reached the persistence call.
    pub error: Option<VoxelStreamError>,
}

impl std::fmt::Display for SaveFlushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stream(e) => write!(f, "terrain stream flush failed: {e}"),
            Self::DataMutation(e) => {
                write!(f, "terrain data mutation failed during flush: {e:?}")
            }
            Self::CompletionDrain { error } => {
                write!(f, "terrain completion drain failed during flush: {error}")
            }
            Self::IndeterminatePersistence { operation } => {
                write!(f, "terrain persistence is indeterminate: {operation:?}")
            }
            Self::SaveAdmission { error } => {
                write!(f, "terrain save admission failed: {error}")
            }
            Self::UnsavedBlocks { count } => {
                write!(f, "{count} terrain block saves remain unsaved")
            }
        }
    }
}

impl std::error::Error for SaveFlushError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalPersistenceState {
    PendingWrite,
    WriteInFlight,
    Indeterminate,
    WrittenUnflushed,
    Durable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewerInputError {
    DuplicateId(ViewerId),
    NegativeHorizontalDistance { id: ViewerId, value: i32 },
    NegativeVerticalDistance { id: ViewerId, value: i32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndeterminateIoResolution {
    AssumeNotWrittenAndRetry,
    AssumeWrittenAndFlush,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistenceOperation {
    Save {
        location: BlockLocation,
        block_revision: u64,
        save_generation: u64,
    },
    Flush {
        checkpoint_generation: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshRefField {
    ResidentViewers,
    VisualViewers,
    CollisionViewers,
    VisualCoverageHolds,
    CollisionCoverageHolds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataRefField {
    ResidentViewers,
    CoverageHolds,
    Total,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariableLodConstructionError {
    LodMath(LodMathError),
    LodResize(VoxelDataLodResizeError),
    Coordinator(CoordinatorError),
    Coverage(CoverageInvariantError),
    UnsupportedLodMesher,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariableLodCoverageHoldError {
    OldManifestMismatch {
        id: super::CoverageHoldId,
    },
    NonCanonicalManifest {
        id: super::CoverageHoldId,
    },
    TargetMismatch {
        id: super::CoverageHoldId,
    },
    InvalidPhaseRelation {
        id: Option<super::CoverageHoldId>,
    },
    MeshRefcountOverflow {
        location: MeshBlockLocation,
        feature: CoverageFeature,
    },
    MeshRefcountUnderflow {
        location: MeshBlockLocation,
        feature: CoverageFeature,
    },
    DataRefcountOverflow {
        location: BlockLocation,
    },
    DataRefcountUnderflow {
        location: BlockLocation,
    },
    CoordinateOverflow {
        location: MeshBlockLocation,
    },
}

/// Exact effects published by one checked terrain edit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxelEditOutcome {
    pub edited_block: BlockLocation,
    pub block_revision: u64,
    pub affected_mesh_blocks: Vec<MeshBlockLocation>,
}

/// Release-mode validation failures for the common prepared runtime publisher.
/// Locations are exact so callers can distinguish a stale draft from an
/// internally duplicated publication target without relying on debug asserts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedPublicationConflict {
    DuplicateMeshKey { location: MeshBlockLocation },
    DuplicateLoadingKey { location: BlockLocation },
    DuplicateDataResidencyKey { location: BlockLocation },
    DuplicatePendingLoadQueueLod { lod_index: u8 },
    DuplicatePendingMeshQueueLod { lod_index: u8 },
    MeshStateMismatch { location: MeshBlockLocation },
    LoadingStateMismatch { location: BlockLocation },
    DataResidencyStateMismatch { location: BlockLocation },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoxelTerrainRuntimeError {
    ViewerInput(ViewerInputError),
    LodMath(LodMathError),
    Coordinator(CoordinatorError),
    Coverage(CoverageInvariantError),
    CoverageHold(VariableLodCoverageHoldError),
    MeshOutputAdmissionFailed,
    MeshOutputApplyFailed,
    CompletionDrainCapacityFailed,
    CompletionNormalizationFailed,
    CompletionFollowUpReservationFailed,
    CompletionDrainStalled,
    /// A typed pre-C1 Variable LOD physical-prepare invariant violation
    /// (ownership mismatch, deferred state, etc.). Surfaced when the
    /// production Variable LOD planner rejects an inconsistent publication.
    VariablePhysicalPrepare(VariablePhysicalPrepareError),
    RequestEpochOverflow,
    ShutdownEpochOverflow,
    RequestGenerationOverflow,
    ShutdownRetryPending,
    LoadRetryCountOverflow {
        location: BlockLocation,
    },
    MeshRevisionOverflow,
    MeshTerminalRetryCountOverflow {
        key: MeshBlockKey,
    },
    RenderTopologyRevisionOverflow,
    MeshRefcountOverflow {
        location: MeshBlockLocation,
        field: MeshRefField,
    },
    MeshRefcountUnderflow {
        location: MeshBlockLocation,
        field: MeshRefField,
    },
    DataRefcountOverflow {
        location: BlockLocation,
        field: DataRefField,
    },
    DataRefcountUnderflow {
        location: BlockLocation,
        field: DataRefField,
    },
    DataResidencyMismatch {
        location: BlockLocation,
        tracked_resident_viewers: Option<u32>,
        tracked_coverage_holds: Option<u32>,
        storage_viewers: u32,
    },
    InvalidVoxelChannel {
        channel_index: usize,
    },
    CoordinateOverflow,
    BlockRevisionOverflow {
        location: BlockLocation,
    },
    LodRevisionOverflow {
        lod_index: u8,
    },
    SettingsRevisionOverflow,
    ConcurrentSettingsMutation {
        expected_revision: u64,
        actual_revision: u64,
    },
    ConcurrentLodMutation {
        lod_index: u8,
        expected_revision: u64,
        actual_revision: u64,
    },
    ConcurrentDataMutation {
        location: BlockLocation,
        expected_revision: VoxelDataKeyRevision,
        actual_revision: VoxelDataKeyRevision,
    },
    PreparedPublicationConflict(PreparedPublicationConflict),
    CapacityReservationFailed,
    DataMutation(SharedVoxelDataMutationError),
    SaveGenerationOverflow,
    MissingSavePayload,
    PersistenceAttemptOverflow {
        operation: PersistenceOperation,
    },
    PersistenceRetryCountOverflow {
        operation: PersistenceOperation,
    },
    TaskCountOverflow,
    StatsOverflow,
    PersistenceRetryLimitExceeded {
        operation: PersistenceOperation,
    },
    PersistenceIndeterminate {
        operation: PersistenceOperation,
    },
    IndeterminatePersistenceMismatch {
        requested: PersistenceOperation,
    },
}

impl std::fmt::Display for VoxelTerrainRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ViewerInput(error) => write!(f, "invalid terrain viewer input: {error:?}"),
            Self::LodMath(error) => write!(f, "invalid terrain LOD math: {error:?}"),
            Self::Coordinator(error) => {
                write!(f, "variable-LOD coordinator update failed: {error:?}")
            }
            Self::Coverage(error) => {
                write!(f, "variable-LOD coverage update failed: {error:?}")
            }
            Self::CoverageHold(error) => {
                write!(f, "variable-LOD coverage hold update failed: {error:?}")
            }
            Self::VariablePhysicalPrepare(error) => {
                write!(f, "variable-LOD physical prepare invariant violated: {error:?}")
            }
            Self::MeshOutputAdmissionFailed => write!(f, "terrain mesh output admission failed"),
            Self::MeshOutputApplyFailed => write!(f, "terrain mesh output apply failed"),
            Self::CompletionDrainCapacityFailed => {
                write!(f, "terrain completion drain capacity reservation failed")
            }
            Self::CompletionNormalizationFailed => {
                write!(f, "terrain completion normalization failed")
            }
            Self::CompletionFollowUpReservationFailed => {
                write!(f, "terrain completion follow-up reservation failed")
            }
            Self::CompletionDrainStalled => {
                write!(f, "terrain completion drain made no ownership progress")
            }
            Self::RequestEpochOverflow => write!(f, "terrain request epoch exhausted"),
            Self::ShutdownEpochOverflow => write!(f, "terrain shutdown epoch exhausted"),
            Self::RequestGenerationOverflow => {
                write!(f, "terrain physical request generation exhausted")
            }
            Self::ShutdownRetryPending => {
                write!(f, "terrain shutdown persistence retry is pending")
            }
            Self::LoadRetryCountOverflow { location } => {
                write!(f, "terrain load retry count exhausted at {location:?}")
            }
            Self::MeshRevisionOverflow => write!(f, "terrain mesh revision exhausted"),
            Self::MeshTerminalRetryCountOverflow { key } => {
                write!(f, "terrain mesh terminal retry count exhausted for {key:?}")
            }
            Self::RenderTopologyRevisionOverflow => {
                write!(f, "terrain render topology revision exhausted")
            }
            Self::MeshRefcountOverflow { location, field } => {
                write!(f, "terrain mesh {field:?} refcount overflow at {location:?}")
            }
            Self::MeshRefcountUnderflow { location, field } => {
                write!(f, "terrain mesh {field:?} refcount underflow at {location:?}")
            }
            Self::DataRefcountOverflow { location, field } => {
                write!(f, "terrain data {field:?} refcount overflow at {location:?}")
            }
            Self::DataRefcountUnderflow { location, field } => {
                write!(f, "terrain data {field:?} refcount underflow at {location:?}")
            }
            Self::DataResidencyMismatch {
                location,
                tracked_resident_viewers,
                tracked_coverage_holds,
                storage_viewers,
            } => write!(
                f,
                "terrain data residency mismatch at {location:?}: tracked viewers {tracked_resident_viewers:?}, tracked coverage holds {tracked_coverage_holds:?}, storage viewers {storage_viewers}"
            ),
            Self::InvalidVoxelChannel { channel_index } => {
                write!(f, "invalid terrain voxel channel index {channel_index}")
            }
            Self::CoordinateOverflow => write!(f, "terrain edit coordinate overflow"),
            Self::BlockRevisionOverflow { location } => {
                write!(f, "terrain data block revision exhausted at {location:?}")
            }
            Self::LodRevisionOverflow { lod_index } => {
                write!(f, "terrain LOD revision exhausted at LOD {lod_index}")
            }
            Self::SettingsRevisionOverflow => {
                write!(f, "terrain shared-data settings revision exhausted")
            }
            Self::ConcurrentSettingsMutation {
                expected_revision,
                actual_revision,
            } => write!(
                f,
                "terrain settings changed concurrently: expected {expected_revision}, got {actual_revision}"
            ),
            Self::ConcurrentLodMutation {
                lod_index,
                expected_revision,
                actual_revision,
            } => write!(
                f,
                "terrain LOD {lod_index} changed concurrently: expected {expected_revision}, got {actual_revision}"
            ),
            Self::ConcurrentDataMutation {
                location,
                expected_revision,
                actual_revision,
            } => write!(
                f,
                "terrain data changed concurrently at {location:?}: expected {expected_revision:?}, got {actual_revision:?}"
            ),
            Self::PreparedPublicationConflict(conflict) => {
                write!(f, "terrain prepared publication conflict: {conflict:?}")
            }
            Self::CapacityReservationFailed => {
                write!(f, "terrain edit capacity reservation failed")
            }
            Self::DataMutation(error) => {
                write!(f, "terrain shared-data transaction failed: {error:?}")
            }
            Self::SaveGenerationOverflow => write!(f, "terrain save generation exhausted"),
            Self::MissingSavePayload => write!(f, "terrain save payload is absent"),
            Self::TaskCountOverflow => write!(f, "terrain prepared task count overflow"),
            Self::StatsOverflow => write!(f, "terrain statistics counter overflow"),
            Self::PersistenceAttemptOverflow { operation } => {
                write!(f, "terrain persistence attempt exhausted for {operation:?}")
            }
            Self::PersistenceRetryCountOverflow { operation } => {
                write!(
                    f,
                    "terrain persistence retry count exhausted for {operation:?}"
                )
            }
            Self::PersistenceRetryLimitExceeded { operation } => {
                write!(
                    f,
                    "terrain persistence retry limit exceeded for {operation:?}"
                )
            }
            Self::PersistenceIndeterminate { operation } => {
                write!(
                    f,
                    "terrain persistence result is indeterminate for {operation:?}"
                )
            }
            Self::IndeterminatePersistenceMismatch { requested } => {
                write!(
                    f,
                    "indeterminate persistence operation did not match {requested:?}"
                )
            }
        }
    }
}

impl std::error::Error for VoxelTerrainRuntimeError {}

impl From<CoordinatorError> for VoxelTerrainRuntimeError {
    fn from(error: CoordinatorError) -> Self {
        Self::Coordinator(error)
    }
}

impl From<CoverageInvariantError> for VoxelTerrainRuntimeError {
    fn from(error: CoverageInvariantError) -> Self {
        Self::Coverage(error)
    }
}

impl From<CoverageHoldLedgerError> for VoxelTerrainRuntimeError {
    fn from(error: CoverageHoldLedgerError) -> Self {
        let error = match error {
            CoverageHoldLedgerError::OldManifestMismatch { id } => {
                VariableLodCoverageHoldError::OldManifestMismatch { id }
            }
            CoverageHoldLedgerError::NonCanonicalManifest { id } => {
                VariableLodCoverageHoldError::NonCanonicalManifest { id }
            }
            CoverageHoldLedgerError::TargetMismatch { id } => {
                VariableLodCoverageHoldError::TargetMismatch { id }
            }
            CoverageHoldLedgerError::InvalidPhaseRelation { id } => {
                VariableLodCoverageHoldError::InvalidPhaseRelation { id }
            }
            CoverageHoldLedgerError::MeshRefcountOverflow { resource } => {
                VariableLodCoverageHoldError::MeshRefcountOverflow {
                    location: resource.location(),
                    feature: resource.feature(),
                }
            }
            CoverageHoldLedgerError::MeshRefcountUnderflow { resource } => {
                VariableLodCoverageHoldError::MeshRefcountUnderflow {
                    location: resource.location(),
                    feature: resource.feature(),
                }
            }
            CoverageHoldLedgerError::DataRefcountOverflow { resource } => {
                VariableLodCoverageHoldError::DataRefcountOverflow {
                    location: coverage_data_location(resource),
                }
            }
            CoverageHoldLedgerError::DataRefcountUnderflow { resource } => {
                VariableLodCoverageHoldError::DataRefcountUnderflow {
                    location: coverage_data_location(resource),
                }
            }
            CoverageHoldLedgerError::CoordinateOverflow { location } => {
                VariableLodCoverageHoldError::CoordinateOverflow { location }
            }
        };
        Self::CoverageHold(error)
    }
}

const fn coverage_data_location(resource: CoverageDataResource) -> BlockLocation {
    BlockLocation {
        position: resource.position_in_blocks(),
        lod_index: resource.lod_index(),
    }
}

impl From<LodMathError> for VoxelTerrainRuntimeError {
    fn from(error: LodMathError) -> Self {
        Self::LodMath(error)
    }
}

fn map_fixed_durability_error(error: VoxelTerrainRuntimeError) -> SaveFlushError {
    match error {
        VoxelTerrainRuntimeError::DataMutation(error) => SaveFlushError::DataMutation(error),
        error @ (VoxelTerrainRuntimeError::SaveGenerationOverflow
        | VoxelTerrainRuntimeError::RequestEpochOverflow
        | VoxelTerrainRuntimeError::ShutdownEpochOverflow
        | VoxelTerrainRuntimeError::RequestGenerationOverflow
        | VoxelTerrainRuntimeError::ShutdownRetryPending
        | VoxelTerrainRuntimeError::MissingSavePayload
        | VoxelTerrainRuntimeError::PersistenceAttemptOverflow { .. }
        | VoxelTerrainRuntimeError::PersistenceRetryCountOverflow { .. }
        | VoxelTerrainRuntimeError::PersistenceRetryLimitExceeded { .. }
        | VoxelTerrainRuntimeError::PersistenceIndeterminate { .. }
        | VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch { .. }) => {
            SaveFlushError::SaveAdmission { error }
        }
        error => SaveFlushError::CompletionDrain { error },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SaveKey {
    position: Vector3i,
    lod_index: u8,
}

impl SaveKey {
    fn new(position: Vector3i, lod_index: u8) -> Self {
        Self {
            position,
            lod_index,
        }
    }
}

#[derive(Debug)]
struct SaveAttemptMeta {
    block_revision: u64,
    generation: u64,
    retry_count: u32,
    last_error: Option<VoxelStreamError>,
}

#[derive(Debug)]
struct PendingSave {
    meta: SaveAttemptMeta,
    payload: VoxelBuffer,
}

struct RetainedSaveAdmissionFailure {
    error: VoxelTerrainRuntimeError,
    // C2c will publish this retained ownership through the runtime event
    // boundary. C2b must keep it even though production cannot consume it yet.
    #[allow(dead_code)]
    save: BlockToSave,
}

// Boxing `PendingSave` would add an allocation and change the exact payload
// owner identity used by the fixed-LOD transaction and its rollback tests.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum ActiveSaveAttempt {
    Pending(PendingSave),
    WriteInFlight {
        meta: SaveAttemptMeta,
        attempt_ordinal: u64,
    },
    Indeterminate {
        meta: SaveAttemptMeta,
        attempt_ordinal: u64,
    },
}

#[derive(Debug)]
struct WrittenSave {
    block_revision: u64,
    generation: u64,
    payload: VoxelBuffer,
}

#[derive(Debug)]
struct SaveJournalEntry {
    written_unflushed: Option<WrittenSave>,
    active: Option<ActiveSaveAttempt>,
    queued_newer: VecDeque<PendingSave>,
}

#[derive(Debug, Clone, Copy)]
struct SaveCheckpointSnapshot {
    key: SaveKey,
    block_revision: u64,
    generation: u64,
}

#[derive(Debug)]
struct SaveCheckpointInFlight {
    checkpoint_generation: u64,
    acknowledged: Vec<SaveCheckpointSnapshot>,
    state: CheckpointAttemptState,
    retry_count: u32,
    max_attempts: u32,
    origin: CheckpointOrigin,
    record_per_block_failure: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckpointAttemptState {
    Pending,
    WriteInFlight { attempt_ordinal: u64 },
    Indeterminate { attempt_ordinal: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckpointOrigin {
    Automatic,
    Explicit,
}

impl SaveJournalEntry {
    #[allow(dead_code)] // retained as a unit-test harness after the planner cutover
    fn new_pending(block_revision: u64, generation: u64, payload: VoxelBuffer) -> Self {
        Self {
            written_unflushed: None,
            active: Some(ActiveSaveAttempt::Pending(PendingSave {
                meta: SaveAttemptMeta {
                    block_revision,
                    generation,
                    retry_count: 0,
                    last_error: None,
                },
                payload,
            })),
            queued_newer: VecDeque::new(),
        }
    }

    fn has_failed_attempt(&self) -> bool {
        matches!(
            &self.active,
            Some(ActiveSaveAttempt::Pending(pending))
                if pending.meta.retry_count > 0
        )
    }

    fn acknowledged_payload_bytes(&self) -> usize {
        self.written_unflushed.as_ref().map_or(0, |written| {
            (0..MAX_CHANNELS)
                .map(|channel| written.payload.channel_bytes(channel).len())
                .sum()
        })
    }

    fn is_empty(&self) -> bool {
        self.written_unflushed.is_none() && self.active.is_none() && self.queued_newer.is_empty()
    }

    fn promote_queued_if_idle(&mut self) {
        if self.written_unflushed.is_none() && self.active.is_none() {
            self.active = self
                .queued_newer
                .pop_front()
                .map(ActiveSaveAttempt::Pending);
        }
    }

    fn active_meta(&self) -> Option<&SaveAttemptMeta> {
        match self.active.as_ref()? {
            ActiveSaveAttempt::Pending(pending) => Some(&pending.meta),
            ActiveSaveAttempt::WriteInFlight { meta, .. }
            | ActiveSaveAttempt::Indeterminate { meta, .. } => Some(meta),
        }
    }

    #[cfg(test)]
    fn state_for_generation(&self, generation: u64) -> Option<JournalPersistenceState> {
        if self
            .written_unflushed
            .as_ref()
            .is_some_and(|written| written.generation == generation)
        {
            return Some(JournalPersistenceState::WrittenUnflushed);
        }
        match &self.active {
            Some(ActiveSaveAttempt::Pending(pending)) if pending.meta.generation == generation => {
                return Some(JournalPersistenceState::PendingWrite);
            }
            Some(ActiveSaveAttempt::WriteInFlight { meta, .. })
                if meta.generation == generation =>
            {
                return Some(JournalPersistenceState::WriteInFlight);
            }
            Some(ActiveSaveAttempt::Indeterminate { meta, .. })
                if meta.generation == generation =>
            {
                return Some(JournalPersistenceState::Indeterminate);
            }
            _ => {}
        }
        self.queued_newer
            .iter()
            .any(|queued| queued.meta.generation == generation)
            .then_some(JournalPersistenceState::PendingWrite)
    }

    #[cfg(test)]
    fn state_for_identity(
        &self,
        block_revision: u64,
        generation: u64,
    ) -> Option<JournalPersistenceState> {
        if self.written_unflushed.as_ref().is_some_and(|written| {
            written.block_revision == block_revision && written.generation == generation
        }) {
            return Some(JournalPersistenceState::WrittenUnflushed);
        }
        match &self.active {
            Some(ActiveSaveAttempt::Pending(pending))
                if pending.meta.block_revision == block_revision
                    && pending.meta.generation == generation =>
            {
                return Some(JournalPersistenceState::PendingWrite);
            }
            Some(ActiveSaveAttempt::WriteInFlight { meta, .. })
                if meta.block_revision == block_revision && meta.generation == generation =>
            {
                return Some(JournalPersistenceState::WriteInFlight);
            }
            Some(ActiveSaveAttempt::Indeterminate { meta, .. })
                if meta.block_revision == block_revision && meta.generation == generation =>
            {
                return Some(JournalPersistenceState::Indeterminate);
            }
            _ => {}
        }
        self.queued_newer
            .iter()
            .any(|queued| {
                queued.meta.block_revision == block_revision && queued.meta.generation == generation
            })
            .then_some(JournalPersistenceState::PendingWrite)
    }

    #[cfg(test)]
    fn write_in_flight_for_test(generation: u64, attempt_ordinal: u64) -> Self {
        Self::write_in_flight_for_test_at_revision(0, generation, attempt_ordinal)
    }

    #[cfg(test)]
    fn write_in_flight_for_test_at_revision(
        block_revision: u64,
        generation: u64,
        attempt_ordinal: u64,
    ) -> Self {
        Self {
            written_unflushed: None,
            active: Some(ActiveSaveAttempt::WriteInFlight {
                meta: SaveAttemptMeta {
                    block_revision,
                    generation,
                    retry_count: 0,
                    last_error: None,
                },
                attempt_ordinal,
            }),
            queued_newer: VecDeque::new(),
        }
    }
}

impl PreparedJournalShadow {
    fn from_entry(entry: &SaveJournalEntry) -> Self {
        let active = match entry.active.as_ref() {
            None => PreparedJournalActiveShadow::None,
            Some(ActiveSaveAttempt::Pending(pending)) => PreparedJournalActiveShadow::Pending {
                block_revision: pending.meta.block_revision,
                generation: pending.meta.generation,
                retry_count: pending.meta.retry_count,
            },
            Some(ActiveSaveAttempt::WriteInFlight {
                meta,
                attempt_ordinal,
            }) => PreparedJournalActiveShadow::WriteInFlight {
                block_revision: meta.block_revision,
                generation: meta.generation,
                attempt_ordinal: *attempt_ordinal,
                retry_count: meta.retry_count,
            },
            Some(ActiveSaveAttempt::Indeterminate {
                meta,
                attempt_ordinal,
            }) => PreparedJournalActiveShadow::Indeterminate {
                block_revision: meta.block_revision,
                generation: meta.generation,
                attempt_ordinal: *attempt_ordinal,
            },
        };
        Self {
            written_block_revision: entry
                .written_unflushed
                .as_ref()
                .map(|written| written.block_revision),
            written_generation: entry
                .written_unflushed
                .as_ref()
                .map(|written| written.generation),
            active,
            queued_len: entry.queued_newer.len(),
            queued_front: entry.queued_newer.front().map(|pending| {
                (
                    pending.meta.block_revision,
                    pending.meta.generation,
                    pending.meta.retry_count,
                )
            }),
        }
    }
}

impl PreparedCheckpointShadow {
    fn from_checkpoint(checkpoint: &SaveCheckpointInFlight) -> Self {
        Self {
            checkpoint_generation: checkpoint.checkpoint_generation,
            state: checkpoint.state,
            retry_count: checkpoint.retry_count,
            max_attempts: checkpoint.max_attempts,
            origin: checkpoint.origin,
            record_per_block_failure: checkpoint.record_per_block_failure,
        }
    }
}

fn try_clone_stream_error(
    error: &VoxelStreamError,
) -> Result<VoxelStreamError, VoxelTerrainRuntimeError> {
    let try_clone_message = |message: &str| -> Result<String, VoxelTerrainRuntimeError> {
        let mut cloned = String::new();
        cloned
            .try_reserve_exact(message.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        cloned.push_str(message);
        Ok(cloned)
    };
    match error {
        VoxelStreamError::InvalidLod { lod, max_lod } => Ok(VoxelStreamError::InvalidLod {
            lod: *lod,
            max_lod: *max_lod,
        }),
        VoxelStreamError::InvalidBlockPosition { position } => {
            Ok(VoxelStreamError::InvalidBlockPosition {
                position: *position,
            })
        }
        VoxelStreamError::BlockFormatMismatch => Ok(VoxelStreamError::BlockFormatMismatch),
        VoxelStreamError::UnsupportedOperation { operation } => {
            Ok(VoxelStreamError::UnsupportedOperation { operation })
        }
        VoxelStreamError::Io(message) => Ok(VoxelStreamError::Io(try_clone_message(message)?)),
        VoxelStreamError::CorruptData(message) => {
            Ok(VoxelStreamError::CorruptData(try_clone_message(message)?))
        }
    }
}

fn try_clone_stream_result(
    result: &StreamResult<()>,
) -> Result<StreamResult<()>, VoxelTerrainRuntimeError> {
    match result {
        Ok(()) => Ok(Ok(())),
        Err(error) => Ok(Err(try_clone_stream_error(error)?)),
    }
}

const MAX_LOAD_RETRIES: u32 = 10;
const MAX_MESH_TERMINAL_RETRIES: u32 = 10;
const MAX_AUTOMATIC_SAVE_ATTEMPTS: u32 = 3;
const MAX_EXPLICIT_SAVE_ATTEMPTS: u32 = 8;
const MAX_AUTOMATIC_CHECKPOINT_ATTEMPTS: u32 = 3;
const MAX_EXPLICIT_CHECKPOINT_ATTEMPTS: u32 = 8;
/// Normal runtime batches this many acknowledged block payloads before asking
/// the stream to make them durable. This bounds healthy retained payloads to
/// at most seven between process ticks without flushing every saved block.
const AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD: usize = 8;
/// A single or small number of unusually dense buffers also checkpoints once
/// their allocated channel storage reaches four MiB.
const AUTOMATIC_SAVE_CHECKPOINT_BYTE_THRESHOLD: usize = 4 * 1024 * 1024;

fn allocate_persistence_generation(
    next_generation: &mut u64,
) -> Result<u64, VoxelTerrainRuntimeError> {
    let generation = *next_generation;
    let Some(successor) = generation.checked_add(1) else {
        return Err(VoxelTerrainRuntimeError::SaveGenerationOverflow);
    };
    *next_generation = successor;
    Ok(generation)
}

fn allocate_request_generation(next_generation: &mut u64) -> Result<u64, VoxelTerrainRuntimeError> {
    let generation = *next_generation;
    let Some(successor) = generation.checked_add(1) else {
        return Err(VoxelTerrainRuntimeError::RequestGenerationOverflow);
    };
    *next_generation = successor;
    Ok(generation)
}

fn allocate_physical_request(
    request_epoch: u64,
    next_generation: &mut u64,
) -> Result<(u64, PhysicalRequest), VoxelTerrainRuntimeError> {
    let generation = allocate_request_generation(next_generation)?;
    Ok((
        generation,
        PhysicalRequest::new(TaskRequestTag::new(request_epoch, generation)),
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadRequestState {
    Queued,
    InFlight,
    NotFound,
    Exhausted,
}

/// One entry-owned physical worker request. Draft clones share the same
/// cancellation bit, but preparation never signals it. Only a committed map
/// replacement/removal or shutdown attempt may cancel the live request.
#[derive(Debug, Clone)]
struct PhysicalRequest {
    tag: TaskRequestTag,
    cancellation: Arc<RequestCancellation>,
}

impl PhysicalRequest {
    fn new(tag: TaskRequestTag) -> Self {
        Self {
            tag,
            cancellation: Arc::new(RequestCancellation::new()),
        }
    }

    fn matches(&self, tag: Option<TaskRequestTag>) -> bool {
        tag == Some(self.tag)
    }

    fn cancel(&self) {
        self.cancellation.cancel();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PersistenceTaskKind {
    Save,
    Flush,
}

// Keep the save payload inline: raw normalization reserves its destination
// before moving the terminal, so boxing here would add the one allocation this
// ownership boundary is designed to avoid.
#[allow(dead_code, clippy::large_enum_variant)]
enum PersistenceTaskTerminal {
    Save(SaveTaskTerminal),
    Flush(FlushTaskTerminal),
}

/// Globally ordered ownership for every completion that C1 cannot apply.
/// Persistence terminals remain available to C2's phase machine, while
/// unknown/malformed tasks remain available for typed diagnostics. One FIFO
/// preserves their original cross-kind order without blocking live ingress.
#[allow(dead_code)]
enum QuarantinedCompletion {
    Persistence {
        kind: PersistenceTaskKind,
        terminal: PersistenceTaskTerminal,
        attempt_ordinal: u64,
        completed: CompletedTask,
    },
    MalformedPersistence {
        kind: PersistenceTaskKind,
        terminal: PersistenceTaskTerminal,
        attempt_ordinal: u64,
        completed: CompletedTask,
    },
    Other {
        kind: CompletionTaskKind,
        completed: CompletedTask,
    },
}

impl QuarantinedCompletion {
    #[cfg(test)]
    fn completed(&self) -> &CompletedTask {
        match self {
            Self::Persistence { completed, .. }
            | Self::MalformedPersistence { completed, .. }
            | Self::Other { completed, .. } => completed,
        }
    }
}

// Durable completions are deliberately inline. `try_reserve` on the VecDeque
// happens before the raw front is touched, so normalization's final payload
// moves perform no allocator calls and cannot strand a half-consumed task.
#[allow(clippy::large_enum_variant)]
enum DurableCompletion {
    LoadFinished {
        completed: CompletedTask,
        output: TerrainLoadOutput,
    },
    LoadTerminal {
        completed: CompletedTask,
        position: Vector3i,
        lod_index: u8,
        request_generation: u64,
        request_tag: Option<TaskRequestTag>,
    },
    MeshFinished {
        completed: CompletedTask,
        output: MeshBlockTaskOutput,
    },
    DirectMesh {
        upload: Arc<MeshUploadSnapshot>,
        dropped: bool,
    },
    MeshTerminal {
        completed: CompletedTask,
        key: MeshBlockKey,
        request_tag: Option<TaskRequestTag>,
    },
    SaveAcknowledged {
        completed: CompletedTask,
        terminal: SaveTaskTerminal,
        attempt_ordinal: u64,
    },
    FlushAcknowledged {
        completed: CompletedTask,
        terminal: FlushTaskTerminal,
        attempt_ordinal: u64,
    },
    PersistenceTerminal {
        completed: CompletedTask,
        kind: PersistenceTaskKind,
        terminal: PersistenceTaskTerminal,
        attempt_ordinal: u64,
    },
    MalformedPersistence {
        completed: CompletedTask,
        kind: PersistenceTaskKind,
        terminal: PersistenceTaskTerminal,
        attempt_ordinal: u64,
    },
    MalformedFinished {
        completed: CompletedTask,
        kind: CompletionTaskKind,
    },
    UnknownTerminal {
        completed: CompletedTask,
    },
}

impl DurableCompletion {
    fn completed_mut(&mut self) -> Option<&mut CompletedTask> {
        match self {
            Self::LoadFinished { completed, .. }
            | Self::LoadTerminal { completed, .. }
            | Self::MeshFinished { completed, .. }
            | Self::MeshTerminal { completed, .. }
            | Self::SaveAcknowledged { completed, .. }
            | Self::FlushAcknowledged { completed, .. }
            | Self::PersistenceTerminal { completed, .. }
            | Self::MalformedPersistence { completed, .. }
            | Self::MalformedFinished { completed, .. }
            | Self::UnknownTerminal { completed } => Some(completed),
            Self::DirectMesh { .. } => None,
        }
    }

    fn descriptor(&self) -> DurableCompletionDescriptor {
        let (kind, owner) = match self {
            Self::LoadFinished { completed, .. } | Self::LoadTerminal { completed, .. } => (
                CompletionTaskKind::Load,
                completion_owner_identity(completed),
            ),
            Self::MeshFinished { completed, .. } | Self::MeshTerminal { completed, .. } => (
                CompletionTaskKind::Mesh,
                completion_owner_identity(completed),
            ),
            Self::SaveAcknowledged { completed, .. }
            | Self::PersistenceTerminal {
                completed,
                kind: PersistenceTaskKind::Save,
                ..
            }
            | Self::MalformedPersistence {
                completed,
                kind: PersistenceTaskKind::Save,
                ..
            } => (
                CompletionTaskKind::Save,
                completion_owner_identity(completed),
            ),
            Self::FlushAcknowledged { completed, .. }
            | Self::PersistenceTerminal {
                completed,
                kind: PersistenceTaskKind::Flush,
                ..
            }
            | Self::MalformedPersistence {
                completed,
                kind: PersistenceTaskKind::Flush,
                ..
            } => (
                CompletionTaskKind::Flush,
                completion_owner_identity(completed),
            ),
            Self::MalformedFinished { completed, kind } => {
                (*kind, completion_owner_identity(completed))
            }
            Self::UnknownTerminal { completed } => (
                CompletionTaskKind::Unknown,
                completion_owner_identity(completed),
            ),
            Self::DirectMesh { upload, .. } => (
                CompletionTaskKind::Mesh,
                CompletionOwnerIdentity::DirectMesh {
                    upload: Arc::as_ptr(upload) as usize,
                    key: upload.key(),
                },
            ),
        };
        DurableCompletionDescriptor { kind, owner }
    }
}

fn completion_owner_identity(completed: &CompletedTask) -> CompletionOwnerIdentity {
    CompletionOwnerIdentity::Runner {
        task: completed.task() as *const dyn ThreadedTask as *const () as usize,
        lane: completed.lane(),
        status: completed.status(),
        follow_up_count: completed.follow_up_count(),
    }
}

fn accepted_completion_has_no_followups(completion: &DurableCompletion) -> bool {
    match completion {
        DurableCompletion::LoadFinished { completed, .. }
        | DurableCompletion::MeshFinished { completed, .. }
        | DurableCompletion::SaveAcknowledged { completed, .. }
        | DurableCompletion::FlushAcknowledged { completed, .. } => {
            completed.follow_up_count() == 0
        }
        _ => true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionTaskKind {
    Load,
    Mesh,
    Save,
    Flush,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionDrainError {
    CapacityReservationFailed,
    #[cfg(test)]
    InjectedNormalizationFailure,
    #[cfg(test)]
    InjectedMeshEventReservationFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedLoadTerminalRecovery {
    Stale,
    Exhausted {
        retry_count: u32,
    },
    Rearm {
        retry_count: u32,
        generation: u64,
        next_generation: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedMeshTerminalRecovery {
    Stale,
    Quiesce {
        retry_count: u32,
    },
    Requeue {
        retry_count: u32,
        generation: u64,
        next_generation: u64,
    },
}

impl From<TryReserveError> for CompletionDrainError {
    fn from(_error: TryReserveError) -> Self {
        Self::CapacityReservationFailed
    }
}

impl From<PreparedTaskBatchError> for CompletionDrainError {
    fn from(error: PreparedTaskBatchError) -> Self {
        match error {
            PreparedTaskBatchError::Capacity(_) => Self::CapacityReservationFailed,
            #[cfg(test)]
            PreparedTaskBatchError::Injected => Self::CapacityReservationFailed,
        }
    }
}

impl From<CompletionDrainError> for VoxelTerrainRuntimeError {
    fn from(error: CompletionDrainError) -> Self {
        match error {
            CompletionDrainError::CapacityReservationFailed => Self::CompletionDrainCapacityFailed,
            #[cfg(test)]
            CompletionDrainError::InjectedNormalizationFailure => {
                Self::CompletionNormalizationFailed
            }
            #[cfg(test)]
            CompletionDrainError::InjectedMeshEventReservationFailure => {
                Self::MeshOutputApplyFailed
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct DataResidencyRefs {
    resident_viewers: u32,
    coverage_holds: u32,
}

impl DataResidencyRefs {
    const fn with_resident_viewers(resident_viewers: u32) -> Self {
        Self {
            resident_viewers,
            coverage_holds: 0,
        }
    }

    fn checked_total(self, location: BlockLocation) -> Result<u32, VoxelTerrainRuntimeError> {
        self.resident_viewers
            .checked_add(self.coverage_holds)
            .ok_or(VoxelTerrainRuntimeError::DataRefcountOverflow {
                location,
                field: DataRefField::Total,
            })
    }

    fn checked_apply_delta(
        self,
        location: BlockLocation,
        field: DataRefField,
        delta: i64,
    ) -> Result<Self, VoxelTerrainRuntimeError> {
        let current = match field {
            DataRefField::ResidentViewers => self.resident_viewers,
            DataRefField::CoverageHolds => self.coverage_holds,
            DataRefField::Total => {
                return Err(VoxelTerrainRuntimeError::DataRefcountOverflow { location, field })
            }
        };
        let next = i128::from(current) + i128::from(delta);
        if next < 0 {
            return Err(VoxelTerrainRuntimeError::DataRefcountUnderflow { location, field });
        }
        let next = u32::try_from(next)
            .map_err(|_| VoxelTerrainRuntimeError::DataRefcountOverflow { location, field })?;
        let mut updated = self;
        match field {
            DataRefField::ResidentViewers => updated.resident_viewers = next,
            DataRefField::CoverageHolds => updated.coverage_holds = next,
            DataRefField::Total => unreachable!("total is derived from its two exact fields"),
        }
        updated.checked_total(location)?;
        Ok(updated)
    }

    const fn is_empty(self) -> bool {
        self.resident_viewers == 0 && self.coverage_holds == 0
    }
}

#[derive(Debug, Clone)]
struct LoadingBlockEntry {
    residency: DataResidencyRefs,
    retry_count: u32,
    request_generation: u64,
    request_state: LoadRequestState,
    physical_request: Option<PhysicalRequest>,
}

impl LoadingBlockEntry {
    fn matches_physical_request(&self, tag: Option<TaskRequestTag>) -> bool {
        match &self.physical_request {
            Some(request) => request.matches(tag),
            None => tag.is_none(),
        }
    }

    fn matches_previous_epoch_request(
        &self,
        tag: Option<TaskRequestTag>,
        current_epoch: u64,
    ) -> bool {
        self.physical_request.as_ref().is_some_and(|request| {
            request.matches(tag) && request.tag.request_epoch != current_epoch
        })
    }

    fn cancel_physical_request_if_superseded_by(&self, next: Option<&PhysicalRequest>) {
        if let Some(current) = &self.physical_request {
            if next.is_none_or(|next| {
                next.tag != current.tag || !Arc::ptr_eq(&next.cancellation, &current.cancellation)
            }) {
                current.cancel();
            }
        }
    }
}

enum PreparedTerrainMode {
    Fixed(FixedLodTransactionDraft),
    #[cfg_attr(not(test), allow(dead_code))]
    Variable(Box<VariableLodTransactionDraft>),
}

struct FixedLodTransactionDraft {
    data_retry_publication: Option<PreparedFixedDataRetryPublication>,
    accepted_feature_updates: Vec<MeshBlockLocation>,
    active_feature_updates: Vec<MeshBlockLocation>,
    topology_event_index: Option<usize>,
}

struct VariableLodTransactionDraft {
    coordinator_update: Option<ValidatedCoordinatorUpdate>,
    coverage_publication: Option<PreparedVariableCoveragePublication>,
}

struct PreparedVariableCoveragePublication {
    preview: Option<ValidatedCoveragePreview>,
    hold_phases: Option<PreparedCoverageHoldPhases>,
    #[allow(dead_code)] // produced by the planner; read by variable-mode tests
    physical_observations: PreparedVariablePhysicalObservations,
}

#[allow(dead_code)] // produced by the planner; read by variable-mode tests
#[derive(Default)]
struct PreparedVariablePhysicalObservations {
    mesh: Vec<VariableMeshPhysicalObservation>,
    data: Vec<VariableDataPhysicalObservation>,
}

#[allow(dead_code)] // produced by the planner; read by variable-mode tests
struct VariableMeshPhysicalObservation {
    location: MeshBlockLocation,
    expected: Option<MeshBlockEntry>,
}

#[allow(dead_code)] // produced by the planner; read by variable-mode tests
enum VariableDataPhysicalObservation {
    Loaded {
        location: BlockLocation,
        expected: DataResidencyRefs,
    },
    Loading {
        location: BlockLocation,
        expected: LoadingBlockEntry,
    },
    Missing {
        location: BlockLocation,
    },
}

struct VariableMeshPhysicalShadow {
    location: MeshBlockLocation,
    expected: Option<MeshBlockEntry>,
    next: Option<MeshBlockEntry>,
}

struct VariableDataPhysicalShadow {
    location: BlockLocation,
    expected_loaded: Option<DataResidencyRefs>,
    expected_loading: Option<LoadingBlockEntry>,
    next: DataResidencyRefs,
    snapshot: SharedVoxelDataTransactionBlockSnapshot,
}

struct PreparedVariablePhysicalSlice {
    mesh_diffs: Vec<PreparedMeshEntryDiff>,
    data_residency_diffs: Vec<PreparedDataResidencyDiff>,
    loading_diffs: Vec<PreparedLoadingEntryDiff>,
    pending_load_queues: Vec<PreparedQueueDiff<Vector3i>>,
    pending_mesh_queues: Vec<PreparedQueueDiff<Vector3i>>,
    data_operations: Vec<SharedVoxelDataTransactionOperation>,
    data_snapshots: Vec<SharedVoxelDataTransactionBlockSnapshot>,
    scheduled_tasks: Vec<ScheduledTask>,
    persistence: PreparedOpaquePersistenceState,
    events_to_append: Vec<VoxelTerrainEvent>,
    next_request_generation: u64,
    next_mesh_revision: u64,
    next_render_topology_revision: u64,
    next_stats: VoxelTerrainStats,
    observations: PreparedVariablePhysicalObservations,
}

struct PreparedFixedDataRetryPublication {
    next_data_view_retries: Vec<PendingDataMutation>,
    next_data_unview_retries: Vec<PendingDataMutation>,
    next_last_data_mutation_error: Option<SharedVoxelDataMutationError>,
}

enum PreparedMapAction<V> {
    Insert(V),
    Replace(V),
    Remove,
}

struct PreparedMeshEntryDiff {
    location: MeshBlockLocation,
    expected_revision: Option<u64>,
    action: PreparedMapAction<MeshBlockEntry>,
}

struct PreparedLoadingEntryDiff {
    location: BlockLocation,
    expected_generation: Option<u64>,
    action: PreparedMapAction<LoadingBlockEntry>,
}

struct PreparedDataResidencyDiff {
    location: BlockLocation,
    expected: Option<DataResidencyRefs>,
    action: PreparedMapAction<DataResidencyRefs>,
}

struct PreparedQueueDiff<T> {
    lod_index: u8,
    final_values: Vec<T>,
}

fn canonical_block_location_key(location: BlockLocation) -> (u8, i32, i32, i32) {
    (
        location.lod_index,
        location.position.x,
        location.position.y,
        location.position.z,
    )
}

fn canonical_mesh_location_key(location: MeshBlockLocation) -> (u8, i32, i32, i32) {
    (
        location.lod_index,
        location.position_in_blocks.x,
        location.position_in_blocks.y,
        location.position_in_blocks.z,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
fn mesh_entry_physical_state_matches(left: &MeshBlockEntry, right: &MeshBlockEntry) -> bool {
    let physical_request_matches = match (&left.physical_request, &right.physical_request) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.tag == right.tag && Arc::ptr_eq(&left.cancellation, &right.cancellation)
        }
        (None, Some(_)) | (Some(_), None) => false,
    };
    let accepted_upload_matches = match (&left.accepted_upload, &right.accepted_upload) {
        (None, None) => true,
        (Some(left), Some(right)) => Arc::ptr_eq(left, right),
        (None, Some(_)) | (Some(_), None) => false,
    };
    left.position == right.position
        && left.resident_viewers == right.resident_viewers
        && left.visual_viewers == right.visual_viewers
        && left.collision_viewers == right.collision_viewers
        && left.visual_coverage_holds == right.visual_coverage_holds
        && left.collision_coverage_holds == right.collision_coverage_holds
        && left.visual_active == right.visual_active
        && left.collision_active == right.collision_active
        && left.is_loaded == right.is_loaded
        && left.requested_revision == right.requested_revision
        && left.request_generation == right.request_generation
        && left.requested_features == right.requested_features
        && left.applied_features == right.applied_features
        && left.applied_revision == right.applied_revision
        && left.has_geometry == right.has_geometry
        && left.is_in_update_list == right.is_in_update_list
        && left.terminal_retry_count == right.terminal_retry_count
        && physical_request_matches
        && accepted_upload_matches
}

#[cfg_attr(not(test), allow(dead_code))]
fn loading_entry_physical_state_matches(
    left: &LoadingBlockEntry,
    right: &LoadingBlockEntry,
) -> bool {
    let request_matches = match (&left.physical_request, &right.physical_request) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.tag == right.tag && Arc::ptr_eq(&left.cancellation, &right.cancellation)
        }
        (None, Some(_)) | (Some(_), None) => false,
    };
    left.residency == right.residency
        && left.retry_count == right.retry_count
        && left.request_generation == right.request_generation
        && left.request_state == right.request_state
        && request_matches
}

#[cfg_attr(not(test), allow(dead_code))]
fn ensure_variable_mesh_shadow<'a>(
    core: &VoxelTerrainCore,
    shadows: &'a mut BTreeMap<(u8, i32, i32, i32), VariableMeshPhysicalShadow>,
    location: MeshBlockLocation,
) -> Result<&'a mut VariableMeshPhysicalShadow, VariableModeTestError> {
    let lod = usize::from(location.lod_index);
    if lod >= usize::from(core.lod_count) || lod >= core.mesh_maps.len() {
        return Err(VoxelTerrainRuntimeError::LodMath(LodMathError::InvalidLodCount).into());
    }
    let key = canonical_mesh_location_key(location);
    if let std::collections::btree_map::Entry::Vacant(slot) = shadows.entry(key) {
        let expected = core.mesh_maps[lod]
            .get(&location.position_in_blocks)
            .map(MeshBlockEntry::clone_for_draft);
        let queued_count = core.blocks_pending_update[lod]
            .iter()
            .filter(|position| **position == location.position_in_blocks)
            .count();
        if let Some(entry) = expected.as_ref() {
            if entry.is_in_update_list != (queued_count == 1) || queued_count > 1 {
                return Err(
                    VariablePhysicalPrepareError::MeshQueueOwnerMismatch { location }.into(),
                );
            }
            if entry.physical_request.as_ref().is_some_and(|request| {
                request.tag != TaskRequestTag::new(core.request_epoch, entry.request_generation)
                    || request.cancellation.is_cancelled()
            }) || (entry.is_in_update_list && entry.physical_request.is_none())
            {
                return Err(
                    VariablePhysicalPrepareError::MeshRequestOwnerMismatch { location }.into(),
                );
            }
        } else if queued_count != 0 {
            return Err(VariablePhysicalPrepareError::MeshQueueOwnerMismatch { location }.into());
        }
        let next = expected.as_ref().map(MeshBlockEntry::clone_for_draft);
        slot.insert(VariableMeshPhysicalShadow {
            location,
            expected,
            next,
        });
    }
    Ok(shadows
        .get_mut(&key)
        .unwrap_or_else(|| unreachable!("inserted variable mesh shadow remains present")))
}

#[cfg_attr(not(test), allow(dead_code))]
fn ensure_variable_data_shadow<'a>(
    core: &VoxelTerrainCore,
    preview: &SharedVoxelDataTransactionPreview,
    shadows: &'a mut BTreeMap<(u8, i32, i32, i32), VariableDataPhysicalShadow>,
    location: BlockLocation,
) -> Result<&'a mut VariableDataPhysicalShadow, VariableModeTestError> {
    let lod = usize::from(location.lod_index);
    if lod >= usize::from(core.lod_count)
        || lod >= core.loading_blocks.len()
        || lod >= core.loaded_data_residency.len()
        || lod >= core.blocks_pending_load.len()
    {
        return Err(VoxelTerrainRuntimeError::LodMath(LodMathError::InvalidLodCount).into());
    }
    let key = canonical_block_location_key(location);
    if let std::collections::btree_map::Entry::Vacant(slot) = shadows.entry(key) {
        let expected_loaded = core.loaded_data_residency[lod]
            .get(&location.position)
            .copied();
        let expected_loading = core.loading_blocks[lod].get(&location.position).cloned();
        if expected_loaded.is_some() && expected_loading.is_some() {
            return Err(VariablePhysicalPrepareError::DataLoadingDeferred { location }.into());
        }
        let Some(snapshot) = preview.block_snapshot(location) else {
            return Err(VariablePhysicalPrepareError::DataInsertDeferred { location }.into());
        };
        let queued_count = core.blocks_pending_load[lod]
            .iter()
            .filter(|position| **position == location.position)
            .count();
        if let Some(entry) = expected_loading.as_ref() {
            let queue_matches = match entry.request_state {
                LoadRequestState::Queued => queued_count == 1,
                LoadRequestState::InFlight
                | LoadRequestState::NotFound
                | LoadRequestState::Exhausted => queued_count == 0,
            };
            if !queue_matches {
                return Err(
                    VariablePhysicalPrepareError::DataQueueOwnerMismatch { location }.into(),
                );
            }
            let request_matches = match entry.request_state {
                LoadRequestState::Queued | LoadRequestState::InFlight => {
                    entry.physical_request.as_ref().is_some_and(|request| {
                        request.tag
                            == TaskRequestTag::new(core.request_epoch, entry.request_generation)
                            && !request.cancellation.is_cancelled()
                    })
                }
                LoadRequestState::NotFound | LoadRequestState::Exhausted => {
                    entry.physical_request.is_none()
                }
            };
            if !request_matches {
                return Err(
                    VariablePhysicalPrepareError::DataRequestOwnerMismatch { location }.into(),
                );
            }
        } else if queued_count != 0 {
            return Err(VariablePhysicalPrepareError::DataQueueOwnerMismatch { location }.into());
        }

        let next = if snapshot.is_present() {
            if expected_loading.is_some() {
                return Err(VariablePhysicalPrepareError::DataLoadingDeferred { location }.into());
            }
            let Some(expected) = expected_loaded else {
                return Err(VariablePhysicalPrepareError::DataInsertDeferred { location }.into());
            };
            let expected_viewers = expected.checked_total(location)?;
            if snapshot.viewers() != expected_viewers {
                return Err(VariablePhysicalPrepareError::DataViewerCountMismatch {
                    location,
                    expected: expected_viewers,
                    actual: snapshot.viewers(),
                }
                .into());
            }
            expected
        } else {
            if expected_loaded.is_some() {
                return Err(VariablePhysicalPrepareError::DataNotReady { location }.into());
            }
            expected_loading
                .as_ref()
                .map_or(DataResidencyRefs::default(), |entry| entry.residency)
        };
        slot.insert(VariableDataPhysicalShadow {
            location,
            expected_loaded,
            expected_loading,
            next,
            snapshot,
        });
    }
    Ok(shadows
        .get_mut(&key)
        .unwrap_or_else(|| unreachable!("inserted variable data shadow remains present")))
}

#[cfg_attr(not(test), allow(dead_code))]
fn mesh_hold_refcount(entry: &MeshBlockEntry, feature: CoverageFeature) -> u32 {
    match feature {
        CoverageFeature::Visual => entry.visual_coverage_holds,
        CoverageFeature::Collision => entry.collision_coverage_holds,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn set_mesh_hold_refcount(entry: &mut MeshBlockEntry, feature: CoverageFeature, next: u32) {
    match feature {
        CoverageFeature::Visual => entry.visual_coverage_holds = next,
        CoverageFeature::Collision => entry.collision_coverage_holds = next,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn mesh_feature_active(entry: &MeshBlockEntry, feature: CoverageFeature) -> bool {
    match feature {
        CoverageFeature::Visual => entry.visual_active,
        CoverageFeature::Collision => entry.collision_active,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn set_mesh_feature_active(entry: &mut MeshBlockEntry, feature: CoverageFeature, active: bool) {
    match feature {
        CoverageFeature::Visual => entry.visual_active = active,
        CoverageFeature::Collision => entry.collision_active = active,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn mesh_feature_needed(entry: &MeshBlockEntry, feature: CoverageFeature) -> bool {
    match feature {
        CoverageFeature::Visual => entry.needs_visual(),
        CoverageFeature::Collision => entry.needs_collision(),
    }
}

fn try_canonical_duplicate_location_key(
    keys: impl ExactSizeIterator<Item = (u8, i32, i32, i32)>,
) -> Result<Option<(u8, i32, i32, i32)>, VoxelTerrainRuntimeError> {
    let len = keys.len();
    if len < 2 {
        return Ok(None);
    }
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(len)
        .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
    scratch.extend(keys);
    scratch.sort_unstable();
    Ok(scratch
        .windows(2)
        .find_map(|pair| (pair[0] == pair[1]).then_some(pair[0])))
}

fn block_location_from_canonical_key(key: (u8, i32, i32, i32)) -> BlockLocation {
    BlockLocation {
        lod_index: key.0,
        position: Vector3i::new(key.1, key.2, key.3),
    }
}

fn mesh_location_from_canonical_key(key: (u8, i32, i32, i32)) -> MeshBlockLocation {
    MeshBlockLocation::new(Vector3i::new(key.1, key.2, key.3), key.0)
}

fn try_canonical_duplicate_mesh_key(
    diffs: &[PreparedMeshEntryDiff],
) -> Result<Option<MeshBlockLocation>, VoxelTerrainRuntimeError> {
    try_canonical_duplicate_location_key(
        diffs
            .iter()
            .map(|diff| canonical_mesh_location_key(diff.location)),
    )
    .map(|key| key.map(mesh_location_from_canonical_key))
}

fn try_canonical_duplicate_loading_key(
    diffs: &[PreparedLoadingEntryDiff],
) -> Result<Option<BlockLocation>, VoxelTerrainRuntimeError> {
    try_canonical_duplicate_location_key(
        diffs
            .iter()
            .map(|diff| canonical_block_location_key(diff.location)),
    )
    .map(|key| key.map(block_location_from_canonical_key))
}

fn try_canonical_duplicate_data_residency_key(
    diffs: &[PreparedDataResidencyDiff],
) -> Result<Option<BlockLocation>, VoxelTerrainRuntimeError> {
    try_canonical_duplicate_location_key(
        diffs
            .iter()
            .map(|diff| canonical_block_location_key(diff.location)),
    )
    .map(|key| key.map(block_location_from_canonical_key))
}

fn canonical_duplicate_queue_lod<T>(diffs: &[PreparedQueueDiff<T>]) -> Option<u8> {
    let mut seen = [false; MAX_LOD];
    let mut duplicate = None;
    for diff in diffs {
        let lod = usize::from(diff.lod_index);
        debug_assert!(lod < MAX_LOD);
        if seen[lod] {
            duplicate =
                Some(duplicate.map_or(diff.lod_index, |current: u8| current.min(diff.lod_index)));
        } else {
            seen[lod] = true;
        }
    }
    duplicate
}

#[derive(Default)]
struct PreparedCompletionPrefix {
    len: usize,
    accepted_followup_count: usize,
    plans: Vec<PreparedCompletionPlan>,
    load_inserts: Vec<PreparedLoadInsert>,
}

struct PreparedCompletionPlan {
    inbox_index: usize,
    descriptor: DurableCompletionDescriptor,
    disposition: PreparedCompletionDisposition,
    followups: Option<PreparedCompletedTaskFollowUps>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DurableCompletionDescriptor {
    kind: CompletionTaskKind,
    owner: CompletionOwnerIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionOwnerIdentity {
    Runner {
        task: usize,
        lane: TaskLane,
        status: TaskCompletionStatus,
        follow_up_count: usize,
    },
    DirectMesh {
        upload: usize,
        key: MeshBlockKey,
    },
}

impl CompletionOwnerIdentity {
    const fn follow_up_count(self) -> usize {
        match self {
            Self::Runner {
                follow_up_count, ..
            } => follow_up_count,
            Self::DirectMesh { .. } => 0,
        }
    }

    const fn with_follow_up_count(self, count: usize) -> Self {
        match self {
            Self::Runner {
                task, lane, status, ..
            } => Self::Runner {
                task,
                lane,
                status,
                follow_up_count: count,
            },
            direct @ Self::DirectMesh { .. } => direct,
        }
    }
}

struct PreparedDirectMeshPlan {
    inbox_index: usize,
    descriptor: DurableCompletionDescriptor,
}

#[derive(Debug, Clone, Copy)]
struct PreparedLoadInsert {
    inbox_index: usize,
    location: BlockLocation,
    final_viewers: u32,
}

enum PreparedCompletionDisposition {
    Retire {
        publish_followups: bool,
    },
    ApplyPersistence {
        publish_followups: bool,
        action: PreparedPersistenceAction,
    },
    Quarantine,
}

impl PreparedCompletionDisposition {
    const fn publishes_followups(&self) -> bool {
        match self {
            Self::Retire { publish_followups }
            | Self::ApplyPersistence {
                publish_followups, ..
            } => *publish_followups,
            Self::Quarantine => false,
        }
    }
}

enum PreparedPersistenceAction {
    Save(PreparedSaveAction),
    Checkpoint(PreparedCheckpointAction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedSaveAction {
    AcknowledgeSuccess {
        key: SaveKey,
        block_revision: u64,
        generation: u64,
        attempt_ordinal: u64,
    },
    AcknowledgeFailure {
        key: SaveKey,
        block_revision: u64,
        generation: u64,
        attempt_ordinal: u64,
        next_retry_count: u32,
    },
    RestoreBeforeIo {
        key: SaveKey,
        block_revision: u64,
        generation: u64,
        attempt_ordinal: u64,
        next_retry_count: u32,
    },
    MarkIndeterminate {
        key: SaveKey,
        block_revision: u64,
        generation: u64,
        attempt_ordinal: u64,
    },
}

enum PreparedCheckpointAction {
    Acknowledge {
        checkpoint_generation: u64,
        attempt_ordinal: u64,
        succeeded: bool,
        origin: CheckpointOrigin,
        explicit_outcome: Option<StreamResult<()>>,
        entry_actions: Vec<PreparedCheckpointEntryAction>,
    },
    RestoreBeforeIo {
        checkpoint_generation: u64,
        attempt_ordinal: u64,
        next_retry_count: u32,
    },
    MarkIndeterminate {
        checkpoint_generation: u64,
        attempt_ordinal: u64,
    },
}

enum PreparedCheckpointEntryAction {
    ClearWritten {
        key: SaveKey,
        block_revision: u64,
        generation: u64,
        promote_queued: bool,
        defer_active: bool,
        remove_entry: bool,
    },
    RestoreWritten {
        key: SaveKey,
        block_revision: u64,
        generation: u64,
        placement: PreparedCheckpointRestorePlacement,
        error: Option<VoxelStreamError>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedCheckpointRestorePlacement {
    Active,
    ReplacePending,
    Queue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedJournalActiveShadow {
    None,
    Pending {
        block_revision: u64,
        generation: u64,
        retry_count: u32,
    },
    WriteInFlight {
        block_revision: u64,
        generation: u64,
        attempt_ordinal: u64,
        retry_count: u32,
    },
    Indeterminate {
        block_revision: u64,
        generation: u64,
        attempt_ordinal: u64,
    },
}

#[derive(Debug, Clone, Copy)]
struct PreparedJournalShadow {
    written_block_revision: Option<u64>,
    written_generation: Option<u64>,
    active: PreparedJournalActiveShadow,
    queued_len: usize,
    queued_front: Option<(u64, u64, u32)>,
}

#[derive(Debug, Clone, Copy)]
struct PreparedCheckpointShadow {
    checkpoint_generation: u64,
    state: CheckpointAttemptState,
    retry_count: u32,
    max_attempts: u32,
    origin: CheckpointOrigin,
    record_per_block_failure: bool,
}

#[derive(Default)]
struct PreparedOpaquePersistenceState {
    removed_owner_routes: Vec<PreparedRemovedOwnerRoute>,
    resident_save_routes: Vec<PreparedResidentSaveRoute>,
    save_dispatches: Vec<PreparedPendingSaveDispatch>,
    save_recovery_updates: Vec<PreparedSaveRecoveryUpdate>,
    resolution: Option<PreparedFixedPersistenceResolution>,
    checkpoint_config_update: Option<PreparedCheckpointConfigUpdate>,
    checkpoint_dispatch: Option<PreparedCheckpointDispatch>,
    checkpoint_begin: Option<SaveCheckpointInFlight>,
    next_deferred_save_dispatch_keys: Vec<SaveKey>,
    next_deferred_checkpoint_dispatch: bool,
    next_automatic_save_checkpoint_blocked: bool,
    next_save_dispatch_error: Option<VoxelTerrainRuntimeError>,
    next_save_generation: u64,
    next_save_checkpoint_generation: u64,
    next_persistence_attempt_ordinal: u64,
    next_force_checkpoint_requested: bool,
    consume_save_panic_hook: bool,
    #[cfg(test)]
    next_panic_flush_before_io_attempts_for_test: usize,
    #[cfg(test)]
    next_panic_flush_after_ack_for_test: bool,
}

enum PreparedRemovedOwnerRoute {
    Clean,
    DirtyDispatch(PreparedDirtyDispatch),
    DirtyPending(PreparedDirtyPending),
    DirtyRetained {
        location: BlockLocation,
        block_revision: u64,
    },
}

enum PreparedResidentSaveRoute {
    Dispatch {
        payload: RevisionedBlockToSave,
        route: PreparedDirtyDispatch,
    },
    Pending {
        payload: RevisionedBlockToSave,
        route: PreparedDirtyPending,
    },
}

#[derive(Debug, Clone, Copy)]
enum PreparedJournalTarget {
    Vacant,
    ActiveIdle,
    Queued,
}

struct PreparedDirtyDispatch {
    location: BlockLocation,
    block_revision: u64,
    save_generation: u64,
    attempt_ordinal: u64,
    target: PreparedJournalTarget,
    installer: PreparedSaveBlockDataPayloadInstaller,
}

struct PreparedDirtyPending {
    location: BlockLocation,
    block_revision: u64,
    save_generation: u64,
    target: PreparedJournalTarget,
}

struct PreparedPendingSaveDispatch {
    key: SaveKey,
    block_revision: u64,
    generation: u64,
    attempt_ordinal: u64,
    installer: PreparedSaveBlockDataPayloadInstaller,
}

#[derive(Debug, Clone, Copy)]
struct PreparedCheckpointDispatch {
    checkpoint_generation: u64,
    attempt_ordinal: u64,
}

#[derive(Debug, Clone, Copy)]
struct FixedCheckpointRequest {
    origin: CheckpointOrigin,
    max_attempts: u32,
    record_per_block_failure: bool,
    reset_pending_retry_count: bool,
}

#[derive(Debug, Clone, Copy)]
struct FixedPersistenceRecoveryRequest {
    reset_pending_save_failures: bool,
    authorize_automatic_checkpoint: bool,
}

#[derive(Debug, Clone, Copy)]
struct PreparedSaveRecoveryUpdate {
    key: SaveKey,
    block_revision: u64,
    generation: u64,
}

#[derive(Debug, Clone, Copy)]
struct FixedPersistenceResolutionRequest {
    operation: PersistenceOperation,
    resolution: IndeterminateIoResolution,
}

enum PreparedFixedPersistenceResolution {
    Save {
        quarantine_index: usize,
        key: SaveKey,
        block_revision: u64,
        generation: u64,
        attempt_ordinal: u64,
        assume_written: bool,
    },
    CheckpointRetry {
        quarantine_index: usize,
        checkpoint_generation: u64,
        attempt_ordinal: u64,
    },
    CheckpointWritten {
        quarantine_index: usize,
        checkpoint_generation: u64,
        attempt_ordinal: u64,
        entry_actions: Vec<PreparedCheckpointEntryAction>,
    },
}

#[derive(Debug, Clone, Copy)]
struct PreparedCheckpointConfigUpdate {
    checkpoint_generation: u64,
    retry_count: u32,
    max_attempts: u32,
    origin: CheckpointOrigin,
    record_per_block_failure: bool,
}

/// Exact variable-LOD publication products whose buffers may still be
/// observed by readers admitted before the storage publication fence. Keep
/// every owner alive through fence release, task linkage and wake.
struct VariableModeRetirement {
    _coordinator_delta: ResidentDemandDelta,
    _coverage_result: Option<CoverageReconcileResult>,
    _before_mesh_updates: Vec<CoverageMeshRefcountUpdate>,
    _before_data_updates: Vec<CoverageDataRefcountUpdate>,
    _after_mesh_updates: Vec<CoverageMeshRefcountUpdate>,
    _after_data_updates: Vec<CoverageDataRefcountUpdate>,
    #[cfg(test)]
    _physical_observations: PreparedVariablePhysicalObservations,
}

#[derive(Default)]
struct RetirementBag {
    mesh_entries: Vec<MeshBlockEntry>,
    loading_entries: Vec<LoadingBlockEntry>,
    completed_tasks: Vec<CompletedTask>,
    mesh_snapshots: Vec<Arc<MeshUploadSnapshot>>,
    durable_completions: Vec<DurableCompletion>,
    data_blocks: Vec<RemovedSharedVoxelDataBlock>,
    paired_viewers: Vec<PairedViewer>,
    data_view_retries: Vec<PendingDataMutation>,
    data_unview_retries: Vec<PendingDataMutation>,
    last_data_mutation_error: Option<SharedVoxelDataMutationError>,
    save_journal_entries: Vec<SaveJournalEntry>,
    followup_escrows: Vec<PreparedCompletedTaskFollowUps>,
    save_attempt_meta: Vec<SaveAttemptMeta>,
    checkpoints: Vec<SaveCheckpointInFlight>,
    written_saves: Vec<WrittenSave>,
    stream_errors: Vec<VoxelStreamError>,
    #[cfg(test)]
    stream_error_drop_probes: Vec<FixedStreamErrorRetirementProbe>,
    checkpoint_outcomes: Vec<(u64, StreamResult<()>)>,
    checkpoint_actions: Vec<PreparedCheckpointAction>,
    deferred_save_keys: Vec<SaveKey>,
    runtime_errors: Vec<VoxelTerrainRuntimeError>,
    coordinator_state: Option<CoordinatorStateRetirement>,
    coverage_state: Option<CoverageStateRetirement>,
    previous_coverage_holds: Option<CoverageHoldLedger>,
    before_topology_coverage_holds: Option<CoverageHoldLedger>,
    variable_mode: Option<VariableModeRetirement>,
}

struct TerrainTransactionDraft {
    paired_viewer_publication: Option<Vec<PairedViewer>>,
    mode: PreparedTerrainMode,
    next_request_generation: u64,
    next_mesh_revision: u64,
    next_render_topology_revision: u64,
    next_stats: VoxelTerrainStats,
    data_tx: PreparedSharedVoxelDataTransaction,
    mesh_diffs: Vec<PreparedMeshEntryDiff>,
    data_residency_diffs: Vec<PreparedDataResidencyDiff>,
    loading_diffs: Vec<PreparedLoadingEntryDiff>,
    pending_load_queues: Vec<PreparedQueueDiff<Vector3i>>,
    pending_mesh_queues: Vec<PreparedQueueDiff<Vector3i>>,
    persistence: PreparedOpaquePersistenceState,
    completion_prefix: PreparedCompletionPrefix,
    direct_mesh_plans: Vec<PreparedDirectMeshPlan>,
    scheduled_task_count: usize,
    prepared_task_batch: FilledPreparedTaskBatch,
    events_to_append: Vec<VoxelTerrainEvent>,
    retirement: RetirementBag,
}

struct PreparedVoxelEditMeshUpdate {
    location: MeshBlockLocation,
    next_entry: MeshBlockEntry,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // retained as a unit-test harness after the planner cutover
struct PreparedVariableRemovalSave {
    location: BlockLocation,
    save_generation: u64,
}

struct PreparedVoxelEditPublication {
    data_transaction: PreparedSharedVoxelDataTransaction,
    edited_block: BlockLocation,
    block_revision: u64,
    inserted_data_residencies: Vec<(BlockLocation, DataResidencyRefs)>,
    loading_generations_to_retire: Vec<(BlockLocation, u64)>,
    pending_load_queue_replacements: Vec<Option<Vec<Vector3i>>>,
    mesh_updates: Vec<PreparedVoxelEditMeshUpdate>,
    pending_queue_replacements: Vec<Option<Vec<Vector3i>>>,
    next_request_generation: u64,
    next_mesh_revision: u64,
    affected_mesh_blocks: Vec<MeshBlockLocation>,
    retired_mesh_entries: Vec<MeshBlockEntry>,
    retired_loading_entries: Vec<LoadingBlockEntry>,
    retired_pending_load_queues: Vec<Vec<Vector3i>>,
    retired_pending_queues: Vec<Vec<Vector3i>>,
}

struct PreparedTerrainPublication {
    paired_viewer_publication: Option<Vec<PairedViewer>>,
    mode: PreparedTerrainMode,
    next_request_generation: u64,
    next_mesh_revision: u64,
    next_render_topology_revision: u64,
    next_stats: VoxelTerrainStats,
    mesh_diffs: Vec<PreparedMeshEntryDiff>,
    data_residency_diffs: Vec<PreparedDataResidencyDiff>,
    loading_diffs: Vec<PreparedLoadingEntryDiff>,
    pending_load_queues: Vec<PreparedQueueDiff<Vector3i>>,
    pending_mesh_queues: Vec<PreparedQueueDiff<Vector3i>>,
    persistence: PreparedOpaquePersistenceState,
    completion_prefix: PreparedCompletionPrefix,
    direct_mesh_plans: Vec<PreparedDirectMeshPlan>,
    scheduled_task_count: usize,
    prepared_task_batch: FilledPreparedTaskBatch,
    events_to_append: Vec<VoxelTerrainEvent>,
    retirement: RetirementBag,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingDataMutation {
    box_in_blocks: Box3i,
    retry_count: u32,
}

/// One paired viewer specification, used to add or update viewers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewerUpdate {
    pub id: ViewerId,
    pub world_position_voxels: Vector3i,
    pub horizontal_view_distance_voxels: i32,
    pub vertical_view_distance_voxels: i32,
    pub demand: MeshDemand,
}

/// Callback-free publication boundaries used by deterministic concurrency
/// tests. The pause control owns only synchronization state, so no test code
/// can run while a terrain or storage publication guard is held.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FixedCommitPausePhase {
    StorageFencedBeforeCorePublish,
    AfterTerrainPublishBeforeFenceFinish,
    BatchLinkedBeforeWake,
    AfterWakeBeforeRetirementDrop,
}

/// Logical allocation destinations in the fixed-LOD prepare phase. Tests arm
/// these typed points instead of relying on allocator-global failure order.
#[allow(dead_code)] // most variants are only constructed under #[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FixedCapacityDestination {
    PairedViewers,
    VariableCoordinatorPreparedState,
    VariableCoverageInputs,
    VariableCoveragePreview,
    VariableCoverageHoldResolution,
    VariableCoverageHoldBind,
    VariablePhysicalResourceSnapshot,
    VariablePhysicalShadows,
    VariableTopologyEvent,
    MeshMap,
    LoadingMap,
    PendingLoadQueue,
    PendingMeshQueue,
    DataViewRetries,
    DataUnviewRetries,
    DurableEffects,
    DirectEffects,
    EventOutbox,
    Quarantine,
    SaveJournal,
    SaveJournalQueue,
    DeferredSaveQueue,
    DirtyRetention,
    Retirement,
    PreparedTaskBatch,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, PartialEq, Eq)]
enum VariableModeTestError {
    Runtime(VoxelTerrainRuntimeError),
    Physical(VariablePhysicalPrepareError),
}

impl From<VoxelTerrainRuntimeError> for VariableModeTestError {
    fn from(error: VoxelTerrainRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<VariablePhysicalPrepareError> for VariableModeTestError {
    fn from(error: VariablePhysicalPrepareError) -> Self {
        Self::Physical(error)
    }
}

/// Production projection of the planner's test-error union into the runtime
/// error type: the `Runtime` variant is unwrapped, the `Physical` variant is
/// surfaced as a typed `VariablePhysicalPrepare` error.
impl From<VariableModeTestError> for VoxelTerrainRuntimeError {
    fn from(error: VariableModeTestError) -> Self {
        match error {
            VariableModeTestError::Runtime(runtime) => runtime,
            VariableModeTestError::Physical(physical) => {
                VoxelTerrainRuntimeError::VariablePhysicalPrepare(physical)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariablePhysicalPrepareError {
    MeshInsertDeferred {
        location: MeshBlockLocation,
    },
    MeshQueueOwnerMismatch {
        location: MeshBlockLocation,
    },
    MeshRequestOwnerMismatch {
        location: MeshBlockLocation,
    },
    DataInsertDeferred {
        location: BlockLocation,
    },
    DataLoadingDeferred {
        location: BlockLocation,
    },
    DataQueueOwnerMismatch {
        location: BlockLocation,
    },
    DataRequestOwnerMismatch {
        location: BlockLocation,
    },
    DataNotReady {
        location: BlockLocation,
    },
    TopologyDataNotReady {
        location: BlockLocation,
    },
    CoordinatorMeshCountMismatch {
        location: MeshBlockLocation,
        field: DemandCountField,
        expected: u32,
        actual: u32,
    },
    CoordinatorDataCountMismatch {
        location: BlockLocation,
        expected: u32,
        actual: u32,
    },
    MeshHoldCountMismatch {
        resource: CoverageMeshResource,
        expected: u32,
        actual: u32,
    },
    DataHoldCountMismatch {
        location: BlockLocation,
        expected: u32,
        actual: u32,
    },
    DataViewerCountMismatch {
        location: BlockLocation,
        expected: u32,
        actual: u32,
    },
    TopologyFinalStateMismatch {
        location: MeshBlockLocation,
        feature: CoverageFeature,
        expected: bool,
        actual: bool,
    },
    ActivationMissingAcceptedSnapshot {
        location: MeshBlockLocation,
        feature: CoverageFeature,
    },
    ActivationNotResident {
        location: MeshBlockLocation,
        feature: CoverageFeature,
    },
    ActivationFeatureNotNeeded {
        location: MeshBlockLocation,
        feature: CoverageFeature,
    },
    ActivationMissingUpload {
        location: MeshBlockLocation,
        feature: CoverageFeature,
    },
    ActivationUploadLocationMismatch {
        location: MeshBlockLocation,
        feature: CoverageFeature,
        actual: MeshBlockLocation,
    },
    ActivationUploadRevisionMismatch {
        location: MeshBlockLocation,
        feature: CoverageFeature,
        expected: u64,
        actual: u64,
    },
    ActivationUploadFeaturesMismatch {
        location: MeshBlockLocation,
        feature: CoverageFeature,
        expected: AcceptedFeatureSnapshot,
        actual: MeshBuildFeatures,
    },
}

#[derive(Debug, Clone, Copy)]
struct FixedCapacityFailpoint {
    destination: FixedCapacityDestination,
    remaining_occurrences: usize,
}

#[cfg(test)]
#[derive(Debug)]
struct FixedCommitPauseState {
    target: FixedCommitPausePhase,
    reached: bool,
    released: bool,
}

/// An external test handle for one armed fixed-LOD commit pause.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct FixedCommitPauseHandle {
    state: Arc<(Mutex<FixedCommitPauseState>, Condvar)>,
    commit_marker: Arc<AtomicBool>,
}

#[cfg(test)]
struct FixedStreamErrorRetirementProbe {
    dropped: Arc<AtomicBool>,
}

#[cfg(test)]
impl Drop for FixedStreamErrorRetirementProbe {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
#[allow(dead_code)]
impl FixedCommitPauseHandle {
    pub(crate) fn wait_until_reached(&self) {
        let (lock, ready) = &*self.state;
        let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !state.reached {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            assert!(
                !remaining.is_zero(),
                "fixed commit never reached the requested pause phase"
            );
            let (next, timeout) = ready
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
            assert!(
                state.reached || !timeout.timed_out(),
                "fixed commit never reached the requested pause phase"
            );
        }
    }

    pub(crate) fn release(&self) {
        let (lock, ready) = &*self.state;
        let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        state.released = true;
        ready.notify_all();
    }

    pub(crate) fn commit_marker(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.commit_marker)
    }

    fn pause_if_target(&self, phase: FixedCommitPausePhase) {
        let (lock, ready) = &*self.state;
        let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.target != phase {
            return;
        }
        state.reached = true;
        ready.notify_all();
        while !state.released {
            state = ready
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}

#[allow(dead_code)] // Constructed now; consumed when variable-LOD routing is activated.
struct VariableLodRuntimeState {
    settings: LodClipboxSettings,
    coordinator: ClipboxCoordinator,
    coverage: VariableLodCoverage,
    coverage_holds: CoverageHoldLedger,
}

/// Cloneable read-only access to storage owned by [`VoxelTerrainCore`].
///
/// The wrapped shared handle is intentionally private. This type does not
/// implement `Deref` and exposes no method that can recover the inner `Arc`,
/// so terrain residency accounting cannot be bypassed by public callers.
#[derive(Clone)]
pub struct VoxelTerrainDataView {
    data: Arc<SharedVoxelData>,
}

impl VoxelTerrainDataView {
    pub fn settings_snapshot(&self) -> SharedVoxelDataSettingsSnapshot {
        self.data.settings_snapshot()
    }

    pub fn bounds(&self) -> Box3i {
        self.data.bounds()
    }

    pub fn block_size(&self) -> u32 {
        self.data.block_size()
    }

    pub fn block_size_po2(&self) -> u8 {
        self.data.block_size_po2()
    }

    pub fn lod_count(&self) -> usize {
        self.data.lod_count()
    }

    pub fn block_snapshot(&self, block_pos: Vector3i, lod_index: usize) -> Option<VoxelDataBlock> {
        self.data.block_snapshot(block_pos, lod_index)
    }
}

/// Engine-agnostic paging terrain core (single- or multi-LOD).
pub struct VoxelTerrainCore {
    data: Arc<SharedVoxelData>,
    /// Number of LOD levels (1 = single-LOD backward compat).
    lod_count: u8,
    /// Per-LOD mesh block maps. Index 0 = LOD 0.
    mesh_maps: Vec<HashMap<Vector3i, MeshBlockEntry>>,
    /// Exact split residency for every loaded data block in every storage LOD.
    loaded_data_residency: Vec<HashMap<Vector3i, DataResidencyRefs>>,
    /// Validated variable-LOD ownership state used by the production planner.
    variable_lod: Option<VariableLodRuntimeState>,
    paired_viewers: Vec<PairedViewer>,
    /// Per-LOD pending load positions.
    blocks_pending_load: Vec<Vec<Vector3i>>,
    /// Per-LOD pending mesh-update positions.
    blocks_pending_update: Vec<Vec<Vector3i>>,
    /// Per-LOD loading state. Viewer demand and retry bookkeeping are kept in
    /// separate typed fields so retry attempts can never become block viewers.
    loading_blocks: Vec<HashMap<Vector3i, LoadingBlockEntry>>,
    /// Storage mutations that rolled back before commit are retried on the
    /// next viewer tick. Duplicates are intentional because each box carries
    /// one exact viewer-count delta.
    data_view_retries: Vec<Vec<PendingDataMutation>>,
    data_unview_retries: Vec<Vec<PendingDataMutation>>,
    last_data_mutation_error: Option<SharedVoxelDataMutationError>,
    /// Runtime incarnation of physical load/mesh requests. Shutdown advances
    /// it before cancellation so no completion from an older attempt can be
    /// accepted solely because a per-entry generation happens to match.
    request_epoch: u64,
    /// Shared checked allocator for physical load and mesh request identities.
    next_request_generation: u64,
    /// Monotonic allocator for the distinct shutdown lifecycle identity.
    next_shutdown_epoch: u64,
    /// Present from the atomic BeginShutdown commit onward. A failed flush
    /// keeps this value and leaves the core permanently retry-only.
    shutdown_epoch: Option<u64>,
    /// Exact storage-lineage authority retained from BeginShutdown through
    /// every persistence retry.
    shutdown_mutation_permit: Option<SharedVoxelDataShutdownMutationPermit>,
    save_journal: HashMap<SaveKey, SaveJournalEntry>,
    retained_save_admission_failures: VecDeque<RetainedSaveAdmissionFailure>,
    next_save_generation: u64,
    next_persistence_attempt_ordinal: u64,
    save_dispatch_error: Option<VoxelTerrainRuntimeError>,
    deferred_save_dispatch_keys: Vec<SaveKey>,
    last_save_checkpoint_error: Option<VoxelStreamError>,
    automatic_save_checkpoint_blocked: bool,
    save_checkpoint_in_flight: Option<SaveCheckpointInFlight>,
    next_save_checkpoint_generation: u64,
    last_checkpoint_outcome: Option<(u64, StreamResult<()>)>,
    deferred_checkpoint_dispatch: bool,
    force_checkpoint_requested: bool,
    automatic_checkpoint_satisfied_empty_flush: bool,
    next_mesh_revision: u64,
    next_render_topology_revision: u64,
    meshing_dependency: Arc<MeshingDependency>,
    stream: Arc<dyn VoxelStream>,
    task_runner: ThreadedTaskRunner,
    /// Lossless FIFO boundary between the runner and terrain-specific
    /// normalization. A failed destination reservation never consumes its
    /// front item.
    raw_completion_inbox: VecDeque<CompletedTask>,
    /// Terrain-owned, already-downcast completions. Payloads and original task
    /// boxes remain owned here until their matching state transition accepts
    /// them and follow-ups can be published.
    durable_completion_inbox: VecDeque<DurableCompletion>,
    /// Direct callers reserve this queue before surrendering their output.
    /// Admitted uploads remain here until one checked apply commits them.
    direct_mesh_retry_inbox: VecDeque<DurableCompletion>,
    /// Variable-LOD compatibility tasks whose runner ingress reservation
    /// failed. Fixed LOD never uses this quarantine.
    legacy_task_admission_retry: Vec<ScheduledTask>,
    /// Save/flush, unknown and malformed completions in their original global
    /// order. Quarantining advances the live FIFO but never discards a Box.
    completion_quarantine: VecDeque<QuarantinedCompletion>,
    shut_down: bool,
    shutdown_in_progress: bool,
    mesh_arrays_pool: Arc<MeshArraysPool>,
    pub max_view_distance_voxels: i32,
    pub automatic_loading_enabled: bool,
    pub stats: VoxelTerrainStats,
    /// When `true`, `try_process` for `lod_count > 1` uses the production
    /// `prepare_variable_physical_slice` planner. The constructor sets this
    /// to `true`; [`Self::use_legacy_variable_path_for_test`] is the
    /// emergency/test switch back to the three-stage route.
    variable_use_planner_path: bool,
    event_outbox: VecDeque<VoxelTerrainEvent>,
    #[cfg(test)]
    fail_next_completion_normalization_for_test: bool,
    #[cfg(test)]
    fail_next_follow_up_reservation_for_test: bool,
    #[cfg(test)]
    fail_next_direct_mesh_reservation_for_test: bool,
    #[cfg(test)]
    fail_next_mesh_event_reservation_for_test: bool,
    #[cfg(test)]
    fixed_after_prepare_data_conflict_for_test: Option<Vector3i>,
    #[cfg(test)]
    fixed_after_prepare_settings_conflict_for_test: bool,
    #[cfg(test)]
    variable_after_prepare_stale_coordinator_for_test: bool,
    #[cfg(test)]
    variable_after_prepare_stale_coverage_for_test: bool,
    #[cfg(test)]
    variable_after_prepare_edit_conflict_for_test: Option<Vector3i>,
    #[cfg(test)]
    edit_after_prepare_voxel_for_test: Option<(u64, Vector3i, usize)>,
    #[cfg(test)]
    panic_next_save_before_io_for_test: bool,
    #[cfg(test)]
    panic_next_flush_before_io_attempts_for_test: usize,
    #[cfg(test)]
    panic_next_flush_after_ack_for_test: bool,
    #[cfg(test)]
    fixed_commit_pause_for_test: Option<FixedCommitPauseHandle>,
    fixed_capacity_failpoint_for_test: Option<FixedCapacityFailpoint>,
    last_fixed_capacity_failure_for_test: Option<FixedCapacityDestination>,
    #[cfg(test)]
    fixed_task_count_bias_for_test: usize,
    #[cfg(test)]
    fixed_dirty_owner_probe_for_test: Option<(BlockLocation, Arc<AtomicUsize>)>,
    #[cfg(test)]
    fixed_stream_error_retirement_probe_for_test: Option<(usize, Arc<AtomicBool>)>,
}

impl VoxelTerrainCore {
    /// Build a new terrain core over the given `VoxelData`. The terrain
    /// shares ownership of the data (so mesh tasks running on the
    /// [`ThreadedTaskRunner`] can lock it), and stores the mesher/stream in
    /// a [`MeshingDependency`] so swapping either invalidates in-flight work.
    pub fn new(
        data: VoxelData,
        stream: Arc<dyn VoxelStream>,
        meshing_dependency: Arc<MeshingDependency>,
    ) -> Self {
        Self::new_common(data, stream, meshing_dependency, None)
    }

    pub fn new_variable_lod(
        mut data: VoxelData,
        stream: Arc<dyn VoxelStream>,
        meshing_dependency: Arc<MeshingDependency>,
        settings: LodClipboxSettings,
    ) -> Result<Self, VariableLodConstructionError> {
        settings
            .validate_for(&data)
            .map_err(VariableLodConstructionError::LodMath)?;
        if !meshing_dependency.mesher().supports_lod() {
            return Err(VariableLodConstructionError::UnsupportedLodMesher);
        }
        data.try_resize_lods_preserving(usize::from(settings.lod_count))
            .map_err(VariableLodConstructionError::LodResize)?;
        let coordinator = ClipboxCoordinator::new(settings, data.bounds())
            .map_err(VariableLodConstructionError::Coordinator)?;
        let coverage = VariableLodCoverage::try_new(settings.lod_count)
            .map_err(VariableLodConstructionError::Coverage)?;
        Ok(Self::new_common(
            data,
            stream,
            meshing_dependency,
            Some(VariableLodRuntimeState {
                settings,
                coordinator,
                coverage,
                coverage_holds: CoverageHoldLedger::default(),
            }),
        ))
    }

    /// Current clipbox settings for a Variable-LOD core.
    pub fn variable_lod_settings(&self) -> Option<LodClipboxSettings> {
        self.variable_lod.as_ref().map(|runtime| runtime.settings)
    }

    /// Rebuild clipboxes after inspector distances change. Viewer demand is
    /// recomputed on the next `try_process` tick.
    pub fn try_reconfigure_variable_clipboxes(
        &mut self,
        settings: LodClipboxSettings,
    ) -> Result<(), VariableLodConstructionError> {
        let Some(runtime) = self.variable_lod.as_mut() else {
            return Err(VariableLodConstructionError::UnsupportedLodMesher);
        };
        let block_size = i32::try_from(self.data.block_size())
            .map_err(|_| VariableLodConstructionError::LodMath(LodMathError::CoordinateOverflow))?;
        settings
            .validate_for_bounds(block_size, self.data.bounds())
            .map_err(VariableLodConstructionError::LodMath)?;
        let coordinator = ClipboxCoordinator::new(settings, self.data.bounds())
            .map_err(VariableLodConstructionError::Coordinator)?;
        runtime.settings = settings;
        runtime.coordinator = coordinator;
        Ok(())
    }

    /// Legacy variable-LOD MVP constructor retained only for the pinned parity
    /// fixture until C2c-E installs the clipbox runtime. Canonical fixed-LOD
    /// production must use [`Self::new`].
    #[doc(hidden)]
    pub fn legacy_variable_lod_for_parity(
        mut data: VoxelData,
        stream: Arc<dyn VoxelStream>,
        meshing_dependency: Arc<MeshingDependency>,
        lod_count: u8,
    ) -> Self {
        data.try_resize_lods_preserving(usize::from(lod_count))
            .expect("legacy parity LOD count must preserve existing maps");
        let block_size = i32::try_from(data.block_size())
            .expect("VoxelData block size must fit legacy clipbox settings");
        let settings = LodClipboxSettings {
            data_block_size: block_size,
            mesh_block_size: block_size,
            lod_count,
            lod0_distance_voxels: 0,
            secondary_distance_voxels: 0,
            unload_hysteresis_blocks: 0,
        };
        let coordinator = ClipboxCoordinator::new(settings, Box3i::default())
            .expect("legacy parity clipbox settings must be valid");
        let coverage = VariableLodCoverage::try_new(lod_count)
            .expect("legacy parity coverage LOD count must be valid");
        Self::new_common(
            data,
            stream,
            meshing_dependency,
            Some(VariableLodRuntimeState {
                settings,
                coordinator,
                coverage,
                coverage_holds: CoverageHoldLedger::default(),
            }),
        )
    }

    fn new_common(
        data: VoxelData,
        stream: Arc<dyn VoxelStream>,
        meshing_dependency: Arc<MeshingDependency>,
        variable_lod: Option<VariableLodRuntimeState>,
    ) -> Self {
        let data = Arc::new(SharedVoxelData::new(data));
        let lod_count = variable_lod
            .as_ref()
            .map_or(1, |runtime| runtime.settings.lod_count);
        let mut loaded_data_residency = Vec::with_capacity(data.lod_count());
        for lod_index in 0..data.lod_count() {
            let sidecar = data.with_lod_map(lod_index, |map| {
                let mut sidecar = HashMap::with_capacity(map.block_count());
                for position in map.block_positions() {
                    let viewers = map
                        .get_block(position)
                        .expect("resident position remains in its map")
                        .viewers
                        .get();
                    sidecar.insert(position, DataResidencyRefs::with_resident_viewers(viewers));
                }
                sidecar
            });
            loaded_data_residency.push(sidecar);
        }
        let task_runner = ThreadedTaskRunner::new(num_threads());
        let n = lod_count as usize;
        Self {
            data,
            lod_count,
            mesh_maps: (0..n).map(|_| HashMap::new()).collect(),
            loaded_data_residency,
            variable_lod,
            paired_viewers: Vec::new(),
            blocks_pending_load: (0..n).map(|_| Vec::new()).collect(),
            blocks_pending_update: (0..n).map(|_| Vec::new()).collect(),
            loading_blocks: (0..n).map(|_| HashMap::new()).collect(),
            data_view_retries: (0..n).map(|_| Vec::new()).collect(),
            data_unview_retries: (0..n).map(|_| Vec::new()).collect(),
            last_data_mutation_error: None,
            request_epoch: 0,
            next_request_generation: 1,
            next_shutdown_epoch: 0,
            shutdown_epoch: None,
            shutdown_mutation_permit: None,
            save_journal: HashMap::new(),
            retained_save_admission_failures: VecDeque::new(),
            next_save_generation: 1,
            next_persistence_attempt_ordinal: 1,
            save_dispatch_error: None,
            deferred_save_dispatch_keys: Vec::new(),
            last_save_checkpoint_error: None,
            automatic_save_checkpoint_blocked: false,
            save_checkpoint_in_flight: None,
            next_save_checkpoint_generation: 1,
            last_checkpoint_outcome: None,
            deferred_checkpoint_dispatch: false,
            force_checkpoint_requested: false,
            automatic_checkpoint_satisfied_empty_flush: false,
            next_mesh_revision: 1,
            next_render_topology_revision: 1,
            meshing_dependency,
            stream,
            task_runner,
            raw_completion_inbox: VecDeque::new(),
            durable_completion_inbox: VecDeque::new(),
            direct_mesh_retry_inbox: VecDeque::new(),
            legacy_task_admission_retry: Vec::new(),
            completion_quarantine: VecDeque::new(),
            shut_down: false,
            shutdown_in_progress: false,
            mesh_arrays_pool: Arc::new(MeshArraysPool::new()),
            max_view_distance_voxels: 192,
            automatic_loading_enabled: true,
            stats: VoxelTerrainStats::default(),
            // Planner is the production multi-LOD route. The legacy
            // three-stage path stays behind `use_legacy_variable_path_for_test`.
            variable_use_planner_path: true,
            event_outbox: VecDeque::new(),
            #[cfg(test)]
            fail_next_completion_normalization_for_test: false,
            #[cfg(test)]
            fail_next_follow_up_reservation_for_test: false,
            #[cfg(test)]
            fail_next_direct_mesh_reservation_for_test: false,
            #[cfg(test)]
            fail_next_mesh_event_reservation_for_test: false,
            #[cfg(test)]
            fixed_after_prepare_data_conflict_for_test: None,
            #[cfg(test)]
            fixed_after_prepare_settings_conflict_for_test: false,
            #[cfg(test)]
            variable_after_prepare_stale_coordinator_for_test: false,
            #[cfg(test)]
            variable_after_prepare_stale_coverage_for_test: false,
            #[cfg(test)]
            variable_after_prepare_edit_conflict_for_test: None,
            #[cfg(test)]
            edit_after_prepare_voxel_for_test: None,
            #[cfg(test)]
            panic_next_save_before_io_for_test: false,
            #[cfg(test)]
            panic_next_flush_before_io_attempts_for_test: 0,
            #[cfg(test)]
            panic_next_flush_after_ack_for_test: false,
            #[cfg(test)]
            fixed_commit_pause_for_test: None,
            fixed_capacity_failpoint_for_test: None,
            last_fixed_capacity_failure_for_test: None,
            #[cfg(test)]
            fixed_task_count_bias_for_test: 0,
            #[cfg(test)]
            fixed_dirty_owner_probe_for_test: None,
            #[cfg(test)]
            fixed_stream_error_retirement_probe_for_test: None,
        }
    }

    /// Convenience constructor for generator-only setups (no stream). A
    /// `MemoryStream` is used as a no-op sink; all loads fall back to the
    /// generator installed on `VoxelData`.
    pub fn new_generator_only(data: VoxelData, meshing_dependency: Arc<MeshingDependency>) -> Self {
        let stream: Arc<dyn VoxelStream> = Arc::new(MemoryStream::new());
        Self::new(data, stream, meshing_dependency)
    }

    /// Returns a cloneable read-only view of the terrain's shared storage.
    ///
    /// Mutation admission remains owned by the terrain core, so the public
    /// view deliberately has no forwarding mutation methods:
    ///
    /// ```compile_fail
    /// use voxel_core::{math::Box3i, terrain::VoxelTerrainCore};
    /// fn cannot_bypass_terrain(core: &VoxelTerrainCore) {
    ///     core.data()
    ///         .view_area(Box3i::default(), 0, None, None, None)
    ///         .unwrap();
    /// }
    /// ```
    ///
    /// Raw map guards are likewise not part of the read-only surface:
    ///
    /// ```compile_fail
    /// use voxel_core::terrain::VoxelTerrainCore;
    /// fn cannot_borrow_the_live_map(core: &VoxelTerrainCore) {
    ///     core.data().with_lod_map(0, |_| ());
    /// }
    /// ```
    pub fn data(&self) -> VoxelTerrainDataView {
        VoxelTerrainDataView {
            data: Arc::clone(&self.data),
        }
    }

    /// Prepares and publishes one voxel edit, its complete storage LOD
    /// cascade, and every viewed mesh invalidation as one transaction.
    pub fn try_edit_voxel(
        &mut self,
        value: u64,
        position: Vector3i,
        channel_index: usize,
    ) -> Result<Option<VoxelEditOutcome>, VoxelTerrainRuntimeError> {
        if self.shutdown_epoch.is_some() {
            return Err(VoxelTerrainRuntimeError::ShutdownRetryPending);
        }
        let Some(publication) =
            self.prepare_voxel_edit_publication(value, position, channel_index)?
        else {
            return Ok(None);
        };

        #[cfg(test)]
        if let Some((conflicting_value, conflicting_position, conflicting_channel)) =
            self.edit_after_prepare_voxel_for_test.take()
        {
            let _ = self.data.try_edit_voxel_checked(
                conflicting_value,
                conflicting_position,
                conflicting_channel,
            );
        }

        self.commit_prepared_voxel_edit(publication).map(Some)
    }

    /// Apply a shape edit to every LOD0 block overlapping the voxel AABB
    /// `[min, max]` (inclusive). One storage transaction is published per
    /// overlapping block instead of one per voxel.
    pub fn try_edit_sphere(
        &mut self,
        center: crate::math::Vector3f,
        radius: f32,
        channel_index: usize,
        mode: crate::edition::EditMode,
        value: u64,
    ) -> Result<u32, VoxelTerrainRuntimeError> {
        if !center.x.is_finite()
            || !center.y.is_finite()
            || !center.z.is_finite()
            || !radius.is_finite()
            || radius < 0.0
        {
            return Ok(0);
        }
        if channel_index >= crate::storage::voxel_buffer::MAX_CHANNELS {
            return Ok(0);
        }
        let min = Vector3i::new(
            (center.x - radius).floor() as i32,
            (center.y - radius).floor() as i32,
            (center.z - radius).floor() as i32,
        );
        let max = Vector3i::new(
            (center.x + radius).ceil() as i32,
            (center.y + radius).ceil() as i32,
            (center.z + radius).ceil() as i32,
        );
        self.try_edit_overlapping_blocks(min, max, |buffer, origin| {
            let local_center = crate::math::Vector3f::new(
                center.x - origin.x as f32,
                center.y - origin.y as f32,
                center.z - origin.z as f32,
            );
            crate::edition::do_sphere(buffer, channel_index, mode, value, local_center, radius);
        })
    }

    /// Apply a box edit to every LOD0 block overlapping `[min, max]`
    /// (inclusive). `do_box` uses an exclusive max, so this converts.
    pub fn try_edit_box(
        &mut self,
        min: Vector3i,
        max: Vector3i,
        channel_index: usize,
        mode: crate::edition::EditMode,
        value: u64,
    ) -> Result<u32, VoxelTerrainRuntimeError> {
        if channel_index >= crate::storage::voxel_buffer::MAX_CHANNELS {
            return Ok(0);
        }
        let lo = Vector3i::new(min.x.min(max.x), min.y.min(max.y), min.z.min(max.z));
        let hi = Vector3i::new(min.x.max(max.x), min.y.max(max.y), min.z.max(max.z));
        // Inclusive GDScript/tool contract → exclusive core `do_box`.
        let exclusive = Vector3i::new(
            hi.x.saturating_add(1),
            hi.y.saturating_add(1),
            hi.z.saturating_add(1),
        );
        self.try_edit_overlapping_blocks(lo, hi, |buffer, origin| {
            let local_min = Vector3i::new(lo.x - origin.x, lo.y - origin.y, lo.z - origin.z);
            let local_max = Vector3i::new(
                exclusive.x - origin.x,
                exclusive.y - origin.y,
                exclusive.z - origin.z,
            );
            crate::edition::do_box(buffer, channel_index, mode, value, local_min, local_max);
        })
    }

    /// Hemisphere brush: sphere cut by the plane whose outward normal is
    /// `flat_direction`. `smoothness` rounds the crease.
    #[allow(clippy::too_many_arguments)]
    pub fn try_edit_hemisphere(
        &mut self,
        center: crate::math::Vector3f,
        radius: f32,
        flat_direction: crate::math::Vector3f,
        smoothness: f32,
        channel_index: usize,
        mode: crate::edition::EditMode,
        value: u64,
    ) -> Result<u32, VoxelTerrainRuntimeError> {
        if !center.x.is_finite()
            || !center.y.is_finite()
            || !center.z.is_finite()
            || !radius.is_finite()
            || radius < 0.0
            || !flat_direction.x.is_finite()
            || !flat_direction.y.is_finite()
            || !flat_direction.z.is_finite()
            || !smoothness.is_finite()
            || smoothness < 0.0
        {
            return Ok(0);
        }
        if channel_index >= crate::storage::voxel_buffer::MAX_CHANNELS {
            return Ok(0);
        }
        let pad = radius + smoothness;
        let min = Vector3i::new(
            (center.x - pad).floor() as i32,
            (center.y - pad).floor() as i32,
            (center.z - pad).floor() as i32,
        );
        let max = Vector3i::new(
            (center.x + pad).ceil() as i32,
            (center.y + pad).ceil() as i32,
            (center.z + pad).ceil() as i32,
        );
        self.try_edit_overlapping_blocks(min, max, |buffer, origin| {
            let local_center = crate::math::Vector3f::new(
                center.x - origin.x as f32,
                center.y - origin.y as f32,
                center.z - origin.z as f32,
            );
            crate::edition::do_hemisphere(
                buffer,
                channel_index,
                mode,
                value,
                local_center,
                radius,
                flat_direction,
                smoothness,
            );
        })
    }

    /// Smooth the SDF channel inside a sphere of influence.
    pub fn try_edit_smooth(
        &mut self,
        center: crate::math::Vector3f,
        radius: f32,
        blur_radius: i32,
        channel_index: usize,
    ) -> Result<u32, VoxelTerrainRuntimeError> {
        if !center.x.is_finite()
            || !center.y.is_finite()
            || !center.z.is_finite()
            || !radius.is_finite()
            || radius < 0.0
            || blur_radius < 0
        {
            return Ok(0);
        }
        if channel_index >= crate::storage::voxel_buffer::MAX_CHANNELS {
            return Ok(0);
        }
        let min = Vector3i::new(
            (center.x - radius).floor() as i32,
            (center.y - radius).floor() as i32,
            (center.z - radius).floor() as i32,
        );
        let max = Vector3i::new(
            (center.x + radius).ceil() as i32,
            (center.y + radius).ceil() as i32,
            (center.z + radius).ceil() as i32,
        );
        self.try_edit_overlapping_blocks(min, max, |buffer, origin| {
            let local_center = crate::math::Vector3f::new(
                center.x - origin.x as f32,
                center.y - origin.y as f32,
                center.z - origin.z as f32,
            );
            crate::edition::do_smooth(buffer, channel_index, local_center, radius, blur_radius);
        })
    }

    fn try_edit_overlapping_blocks(
        &mut self,
        min: Vector3i,
        max: Vector3i,
        mut apply: impl FnMut(&mut crate::storage::VoxelBuffer, Vector3i),
    ) -> Result<u32, VoxelTerrainRuntimeError> {
        if self.shutdown_epoch.is_some() {
            return Err(VoxelTerrainRuntimeError::ShutdownRetryPending);
        }
        let block_size = self.data_block_size();
        if block_size <= 0 {
            return Ok(0);
        }
        let min_block = Vector3i::new(
            min.x.div_euclid(block_size),
            min.y.div_euclid(block_size),
            min.z.div_euclid(block_size),
        );
        let max_block = Vector3i::new(
            max.x.div_euclid(block_size),
            max.y.div_euclid(block_size),
            max.z.div_euclid(block_size),
        );
        let mut edited = 0u32;
        let mut z = min_block.z;
        while z <= max_block.z {
            let mut y = min_block.y;
            while y <= max_block.y {
                let mut x = min_block.x;
                while x <= max_block.x {
                    let block_pos = Vector3i::new(x, y, z);
                    if self
                        .try_edit_lod0_block(block_pos, |buffer, origin| apply(buffer, origin))?
                        .is_some()
                    {
                        edited = edited.saturating_add(1);
                    }
                    x = match x.checked_add(1) {
                        Some(next) => next,
                        None => break,
                    };
                }
                y = match y.checked_add(1) {
                    Some(next) => next,
                    None => break,
                };
            }
            z = match z.checked_add(1) {
                Some(next) => next,
                None => break,
            };
        }
        Ok(edited)
    }

    fn try_edit_lod0_block(
        &mut self,
        block_position: Vector3i,
        apply: impl FnOnce(&mut crate::storage::VoxelBuffer, Vector3i),
    ) -> Result<Option<VoxelEditOutcome>, VoxelTerrainRuntimeError> {
        if self.shutdown_epoch.is_some() {
            return Err(VoxelTerrainRuntimeError::ShutdownRetryPending);
        }
        let Some(prepared_edit) = self
            .data
            .prepare_lod0_block_edit(block_position, apply)
            .map_err(map_voxel_edit_storage_error)?
        else {
            return Ok(None);
        };
        let block_size = self.data_block_size();
        let origin = Vector3i::new(
            block_position.x.saturating_mul(block_size),
            block_position.y.saturating_mul(block_size),
            block_position.z.saturating_mul(block_size),
        );
        let far = Vector3i::new(
            origin.x.saturating_add(block_size.saturating_sub(1)),
            origin.y.saturating_add(block_size.saturating_sub(1)),
            origin.z.saturating_add(block_size.saturating_sub(1)),
        );
        let Some(publication) =
            self.finish_prepared_edit_publication(prepared_edit, origin, far)?
        else {
            return Ok(None);
        };
        self.commit_prepared_voxel_edit(publication).map(Some)
    }

    fn prepare_voxel_edit_publication(
        &mut self,
        value: u64,
        position: Vector3i,
        channel_index: usize,
    ) -> Result<Option<PreparedVoxelEditPublication>, VoxelTerrainRuntimeError> {
        let Some(prepared_edit) = self
            .data
            .prepare_voxel_edit(value, position, channel_index)
            .map_err(map_voxel_edit_storage_error)?
        else {
            return Ok(None);
        };
        self.finish_prepared_edit_publication(prepared_edit, position, position)
    }

    fn finish_prepared_edit_publication(
        &mut self,
        mut prepared_edit: crate::storage::voxel_data::PreparedSharedVoxelDataEdit,
        edit_min: Vector3i,
        edit_max: Vector3i,
    ) -> Result<Option<PreparedVoxelEditPublication>, VoxelTerrainRuntimeError> {
        let edited_block = prepared_edit.edited_block();
        let block_revision = prepared_edit.block_revision();
        let mut inserted_data_locations = Vec::new();
        inserted_data_locations
            .try_reserve_exact(prepared_edit.inserted_locations().len())
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        inserted_data_locations.extend_from_slice(prepared_edit.inserted_locations());
        let mut inserted_data_residencies = Vec::new();
        inserted_data_residencies
            .try_reserve_exact(inserted_data_locations.len())
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        let mut loading_generations_to_retire = Vec::new();
        loading_generations_to_retire
            .try_reserve_exact(inserted_data_locations.len())
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        let mut loading_positions_by_lod = Vec::new();
        loading_positions_by_lod
            .try_reserve_exact(self.loading_blocks.len())
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        for _ in 0..self.loading_blocks.len() {
            loading_positions_by_lod.push(Vec::new());
        }
        for location in &inserted_data_locations {
            let lod_index = usize::from(location.lod_index);
            if let Some(tracked) = self.loaded_data_residency[lod_index].get(&location.position) {
                return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                    location: *location,
                    tracked_resident_viewers: Some(tracked.resident_viewers),
                    tracked_coverage_holds: Some(tracked.coverage_holds),
                    storage_viewers: 0,
                });
            }
            self.loaded_data_residency[lod_index]
                .try_reserve(1)
                .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
            let residency = self.loading_blocks[lod_index]
                .get(&location.position)
                .map_or(DataResidencyRefs::default(), |loading| loading.residency);
            let final_viewers = residency.checked_total(*location)?;
            if !prepared_edit.set_insert_final_viewers(*location, final_viewers) {
                return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                    location: *location,
                    tracked_resident_viewers: Some(residency.resident_viewers),
                    tracked_coverage_holds: Some(residency.coverage_holds),
                    storage_viewers: 0,
                });
            }
            inserted_data_residencies.push((*location, residency));
            if let Some(loading) = self.loading_blocks[lod_index].get(&location.position) {
                loading_positions_by_lod[lod_index]
                    .try_reserve(1)
                    .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
                loading_positions_by_lod[lod_index].push(location.position);
                loading_generations_to_retire.push((*location, loading.request_generation));
            }
        }
        let mut pending_load_queue_replacements = Vec::new();
        pending_load_queue_replacements
            .try_reserve_exact(self.blocks_pending_load.len())
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        let loading_retirement_count = loading_generations_to_retire.len();
        let mut retired_loading_entries = Vec::new();
        retired_loading_entries
            .try_reserve_exact(loading_retirement_count)
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        let mut retired_pending_load_queues = Vec::new();
        retired_pending_load_queues
            .try_reserve_exact(
                loading_positions_by_lod
                    .iter()
                    .filter(|positions| !positions.is_empty())
                    .count(),
            )
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        for (lod_index, positions) in loading_positions_by_lod.iter().enumerate() {
            if positions.is_empty() {
                pending_load_queue_replacements.push(None);
                continue;
            }
            let pending = &self.blocks_pending_load[lod_index];
            let mut replacement = Vec::new();
            replacement
                .try_reserve_exact(pending.len())
                .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
            replacement.extend(
                pending
                    .iter()
                    .copied()
                    .filter(|pending_position| !positions.contains(pending_position)),
            );
            pending_load_queue_replacements.push(Some(replacement));
        }
        let mesher = self.meshing_dependency.mesher();
        let mut mesh_updates = Vec::new();
        let mut affected_mesh_blocks = Vec::new();
        let mut queue_additions = Vec::new();
        queue_additions
            .try_reserve_exact(self.mesh_maps.len())
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        for _ in 0..self.mesh_maps.len() {
            queue_additions.push(Vec::new());
        }

        let mut next_request_generation = self.next_request_generation;
        let mut next_mesh_revision = self.next_mesh_revision;
        for (lod_index, queue_additions) in queue_additions.iter_mut().enumerate() {
            let (minimum, maximum, candidate_count) = checked_edit_mesh_block_bounds_span(
                edit_min,
                edit_max,
                mesher.minimum_padding(),
                mesher.maximum_padding(),
                self.data_block_size(),
                lod_index,
            )?;
            #[cfg(test)]
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::MeshMap)
                .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
            mesh_updates
                .try_reserve(candidate_count)
                .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
            affected_mesh_blocks
                .try_reserve(candidate_count)
                .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
            queue_additions
                .try_reserve(candidate_count)
                .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;

            for x in minimum.x..=maximum.x {
                for y in minimum.y..=maximum.y {
                    for z in minimum.z..=maximum.z {
                        let block_position = Vector3i::new(x, y, z);
                        let Some(entry) = self.mesh_maps[lod_index].get(&block_position) else {
                            continue;
                        };
                        if entry.resident_refcount() == 0 {
                            continue;
                        }
                        let revision = next_mesh_revision;
                        next_mesh_revision = revision
                            .checked_add(1)
                            .ok_or(VoxelTerrainRuntimeError::MeshRevisionOverflow)?;
                        let location = MeshBlockLocation::new(block_position, lod_index as u8);
                        let mut next_entry = entry.clone_for_draft();
                        next_entry.requested_revision = Some(revision);
                        next_entry.terminal_retry_count = 0;
                        let (generation, physical_request) = allocate_physical_request(
                            self.request_epoch,
                            &mut next_request_generation,
                        )?;
                        next_entry.request_generation = generation;
                        next_entry.physical_request = Some(physical_request);
                        if !next_entry.is_in_update_list {
                            next_entry.is_in_update_list = true;
                            queue_additions.push(block_position);
                        }
                        mesh_updates.push(PreparedVoxelEditMeshUpdate {
                            location,
                            next_entry,
                        });
                        affected_mesh_blocks.push(location);
                    }
                }
            }
        }

        let mut pending_queue_replacements = Vec::new();
        pending_queue_replacements
            .try_reserve_exact(self.blocks_pending_update.len())
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        let replacement_count = queue_additions
            .iter()
            .filter(|additions| !additions.is_empty())
            .count();
        let mut retired_pending_queues = Vec::new();
        retired_pending_queues
            .try_reserve_exact(replacement_count)
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        for (lod_index, additions) in queue_additions.into_iter().enumerate() {
            if additions.is_empty() {
                pending_queue_replacements.push(None);
                continue;
            }
            #[cfg(test)]
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::PendingMeshQueue)
                .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
            let final_len = self.blocks_pending_update[lod_index]
                .len()
                .checked_add(additions.len())
                .ok_or(VoxelTerrainRuntimeError::CapacityReservationFailed)?;
            let mut replacement = Vec::new();
            replacement
                .try_reserve_exact(final_len)
                .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
            replacement.extend(self.blocks_pending_update[lod_index].iter().copied());
            replacement.extend(additions);
            pending_queue_replacements.push(Some(replacement));
        }

        #[cfg(test)]
        self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::Retirement)
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        let mut retired_mesh_entries = Vec::new();
        retired_mesh_entries
            .try_reserve_exact(mesh_updates.len())
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;

        Ok(Some(PreparedVoxelEditPublication {
            data_transaction: prepared_edit.into_transaction(),
            edited_block,
            block_revision,
            inserted_data_residencies,
            loading_generations_to_retire,
            pending_load_queue_replacements,
            mesh_updates,
            pending_queue_replacements,
            next_request_generation,
            next_mesh_revision,
            affected_mesh_blocks,
            retired_mesh_entries,
            retired_loading_entries,
            retired_pending_load_queues,
            retired_pending_queues,
        }))
    }

    fn commit_prepared_voxel_edit(
        &mut self,
        publication: PreparedVoxelEditPublication,
    ) -> Result<VoxelEditOutcome, VoxelTerrainRuntimeError> {
        #[cfg(test)]
        if let Some(pause) = &self.fixed_commit_pause_for_test {
            pause.commit_marker.store(false, Ordering::SeqCst);
        }
        let PreparedVoxelEditPublication {
            mut data_transaction,
            edited_block,
            block_revision,
            inserted_data_residencies,
            loading_generations_to_retire,
            mut pending_load_queue_replacements,
            mesh_updates,
            mut pending_queue_replacements,
            next_request_generation,
            next_mesh_revision,
            affected_mesh_blocks,
            mut retired_mesh_entries,
            mut retired_loading_entries,
            mut retired_pending_load_queues,
            mut retired_pending_queues,
        } = publication;
        let fence = data_transaction
            .commit_holding_publication_fence()
            .map_err(map_voxel_edit_storage_error)?;
        #[cfg(test)]
        if let Some(pause) = &self.fixed_commit_pause_for_test {
            pause.pause_if_target(FixedCommitPausePhase::StorageFencedBeforeCorePublish);
        }

        for (location, residency) in inserted_data_residencies {
            let previous = self.loaded_data_residency[usize::from(location.lod_index)]
                .insert(location.position, residency);
            debug_assert!(previous.is_none());
        }
        for (location, expected_generation) in loading_generations_to_retire {
            let retired = self.loading_blocks[usize::from(location.lod_index)]
                .remove(&location.position)
                .expect("prepared edit loading owner remains until publication");
            debug_assert_eq!(retired.request_generation, expected_generation);
            retired.cancel_physical_request_if_superseded_by(None);
            retired_loading_entries.push(retired);
        }
        for (lod_index, replacement) in pending_load_queue_replacements.iter_mut().enumerate() {
            if let Some(replacement) = replacement.take() {
                retired_pending_load_queues.push(std::mem::replace(
                    &mut self.blocks_pending_load[lod_index],
                    replacement,
                ));
            }
        }
        for update in mesh_updates {
            let lod_index = usize::from(update.location.lod_index);
            let resident = self.mesh_maps[lod_index]
                .get_mut(&update.location.position_in_blocks)
                .expect("prepared edit mesh entry remains resident until publication");
            let retired = std::mem::replace(resident, update.next_entry);
            retired.cancel_physical_request_if_superseded_by(
                self.mesh_maps[lod_index][&update.location.position_in_blocks]
                    .physical_request
                    .as_ref(),
            );
            retired_mesh_entries.push(retired);
        }
        for (lod_index, replacement) in pending_queue_replacements.iter_mut().enumerate() {
            if let Some(replacement) = replacement.take() {
                retired_pending_queues.push(std::mem::replace(
                    &mut self.blocks_pending_update[lod_index],
                    replacement,
                ));
            }
        }
        self.next_request_generation = next_request_generation;
        self.next_mesh_revision = next_mesh_revision;

        #[cfg(test)]
        if let Some(pause) = &self.fixed_commit_pause_for_test {
            pause.commit_marker.store(true, Ordering::SeqCst);
            pause.pause_if_target(FixedCommitPausePhase::AfterTerrainPublishBeforeFenceFinish);
        }
        let storage_outcome = fence.finish();
        let outcome = VoxelEditOutcome {
            edited_block,
            block_revision,
            affected_mesh_blocks,
        };
        drop((
            storage_outcome,
            retired_mesh_entries,
            retired_loading_entries,
            retired_pending_load_queues,
            pending_load_queue_replacements,
            retired_pending_queues,
            pending_queue_replacements,
            data_transaction,
        ));
        Ok(outcome)
    }

    /// Returns the data block size (in voxels) used by the underlying
    /// `VoxelData`. The current port assumes mesh block size == data block
    /// size (factor 1).
    fn data_block_size(&self) -> i32 {
        // Matches the C++ inline `get_data_block_size()`.
        self.data.block_size() as i32
    }

    /// Validates one explicitly captured storage key against the terrain's
    /// split residency sidecar. Callers intentionally supply the captured key
    /// set, so persistence-only work never pays for an unrelated full-map
    /// scan.
    fn validate_loaded_data_residency_snapshot(
        &self,
        snapshot: &SharedVoxelDataTransactionBlockSnapshot,
    ) -> Result<DataResidencyRefs, VoxelTerrainRuntimeError> {
        let location = snapshot.location();
        let tracked = self
            .loaded_data_residency
            .get(usize::from(location.lod_index))
            .and_then(|sidecar| sidecar.get(&location.position))
            .copied();
        let storage_viewers = if snapshot.is_present() {
            snapshot.viewers()
        } else {
            0
        };
        let Some(tracked) = tracked.filter(|_| snapshot.is_present()) else {
            return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                location,
                tracked_resident_viewers: tracked.map(|refs| refs.resident_viewers),
                tracked_coverage_holds: tracked.map(|refs| refs.coverage_holds),
                storage_viewers,
            });
        };
        if tracked.checked_total(location)? != storage_viewers {
            return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                location,
                tracked_resident_viewers: Some(tracked.resident_viewers),
                tracked_coverage_holds: Some(tracked.coverage_holds),
                storage_viewers,
            });
        }
        Ok(tracked)
    }

    /// Returns whether every in-bound data block required to mesh `location`
    /// is resident. Finite world bounds clip only the out-of-bound side of the
    /// ordinary one-block halo; every in-bound neighbour remains mandatory.
    pub fn meshing_data_is_ready(&self, location: MeshBlockLocation) -> Result<bool, LodMathError> {
        let data_box = self.meshing_data_box(location)?;
        if data_box.is_empty() {
            return Ok(false);
        }
        Ok(self
            .data
            .has_all_blocks_in_area(data_box, usize::from(location.lod_index)))
    }

    /// Exact finite-boundary data halo used by readiness and Task 6's
    /// join-target hold acquisition.
    pub(crate) fn meshing_data_box(
        &self,
        location: MeshBlockLocation,
    ) -> Result<Box3i, LodMathError> {
        if usize::from(location.lod_index) >= self.data.lod_count() {
            return Err(LodMathError::InvalidLodCount);
        }
        clipped_meshing_data_box(location, self.data_block_size(), 1, self.data.bounds())
    }

    fn validate_meshing_bounds(&self) -> Result<(), LodMathError> {
        let bounds = self.data.bounds();
        let block_size = self.data_block_size();
        for lod_index in 0..self.lod_count {
            let lod_bounds = bounds_in_lod_blocks(bounds, block_size, lod_index)?;
            if lod_bounds.is_empty() {
                continue;
            }
            let last = Vector3i::new(
                lod_bounds
                    .position
                    .x
                    .checked_add(lod_bounds.size.x - 1)
                    .ok_or(LodMathError::CoordinateOverflow)?,
                lod_bounds
                    .position
                    .y
                    .checked_add(lod_bounds.size.y - 1)
                    .ok_or(LodMathError::CoordinateOverflow)?,
                lod_bounds
                    .position
                    .z
                    .checked_add(lod_bounds.size.z - 1)
                    .ok_or(LodMathError::CoordinateOverflow)?,
            );
            // The helper additionally validates the LOD-scaled voxel extent
            // and padded halo. Checking both extreme corners proves every
            // interior block representable before any variable-LOD mutation.
            for position in [lod_bounds.position, last] {
                clipped_meshing_data_box(
                    MeshBlockLocation::new(position, lod_index),
                    block_size,
                    1,
                    bounds,
                )?;
            }
        }
        Ok(())
    }

    /// Read-only diagnostic view of the LOD-0 mesh-block state. Runtime upload
    /// consumers must use the snapshot owned by each lifecycle event instead
    /// of querying this map.
    pub fn mesh_blocks(&self) -> &HashMap<Vector3i, MeshBlockEntry> {
        &self.mesh_maps[0]
    }

    /// Reference to the mesh-block hashmap for a specific LOD.
    pub fn mesh_blocks_at_lod(&self, lod: u8) -> &HashMap<Vector3i, MeshBlockEntry> {
        &self.mesh_maps[lod as usize]
    }

    /// Number of LOD levels.
    pub fn lod_count(&self) -> u8 {
        self.lod_count
    }

    /// Cumulative terrain statistics (blocks loaded/unloaded, meshes built/dropped).
    /// Mirrors the C++ `_stats` snapshot exposed via `VoxelTerrain::get_statistics`.
    pub fn stats(&self) -> &VoxelTerrainStats {
        &self.stats
    }

    /// Number of background tasks still queued, running, or awaiting a checked
    /// terrain commit. Returns 0 once all pending load/mesh work has completed
    /// and every finished output has left the terrain-owned retry FIFOs. Tests
    /// and the Godot binding use this to wait for paging convergence before
    /// asserting on mesh output.
    pub fn pending_task_count(&self) -> usize {
        self.task_runner
            .remaining_task_count()
            .saturating_add(self.raw_completion_inbox.len())
            .saturating_add(self.durable_completion_inbox.len())
            .saturating_add(self.direct_mesh_retry_inbox.len())
            .saturating_add(self.legacy_task_admission_retry.len())
    }

    /// Block until every queued background task has finished, then return. Use
    /// after a batch of [`try_process`](Self::try_process) ticks to reach a stable,
    /// deterministic state (all mesh outputs applied) before asserting on
    /// vertex counts or block contents.
    pub fn wait_for_pending_tasks(&mut self) {
        if self.lod_count > 1 && !self.legacy_task_admission_retry.is_empty() {
            self.legacy_link_or_retain_task_batch(Vec::new());
        }
        self.task_runner.wait_for_all_tasks();
    }

    fn prepare_save_completion_action(
        shadow: &mut HashMap<SaveKey, PreparedJournalShadow>,
        terminal: &SaveTaskTerminal,
        attempt_ordinal: u64,
    ) -> Result<Option<PreparedSaveAction>, VoxelTerrainRuntimeError> {
        let key = SaveKey::new(terminal.location.position, terminal.location.lod_index);
        let Some(entry) = shadow.get_mut(&key) else {
            return Ok(None);
        };
        let PreparedJournalActiveShadow::WriteInFlight {
            block_revision,
            generation,
            attempt_ordinal: current_attempt,
            retry_count,
        } = entry.active
        else {
            return Ok(None);
        };
        if block_revision != terminal.block_revision
            || generation != terminal.save_generation
            || current_attempt != attempt_ordinal
        {
            return Ok(None);
        }
        debug_assert_eq!(
            entry.active,
            PreparedJournalActiveShadow::WriteInFlight {
                block_revision: terminal.block_revision,
                generation: terminal.save_generation,
                attempt_ordinal,
                retry_count,
            }
        );
        let common = (
            key,
            terminal.block_revision,
            terminal.save_generation,
            attempt_ordinal,
        );
        let action = match (terminal.phase, terminal.acknowledgement.as_ref()) {
            (PersistenceIoPhase::Acknowledged, Some(PersistenceAcknowledgement::Save(Ok(()))))
                if entry.written_generation.is_none() =>
            {
                entry.written_generation = Some(terminal.save_generation);
                entry.written_block_revision = Some(terminal.block_revision);
                entry.active = PreparedJournalActiveShadow::None;
                Some(PreparedSaveAction::AcknowledgeSuccess {
                    key: common.0,
                    block_revision: common.1,
                    generation: common.2,
                    attempt_ordinal: common.3,
                })
            }
            (PersistenceIoPhase::Acknowledged, Some(PersistenceAcknowledgement::Save(Err(_)))) => {
                let next_retry_count = retry_count.checked_add(1).ok_or(
                    VoxelTerrainRuntimeError::PersistenceRetryCountOverflow {
                        operation: PersistenceOperation::Save {
                            location: terminal.location,
                            block_revision: terminal.block_revision,
                            save_generation: terminal.save_generation,
                        },
                    },
                )?;
                entry.active = PreparedJournalActiveShadow::Pending {
                    block_revision: terminal.block_revision,
                    generation: terminal.save_generation,
                    retry_count: next_retry_count,
                };
                Some(PreparedSaveAction::AcknowledgeFailure {
                    key: common.0,
                    block_revision: common.1,
                    generation: common.2,
                    attempt_ordinal: common.3,
                    next_retry_count,
                })
            }
            (PersistenceIoPhase::BeforeIo, None) => {
                let next_retry_count = retry_count.checked_add(1).ok_or(
                    VoxelTerrainRuntimeError::PersistenceRetryCountOverflow {
                        operation: PersistenceOperation::Save {
                            location: terminal.location,
                            block_revision: terminal.block_revision,
                            save_generation: terminal.save_generation,
                        },
                    },
                )?;
                entry.active = PreparedJournalActiveShadow::Pending {
                    block_revision: terminal.block_revision,
                    generation: terminal.save_generation,
                    retry_count: next_retry_count,
                };
                Some(PreparedSaveAction::RestoreBeforeIo {
                    key: common.0,
                    block_revision: common.1,
                    generation: common.2,
                    attempt_ordinal: common.3,
                    next_retry_count,
                })
            }
            (PersistenceIoPhase::CallEntered, None) => {
                entry.active = PreparedJournalActiveShadow::Indeterminate {
                    block_revision: terminal.block_revision,
                    generation: terminal.save_generation,
                    attempt_ordinal,
                };
                Some(PreparedSaveAction::MarkIndeterminate {
                    key: common.0,
                    block_revision: common.1,
                    generation: common.2,
                    attempt_ordinal: common.3,
                })
            }
            _ => None,
        };
        Ok(action)
    }

    fn prepare_checkpoint_completion_action(
        &self,
        checkpoint_shadow: &mut Option<PreparedCheckpointShadow>,
        journal_shadow: &mut HashMap<SaveKey, PreparedJournalShadow>,
        terminal: &FlushTaskTerminal,
        attempt_ordinal: u64,
    ) -> Result<Option<PreparedCheckpointAction>, VoxelTerrainRuntimeError> {
        let Some(shadow) = checkpoint_shadow.as_mut() else {
            return Ok(None);
        };
        if shadow.checkpoint_generation != terminal.checkpoint_generation
            || shadow.state != (CheckpointAttemptState::WriteInFlight { attempt_ordinal })
        {
            return Ok(None);
        }

        match (terminal.phase, terminal.acknowledgement.as_ref()) {
            (PersistenceIoPhase::BeforeIo, None) => {
                shadow.retry_count = shadow.retry_count.checked_add(1).ok_or(
                    VoxelTerrainRuntimeError::PersistenceRetryCountOverflow {
                        operation: PersistenceOperation::Flush {
                            checkpoint_generation: terminal.checkpoint_generation,
                        },
                    },
                )?;
                shadow.state = CheckpointAttemptState::Pending;
                Ok(Some(PreparedCheckpointAction::RestoreBeforeIo {
                    checkpoint_generation: terminal.checkpoint_generation,
                    attempt_ordinal,
                    next_retry_count: shadow.retry_count,
                }))
            }
            (PersistenceIoPhase::CallEntered, None) => {
                shadow.state = CheckpointAttemptState::Indeterminate { attempt_ordinal };
                Ok(Some(PreparedCheckpointAction::MarkIndeterminate {
                    checkpoint_generation: terminal.checkpoint_generation,
                    attempt_ordinal,
                }))
            }
            (PersistenceIoPhase::Acknowledged, Some(PersistenceAcknowledgement::Flush(result))) => {
                let Some(checkpoint) = self.save_checkpoint_in_flight.as_ref() else {
                    return Ok(None);
                };
                let succeeded = result.is_ok();
                let explicit_outcome = if shadow.origin == CheckpointOrigin::Explicit {
                    Some(try_clone_stream_result(result)?)
                } else {
                    None
                };
                let mut entry_actions = Vec::new();
                entry_actions
                    .try_reserve_exact(checkpoint.acknowledged.len())
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                for snapshot in &checkpoint.acknowledged {
                    let Some(entry) = journal_shadow.get_mut(&snapshot.key) else {
                        continue;
                    };
                    if entry.written_block_revision != Some(snapshot.block_revision)
                        || entry.written_generation != Some(snapshot.generation)
                    {
                        continue;
                    }
                    entry.written_block_revision = None;
                    entry.written_generation = None;
                    if succeeded {
                        let promote_queued = entry.active == PreparedJournalActiveShadow::None
                            && entry.queued_len != 0;
                        if promote_queued {
                            entry.queued_len -= 1;
                            let (block_revision, generation, retry_count) =
                                entry.queued_front.unwrap_or_else(|| {
                                    unreachable!("prepared queued save retains front metadata")
                                });
                            entry.active = PreparedJournalActiveShadow::Pending {
                                block_revision,
                                generation,
                                retry_count,
                            };
                        }
                        let defer_active =
                            matches!(entry.active, PreparedJournalActiveShadow::Pending { .. });
                        let remove_entry = entry.active == PreparedJournalActiveShadow::None
                            && entry.queued_len == 0;
                        entry_actions.push(PreparedCheckpointEntryAction::ClearWritten {
                            key: snapshot.key,
                            block_revision: snapshot.block_revision,
                            generation: snapshot.generation,
                            promote_queued,
                            defer_active,
                            remove_entry,
                        });
                    } else {
                        let restored_retry_count = u32::from(shadow.record_per_block_failure);
                        let placement = match entry.active {
                            PreparedJournalActiveShadow::None => {
                                entry.active = PreparedJournalActiveShadow::Pending {
                                    block_revision: snapshot.block_revision,
                                    generation: snapshot.generation,
                                    retry_count: restored_retry_count,
                                };
                                PreparedCheckpointRestorePlacement::Active
                            }
                            PreparedJournalActiveShadow::Pending {
                                block_revision,
                                generation,
                                retry_count,
                            } => {
                                entry.queued_len = entry
                                    .queued_len
                                    .checked_add(1)
                                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                                entry.queued_front =
                                    Some((block_revision, generation, retry_count));
                                entry.active = PreparedJournalActiveShadow::Pending {
                                    block_revision: snapshot.block_revision,
                                    generation: snapshot.generation,
                                    retry_count: restored_retry_count,
                                };
                                PreparedCheckpointRestorePlacement::ReplacePending
                            }
                            PreparedJournalActiveShadow::WriteInFlight { .. }
                            | PreparedJournalActiveShadow::Indeterminate { .. } => {
                                entry.queued_len = entry
                                    .queued_len
                                    .checked_add(1)
                                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                                entry.queued_front = Some((
                                    snapshot.block_revision,
                                    snapshot.generation,
                                    restored_retry_count,
                                ));
                                PreparedCheckpointRestorePlacement::Queue
                            }
                        };
                        let error = if shadow.record_per_block_failure {
                            let Err(error) = result else {
                                unreachable!("failed checkpoint action has an error")
                            };
                            Some(try_clone_stream_error(error)?)
                        } else {
                            None
                        };
                        entry_actions.push(PreparedCheckpointEntryAction::RestoreWritten {
                            key: snapshot.key,
                            block_revision: snapshot.block_revision,
                            generation: snapshot.generation,
                            placement,
                            error,
                        });
                    }
                }
                let action = PreparedCheckpointAction::Acknowledge {
                    checkpoint_generation: terminal.checkpoint_generation,
                    attempt_ordinal,
                    succeeded,
                    origin: shadow.origin,
                    explicit_outcome,
                    entry_actions,
                };
                *checkpoint_shadow = None;
                Ok(Some(action))
            }
            _ => Ok(None),
        }
    }

    fn fixed_draft_mesh_data_is_ready(
        &self,
        data_preview: &crate::storage::voxel_data::SharedVoxelDataTransactionPreview,
        mesh_position: Vector3i,
        data_presence: &mut HashMap<Vector3i, bool>,
        data_snapshots: &mut Vec<SharedVoxelDataTransactionBlockSnapshot>,
    ) -> Result<bool, VoxelTerrainRuntimeError> {
        let data_box = clipped_meshing_data_box(
            MeshBlockLocation::new(mesh_position, 0),
            i32::try_from(data_preview.block_size())
                .map_err(|_| VoxelTerrainRuntimeError::CoordinateOverflow)?,
            1,
            data_preview.settings().bounds_in_voxels,
        )
        .map_err(VoxelTerrainRuntimeError::from)?;
        if data_box.is_empty() {
            return Ok(false);
        }
        data_presence
            .try_reserve(27)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        data_snapshots
            .try_reserve(27)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for position in data_box.iter_cells_zxy() {
            let location = BlockLocation {
                position,
                lod_index: 0,
            };
            let present = if let Some(present) = data_presence.get(&position) {
                *present
            } else {
                let snapshot = data_preview
                    .block_snapshot(location)
                    .expect("fixed LOD halo location is valid");
                data_snapshots.push(snapshot.clone());
                data_presence.insert(position, snapshot.is_present());
                snapshot.is_present()
            };
            if !present {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn prepare_fixed_viewer_transaction(
        &mut self,
        viewers: &[ViewerUpdate],
        replace_viewers: bool,
        dispatch_load_mesh_tasks: bool,
        dispatch_persistence_tasks: bool,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        self.prepare_fixed_viewer_transaction_with_checkpoint(
            viewers,
            replace_viewers,
            dispatch_load_mesh_tasks,
            dispatch_persistence_tasks,
            None,
            None,
            None,
            false,
        )
    }

    // These are independent transaction-boundary controls. Keeping them
    // explicit avoids a broad mechanical redesign of the ownership-sensitive
    // C1/C2 entry point while preserving typed, allocation-free inputs.
    #[allow(clippy::too_many_arguments)]
    fn prepare_fixed_viewer_transaction_with_checkpoint(
        &mut self,
        viewers: &[ViewerUpdate],
        replace_viewers: bool,
        dispatch_load_mesh_tasks: bool,
        dispatch_persistence_tasks: bool,
        checkpoint_request: Option<FixedCheckpointRequest>,
        persistence_recovery_request: Option<FixedPersistenceRecoveryRequest>,
        persistence_resolution_request: Option<FixedPersistenceResolutionRequest>,
        capture_resident_dirty: bool,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        self.prepare_fixed_viewer_transaction_with_checkpoint_and_admission(
            viewers,
            replace_viewers,
            dispatch_load_mesh_tasks,
            dispatch_persistence_tasks,
            checkpoint_request,
            persistence_recovery_request,
            persistence_resolution_request,
            capture_resident_dirty,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_fixed_viewer_transaction_with_checkpoint_and_admission(
        &mut self,
        viewers: &[ViewerUpdate],
        replace_viewers: bool,
        dispatch_load_mesh_tasks: bool,
        dispatch_persistence_tasks: bool,
        checkpoint_request: Option<FixedCheckpointRequest>,
        persistence_recovery_request: Option<FixedPersistenceRecoveryRequest>,
        persistence_resolution_request: Option<FixedPersistenceResolutionRequest>,
        capture_resident_dirty: bool,
        shutdown_owned_capture: bool,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        debug_assert_eq!(self.lod_count, 1);
        if shutdown_owned_capture {
            let permit = self.shutdown_mutation_permit.as_ref().ok_or(
                VoxelTerrainRuntimeError::DataMutation(
                    SharedVoxelDataMutationError::MutationAdmissionClosed,
                ),
            )?;
            self.data
                .validate_shutdown_mutation_permit(permit)
                .map_err(VoxelTerrainRuntimeError::DataMutation)?;
        }
        let data_preview = self.data.begin_transaction_preview();
        let (resident_operations, resident_snapshots, mut resident_payloads) =
            if capture_resident_dirty {
                data_preview
                    .prepare_resident_dirty_copies()
                    .map_err(VoxelTerrainRuntimeError::DataMutation)?
                    .into_parts()
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
        for snapshot in &resident_snapshots {
            self.validate_loaded_data_residency_snapshot(snapshot)?;
        }
        let data_block_size = data_preview.block_size() as i32;
        let mut next_paired_viewers = Vec::new();
        if replace_viewers {
            #[cfg(test)]
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::PairedViewers)?;
            next_paired_viewers
                .try_reserve_exact(viewers.len())
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        }
        let mut data_deltas = HashMap::<Vector3i, i64>::new();
        let mut mesh_deltas = HashMap::<Vector3i, (i64, i64, i64)>::new();
        let consume_data_retries = (replace_viewers || dispatch_load_mesh_tasks)
            && (!self.data_view_retries[0].is_empty() || !self.data_unview_retries[0].is_empty());
        #[cfg(test)]
        if consume_data_retries && !self.data_unview_retries[0].is_empty() {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::DataUnviewRetries)?;
        }
        for pending in self.data_unview_retries[0]
            .iter()
            .filter(|_| consume_data_retries)
        {
            let count = usize::try_from(pending.box_in_blocks.size.volume_u64())
                .map_err(|_| VoxelTerrainRuntimeError::TaskCountOverflow)?;
            data_deltas
                .try_reserve(count)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            for position in pending.box_in_blocks.iter_cells_zxy() {
                let delta = data_deltas.entry(position).or_default();
                *delta = delta
                    .checked_sub(1)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
            }
        }
        #[cfg(test)]
        if consume_data_retries && !self.data_view_retries[0].is_empty() {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::DataViewRetries)?;
        }
        for pending in self.data_view_retries[0]
            .iter()
            .filter(|_| consume_data_retries)
        {
            let count = usize::try_from(pending.box_in_blocks.size.volume_u64())
                .map_err(|_| VoxelTerrainRuntimeError::TaskCountOverflow)?;
            data_deltas
                .try_reserve(count)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            for position in pending.box_in_blocks.iter_cells_zxy() {
                let delta = data_deltas.entry(position).or_default();
                *delta = delta
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
            }
        }
        let fixed_state_copy = |state: &ViewerState| ViewerState {
            local_position_voxels: state.local_position_voxels,
            data_box: state.data_box,
            mesh_box: state.mesh_box,
            data_box_per_lod: Vec::new(),
            mesh_box_per_lod: Vec::new(),
            horizontal_view_distance_voxels: state.horizontal_view_distance_voxels,
            vertical_view_distance_voxels: state.vertical_view_distance_voxels,
            demand: state.demand,
        };
        let mut current_index = 0usize;
        let mut desired_index = 0usize;
        while replace_viewers
            && (current_index < self.paired_viewers.len() || desired_index < viewers.len())
        {
            let current = self.paired_viewers.get(current_index);
            let desired = viewers.get(desired_index);
            let take_current = match (current, desired) {
                (Some(current), Some(desired)) => current.id <= desired.id,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => unreachable!("merge loop has one remaining viewer"),
            };
            let take_desired = match (current, desired) {
                (Some(current), Some(desired)) => desired.id <= current.id,
                (None, Some(_)) => true,
                (Some(_), None) => false,
                (None, None) => unreachable!("merge loop has one remaining viewer"),
            };
            let old_state = if take_current {
                let state = fixed_state_copy(
                    &current
                        .unwrap_or_else(|| unreachable!("current merge item remains present"))
                        .state,
                );
                current_index += 1;
                state
            } else {
                ViewerState::default()
            };
            let mut new_state = ViewerState::default();
            if take_desired {
                let update = desired
                    .copied()
                    .unwrap_or_else(|| unreachable!("desired merge item remains present"));
                desired_index += 1;
                new_state.local_position_voxels = update.world_position_voxels;
                new_state.horizontal_view_distance_voxels = update
                    .horizontal_view_distance_voxels
                    .min(self.max_view_distance_voxels);
                new_state.vertical_view_distance_voxels = update
                    .vertical_view_distance_voxels
                    .min(self.max_view_distance_voxels);
                new_state.demand = update.demand;
                compute_viewer_boxes(&mut new_state, data_block_size, data_block_size);
                next_paired_viewers.push(PairedViewer {
                    id: update.id,
                    state: fixed_state_copy(&new_state),
                    prev_state: fixed_state_copy(&old_state),
                });
            }

            if old_state.data_box != new_state.data_box {
                let data_upper = usize::try_from(old_state.data_box.size.volume_u64())
                    .ok()
                    .and_then(|count| {
                        usize::try_from(new_state.data_box.size.volume_u64())
                            .ok()
                            .and_then(|other| count.checked_add(other))
                    })
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                data_deltas
                    .try_reserve(data_upper)
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                for position in old_state.data_box.iter_cells_zxy() {
                    let delta = data_deltas.entry(position).or_default();
                    *delta = delta
                        .checked_sub(1)
                        .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                }
                for position in new_state.data_box.iter_cells_zxy() {
                    let delta = data_deltas.entry(position).or_default();
                    *delta = delta
                        .checked_add(1)
                        .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                }
            }
            if old_state.mesh_box != new_state.mesh_box || old_state.demand != new_state.demand {
                let mesh_upper = usize::try_from(old_state.mesh_box.size.volume_u64())
                    .ok()
                    .and_then(|count| {
                        usize::try_from(new_state.mesh_box.size.volume_u64())
                            .ok()
                            .and_then(|other| count.checked_add(other))
                    })
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                mesh_deltas
                    .try_reserve(mesh_upper)
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                for position in old_state.mesh_box.iter_cells_zxy() {
                    let delta = mesh_deltas.entry(position).or_default();
                    delta.0 = delta
                        .0
                        .checked_sub(1)
                        .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                    delta.1 = delta
                        .1
                        .checked_sub(i64::from(old_state.demand.visuals))
                        .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                    delta.2 = delta
                        .2
                        .checked_sub(i64::from(old_state.demand.collisions))
                        .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                }
                for position in new_state.mesh_box.iter_cells_zxy() {
                    let delta = mesh_deltas.entry(position).or_default();
                    delta.0 = delta
                        .0
                        .checked_add(1)
                        .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                    delta.1 = delta
                        .1
                        .checked_add(i64::from(new_state.demand.visuals))
                        .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                    delta.2 = delta
                        .2
                        .checked_add(i64::from(new_state.demand.collisions))
                        .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                }
            }
        }

        let mut next_request_generation = self.next_request_generation;
        let mut next_mesh_revision = self.next_mesh_revision;
        let mut next_render_topology_revision = self.next_render_topology_revision;
        let mut next_stats = self.stats;
        let mut loading_shadow = HashMap::<Vector3i, Option<LoadingBlockEntry>>::new();
        let mut data_residency_shadow = HashMap::<Vector3i, Option<DataResidencyRefs>>::new();
        let mut mesh_shadow = HashMap::<Vector3i, Option<MeshBlockEntry>>::new();
        let mut pending_load = HashSet::new();
        pending_load
            .try_reserve(self.blocks_pending_load[0].len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        pending_load.extend(self.blocks_pending_load[0].iter().copied());
        let mut pending_mesh = HashSet::new();
        pending_mesh
            .try_reserve(self.blocks_pending_update[0].len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        pending_mesh.extend(self.blocks_pending_update[0].iter().copied());
        let mut data_operations = resident_operations;
        let mut data_snapshots = resident_snapshots;
        let mut data_presence = HashMap::<Vector3i, bool>::new();
        data_presence
            .try_reserve(data_snapshots.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        data_presence.extend(
            data_snapshots
                .iter()
                .map(|snapshot| (snapshot.location().position, snapshot.is_present())),
        );
        let mut events_to_append = Vec::new();
        let mut mesh_exit_events = Vec::new();
        let mut topology_changes = Vec::<(CoverageFeature, bool, MeshBlockLocation)>::new();
        let mut removed_snapshots = Vec::<SharedVoxelDataTransactionBlockSnapshot>::new();
        loading_shadow
            .try_reserve(self.blocks_pending_load[0].len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for position in &self.blocks_pending_load[0] {
            loading_shadow
                .entry(*position)
                .or_insert_with(|| self.loading_blocks[0].get(position).cloned());
        }
        mesh_shadow
            .try_reserve(self.blocks_pending_update[0].len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for position in &self.blocks_pending_update[0] {
            mesh_shadow.entry(*position).or_insert_with(|| {
                self.mesh_maps[0]
                    .get(position)
                    .map(MeshBlockEntry::clone_for_draft)
            });
        }

        let mut data_positions = Vec::new();
        data_positions
            .try_reserve_exact(data_deltas.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        // Keep the complete touched union, including zero-net replacements.
        // A remove/add pair that cancels numerically still crossed the public
        // ownership boundary and must validate storage against the exact
        // sidecar before any other part of the draft can publish.
        data_positions.extend(data_deltas);
        data_positions.sort_unstable_by_key(|(position, delta)| {
            (*delta <= 0, position.x, position.y, position.z)
        });
        loading_shadow
            .try_reserve(data_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        pending_load
            .try_reserve(data_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        events_to_append
            .try_reserve(data_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        data_operations
            .try_reserve_exact(data_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        data_snapshots
            .try_reserve_exact(data_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        data_presence
            .try_reserve(data_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        removed_snapshots
            .try_reserve_exact(data_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (position, delta) in data_positions {
            let location = BlockLocation {
                position,
                lod_index: 0,
            };
            let snapshot = data_snapshots
                .iter()
                .find(|snapshot| snapshot.location() == location)
                .cloned()
                .unwrap_or_else(|| {
                    let snapshot = data_preview
                        .block_snapshot(location)
                        .expect("fixed LOD is present");
                    data_snapshots.push(snapshot.clone());
                    snapshot
                });
            if snapshot.is_present() {
                if let Some(loading) = self.loading_blocks[0].get(&position) {
                    return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                        location,
                        tracked_resident_viewers: Some(loading.residency.resident_viewers),
                        tracked_coverage_holds: Some(loading.residency.coverage_holds),
                        storage_viewers: snapshot.viewers(),
                    });
                }
                let Some(original_residency) =
                    self.loaded_data_residency[0].get(&position).copied()
                else {
                    return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                        location,
                        tracked_resident_viewers: None,
                        tracked_coverage_holds: None,
                        storage_viewers: snapshot.viewers(),
                    });
                };
                let tracked_total = original_residency.checked_total(location)?;
                if tracked_total != snapshot.viewers() {
                    return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                        location,
                        tracked_resident_viewers: Some(original_residency.resident_viewers),
                        tracked_coverage_holds: Some(original_residency.coverage_holds),
                        storage_viewers: snapshot.viewers(),
                    });
                }
                let next_residency = original_residency.checked_apply_delta(
                    location,
                    DataRefField::ResidentViewers,
                    delta,
                )?;
                let final_viewers = next_residency.checked_total(location)?;
                data_residency_shadow.insert(
                    position,
                    (!next_residency.is_empty()).then_some(next_residency),
                );
                if next_residency.is_empty() {
                    data_presence.insert(position, false);
                    if let Some(operation) = data_operations
                        .iter_mut()
                        .find(|operation| operation.location() == location)
                    {
                        debug_assert!(matches!(
                            operation,
                            SharedVoxelDataTransactionOperation::ClearModified { .. }
                        ));
                        *operation = SharedVoxelDataTransactionOperation::Remove { location };
                        resident_payloads.retain(|payload| payload.location != location);
                    } else {
                        data_operations
                            .push(SharedVoxelDataTransactionOperation::Remove { location });
                    }
                    removed_snapshots.push(snapshot);
                    events_to_append.push(VoxelTerrainEvent::DataBlockUnloaded(location));
                    next_stats.blocks_unloaded = next_stats
                        .blocks_unloaded
                        .checked_add(1)
                        .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
                } else if final_viewers != snapshot.viewers() {
                    data_presence.insert(position, true);
                    if let Some(operation) = data_operations
                        .iter_mut()
                        .find(|operation| operation.location() == location)
                    {
                        debug_assert!(matches!(
                            operation,
                            SharedVoxelDataTransactionOperation::ClearModified { .. }
                        ));
                        *operation =
                            SharedVoxelDataTransactionOperation::SetViewersExactAndClearModified {
                                location,
                                final_viewers,
                            };
                    } else {
                        data_operations.push(
                            SharedVoxelDataTransactionOperation::SetViewersExact {
                                location,
                                final_viewers,
                            },
                        );
                    }
                }
            } else {
                if let Some(tracked) = self.loaded_data_residency[0].get(&position).copied() {
                    return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                        location,
                        tracked_resident_viewers: Some(tracked.resident_viewers),
                        tracked_coverage_holds: Some(tracked.coverage_holds),
                        storage_viewers: 0,
                    });
                }
                data_presence.insert(position, false);
                let entry = loading_shadow
                    .entry(position)
                    .or_insert_with(|| self.loading_blocks[0].get(&position).cloned());
                let current = entry
                    .as_ref()
                    .map_or(DataResidencyRefs::default(), |entry| entry.residency);
                let next_residency =
                    current.checked_apply_delta(location, DataRefField::ResidentViewers, delta)?;
                if next_residency.is_empty() {
                    *entry = None;
                    pending_load.remove(&position);
                } else if let Some(entry) = entry.as_mut() {
                    entry.residency = next_residency;
                    if matches!(
                        entry.request_state,
                        LoadRequestState::NotFound | LoadRequestState::Exhausted
                    ) && delta > 0
                    {
                        let generation = next_request_generation;
                        next_request_generation = generation
                            .checked_add(1)
                            .ok_or(VoxelTerrainRuntimeError::RequestGenerationOverflow)?;
                        entry.retry_count = 0;
                        entry.request_generation = generation;
                        entry.request_state = LoadRequestState::Queued;
                        entry.physical_request = Some(PhysicalRequest::new(TaskRequestTag::new(
                            self.request_epoch,
                            generation,
                        )));
                        pending_load.insert(position);
                    }
                } else {
                    let generation = next_request_generation;
                    next_request_generation = generation
                        .checked_add(1)
                        .ok_or(VoxelTerrainRuntimeError::RequestGenerationOverflow)?;
                    *entry = Some(LoadingBlockEntry {
                        residency: next_residency,
                        retry_count: 0,
                        request_generation: generation,
                        request_state: LoadRequestState::Queued,
                        physical_request: Some(PhysicalRequest::new(TaskRequestTag::new(
                            self.request_epoch,
                            generation,
                        ))),
                    });
                    pending_load.insert(position);
                }
            }
        }

        let mut mesh_positions = Vec::new();
        mesh_positions
            .try_reserve_exact(mesh_deltas.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for delta in mesh_deltas {
            if delta.1 != (0, 0, 0) {
                mesh_positions.push(delta);
            }
        }
        mesh_positions.sort_unstable_by_key(|(position, delta)| {
            (
                !(delta.0 > 0 || delta.1 > 0 || delta.2 > 0),
                position.x,
                position.y,
                position.z,
            )
        });
        mesh_shadow
            .try_reserve(mesh_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        pending_mesh
            .try_reserve(mesh_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        mesh_exit_events
            .try_reserve_exact(mesh_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (position, (resident_delta, visual_delta, collision_delta)) in mesh_positions {
            let location = MeshBlockLocation::new(position, 0);
            let shadow = mesh_shadow.entry(position).or_insert_with(|| {
                self.mesh_maps[0]
                    .get(&position)
                    .map(MeshBlockEntry::clone_for_draft)
            });
            let mut entry = shadow.take().unwrap_or_else(|| MeshBlockEntry {
                position,
                ..MeshBlockEntry::default()
            });
            entry.checked_apply_ref_delta(
                location,
                MeshRefField::ResidentViewers,
                resident_delta,
            )?;
            entry.checked_apply_ref_delta(location, MeshRefField::VisualViewers, visual_delta)?;
            entry.checked_apply_ref_delta(
                location,
                MeshRefField::CollisionViewers,
                collision_delta,
            )?;

            if entry.resident_refcount() == 0 {
                pending_mesh.remove(&position);
                if entry.accepted_upload.is_some() {
                    mesh_exit_events.push(VoxelTerrainEvent::MeshBlockExited(location));
                }
                *shadow = None;
                continue;
            }

            let required_features = MeshBuildFeatures {
                visuals: entry.needs_visual(),
                collisions: entry.needs_collision(),
                variable_lod: false,
            };
            let accepted_contains = entry.accepted_upload.is_some()
                && entry.applied_features.contains(required_features);
            if !accepted_contains && !entry.requested_features.contains(required_features) {
                let revision = next_mesh_revision;
                next_mesh_revision = revision
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::MeshRevisionOverflow)?;
                let (generation, physical_request) =
                    allocate_physical_request(self.request_epoch, &mut next_request_generation)?;
                entry.requested_revision = Some(revision);
                entry.request_generation = generation;
                entry.requested_features = required_features;
                entry.terminal_retry_count = 0;
                entry.physical_request = Some(physical_request);
                if !entry.is_in_update_list {
                    entry.is_in_update_list = true;
                    pending_mesh.insert(position);
                }
            }

            let visual_ready = entry
                .accepted_upload
                .as_ref()
                .is_some_and(|upload| upload.visual_state() != PayloadState::NotBuilt);
            let collision_ready = entry
                .accepted_upload
                .as_ref()
                .is_some_and(|upload| upload.collision_state() != PayloadState::NotBuilt);
            let next_visual_active = entry.needs_visual() && visual_ready;
            let next_collision_active = entry.needs_collision() && collision_ready;
            if next_visual_active != entry.visual_active {
                entry.visual_active = next_visual_active;
            }
            if next_collision_active != entry.collision_active {
                entry.collision_active = next_collision_active;
            }
            *shadow = Some(entry);
        }

        let completion_count = self.durable_completion_inbox.len();
        let direct_mesh_prefix_len = self.direct_mesh_retry_inbox.len();
        #[cfg(test)]
        if completion_count != 0 {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::DurableEffects)?;
        }
        #[cfg(test)]
        if direct_mesh_prefix_len != 0 {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::DirectEffects)?;
        }
        let completion_and_direct = completion_count
            .checked_add(direct_mesh_prefix_len)
            .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
        let mut accepted_feature_updates = Vec::new();
        accepted_feature_updates
            .try_reserve_exact(completion_and_direct)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        loading_shadow
            .try_reserve(completion_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        mesh_shadow
            .try_reserve(
                completion_count
                    .checked_mul(27)
                    .and_then(|count| count.checked_add(direct_mesh_prefix_len))
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        pending_load
            .try_reserve(completion_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        pending_mesh
            .try_reserve(
                completion_count
                    .checked_mul(27)
                    .and_then(|count| count.checked_add(direct_mesh_prefix_len))
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        data_snapshots
            .try_reserve(completion_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        data_presence
            .try_reserve(completion_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        data_operations
            .try_reserve(completion_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        events_to_append
            .try_reserve(
                completion_and_direct
                    .checked_mul(3)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        topology_changes
            .try_reserve(
                completion_and_direct
                    .checked_mul(2)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut completion_plans = Vec::new();
        completion_plans
            .try_reserve_exact(completion_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut load_inserts = Vec::new();
        load_inserts
            .try_reserve_exact(completion_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut accepted_followup_count = 0usize;
        let mut quarantine_count = 0usize;
        let checkpoint_key_count = self
            .save_checkpoint_in_flight
            .as_ref()
            .map_or(0, |checkpoint| checkpoint.acknowledged.len());
        let mut journal_keys = Vec::new();
        journal_keys
            .try_reserve_exact(
                completion_count
                    .checked_add(checkpoint_key_count)
                    .and_then(|count| count.checked_add(self.save_journal.len()))
                    .and_then(|count| count.checked_add(self.deferred_save_dispatch_keys.len()))
                    .and_then(|count| count.checked_add(removed_snapshots.len()))
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for completion in &self.durable_completion_inbox {
            let terminal = match completion {
                DurableCompletion::SaveAcknowledged { terminal, .. } => Some(terminal),
                DurableCompletion::PersistenceTerminal {
                    terminal: PersistenceTaskTerminal::Save(terminal),
                    ..
                }
                | DurableCompletion::MalformedPersistence {
                    terminal: PersistenceTaskTerminal::Save(terminal),
                    ..
                } => Some(terminal),
                _ => None,
            };
            if let Some(terminal) = terminal {
                journal_keys.push(SaveKey::new(
                    terminal.location.position,
                    terminal.location.lod_index,
                ));
            }
        }
        if let Some(checkpoint) = self.save_checkpoint_in_flight.as_ref() {
            journal_keys.extend(checkpoint.acknowledged.iter().map(|snapshot| snapshot.key));
        }
        journal_keys.extend(self.save_journal.keys().copied());
        journal_keys.extend(self.deferred_save_dispatch_keys.iter().copied());
        journal_keys.extend(
            removed_snapshots
                .iter()
                .filter(|snapshot| snapshot.is_modified() && snapshot.has_voxels())
                .map(|snapshot| {
                    let location = snapshot.location();
                    SaveKey::new(location.position, location.lod_index)
                }),
        );
        journal_keys.extend(
            resident_payloads
                .iter()
                .map(|payload| SaveKey::new(payload.location.position, payload.location.lod_index)),
        );
        journal_keys.sort_unstable_by_key(|key| {
            (
                key.lod_index,
                key.position.x,
                key.position.y,
                key.position.z,
            )
        });
        journal_keys.dedup();
        let mut journal_shadow = HashMap::<SaveKey, PreparedJournalShadow>::new();
        journal_shadow
            .try_reserve(journal_keys.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for key in journal_keys {
            if let Some(entry) = self.save_journal.get(&key) {
                journal_shadow.insert(key, PreparedJournalShadow::from_entry(entry));
            }
        }
        let mut checkpoint_shadow = self
            .save_checkpoint_in_flight
            .as_ref()
            .map(PreparedCheckpointShadow::from_checkpoint);

        // Viewer exits establish their journal metadata before the durable
        // prefix is replayed, so a completion observes the same sequential
        // viewer -> durable -> direct state order as the eventual commit.
        let mut next_save_generation = self.next_save_generation;
        let mut next_save_checkpoint_generation = self.next_save_checkpoint_generation;
        let mut next_persistence_attempt_ordinal = self.next_persistence_attempt_ordinal;
        let mut persistence_tasks = Vec::new();
        persistence_tasks
            .try_reserve_exact(
                removed_snapshots
                    .len()
                    .checked_add(resident_payloads.len())
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut removed_owner_routes = Vec::new();
        removed_owner_routes
            .try_reserve_exact(removed_snapshots.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut resident_save_routes = Vec::new();
        resident_save_routes
            .try_reserve_exact(resident_payloads.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut vacant_journal_count = 0usize;
        let mut dirty_retained_count = 0usize;
        #[cfg(test)]
        let mut save_panic_hook_available = self.panic_next_save_before_io_for_test;
        #[cfg(not(test))]
        let save_panic_hook_available = false;
        for snapshot in &removed_snapshots {
            if !snapshot.is_modified() {
                removed_owner_routes.push(PreparedRemovedOwnerRoute::Clean);
                continue;
            }
            if !snapshot.has_voxels() {
                let VoxelDataKeyRevision::Present(block_revision) = snapshot.revision() else {
                    unreachable!("a removed resident snapshot has a present revision")
                };
                dirty_retained_count = dirty_retained_count
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                removed_owner_routes.push(PreparedRemovedOwnerRoute::DirtyRetained {
                    location: snapshot.location(),
                    block_revision,
                });
                continue;
            }
            let VoxelDataKeyRevision::Present(block_revision) = snapshot.revision() else {
                unreachable!("a removed resident snapshot has a present revision")
            };
            let save_generation = allocate_persistence_generation(&mut next_save_generation)?;
            let snapshot_location = snapshot.location();
            let key = SaveKey::new(snapshot_location.position, snapshot_location.lod_index);
            let shadow = journal_shadow.entry(key).or_insert(PreparedJournalShadow {
                written_block_revision: None,
                written_generation: None,
                active: PreparedJournalActiveShadow::None,
                queued_len: 0,
                queued_front: None,
            });
            let target = if !self.save_journal.contains_key(&key) {
                vacant_journal_count = vacant_journal_count
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                PreparedJournalTarget::Vacant
            } else if shadow.active == PreparedJournalActiveShadow::None
                && shadow.written_generation.is_none()
            {
                PreparedJournalTarget::ActiveIdle
            } else {
                PreparedJournalTarget::Queued
            };
            let dispatch = dispatch_persistence_tasks
                && checkpoint_shadow.is_none()
                && !self.automatic_save_checkpoint_blocked
                && !matches!(target, PreparedJournalTarget::Queued);
            if dispatch {
                let attempt_ordinal = next_persistence_attempt_ordinal;
                next_persistence_attempt_ordinal = attempt_ordinal.checked_add(1).ok_or(
                    VoxelTerrainRuntimeError::PersistenceAttemptOverflow {
                        operation: PersistenceOperation::Save {
                            location: snapshot.location(),
                            block_revision,
                            save_generation,
                        },
                    },
                )?;
                #[allow(unused_mut)]
                let mut task = PreparedSaveBlockDataTask::new(
                    snapshot.location(),
                    block_revision,
                    save_generation,
                    StreamingDependency::new(self.stream.clone()),
                    None,
                    attempt_ordinal,
                );
                #[cfg(test)]
                if save_panic_hook_available {
                    task.set_panic_before_io_for_test(true);
                    save_panic_hook_available = false;
                }
                let (scheduled, installer) = task.into_scheduled();
                persistence_tasks.push((key, save_generation, scheduled));
                shadow.active = PreparedJournalActiveShadow::WriteInFlight {
                    block_revision,
                    generation: save_generation,
                    attempt_ordinal,
                    retry_count: 0,
                };
                removed_owner_routes.push(PreparedRemovedOwnerRoute::DirtyDispatch(
                    PreparedDirtyDispatch {
                        location: snapshot.location(),
                        block_revision,
                        save_generation,
                        attempt_ordinal,
                        target,
                        installer,
                    },
                ));
            } else {
                match target {
                    PreparedJournalTarget::Vacant | PreparedJournalTarget::ActiveIdle => {
                        shadow.active = PreparedJournalActiveShadow::Pending {
                            block_revision,
                            generation: save_generation,
                            retry_count: 0,
                        };
                    }
                    PreparedJournalTarget::Queued => {
                        if shadow.queued_len == 0 {
                            shadow.queued_front = Some((block_revision, save_generation, 0));
                        }
                        shadow.queued_len = shadow
                            .queued_len
                            .checked_add(1)
                            .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                    }
                }
                removed_owner_routes.push(PreparedRemovedOwnerRoute::DirtyPending(
                    PreparedDirtyPending {
                        location: snapshot.location(),
                        block_revision,
                        save_generation,
                        target,
                    },
                ));
            }
        }

        for payload in resident_payloads.drain(..) {
            let location = payload.location;
            let block_revision = payload.block_revision;
            let save_generation = allocate_persistence_generation(&mut next_save_generation)?;
            let key = SaveKey::new(location.position, location.lod_index);
            let shadow = journal_shadow.entry(key).or_insert(PreparedJournalShadow {
                written_block_revision: None,
                written_generation: None,
                active: PreparedJournalActiveShadow::None,
                queued_len: 0,
                queued_front: None,
            });
            let target = if !self.save_journal.contains_key(&key) {
                vacant_journal_count = vacant_journal_count
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                PreparedJournalTarget::Vacant
            } else if shadow.active == PreparedJournalActiveShadow::None
                && shadow.written_generation.is_none()
            {
                PreparedJournalTarget::ActiveIdle
            } else {
                PreparedJournalTarget::Queued
            };
            let dispatch = dispatch_persistence_tasks
                && checkpoint_shadow.is_none()
                && !self.automatic_save_checkpoint_blocked
                && !matches!(target, PreparedJournalTarget::Queued);
            if dispatch {
                let attempt_ordinal = next_persistence_attempt_ordinal;
                next_persistence_attempt_ordinal = attempt_ordinal.checked_add(1).ok_or(
                    VoxelTerrainRuntimeError::PersistenceAttemptOverflow {
                        operation: PersistenceOperation::Save {
                            location,
                            block_revision,
                            save_generation,
                        },
                    },
                )?;
                #[allow(unused_mut)]
                let mut task = PreparedSaveBlockDataTask::new(
                    location,
                    block_revision,
                    save_generation,
                    StreamingDependency::new(self.stream.clone()),
                    None,
                    attempt_ordinal,
                );
                #[cfg(test)]
                if save_panic_hook_available {
                    task.set_panic_before_io_for_test(true);
                    save_panic_hook_available = false;
                }
                let (scheduled, installer) = task.into_scheduled();
                persistence_tasks.push((key, save_generation, scheduled));
                shadow.active = PreparedJournalActiveShadow::WriteInFlight {
                    block_revision,
                    generation: save_generation,
                    attempt_ordinal,
                    retry_count: 0,
                };
                resident_save_routes.push(PreparedResidentSaveRoute::Dispatch {
                    payload,
                    route: PreparedDirtyDispatch {
                        location,
                        block_revision,
                        save_generation,
                        attempt_ordinal,
                        target,
                        installer,
                    },
                });
            } else {
                match target {
                    PreparedJournalTarget::Vacant | PreparedJournalTarget::ActiveIdle => {
                        shadow.active = PreparedJournalActiveShadow::Pending {
                            block_revision,
                            generation: save_generation,
                            retry_count: 0,
                        };
                    }
                    PreparedJournalTarget::Queued => {
                        if shadow.queued_len == 0 {
                            shadow.queued_front = Some((block_revision, save_generation, 0));
                        }
                        shadow.queued_len = shadow
                            .queued_len
                            .checked_add(1)
                            .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                    }
                }
                resident_save_routes.push(PreparedResidentSaveRoute::Pending {
                    payload,
                    route: PreparedDirtyPending {
                        location,
                        block_revision,
                        save_generation,
                        target,
                    },
                });
            }
        }

        for (inbox_index, completion) in self.durable_completion_inbox.iter().enumerate() {
            let mut publish_followups = false;
            let disposition = match completion {
                DurableCompletion::LoadFinished {
                    completed: _,
                    output,
                } => {
                    let bpos = output.block_data.position_in_blocks;
                    let lod_index = output.block_data.lod_index;
                    if lod_index != 0 {
                        PreparedCompletionDisposition::Retire {
                            publish_followups: false,
                        }
                    } else {
                        let shadow = loading_shadow
                            .entry(bpos)
                            .or_insert_with(|| self.loading_blocks[0].get(&bpos).cloned());
                        let is_current = shadow.as_ref().is_some_and(|entry| {
                            entry.request_generation == output.request_generation
                                && entry.matches_physical_request(output.request_tag)
                                && output
                                    .request_tag
                                    .is_none_or(|tag| tag.request_epoch == self.request_epoch)
                        });
                        if !is_current {
                            if !self.shutdown_in_progress {
                                if let Some(entry) = shadow.as_mut().filter(|entry| {
                                    !entry.residency.is_empty()
                                        && entry.request_generation == output.request_generation
                                        && entry.matches_previous_epoch_request(
                                            output.request_tag,
                                            self.request_epoch,
                                        )
                                }) {
                                    let (generation, physical_request) = allocate_physical_request(
                                        self.request_epoch,
                                        &mut next_request_generation,
                                    )?;
                                    entry.request_generation = generation;
                                    entry.physical_request = Some(physical_request);
                                    entry.request_state = LoadRequestState::Queued;
                                    pending_load.insert(bpos);
                                }
                            }
                            PreparedCompletionDisposition::Retire {
                                publish_followups: false,
                            }
                        } else {
                            match output.block_data.kind {
                                BlockDataOutputKind::Loaded
                                | BlockDataOutputKind::NeedsGeneration
                                    if output.block_data.dropped =>
                                {
                                    let entry = shadow
                                        .as_mut()
                                        .expect("current completion retained loading shadow");
                                    if self.shutdown_in_progress {
                                        entry.request_state = LoadRequestState::Exhausted;
                                    } else {
                                        entry.retry_count =
                                            entry.retry_count.checked_add(1).ok_or(
                                                VoxelTerrainRuntimeError::LoadRetryCountOverflow {
                                                    location: BlockLocation {
                                                        position: bpos,
                                                        lod_index: 0,
                                                    },
                                                },
                                            )?;
                                    }
                                    if entry.retry_count <= MAX_LOAD_RETRIES
                                        && !self.shutdown_in_progress
                                    {
                                        let (generation, physical_request) =
                                            allocate_physical_request(
                                                self.request_epoch,
                                                &mut next_request_generation,
                                            )?;
                                        entry.request_generation = generation;
                                        entry.request_state = LoadRequestState::Queued;
                                        entry.physical_request = Some(physical_request);
                                        pending_load.insert(bpos);
                                    } else {
                                        entry.request_state = LoadRequestState::Exhausted;
                                        entry.physical_request = None;
                                    }
                                    publish_followups = true;
                                    PreparedCompletionDisposition::Retire { publish_followups }
                                }
                                BlockDataOutputKind::Loaded
                                | BlockDataOutputKind::NeedsGeneration
                                    if output.block_data.voxels.is_some() =>
                                {
                                    let residency = shadow
                                        .as_ref()
                                        .map_or(DataResidencyRefs::default(), |entry| {
                                            entry.residency
                                        });
                                    if residency.is_empty() {
                                        PreparedCompletionDisposition::Retire {
                                            publish_followups: false,
                                        }
                                    } else {
                                        let location = BlockLocation {
                                            position: bpos,
                                            lod_index: 0,
                                        };
                                        let present = if let Some(present) =
                                            data_presence.get(&location.position)
                                        {
                                            *present
                                        } else {
                                            data_presence.try_reserve(1).map_err(|_| {
                                                VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
                                            })?;
                                            data_snapshots.try_reserve(1).map_err(|_| {
                                                VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
                                            })?;
                                            let snapshot = data_preview
                                                .block_snapshot(location)
                                                .expect("fixed LOD is present");
                                            data_snapshots.push(snapshot.clone());
                                            data_presence
                                                .insert(location.position, snapshot.is_present());
                                            snapshot.is_present()
                                        };
                                        if present {
                                            *shadow = None;
                                            pending_load.remove(&bpos);
                                            PreparedCompletionDisposition::Retire {
                                                publish_followups: false,
                                            }
                                        } else {
                                            let final_viewers =
                                                residency.checked_total(location)?;
                                            data_presence.insert(location.position, true);
                                            data_residency_shadow
                                                .insert(location.position, Some(residency));
                                            load_inserts.push(PreparedLoadInsert {
                                                inbox_index,
                                                location,
                                                final_viewers,
                                            });
                                            *shadow = None;
                                            pending_load.remove(&bpos);
                                            let loaded_voxel_box = block_box_to_voxel_box(
                                                Box3i::new(bpos, Vector3i::splat(1)),
                                                data_block_size,
                                                0,
                                            );
                                            let affected_mesh_box = loaded_voxel_box
                                                .padded(1)
                                                .downscaled(data_block_size);
                                            for mesh_position in affected_mesh_box.iter_cells_zxy()
                                            {
                                                let mesh = mesh_shadow
                                                    .entry(mesh_position)
                                                    .or_insert_with(|| {
                                                        self.mesh_maps[0]
                                                            .get(&mesh_position)
                                                            .map(MeshBlockEntry::clone_for_draft)
                                                    });
                                                let Some(entry) = mesh
                                                    .as_mut()
                                                    .filter(|entry| entry.resident_refcount() > 0)
                                                else {
                                                    continue;
                                                };
                                                if !self.fixed_draft_mesh_data_is_ready(
                                                    &data_preview,
                                                    mesh_position,
                                                    &mut data_presence,
                                                    &mut data_snapshots,
                                                )? {
                                                    continue;
                                                }
                                                if entry.requested_revision
                                                    == entry.applied_revision
                                                {
                                                    let revision = next_mesh_revision;
                                                    next_mesh_revision = revision
                                                        .checked_add(1)
                                                        .ok_or(
                                                            VoxelTerrainRuntimeError::MeshRevisionOverflow,
                                                        )?;
                                                    let (generation, physical_request) =
                                                        allocate_physical_request(
                                                            self.request_epoch,
                                                            &mut next_request_generation,
                                                        )?;
                                                    entry.requested_revision = Some(revision);
                                                    entry.request_generation = generation;
                                                    entry.requested_features = MeshBuildFeatures {
                                                        visuals: entry.needs_visual(),
                                                        collisions: entry.needs_collision(),
                                                        variable_lod: false,
                                                    };
                                                    entry.terminal_retry_count = 0;
                                                    entry.physical_request = Some(physical_request);
                                                }
                                                if !entry.is_in_update_list {
                                                    entry.is_in_update_list = true;
                                                    pending_mesh.insert(mesh_position);
                                                }
                                            }
                                            next_stats.blocks_loaded = next_stats
                                                .blocks_loaded
                                                .checked_add(1)
                                                .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
                                            events_to_append
                                                .push(VoxelTerrainEvent::DataBlockLoaded(location));
                                            publish_followups = true;
                                            PreparedCompletionDisposition::Retire {
                                                publish_followups,
                                            }
                                        }
                                    }
                                }
                                BlockDataOutputKind::NotFound => {
                                    let entry = shadow
                                        .as_mut()
                                        .expect("current completion retained loading shadow");
                                    entry.request_state = LoadRequestState::NotFound;
                                    entry.physical_request = None;
                                    pending_load.remove(&bpos);
                                    publish_followups = true;
                                    PreparedCompletionDisposition::Retire { publish_followups }
                                }
                                BlockDataOutputKind::Loaded
                                | BlockDataOutputKind::NeedsGeneration
                                | BlockDataOutputKind::Saved => {
                                    PreparedCompletionDisposition::Retire {
                                        publish_followups: false,
                                    }
                                }
                            }
                        }
                    }
                }
                DurableCompletion::LoadTerminal {
                    position,
                    lod_index,
                    request_generation,
                    request_tag,
                    ..
                } => {
                    if *lod_index == 0 {
                        let shadow = loading_shadow
                            .entry(*position)
                            .or_insert_with(|| self.loading_blocks[0].get(position).cloned());
                        if let Some(entry) = shadow.as_mut().filter(|entry| {
                            !entry.residency.is_empty()
                                && entry.request_generation == *request_generation
                                && entry.request_state == LoadRequestState::InFlight
                        }) {
                            if !self.shutdown_in_progress
                                && entry.matches_previous_epoch_request(
                                    *request_tag,
                                    self.request_epoch,
                                )
                            {
                                let (generation, physical_request) = allocate_physical_request(
                                    self.request_epoch,
                                    &mut next_request_generation,
                                )?;
                                entry.request_generation = generation;
                                entry.physical_request = Some(physical_request);
                                entry.request_state = LoadRequestState::Queued;
                                pending_load.insert(*position);
                            } else if entry.matches_physical_request(*request_tag)
                                && request_tag
                                    .is_none_or(|tag| tag.request_epoch == self.request_epoch)
                            {
                                if self.shutdown_in_progress {
                                    entry.request_state = LoadRequestState::Exhausted;
                                } else {
                                    entry.retry_count = entry.retry_count.checked_add(1).ok_or(
                                        VoxelTerrainRuntimeError::LoadRetryCountOverflow {
                                            location: BlockLocation {
                                                position: *position,
                                                lod_index: 0,
                                            },
                                        },
                                    )?;
                                    if entry.retry_count > MAX_LOAD_RETRIES {
                                        entry.request_state = LoadRequestState::Exhausted;
                                        entry.physical_request = None;
                                    } else {
                                        let (generation, physical_request) =
                                            allocate_physical_request(
                                                self.request_epoch,
                                                &mut next_request_generation,
                                            )?;
                                        entry.request_generation = generation;
                                        entry.request_state = LoadRequestState::Queued;
                                        entry.physical_request = Some(physical_request);
                                        pending_load.insert(*position);
                                    }
                                }
                            }
                        }
                    }
                    PreparedCompletionDisposition::Retire {
                        publish_followups: false,
                    }
                }
                DurableCompletion::MeshFinished {
                    completed: _,
                    output,
                } => {
                    let upload = output.upload();
                    let key = upload.key();
                    if key.location.lod_index != 0 {
                        next_stats.meshes_dropped = next_stats
                            .meshes_dropped
                            .checked_add(1)
                            .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
                        PreparedCompletionDisposition::Retire {
                            publish_followups: false,
                        }
                    } else {
                        let position = key.location.position_in_blocks;
                        let shadow = mesh_shadow.entry(position).or_insert_with(|| {
                            self.mesh_maps[0]
                                .get(&position)
                                .map(MeshBlockEntry::clone_for_draft)
                        });
                        let is_current = shadow.as_ref().is_some_and(|entry| {
                            entry.requested_revision == Some(key.revision)
                                && entry.matches_physical_request(output.request_tag())
                                && output
                                    .request_tag()
                                    .is_none_or(|tag| tag.request_epoch == self.request_epoch)
                        });
                        if !is_current {
                            next_stats.meshes_dropped = next_stats
                                .meshes_dropped
                                .checked_add(1)
                                .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
                            if !self.shutdown_in_progress {
                                if let Some(entry) = shadow.as_mut().filter(|entry| {
                                    entry.resident_refcount() > 0
                                        && entry.requested_revision == Some(key.revision)
                                        && entry.matches_previous_epoch_request(
                                            output.request_tag(),
                                            self.request_epoch,
                                        )
                                }) {
                                    let (generation, physical_request) = allocate_physical_request(
                                        self.request_epoch,
                                        &mut next_request_generation,
                                    )?;
                                    entry.request_generation = generation;
                                    entry.physical_request = Some(physical_request);
                                    if !entry.is_in_update_list {
                                        entry.is_in_update_list = true;
                                        pending_mesh.insert(position);
                                    }
                                }
                            }
                            PreparedCompletionDisposition::Retire {
                                publish_followups: false,
                            }
                        } else {
                            let entry = shadow
                                .as_mut()
                                .expect("current completion retained mesh shadow");
                            entry.terminal_retry_count = 0;
                            if output.dropped() {
                                next_stats.meshes_dropped = next_stats
                                    .meshes_dropped
                                    .checked_add(1)
                                    .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
                                if !self.shutdown_in_progress && entry.resident_refcount() > 0 {
                                    let (generation, physical_request) = allocate_physical_request(
                                        self.request_epoch,
                                        &mut next_request_generation,
                                    )?;
                                    entry.request_generation = generation;
                                    entry.physical_request = Some(physical_request);
                                    entry.is_in_update_list = true;
                                    pending_mesh.insert(position);
                                }
                            } else {
                                // Accepting a usable result terminalizes this
                                // exact physical request. A direct/runner peer
                                // for the same content revision is stale after
                                // this draft commits.
                                entry.physical_request = None;
                                let had_built_payload = entry
                                    .accepted_upload
                                    .as_ref()
                                    .is_some_and(|accepted| accepted.has_built_payload());
                                let has_built_payload = upload.has_built_payload();
                                entry.accepted_upload = Some(upload.clone());
                                entry.applied_revision = Some(key.revision);
                                entry.requested_features = upload.features();
                                entry.applied_features = upload.features();
                                entry.has_geometry =
                                    upload.visual_state() == PayloadState::NonEmpty;
                                entry.is_loaded = true;
                                next_stats.meshes_built = next_stats
                                    .meshes_built
                                    .checked_add(1)
                                    .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
                                events_to_append.push(
                                    match (had_built_payload, has_built_payload) {
                                        (false, true) => {
                                            VoxelTerrainEvent::MeshBlockEntered(upload.clone())
                                        }
                                        (true, true) => {
                                            VoxelTerrainEvent::MeshBlockUpdated(upload.clone())
                                        }
                                        (_, false) => {
                                            VoxelTerrainEvent::MeshBlockBecameEmpty(upload.clone())
                                        }
                                    },
                                );
                                accepted_feature_updates.push(key.location);
                                let demanded_features = MeshBuildFeatures {
                                    visuals: entry.needs_visual(),
                                    collisions: entry.needs_collision(),
                                    variable_lod: false,
                                };
                                if !self.shutdown_in_progress
                                    && !upload.features().contains(demanded_features)
                                {
                                    let revision = next_mesh_revision;
                                    next_mesh_revision = revision
                                        .checked_add(1)
                                        .ok_or(VoxelTerrainRuntimeError::MeshRevisionOverflow)?;
                                    let (generation, physical_request) = allocate_physical_request(
                                        self.request_epoch,
                                        &mut next_request_generation,
                                    )?;
                                    entry.requested_revision = Some(revision);
                                    entry.request_generation = generation;
                                    entry.requested_features = demanded_features;
                                    entry.terminal_retry_count = 0;
                                    entry.physical_request = Some(physical_request);
                                    if !entry.is_in_update_list {
                                        entry.is_in_update_list = true;
                                        pending_mesh.insert(position);
                                    }
                                }
                            }
                            let visual_ready =
                                entry.accepted_upload.as_ref().is_some_and(|value| {
                                    value.visual_state() != PayloadState::NotBuilt
                                });
                            let collision_ready =
                                entry.accepted_upload.as_ref().is_some_and(|value| {
                                    value.collision_state() != PayloadState::NotBuilt
                                });
                            let next_visual_active = entry.needs_visual() && visual_ready;
                            let next_collision_active = entry.needs_collision() && collision_ready;
                            if next_visual_active != entry.visual_active {
                                entry.visual_active = next_visual_active;
                            }
                            if next_collision_active != entry.collision_active {
                                entry.collision_active = next_collision_active;
                            }
                            publish_followups = true;
                            PreparedCompletionDisposition::Retire { publish_followups }
                        }
                    }
                }
                DurableCompletion::MeshTerminal {
                    key, request_tag, ..
                } => {
                    if key.location.lod_index == 0 {
                        let position = key.location.position_in_blocks;
                        let shadow = mesh_shadow.entry(position).or_insert_with(|| {
                            self.mesh_maps[0]
                                .get(&position)
                                .map(MeshBlockEntry::clone_for_draft)
                        });
                        if let Some(entry) = shadow.as_mut().filter(|entry| {
                            entry.resident_refcount() > 0
                                && entry.requested_revision == Some(key.revision)
                                && !entry.is_in_update_list
                        }) {
                            if !self.shutdown_in_progress
                                && entry.matches_previous_epoch_request(
                                    *request_tag,
                                    self.request_epoch,
                                )
                            {
                                let (generation, physical_request) = allocate_physical_request(
                                    self.request_epoch,
                                    &mut next_request_generation,
                                )?;
                                entry.request_generation = generation;
                                entry.physical_request = Some(physical_request);
                                entry.is_in_update_list = true;
                                pending_mesh.insert(position);
                            } else if !self.shutdown_in_progress
                                && entry.matches_physical_request(*request_tag)
                                && request_tag
                                    .is_none_or(|tag| tag.request_epoch == self.request_epoch)
                            {
                                entry.terminal_retry_count =
                                    entry.terminal_retry_count.checked_add(1).ok_or(
                                        VoxelTerrainRuntimeError::MeshTerminalRetryCountOverflow {
                                            key: *key,
                                        },
                                    )?;
                                if entry.terminal_retry_count <= MAX_MESH_TERMINAL_RETRIES {
                                    let (generation, physical_request) = allocate_physical_request(
                                        self.request_epoch,
                                        &mut next_request_generation,
                                    )?;
                                    entry.request_generation = generation;
                                    entry.physical_request = Some(physical_request);
                                    entry.is_in_update_list = true;
                                    pending_mesh.insert(position);
                                } else {
                                    entry.physical_request = None;
                                }
                            }
                        }
                    }
                    PreparedCompletionDisposition::Retire {
                        publish_followups: false,
                    }
                }
                DurableCompletion::SaveAcknowledged {
                    terminal,
                    attempt_ordinal,
                    ..
                } => {
                    if let Some(action) = Self::prepare_save_completion_action(
                        &mut journal_shadow,
                        terminal,
                        *attempt_ordinal,
                    )? {
                        publish_followups = true;
                        PreparedCompletionDisposition::ApplyPersistence {
                            publish_followups,
                            action: PreparedPersistenceAction::Save(action),
                        }
                    } else {
                        PreparedCompletionDisposition::Retire {
                            publish_followups: false,
                        }
                    }
                }
                DurableCompletion::FlushAcknowledged {
                    terminal,
                    attempt_ordinal,
                    ..
                } => {
                    if let Some(action) = self.prepare_checkpoint_completion_action(
                        &mut checkpoint_shadow,
                        &mut journal_shadow,
                        terminal,
                        *attempt_ordinal,
                    )? {
                        publish_followups = true;
                        PreparedCompletionDisposition::ApplyPersistence {
                            publish_followups,
                            action: PreparedPersistenceAction::Checkpoint(action),
                        }
                    } else {
                        PreparedCompletionDisposition::Retire {
                            publish_followups: false,
                        }
                    }
                }
                DurableCompletion::PersistenceTerminal {
                    terminal,
                    attempt_ordinal,
                    ..
                } => {
                    let action = match terminal {
                        PersistenceTaskTerminal::Save(terminal) => {
                            Self::prepare_save_completion_action(
                                &mut journal_shadow,
                                terminal,
                                *attempt_ordinal,
                            )?
                            .map(PreparedPersistenceAction::Save)
                        }
                        PersistenceTaskTerminal::Flush(terminal) => self
                            .prepare_checkpoint_completion_action(
                                &mut checkpoint_shadow,
                                &mut journal_shadow,
                                terminal,
                                *attempt_ordinal,
                            )?
                            .map(PreparedPersistenceAction::Checkpoint),
                    };
                    if let Some(action) = action {
                        let quarantine_after = matches!(
                            action,
                            PreparedPersistenceAction::Save(
                                PreparedSaveAction::MarkIndeterminate { .. }
                            ) | PreparedPersistenceAction::Checkpoint(
                                PreparedCheckpointAction::MarkIndeterminate { .. }
                            )
                        );
                        if quarantine_after {
                            quarantine_count = quarantine_count
                                .checked_add(1)
                                .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                        }
                        PreparedCompletionDisposition::ApplyPersistence {
                            publish_followups: false,
                            action,
                        }
                    } else {
                        quarantine_count = quarantine_count
                            .checked_add(1)
                            .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                        PreparedCompletionDisposition::Quarantine
                    }
                }
                DurableCompletion::MalformedPersistence { .. }
                | DurableCompletion::MalformedFinished { .. }
                | DurableCompletion::UnknownTerminal { .. } => {
                    quarantine_count = quarantine_count
                        .checked_add(1)
                        .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                    PreparedCompletionDisposition::Quarantine
                }
                DurableCompletion::DirectMesh { .. } => {
                    unreachable!("direct completions use their own admitted FIFO")
                }
            };
            if publish_followups {
                let follow_up_count = match completion {
                    DurableCompletion::LoadFinished { completed, .. }
                    | DurableCompletion::MeshFinished { completed, .. }
                    | DurableCompletion::SaveAcknowledged { completed, .. }
                    | DurableCompletion::FlushAcknowledged { completed, .. } => {
                        completed.follow_up_count()
                    }
                    _ => 0,
                };
                accepted_followup_count = accepted_followup_count
                    .checked_add(follow_up_count)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
            }
            completion_plans.push(PreparedCompletionPlan {
                inbox_index,
                descriptor: completion.descriptor(),
                disposition,
                followups: None,
            });
        }

        let mut direct_mesh_plans = Vec::new();
        direct_mesh_plans
            .try_reserve_exact(direct_mesh_prefix_len)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (inbox_index, completion) in self.direct_mesh_retry_inbox.iter().enumerate() {
            direct_mesh_plans.push(PreparedDirectMeshPlan {
                inbox_index,
                descriptor: completion.descriptor(),
            });
            let DurableCompletion::DirectMesh { upload, dropped } = completion else {
                unreachable!("direct mesh FIFO contains only admitted direct uploads")
            };
            let key = upload.key();
            if key.location.lod_index != 0 {
                next_stats.meshes_dropped = next_stats
                    .meshes_dropped
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
                continue;
            }
            let position = key.location.position_in_blocks;
            let shadow = mesh_shadow.entry(position).or_insert_with(|| {
                self.mesh_maps[0]
                    .get(&position)
                    .map(MeshBlockEntry::clone_for_draft)
            });
            let Some(entry) = shadow.as_mut().filter(|entry| {
                entry.requested_revision == Some(key.revision)
                    && entry.applied_revision != Some(key.revision)
            }) else {
                next_stats.meshes_dropped = next_stats
                    .meshes_dropped
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
                continue;
            };
            entry.terminal_retry_count = 0;
            if *dropped {
                next_stats.meshes_dropped = next_stats
                    .meshes_dropped
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
                if !self.shutdown_in_progress && entry.resident_refcount() > 0 {
                    let (generation, physical_request) = allocate_physical_request(
                        self.request_epoch,
                        &mut next_request_generation,
                    )?;
                    entry.request_generation = generation;
                    entry.physical_request = Some(physical_request);
                    entry.is_in_update_list = true;
                    pending_mesh.insert(position);
                }
                continue;
            }
            entry.physical_request = None;
            let had_built_payload = entry
                .accepted_upload
                .as_ref()
                .is_some_and(|accepted| accepted.has_built_payload());
            let has_built_payload = upload.has_built_payload();
            entry.accepted_upload = Some(upload.clone());
            entry.applied_revision = Some(key.revision);
            entry.requested_features = upload.features();
            entry.applied_features = upload.features();
            entry.has_geometry = upload.visual_state() == PayloadState::NonEmpty;
            entry.is_loaded = true;
            next_stats.meshes_built = next_stats
                .meshes_built
                .checked_add(1)
                .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
            events_to_append.push(match (had_built_payload, has_built_payload) {
                (false, true) => VoxelTerrainEvent::MeshBlockEntered(upload.clone()),
                (true, true) => VoxelTerrainEvent::MeshBlockUpdated(upload.clone()),
                (_, false) => VoxelTerrainEvent::MeshBlockBecameEmpty(upload.clone()),
            });
            accepted_feature_updates.push(key.location);
            let demanded_features = MeshBuildFeatures {
                visuals: entry.needs_visual(),
                collisions: entry.needs_collision(),
                variable_lod: false,
            };
            if !self.shutdown_in_progress && !upload.features().contains(demanded_features) {
                let revision = next_mesh_revision;
                next_mesh_revision = revision
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::MeshRevisionOverflow)?;
                let (generation, physical_request) =
                    allocate_physical_request(self.request_epoch, &mut next_request_generation)?;
                entry.requested_revision = Some(revision);
                entry.request_generation = generation;
                entry.requested_features = demanded_features;
                entry.terminal_retry_count = 0;
                entry.physical_request = Some(physical_request);
                if !entry.is_in_update_list {
                    entry.is_in_update_list = true;
                    pending_mesh.insert(position);
                }
            }
            let next_visual_active =
                entry.needs_visual() && upload.visual_state() != PayloadState::NotBuilt;
            let next_collision_active =
                entry.needs_collision() && upload.collision_state() != PayloadState::NotBuilt;
            if next_visual_active != entry.visual_active {
                entry.visual_active = next_visual_active;
            }
            if next_collision_active != entry.collision_active {
                entry.collision_active = next_collision_active;
            }
        }

        topology_changes
            .try_reserve_exact(
                mesh_shadow
                    .len()
                    .checked_mul(2)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (position, next) in &mesh_shadow {
            let location = MeshBlockLocation::new(*position, 0);
            let original = self.mesh_maps[0].get(position);
            let next_visual_active = next.as_ref().is_some_and(|entry| entry.visual_active);
            let next_collision_active = next.as_ref().is_some_and(|entry| entry.collision_active);
            if original.is_some_and(|entry| entry.visual_active) != next_visual_active {
                topology_changes.push((CoverageFeature::Visual, next_visual_active, location));
            }
            if original.is_some_and(|entry| entry.collision_active) != next_collision_active {
                topology_changes.push((
                    CoverageFeature::Collision,
                    next_collision_active,
                    location,
                ));
            }
        }
        topology_changes.sort_unstable_by_key(|(feature, active, location)| {
            (
                match feature {
                    CoverageFeature::Visual => 0,
                    CoverageFeature::Collision => 1,
                },
                !*active,
                location.lod_index,
                location.position_in_blocks.x,
                location.position_in_blocks.y,
                location.position_in_blocks.z,
            )
        });
        let topology_event_index = if topology_changes.is_empty() {
            None
        } else {
            let revision = next_render_topology_revision;
            next_render_topology_revision = revision
                .checked_add(1)
                .ok_or(VoxelTerrainRuntimeError::RenderTopologyRevisionOverflow)?;
            let mut groups = Vec::new();
            groups
                .try_reserve_exact(topology_changes.len())
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            for (feature, active, location) in &topology_changes {
                let mut activate = Vec::new();
                let mut deactivate = Vec::new();
                if *active {
                    activate
                        .try_reserve_exact(1)
                        .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                    activate.push(*location);
                } else {
                    deactivate
                        .try_reserve_exact(1)
                        .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                    deactivate.push(*location);
                }
                groups.push(FeatureTopologyGroup {
                    feature: *feature,
                    operation: if *active {
                        TopologyOperation::RootActivate
                    } else {
                        TopologyOperation::RootDeactivate
                    },
                    anchor: *location,
                    activate,
                    deactivate,
                });
            }
            let batch = RenderTopologyBatch {
                revision,
                groups,
                transition_masks: Vec::new(),
            };
            let event_index = events_to_append.len();
            events_to_append
                .try_reserve(1)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            events_to_append.push(VoxelTerrainEvent::RenderTopologyChanged(batch));
            Some(event_index)
        };
        events_to_append
            .try_reserve(mesh_exit_events.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        events_to_append.extend(mesh_exit_events);

        let mut data_residency_diffs = Vec::new();
        data_residency_diffs
            .try_reserve_exact(data_residency_shadow.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (position, next) in data_residency_shadow {
            let expected = self.loaded_data_residency[0].get(&position).copied();
            if expected == next {
                continue;
            }
            let action = match (expected, next) {
                (None, Some(residency)) => PreparedMapAction::Insert(residency),
                (Some(_), Some(residency)) => PreparedMapAction::Replace(residency),
                (Some(_), None) => PreparedMapAction::Remove,
                (None, None) => continue,
            };
            data_residency_diffs.push(PreparedDataResidencyDiff {
                location: BlockLocation {
                    position,
                    lod_index: 0,
                },
                expected,
                action,
            });
        }
        data_residency_diffs.sort_unstable_by_key(|diff| {
            (
                diff.location.position.z,
                diff.location.position.x,
                diff.location.position.y,
            )
        });

        let mut loading_diffs = Vec::new();
        loading_diffs
            .try_reserve_exact(loading_shadow.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (position, next) in loading_shadow {
            let expected = self.loading_blocks[0]
                .get(&position)
                .map(|entry| entry.request_generation);
            let action = match (expected, next) {
                (None, Some(entry)) => PreparedMapAction::Insert(entry),
                (Some(_), Some(entry)) => PreparedMapAction::Replace(entry),
                (Some(_), None) => PreparedMapAction::Remove,
                (None, None) => continue,
            };
            loading_diffs.push(PreparedLoadingEntryDiff {
                location: BlockLocation {
                    position,
                    lod_index: 0,
                },
                expected_generation: expected,
                action,
            });
        }
        loading_diffs.sort_unstable_by_key(|diff| {
            (
                diff.location.position.z,
                diff.location.position.x,
                diff.location.position.y,
            )
        });
        let mut mesh_diffs = Vec::new();
        mesh_diffs
            .try_reserve_exact(mesh_shadow.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (position, next) in mesh_shadow {
            let expected = self.mesh_maps[0]
                .get(&position)
                .and_then(|entry| entry.requested_revision);
            let action = match (self.mesh_maps[0].contains_key(&position), next) {
                (false, Some(entry)) => PreparedMapAction::Insert(entry),
                (true, Some(entry)) => PreparedMapAction::Replace(entry),
                (true, None) => PreparedMapAction::Remove,
                (false, None) => continue,
            };
            mesh_diffs.push(PreparedMeshEntryDiff {
                location: MeshBlockLocation::new(position, 0),
                expected_revision: expected,
                action,
            });
        }
        mesh_diffs.sort_unstable_by_key(|diff| {
            (
                diff.location.position_in_blocks.z,
                diff.location.position_in_blocks.x,
                diff.location.position_in_blocks.y,
            )
        });

        let mut pending_load_positions = Vec::new();
        pending_load_positions
            .try_reserve_exact(pending_load.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        pending_load_positions.extend(pending_load);
        pending_load_positions
            .sort_unstable_by_key(|position| (position.x, position.y, position.z));
        let mut pending_mesh_positions = Vec::new();
        pending_mesh_positions
            .try_reserve_exact(pending_mesh.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        pending_mesh_positions.extend(pending_mesh);
        pending_mesh_positions
            .sort_unstable_by_key(|position| (position.x, position.y, position.z));

        let mut loading_diff_indices = HashMap::new();
        loading_diff_indices
            .try_reserve(loading_diffs.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (index, diff) in loading_diffs.iter().enumerate() {
            loading_diff_indices.insert(diff.location.position, index);
        }
        let mut mesh_diff_indices = HashMap::new();
        mesh_diff_indices
            .try_reserve(mesh_diffs.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (index, diff) in mesh_diffs.iter().enumerate() {
            mesh_diff_indices.insert(diff.location.position_in_blocks, index);
        }

        let mut load_tasks = Vec::new();
        load_tasks
            .try_reserve_exact(pending_load_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut mesh_tasks = Vec::new();
        mesh_tasks
            .try_reserve_exact(pending_mesh_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        #[cfg(test)]
        if !pending_load_positions.is_empty() {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::PendingLoadQueue)?;
        }
        let mut final_pending_load = Vec::new();
        final_pending_load
            .try_reserve_exact(pending_load_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for position in pending_load_positions {
            let diff_index = loading_diff_indices.get(&position).copied();
            let entry = match diff_index.map(|index| &mut loading_diffs[index].action) {
                Some(PreparedMapAction::Insert(entry) | PreparedMapAction::Replace(entry)) => entry,
                Some(PreparedMapAction::Remove) => continue,
                None => continue,
            };
            if entry.request_state != LoadRequestState::Queued {
                continue;
            }
            if !dispatch_load_mesh_tasks {
                final_pending_load.push(position);
                continue;
            }
            entry.request_state = LoadRequestState::InFlight;
            if entry.physical_request.as_ref().is_none_or(|request| {
                request.tag.request_epoch != self.request_epoch
                    || request.tag.request_generation != entry.request_generation
                    || request.cancellation.is_cancelled()
            }) {
                let (generation, physical_request) =
                    allocate_physical_request(self.request_epoch, &mut next_request_generation)?;
                entry.request_generation = generation;
                entry.physical_request = Some(physical_request);
            }
            let request_generation = entry.request_generation;
            let physical_request = entry
                .physical_request
                .as_ref()
                .expect("fixed queued load owns its physical request")
                .clone();
            load_tasks.push(ScheduledTask::new(
                Box::new(
                    LoadBlockForTerrainTask::new(
                        position,
                        0,
                        request_generation,
                        self.data.clone(),
                        self.stream.clone(),
                    )
                    .with_request_control(physical_request.tag, physical_request.cancellation),
                ),
                TaskLane::Parallel,
            ));
        }
        #[cfg(test)]
        if !pending_mesh_positions.is_empty() {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::PendingMeshQueue)?;
        }
        let mut final_pending_mesh = Vec::new();
        final_pending_mesh
            .try_reserve_exact(pending_mesh_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for position in pending_mesh_positions {
            let data_is_ready = self.fixed_draft_mesh_data_is_ready(
                &data_preview,
                position,
                &mut data_presence,
                &mut data_snapshots,
            )?;
            let diff_index = mesh_diff_indices.get(&position).copied();
            let entry = match diff_index.map(|index| &mut mesh_diffs[index].action) {
                Some(PreparedMapAction::Insert(entry) | PreparedMapAction::Replace(entry)) => entry,
                Some(PreparedMapAction::Remove) => continue,
                None => {
                    continue;
                }
            };
            if entry.resident_refcount() == 0 || entry.requested_revision.is_none() {
                entry.is_in_update_list = false;
                continue;
            }
            if !data_is_ready {
                entry.is_in_update_list = true;
                final_pending_mesh.push(position);
                continue;
            }
            if !dispatch_load_mesh_tasks {
                entry.is_in_update_list = true;
                final_pending_mesh.push(position);
                continue;
            }
            entry.is_in_update_list = false;
            let key = MeshBlockKey {
                location: MeshBlockLocation::new(position, 0),
                revision: entry
                    .requested_revision
                    .expect("prepared pending mesh retains its checked revision"),
            };
            if entry.physical_request.as_ref().is_none_or(|request| {
                request.tag.request_epoch != self.request_epoch
                    || request.tag.request_generation != entry.request_generation
                    || request.cancellation.is_cancelled()
            }) {
                let (generation, physical_request) =
                    allocate_physical_request(self.request_epoch, &mut next_request_generation)?;
                entry.request_generation = generation;
                entry.physical_request = Some(physical_request);
            }
            let physical_request = entry
                .physical_request
                .as_ref()
                .expect("fixed queued mesh owns its physical request")
                .clone();
            mesh_tasks.push(ScheduledTask::new(
                Box::new(
                    MeshBlockTask::new(MeshBlockTaskParams {
                        key,
                        data: self.data.clone(),
                        meshing_dependency: self.meshing_dependency.clone(),
                        collision_hint: entry.needs_collision(),
                        lod_hint: false,
                        mesh_arrays_pool: Some(self.mesh_arrays_pool.clone()),
                    })
                    .with_request_control(physical_request.tag, physical_request.cancellation),
                ),
                TaskLane::Parallel,
            ));
        }
        data_snapshots.sort_unstable_by_key(|snapshot| {
            (
                snapshot.location().lod_index,
                snapshot.location().position.x,
                snapshot.location().position.y,
                snapshot.location().position.z,
            )
        });
        data_snapshots.dedup_by_key(|snapshot| snapshot.location());

        #[cfg(test)]
        if vacant_journal_count != 0 {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::SaveJournal)?;
        }
        self.save_journal
            .try_reserve(vacant_journal_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        #[cfg(test)]
        if journal_shadow.iter().any(|(key, shadow)| {
            self.save_journal
                .get(key)
                .is_some_and(|entry| shadow.queued_len > entry.queued_newer.len())
        }) {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::SaveJournalQueue)?;
        }
        for (key, shadow) in &journal_shadow {
            let Some(entry) = self.save_journal.get_mut(key) else {
                continue;
            };
            // Only reserve additional capacity. A prepared action may consume
            // queued successors, in which case the required growth is zero.
            let queue_growth = shadow.queued_len.saturating_sub(entry.queued_newer.len());
            entry
                .queued_newer
                .try_reserve(queue_growth)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        }
        #[cfg(test)]
        if dirty_retained_count != 0 {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::DirtyRetention)?;
        }
        self.retained_save_admission_failures
            .try_reserve(dirty_retained_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        #[cfg(test)]
        if quarantine_count != 0 {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::Quarantine)?;
        }
        self.completion_quarantine
            .try_reserve(quarantine_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;

        let mut resolution_force_checkpoint = false;
        let mut resolution_clears_force_checkpoint = false;
        let mut resolution_unblocks_checkpoint = false;
        let mut resolution = None;
        if let Some(request) = persistence_resolution_request {
            match request.operation {
                PersistenceOperation::Save {
                    location,
                    block_revision,
                    save_generation,
                } => {
                    let key = SaveKey::new(location.position, location.lod_index);
                    let Some(entry) = self.save_journal.get(&key) else {
                        return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                            requested: request.operation,
                        });
                    };
                    let Some(ActiveSaveAttempt::Indeterminate {
                        meta,
                        attempt_ordinal,
                    }) = entry.active.as_ref()
                    else {
                        return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                            requested: request.operation,
                        });
                    };
                    if meta.block_revision != block_revision || meta.generation != save_generation {
                        return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                            requested: request.operation,
                        });
                    }
                    let attempt_ordinal = *attempt_ordinal;
                    let retry_count = meta.retry_count;
                    let Some(quarantine_index) =
                        self.completion_quarantine.iter().position(|completion| {
                            matches!(
                                completion,
                                QuarantinedCompletion::Persistence {
                                    kind: PersistenceTaskKind::Save,
                                    terminal: PersistenceTaskTerminal::Save(terminal),
                                    attempt_ordinal: terminal_attempt,
                                    ..
                                } if terminal.location == location
                                    && terminal.block_revision == block_revision
                                    && terminal.save_generation == save_generation
                                    && *terminal_attempt == attempt_ordinal
                                    && terminal.phase == PersistenceIoPhase::CallEntered
                                    && terminal.acknowledgement.is_none()
                            )
                        })
                    else {
                        return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                            requested: request.operation,
                        });
                    };
                    let shadow = journal_shadow.get_mut(&key).unwrap_or_else(|| {
                        unreachable!("indeterminate save shadow remains present")
                    });
                    if shadow.active
                        != (PreparedJournalActiveShadow::Indeterminate {
                            block_revision,
                            generation: save_generation,
                            attempt_ordinal,
                        })
                    {
                        return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                            requested: request.operation,
                        });
                    }
                    match request.resolution {
                        IndeterminateIoResolution::AssumeNotWrittenAndRetry => {
                            if self
                                .next_persistence_attempt_ordinal
                                .checked_add(1)
                                .is_none()
                            {
                                return Err(VoxelTerrainRuntimeError::PersistenceAttemptOverflow {
                                    operation: request.operation,
                                });
                            }
                            shadow.active = PreparedJournalActiveShadow::Pending {
                                block_revision,
                                generation: save_generation,
                                retry_count,
                            };
                            resolution = Some(PreparedFixedPersistenceResolution::Save {
                                quarantine_index,
                                key,
                                block_revision,
                                generation: save_generation,
                                attempt_ordinal,
                                assume_written: false,
                            });
                        }
                        IndeterminateIoResolution::AssumeWrittenAndFlush => {
                            if shadow.written_generation.is_some() {
                                return Err(
                                    VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                                        requested: request.operation,
                                    },
                                );
                            }
                            shadow.active = PreparedJournalActiveShadow::None;
                            shadow.written_block_revision = Some(block_revision);
                            shadow.written_generation = Some(save_generation);
                            resolution_force_checkpoint = true;
                            resolution_unblocks_checkpoint = true;
                            resolution = Some(PreparedFixedPersistenceResolution::Save {
                                quarantine_index,
                                key,
                                block_revision,
                                generation: save_generation,
                                attempt_ordinal,
                                assume_written: true,
                            });
                        }
                    }
                }
                PersistenceOperation::Flush {
                    checkpoint_generation,
                } => {
                    let Some(checkpoint) = self.save_checkpoint_in_flight.as_ref() else {
                        return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                            requested: request.operation,
                        });
                    };
                    let CheckpointAttemptState::Indeterminate { attempt_ordinal } =
                        checkpoint.state
                    else {
                        return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                            requested: request.operation,
                        });
                    };
                    if checkpoint.checkpoint_generation != checkpoint_generation {
                        return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                            requested: request.operation,
                        });
                    }
                    let Some(quarantine_index) =
                        self.completion_quarantine.iter().position(|completion| {
                            matches!(
                                completion,
                                QuarantinedCompletion::Persistence {
                                    kind: PersistenceTaskKind::Flush,
                                    terminal: PersistenceTaskTerminal::Flush(terminal),
                                    attempt_ordinal: terminal_attempt,
                                    ..
                                } if terminal.checkpoint_generation == checkpoint_generation
                                    && *terminal_attempt == attempt_ordinal
                                    && terminal.phase == PersistenceIoPhase::CallEntered
                                    && terminal.acknowledgement.is_none()
                            )
                        })
                    else {
                        return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                            requested: request.operation,
                        });
                    };
                    let shadow = checkpoint_shadow.as_mut().unwrap_or_else(|| {
                        unreachable!("indeterminate checkpoint shadow remains present")
                    });
                    if shadow.checkpoint_generation != checkpoint_generation
                        || shadow.state
                            != (CheckpointAttemptState::Indeterminate { attempt_ordinal })
                    {
                        return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                            requested: request.operation,
                        });
                    }
                    match request.resolution {
                        IndeterminateIoResolution::AssumeNotWrittenAndRetry => {
                            if self
                                .next_persistence_attempt_ordinal
                                .checked_add(1)
                                .is_none()
                            {
                                return Err(VoxelTerrainRuntimeError::PersistenceAttemptOverflow {
                                    operation: request.operation,
                                });
                            }
                            shadow.state = CheckpointAttemptState::Pending;
                            resolution =
                                Some(PreparedFixedPersistenceResolution::CheckpointRetry {
                                    quarantine_index,
                                    checkpoint_generation,
                                    attempt_ordinal,
                                });
                        }
                        IndeterminateIoResolution::AssumeWrittenAndFlush => {
                            let mut entry_actions = Vec::new();
                            entry_actions
                                .try_reserve_exact(checkpoint.acknowledged.len())
                                .map_err(|_| {
                                    VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
                                })?;
                            for snapshot in &checkpoint.acknowledged {
                                let Some(entry) = journal_shadow.get_mut(&snapshot.key) else {
                                    continue;
                                };
                                if entry.written_block_revision != Some(snapshot.block_revision)
                                    || entry.written_generation != Some(snapshot.generation)
                                {
                                    continue;
                                }
                                entry.written_block_revision = None;
                                entry.written_generation = None;
                                let promote_queued = entry.active
                                    == PreparedJournalActiveShadow::None
                                    && entry.queued_len != 0;
                                if promote_queued {
                                    entry.queued_len -= 1;
                                    let (block_revision, generation, retry_count) =
                                        entry.queued_front.unwrap_or_else(|| {
                                            unreachable!(
                                                "prepared queued save retains front metadata"
                                            )
                                        });
                                    entry.active = PreparedJournalActiveShadow::Pending {
                                        block_revision,
                                        generation,
                                        retry_count,
                                    };
                                }
                                let defer_active = matches!(
                                    entry.active,
                                    PreparedJournalActiveShadow::Pending { .. }
                                );
                                let remove_entry = entry.active
                                    == PreparedJournalActiveShadow::None
                                    && entry.queued_len == 0;
                                entry_actions.push(PreparedCheckpointEntryAction::ClearWritten {
                                    key: snapshot.key,
                                    block_revision: snapshot.block_revision,
                                    generation: snapshot.generation,
                                    promote_queued,
                                    defer_active,
                                    remove_entry,
                                });
                            }
                            checkpoint_shadow = None;
                            resolution_unblocks_checkpoint = true;
                            resolution_clears_force_checkpoint = true;
                            resolution =
                                Some(PreparedFixedPersistenceResolution::CheckpointWritten {
                                    quarantine_index,
                                    checkpoint_generation,
                                    attempt_ordinal,
                                    entry_actions,
                                });
                        }
                    }
                }
            }
        }

        let mut save_recovery_updates = Vec::new();
        if persistence_recovery_request.is_some_and(|request| request.reset_pending_save_failures) {
            save_recovery_updates
                .try_reserve_exact(journal_shadow.len())
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            for (key, shadow) in &mut journal_shadow {
                let PreparedJournalActiveShadow::Pending {
                    block_revision,
                    generation,
                    retry_count,
                } = &mut shadow.active
                else {
                    continue;
                };
                *retry_count = 0;
                save_recovery_updates.push(PreparedSaveRecoveryUpdate {
                    key: *key,
                    block_revision: *block_revision,
                    generation: *generation,
                });
            }
            save_recovery_updates.sort_unstable_by_key(|update| {
                (
                    update.key.lod_index,
                    update.key.position.x,
                    update.key.position.y,
                    update.key.position.z,
                    update.generation,
                )
            });
        }

        let mut next_automatic_save_checkpoint_blocked =
            completion_plans
                .iter()
                .filter_map(|plan| match &plan.disposition {
                    PreparedCompletionDisposition::ApplyPersistence {
                        action:
                            PreparedPersistenceAction::Checkpoint(
                                PreparedCheckpointAction::Acknowledge { succeeded, .. },
                            ),
                        ..
                    } => Some(!*succeeded),
                    _ => None,
                })
                .next_back()
                .unwrap_or(self.automatic_save_checkpoint_blocked);
        if persistence_recovery_request
            .is_some_and(|request| request.authorize_automatic_checkpoint)
            || resolution_unblocks_checkpoint
        {
            next_automatic_save_checkpoint_blocked = false;
        }
        let checkpoint_completion_in_prefix = completion_plans.iter().any(|plan| {
            matches!(
                &plan.disposition,
                PreparedCompletionDisposition::ApplyPersistence {
                    action: PreparedPersistenceAction::Checkpoint(
                        PreparedCheckpointAction::Acknowledge { .. }
                    ),
                    ..
                }
            )
        });
        let next_force_checkpoint_requested =
            if checkpoint_completion_in_prefix || resolution_clears_force_checkpoint {
                false
            } else {
                self.force_checkpoint_requested || resolution_force_checkpoint
            };

        let mut defer_save_this_transaction = Vec::new();
        defer_save_this_transaction
            .try_reserve_exact(
                completion_count
                    .checked_add(checkpoint_key_count)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut defer_checkpoint_this_transaction = false;
        for plan in &completion_plans {
            let PreparedCompletionDisposition::ApplyPersistence { action, .. } = &plan.disposition
            else {
                continue;
            };
            match action {
                PreparedPersistenceAction::Save(
                    PreparedSaveAction::AcknowledgeFailure { key, .. }
                    | PreparedSaveAction::RestoreBeforeIo { key, .. },
                ) => {
                    if !defer_save_this_transaction.contains(key) {
                        defer_save_this_transaction.push(*key);
                    }
                }
                PreparedPersistenceAction::Checkpoint(PreparedCheckpointAction::Acknowledge {
                    entry_actions,
                    ..
                }) => {
                    for entry_action in entry_actions {
                        let PreparedCheckpointEntryAction::ClearWritten {
                            key,
                            defer_active: true,
                            ..
                        } = entry_action
                        else {
                            continue;
                        };
                        if !defer_save_this_transaction.contains(key) {
                            defer_save_this_transaction.push(*key);
                        }
                    }
                }
                PreparedPersistenceAction::Checkpoint(
                    PreparedCheckpointAction::RestoreBeforeIo { .. },
                ) => defer_checkpoint_this_transaction = true,
                PreparedPersistenceAction::Save(
                    PreparedSaveAction::AcknowledgeSuccess { .. }
                    | PreparedSaveAction::MarkIndeterminate { .. },
                )
                | PreparedPersistenceAction::Checkpoint(
                    PreparedCheckpointAction::MarkIndeterminate { .. },
                ) => {}
            }
        }
        if let Some(PreparedFixedPersistenceResolution::CheckpointWritten {
            entry_actions, ..
        }) = &resolution
        {
            for entry_action in entry_actions {
                let PreparedCheckpointEntryAction::ClearWritten {
                    key,
                    defer_active: true,
                    ..
                } = entry_action
                else {
                    continue;
                };
                if !defer_save_this_transaction.contains(key) {
                    defer_save_this_transaction.push(*key);
                }
            }
        }

        let mut checkpoint_config_update = None;
        if let (Some(request), Some(shadow)) = (checkpoint_request, checkpoint_shadow.as_mut()) {
            if request.reset_pending_retry_count && shadow.state == CheckpointAttemptState::Pending
            {
                shadow.retry_count = 0;
            }
            shadow.max_attempts = request.max_attempts;
            shadow.origin = request.origin;
            shadow.record_per_block_failure = request.record_per_block_failure;
            checkpoint_config_update = Some(PreparedCheckpointConfigUpdate {
                checkpoint_generation: shadow.checkpoint_generation,
                retry_count: shadow.retry_count,
                max_attempts: shadow.max_attempts,
                origin: shadow.origin,
                record_per_block_failure: shadow.record_per_block_failure,
            });
        }

        let mut checkpoint_begin = None;
        if checkpoint_shadow.is_none()
            && (checkpoint_request.is_some()
                || (!next_automatic_save_checkpoint_blocked && !checkpoint_completion_in_prefix))
        {
            let mut acknowledged = Vec::new();
            acknowledged
                .try_reserve_exact(self.save_journal.len())
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            let mut allocated_bytes = 0usize;
            let mut successor_is_blocked = false;
            let mut persistence_in_flight = false;
            for (key, live_entry) in &self.save_journal {
                let shadow = journal_shadow.get(key);
                let written_block_revision = shadow
                    .map(|shadow| shadow.written_block_revision)
                    .unwrap_or_else(|| {
                        live_entry
                            .written_unflushed
                            .as_ref()
                            .map(|written| written.block_revision)
                    });
                let written_generation = shadow
                    .map(|shadow| shadow.written_generation)
                    .unwrap_or_else(|| {
                        live_entry
                            .written_unflushed
                            .as_ref()
                            .map(|written| written.generation)
                    });
                let active = shadow
                    .map(|shadow| shadow.active)
                    .unwrap_or_else(|| PreparedJournalShadow::from_entry(live_entry).active);
                persistence_in_flight |= matches!(
                    active,
                    PreparedJournalActiveShadow::WriteInFlight { .. }
                        | PreparedJournalActiveShadow::Indeterminate { .. }
                );
                let queued_len =
                    shadow.map_or(live_entry.queued_newer.len(), |shadow| shadow.queued_len);
                let Some(generation) = written_generation else {
                    continue;
                };
                let block_revision = written_block_revision
                    .expect("written save generation retains its paired block revision");
                acknowledged.push(SaveCheckpointSnapshot {
                    key: *key,
                    block_revision,
                    generation,
                });
                successor_is_blocked |= queued_len != 0;
                let payload_bytes = live_entry
                    .written_unflushed
                    .as_ref()
                    .filter(|written| {
                        written.block_revision == block_revision && written.generation == generation
                    })
                    .map(|written| {
                        (0..MAX_CHANNELS)
                            .map(|channel| written.payload.channel_bytes(channel).len())
                            .sum::<usize>()
                    })
                    .or_else(|| {
                        self.durable_completion_inbox.iter().find_map(|completion| {
                            let terminal = match completion {
                                DurableCompletion::SaveAcknowledged { terminal, .. }
                                | DurableCompletion::PersistenceTerminal {
                                    terminal: PersistenceTaskTerminal::Save(terminal),
                                    ..
                                } => terminal,
                                _ => return None,
                            };
                            (terminal.block_revision == block_revision
                                && terminal.save_generation == generation
                                && terminal.location.position == key.position
                                && terminal.location.lod_index == key.lod_index)
                                .then(|| {
                                    (0..MAX_CHANNELS)
                                        .map(|channel| {
                                            terminal.payload.channel_bytes(channel).len()
                                        })
                                        .sum::<usize>()
                                })
                        })
                    })
                    .unwrap_or(0);
                allocated_bytes = allocated_bytes
                    .checked_add(payload_bytes)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
            }
            acknowledged.sort_unstable_by_key(|snapshot| {
                (
                    snapshot.key.lod_index,
                    snapshot.key.position.x,
                    snapshot.key.position.y,
                    snapshot.key.position.z,
                )
            });
            let should_checkpoint = checkpoint_request.is_some()
                || (!persistence_in_flight
                    && (next_force_checkpoint_requested
                        || successor_is_blocked
                        || acknowledged.len() >= AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD
                        || allocated_bytes >= AUTOMATIC_SAVE_CHECKPOINT_BYTE_THRESHOLD));
            if should_checkpoint {
                let checkpoint_generation =
                    allocate_persistence_generation(&mut next_save_checkpoint_generation)?;
                let request = checkpoint_request.unwrap_or(FixedCheckpointRequest {
                    origin: CheckpointOrigin::Automatic,
                    max_attempts: MAX_AUTOMATIC_CHECKPOINT_ATTEMPTS,
                    record_per_block_failure: false,
                    reset_pending_retry_count: false,
                });
                let checkpoint = SaveCheckpointInFlight {
                    checkpoint_generation,
                    acknowledged,
                    state: CheckpointAttemptState::Pending,
                    retry_count: 0,
                    max_attempts: request.max_attempts,
                    origin: request.origin,
                    record_per_block_failure: request.record_per_block_failure,
                };
                checkpoint_shadow = Some(PreparedCheckpointShadow::from_checkpoint(&checkpoint));
                checkpoint_begin = Some(checkpoint);
            }
        }

        let mut save_dispatch_candidates = Vec::new();
        save_dispatch_candidates
            .try_reserve_exact(journal_shadow.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (key, shadow) in &journal_shadow {
            let PreparedJournalActiveShadow::Pending {
                block_revision,
                generation,
                retry_count,
            } = shadow.active
            else {
                continue;
            };
            let max_save_attempts = if self.shutdown_in_progress {
                MAX_EXPLICIT_SAVE_ATTEMPTS
            } else {
                MAX_AUTOMATIC_SAVE_ATTEMPTS
            };
            if shadow.written_generation.is_none()
                && retry_count < max_save_attempts
                && !defer_save_this_transaction.contains(key)
            {
                save_dispatch_candidates.push((*key, block_revision, generation, retry_count));
            }
        }
        save_dispatch_candidates.sort_unstable_by_key(|(key, block_revision, generation, _)| {
            (
                key.lod_index,
                key.position.x,
                key.position.y,
                key.position.z,
                *block_revision,
                *generation,
            )
        });
        let mut save_dispatches = Vec::new();
        save_dispatches
            .try_reserve_exact(save_dispatch_candidates.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        persistence_tasks
            .try_reserve(
                save_dispatch_candidates
                    .len()
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        if checkpoint_shadow.is_none()
            && !next_automatic_save_checkpoint_blocked
            && dispatch_persistence_tasks
        {
            for (key, block_revision, generation, retry_count) in save_dispatch_candidates {
                let attempt_ordinal = next_persistence_attempt_ordinal;
                next_persistence_attempt_ordinal = attempt_ordinal.checked_add(1).ok_or(
                    VoxelTerrainRuntimeError::PersistenceAttemptOverflow {
                        operation: PersistenceOperation::Save {
                            location: BlockLocation {
                                position: key.position,
                                lod_index: key.lod_index,
                            },
                            block_revision,
                            save_generation: generation,
                        },
                    },
                )?;
                #[allow(unused_mut)]
                let mut task = PreparedSaveBlockDataTask::new(
                    BlockLocation {
                        position: key.position,
                        lod_index: key.lod_index,
                    },
                    block_revision,
                    generation,
                    StreamingDependency::new(self.stream.clone()),
                    None,
                    attempt_ordinal,
                );
                #[cfg(test)]
                if save_panic_hook_available {
                    task.set_panic_before_io_for_test(true);
                    save_panic_hook_available = false;
                }
                let (scheduled, installer) = task.into_scheduled();
                persistence_tasks.push((key, generation, scheduled));
                save_dispatches.push(PreparedPendingSaveDispatch {
                    key,
                    block_revision,
                    generation,
                    attempt_ordinal,
                    installer,
                });
                let shadow = journal_shadow
                    .get_mut(&key)
                    .unwrap_or_else(|| unreachable!("prepared save dispatch key remains live"));
                shadow.active = PreparedJournalActiveShadow::WriteInFlight {
                    block_revision,
                    generation,
                    attempt_ordinal,
                    retry_count,
                };
            }
        }

        let mut checkpoint_dispatch = None;
        let mut checkpoint_task = None;
        #[cfg(test)]
        let mut next_panic_flush_before_io_attempts_for_test =
            self.panic_next_flush_before_io_attempts_for_test;
        #[cfg(test)]
        let mut next_panic_flush_after_ack_for_test = self.panic_next_flush_after_ack_for_test;
        if let Some(shadow) = checkpoint_shadow.as_mut().filter(|shadow| {
            shadow.state == CheckpointAttemptState::Pending
                && shadow.retry_count < shadow.max_attempts
                && !defer_checkpoint_this_transaction
                && dispatch_persistence_tasks
        }) {
            let attempt_ordinal = next_persistence_attempt_ordinal;
            next_persistence_attempt_ordinal = attempt_ordinal.checked_add(1).ok_or(
                VoxelTerrainRuntimeError::PersistenceAttemptOverflow {
                    operation: PersistenceOperation::Flush {
                        checkpoint_generation: shadow.checkpoint_generation,
                    },
                },
            )?;
            #[allow(unused_mut)]
            let mut task = FlushVoxelStreamTask::new_with_attempt_ordinal(
                self.stream.clone(),
                shadow.checkpoint_generation,
                attempt_ordinal,
            );
            #[cfg(test)]
            {
                if next_panic_flush_before_io_attempts_for_test != 0 {
                    task.set_panic_before_io_for_test(true);
                    next_panic_flush_before_io_attempts_for_test -= 1;
                }
                if next_panic_flush_after_ack_for_test {
                    task.set_panic_after_ack_for_test(true);
                    next_panic_flush_after_ack_for_test = false;
                }
            }
            checkpoint_task = Some(ScheduledTask::new(Box::new(task), TaskLane::Serial));
            checkpoint_dispatch = Some(PreparedCheckpointDispatch {
                checkpoint_generation: shadow.checkpoint_generation,
                attempt_ordinal,
            });
            shadow.state = CheckpointAttemptState::WriteInFlight { attempt_ordinal };
        }
        persistence_tasks.sort_unstable_by_key(|(key, generation, _)| {
            (
                key.lod_index,
                key.position.x,
                key.position.y,
                key.position.z,
                *generation,
            )
        });
        let next_deferred_save_dispatch_keys = defer_save_this_transaction;
        let next_deferred_checkpoint_dispatch = defer_checkpoint_this_transaction;
        #[cfg(test)]
        if !next_deferred_save_dispatch_keys.is_empty() {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::DeferredSaveQueue)?;
        }

        let mut pending_load_queues = Vec::new();
        pending_load_queues
            .try_reserve_exact(1)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        pending_load_queues.push(PreparedQueueDiff {
            lod_index: 0,
            final_values: final_pending_load,
        });
        let mut pending_mesh_queues = Vec::new();
        pending_mesh_queues
            .try_reserve_exact(1)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        pending_mesh_queues.push(PreparedQueueDiff {
            lod_index: 0,
            final_values: final_pending_mesh,
        });
        self.try_reserve_prepared_runtime_publication(
            &mesh_diffs,
            &data_residency_diffs,
            &loading_diffs,
            &pending_load_queues,
            &pending_mesh_queues,
        )?;
        #[cfg(test)]
        if events_to_append
            .iter()
            .any(|event| event.mesh_descriptor().is_some())
            && std::mem::take(&mut self.fail_next_mesh_event_reservation_for_test)
        {
            return Err(VoxelTerrainRuntimeError::MeshOutputApplyFailed);
        }
        #[cfg(test)]
        if !events_to_append.is_empty() {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::EventOutbox)?;
        }
        self.event_outbox
            .try_reserve(events_to_append.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let retirement_count = mesh_diffs
            .len()
            .checked_add(loading_diffs.len())
            .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
        let mut checkpoint_entry_action_count = 0usize;
        let mut checkpoint_remove_entry_count = 0usize;
        for plan in &completion_plans {
            if let PreparedCompletionDisposition::ApplyPersistence {
                action:
                    PreparedPersistenceAction::Checkpoint(PreparedCheckpointAction::Acknowledge {
                        entry_actions,
                        ..
                    }),
                ..
            } = &plan.disposition
            {
                checkpoint_entry_action_count = checkpoint_entry_action_count
                    .checked_add(entry_actions.len())
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                checkpoint_remove_entry_count = checkpoint_remove_entry_count
                    .checked_add(
                        entry_actions
                            .iter()
                            .filter(|entry_action| {
                                matches!(
                                    entry_action,
                                    PreparedCheckpointEntryAction::ClearWritten {
                                        remove_entry: true,
                                        ..
                                    }
                                )
                            })
                            .count(),
                    )
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
            }
        }
        if let Some(PreparedFixedPersistenceResolution::CheckpointWritten {
            entry_actions, ..
        }) = &resolution
        {
            checkpoint_entry_action_count = checkpoint_entry_action_count
                .checked_add(entry_actions.len())
                .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
            checkpoint_remove_entry_count = checkpoint_remove_entry_count
                .checked_add(
                    entry_actions
                        .iter()
                        .filter(|entry_action| {
                            matches!(
                                entry_action,
                                PreparedCheckpointEntryAction::ClearWritten {
                                    remove_entry: true,
                                    ..
                                }
                            )
                        })
                        .count(),
                )
                .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
        }
        let resolution_count = usize::from(resolution.is_some());
        #[cfg(test)]
        self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::Retirement)?;
        let mut retirement = RetirementBag::default();
        retirement
            .mesh_entries
            .try_reserve_exact(retirement_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .loading_entries
            .try_reserve_exact(retirement_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .data_blocks
            .try_reserve_exact(removed_owner_routes.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .completed_tasks
            .try_reserve_exact(
                completion_count
                    .checked_add(resolution_count)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .durable_completions
            .try_reserve_exact(completion_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .mesh_snapshots
            .try_reserve_exact(direct_mesh_prefix_len)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .save_journal_entries
            .try_reserve_exact(
                vacant_journal_count
                    .checked_add(checkpoint_remove_entry_count)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .followup_escrows
            .try_reserve_exact(completion_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .save_attempt_meta
            .try_reserve_exact(
                completion_count
                    .checked_add(resolution_count)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .checkpoints
            .try_reserve_exact(
                completion_count
                    .checked_add(resolution_count)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .written_saves
            .try_reserve_exact(checkpoint_entry_action_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .stream_errors
            .try_reserve_exact(
                completion_count
                    .checked_add(resolution_count)
                    .and_then(|count| count.checked_add(save_recovery_updates.len()))
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        #[cfg(test)]
        retirement
            .stream_error_drop_probes
            .try_reserve_exact(save_recovery_updates.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .checkpoint_outcomes
            .try_reserve_exact(
                completion_count
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .checkpoint_actions
            .try_reserve_exact(
                completion_count
                    .checked_add(resolution_count)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .runtime_errors
            .try_reserve_exact(1)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;

        let mut active_feature_updates = Vec::new();
        active_feature_updates
            .try_reserve_exact(topology_changes.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        active_feature_updates.extend(topology_changes.iter().map(|(_, _, location)| *location));

        let scheduled_task_count = persistence_tasks
            .len()
            .checked_add(load_tasks.len())
            .and_then(|count| count.checked_add(mesh_tasks.len()))
            .and_then(|count| count.checked_add(usize::from(checkpoint_task.is_some())))
            .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
        let task_count = scheduled_task_count
            .checked_add(accepted_followup_count)
            .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
        #[cfg(test)]
        let task_count = task_count
            .checked_add(std::mem::take(&mut self.fixed_task_count_bias_for_test))
            .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
        #[cfg(test)]
        if accepted_followup_count != 0
            && std::mem::take(&mut self.fail_next_follow_up_reservation_for_test)
        {
            return Err(VoxelTerrainRuntimeError::CompletionFollowUpReservationFailed);
        }
        #[cfg(test)]
        if task_count != 0 {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::PreparedTaskBatch)?;
        }
        let mut prepared_task_batch = self
            .task_runner
            .try_prepare_enqueue(task_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for plan in &mut completion_plans {
            if !plan.disposition.publishes_followups() {
                continue;
            }
            let completion = self
                .durable_completion_inbox
                .get_mut(plan.inbox_index)
                .expect("prepared completion index remains in the durable prefix");
            debug_assert_eq!(completion.descriptor(), plan.descriptor);
            let completed = completion
                .completed_mut()
                .expect("accepted completion retains its runner task owner");
            let mut escrow = completed.prepare_follow_up_take();
            debug_assert_eq!(escrow.len(), plan.descriptor.owner.follow_up_count());
            for task in escrow.drain() {
                if prepared_task_batch.push_reserved(task).is_err() {
                    unreachable!("checked exact follow-up count cannot overfill the batch");
                }
            }
            plan.followups = Some(escrow);
        }
        for task in persistence_tasks
            .into_iter()
            .map(|(_, _, task)| task)
            .chain(load_tasks)
            .chain(mesh_tasks)
            .chain(checkpoint_task)
        {
            if prepared_task_batch.push_reserved(task).is_err() {
                unreachable!("checked exact task count cannot overfill scheduled suffix");
            }
        }
        let prepared_task_batch = prepared_task_batch
            .try_into_filled()
            .unwrap_or_else(|_| unreachable!("checked exact task count fills the batch"));

        let consume_save_panic_hook = {
            #[cfg(test)]
            {
                self.panic_next_save_before_io_for_test && !save_panic_hook_available
            }
            #[cfg(not(test))]
            {
                let _ = save_panic_hook_available;
                false
            }
        };
        // This is the final ownership take before C1 preparation. All other
        // fallible allocations and reservations above have succeeded.
        for insert in &load_inserts {
            let completion = self
                .durable_completion_inbox
                .get_mut(insert.inbox_index)
                .expect("prepared load escrow index remains in the durable prefix");
            let DurableCompletion::LoadFinished { output, .. } = completion else {
                unreachable!("prepared load escrow points to a load completion")
            };
            let voxels = output
                .block_data
                .voxels
                .take()
                .expect("prepared load escrow retained its exact voxel payload");
            let mut block = VoxelDataBlock::with_voxels(voxels, insert.location.lod_index);
            block.set_edited(true);
            data_operations.push(SharedVoxelDataTransactionOperation::Insert {
                location: insert.location,
                block,
                final_viewers: insert.final_viewers,
            });
        }
        let mut data_tx = match data_preview.prepare_transaction(data_operations, &data_snapshots) {
            Ok(transaction) => transaction,
            Err(error) => {
                let (storage_error, mut operations) = error.into_parts();
                self.restore_load_insert_escrow(&mut operations, &load_inserts);
                self.restore_prepared_followups(
                    prepared_task_batch,
                    scheduled_task_count,
                    &mut completion_plans,
                );
                return Err(VoxelTerrainRuntimeError::DataMutation(storage_error));
            }
        };
        if shutdown_owned_capture {
            let permit = self.shutdown_mutation_permit.as_ref().ok_or(
                VoxelTerrainRuntimeError::DataMutation(
                    SharedVoxelDataMutationError::MutationAdmissionClosed,
                ),
            )?;
            data_tx
                .authorize_shutdown_mutation(permit)
                .map_err(VoxelTerrainRuntimeError::DataMutation)?;
        }
        let draft = TerrainTransactionDraft {
            paired_viewer_publication: replace_viewers.then_some(next_paired_viewers),
            mode: PreparedTerrainMode::Fixed(FixedLodTransactionDraft {
                data_retry_publication: consume_data_retries.then(|| {
                    PreparedFixedDataRetryPublication {
                        next_data_view_retries: Vec::new(),
                        next_data_unview_retries: Vec::new(),
                        next_last_data_mutation_error: None,
                    }
                }),
                accepted_feature_updates,
                active_feature_updates,
                topology_event_index,
            }),
            next_request_generation,
            next_mesh_revision,
            next_render_topology_revision,
            next_stats,
            data_tx,
            mesh_diffs,
            data_residency_diffs,
            loading_diffs,
            pending_load_queues,
            pending_mesh_queues,
            persistence: PreparedOpaquePersistenceState {
                removed_owner_routes,
                resident_save_routes,
                save_dispatches,
                save_recovery_updates,
                resolution,
                checkpoint_config_update,
                checkpoint_dispatch,
                checkpoint_begin,
                next_deferred_save_dispatch_keys,
                next_deferred_checkpoint_dispatch,
                next_automatic_save_checkpoint_blocked,
                next_save_dispatch_error: None,
                next_save_generation,
                next_save_checkpoint_generation,
                next_persistence_attempt_ordinal,
                next_force_checkpoint_requested,
                consume_save_panic_hook,
                #[cfg(test)]
                next_panic_flush_before_io_attempts_for_test,
                #[cfg(test)]
                next_panic_flush_after_ack_for_test,
            },
            completion_prefix: PreparedCompletionPrefix {
                len: completion_count,
                accepted_followup_count,
                plans: completion_plans,
                load_inserts,
            },
            direct_mesh_plans,
            scheduled_task_count,
            prepared_task_batch,
            events_to_append,
            retirement,
        };
        #[cfg(test)]
        if let Some(position) = self.fixed_after_prepare_data_conflict_for_test.take() {
            let data = &self.data;
            let _ = data.try_set_block(position, VoxelDataBlock::empty(0));
        }
        #[cfg(test)]
        if std::mem::take(&mut self.fixed_after_prepare_settings_conflict_for_test) {
            self.data.increment_test_settings_revision();
        }
        self.commit_terrain_draft_no_fail(draft)
    }

    fn restore_load_insert_escrow(
        &mut self,
        operations: &mut [SharedVoxelDataTransactionOperation],
        inserts: &[PreparedLoadInsert],
    ) {
        for insert in inserts {
            let operation = operations
                .iter_mut()
                .find(|operation| match operation {
                    SharedVoxelDataTransactionOperation::Insert { location, .. } => {
                        *location == insert.location
                    }
                    SharedVoxelDataTransactionOperation::SetViewersExact { .. }
                    | SharedVoxelDataTransactionOperation::SetViewersExactAndClearModified {
                        ..
                    }
                    | SharedVoxelDataTransactionOperation::Replace { .. }
                    | SharedVoxelDataTransactionOperation::ClearModified { .. }
                    | SharedVoxelDataTransactionOperation::Remove { .. } => false,
                })
                .expect("prepared load insert remains in retry-owned C1 operations");
            let SharedVoxelDataTransactionOperation::Insert { block, .. } = operation else {
                unreachable!("prepared load escrow resolves only insert operations")
            };
            let voxels = std::mem::replace(block, VoxelDataBlock::empty(insert.location.lod_index))
                .into_voxels()
                .expect("prepared C1 insert retained the exact load payload");
            let completion = self
                .durable_completion_inbox
                .get_mut(insert.inbox_index)
                .expect("failed draft consumes no durable prefix");
            let DurableCompletion::LoadFinished { output, .. } = completion else {
                unreachable!("prepared load escrow restores its load completion")
            };
            debug_assert!(output.block_data.voxels.is_none());
            output.block_data.voxels = Some(voxels);
        }
    }

    fn restore_prepared_followups(
        &mut self,
        batch: FilledPreparedTaskBatch,
        scheduled_task_count: usize,
        plans: &mut [PreparedCompletionPlan],
    ) {
        let mut recovered = batch.into_scheduled_tasks();
        for plan in plans {
            let Some(escrow) = plan.followups.take() else {
                continue;
            };
            let completion = self
                .durable_completion_inbox
                .get_mut(plan.inbox_index)
                .expect("failed publication retains the exact durable prefix");
            let completed = completion
                .completed_mut()
                .expect("accepted rollback plan retains its runner task owner");
            if escrow.restore(completed, &mut recovered).is_err() {
                unreachable!("typed follow-up escrow restores its exact FIFO owners");
            }
            debug_assert_eq!(completion.descriptor(), plan.descriptor);
        }
        for _ in 0..scheduled_task_count {
            let _ = recovered.next();
        }
        debug_assert_eq!(recovered.len(), 0);
    }

    fn try_reserve_prepared_runtime_publication(
        &mut self,
        mesh_diffs: &[PreparedMeshEntryDiff],
        data_residency_diffs: &[PreparedDataResidencyDiff],
        loading_diffs: &[PreparedLoadingEntryDiff],
        pending_load_queues: &[PreparedQueueDiff<Vector3i>],
        pending_mesh_queues: &[PreparedQueueDiff<Vector3i>],
    ) -> Result<(), VoxelTerrainRuntimeError> {
        let lod_count = usize::from(self.lod_count);
        let lod_is_valid = |lod_index: u8| {
            let lod = usize::from(lod_index);
            lod < lod_count
                && lod < MAX_LOD
                && lod < self.mesh_maps.len()
                && lod < self.loading_blocks.len()
                && lod < self.loaded_data_residency.len()
                && lod < self.blocks_pending_load.len()
                && lod < self.blocks_pending_update.len()
        };
        if mesh_diffs
            .iter()
            .any(|diff| !lod_is_valid(diff.location.lod_index))
            || data_residency_diffs
                .iter()
                .any(|diff| !lod_is_valid(diff.location.lod_index))
            || loading_diffs
                .iter()
                .any(|diff| !lod_is_valid(diff.location.lod_index))
            || pending_load_queues
                .iter()
                .any(|diff| !lod_is_valid(diff.lod_index))
            || pending_mesh_queues
                .iter()
                .any(|diff| !lod_is_valid(diff.lod_index))
        {
            return Err(VoxelTerrainRuntimeError::LodMath(
                LodMathError::InvalidLodCount,
            ));
        }

        // Keep conflict selection independent from draft iteration order. The
        // fixed kind priority is maps first (mesh, loading, residency), then
        // queues (load, mesh); each kind reports its smallest exact key.
        if let Some(location) = try_canonical_duplicate_mesh_key(mesh_diffs)? {
            return Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
                PreparedPublicationConflict::DuplicateMeshKey { location },
            ));
        }
        if let Some(location) = try_canonical_duplicate_loading_key(loading_diffs)? {
            return Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
                PreparedPublicationConflict::DuplicateLoadingKey { location },
            ));
        }
        if let Some(location) = try_canonical_duplicate_data_residency_key(data_residency_diffs)? {
            return Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
                PreparedPublicationConflict::DuplicateDataResidencyKey { location },
            ));
        }
        if let Some(lod_index) = canonical_duplicate_queue_lod(pending_load_queues) {
            return Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
                PreparedPublicationConflict::DuplicatePendingLoadQueueLod { lod_index },
            ));
        }
        if let Some(lod_index) = canonical_duplicate_queue_lod(pending_mesh_queues) {
            return Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
                PreparedPublicationConflict::DuplicatePendingMeshQueueLod { lod_index },
            ));
        }

        let mesh_state_mismatch = mesh_diffs
            .iter()
            .filter(|diff| {
                let current = self.mesh_maps[usize::from(diff.location.lod_index)]
                    .get(&diff.location.position_in_blocks);
                match &diff.action {
                    PreparedMapAction::Insert(_) => {
                        diff.expected_revision.is_some() || current.is_some()
                    }
                    PreparedMapAction::Replace(_) | PreparedMapAction::Remove => {
                        current.is_none()
                            || current.and_then(|entry| entry.requested_revision)
                                != diff.expected_revision
                    }
                }
            })
            .map(|diff| diff.location)
            .min_by_key(|location| canonical_mesh_location_key(*location));
        if let Some(location) = mesh_state_mismatch {
            return Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
                PreparedPublicationConflict::MeshStateMismatch { location },
            ));
        }

        let loading_state_mismatch = loading_diffs
            .iter()
            .filter(|diff| {
                let current = self.loading_blocks[usize::from(diff.location.lod_index)]
                    .get(&diff.location.position);
                match &diff.action {
                    PreparedMapAction::Insert(_) => {
                        diff.expected_generation.is_some() || current.is_some()
                    }
                    PreparedMapAction::Replace(_) | PreparedMapAction::Remove => {
                        current.is_none()
                            || current.map(|entry| entry.request_generation)
                                != diff.expected_generation
                    }
                }
            })
            .map(|diff| diff.location)
            .min_by_key(|location| canonical_block_location_key(*location));
        if let Some(location) = loading_state_mismatch {
            return Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
                PreparedPublicationConflict::LoadingStateMismatch { location },
            ));
        }

        let data_residency_state_mismatch = data_residency_diffs
            .iter()
            .filter(|diff| {
                let current = self.loaded_data_residency[usize::from(diff.location.lod_index)]
                    .get(&diff.location.position)
                    .copied();
                match &diff.action {
                    PreparedMapAction::Insert(_) => diff.expected.is_some() || current.is_some(),
                    PreparedMapAction::Replace(_) | PreparedMapAction::Remove => {
                        current.is_none() || current != diff.expected
                    }
                }
            })
            .map(|diff| diff.location)
            .min_by_key(|location| canonical_block_location_key(*location));
        if let Some(location) = data_residency_state_mismatch {
            return Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
                PreparedPublicationConflict::DataResidencyStateMismatch { location },
            ));
        }

        #[cfg(test)]
        let mesh_insertions = mesh_diffs
            .iter()
            .filter(|diff| matches!(diff.action, PreparedMapAction::Insert(_)))
            .count();
        #[cfg(test)]
        if mesh_insertions != 0 {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::MeshMap)?;
        }
        #[cfg(test)]
        let loading_insertions = loading_diffs
            .iter()
            .filter(|diff| matches!(diff.action, PreparedMapAction::Insert(_)))
            .count();
        #[cfg(test)]
        if loading_insertions != 0 {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::LoadingMap)?;
        }
        for lod in 0..lod_count {
            let mesh_insertions_at_lod = mesh_diffs
                .iter()
                .filter(|diff| {
                    usize::from(diff.location.lod_index) == lod
                        && matches!(diff.action, PreparedMapAction::Insert(_))
                })
                .count();
            self.mesh_maps[lod]
                .try_reserve(mesh_insertions_at_lod)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            let loading_insertions_at_lod = loading_diffs
                .iter()
                .filter(|diff| {
                    usize::from(diff.location.lod_index) == lod
                        && matches!(diff.action, PreparedMapAction::Insert(_))
                })
                .count();
            self.loading_blocks[lod]
                .try_reserve(loading_insertions_at_lod)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            let data_residency_insertions_at_lod = data_residency_diffs
                .iter()
                .filter(|diff| {
                    usize::from(diff.location.lod_index) == lod
                        && matches!(diff.action, PreparedMapAction::Insert(_))
                })
                .count();
            self.loaded_data_residency[lod]
                .try_reserve(data_residency_insertions_at_lod)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        }
        Ok(())
    }

    fn publish_prepared_runtime_diffs_no_fail(
        &mut self,
        mesh_diffs: &mut Vec<PreparedMeshEntryDiff>,
        data_residency_diffs: &mut Vec<PreparedDataResidencyDiff>,
        loading_diffs: &mut Vec<PreparedLoadingEntryDiff>,
        retirement: &mut RetirementBag,
    ) {
        for diff in mesh_diffs.drain(..) {
            let lod = usize::from(diff.location.lod_index);
            let position = diff.location.position_in_blocks;
            let mesh_map = self
                .mesh_maps
                .get_mut(lod)
                .unwrap_or_else(|| unreachable!("prepared mesh LOD remains valid"));
            debug_assert_eq!(
                mesh_map
                    .get(&position)
                    .and_then(|entry| entry.requested_revision),
                diff.expected_revision
            );
            match diff.action {
                PreparedMapAction::Insert(entry) => {
                    let previous = mesh_map.insert(position, entry);
                    debug_assert!(previous.is_none());
                    if let Some(previous) = previous {
                        previous.cancel_physical_request_if_superseded_by(
                            mesh_map[&position].physical_request.as_ref(),
                        );
                        retirement.mesh_entries.push(previous);
                    }
                }
                PreparedMapAction::Replace(entry) => {
                    let old = mesh_map
                        .insert(position, entry)
                        .expect("prepared mesh replacement retained its entry");
                    old.cancel_physical_request_if_superseded_by(
                        mesh_map[&position].physical_request.as_ref(),
                    );
                    retirement.mesh_entries.push(old);
                }
                PreparedMapAction::Remove => {
                    let old = mesh_map
                        .remove(&position)
                        .expect("prepared mesh removal retained its entry");
                    old.cancel_physical_request_if_superseded_by(None);
                    retirement.mesh_entries.push(old);
                }
            }
        }
        for diff in data_residency_diffs.drain(..) {
            let lod = usize::from(diff.location.lod_index);
            let position = diff.location.position;
            let residency = self
                .loaded_data_residency
                .get_mut(lod)
                .unwrap_or_else(|| unreachable!("prepared data-residency LOD remains valid"));
            debug_assert_eq!(residency.get(&position).copied(), diff.expected);
            match diff.action {
                PreparedMapAction::Insert(next) => {
                    let previous = residency.insert(position, next);
                    debug_assert!(previous.is_none());
                }
                PreparedMapAction::Replace(next) => {
                    let previous = residency.insert(position, next);
                    debug_assert!(previous.is_some());
                }
                PreparedMapAction::Remove => {
                    let previous = residency.remove(&position);
                    debug_assert!(previous.is_some());
                }
            }
        }
        for diff in loading_diffs.drain(..) {
            let lod = usize::from(diff.location.lod_index);
            let position = diff.location.position;
            let loading = self
                .loading_blocks
                .get_mut(lod)
                .unwrap_or_else(|| unreachable!("prepared loading LOD remains valid"));
            debug_assert_eq!(
                loading.get(&position).map(|entry| entry.request_generation),
                diff.expected_generation
            );
            match diff.action {
                PreparedMapAction::Insert(entry) => {
                    let previous = loading.insert(position, entry);
                    debug_assert!(previous.is_none());
                    if let Some(previous) = previous {
                        previous.cancel_physical_request_if_superseded_by(
                            loading[&position].physical_request.as_ref(),
                        );
                        retirement.loading_entries.push(previous);
                    }
                }
                PreparedMapAction::Replace(entry) => {
                    let old = loading
                        .insert(position, entry)
                        .expect("prepared loading replacement retained its entry");
                    old.cancel_physical_request_if_superseded_by(
                        loading[&position].physical_request.as_ref(),
                    );
                    retirement.loading_entries.push(old);
                }
                PreparedMapAction::Remove => {
                    let old = loading
                        .remove(&position)
                        .expect("prepared loading removal retained its entry");
                    old.cancel_physical_request_if_superseded_by(None);
                    retirement.loading_entries.push(old);
                }
            }
        }
    }

    fn publish_prepared_queue_diffs_no_fail(
        &mut self,
        pending_load_queues: &mut [PreparedQueueDiff<Vector3i>],
        pending_mesh_queues: &mut [PreparedQueueDiff<Vector3i>],
    ) {
        for prepared in pending_load_queues {
            let live = self
                .blocks_pending_load
                .get_mut(usize::from(prepared.lod_index))
                .unwrap_or_else(|| unreachable!("prepared load-queue LOD remains valid"));
            std::mem::swap(live, &mut prepared.final_values);
        }
        for prepared in pending_mesh_queues {
            let live = self
                .blocks_pending_update
                .get_mut(usize::from(prepared.lod_index))
                .unwrap_or_else(|| unreachable!("prepared mesh-queue LOD remains valid"));
            std::mem::swap(live, &mut prepared.final_values);
        }
    }

    fn revalidate_prepared_terrain_mode(
        &self,
        mode: &PreparedTerrainMode,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        let PreparedTerrainMode::Variable(variable) = mode else {
            return Ok(());
        };
        let runtime = self
            .variable_lod
            .as_ref()
            .unwrap_or_else(|| unreachable!("validated variable mode retains its runtime"));
        variable
            .coordinator_update
            .as_ref()
            .unwrap_or_else(|| unreachable!("variable mode retains one coordinator token"))
            .revalidate_for(&runtime.coordinator)?;
        if let Some(coverage) = &variable.coverage_publication {
            coverage
                .preview
                .as_ref()
                .unwrap_or_else(|| unreachable!("coverage publication retains one preview token"))
                .revalidate_for(&runtime.coverage)?;
            #[cfg(test)]
            {
                for observation in &coverage.physical_observations.mesh {
                    let current = self
                        .mesh_maps
                        .get(usize::from(observation.location.lod_index))
                        .and_then(|map| map.get(&observation.location.position_in_blocks));
                    let matches = match (current, observation.expected.as_ref()) {
                        (None, None) => true,
                        (Some(current), Some(expected)) => {
                            mesh_entry_physical_state_matches(current, expected)
                        }
                        (None, Some(_)) | (Some(_), None) => false,
                    };
                    if !matches {
                        return Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
                            PreparedPublicationConflict::MeshStateMismatch {
                                location: observation.location,
                            },
                        ));
                    }
                }
                for observation in &coverage.physical_observations.data {
                    match observation {
                        VariableDataPhysicalObservation::Loaded { location, expected } => {
                            let current = self.loaded_data_residency
                                [usize::from(location.lod_index)]
                            .get(&location.position)
                            .copied();
                            if current != Some(*expected)
                                || self.loading_blocks[usize::from(location.lod_index)]
                                    .contains_key(&location.position)
                            {
                                return Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
                                    PreparedPublicationConflict::DataResidencyStateMismatch {
                                        location: *location,
                                    },
                                ));
                            }
                        }
                        VariableDataPhysicalObservation::Loading { location, expected } => {
                            let current = self.loading_blocks[usize::from(location.lod_index)]
                                .get(&location.position);
                            if current.is_none_or(|current| {
                                !loading_entry_physical_state_matches(current, expected)
                            }) || self.loaded_data_residency[usize::from(location.lod_index)]
                                .contains_key(&location.position)
                            {
                                return Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
                                    PreparedPublicationConflict::LoadingStateMismatch {
                                        location: *location,
                                    },
                                ));
                            }
                        }
                        VariableDataPhysicalObservation::Missing { location } => {
                            if self.loaded_data_residency[usize::from(location.lod_index)]
                                .contains_key(&location.position)
                                || self.loading_blocks[usize::from(location.lod_index)]
                                    .contains_key(&location.position)
                            {
                                return Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
                                    PreparedPublicationConflict::DataResidencyStateMismatch {
                                        location: *location,
                                    },
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn commit_terrain_draft_no_fail(
        &mut self,
        draft: TerrainTransactionDraft,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        #[cfg(test)]
        if let Some(pause) = &self.fixed_commit_pause_for_test {
            pause.commit_marker.store(false, Ordering::SeqCst);
        }
        let TerrainTransactionDraft {
            paired_viewer_publication,
            mode,
            next_request_generation,
            next_mesh_revision,
            next_render_topology_revision,
            next_stats,
            mut data_tx,
            mesh_diffs,
            data_residency_diffs,
            loading_diffs,
            pending_load_queues,
            pending_mesh_queues,
            persistence,
            completion_prefix,
            direct_mesh_plans,
            scheduled_task_count,
            prepared_task_batch,
            events_to_append,
            retirement,
        } = draft;
        let publication = PreparedTerrainPublication {
            paired_viewer_publication,
            mode,
            next_request_generation,
            next_mesh_revision,
            next_render_topology_revision,
            next_stats,
            mesh_diffs,
            data_residency_diffs,
            loading_diffs,
            pending_load_queues,
            pending_mesh_queues,
            persistence,
            completion_prefix,
            direct_mesh_plans,
            scheduled_task_count,
            prepared_task_batch,
            events_to_append,
            retirement,
        };

        if let Err(error) = self.revalidate_prepared_terrain_mode(&publication.mode) {
            let mut operations = data_tx
                .into_operations()
                .expect("pre-C1 revalidation retains every retry-owned operation");
            let PreparedTerrainPublication {
                mut completion_prefix,
                scheduled_task_count,
                prepared_task_batch,
                ..
            } = publication;
            self.restore_load_insert_escrow(&mut operations, &completion_prefix.load_inserts);
            self.restore_prepared_followups(
                prepared_task_batch,
                scheduled_task_count,
                &mut completion_prefix.plans,
            );
            return Err(error);
        }

        let error = match data_tx.commit_holding_publication_fence() {
            Ok(fence) => {
                #[cfg(test)]
                if let Some(pause) = &self.fixed_commit_pause_for_test {
                    pause.pause_if_target(FixedCommitPausePhase::StorageFencedBeforeCorePublish);
                }
                return self.publish_committed_terrain_draft(fence, publication);
            }
            Err(error) => error,
        };
        let mut operations = data_tx
            .into_operations()
            .expect("failed C1 fence retains every retry-owned operation");
        let PreparedTerrainPublication {
            mut completion_prefix,
            scheduled_task_count,
            prepared_task_batch,
            ..
        } = publication;
        self.restore_load_insert_escrow(&mut operations, &completion_prefix.load_inserts);
        self.restore_prepared_followups(
            prepared_task_batch,
            scheduled_task_count,
            &mut completion_prefix.plans,
        );
        Err(VoxelTerrainRuntimeError::DataMutation(error))
    }

    fn publish_committed_terrain_draft(
        &mut self,
        mut fence: CommittedSharedVoxelDataTransaction<'_>,
        publication: PreparedTerrainPublication,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        let PreparedTerrainPublication {
            paired_viewer_publication,
            mut mode,
            next_request_generation,
            next_mesh_revision,
            next_render_topology_revision,
            next_stats,
            mut mesh_diffs,
            mut data_residency_diffs,
            mut loading_diffs,
            mut pending_load_queues,
            mut pending_mesh_queues,
            persistence,
            completion_prefix,
            mut direct_mesh_plans,
            scheduled_task_count,
            prepared_task_batch,
            mut events_to_append,
            mut retirement,
        } = publication;
        let PreparedOpaquePersistenceState {
            mut removed_owner_routes,
            mut resident_save_routes,
            mut save_dispatches,
            mut save_recovery_updates,
            mut resolution,
            checkpoint_config_update,
            checkpoint_dispatch,
            checkpoint_begin,
            next_deferred_save_dispatch_keys,
            next_deferred_checkpoint_dispatch,
            next_automatic_save_checkpoint_blocked,
            next_save_dispatch_error,
            next_save_generation,
            next_save_checkpoint_generation,
            next_persistence_attempt_ordinal,
            next_force_checkpoint_requested,
            consume_save_panic_hook,
            #[cfg(test)]
            next_panic_flush_before_io_attempts_for_test,
            #[cfg(test)]
            next_panic_flush_after_ack_for_test,
        } = persistence;
        let PreparedCompletionPrefix {
            len: completion_prefix_len,
            accepted_followup_count,
            plans: mut completion_plans,
            load_inserts,
        } = completion_prefix;
        let mut removed = fence.take_removed_blocks();
        debug_assert_eq!(
            removed.len(),
            removed_owner_routes.len(),
            "every prepared C1 removal has one canonical owner route"
        );
        let had_dirty_owner = !resident_save_routes.is_empty()
            || removed_owner_routes
                .iter()
                .any(|route| !matches!(route, PreparedRemovedOwnerRoute::Clean));
        for (removed, route) in removed.drain(..).zip(removed_owner_routes.drain(..)) {
            match route {
                PreparedRemovedOwnerRoute::Clean => retirement.data_blocks.push(removed),
                PreparedRemovedOwnerRoute::DirtyRetained {
                    location: expected_location,
                    block_revision,
                } => {
                    let (location, block) = removed.into_parts();
                    debug_assert_eq!(location, expected_location);
                    let voxels = block.into_voxels();
                    #[cfg(test)]
                    if let Some(payload) = voxels.as_ref() {
                        self.record_fixed_dirty_owner_for_test(location, payload);
                    }
                    let save = BlockToSave {
                        voxels,
                        position: location.position,
                        lod_index: location.lod_index,
                        block_revision,
                    };
                    self.retained_save_admission_failures
                        .push_back(RetainedSaveAdmissionFailure {
                            error: VoxelTerrainRuntimeError::MissingSavePayload,
                            save,
                        });
                }
                PreparedRemovedOwnerRoute::DirtyDispatch(route) => {
                    let (location, block) = removed.into_parts();
                    debug_assert_eq!(location, route.location);
                    let payload = block
                        .into_voxels()
                        .expect("C1 revalidated the prepared dirty payload presence");
                    #[cfg(test)]
                    self.record_fixed_dirty_owner_for_test(location, &payload);
                    let meta = SaveAttemptMeta {
                        block_revision: route.block_revision,
                        generation: route.save_generation,
                        retry_count: 0,
                        last_error: None,
                    };
                    match route.target {
                        PreparedJournalTarget::Vacant => {
                            let replaced = self.save_journal.insert(
                                SaveKey::new(location.position, location.lod_index),
                                SaveJournalEntry {
                                    written_unflushed: None,
                                    active: Some(ActiveSaveAttempt::WriteInFlight {
                                        meta,
                                        attempt_ordinal: route.attempt_ordinal,
                                    }),
                                    queued_newer: VecDeque::new(),
                                },
                            );
                            if let Some(replaced) = replaced {
                                retirement.save_journal_entries.push(replaced);
                            }
                        }
                        PreparedJournalTarget::ActiveIdle => {
                            if let Some(entry) = self
                                .save_journal
                                .get_mut(&SaveKey::new(location.position, location.lod_index))
                            {
                                entry.active = Some(ActiveSaveAttempt::WriteInFlight {
                                    meta,
                                    attempt_ordinal: route.attempt_ordinal,
                                });
                            }
                        }
                        PreparedJournalTarget::Queued => {
                            unreachable!("queued dirty ownership is prepared as pending")
                        }
                    }
                    route.installer.install_payload(payload);
                }
                PreparedRemovedOwnerRoute::DirtyPending(route) => {
                    let (location, block) = removed.into_parts();
                    debug_assert_eq!(location, route.location);
                    let payload = block
                        .into_voxels()
                        .expect("C1 revalidated the prepared dirty payload presence");
                    #[cfg(test)]
                    self.record_fixed_dirty_owner_for_test(location, &payload);
                    let pending = PendingSave {
                        meta: SaveAttemptMeta {
                            block_revision: route.block_revision,
                            generation: route.save_generation,
                            retry_count: 0,
                            last_error: None,
                        },
                        payload,
                    };
                    let key = SaveKey::new(location.position, location.lod_index);
                    match route.target {
                        PreparedJournalTarget::Vacant => {
                            let replaced = self.save_journal.insert(
                                key,
                                SaveJournalEntry {
                                    written_unflushed: None,
                                    active: Some(ActiveSaveAttempt::Pending(pending)),
                                    queued_newer: VecDeque::new(),
                                },
                            );
                            if let Some(replaced) = replaced {
                                retirement.save_journal_entries.push(replaced);
                            }
                        }
                        PreparedJournalTarget::ActiveIdle => {
                            if let Some(entry) = self.save_journal.get_mut(&key) {
                                entry.active = Some(ActiveSaveAttempt::Pending(pending));
                            }
                        }
                        PreparedJournalTarget::Queued => {
                            if let Some(entry) = self.save_journal.get_mut(&key) {
                                entry.queued_newer.push_back(pending);
                            }
                        }
                    }
                }
            }
        }
        for resident in resident_save_routes.drain(..) {
            match resident {
                PreparedResidentSaveRoute::Dispatch { payload, route } => {
                    debug_assert_eq!(payload.location, route.location);
                    debug_assert_eq!(payload.block_revision, route.block_revision);
                    let location = payload.location;
                    #[cfg(test)]
                    self.record_fixed_dirty_owner_for_test(location, &payload.voxels);
                    let meta = SaveAttemptMeta {
                        block_revision: route.block_revision,
                        generation: route.save_generation,
                        retry_count: 0,
                        last_error: None,
                    };
                    match route.target {
                        PreparedJournalTarget::Vacant => {
                            let replaced = self.save_journal.insert(
                                SaveKey::new(location.position, location.lod_index),
                                SaveJournalEntry {
                                    written_unflushed: None,
                                    active: Some(ActiveSaveAttempt::WriteInFlight {
                                        meta,
                                        attempt_ordinal: route.attempt_ordinal,
                                    }),
                                    queued_newer: VecDeque::new(),
                                },
                            );
                            if let Some(replaced) = replaced {
                                retirement.save_journal_entries.push(replaced);
                            }
                        }
                        PreparedJournalTarget::ActiveIdle => {
                            if let Some(entry) = self
                                .save_journal
                                .get_mut(&SaveKey::new(location.position, location.lod_index))
                            {
                                entry.active = Some(ActiveSaveAttempt::WriteInFlight {
                                    meta,
                                    attempt_ordinal: route.attempt_ordinal,
                                });
                            }
                        }
                        PreparedJournalTarget::Queued => {
                            unreachable!("a dispatched resident save cannot target the queue")
                        }
                    }
                    route.installer.install_payload(payload.voxels);
                }
                PreparedResidentSaveRoute::Pending { payload, route } => {
                    debug_assert_eq!(payload.location, route.location);
                    debug_assert_eq!(payload.block_revision, route.block_revision);
                    let location = payload.location;
                    #[cfg(test)]
                    self.record_fixed_dirty_owner_for_test(location, &payload.voxels);
                    let pending = PendingSave {
                        meta: SaveAttemptMeta {
                            block_revision: route.block_revision,
                            generation: route.save_generation,
                            retry_count: 0,
                            last_error: None,
                        },
                        payload: payload.voxels,
                    };
                    let key = SaveKey::new(location.position, location.lod_index);
                    match route.target {
                        PreparedJournalTarget::Vacant => {
                            let replaced = self.save_journal.insert(
                                key,
                                SaveJournalEntry {
                                    written_unflushed: None,
                                    active: Some(ActiveSaveAttempt::Pending(pending)),
                                    queued_newer: VecDeque::new(),
                                },
                            );
                            if let Some(replaced) = replaced {
                                retirement.save_journal_entries.push(replaced);
                            }
                        }
                        PreparedJournalTarget::ActiveIdle => {
                            if let Some(entry) = self.save_journal.get_mut(&key) {
                                entry.active = Some(ActiveSaveAttempt::Pending(pending));
                            }
                        }
                        PreparedJournalTarget::Queued => {
                            if let Some(entry) = self.save_journal.get_mut(&key) {
                                entry.queued_newer.push_back(pending);
                            }
                        }
                    }
                }
            }
        }
        self.next_save_generation = next_save_generation;
        self.next_save_checkpoint_generation = next_save_checkpoint_generation;
        self.next_persistence_attempt_ordinal = next_persistence_attempt_ordinal;
        self.force_checkpoint_requested = next_force_checkpoint_requested;
        if let Some(previous) =
            std::mem::replace(&mut self.save_dispatch_error, next_save_dispatch_error)
        {
            retirement.runtime_errors.push(previous);
        }
        if consume_save_panic_hook {
            #[cfg(test)]
            {
                self.panic_next_save_before_io_for_test = false;
            }
        }
        #[cfg(test)]
        {
            self.panic_next_flush_before_io_attempts_for_test =
                next_panic_flush_before_io_attempts_for_test;
            self.panic_next_flush_after_ack_for_test = next_panic_flush_after_ack_for_test;
        }
        if had_dirty_owner {
            self.automatic_checkpoint_satisfied_empty_flush = false;
        }

        match &mode {
            PreparedTerrainMode::Fixed(fixed) => {
                debug_assert!(fixed.topology_event_index.is_none_or(|index| matches!(
                    events_to_append.get(index),
                    Some(VoxelTerrainEvent::RenderTopologyChanged(_))
                )));
                let _ = (
                    fixed.accepted_feature_updates.len(),
                    fixed.active_feature_updates.len(),
                );
            }
            PreparedTerrainMode::Variable(_) => {}
        }
        if let Some(next_paired_viewers) = paired_viewer_publication {
            retirement.paired_viewers =
                std::mem::replace(&mut self.paired_viewers, next_paired_viewers);
        }
        self.publish_prepared_runtime_diffs_no_fail(
            &mut mesh_diffs,
            &mut data_residency_diffs,
            &mut loading_diffs,
            &mut retirement,
        );
        self.publish_prepared_queue_diffs_no_fail(
            &mut pending_load_queues,
            &mut pending_mesh_queues,
        );
        self.next_request_generation = next_request_generation;
        self.next_mesh_revision = next_mesh_revision;
        self.next_render_topology_revision = next_render_topology_revision;
        self.stats = next_stats;
        match &mut mode {
            PreparedTerrainMode::Fixed(fixed) => {
                if let Some(retries) = fixed.data_retry_publication.take() {
                    retirement.data_view_retries = std::mem::replace(
                        &mut self.data_view_retries[0],
                        retries.next_data_view_retries,
                    );
                    retirement.data_unview_retries = std::mem::replace(
                        &mut self.data_unview_retries[0],
                        retries.next_data_unview_retries,
                    );
                    retirement.last_data_mutation_error = std::mem::replace(
                        &mut self.last_data_mutation_error,
                        retries.next_last_data_mutation_error,
                    );
                }
            }
            PreparedTerrainMode::Variable(variable) => {
                self.publish_variable_mode_no_fail(variable, &mut retirement);
            }
        }
        debug_assert_eq!(completion_prefix_len, completion_plans.len());
        for (committed_index, mut plan) in completion_plans.drain(..).enumerate() {
            debug_assert_eq!(plan.inbox_index, committed_index);
            let completion = self
                .durable_completion_inbox
                .pop_front()
                .expect("prepared durable prefix remains complete until commit");
            let mut committed_descriptor = completion.descriptor();
            if plan.followups.is_some() {
                committed_descriptor.owner = committed_descriptor
                    .owner
                    .with_follow_up_count(plan.descriptor.owner.follow_up_count());
            }
            debug_assert_eq!(committed_descriptor, plan.descriptor);
            if let Some(escrow) = plan.followups.take() {
                retirement.followup_escrows.push(escrow);
            }
            self.commit_prepared_completion(completion, plan.disposition, &mut retirement);
        }
        if let Some(resolution) = resolution.as_mut() {
            match resolution {
                PreparedFixedPersistenceResolution::Save {
                    quarantine_index,
                    key,
                    block_revision,
                    generation,
                    attempt_ordinal,
                    assume_written,
                } => {
                    let quarantined = self
                        .completion_quarantine
                        .remove(*quarantine_index)
                        .expect("prepared save resolution quarantine remains present");
                    let QuarantinedCompletion::Persistence {
                        terminal: PersistenceTaskTerminal::Save(terminal),
                        attempt_ordinal: terminal_attempt,
                        completed,
                        ..
                    } = quarantined
                    else {
                        unreachable!("prepared save resolution retains exact terminal")
                    };
                    debug_assert_eq!(terminal.block_revision, *block_revision);
                    debug_assert_eq!(terminal.save_generation, *generation);
                    debug_assert_eq!(terminal_attempt, *attempt_ordinal);
                    let entry = self.save_journal.get_mut(key).unwrap_or_else(|| {
                        unreachable!("prepared save resolution journal remains live")
                    });
                    let active = entry
                        .active
                        .take()
                        .unwrap_or_else(|| unreachable!("prepared save resolution remains active"));
                    let ActiveSaveAttempt::Indeterminate {
                        meta,
                        attempt_ordinal: active_attempt,
                    } = active
                    else {
                        unreachable!("prepared save resolution remains indeterminate")
                    };
                    debug_assert_eq!(meta.block_revision, *block_revision);
                    debug_assert_eq!(meta.generation, *generation);
                    debug_assert_eq!(active_attempt, *attempt_ordinal);
                    if *assume_written {
                        debug_assert!(entry.written_unflushed.is_none());
                        entry.written_unflushed = Some(WrittenSave {
                            block_revision: *block_revision,
                            generation: *generation,
                            payload: terminal.payload,
                        });
                        retirement.save_attempt_meta.push(meta);
                    } else {
                        entry.active = Some(ActiveSaveAttempt::Pending(PendingSave {
                            meta,
                            payload: terminal.payload,
                        }));
                    }
                    retirement.completed_tasks.push(completed);
                }
                PreparedFixedPersistenceResolution::CheckpointRetry {
                    quarantine_index,
                    checkpoint_generation,
                    attempt_ordinal,
                } => {
                    let quarantined = self
                        .completion_quarantine
                        .remove(*quarantine_index)
                        .expect("prepared checkpoint retry quarantine remains present");
                    let QuarantinedCompletion::Persistence {
                        terminal: PersistenceTaskTerminal::Flush(terminal),
                        attempt_ordinal: terminal_attempt,
                        completed,
                        ..
                    } = quarantined
                    else {
                        unreachable!("prepared checkpoint retry retains exact terminal")
                    };
                    debug_assert_eq!(terminal.checkpoint_generation, *checkpoint_generation);
                    debug_assert_eq!(terminal_attempt, *attempt_ordinal);
                    let checkpoint = self
                        .save_checkpoint_in_flight
                        .as_mut()
                        .unwrap_or_else(|| unreachable!("prepared checkpoint retry remains live"));
                    debug_assert_eq!(checkpoint.checkpoint_generation, *checkpoint_generation);
                    debug_assert_eq!(
                        checkpoint.state,
                        CheckpointAttemptState::Indeterminate {
                            attempt_ordinal: *attempt_ordinal
                        }
                    );
                    checkpoint.state = CheckpointAttemptState::Pending;
                    retirement.completed_tasks.push(completed);
                }
                PreparedFixedPersistenceResolution::CheckpointWritten {
                    quarantine_index,
                    checkpoint_generation,
                    attempt_ordinal,
                    entry_actions,
                } => {
                    let quarantined = self
                        .completion_quarantine
                        .remove(*quarantine_index)
                        .expect("prepared checkpoint acknowledgement remains present");
                    let QuarantinedCompletion::Persistence {
                        terminal: PersistenceTaskTerminal::Flush(terminal),
                        attempt_ordinal: terminal_attempt,
                        completed,
                        ..
                    } = quarantined
                    else {
                        unreachable!("prepared checkpoint acknowledgement retains exact terminal")
                    };
                    debug_assert_eq!(terminal.checkpoint_generation, *checkpoint_generation);
                    debug_assert_eq!(terminal_attempt, *attempt_ordinal);
                    let checkpoint = self.save_checkpoint_in_flight.take().unwrap_or_else(|| {
                        unreachable!("prepared checkpoint acknowledgement remains live")
                    });
                    debug_assert_eq!(checkpoint.checkpoint_generation, *checkpoint_generation);
                    debug_assert_eq!(
                        checkpoint.state,
                        CheckpointAttemptState::Indeterminate {
                            attempt_ordinal: *attempt_ordinal
                        }
                    );
                    if checkpoint.origin == CheckpointOrigin::Explicit {
                        if let Some(previous) = self
                            .last_checkpoint_outcome
                            .replace((*checkpoint_generation, Ok(())))
                        {
                            retirement.checkpoint_outcomes.push(previous);
                        }
                    }
                    if let Some(previous) = self.last_save_checkpoint_error.take() {
                        retirement.stream_errors.push(previous);
                    }
                    for action in entry_actions.drain(..) {
                        let PreparedCheckpointEntryAction::ClearWritten {
                            key,
                            generation,
                            promote_queued,
                            remove_entry,
                            ..
                        } = action
                        else {
                            unreachable!("successful checkpoint resolution only clears writes")
                        };
                        let entry = self.save_journal.get_mut(&key).unwrap_or_else(|| {
                            unreachable!("prepared checkpoint resolution key remains live")
                        });
                        let written = entry.written_unflushed.take().unwrap_or_else(|| {
                            unreachable!("prepared checkpoint resolution write remains live")
                        });
                        debug_assert_eq!(written.generation, generation);
                        retirement.written_saves.push(written);
                        if promote_queued {
                            debug_assert!(entry.active.is_none());
                            entry.active = entry
                                .queued_newer
                                .pop_front()
                                .map(ActiveSaveAttempt::Pending);
                        }
                        if remove_entry {
                            let removed = self.save_journal.remove(&key).unwrap_or_else(|| {
                                unreachable!("prepared checkpoint resolution removal remains live")
                            });
                            retirement.save_journal_entries.push(removed);
                        }
                    }
                    retirement.completed_tasks.push(completed);
                    retirement.checkpoints.push(checkpoint);
                    retirement
                        .checkpoint_actions
                        .push(PreparedCheckpointAction::Acknowledge {
                            checkpoint_generation: *checkpoint_generation,
                            attempt_ordinal: *attempt_ordinal,
                            succeeded: true,
                            origin: CheckpointOrigin::Automatic,
                            explicit_outcome: None,
                            entry_actions: std::mem::take(entry_actions),
                        });
                    self.automatic_checkpoint_satisfied_empty_flush = self.save_journal.is_empty();
                }
            }
        }
        for update in save_recovery_updates.drain(..) {
            let entry = self
                .save_journal
                .get_mut(&update.key)
                .unwrap_or_else(|| unreachable!("prepared save recovery key remains live"));
            let Some(ActiveSaveAttempt::Pending(pending)) = entry.active.as_mut() else {
                unreachable!("prepared save recovery remains pending until commit")
            };
            debug_assert_eq!(pending.meta.block_revision, update.block_revision);
            debug_assert_eq!(pending.meta.generation, update.generation);
            pending.meta.retry_count = 0;
            if let Some(error) = pending.meta.last_error.take() {
                #[cfg(test)]
                if let Some((expected_ptr, dropped)) =
                    &self.fixed_stream_error_retirement_probe_for_test
                {
                    let actual_ptr = match &error {
                        VoxelStreamError::Io(message) | VoxelStreamError::CorruptData(message) => {
                            message.as_ptr() as usize
                        }
                        _ => 0,
                    };
                    if actual_ptr == *expected_ptr {
                        retirement
                            .stream_error_drop_probes
                            .push(FixedStreamErrorRetirementProbe {
                                dropped: Arc::clone(dropped),
                            });
                    }
                }
                retirement.stream_errors.push(error);
            }
        }
        self.automatic_save_checkpoint_blocked = next_automatic_save_checkpoint_blocked;
        if let Some(update) = checkpoint_config_update {
            let checkpoint = self
                .save_checkpoint_in_flight
                .as_mut()
                .unwrap_or_else(|| unreachable!("prepared checkpoint config remains live"));
            debug_assert_eq!(
                checkpoint.checkpoint_generation,
                update.checkpoint_generation
            );
            checkpoint.retry_count = update.retry_count;
            checkpoint.max_attempts = update.max_attempts;
            checkpoint.origin = update.origin;
            checkpoint.record_per_block_failure = update.record_per_block_failure;
        }
        if let Some(checkpoint) = checkpoint_begin {
            debug_assert!(self.save_checkpoint_in_flight.is_none());
            if let Some(previous) = self.last_checkpoint_outcome.take() {
                retirement.checkpoint_outcomes.push(previous);
            }
            self.save_checkpoint_in_flight = Some(checkpoint);
        }
        for dispatch in save_dispatches.drain(..) {
            let entry = self
                .save_journal
                .get_mut(&dispatch.key)
                .unwrap_or_else(|| unreachable!("prepared save dispatch key remains live"));
            let active = entry
                .active
                .take()
                .unwrap_or_else(|| unreachable!("prepared pending save remains active"));
            let ActiveSaveAttempt::Pending(pending) = active else {
                unreachable!("prepared save dispatch retains pending ownership")
            };
            debug_assert_eq!(pending.meta.block_revision, dispatch.block_revision);
            debug_assert_eq!(pending.meta.generation, dispatch.generation);
            dispatch.installer.install_payload(pending.payload);
            entry.active = Some(ActiveSaveAttempt::WriteInFlight {
                meta: pending.meta,
                attempt_ordinal: dispatch.attempt_ordinal,
            });
        }
        if let Some(dispatch) = checkpoint_dispatch {
            let checkpoint = self
                .save_checkpoint_in_flight
                .as_mut()
                .unwrap_or_else(|| unreachable!("prepared checkpoint dispatch remains live"));
            debug_assert_eq!(
                checkpoint.checkpoint_generation,
                dispatch.checkpoint_generation
            );
            debug_assert_eq!(checkpoint.state, CheckpointAttemptState::Pending);
            checkpoint.state = CheckpointAttemptState::WriteInFlight {
                attempt_ordinal: dispatch.attempt_ordinal,
            };
        }
        retirement.deferred_save_keys = std::mem::replace(
            &mut self.deferred_save_dispatch_keys,
            next_deferred_save_dispatch_keys,
        );
        self.deferred_checkpoint_dispatch = next_deferred_checkpoint_dispatch;
        for (committed_index, plan) in direct_mesh_plans.drain(..).enumerate() {
            debug_assert_eq!(plan.inbox_index, committed_index);
            let completion = self
                .direct_mesh_retry_inbox
                .pop_front()
                .expect("prepared direct prefix remains complete until commit");
            debug_assert_eq!(completion.descriptor(), plan.descriptor);
            let DurableCompletion::DirectMesh { upload, .. } = completion else {
                unreachable!("direct mesh FIFO contains only admitted direct uploads")
            };
            retirement.mesh_snapshots.push(upload);
        }
        self.event_outbox.extend(events_to_append.drain(..));

        #[cfg(test)]
        if let Some(pause) = &self.fixed_commit_pause_for_test {
            pause.commit_marker.store(true, Ordering::SeqCst);
            pause.pause_if_target(FixedCommitPausePhase::AfterTerrainPublishBeforeFenceFinish);
        }
        let _ = fence.finish();
        let wake = self.task_runner.link_prepared(prepared_task_batch);
        #[cfg(test)]
        if let Some(pause) = &self.fixed_commit_pause_for_test {
            pause.pause_if_target(FixedCommitPausePhase::BatchLinkedBeforeWake);
        }
        wake.wake();
        #[cfg(test)]
        if let Some(pause) = &self.fixed_commit_pause_for_test {
            pause.pause_if_target(FixedCommitPausePhase::AfterWakeBeforeRetirementDrop);
        }
        drop((
            mode,
            load_inserts,
            completion_plans,
            direct_mesh_plans,
            removed,
            removed_owner_routes,
            save_dispatches,
            save_recovery_updates,
            resolution,
            mesh_diffs,
            data_residency_diffs,
            loading_diffs,
            pending_load_queues,
            pending_mesh_queues,
            events_to_append,
            accepted_followup_count,
            scheduled_task_count,
        ));
        drop(retirement);
        Ok(())
    }

    fn publish_variable_mode_no_fail(
        &mut self,
        variable: &mut VariableLodTransactionDraft,
        retirement: &mut RetirementBag,
    ) {
        let runtime = self
            .variable_lod
            .as_mut()
            .unwrap_or_else(|| unreachable!("validated variable publication retains its runtime"));
        let coordinator = variable
            .coordinator_update
            .take()
            .unwrap_or_else(|| unreachable!("variable publication owns one coordinator token"))
            .publish(&mut runtime.coordinator);
        let (coordinator_delta, coordinator_state) = coordinator.into_parts();
        debug_assert!(retirement.coordinator_state.is_none());
        retirement.coordinator_state = Some(coordinator_state);

        let Some(mut coverage_publication) = variable.coverage_publication.take() else {
            debug_assert!(retirement.variable_mode.is_none());
            retirement.variable_mode = Some(VariableModeRetirement {
                _coordinator_delta: coordinator_delta,
                _coverage_result: None,
                _before_mesh_updates: Vec::new(),
                _before_data_updates: Vec::new(),
                _after_mesh_updates: Vec::new(),
                _after_data_updates: Vec::new(),
                #[cfg(test)]
                _physical_observations: PreparedVariablePhysicalObservations::default(),
            });
            return;
        };
        let (before, after, _) = coverage_publication
            .hold_phases
            .take()
            .unwrap_or_else(|| unreachable!("coverage publication owns its hold phases"))
            .into_parts();
        let (before_ledger, before_mesh, before_data, _) = before.into_parts();
        let (after_ledger, after_mesh, after_data, _) = after.into_parts();
        #[cfg(test)]
        let physical_observations = std::mem::take(&mut coverage_publication.physical_observations);

        debug_assert!(retirement.previous_coverage_holds.is_none());
        retirement.previous_coverage_holds = Some(std::mem::replace(
            &mut runtime.coverage_holds,
            before_ledger,
        ));
        let coverage = coverage_publication
            .preview
            .take()
            .unwrap_or_else(|| unreachable!("coverage publication owns one preview token"))
            .publish(&mut runtime.coverage);
        let (coverage_result, coverage_state) = coverage.into_parts();
        debug_assert!(retirement.coverage_state.is_none());
        retirement.coverage_state = Some(coverage_state);
        debug_assert!(retirement.before_topology_coverage_holds.is_none());
        retirement.before_topology_coverage_holds =
            Some(std::mem::replace(&mut runtime.coverage_holds, after_ledger));
        debug_assert!(retirement.variable_mode.is_none());
        retirement.variable_mode = Some(VariableModeRetirement {
            _coordinator_delta: coordinator_delta,
            _coverage_result: Some(coverage_result),
            _before_mesh_updates: before_mesh,
            _before_data_updates: before_data,
            _after_mesh_updates: after_mesh,
            _after_data_updates: after_data,
            #[cfg(test)]
            _physical_observations: physical_observations,
        });
    }

    fn commit_prepared_completion(
        &mut self,
        completion: DurableCompletion,
        disposition: PreparedCompletionDisposition,
        retirement: &mut RetirementBag,
    ) {
        let publish_followups = match disposition {
            PreparedCompletionDisposition::Retire { publish_followups }
            | PreparedCompletionDisposition::ApplyPersistence {
                publish_followups, ..
            } => publish_followups,
            PreparedCompletionDisposition::Quarantine => false,
        };
        debug_assert!(!publish_followups || accepted_completion_has_no_followups(&completion));

        match disposition {
            PreparedCompletionDisposition::Retire { .. } => {
                retirement.durable_completions.push(completion);
            }
            PreparedCompletionDisposition::Quarantine => {
                self.commit_quarantined_completion(completion, retirement);
            }
            PreparedCompletionDisposition::ApplyPersistence { action, .. } => {
                match (action, completion) {
                    (
                        PreparedPersistenceAction::Save(action),
                        DurableCompletion::SaveAcknowledged {
                            completed,
                            terminal,
                            attempt_ordinal,
                        },
                    ) => {
                        let terminal = self.commit_prepared_save_action(
                            action,
                            terminal,
                            attempt_ordinal,
                            retirement,
                        );
                        debug_assert!(terminal.is_none());
                        retirement.completed_tasks.push(completed);
                    }
                    (
                        PreparedPersistenceAction::Checkpoint(action),
                        DurableCompletion::FlushAcknowledged {
                            completed,
                            terminal,
                            attempt_ordinal,
                        },
                    ) => {
                        let terminal = self.commit_prepared_checkpoint_action(
                            action,
                            terminal,
                            attempt_ordinal,
                            retirement,
                        );
                        debug_assert!(terminal.is_none());
                        retirement.completed_tasks.push(completed);
                    }
                    (
                        PreparedPersistenceAction::Save(action),
                        DurableCompletion::PersistenceTerminal {
                            completed,
                            kind,
                            terminal: PersistenceTaskTerminal::Save(terminal),
                            attempt_ordinal,
                        },
                    ) => {
                        let quarantine =
                            matches!(action, PreparedSaveAction::MarkIndeterminate { .. });
                        let terminal = self.commit_prepared_save_action(
                            action,
                            terminal,
                            attempt_ordinal,
                            retirement,
                        );
                        if quarantine {
                            let terminal = terminal
                                .expect("indeterminate prepared save retains its terminal owner");
                            self.completion_quarantine.push_back(
                                QuarantinedCompletion::Persistence {
                                    kind,
                                    terminal: PersistenceTaskTerminal::Save(terminal),
                                    attempt_ordinal,
                                    completed,
                                },
                            );
                        } else {
                            retirement.completed_tasks.push(completed);
                        }
                    }
                    (
                        PreparedPersistenceAction::Checkpoint(action),
                        DurableCompletion::PersistenceTerminal {
                            completed,
                            kind,
                            terminal: PersistenceTaskTerminal::Flush(terminal),
                            attempt_ordinal,
                        },
                    ) => {
                        let quarantine =
                            matches!(action, PreparedCheckpointAction::MarkIndeterminate { .. });
                        let terminal = self.commit_prepared_checkpoint_action(
                            action,
                            terminal,
                            attempt_ordinal,
                            retirement,
                        );
                        if quarantine {
                            let terminal = terminal.expect(
                                "indeterminate prepared checkpoint retains its terminal owner",
                            );
                            self.completion_quarantine.push_back(
                                QuarantinedCompletion::Persistence {
                                    kind,
                                    terminal: PersistenceTaskTerminal::Flush(terminal),
                                    attempt_ordinal,
                                    completed,
                                },
                            );
                        } else {
                            retirement.completed_tasks.push(completed);
                        }
                    }
                    (_, other) => {
                        debug_assert!(false, "prepared persistence owner identity changed");
                        retirement.durable_completions.push(other);
                    }
                }
            }
        }
    }

    fn commit_prepared_save_action(
        &mut self,
        action: PreparedSaveAction,
        terminal: SaveTaskTerminal,
        attempt_ordinal: u64,
        retirement: &mut RetirementBag,
    ) -> Option<SaveTaskTerminal> {
        match action {
            PreparedSaveAction::AcknowledgeSuccess {
                key,
                block_revision,
                generation,
                attempt_ordinal: expected_attempt,
            } => {
                debug_assert_eq!(terminal.block_revision, block_revision);
                debug_assert_eq!(terminal.save_generation, generation);
                debug_assert_eq!(attempt_ordinal, expected_attempt);
                let SaveTaskTerminal {
                    location,
                    save_generation,
                    payload,
                    acknowledgement,
                    ..
                } = terminal;
                debug_assert_eq!(key, SaveKey::new(location.position, location.lod_index));
                debug_assert!(matches!(
                    acknowledgement,
                    Some(PersistenceAcknowledgement::Save(Ok(())))
                ));
                let entry = self
                    .save_journal
                    .get_mut(&key)
                    .unwrap_or_else(|| unreachable!("prepared save key remains live"));
                let active = entry
                    .active
                    .take()
                    .unwrap_or_else(|| unreachable!("prepared save attempt remains active"));
                let ActiveSaveAttempt::WriteInFlight {
                    meta,
                    attempt_ordinal: current_attempt,
                } = active
                else {
                    unreachable!("prepared save attempt retains its in-flight state")
                };
                debug_assert_eq!(meta.block_revision, block_revision);
                debug_assert_eq!(meta.generation, generation);
                debug_assert_eq!(current_attempt, expected_attempt);
                debug_assert!(entry.written_unflushed.is_none());
                retirement.save_attempt_meta.push(meta);
                entry.written_unflushed = Some(WrittenSave {
                    block_revision,
                    generation: save_generation,
                    payload,
                });
                None
            }
            PreparedSaveAction::AcknowledgeFailure {
                key,
                block_revision,
                generation,
                attempt_ordinal: expected_attempt,
                next_retry_count,
            } => {
                debug_assert_eq!(terminal.block_revision, block_revision);
                debug_assert_eq!(terminal.save_generation, generation);
                debug_assert_eq!(attempt_ordinal, expected_attempt);
                let SaveTaskTerminal {
                    location,
                    payload,
                    acknowledgement,
                    ..
                } = terminal;
                debug_assert_eq!(key, SaveKey::new(location.position, location.lod_index));
                let Some(PersistenceAcknowledgement::Save(Err(error))) = acknowledgement else {
                    unreachable!("prepared failed save retains its stream error")
                };
                let entry = self
                    .save_journal
                    .get_mut(&key)
                    .unwrap_or_else(|| unreachable!("prepared save key remains live"));
                let active = entry
                    .active
                    .take()
                    .unwrap_or_else(|| unreachable!("prepared save attempt remains active"));
                let ActiveSaveAttempt::WriteInFlight {
                    mut meta,
                    attempt_ordinal: current_attempt,
                } = active
                else {
                    unreachable!("prepared save attempt retains its in-flight state")
                };
                debug_assert_eq!(meta.block_revision, block_revision);
                debug_assert_eq!(meta.generation, generation);
                debug_assert_eq!(current_attempt, expected_attempt);
                meta.retry_count = next_retry_count;
                if let Some(previous) = meta.last_error.replace(error) {
                    retirement.stream_errors.push(previous);
                }
                entry.active = Some(ActiveSaveAttempt::Pending(PendingSave { meta, payload }));
                None
            }
            PreparedSaveAction::RestoreBeforeIo {
                key,
                block_revision,
                generation,
                attempt_ordinal: expected_attempt,
                next_retry_count,
            } => {
                debug_assert_eq!(terminal.block_revision, block_revision);
                debug_assert_eq!(terminal.save_generation, generation);
                debug_assert_eq!(attempt_ordinal, expected_attempt);
                let SaveTaskTerminal {
                    location,
                    payload,
                    acknowledgement,
                    ..
                } = terminal;
                debug_assert_eq!(key, SaveKey::new(location.position, location.lod_index));
                debug_assert!(acknowledgement.is_none());
                let entry = self
                    .save_journal
                    .get_mut(&key)
                    .unwrap_or_else(|| unreachable!("prepared save key remains live"));
                let active = entry
                    .active
                    .take()
                    .unwrap_or_else(|| unreachable!("prepared save attempt remains active"));
                let ActiveSaveAttempt::WriteInFlight {
                    mut meta,
                    attempt_ordinal: current_attempt,
                } = active
                else {
                    unreachable!("prepared save attempt retains its in-flight state")
                };
                debug_assert_eq!(meta.block_revision, block_revision);
                debug_assert_eq!(meta.generation, generation);
                debug_assert_eq!(current_attempt, expected_attempt);
                meta.retry_count = next_retry_count;
                entry.active = Some(ActiveSaveAttempt::Pending(PendingSave { meta, payload }));
                None
            }
            PreparedSaveAction::MarkIndeterminate {
                key,
                block_revision,
                generation,
                attempt_ordinal: expected_attempt,
            } => {
                debug_assert_eq!(terminal.block_revision, block_revision);
                debug_assert_eq!(terminal.save_generation, generation);
                debug_assert_eq!(attempt_ordinal, expected_attempt);
                debug_assert_eq!(
                    key,
                    SaveKey::new(terminal.location.position, terminal.location.lod_index)
                );
                let entry = self
                    .save_journal
                    .get_mut(&key)
                    .unwrap_or_else(|| unreachable!("prepared save key remains live"));
                let active = entry
                    .active
                    .take()
                    .unwrap_or_else(|| unreachable!("prepared save attempt remains active"));
                let ActiveSaveAttempt::WriteInFlight {
                    meta,
                    attempt_ordinal: current_attempt,
                } = active
                else {
                    unreachable!("prepared save attempt retains its in-flight state")
                };
                debug_assert_eq!(meta.block_revision, block_revision);
                debug_assert_eq!(meta.generation, generation);
                debug_assert_eq!(current_attempt, expected_attempt);
                entry.active = Some(ActiveSaveAttempt::Indeterminate {
                    meta,
                    attempt_ordinal: current_attempt,
                });
                Some(terminal)
            }
        }
    }

    fn commit_prepared_checkpoint_action(
        &mut self,
        action: PreparedCheckpointAction,
        terminal: FlushTaskTerminal,
        attempt_ordinal: u64,
        retirement: &mut RetirementBag,
    ) -> Option<FlushTaskTerminal> {
        match action {
            PreparedCheckpointAction::Acknowledge {
                checkpoint_generation,
                attempt_ordinal: expected_attempt,
                succeeded,
                origin,
                mut explicit_outcome,
                mut entry_actions,
            } => {
                debug_assert_eq!(terminal.checkpoint_generation, checkpoint_generation);
                debug_assert_eq!(attempt_ordinal, expected_attempt);
                let FlushTaskTerminal {
                    acknowledgement, ..
                } = terminal;
                let Some(PersistenceAcknowledgement::Flush(result)) = acknowledgement else {
                    unreachable!("prepared checkpoint acknowledgement retains its result")
                };
                debug_assert_eq!(result.is_ok(), succeeded);
                let checkpoint = self
                    .save_checkpoint_in_flight
                    .take()
                    .unwrap_or_else(|| unreachable!("prepared checkpoint remains live"));
                debug_assert_eq!(checkpoint.checkpoint_generation, checkpoint_generation);
                debug_assert_eq!(
                    checkpoint.state,
                    CheckpointAttemptState::WriteInFlight {
                        attempt_ordinal: expected_attempt,
                    }
                );
                debug_assert_eq!(checkpoint.origin, origin);

                if let Some(outcome) = explicit_outcome.take() {
                    if let Some(previous) = self
                        .last_checkpoint_outcome
                        .replace((checkpoint_generation, outcome))
                    {
                        retirement.checkpoint_outcomes.push(previous);
                    }
                }
                match result {
                    Ok(()) => {
                        if let Some(previous) = self.last_save_checkpoint_error.take() {
                            retirement.stream_errors.push(previous);
                        }
                        self.automatic_save_checkpoint_blocked = false;
                    }
                    Err(error) => {
                        if let Some(previous) = self.last_save_checkpoint_error.replace(error) {
                            retirement.stream_errors.push(previous);
                        }
                        self.automatic_save_checkpoint_blocked = true;
                    }
                }

                for entry_action in entry_actions.drain(..) {
                    match entry_action {
                        PreparedCheckpointEntryAction::ClearWritten {
                            key,
                            block_revision,
                            generation,
                            promote_queued,
                            remove_entry,
                            ..
                        } => {
                            let entry = self.save_journal.get_mut(&key).unwrap_or_else(|| {
                                unreachable!("prepared checkpoint key remains live")
                            });
                            let written = entry.written_unflushed.take().unwrap_or_else(|| {
                                unreachable!("prepared acknowledged payload remains live")
                            });
                            debug_assert_eq!(written.block_revision, block_revision);
                            debug_assert_eq!(written.generation, generation);
                            retirement.written_saves.push(written);
                            if promote_queued {
                                debug_assert!(entry.active.is_none());
                                let pending = entry.queued_newer.pop_front().unwrap_or_else(|| {
                                    unreachable!("prepared queued save remains live")
                                });
                                entry.active = Some(ActiveSaveAttempt::Pending(pending));
                            }
                            if remove_entry {
                                let removed = self.save_journal.remove(&key).unwrap_or_else(|| {
                                    unreachable!("prepared empty journal entry remains live")
                                });
                                retirement.save_journal_entries.push(removed);
                            }
                        }
                        PreparedCheckpointEntryAction::RestoreWritten {
                            key,
                            block_revision,
                            generation,
                            placement,
                            error,
                        } => {
                            let entry = self.save_journal.get_mut(&key).unwrap_or_else(|| {
                                unreachable!("prepared checkpoint key remains live")
                            });
                            let written = entry.written_unflushed.take().unwrap_or_else(|| {
                                unreachable!("prepared acknowledged payload remains live")
                            });
                            debug_assert_eq!(written.block_revision, block_revision);
                            debug_assert_eq!(written.generation, generation);
                            let retry_count = u32::from(error.is_some());
                            let restored = PendingSave {
                                meta: SaveAttemptMeta {
                                    block_revision,
                                    generation,
                                    retry_count,
                                    last_error: error,
                                },
                                payload: written.payload,
                            };
                            match placement {
                                PreparedCheckpointRestorePlacement::Active => {
                                    debug_assert!(entry.active.is_none());
                                    entry.active = Some(ActiveSaveAttempt::Pending(restored));
                                }
                                PreparedCheckpointRestorePlacement::ReplacePending => {
                                    let active = entry.active.take().unwrap_or_else(|| {
                                        unreachable!("prepared pending save remains live")
                                    });
                                    let ActiveSaveAttempt::Pending(pending) = active else {
                                        unreachable!("prepared replacement target remains pending")
                                    };
                                    entry.queued_newer.push_front(pending);
                                    entry.active = Some(ActiveSaveAttempt::Pending(restored));
                                }
                                PreparedCheckpointRestorePlacement::Queue => {
                                    debug_assert!(entry.active.is_some());
                                    entry.queued_newer.push_front(restored);
                                }
                            }
                        }
                    }
                }
                retirement
                    .checkpoint_actions
                    .push(PreparedCheckpointAction::Acknowledge {
                        checkpoint_generation,
                        attempt_ordinal: expected_attempt,
                        succeeded,
                        origin,
                        explicit_outcome,
                        entry_actions,
                    });
                retirement.checkpoints.push(checkpoint);
                self.automatic_checkpoint_satisfied_empty_flush =
                    succeeded && self.save_journal.is_empty();
                self.force_checkpoint_requested = false;
                None
            }
            PreparedCheckpointAction::RestoreBeforeIo {
                checkpoint_generation,
                attempt_ordinal: expected_attempt,
                next_retry_count,
            } => {
                debug_assert_eq!(terminal.checkpoint_generation, checkpoint_generation);
                debug_assert_eq!(attempt_ordinal, expected_attempt);
                let checkpoint = self
                    .save_checkpoint_in_flight
                    .as_mut()
                    .unwrap_or_else(|| unreachable!("prepared checkpoint remains live"));
                debug_assert_eq!(checkpoint.checkpoint_generation, checkpoint_generation);
                debug_assert_eq!(
                    checkpoint.state,
                    CheckpointAttemptState::WriteInFlight {
                        attempt_ordinal: expected_attempt,
                    }
                );
                checkpoint.retry_count = next_retry_count;
                checkpoint.state = CheckpointAttemptState::Pending;
                self.deferred_checkpoint_dispatch = true;
                None
            }
            PreparedCheckpointAction::MarkIndeterminate {
                checkpoint_generation,
                attempt_ordinal: expected_attempt,
            } => {
                debug_assert_eq!(terminal.checkpoint_generation, checkpoint_generation);
                debug_assert_eq!(attempt_ordinal, expected_attempt);
                let checkpoint = self
                    .save_checkpoint_in_flight
                    .as_mut()
                    .unwrap_or_else(|| unreachable!("prepared checkpoint remains live"));
                debug_assert_eq!(checkpoint.checkpoint_generation, checkpoint_generation);
                debug_assert_eq!(
                    checkpoint.state,
                    CheckpointAttemptState::WriteInFlight {
                        attempt_ordinal: expected_attempt,
                    }
                );
                checkpoint.state = CheckpointAttemptState::Indeterminate {
                    attempt_ordinal: expected_attempt,
                };
                Some(terminal)
            }
        }
    }

    fn commit_quarantined_completion(
        &mut self,
        completion: DurableCompletion,
        retirement: &mut RetirementBag,
    ) {
        match completion {
            DurableCompletion::PersistenceTerminal {
                completed,
                kind,
                terminal,
                attempt_ordinal,
            } => self
                .completion_quarantine
                .push_back(QuarantinedCompletion::Persistence {
                    kind,
                    terminal,
                    attempt_ordinal,
                    completed,
                }),
            DurableCompletion::MalformedPersistence {
                completed,
                kind,
                terminal,
                attempt_ordinal,
            } => {
                self.completion_quarantine
                    .push_back(QuarantinedCompletion::MalformedPersistence {
                        kind,
                        terminal,
                        attempt_ordinal,
                        completed,
                    })
            }
            DurableCompletion::MalformedFinished { completed, kind } => self
                .completion_quarantine
                .push_back(QuarantinedCompletion::Other { kind, completed }),
            DurableCompletion::UnknownTerminal { completed } => self
                .completion_quarantine
                .push_back(QuarantinedCompletion::Other {
                    kind: CompletionTaskKind::Unknown,
                    completed,
                }),
            other => {
                debug_assert!(false, "only quarantine dispositions reach this branch");
                // Preserve unexpected ownership until every publication guard
                // is gone instead of dropping it in the commit tail.
                retirement.durable_completions.push(other);
            }
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn prepare_variable_physical_slice(
        &mut self,
        coordinator_update: &ValidatedCoordinatorUpdate,
        coverage_preview: &ValidatedCoveragePreview,
        coverage_inputs: &[CoverageInput],
        hold_resolution: PreparedCoverageHoldResolution,
        data_preview: &SharedVoxelDataTransactionPreview,
    ) -> Result<(PreparedVariablePhysicalSlice, PreparedCoverageHoldPhases), VariableModeTestError>
    {
        let mut mesh_shadows = BTreeMap::new();
        let mut data_shadows = BTreeMap::new();
        let mut mesh_insert_allowed = BTreeSet::new();

        for change in &coordinator_update.delta().changes {
            match change.key.kind {
                ResidentBlockKind::Mesh => {
                    ensure_variable_mesh_shadow(self, &mut mesh_shadows, change.key.location)?;
                    if change.new_counts.resident != 0 {
                        mesh_insert_allowed
                            .insert(canonical_mesh_location_key(change.key.location));
                    }
                }
                ResidentBlockKind::Data => {
                    ensure_variable_data_shadow(
                        self,
                        data_preview,
                        &mut data_shadows,
                        BlockLocation {
                            position: change.key.location.position_in_blocks,
                            lod_index: change.key.location.lod_index,
                        },
                    )?;
                }
            }
        }

        let mut mesh_resources = Vec::new();
        let mesh_resource_count = hold_resolution.mesh_resources().count();
        mesh_resources
            .try_reserve_exact(mesh_resource_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        mesh_resources.extend(hold_resolution.mesh_resources());
        let mut data_resources = Vec::new();
        let data_resource_count = hold_resolution.data_resources().count();
        data_resources
            .try_reserve_exact(data_resource_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        data_resources.extend(hold_resolution.data_resources());

        for resource in &mesh_resources {
            let location = resource.location();
            if hold_resolution
                .after_topology_ledger()
                .target_owner_ids(location)
                .next()
                .is_some()
            {
                mesh_insert_allowed.insert(canonical_mesh_location_key(location));
            }
        }

        for resource in &mesh_resources {
            ensure_variable_mesh_shadow(self, &mut mesh_shadows, resource.location())?;
        }
        for resource in &data_resources {
            ensure_variable_data_shadow(
                self,
                data_preview,
                &mut data_shadows,
                BlockLocation {
                    position: resource.position_in_blocks(),
                    lod_index: resource.lod_index(),
                },
            )?;
        }
        for group in &coverage_preview.result().topology.groups {
            for location in group.activate.iter().chain(&group.deactivate).copied() {
                ensure_variable_mesh_shadow(self, &mut mesh_shadows, location)?;
            }
        }
        for (location, _) in &coverage_preview.result().topology.transition_masks {
            ensure_variable_mesh_shadow(self, &mut mesh_shadows, *location)?;
        }
        for input in coverage_inputs {
            let location = match input {
                CoverageInput::SetDemand { location, .. }
                | CoverageInput::Accept { location, .. }
                | CoverageInput::Evict { location } => *location,
                CoverageInput::SetJoinTargetState { id, .. } => id.join_parent,
            };
            ensure_variable_mesh_shadow(self, &mut mesh_shadows, location)?;
        }
        for (key, shadow) in &mut mesh_shadows {
            if shadow.next.is_none() && mesh_insert_allowed.contains(key) {
                shadow.next = Some(MeshBlockEntry {
                    position: shadow.location.position_in_blocks,
                    ..MeshBlockEntry::default()
                });
            }
        }
        self.fixed_capacity_checkpoint_for_test(
            FixedCapacityDestination::VariablePhysicalResourceSnapshot,
        )?;

        let mut resources = CoverageHoldResourceSnapshot::default();
        for resource in &mesh_resources {
            let shadow = mesh_shadows
                .get(&canonical_mesh_location_key(resource.location()))
                .unwrap_or_else(|| unreachable!("resolved mesh resource owns one shadow"));
            resources.set_mesh_refcount(
                *resource,
                shadow
                    .expected
                    .as_ref()
                    .map_or(0, |entry| mesh_hold_refcount(entry, resource.feature())),
            );
        }
        for resource in &data_resources {
            let location = BlockLocation {
                position: resource.position_in_blocks(),
                lod_index: resource.lod_index(),
            };
            let shadow = data_shadows
                .get(&canonical_block_location_key(location))
                .unwrap_or_else(|| unreachable!("resolved data resource owns one shadow"));
            resources.set_data_resource(
                *resource,
                shadow.next.coverage_holds,
                shadow.snapshot.is_present(),
            );
        }

        // Coordinator CAS is deliberately first. Coverage-hold counts are a
        // disjoint field family, so the ledger snapshot remains the exact
        // pre-transaction hold state while viewer residency moves first.
        for change in &coordinator_update.delta().changes {
            match change.key.kind {
                ResidentBlockKind::Mesh => {
                    let location = change.key.location;
                    let shadow = mesh_shadows
                        .get_mut(&canonical_mesh_location_key(location))
                        .unwrap_or_else(|| unreachable!("coordinator mesh owns one shadow"));
                    let entry = shadow
                        .next
                        .as_mut()
                        .ok_or(VariablePhysicalPrepareError::MeshInsertDeferred { location })?;
                    for (field, expected, actual) in [
                        (
                            DemandCountField::Resident,
                            change.old_counts.resident,
                            entry.resident_viewers,
                        ),
                        (
                            DemandCountField::Visuals,
                            change.old_counts.visuals,
                            entry.visual_viewers,
                        ),
                        (
                            DemandCountField::Collisions,
                            change.old_counts.collisions,
                            entry.collision_viewers,
                        ),
                    ] {
                        if actual != expected {
                            return Err(
                                VariablePhysicalPrepareError::CoordinatorMeshCountMismatch {
                                    location,
                                    field,
                                    expected,
                                    actual,
                                }
                                .into(),
                            );
                        }
                    }
                    entry.resident_viewers = change.new_counts.resident;
                    entry.visual_viewers = change.new_counts.visuals;
                    entry.collision_viewers = change.new_counts.collisions;
                }
                ResidentBlockKind::Data => {
                    let location = BlockLocation {
                        position: change.key.location.position_in_blocks,
                        lod_index: change.key.location.lod_index,
                    };
                    let shadow = data_shadows
                        .get_mut(&canonical_block_location_key(location))
                        .unwrap_or_else(|| unreachable!("coordinator data owns one shadow"));
                    if shadow.next.resident_viewers != change.old_counts.resident {
                        return Err(VariablePhysicalPrepareError::CoordinatorDataCountMismatch {
                            location,
                            expected: change.old_counts.resident,
                            actual: shadow.next.resident_viewers,
                        }
                        .into());
                    }
                    shadow.next.resident_viewers = change.new_counts.resident;
                    shadow.next.checked_total(location)?;
                }
            }
        }

        self.fixed_capacity_checkpoint_for_test(
            FixedCapacityDestination::VariableCoverageHoldBind,
        )?;
        let hold_phases = hold_resolution
            .bind(&resources)
            .map_err(VoxelTerrainRuntimeError::from)?;
        for update in hold_phases.before_topology().mesh_refcount_updates() {
            let resource = update.resource();
            let shadow = mesh_shadows
                .get_mut(&canonical_mesh_location_key(resource.location()))
                .unwrap_or_else(|| unreachable!("before-hold mesh owns one shadow"));
            let entry =
                shadow
                    .next
                    .as_mut()
                    .ok_or(VariablePhysicalPrepareError::MeshInsertDeferred {
                        location: resource.location(),
                    })?;
            let actual = mesh_hold_refcount(entry, resource.feature());
            if actual != update.expected() {
                return Err(VariablePhysicalPrepareError::MeshHoldCountMismatch {
                    resource,
                    expected: update.expected(),
                    actual,
                }
                .into());
            }
            set_mesh_hold_refcount(entry, resource.feature(), update.next());
        }
        for update in hold_phases.before_topology().data_refcount_updates() {
            let resource = update.resource();
            let location = BlockLocation {
                position: resource.position_in_blocks(),
                lod_index: resource.lod_index(),
            };
            let shadow = data_shadows
                .get_mut(&canonical_block_location_key(location))
                .unwrap_or_else(|| unreachable!("before-hold data owns one shadow"));
            if shadow.next.coverage_holds != update.expected() {
                return Err(VariablePhysicalPrepareError::DataHoldCountMismatch {
                    location,
                    expected: update.expected(),
                    actual: shadow.next.coverage_holds,
                }
                .into());
            }
            shadow.next.coverage_holds = update.next();
            shadow.next.checked_total(location)?;
        }

        let topology_activates = coverage_preview
            .result()
            .topology
            .groups
            .iter()
            .any(|group| !group.activate.is_empty());
        if topology_activates
            && (!hold_phases.before_topology().all_required_data_ready()
                || !hold_phases.after_topology().all_required_data_ready())
        {
            let location = data_shadows
                .values()
                .find(|shadow| !shadow.snapshot.is_present())
                .map(|shadow| shadow.location)
                .unwrap_or(BlockLocation {
                    position: Vector3i::zero(),
                    lod_index: 0,
                });
            return Err(VariablePhysicalPrepareError::TopologyDataNotReady { location }.into());
        }

        for group in &coverage_preview.result().topology.groups {
            for location in &group.activate {
                let shadow = mesh_shadows
                    .get_mut(&canonical_mesh_location_key(*location))
                    .unwrap_or_else(|| unreachable!("topology activation owns one shadow"));
                let entry = shadow.next.as_mut().ok_or(
                    VariablePhysicalPrepareError::MeshInsertDeferred {
                        location: *location,
                    },
                )?;
                set_mesh_feature_active(entry, group.feature, true);
            }
            for location in &group.deactivate {
                let shadow = mesh_shadows
                    .get_mut(&canonical_mesh_location_key(*location))
                    .unwrap_or_else(|| unreachable!("topology deactivation owns one shadow"));
                let entry = shadow.next.as_mut().ok_or(
                    VariablePhysicalPrepareError::MeshInsertDeferred {
                        location: *location,
                    },
                )?;
                set_mesh_feature_active(entry, group.feature, false);
            }
        }

        for update in hold_phases.after_topology().mesh_refcount_updates() {
            let resource = update.resource();
            let shadow = mesh_shadows
                .get_mut(&canonical_mesh_location_key(resource.location()))
                .unwrap_or_else(|| unreachable!("after-hold mesh owns one shadow"));
            let entry =
                shadow
                    .next
                    .as_mut()
                    .ok_or(VariablePhysicalPrepareError::MeshInsertDeferred {
                        location: resource.location(),
                    })?;
            let actual = mesh_hold_refcount(entry, resource.feature());
            if actual != update.expected() {
                return Err(VariablePhysicalPrepareError::MeshHoldCountMismatch {
                    resource,
                    expected: update.expected(),
                    actual,
                }
                .into());
            }
            set_mesh_hold_refcount(entry, resource.feature(), update.next());
        }
        for update in hold_phases.after_topology().data_refcount_updates() {
            let resource = update.resource();
            let location = BlockLocation {
                position: resource.position_in_blocks(),
                lod_index: resource.lod_index(),
            };
            let shadow = data_shadows
                .get_mut(&canonical_block_location_key(location))
                .unwrap_or_else(|| unreachable!("after-hold data owns one shadow"));
            if shadow.next.coverage_holds != update.expected() {
                return Err(VariablePhysicalPrepareError::DataHoldCountMismatch {
                    location,
                    expected: update.expected(),
                    actual: shadow.next.coverage_holds,
                }
                .into());
            }
            shadow.next.coverage_holds = update.next();
            shadow.next.checked_total(location)?;
        }
        self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::VariablePhysicalShadows)?;

        // Validate the entire touched mesh union, including unchanged-active
        // fallbacks and zero-net resources. An activation-only check would
        // miss a coordinator/hold CAS that accidentally cleared a stable
        // frontier member.
        for shadow in mesh_shadows.values() {
            for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
                let location = shadow.location;
                let expected = coverage_preview.next_is_active(location, feature);
                let actual = shadow
                    .next
                    .as_ref()
                    .is_some_and(|entry| mesh_feature_active(entry, feature));
                if actual != expected {
                    return Err(VariablePhysicalPrepareError::TopologyFinalStateMismatch {
                        location,
                        feature,
                        expected,
                        actual,
                    }
                    .into());
                }
                if !expected {
                    continue;
                }
                let entry = shadow.next.as_ref().unwrap_or_else(|| {
                    unreachable!("active topology validation retained its mesh entry")
                });
                let Some(accepted) = coverage_preview.next_accepted_snapshot(location) else {
                    return Err(
                        VariablePhysicalPrepareError::ActivationMissingAcceptedSnapshot {
                            location,
                            feature,
                        }
                        .into(),
                    );
                };
                if entry.resident_refcount() == 0 {
                    return Err(VariablePhysicalPrepareError::ActivationNotResident {
                        location,
                        feature,
                    }
                    .into());
                }
                if !mesh_feature_needed(entry, feature) {
                    return Err(VariablePhysicalPrepareError::ActivationFeatureNotNeeded {
                        location,
                        feature,
                    }
                    .into());
                }
                let Some(upload) = entry.accepted_upload.as_ref() else {
                    return Err(VariablePhysicalPrepareError::ActivationMissingUpload {
                        location,
                        feature,
                    }
                    .into());
                };
                let key = upload.key();
                if key.location != location {
                    return Err(
                        VariablePhysicalPrepareError::ActivationUploadLocationMismatch {
                            location,
                            feature,
                            actual: key.location,
                        }
                        .into(),
                    );
                }
                if key.revision != accepted.revision {
                    return Err(
                        VariablePhysicalPrepareError::ActivationUploadRevisionMismatch {
                            location,
                            feature,
                            expected: accepted.revision,
                            actual: key.revision,
                        }
                        .into(),
                    );
                }
                let features = upload.features();
                let visual_built = upload.visual_state() != PayloadState::NotBuilt;
                let collision_built = upload.collision_state() != PayloadState::NotBuilt;
                if visual_built != accepted.visuals
                    || collision_built != accepted.collisions
                    || features.visuals != accepted.visuals
                    || features.collisions != accepted.collisions
                    || !features.variable_lod
                {
                    return Err(
                        VariablePhysicalPrepareError::ActivationUploadFeaturesMismatch {
                            location,
                            feature,
                            expected: accepted,
                            actual: features,
                        }
                        .into(),
                    );
                }
            }
        }

        let topology = &coverage_preview.result().topology;
        let mut next_render_topology_revision = self.next_render_topology_revision;
        let mut events_to_append = Vec::new();
        if !topology.groups.is_empty() || !topology.transition_masks.is_empty() {
            self.fixed_capacity_checkpoint_for_test(
                FixedCapacityDestination::VariableTopologyEvent,
            )?;
            let revision = next_render_topology_revision;
            next_render_topology_revision = revision
                .checked_add(1)
                .ok_or(VoxelTerrainRuntimeError::RenderTopologyRevisionOverflow)?;
            let mut batch = topology.clone();
            batch.revision = revision;
            events_to_append
                .try_reserve_exact(1)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            events_to_append.push(VoxelTerrainEvent::RenderTopologyChanged(batch));
        }

        let lod_count = usize::from(self.lod_count);
        let mut next_pending_load = Vec::new();
        let mut next_pending_mesh = Vec::new();
        next_pending_load
            .try_reserve_exact(lod_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        next_pending_mesh
            .try_reserve_exact(lod_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for lod in 0..lod_count {
            let mut load = Vec::new();
            load.try_reserve_exact(self.blocks_pending_load[lod].len())
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            load.extend_from_slice(&self.blocks_pending_load[lod]);
            next_pending_load.push(load);
            let mut mesh = Vec::new();
            mesh.try_reserve_exact(self.blocks_pending_update[lod].len())
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            mesh.extend_from_slice(&self.blocks_pending_update[lod]);
            next_pending_mesh.push(mesh);
        }

        let mut next_request_generation = self.next_request_generation;
        let mut next_mesh_revision = self.next_mesh_revision;
        let mut next_stats = self.stats;
        let mut scheduled_tasks = Vec::new();
        let mut loading_diffs = Vec::new();
        let mut data_residency_diffs = Vec::new();
        let mut data_operations = Vec::new();
        let mut removed_routes = Vec::new();
        let mut next_save_generation = self.next_save_generation;
        let mut vacant_save_count = 0usize;
        let mut dirty_retained_count = 0usize;
        let mut queued_save_keys = Vec::new();

        // Discover the complete mesh halo union before planning data so every
        // newly observed missing key participates in this transaction's load
        // planner instead of becoming an observation-only orphan.
        let mut mesh_work = Vec::new();
        mesh_work
            .try_reserve_exact(mesh_shadows.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for shadow in mesh_shadows.values() {
            let Some(entry) = shadow.next.as_ref() else {
                continue;
            };
            if entry.resident_refcount() == 0 || (!entry.needs_visual() && !entry.needs_collision())
            {
                continue;
            }
            let required = MeshBuildFeatures {
                visuals: entry.needs_visual(),
                collisions: entry.needs_collision(),
                variable_lod: true,
            };
            let accepted =
                entry.accepted_upload.is_some() && entry.applied_features.contains(required);
            if !accepted {
                mesh_work.push(shadow.location);
            }
        }
        for location in &mesh_work {
            let halo =
                clipped_meshing_data_box(*location, self.data_block_size(), 1, self.data.bounds())
                    .map_err(VoxelTerrainRuntimeError::LodMath)?;
            for position in halo.iter_cells_zxy() {
                ensure_variable_data_shadow(
                    self,
                    data_preview,
                    &mut data_shadows,
                    BlockLocation {
                        position,
                        lod_index: location.lod_index,
                    },
                )?;
            }
        }

        let physical_upper = data_shadows
            .len()
            .checked_add(mesh_shadows.len())
            .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
        scheduled_tasks
            .try_reserve_exact(physical_upper)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        loading_diffs
            .try_reserve_exact(data_shadows.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        data_residency_diffs
            .try_reserve_exact(data_shadows.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        data_operations
            .try_reserve_exact(data_shadows.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        removed_routes
            .try_reserve_exact(data_shadows.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        queued_save_keys
            .try_reserve_exact(data_shadows.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        events_to_append
            .try_reserve(physical_upper)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for queue in &mut next_pending_mesh {
            queue
                .try_reserve(mesh_shadows.len())
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        }

        // Data is planned before mesh dispatch so every newly missing halo
        // owns exactly one load task in the same prepared batch.
        for shadow in data_shadows.values_mut() {
            let lod = usize::from(shadow.location.lod_index);
            if shadow.snapshot.is_present() {
                if shadow.next.is_empty() {
                    data_operations.push(SharedVoxelDataTransactionOperation::Remove {
                        location: shadow.location,
                    });
                    data_residency_diffs.push(PreparedDataResidencyDiff {
                        location: shadow.location,
                        expected: shadow.expected_loaded,
                        action: PreparedMapAction::Remove,
                    });
                    let route = if !shadow.snapshot.is_modified() {
                        PreparedRemovedOwnerRoute::Clean
                    } else {
                        let VoxelDataKeyRevision::Present(block_revision) =
                            shadow.snapshot.revision()
                        else {
                            unreachable!("present removal snapshot has a present revision")
                        };
                        if !shadow.snapshot.has_voxels() {
                            dirty_retained_count = dirty_retained_count
                                .checked_add(1)
                                .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                            PreparedRemovedOwnerRoute::DirtyRetained {
                                location: shadow.location,
                                block_revision,
                            }
                        } else {
                            let save_generation =
                                allocate_persistence_generation(&mut next_save_generation)?;
                            let key =
                                SaveKey::new(shadow.location.position, shadow.location.lod_index);
                            let target = match self.save_journal.get(&key) {
                                None => {
                                    vacant_save_count = vacant_save_count
                                        .checked_add(1)
                                        .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                                    PreparedJournalTarget::Vacant
                                }
                                Some(entry)
                                    if entry.active.is_none()
                                        && entry.written_unflushed.is_none() =>
                                {
                                    PreparedJournalTarget::ActiveIdle
                                }
                                Some(_) => {
                                    queued_save_keys.push(key);
                                    PreparedJournalTarget::Queued
                                }
                            };
                            PreparedRemovedOwnerRoute::DirtyPending(PreparedDirtyPending {
                                location: shadow.location,
                                block_revision,
                                save_generation,
                                target,
                            })
                        }
                    };
                    removed_routes.push(route);
                    next_stats.blocks_unloaded = next_stats
                        .blocks_unloaded
                        .checked_add(1)
                        .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
                    events_to_append.push(VoxelTerrainEvent::DataBlockUnloaded(shadow.location));
                } else {
                    let final_viewers = shadow.next.checked_total(shadow.location)?;
                    if final_viewers != shadow.snapshot.viewers() {
                        data_operations.push(
                            SharedVoxelDataTransactionOperation::SetViewersExact {
                                location: shadow.location,
                                final_viewers,
                            },
                        );
                    }
                    data_residency_diffs.push(PreparedDataResidencyDiff {
                        location: shadow.location,
                        expected: shadow.expected_loaded,
                        action: PreparedMapAction::Replace(shadow.next),
                    });
                }
                continue;
            }

            next_pending_load[lod].retain(|position| *position != shadow.location.position);
            if shadow.next.is_empty() {
                if let Some(expected) = shadow.expected_loading.as_ref() {
                    loading_diffs.push(PreparedLoadingEntryDiff {
                        location: shadow.location,
                        expected_generation: Some(expected.request_generation),
                        action: PreparedMapAction::Remove,
                    });
                }
                continue;
            }

            let mut entry = shadow
                .expected_loading
                .clone()
                .unwrap_or(LoadingBlockEntry {
                    residency: shadow.next,
                    retry_count: 0,
                    request_generation: 0,
                    request_state: LoadRequestState::Queued,
                    physical_request: None,
                });
            entry.residency = shadow.next;
            let residency_grew = shadow.expected_loading.as_ref().is_none_or(|expected| {
                shadow.next.resident_viewers > expected.residency.resident_viewers
                    || shadow.next.coverage_holds > expected.residency.coverage_holds
            });
            let needs_fresh = shadow.expected_loading.is_none()
                || (matches!(
                    entry.request_state,
                    LoadRequestState::NotFound | LoadRequestState::Exhausted
                ) && residency_grew);
            let dispatch = entry.request_state == LoadRequestState::Queued || needs_fresh;
            if dispatch {
                if needs_fresh {
                    let (generation, request) = allocate_physical_request(
                        self.request_epoch,
                        &mut next_request_generation,
                    )?;
                    entry.request_generation = generation;
                    entry.physical_request = Some(request);
                    entry.retry_count = 0;
                }
                entry.request_state = LoadRequestState::InFlight;
                let request = entry
                    .physical_request
                    .as_ref()
                    .expect("queued variable load retained its validated request")
                    .clone();
                scheduled_tasks.push(ScheduledTask::new(
                    Box::new(
                        LoadBlockForTerrainTask::new(
                            shadow.location.position,
                            shadow.location.lod_index,
                            entry.request_generation,
                            self.data.clone(),
                            self.stream.clone(),
                        )
                        .with_request_control(request.tag, request.cancellation),
                    ),
                    TaskLane::Parallel,
                ));
            }
            let expected_generation = shadow
                .expected_loading
                .as_ref()
                .map(|expected| expected.request_generation);
            loading_diffs.push(PreparedLoadingEntryDiff {
                location: shadow.location,
                expected_generation,
                action: if expected_generation.is_some() {
                    PreparedMapAction::Replace(entry)
                } else {
                    PreparedMapAction::Insert(entry)
                },
            });
        }

        for shadow in mesh_shadows.values_mut() {
            let Some(entry) = shadow.next.as_mut() else {
                continue;
            };
            let lod = usize::from(shadow.location.lod_index);
            next_pending_mesh[lod]
                .retain(|position| *position != shadow.location.position_in_blocks);
            if entry.resident_refcount() == 0 {
                entry.is_in_update_list = false;
                entry.physical_request = None;
                shadow.next = None;
                continue;
            }
            if !entry.needs_visual() && !entry.needs_collision() {
                entry.is_in_update_list = false;
                entry.physical_request = None;
                continue;
            }
            let required = MeshBuildFeatures {
                visuals: entry.needs_visual(),
                collisions: entry.needs_collision(),
                variable_lod: true,
            };
            let accepted =
                entry.accepted_upload.is_some() && entry.applied_features.contains(required);
            if accepted {
                entry.is_in_update_list = false;
                entry.physical_request = None;
                continue;
            }
            if entry.requested_revision.is_none() || !entry.requested_features.contains(required) {
                let revision = next_mesh_revision;
                next_mesh_revision = revision
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::MeshRevisionOverflow)?;
                entry.requested_revision = Some(revision);
                entry.requested_features = required;
                entry.terminal_retry_count = 0;
                entry.physical_request = None;
                entry.is_in_update_list = false;
            }
            let halo = clipped_meshing_data_box(
                shadow.location,
                self.data_block_size(),
                1,
                self.data.bounds(),
            )
            .map_err(VoxelTerrainRuntimeError::LodMath)?;
            let data_ready = halo.iter_cells_zxy().all(|position| {
                data_shadows
                    .get(&canonical_block_location_key(BlockLocation {
                        position,
                        lod_index: shadow.location.lod_index,
                    }))
                    .is_some_and(|data| data.snapshot.is_present() && !data.next.is_empty())
            });
            let valid_in_flight = !entry.is_in_update_list
                && entry.physical_request.as_ref().is_some_and(|request| {
                    request.tag == TaskRequestTag::new(self.request_epoch, entry.request_generation)
                        && !request.cancellation.is_cancelled()
                });
            if valid_in_flight {
                continue;
            }
            if entry.physical_request.is_none() {
                let (generation, request) =
                    allocate_physical_request(self.request_epoch, &mut next_request_generation)?;
                entry.request_generation = generation;
                entry.physical_request = Some(request);
            }
            if !data_ready {
                entry.is_in_update_list = true;
                next_pending_mesh[lod].push(shadow.location.position_in_blocks);
                continue;
            }
            entry.is_in_update_list = false;
            let request = entry
                .physical_request
                .as_ref()
                .expect("prepared variable mesh owns one exact request")
                .clone();
            scheduled_tasks.push(ScheduledTask::new(
                Box::new(
                    MeshBlockTask::new(MeshBlockTaskParams {
                        key: MeshBlockKey {
                            location: shadow.location,
                            revision: entry
                                .requested_revision
                                .expect("prepared variable mesh owns one revision"),
                        },
                        data: self.data.clone(),
                        meshing_dependency: self.meshing_dependency.clone(),
                        collision_hint: entry.needs_collision(),
                        lod_hint: true,
                        mesh_arrays_pool: Some(self.mesh_arrays_pool.clone()),
                    })
                    .with_request_control(request.tag, request.cancellation),
                ),
                TaskLane::Parallel,
            ));
        }

        let mut mesh_diffs = Vec::new();
        mesh_diffs
            .try_reserve_exact(mesh_shadows.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut mesh_observations = Vec::new();
        mesh_observations
            .try_reserve_exact(mesh_shadows.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (_, shadow) in mesh_shadows {
            let expected_revision = shadow
                .expected
                .as_ref()
                .and_then(|entry| entry.requested_revision);
            let had_expected = shadow.expected.is_some();
            let emit_exit = shadow
                .expected
                .as_ref()
                .is_some_and(|entry| entry.accepted_upload.is_some());
            mesh_observations.push(VariableMeshPhysicalObservation {
                location: shadow.location,
                expected: shadow.expected,
            });
            let action = match (had_expected, shadow.next) {
                (false, Some(entry)) if entry.resident_refcount() != 0 => {
                    PreparedMapAction::Insert(entry)
                }
                (true, Some(entry)) if entry.resident_refcount() != 0 => {
                    PreparedMapAction::Replace(entry)
                }
                (true, _) => {
                    if emit_exit {
                        events_to_append.push(VoxelTerrainEvent::MeshBlockExited(shadow.location));
                    }
                    PreparedMapAction::Remove
                }
                (false, _) => continue,
            };
            mesh_diffs.push(PreparedMeshEntryDiff {
                location: shadow.location,
                expected_revision,
                action,
            });
        }

        let mut data_snapshots = Vec::new();
        data_snapshots
            .try_reserve_exact(data_shadows.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut data_observations = Vec::new();
        data_observations
            .try_reserve_exact(data_shadows.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (_, shadow) in data_shadows {
            data_snapshots.push(shadow.snapshot);
            data_observations.push(if let Some(expected) = shadow.expected_loaded {
                VariableDataPhysicalObservation::Loaded {
                    location: shadow.location,
                    expected,
                }
            } else if let Some(expected) = shadow.expected_loading {
                VariableDataPhysicalObservation::Loading {
                    location: shadow.location,
                    expected,
                }
            } else {
                VariableDataPhysicalObservation::Missing {
                    location: shadow.location,
                }
            });
        }

        #[cfg(test)]
        if vacant_save_count != 0 {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::SaveJournal)?;
        }
        self.save_journal
            .try_reserve(vacant_save_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        #[cfg(test)]
        if !queued_save_keys.is_empty() {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::SaveJournalQueue)?;
        }
        for key in queued_save_keys {
            self.save_journal
                .get_mut(&key)
                .unwrap_or_else(|| unreachable!("queued save target remains live"))
                .queued_newer
                .try_reserve(1)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        }
        #[cfg(test)]
        if dirty_retained_count != 0 {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::DirtyRetention)?;
        }
        self.retained_save_admission_failures
            .try_reserve(dirty_retained_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;

        let mut pending_load_queues = Vec::new();
        pending_load_queues
            .try_reserve_exact(lod_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut has_pending_load_diff = false;
        for (lod, final_values) in next_pending_load.into_iter().enumerate() {
            if final_values != self.blocks_pending_load[lod] {
                if !has_pending_load_diff {
                    has_pending_load_diff = true;
                }
                pending_load_queues.push(PreparedQueueDiff {
                    lod_index: u8::try_from(lod).map_err(|_| {
                        VoxelTerrainRuntimeError::LodMath(LodMathError::InvalidLodCount)
                    })?,
                    final_values,
                });
            }
        }
        #[cfg(test)]
        if has_pending_load_diff {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::PendingLoadQueue)?;
        }
        let mut pending_mesh_queues = Vec::new();
        pending_mesh_queues
            .try_reserve_exact(lod_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut has_pending_mesh_diff = false;
        for (lod, final_values) in next_pending_mesh.into_iter().enumerate() {
            if final_values != self.blocks_pending_update[lod] {
                if !has_pending_mesh_diff {
                    has_pending_mesh_diff = true;
                }
                pending_mesh_queues.push(PreparedQueueDiff {
                    lod_index: u8::try_from(lod).map_err(|_| {
                        VoxelTerrainRuntimeError::LodMath(LodMathError::InvalidLodCount)
                    })?,
                    final_values,
                });
            }
        }
        #[cfg(test)]
        if has_pending_mesh_diff {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::PendingMeshQueue)?;
        }

        let mut next_deferred_save_dispatch_keys = Vec::new();
        next_deferred_save_dispatch_keys
            .try_reserve_exact(self.deferred_save_dispatch_keys.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        next_deferred_save_dispatch_keys.extend_from_slice(&self.deferred_save_dispatch_keys);

        let persistence = PreparedOpaquePersistenceState {
            removed_owner_routes: removed_routes,
            next_deferred_save_dispatch_keys,
            next_deferred_checkpoint_dispatch: self.deferred_checkpoint_dispatch,
            next_automatic_save_checkpoint_blocked: self.automatic_save_checkpoint_blocked,
            next_save_dispatch_error: self.save_dispatch_error.clone(),
            next_save_generation,
            next_save_checkpoint_generation: self.next_save_checkpoint_generation,
            next_persistence_attempt_ordinal: self.next_persistence_attempt_ordinal,
            next_force_checkpoint_requested: self.force_checkpoint_requested,
            #[cfg(test)]
            next_panic_flush_before_io_attempts_for_test: self
                .panic_next_flush_before_io_attempts_for_test,
            #[cfg(test)]
            next_panic_flush_after_ack_for_test: self.panic_next_flush_after_ack_for_test,
            ..PreparedOpaquePersistenceState::default()
        };

        Ok((
            PreparedVariablePhysicalSlice {
                mesh_diffs,
                data_residency_diffs,
                loading_diffs,
                pending_load_queues,
                pending_mesh_queues,
                data_operations,
                data_snapshots,
                scheduled_tasks,
                persistence,
                events_to_append,
                next_request_generation,
                next_mesh_revision,
                next_render_topology_revision,
                next_stats,
                observations: PreparedVariablePhysicalObservations {
                    mesh: mesh_observations,
                    data: data_observations,
                },
            },
            hold_phases,
        ))
    }

    #[cfg(test)]
    fn try_fixed_viewer_transaction_for_test(
        &mut self,
        viewers: &[ViewerUpdate],
    ) -> Result<(), VoxelTerrainRuntimeError> {
        let viewers = normalize_and_validate_viewer_updates(viewers)
            .map_err(VoxelTerrainRuntimeError::ViewerInput)?;
        self.prepare_fixed_viewer_transaction(&viewers, true, true, true)
    }

    #[cfg(test)]
    fn try_variable_mode_transaction_for_test(
        &mut self,
        paired_viewers: &[PairedViewer],
        viewers: &[ClipboxViewerUpdate],
    ) -> Result<(), VariableModeTestError> {
        self.try_variable_mode_transaction_with_coverage_inputs_for_test(
            paired_viewers,
            viewers,
            &[],
        )
    }

    #[cfg(test)]
    fn try_variable_mode_transaction_with_coverage_inputs_for_test(
        &mut self,
        paired_viewers: &[PairedViewer],
        viewers: &[ClipboxViewerUpdate],
        extra_coverage_inputs: &[CoverageInput],
    ) -> Result<(), VariableModeTestError> {
        debug_assert!(self.variable_lod.is_some());
        self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::PairedViewers)?;
        let mut next_paired_viewers = Vec::new();
        next_paired_viewers
            .try_reserve_exact(paired_viewers.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        next_paired_viewers.extend_from_slice(paired_viewers);

        let coordinator_update = self
            .variable_lod
            .as_ref()
            .unwrap_or_else(|| unreachable!("test seam requires variable LOD"))
            .coordinator
            .prepare_update(viewers)
            .map_err(VoxelTerrainRuntimeError::Coordinator)?;
        self.fixed_capacity_checkpoint_for_test(
            FixedCapacityDestination::VariableCoordinatorPreparedState,
        )?;

        let mesh_change_count = coordinator_update
            .delta()
            .changes
            .iter()
            .filter(|change| change.key.kind == ResidentBlockKind::Mesh)
            .count();
        let coverage_input_count = mesh_change_count
            .checked_add(extra_coverage_inputs.len())
            .ok_or(VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let coordinator_update = coordinator_update
            .validate_for(
                &self
                    .variable_lod
                    .as_ref()
                    .unwrap_or_else(|| unreachable!("test seam requires variable LOD"))
                    .coordinator,
            )
            .map_err(VoxelTerrainRuntimeError::from)?;
        let data_preview = self.data.begin_transaction_preview();
        let mut physical_slice = None;
        let needs_coverage_preview =
            coverage_input_count != 0 || !coordinator_update.delta().changes.is_empty();
        let coverage_publication = if !needs_coverage_preview {
            None
        } else {
            self.fixed_capacity_checkpoint_for_test(
                FixedCapacityDestination::VariableCoverageInputs,
            )?;
            let mut inputs = Vec::new();
            inputs
                .try_reserve_exact(coverage_input_count)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            inputs.extend(
                coordinator_update
                    .delta()
                    .changes
                    .iter()
                    .filter_map(|change| {
                        (change.key.kind == ResidentBlockKind::Mesh).then_some(
                            CoverageInput::SetDemand {
                                location: change.key.location,
                                counts: change.new_counts,
                            },
                        )
                    }),
            );
            inputs.extend_from_slice(extra_coverage_inputs);
            inputs.sort_unstable_by_key(|input| match input {
                CoverageInput::SetDemand { location, .. }
                | CoverageInput::Accept { location, .. }
                | CoverageInput::Evict { location } => canonical_mesh_location_key(*location),
                CoverageInput::SetJoinTargetState { id, .. } => {
                    canonical_mesh_location_key(id.join_parent)
                }
            });

            self.fixed_capacity_checkpoint_for_test(
                FixedCapacityDestination::VariableCoveragePreview,
            )?;
            let preview = self
                .variable_lod
                .as_ref()
                .unwrap_or_else(|| unreachable!("test seam requires variable LOD"))
                .coverage
                .preview_reconcile(&inputs)
                .map_err(VoxelTerrainRuntimeError::Coverage)?;
            self.fixed_capacity_checkpoint_for_test(
                FixedCapacityDestination::VariableCoverageHoldResolution,
            )?;
            let hold_resolution = {
                let runtime = self
                    .variable_lod
                    .as_ref()
                    .unwrap_or_else(|| unreachable!("test seam requires variable LOD"));
                runtime.coverage_holds.prepare_resolution(
                    &preview.result().hold_deltas,
                    runtime.settings,
                    self.data.bounds(),
                )
            }
            .map_err(VoxelTerrainRuntimeError::from)?;
            let preview = preview
                .validate_for(
                    &self
                        .variable_lod
                        .as_ref()
                        .unwrap_or_else(|| unreachable!("test seam requires variable LOD"))
                        .coverage,
                )
                .map_err(VoxelTerrainRuntimeError::from)?;
            let (mut prepared_physical, hold_phases) = self.prepare_variable_physical_slice(
                &coordinator_update,
                &preview,
                &inputs,
                hold_resolution,
                &data_preview,
            )?;
            let observations = std::mem::take(&mut prepared_physical.observations);
            physical_slice = Some(prepared_physical);
            Some(PreparedVariableCoveragePublication {
                preview: Some(preview),
                hold_phases: Some(hold_phases),
                physical_observations: observations,
            })
        };

        let PreparedVariablePhysicalSlice {
            mesh_diffs,
            data_residency_diffs,
            loading_diffs,
            pending_load_queues,
            pending_mesh_queues,
            data_operations,
            data_snapshots,
            scheduled_tasks,
            persistence,
            events_to_append,
            next_request_generation,
            next_mesh_revision,
            next_render_topology_revision,
            next_stats,
            observations: _,
        } = physical_slice.unwrap_or_else(|| PreparedVariablePhysicalSlice {
            mesh_diffs: Vec::new(),
            data_residency_diffs: Vec::new(),
            loading_diffs: Vec::new(),
            pending_load_queues: Vec::new(),
            pending_mesh_queues: Vec::new(),
            data_operations: Vec::new(),
            data_snapshots: Vec::new(),
            scheduled_tasks: Vec::new(),
            persistence: PreparedOpaquePersistenceState {
                next_deferred_save_dispatch_keys: self.deferred_save_dispatch_keys.clone(),
                next_deferred_checkpoint_dispatch: self.deferred_checkpoint_dispatch,
                next_automatic_save_checkpoint_blocked: self.automatic_save_checkpoint_blocked,
                next_save_dispatch_error: self.save_dispatch_error.clone(),
                next_save_generation: self.next_save_generation,
                next_save_checkpoint_generation: self.next_save_checkpoint_generation,
                next_persistence_attempt_ordinal: self.next_persistence_attempt_ordinal,
                next_force_checkpoint_requested: self.force_checkpoint_requested,
                #[cfg(test)]
                next_panic_flush_before_io_attempts_for_test: self
                    .panic_next_flush_before_io_attempts_for_test,
                #[cfg(test)]
                next_panic_flush_after_ack_for_test: self.panic_next_flush_after_ack_for_test,
                ..PreparedOpaquePersistenceState::default()
            },
            events_to_append: Vec::new(),
            next_request_generation: self.next_request_generation,
            next_mesh_revision: self.next_mesh_revision,
            next_render_topology_revision: self.next_render_topology_revision,
            next_stats: self.stats,
            observations: PreparedVariablePhysicalObservations::default(),
        });

        self.try_reserve_prepared_runtime_publication(
            &mesh_diffs,
            &data_residency_diffs,
            &loading_diffs,
            &pending_load_queues,
            &pending_mesh_queues,
        )?;
        if !events_to_append.is_empty() {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::EventOutbox)?;
        }
        self.event_outbox
            .try_reserve(events_to_append.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut retirement = RetirementBag::default();
        if !mesh_diffs.is_empty() {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::Retirement)?;
        }
        retirement
            .mesh_entries
            .try_reserve_exact(mesh_diffs.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .loading_entries
            .try_reserve_exact(loading_diffs.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .data_blocks
            .try_reserve_exact(persistence.removed_owner_routes.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .save_journal_entries
            .try_reserve_exact(persistence.removed_owner_routes.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .runtime_errors
            .try_reserve_exact(1)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;

        let scheduled_task_count = scheduled_tasks.len();
        #[cfg(test)]
        if scheduled_task_count != 0 {
            // Mirror the fixed paging path: the `PreparedTaskBatch` capacity
            // failpoint only applies when a batch is actually being prepared,
            // so cache-only/coordinator-only variable transactions that stage
            // zero tasks do not trip the injected failpoint spuriously.
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::PreparedTaskBatch)?;
        }
        let mut prepared_task_batch = self
            .task_runner
            .try_prepare_enqueue(scheduled_task_count)
            .map_err(CompletionDrainError::from)
            .map_err(VoxelTerrainRuntimeError::from)?;
        for task in scheduled_tasks {
            prepared_task_batch
                .push_reserved(task)
                .unwrap_or_else(|_| unreachable!("exact variable task count was reserved"));
        }
        let prepared_task_batch = prepared_task_batch
            .try_into_filled()
            .unwrap_or_else(|_| unreachable!("variable task batch is exactly filled"));
        let data_tx = data_preview
            .prepare_transaction(data_operations, &data_snapshots)
            .map_err(|error| VoxelTerrainRuntimeError::DataMutation(error.into_parts().0))?;
        let draft = TerrainTransactionDraft {
            paired_viewer_publication: Some(next_paired_viewers),
            mode: PreparedTerrainMode::Variable(Box::new(VariableLodTransactionDraft {
                coordinator_update: Some(coordinator_update),
                coverage_publication,
            })),
            next_request_generation,
            next_mesh_revision,
            next_render_topology_revision,
            next_stats,
            data_tx,
            mesh_diffs,
            data_residency_diffs,
            loading_diffs,
            pending_load_queues,
            pending_mesh_queues,
            persistence,
            completion_prefix: PreparedCompletionPrefix::default(),
            direct_mesh_plans: Vec::new(),
            scheduled_task_count,
            prepared_task_batch,
            events_to_append,
            retirement,
        };
        if std::mem::take(&mut self.fixed_after_prepare_settings_conflict_for_test) {
            self.data.increment_test_settings_revision();
        }
        if std::mem::take(&mut self.variable_after_prepare_stale_coverage_for_test) {
            let input = CoverageInput::SetDemand {
                location: MeshBlockLocation::new(Vector3i::new(7, 0, 0), 2),
                counts: DemandCounts {
                    resident: 1,
                    visuals: 1,
                    ..DemandCounts::default()
                },
            };
            let runtime = self
                .variable_lod
                .as_mut()
                .unwrap_or_else(|| unreachable!("test seam requires variable LOD"));
            let preview = runtime
                .coverage
                .preview_reconcile(&[input])
                .unwrap_or_else(|error| unreachable!("valid test interference: {error:?}"));
            runtime
                .coverage
                .apply_preview(preview)
                .unwrap_or_else(|error| unreachable!("valid test interference: {error:?}"));
        }
        if std::mem::take(&mut self.variable_after_prepare_stale_coordinator_for_test) {
            let interference = ClipboxViewerUpdate {
                id: ViewerId::MAX,
                position_voxels: Vector3i::new(256, 0, 0),
                view_distance_voxels: Vector3i::splat(16),
                demand: MeshDemand {
                    visuals: true,
                    collisions: false,
                },
            };
            self.variable_lod
                .as_mut()
                .unwrap_or_else(|| unreachable!("test seam requires variable LOD"))
                .coordinator
                .update_viewers(&[interference])
                .unwrap_or_else(|error| unreachable!("valid test interference: {error:?}"));
        }
        self.commit_terrain_draft_no_fail(draft)
            .map_err(VariableModeTestError::from)
    }

    /// Test-only switch: keep `try_process` on the production planner path.
    /// The constructor already enables this; the helper exists so acceptance
    /// tests can name the route explicitly.
    #[doc(hidden)]
    pub fn use_variable_planner_path_for_test(&mut self) {
        self.variable_use_planner_path = true;
    }

    /// Test-only switch: force `try_process` for `lod_count > 1` back onto the
    /// legacy three-stage route. Used by parity/legacy-behavior tests that pin
    /// the pre-cutover route's observable output.
    #[doc(hidden)]
    pub fn use_legacy_variable_path_for_test(&mut self) {
        self.variable_use_planner_path = false;
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn install_fixed_commit_pause_for_test(
        &mut self,
        phase: FixedCommitPausePhase,
    ) -> FixedCommitPauseHandle {
        let handle = FixedCommitPauseHandle {
            state: Arc::new((
                Mutex::new(FixedCommitPauseState {
                    target: phase,
                    reached: false,
                    released: false,
                }),
                Condvar::new(),
            )),
            commit_marker: Arc::new(AtomicBool::new(false)),
        };
        self.fixed_commit_pause_for_test = Some(handle.clone());
        handle
    }

    #[cfg(test)]
    pub(crate) fn fail_fixed_capacity_for_test(
        &mut self,
        destination: FixedCapacityDestination,
        occurrence: usize,
    ) {
        assert!(
            occurrence != 0,
            "capacity failpoint occurrence is one-based"
        );
        self.fixed_capacity_failpoint_for_test = Some(FixedCapacityFailpoint {
            destination,
            remaining_occurrences: occurrence,
        });
        self.last_fixed_capacity_failure_for_test = None;
    }

    #[cfg(test)]
    pub(crate) fn last_fixed_capacity_failure_for_test(&self) -> Option<FixedCapacityDestination> {
        self.last_fixed_capacity_failure_for_test
    }

    #[cfg(test)]
    fn clear_fixed_capacity_failpoint_for_test(&mut self) {
        self.fixed_capacity_failpoint_for_test = None;
        self.last_fixed_capacity_failure_for_test = None;
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn set_fixed_task_count_bias_for_test(&mut self, bias: usize) {
        self.fixed_task_count_bias_for_test = bias;
    }

    #[cfg(test)]
    pub(crate) fn install_fixed_dirty_owner_probe_for_test(
        &mut self,
        location: BlockLocation,
    ) -> Arc<AtomicUsize> {
        let observed = Arc::new(AtomicUsize::new(0));
        self.fixed_dirty_owner_probe_for_test = Some((location, Arc::clone(&observed)));
        observed
    }

    #[cfg(test)]
    fn install_fixed_stream_error_retirement_probe_for_test(
        &mut self,
        error_heap_ptr: usize,
    ) -> Arc<AtomicBool> {
        let dropped = Arc::new(AtomicBool::new(false));
        self.fixed_stream_error_retirement_probe_for_test =
            Some((error_heap_ptr, Arc::clone(&dropped)));
        dropped
    }

    #[cfg(test)]
    fn record_fixed_dirty_owner_for_test(&self, location: BlockLocation, payload: &VoxelBuffer) {
        let Some((expected, observed)) = &self.fixed_dirty_owner_probe_for_test else {
            return;
        };
        if *expected == location {
            observed.store(payload.channel_bytes(0).as_ptr() as usize, Ordering::SeqCst);
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn fixed_capacity_checkpoint_for_test(
        &mut self,
        destination: FixedCapacityDestination,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        let Some(failpoint) = self.fixed_capacity_failpoint_for_test.as_mut() else {
            return Ok(());
        };
        if failpoint.destination != destination {
            return Ok(());
        }
        failpoint.remaining_occurrences = failpoint
            .remaining_occurrences
            .checked_sub(1)
            .expect("armed capacity failpoint has a positive occurrence");
        if failpoint.remaining_occurrences != 0 {
            return Ok(());
        }
        self.fixed_capacity_failpoint_for_test = None;
        self.last_fixed_capacity_failure_for_test = Some(destination);
        Err(VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)
    }

    fn try_run_fixed_transaction(
        &mut self,
        viewers: &[ViewerUpdate],
        replace_viewers: bool,
        dispatch_tasks: bool,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        self.try_run_fixed_transaction_internal(
            viewers,
            replace_viewers,
            dispatch_tasks,
            false,
            false,
        )
    }

    fn try_run_fixed_transaction_capturing_resident(
        &mut self,
        viewers: &[ViewerUpdate],
        replace_viewers: bool,
        dispatch_tasks: bool,
        shutdown_owned_capture: bool,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        self.try_run_fixed_transaction_internal(
            viewers,
            replace_viewers,
            dispatch_tasks,
            true,
            shutdown_owned_capture,
        )
    }

    fn try_run_fixed_transaction_internal(
        &mut self,
        viewers: &[ViewerUpdate],
        replace_viewers: bool,
        dispatch_tasks: bool,
        capture_resident_dirty: bool,
        shutdown_owned_capture: bool,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        self.task_runner
            .try_drain_completed_into(&mut self.raw_completion_inbox)
            .map_err(CompletionDrainError::from)?;
        self.try_normalize_raw_completions()
            .map_err(VoxelTerrainRuntimeError::from)?;
        if !capture_resident_dirty {
            return self.prepare_fixed_viewer_transaction(
                viewers,
                replace_viewers,
                dispatch_tasks,
                true,
            );
        }
        self.prepare_fixed_viewer_transaction_with_checkpoint_and_admission(
            viewers,
            replace_viewers,
            dispatch_tasks,
            true,
            None,
            None,
            None,
            capture_resident_dirty,
            shutdown_owned_capture,
        )
    }

    /// Production Variable LOD transaction: routes the multi-LOD viewer update
    /// through the `prepare_variable_physical_slice` planner instead of the
    /// legacy three-stage multi-LOD path. Mirrors the fixed-LOD
    /// `try_run_fixed_transaction` shape (drain completions first, run one
    /// prepared transaction, then the caller drains the event outbox).
    ///
    /// Drain ordering (Phase B Section 4): the raw completion inbox is drained
    /// and normalized, then `legacy_variable_apply_durable_fifo` is applied
    /// BEFORE the planner shadows the live maps (so completed loads/meshes are
    /// resident when the planner observes them). Direct-FIFO and save/checkpoint
    /// dispatch are deferred to AFTER `commit_terrain_draft_no_fail` so they
    /// cannot race the publication fence.
    fn try_run_variable_transaction(
        &mut self,
        viewers: &[ViewerUpdate],
        dispatch_tasks: bool,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        debug_assert!(self.lod_count > 1);
        debug_assert!(
            self.variable_lod.is_some(),
            "variable planner requires variable LOD"
        );

        // Drain raw completions and normalize them into the durable inbox,
        // exactly like the fixed path.
        self.task_runner
            .try_drain_completed_into(&mut self.raw_completion_inbox)
            .map_err(CompletionDrainError::from)?;
        self.try_normalize_raw_completions()
            .map_err(VoxelTerrainRuntimeError::from)?;
        // Apply the durable inbox to the live maps BEFORE the planner shadows
        // them. This is the load-bearing drain-ordering step: it makes every
        // completion that has arrived by this tick visible to the planner.
        self.legacy_variable_apply_durable_fifo()?;

        // Build the paired-viewer publication from the incoming viewer updates
        // (Section 2). The coordinator/coverage model owns residency; the
        // paired viewers are kept consistent for telemetry/shutdown bookkeeping.
        let mut next_paired_viewers = Vec::new();
        next_paired_viewers
            .try_reserve_exact(viewers.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut clipbox_viewers = Vec::new();
        clipbox_viewers
            .try_reserve_exact(viewers.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for update in viewers {
            let horizontal = update
                .horizontal_view_distance_voxels
                .min(self.max_view_distance_voxels);
            let vertical = update
                .vertical_view_distance_voxels
                .min(self.max_view_distance_voxels);
            let prev_state = self
                .paired_viewers
                .iter()
                .find(|paired| paired.id == update.id)
                .map_or(ViewerState::default(), |paired| paired.state.clone());
            next_paired_viewers.push(PairedViewer {
                id: update.id,
                state: ViewerState {
                    local_position_voxels: update.world_position_voxels,
                    horizontal_view_distance_voxels: horizontal,
                    vertical_view_distance_voxels: vertical,
                    demand: update.demand,
                    ..ViewerState::default()
                },
                prev_state,
            });
            clipbox_viewers.push(ClipboxViewerUpdate {
                id: update.id,
                position_voxels: update.world_position_voxels,
                view_distance_voxels: Vector3i::new(horizontal, vertical, horizontal),
                demand: update.demand,
            });
        }

        let coordinator_update = self
            .variable_lod
            .as_ref()
            .unwrap_or_else(|| unreachable!("variable planner requires variable LOD"))
            .coordinator
            .prepare_update(&clipbox_viewers)
            .map_err(VoxelTerrainRuntimeError::Coordinator)?;
        let mesh_change_count = coordinator_update
            .delta()
            .changes
            .iter()
            .filter(|change| change.key.kind == ResidentBlockKind::Mesh)
            .count();
        let coordinator_update = coordinator_update
            .validate_for(
                &self
                    .variable_lod
                    .as_ref()
                    .unwrap_or_else(|| unreachable!("variable planner requires variable LOD"))
                    .coordinator,
            )
            .map_err(VoxelTerrainRuntimeError::from)?;

        let data_preview = self.data.begin_transaction_preview();
        let needs_coverage_preview =
            mesh_change_count != 0 || !coordinator_update.delta().changes.is_empty();
        let mut physical_slice = None;
        let coverage_publication = if !needs_coverage_preview {
            None
        } else {
            let mut inputs = Vec::new();
            inputs
                .try_reserve_exact(mesh_change_count)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            inputs.extend(
                coordinator_update
                    .delta()
                    .changes
                    .iter()
                    .filter_map(|change| {
                        (change.key.kind == ResidentBlockKind::Mesh).then_some(
                            CoverageInput::SetDemand {
                                location: change.key.location,
                                counts: change.new_counts,
                            },
                        )
                    }),
            );
            inputs.sort_unstable_by_key(|input| match input {
                CoverageInput::SetDemand { location, .. }
                | CoverageInput::Accept { location, .. }
                | CoverageInput::Evict { location } => canonical_mesh_location_key(*location),
                CoverageInput::SetJoinTargetState { id, .. } => {
                    canonical_mesh_location_key(id.join_parent)
                }
            });
            let preview = self
                .variable_lod
                .as_ref()
                .unwrap_or_else(|| unreachable!("variable planner requires variable LOD"))
                .coverage
                .preview_reconcile(&inputs)
                .map_err(VoxelTerrainRuntimeError::Coverage)?;
            let hold_resolution = {
                let runtime = self
                    .variable_lod
                    .as_ref()
                    .unwrap_or_else(|| unreachable!("variable planner requires variable LOD"));
                runtime.coverage_holds.prepare_resolution(
                    &preview.result().hold_deltas,
                    runtime.settings,
                    self.data.bounds(),
                )
            }
            .map_err(VoxelTerrainRuntimeError::from)?;
            let preview = preview
                .validate_for(
                    &self
                        .variable_lod
                        .as_ref()
                        .unwrap_or_else(|| unreachable!("variable planner requires variable LOD"))
                        .coverage,
                )
                .map_err(VoxelTerrainRuntimeError::from)?;
            let (mut prepared_physical, hold_phases) = self.prepare_variable_physical_slice(
                &coordinator_update,
                &preview,
                &inputs,
                hold_resolution,
                &data_preview,
            )?;
            let observations = std::mem::take(&mut prepared_physical.observations);
            physical_slice = Some(prepared_physical);
            Some(PreparedVariableCoveragePublication {
                preview: Some(preview),
                hold_phases: Some(hold_phases),
                physical_observations: observations,
            })
        };

        let PreparedVariablePhysicalSlice {
            mesh_diffs,
            data_residency_diffs,
            loading_diffs,
            pending_load_queues,
            pending_mesh_queues,
            data_operations,
            data_snapshots,
            mut scheduled_tasks,
            persistence,
            events_to_append,
            next_request_generation,
            next_mesh_revision,
            next_render_topology_revision,
            next_stats,
            observations: _,
        } = physical_slice.unwrap_or_else(|| PreparedVariablePhysicalSlice {
            mesh_diffs: Vec::new(),
            data_residency_diffs: Vec::new(),
            loading_diffs: Vec::new(),
            pending_load_queues: Vec::new(),
            pending_mesh_queues: Vec::new(),
            data_operations: Vec::new(),
            data_snapshots: Vec::new(),
            scheduled_tasks: Vec::new(),
            persistence: PreparedOpaquePersistenceState {
                next_deferred_save_dispatch_keys: self.deferred_save_dispatch_keys.clone(),
                next_deferred_checkpoint_dispatch: self.deferred_checkpoint_dispatch,
                next_automatic_save_checkpoint_blocked: self.automatic_save_checkpoint_blocked,
                next_save_dispatch_error: self.save_dispatch_error.clone(),
                next_save_generation: self.next_save_generation,
                next_save_checkpoint_generation: self.next_save_checkpoint_generation,
                next_persistence_attempt_ordinal: self.next_persistence_attempt_ordinal,
                next_force_checkpoint_requested: self.force_checkpoint_requested,
                #[cfg(test)]
                next_panic_flush_before_io_attempts_for_test: self
                    .panic_next_flush_before_io_attempts_for_test,
                #[cfg(test)]
                next_panic_flush_after_ack_for_test: self.panic_next_flush_after_ack_for_test,
                ..PreparedOpaquePersistenceState::default()
            },
            events_to_append: Vec::new(),
            next_request_generation: self.next_request_generation,
            next_mesh_revision: self.next_mesh_revision,
            next_render_topology_revision: self.next_render_topology_revision,
            next_stats: self.stats,
            observations: PreparedVariablePhysicalObservations::default(),
        });

        // Honor the dispatch gate: when tasks should not be dispatched, drop
        // the prepared tasks before reservation (mirrors the fixed path's
        // `dispatch_tasks` semantics).
        if !dispatch_tasks {
            scheduled_tasks.clear();
        }

        self.try_reserve_prepared_runtime_publication(
            &mesh_diffs,
            &data_residency_diffs,
            &loading_diffs,
            &pending_load_queues,
            &pending_mesh_queues,
        )?;
        if !events_to_append.is_empty() {
            self.event_outbox
                .try_reserve(events_to_append.len())
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        }
        let mut retirement = RetirementBag::default();
        retirement
            .mesh_entries
            .try_reserve_exact(mesh_diffs.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .loading_entries
            .try_reserve_exact(loading_diffs.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .data_blocks
            .try_reserve_exact(persistence.removed_owner_routes.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .save_journal_entries
            .try_reserve_exact(persistence.removed_owner_routes.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        retirement
            .runtime_errors
            .try_reserve_exact(1)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;

        let scheduled_task_count = scheduled_tasks.len();
        let mut prepared_task_batch = self
            .task_runner
            .try_prepare_enqueue(scheduled_task_count)
            .map_err(CompletionDrainError::from)
            .map_err(VoxelTerrainRuntimeError::from)?;
        for task in scheduled_tasks {
            prepared_task_batch
                .push_reserved(task)
                .unwrap_or_else(|_| unreachable!("exact variable task count was reserved"));
        }
        let prepared_task_batch = prepared_task_batch
            .try_into_filled()
            .unwrap_or_else(|_| unreachable!("variable task batch is exactly filled"));
        let data_tx = data_preview
            .prepare_transaction(data_operations, &data_snapshots)
            .map_err(|error| VoxelTerrainRuntimeError::DataMutation(error.into_parts().0))?;
        let draft = TerrainTransactionDraft {
            paired_viewer_publication: Some(next_paired_viewers),
            mode: PreparedTerrainMode::Variable(Box::new(VariableLodTransactionDraft {
                coordinator_update: Some(coordinator_update),
                coverage_publication,
            })),
            next_request_generation,
            next_mesh_revision,
            next_render_topology_revision,
            next_stats,
            data_tx,
            mesh_diffs,
            data_residency_diffs,
            loading_diffs,
            pending_load_queues,
            pending_mesh_queues,
            persistence,
            completion_prefix: PreparedCompletionPrefix::default(),
            direct_mesh_plans: Vec::new(),
            scheduled_task_count,
            prepared_task_batch,
            events_to_append,
            retirement,
        };
        self.commit_terrain_draft_no_fail(draft)?;

        // AFTER the publication fence: dispatch queued saves/checkpoints and
        // apply any direct mesh uploads that arrived. Deferring these keeps
        // them from racing the planner's map publication.
        let deferred_keys = std::mem::take(&mut self.deferred_save_dispatch_keys);
        self.dispatch_queued_saves_except(&deferred_keys);
        if !std::mem::take(&mut self.deferred_checkpoint_dispatch) {
            self.dispatch_pending_checkpoint();
        }
        self.legacy_variable_apply_direct_fifo()?;
        // Stationary ticks skip the coordinator slice. Halo data that became
        // resident after the last delta is dispatched here with the same
        // MeshBlockTask hints the planner uses, so remesh features cannot
        // drift from the production path.
        self.enqueue_ready_mesh_block_tasks()?;
        Ok(())
    }

    fn try_run_fixed_transaction_with_checkpoint(
        &mut self,
        checkpoint_request: FixedCheckpointRequest,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        debug_assert_eq!(self.lod_count, 1);
        self.task_runner
            .try_drain_completed_into(&mut self.raw_completion_inbox)
            .map_err(CompletionDrainError::from)?;
        self.try_normalize_raw_completions()
            .map_err(VoxelTerrainRuntimeError::from)?;
        self.prepare_fixed_viewer_transaction_with_checkpoint(
            &[],
            false,
            false,
            true,
            Some(checkpoint_request),
            None,
            None,
            false,
        )
    }

    fn try_run_fixed_transaction_with_persistence_recovery(
        &mut self,
        recovery_request: FixedPersistenceRecoveryRequest,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        debug_assert_eq!(self.lod_count, 1);
        self.task_runner
            .try_drain_completed_into(&mut self.raw_completion_inbox)
            .map_err(CompletionDrainError::from)?;
        self.try_normalize_raw_completions()
            .map_err(VoxelTerrainRuntimeError::from)?;
        self.prepare_fixed_viewer_transaction_with_checkpoint(
            &[],
            false,
            false,
            true,
            None,
            Some(recovery_request),
            None,
            false,
        )
    }

    fn try_run_fixed_transaction_with_persistence_resolution(
        &mut self,
        resolution_request: FixedPersistenceResolutionRequest,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        debug_assert_eq!(self.lod_count, 1);
        self.task_runner
            .try_drain_completed_into(&mut self.raw_completion_inbox)
            .map_err(CompletionDrainError::from)?;
        self.try_normalize_raw_completions()
            .map_err(VoxelTerrainRuntimeError::from)?;
        self.prepare_fixed_viewer_transaction_with_checkpoint(
            &[],
            false,
            false,
            true,
            None,
            None,
            Some(resolution_request),
            false,
        )
    }

    /// Atomically publishes variable-LOD resident dirty copies into the save
    /// journal while clearing the matching storage flags under one C1 fence.
    /// Explicit flush/shutdown keeps persistence dispatch paused until this
    /// returns, so no task can observe an empty prepared payload slot.
    fn try_capture_variable_resident_saves(
        &mut self,
        shutdown_owned_capture: bool,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        debug_assert!(self.lod_count > 1);
        if shutdown_owned_capture {
            let permit = self.shutdown_mutation_permit.as_ref().ok_or(
                VoxelTerrainRuntimeError::DataMutation(
                    SharedVoxelDataMutationError::MutationAdmissionClosed,
                ),
            )?;
            self.data
                .validate_shutdown_mutation_permit(permit)
                .map_err(VoxelTerrainRuntimeError::DataMutation)?;
        }
        let preview = self.data.begin_transaction_preview();
        let (operations, snapshots, payloads) = preview
            .prepare_resident_dirty_copies()
            .map_err(VoxelTerrainRuntimeError::DataMutation)?
            .into_parts();
        for snapshot in &snapshots {
            self.validate_loaded_data_residency_snapshot(snapshot)?;
        }
        let mut next_save_generation = self.next_save_generation;
        let mut routes = Vec::new();
        routes
            .try_reserve_exact(payloads.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut vacant_count = 0usize;
        for payload in payloads {
            let generation = allocate_persistence_generation(&mut next_save_generation)?;
            let key = SaveKey::new(payload.location.position, payload.location.lod_index);
            let target = match self.save_journal.get(&key) {
                None => {
                    vacant_count = vacant_count
                        .checked_add(1)
                        .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?;
                    PreparedJournalTarget::Vacant
                }
                Some(entry) if entry.active.is_none() && entry.written_unflushed.is_none() => {
                    PreparedJournalTarget::ActiveIdle
                }
                Some(_) => PreparedJournalTarget::Queued,
            };
            routes.push(PreparedResidentSaveRoute::Pending {
                route: PreparedDirtyPending {
                    location: payload.location,
                    block_revision: payload.block_revision,
                    save_generation: generation,
                    target,
                },
                payload,
            });
        }
        #[cfg(test)]
        if vacant_count != 0 {
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::SaveJournal)?;
        }
        self.save_journal
            .try_reserve(vacant_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for route in &routes {
            let PreparedResidentSaveRoute::Pending { payload, route } = route else {
                unreachable!("variable resident capture prepares only pending owners")
            };
            if !matches!(route.target, PreparedJournalTarget::Queued) {
                continue;
            }
            #[cfg(test)]
            self.fixed_capacity_checkpoint_for_test(FixedCapacityDestination::SaveJournalQueue)?;
            self.save_journal
                .get_mut(&SaveKey::new(
                    payload.location.position,
                    payload.location.lod_index,
                ))
                .expect("queued variable resident target remains live")
                .queued_newer
                .try_reserve(1)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        }
        let mut unexpected_replacements = Vec::new();
        unexpected_replacements
            .try_reserve_exact(vacant_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut transaction = preview
            .prepare_transaction(operations, &snapshots)
            .map_err(|error| VoxelTerrainRuntimeError::DataMutation(error.into_parts().0))?;
        if shutdown_owned_capture {
            let permit = self.shutdown_mutation_permit.as_ref().ok_or(
                VoxelTerrainRuntimeError::DataMutation(
                    SharedVoxelDataMutationError::MutationAdmissionClosed,
                ),
            )?;
            transaction
                .authorize_shutdown_mutation(permit)
                .map_err(VoxelTerrainRuntimeError::DataMutation)?;
        }
        #[cfg(test)]
        if let Some(position) = self.variable_after_prepare_edit_conflict_for_test.take() {
            let _ = self.data.try_edit_voxel_checked(
                0xDC,
                position,
                crate::storage::ChannelId::Type.index(),
            );
        }
        let fence = transaction
            .commit_holding_publication_fence()
            .map_err(VoxelTerrainRuntimeError::DataMutation)?;
        debug_assert!(fence.removed_blocks().is_empty());
        for route in routes {
            let PreparedResidentSaveRoute::Pending { payload, route } = route else {
                unreachable!("variable resident capture publishes only pending owners")
            };
            debug_assert_eq!(payload.location, route.location);
            debug_assert_eq!(payload.block_revision, route.block_revision);
            let key = SaveKey::new(payload.location.position, payload.location.lod_index);
            #[cfg(test)]
            self.record_fixed_dirty_owner_for_test(payload.location, &payload.voxels);
            let pending = PendingSave {
                meta: SaveAttemptMeta {
                    block_revision: route.block_revision,
                    generation: route.save_generation,
                    retry_count: 0,
                    last_error: None,
                },
                payload: payload.voxels,
            };
            match route.target {
                PreparedJournalTarget::Vacant => {
                    if let Some(replaced) = self.save_journal.insert(
                        key,
                        SaveJournalEntry {
                            written_unflushed: None,
                            active: Some(ActiveSaveAttempt::Pending(pending)),
                            queued_newer: VecDeque::new(),
                        },
                    ) {
                        unexpected_replacements.push(replaced);
                    }
                }
                PreparedJournalTarget::ActiveIdle => {
                    self.save_journal
                        .get_mut(&key)
                        .expect("prepared active-idle journal target remains live")
                        .active = Some(ActiveSaveAttempt::Pending(pending));
                }
                PreparedJournalTarget::Queued => {
                    self.save_journal
                        .get_mut(&key)
                        .expect("prepared queued journal target remains live")
                        .queued_newer
                        .push_back(pending);
                }
            }
        }
        self.next_save_generation = next_save_generation;
        if !self.save_journal.is_empty() {
            self.automatic_checkpoint_satisfied_empty_flush = false;
        }
        #[cfg(test)]
        if let Some(pause) = &self.fixed_commit_pause_for_test {
            pause.commit_marker.store(true, Ordering::SeqCst);
            pause.pause_if_target(FixedCommitPausePhase::AfterTerrainPublishBeforeFenceFinish);
        }
        let storage_outcome = fence.finish();
        drop((storage_outcome, unexpected_replacements, transaction));
        Ok(())
    }

    /// Per-frame entry point: pump viewer updates, enqueue pending work, and
    /// drain any task outputs that have completed so far. Returns the events
    /// emitted this tick.
    ///
    /// Pass the desired viewer set via `viewers`; any paired viewer not in
    /// the list is treated as removed (its boxes shrink to empty, triggering
    /// unloads). This mirrors the C++ `process_viewers` + `process_meshing`
    /// pair, plus the `apply_*_response` callbacks folded in.
    pub fn try_process(
        &mut self,
        viewers: &[ViewerUpdate],
    ) -> Result<Vec<VoxelTerrainEvent>, VoxelTerrainRuntimeError> {
        if !self.shut_down && self.shutdown_epoch.is_some() {
            return Err(VoxelTerrainRuntimeError::ShutdownRetryPending);
        }
        let viewers = normalize_and_validate_viewer_updates(viewers)
            .map_err(VoxelTerrainRuntimeError::ViewerInput)?;
        if self.lod_count == 1 {
            let committed_tail = if self.shut_down {
                self.event_outbox.len()
            } else {
                self.try_run_fixed_transaction(
                    &viewers,
                    self.automatic_loading_enabled,
                    self.automatic_loading_enabled && !self.shutdown_in_progress,
                )?;
                self.event_outbox.len()
            };
            return Ok(self.drain_event_outbox_through(committed_tail));
        }
        self.validate_meshing_bounds()?;
        // Multi-LOD always uses the production clipbox planner.
        let previous_event_count = self.event_outbox.len();
        if self.shut_down {
            return Ok(self.drain_event_outbox_through(previous_event_count));
        }
        let dispatch_tasks = self.automatic_loading_enabled && !self.shutdown_in_progress;
        self.try_run_variable_transaction(&viewers, dispatch_tasks)?;
        self.checkpoint_acknowledged_saves_if_needed();
        let committed_tail = self.event_outbox.len();
        Ok(self.drain_event_outbox_through(committed_tail))
    }

    /// Apply every currently available completion and publish the committed
    /// event prefix in FIFO order. A checked failure retains the complete
    /// pre-call event outbox and every unconsumed completion.
    pub fn try_drain_completed_tasks(
        &mut self,
    ) -> Result<Vec<VoxelTerrainEvent>, VoxelTerrainRuntimeError> {
        if !self.shut_down && self.shutdown_epoch.is_some() {
            return Err(VoxelTerrainRuntimeError::ShutdownRetryPending);
        }
        if self.lod_count == 1 {
            self.try_run_fixed_transaction(
                &[],
                false,
                self.automatic_loading_enabled && !self.shutdown_in_progress,
            )?;
            let committed_tail = self.event_outbox.len();
            return Ok(self.drain_event_outbox_through(committed_tail));
        }
        self.validate_meshing_bounds()?;
        match self.try_drain_completion_work() {
            Ok(()) => {
                let committed_tail = self.event_outbox.len();
                Ok(self.drain_event_outbox_through(committed_tail))
            }
            Err(error) => Err(error),
        }
    }

    fn drain_event_outbox_through(&mut self, committed_tail: usize) -> Vec<VoxelTerrainEvent> {
        self.event_outbox.drain(..committed_tail).collect()
    }

    pub fn shutdown_and_flush(&mut self) -> Result<(), SaveFlushError> {
        if self.shut_down {
            return Ok(());
        }

        self.begin_shutdown_attempt()?;
        // Fixed-LOD shutdown has one public logical boundary: after waiting
        // for physical work, the completed prefix, DesiredEmpty viewer
        // collapse, dirty-owner routing and combined task batch commit in one
        // common transaction. A late C1 conflict therefore cannot expose a
        // completion-only prefix.
        let fixed_shutdown = self.lod_count == 1;
        let mut checkpoint_already_flushed = false;
        if fixed_shutdown {
            self.task_runner.wait_for_all_tasks();
        } else {
            checkpoint_already_flushed = match self.wait_and_drain_before_explicit_flush() {
                Ok(flushed) => flushed,
                Err(error) => {
                    self.shutdown_in_progress = false;
                    return Err(error);
                }
            };
        }

        if fixed_shutdown {
            if let Err(error) =
                self.try_run_fixed_transaction_capturing_resident(&[], true, false, true)
            {
                self.shutdown_in_progress = false;
                return Err(map_fixed_durability_error(error));
            }
            checkpoint_already_flushed = self.automatic_checkpoint_satisfied_empty_flush
                && self.save_journal.is_empty()
                && self.last_save_checkpoint_error.is_none();
            self.automatic_checkpoint_satisfied_empty_flush = false;
        } else if let Err(error) = self.try_capture_variable_resident_saves(true) {
            self.shutdown_in_progress = false;
            return Err(map_fixed_durability_error(error));
        }

        let mut result = self.flush_save_journal(checkpoint_already_flushed);
        if result.is_ok() && !fixed_shutdown {
            result = self
                .retire_variable_loaded_residency_after_shutdown()
                .map_err(map_fixed_durability_error);
        }
        if result.is_ok() {
            self.task_runner.shutdown();
            if !fixed_shutdown {
                // Variable-LOD buffers stay as a readable retired snapshot,
                // but the preceding fenced transaction has already retired
                // every storage-side residency owner to zero.
                self.retire_variable_runtime_after_shutdown();
            }
            // Persistence diagnostics may retain the exact completed task and
            // its follow-ups while the terrain is live. Successful shutdown
            // is the terminal ownership boundary, so no executable owner or
            // shutdown-only capability may survive it.
            self.completion_quarantine.clear();
            self.shutdown_mutation_permit = None;
            self.shut_down = true;
        }
        self.shutdown_in_progress = false;
        result
    }

    /// Retires the final variable-LOD residency owners after persistence is
    /// durable. Voxel payloads remain readable, while both storage viewer
    /// counts and the split terrain sidecar become canonical zero/zero under
    /// one publication fence.
    fn retire_variable_loaded_residency_after_shutdown(
        &mut self,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        let owner_count = self
            .loaded_data_residency
            .iter()
            .try_fold(0usize, |total, sidecar| total.checked_add(sidecar.len()))
            .ok_or(VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        let preview = self.data.begin_transaction_preview();
        let mut snapshots = Vec::new();
        snapshots
            .try_reserve_exact(owner_count)
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;
        let mut operations = Vec::new();
        operations
            .try_reserve_exact(owner_count)
            .map_err(|_| VoxelTerrainRuntimeError::CapacityReservationFailed)?;

        for (lod_index, sidecar) in self.loaded_data_residency.iter().enumerate() {
            for (&position, &residency) in sidecar {
                let location = BlockLocation {
                    position,
                    lod_index: lod_index as u8,
                };
                let snapshot = preview.block_snapshot(location).ok_or(
                    VoxelTerrainRuntimeError::DataMutation(
                        SharedVoxelDataMutationError::LodDestinationUnavailable {
                            position,
                            lod_index,
                        },
                    ),
                )?;
                let storage_viewers = if snapshot.is_present() {
                    snapshot.viewers()
                } else {
                    0
                };
                if !snapshot.is_present() || residency.checked_total(location)? != storage_viewers {
                    return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                        location,
                        tracked_resident_viewers: Some(residency.resident_viewers),
                        tracked_coverage_holds: Some(residency.coverage_holds),
                        storage_viewers,
                    });
                }
                if storage_viewers != 0 {
                    operations.push(SharedVoxelDataTransactionOperation::SetViewersExact {
                        location,
                        final_viewers: 0,
                    });
                }
                snapshots.push(snapshot);
            }
        }

        let mut transaction = match preview.prepare_transaction(operations, &snapshots) {
            Ok(transaction) => transaction,
            Err(error) => {
                let (storage_error, operations) = error.into_parts();
                drop(operations);
                return Err(VoxelTerrainRuntimeError::DataMutation(storage_error));
            }
        };
        let permit = self.shutdown_mutation_permit.as_ref().ok_or(
            VoxelTerrainRuntimeError::DataMutation(
                SharedVoxelDataMutationError::MutationAdmissionClosed,
            ),
        )?;
        transaction
            .authorize_shutdown_mutation(permit)
            .map_err(VoxelTerrainRuntimeError::DataMutation)?;
        let fence = transaction
            .commit_holding_publication_fence()
            .map_err(VoxelTerrainRuntimeError::DataMutation)?;
        for sidecar in &mut self.loaded_data_residency {
            for residency in sidecar.values_mut() {
                *residency = DataResidencyRefs::default();
            }
        }
        let outcome = fence.finish();
        drop((outcome, transaction));
        Ok(())
    }

    fn begin_shutdown_attempt(&mut self) -> Result<(), SaveFlushError> {
        if self.shutdown_epoch.is_none() {
            // BeginShutdown is one atomic lifecycle publication. Neither
            // counter nor state changes unless both successors exist.
            let next_request_epoch =
                self.request_epoch
                    .checked_add(1)
                    .ok_or(SaveFlushError::SaveAdmission {
                        error: VoxelTerrainRuntimeError::RequestEpochOverflow,
                    })?;
            let next_shutdown_epoch =
                self.next_shutdown_epoch
                    .checked_add(1)
                    .ok_or(SaveFlushError::SaveAdmission {
                        error: VoxelTerrainRuntimeError::ShutdownEpochOverflow,
                    })?;
            let shutdown_mutation_permit = self.data.close_mutation_admission_for_shutdown();
            self.shutdown_mutation_permit = Some(shutdown_mutation_permit);
            self.request_epoch = next_request_epoch;
            self.next_shutdown_epoch = next_shutdown_epoch;
            self.shutdown_epoch = Some(next_shutdown_epoch);
        }
        self.shutdown_in_progress = true;
        // Signal after the new epoch is visible and before waiting. Workers
        // can stop at their next cooperative checkpoint, while any already
        // completed old-epoch output remains deterministically stale.
        for loading in &self.loading_blocks {
            for entry in loading.values() {
                if let Some(request) = &entry.physical_request {
                    request.cancel();
                }
            }
        }
        for meshes in &self.mesh_maps {
            for entry in meshes.values() {
                if let Some(request) = &entry.physical_request {
                    request.cancel();
                }
            }
        }
        Ok(())
    }

    /// Canonically aborts every non-persistence variable-LOD owner after the
    /// runner is quiescent and the save journal is durable. Loaded voxel
    /// buffers intentionally remain as a read-only retired snapshot with
    /// canonical zero/zero residency; request maps, queues, and completion
    /// payloads do not.
    fn retire_variable_runtime_after_shutdown(&mut self) {
        self.paired_viewers.clear();
        for loading in &mut self.loading_blocks {
            for entry in loading.values() {
                if let Some(request) = &entry.physical_request {
                    request.cancel();
                }
            }
            loading.clear();
        }
        for meshes in &mut self.mesh_maps {
            for entry in meshes.values() {
                if let Some(request) = &entry.physical_request {
                    request.cancel();
                }
            }
            meshes.clear();
        }
        for pending in &mut self.blocks_pending_load {
            pending.clear();
        }
        for pending in &mut self.blocks_pending_update {
            pending.clear();
        }
        for retries in &mut self.data_view_retries {
            retries.clear();
        }
        for retries in &mut self.data_unview_retries {
            retries.clear();
        }
        self.raw_completion_inbox.clear();
        self.durable_completion_inbox.clear();
        self.direct_mesh_retry_inbox.clear();
        self.legacy_task_admission_retry.clear();
    }

    /// Persist all dirty resident blocks and already-journaled saves while
    /// leaving paging, viewers, and the task runner active.
    ///
    /// Failures retain their owned voxel payloads in the save journal, so the
    /// caller can invoke this method again after correcting the stream error.
    pub fn flush_pending_saves(&mut self) -> Result<(), SaveFlushError> {
        if self.shut_down {
            return Ok(());
        }
        if self.shutdown_epoch.is_some() {
            return Err(SaveFlushError::SaveAdmission {
                error: VoxelTerrainRuntimeError::ShutdownRetryPending,
            });
        }

        self.shutdown_in_progress = true;
        let checkpoint_already_flushed = match self.wait_and_drain_before_explicit_flush() {
            Ok(flushed) => flushed,
            Err(error) => {
                self.shutdown_in_progress = false;
                return Err(error);
            }
        };
        if self.lod_count == 1 {
            if let Err(error) =
                self.try_run_fixed_transaction_capturing_resident(&[], false, true, false)
            {
                self.shutdown_in_progress = false;
                return Err(map_fixed_durability_error(error));
            }
        } else {
            if let Err(error) = self.try_capture_variable_resident_saves(false) {
                self.shutdown_in_progress = false;
                return Err(map_fixed_durability_error(error));
            }
        }
        let result = self.flush_save_journal(checkpoint_already_flushed);
        if result.is_err() {
            // Ordinary flush is not BeginShutdown and must leave mutation
            // admission available for the active runtime.
            self.data.reopen_mutation_admission();
        }
        result
    }

    fn wait_and_drain_before_explicit_flush(&mut self) -> Result<bool, SaveFlushError> {
        if self.lod_count > 1 && !self.legacy_task_admission_retry.is_empty() {
            self.legacy_link_or_retain_task_batch(Vec::new());
        }
        self.task_runner.wait_for_all_tasks();
        self.drain_completed_tasks().map_err(|error| {
            if self.lod_count == 1 {
                map_fixed_durability_error(error)
            } else {
                SaveFlushError::CompletionDrain { error }
            }
        })?;
        let checkpoint_already_flushed = self.automatic_checkpoint_satisfied_empty_flush
            && self.save_journal.is_empty()
            && self.last_save_checkpoint_error.is_none();
        self.automatic_checkpoint_satisfied_empty_flush = false;
        Ok(checkpoint_already_flushed)
    }

    fn flush_save_journal(
        &mut self,
        checkpoint_already_flushed: bool,
    ) -> Result<(), SaveFlushError> {
        let result = if let Some(operation) = self.first_indeterminate_persistence_operation() {
            Err(SaveFlushError::IndeterminatePersistence { operation })
        } else if let Some(error) = self.save_dispatch_error.clone() {
            Err(SaveFlushError::SaveAdmission { error })
        } else if let Some(failure) = self.retained_save_admission_failures.front() {
            Err(SaveFlushError::SaveAdmission {
                error: failure.error.clone(),
            })
        } else {
            // An explicit flush or shutdown is the caller-authorized recovery
            // path after an automatic checkpoint failed.
            if self.lod_count == 1 {
                if let Err(error) = self.try_run_fixed_transaction_with_persistence_recovery(
                    FixedPersistenceRecoveryRequest {
                        reset_pending_save_failures: true,
                        authorize_automatic_checkpoint: true,
                    },
                ) {
                    self.shutdown_in_progress = false;
                    return Err(map_fixed_durability_error(error));
                }
            } else {
                self.automatic_save_checkpoint_blocked = false;
                for entry in self.save_journal.values_mut() {
                    if let Some(ActiveSaveAttempt::Pending(pending)) = &mut entry.active {
                        pending.meta.retry_count = 0;
                        pending.meta.last_error = None;
                    }
                }
            }
            if checkpoint_already_flushed && self.save_journal.is_empty() {
                Ok(())
            } else {
                self.flush_save_journal_with_attempts(MAX_EXPLICIT_SAVE_ATTEMPTS as usize)
            }
        };
        self.shutdown_in_progress = false;
        result
    }

    fn first_indeterminate_persistence_operation(&self) -> Option<PersistenceOperation> {
        if let Some(checkpoint) = self.save_checkpoint_in_flight.as_ref() {
            if matches!(
                checkpoint.state,
                CheckpointAttemptState::Indeterminate { .. }
            ) {
                return Some(PersistenceOperation::Flush {
                    checkpoint_generation: checkpoint.checkpoint_generation,
                });
            }
        }
        self.save_journal
            .iter()
            .filter_map(|(key, entry)| {
                let Some(ActiveSaveAttempt::Indeterminate { meta, .. }) = &entry.active else {
                    return None;
                };
                Some((*key, meta.block_revision, meta.generation))
            })
            .min_by_key(|(key, block_revision, generation)| {
                (
                    key.lod_index,
                    key.position.x,
                    key.position.y,
                    key.position.z,
                    *block_revision,
                    *generation,
                )
            })
            .map(
                |(key, block_revision, save_generation)| PersistenceOperation::Save {
                    location: BlockLocation {
                        position: key.position,
                        lod_index: key.lod_index,
                    },
                    block_revision,
                    save_generation,
                },
            )
    }

    fn flush_save_journal_with_attempts(
        &mut self,
        max_save_attempts: usize,
    ) -> Result<(), SaveFlushError> {
        for _ in 0..max_save_attempts {
            if self.lod_count == 1 {
                self.try_run_fixed_transaction(&[], false, true)
                    .map_err(map_fixed_durability_error)?;
            } else {
                let keys: Vec<SaveKey> = self.save_journal.keys().copied().collect();
                for key in keys {
                    self.dispatch_queued_save(key);
                }
            }

            if let Some(error) = self.save_dispatch_error.clone() {
                return Err(SaveFlushError::SaveAdmission { error });
            }

            if self.lod_count > 1 && !self.legacy_task_admission_retry.is_empty() {
                self.legacy_link_or_retain_task_batch(Vec::new());
            }
            self.task_runner.wait_for_all_tasks();
            self.drain_completed_tasks().map_err(|error| {
                if self.lod_count == 1 {
                    map_fixed_durability_error(error)
                } else {
                    SaveFlushError::CompletionDrain { error }
                }
            })?;

            let ready_to_flush = self
                .save_journal
                .values()
                .all(|entry| entry.active.is_none());
            if ready_to_flush {
                self.flush_acknowledged_saves()?;
                if self.save_journal.is_empty() {
                    return Ok(());
                }
            }
        }

        // A permanently failing block must not strand successful saves from
        // other keys. Make the acknowledged subset durable before reporting
        // only the entries whose block-save attempts actually failed.
        if self
            .save_journal
            .values()
            .any(|entry| entry.written_unflushed.is_some())
        {
            self.flush_acknowledged_saves()?;
        }
        let failed_count = self
            .save_journal
            .values()
            .filter(|entry| entry.has_failed_attempt())
            .count();
        debug_assert!(
            failed_count > 0 || self.save_journal.is_empty(),
            "bounded save loop ended without an acknowledged or failed entry"
        );
        Err(SaveFlushError::UnsavedBlocks {
            count: failed_count,
        })
    }

    fn flush_acknowledged_saves(&mut self) -> Result<(), SaveFlushError> {
        self.flush_acknowledged_saves_with_failure_diagnostics(true)
    }

    fn flush_acknowledged_saves_with_failure_diagnostics(
        &mut self,
        record_per_block_failure: bool,
    ) -> Result<(), SaveFlushError> {
        let checkpoint_generation = if self.lod_count == 1 {
            self.try_run_fixed_transaction_with_checkpoint(FixedCheckpointRequest {
                origin: CheckpointOrigin::Explicit,
                max_attempts: MAX_EXPLICIT_CHECKPOINT_ATTEMPTS,
                record_per_block_failure,
                reset_pending_retry_count: true,
            })
            .map_err(map_fixed_durability_error)?;
            self.save_checkpoint_in_flight
                .as_ref()
                .map(|checkpoint| checkpoint.checkpoint_generation)
                .ok_or(SaveFlushError::SaveAdmission {
                    error: VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                        requested: PersistenceOperation::Flush {
                            checkpoint_generation: self.next_save_checkpoint_generation,
                        },
                    },
                })?
        } else if let Some(checkpoint) = self.save_checkpoint_in_flight.as_mut() {
            if checkpoint.state == CheckpointAttemptState::Pending {
                checkpoint.retry_count = 0;
            }
            checkpoint.origin = CheckpointOrigin::Explicit;
            checkpoint.max_attempts = MAX_EXPLICIT_CHECKPOINT_ATTEMPTS;
            checkpoint.record_per_block_failure = record_per_block_failure;
            checkpoint.checkpoint_generation
        } else {
            let operation = self
                .begin_checkpoint(
                    CheckpointOrigin::Explicit,
                    MAX_EXPLICIT_CHECKPOINT_ATTEMPTS,
                    record_per_block_failure,
                )
                .map_err(|error| SaveFlushError::SaveAdmission { error })?;
            let PersistenceOperation::Flush {
                checkpoint_generation,
            } = operation
            else {
                unreachable!("begin_checkpoint only creates flush operations")
            };
            checkpoint_generation
        };
        let operation = PersistenceOperation::Flush {
            checkpoint_generation,
        };

        loop {
            if let Some((completed_generation, result)) = self.last_checkpoint_outcome.take() {
                if completed_generation == checkpoint_generation {
                    return result.map_err(SaveFlushError::Stream);
                }
                self.last_checkpoint_outcome = Some((completed_generation, result));
            }

            let Some(checkpoint) = self.save_checkpoint_in_flight.as_ref() else {
                return Err(SaveFlushError::SaveAdmission {
                    error: VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                        requested: operation,
                    },
                });
            };
            match checkpoint.state {
                CheckpointAttemptState::Pending => {
                    if checkpoint.retry_count >= checkpoint.max_attempts {
                        return Err(SaveFlushError::SaveAdmission {
                            error: VoxelTerrainRuntimeError::PersistenceRetryLimitExceeded {
                                operation,
                            },
                        });
                    }
                    if self.lod_count == 1 {
                        self.try_run_fixed_transaction(&[], false, true)
                            .map_err(map_fixed_durability_error)?;
                    } else {
                        self.dispatch_pending_checkpoint();
                    }
                    if let Some(error) = self.save_dispatch_error.clone() {
                        return Err(SaveFlushError::SaveAdmission { error });
                    }
                }
                CheckpointAttemptState::WriteInFlight { .. } => {}
                CheckpointAttemptState::Indeterminate { .. } => {
                    return Err(SaveFlushError::IndeterminatePersistence { operation });
                }
            }
            if self.lod_count > 1 && !self.legacy_task_admission_retry.is_empty() {
                self.legacy_link_or_retain_task_batch(Vec::new());
            }
            self.task_runner.wait_for_all_tasks();
            self.drain_completed_tasks().map_err(|error| {
                if self.lod_count == 1 {
                    map_fixed_durability_error(error)
                } else {
                    SaveFlushError::CompletionDrain { error }
                }
            })?;
        }
    }

    fn acknowledged_checkpoint_snapshot(&self) -> Vec<SaveCheckpointSnapshot> {
        let mut snapshots = self
            .save_journal
            .iter()
            .filter_map(|(key, entry)| {
                entry
                    .written_unflushed
                    .as_ref()
                    .map(|written| SaveCheckpointSnapshot {
                        key: *key,
                        block_revision: written.block_revision,
                        generation: written.generation,
                    })
            })
            .collect::<Vec<_>>();
        snapshots.sort_unstable_by_key(|snapshot| {
            (
                snapshot.key.lod_index,
                snapshot.key.position.x,
                snapshot.key.position.y,
                snapshot.key.position.z,
            )
        });
        snapshots
    }

    fn apply_acknowledged_checkpoint_result(
        &mut self,
        acknowledged: Vec<SaveCheckpointSnapshot>,
        result: StreamResult<()>,
        record_per_block_failure: bool,
    ) -> Result<(), SaveFlushError> {
        match result {
            Ok(()) => {
                self.last_save_checkpoint_error = None;
                self.automatic_save_checkpoint_blocked = false;
                for snapshot in acknowledged {
                    let Some(entry) = self.save_journal.get_mut(&snapshot.key) else {
                        continue;
                    };
                    if entry.written_unflushed.as_ref().is_none_or(|written| {
                        written.block_revision != snapshot.block_revision
                            || written.generation != snapshot.generation
                    }) {
                        continue;
                    }
                    entry.written_unflushed = None;
                    entry.promote_queued_if_idle();
                    let should_defer = matches!(entry.active, Some(ActiveSaveAttempt::Pending(_)));
                    let should_remove = entry.is_empty();
                    if should_defer && !self.deferred_save_dispatch_keys.contains(&snapshot.key) {
                        self.deferred_save_dispatch_keys.push(snapshot.key);
                    }
                    if should_remove {
                        self.save_journal.remove(&snapshot.key);
                    }
                }
                Ok(())
            }
            Err(error) => {
                self.last_save_checkpoint_error = Some(error.clone());
                self.automatic_save_checkpoint_blocked = true;
                for snapshot in acknowledged {
                    let Some(entry) = self.save_journal.get_mut(&snapshot.key) else {
                        continue;
                    };
                    if entry.written_unflushed.as_ref().is_none_or(|written| {
                        written.block_revision != snapshot.block_revision
                            || written.generation != snapshot.generation
                    }) {
                        continue;
                    }
                    let Some(written) = entry.written_unflushed.take() else {
                        continue;
                    };
                    let mut meta = SaveAttemptMeta {
                        block_revision: written.block_revision,
                        generation: written.generation,
                        retry_count: 0,
                        last_error: None,
                    };
                    if record_per_block_failure {
                        meta.retry_count = meta.retry_count.saturating_add(1);
                        meta.last_error = Some(error.clone());
                    }
                    let restored = PendingSave {
                        meta,
                        payload: written.payload,
                    };
                    if let Some(active) = entry.active.take() {
                        match active {
                            ActiveSaveAttempt::Pending(pending) => {
                                entry.queued_newer.push_front(pending)
                            }
                            other => entry.active = Some(other),
                        }
                    }
                    if entry.active.is_none() {
                        entry.active = Some(ActiveSaveAttempt::Pending(restored));
                    } else {
                        entry.queued_newer.push_front(restored);
                    }
                }
                Err(SaveFlushError::Stream(error))
            }
        }
    }

    fn apply_save_checkpoint_response(
        &mut self,
        terminal: FlushTaskTerminal,
        attempt_ordinal: u64,
    ) -> bool {
        let Some(PersistenceAcknowledgement::Flush(result)) = terminal.acknowledgement else {
            return false;
        };
        let Some(checkpoint) = self.save_checkpoint_in_flight.as_ref() else {
            return false;
        };
        if checkpoint.checkpoint_generation != terminal.checkpoint_generation
            || checkpoint.state != (CheckpointAttemptState::WriteInFlight { attempt_ordinal })
        {
            return false;
        }
        let Some(checkpoint) = self.save_checkpoint_in_flight.take() else {
            return false;
        };
        let checkpoint_succeeded = result.is_ok();
        if checkpoint.origin == CheckpointOrigin::Explicit {
            self.last_checkpoint_outcome = Some((terminal.checkpoint_generation, result.clone()));
        }
        let _ = self.apply_acknowledged_checkpoint_result(
            checkpoint.acknowledged,
            result,
            checkpoint.record_per_block_failure,
        );
        self.automatic_checkpoint_satisfied_empty_flush =
            checkpoint_succeeded && self.save_journal.is_empty();
        self.force_checkpoint_requested = false;
        true
    }

    fn checkpoint_attempt_matches(
        &self,
        terminal: &FlushTaskTerminal,
        attempt_ordinal: u64,
    ) -> bool {
        self.save_checkpoint_in_flight
            .as_ref()
            .is_some_and(|checkpoint| {
                checkpoint.checkpoint_generation == terminal.checkpoint_generation
                    && checkpoint.state
                        == (CheckpointAttemptState::WriteInFlight { attempt_ordinal })
            })
    }

    fn begin_checkpoint(
        &mut self,
        origin: CheckpointOrigin,
        max_attempts: u32,
        record_per_block_failure: bool,
    ) -> Result<PersistenceOperation, VoxelTerrainRuntimeError> {
        if let Some(checkpoint) = self.save_checkpoint_in_flight.as_ref() {
            return Ok(PersistenceOperation::Flush {
                checkpoint_generation: checkpoint.checkpoint_generation,
            });
        }
        let acknowledged = self.acknowledged_checkpoint_snapshot();
        let checkpoint_generation =
            allocate_persistence_generation(&mut self.next_save_checkpoint_generation)?;
        self.last_checkpoint_outcome = None;
        self.save_checkpoint_in_flight = Some(SaveCheckpointInFlight {
            checkpoint_generation,
            acknowledged,
            state: CheckpointAttemptState::Pending,
            retry_count: 0,
            max_attempts,
            origin,
            record_per_block_failure,
        });
        self.dispatch_pending_checkpoint();
        Ok(PersistenceOperation::Flush {
            checkpoint_generation,
        })
    }

    fn dispatch_pending_checkpoint(&mut self) {
        let Some(checkpoint) = self.save_checkpoint_in_flight.as_ref() else {
            return;
        };
        if checkpoint.state != CheckpointAttemptState::Pending
            || checkpoint.retry_count >= checkpoint.max_attempts
        {
            return;
        }
        let operation = PersistenceOperation::Flush {
            checkpoint_generation: checkpoint.checkpoint_generation,
        };
        let attempt_ordinal = self.next_persistence_attempt_ordinal;
        let Some(next_attempt_ordinal) = attempt_ordinal.checked_add(1) else {
            self.save_dispatch_error =
                Some(VoxelTerrainRuntimeError::PersistenceAttemptOverflow { operation });
            return;
        };
        let Some(checkpoint) = self.save_checkpoint_in_flight.as_mut() else {
            return;
        };
        self.next_persistence_attempt_ordinal = next_attempt_ordinal;
        checkpoint.state = CheckpointAttemptState::WriteInFlight { attempt_ordinal };
        let task = FlushVoxelStreamTask::new_with_attempt_ordinal(
            self.stream.clone(),
            checkpoint.checkpoint_generation,
            attempt_ordinal,
        );
        #[cfg(test)]
        let mut task = task;
        #[cfg(test)]
        {
            if self.panic_next_flush_before_io_attempts_for_test != 0 {
                self.panic_next_flush_before_io_attempts_for_test -= 1;
                task.set_panic_before_io_for_test(true);
            }
            if std::mem::take(&mut self.panic_next_flush_after_ack_for_test) {
                task.set_panic_after_ack_for_test(true);
            }
        }
        self.legacy_link_or_retain_task_batch(vec![ScheduledTask::new(
            Box::new(task),
            TaskLane::Serial,
        )]);
    }

    fn restore_checkpoint_before_io(
        &mut self,
        terminal: &FlushTaskTerminal,
        attempt_ordinal: u64,
    ) -> bool {
        let Some(checkpoint) = self.save_checkpoint_in_flight.as_mut() else {
            return false;
        };
        if checkpoint.checkpoint_generation != terminal.checkpoint_generation
            || checkpoint.state != (CheckpointAttemptState::WriteInFlight { attempt_ordinal })
        {
            return false;
        }
        checkpoint.retry_count = checkpoint.retry_count.saturating_add(1);
        checkpoint.state = CheckpointAttemptState::Pending;
        self.deferred_checkpoint_dispatch = true;
        true
    }

    fn mark_checkpoint_indeterminate(
        &mut self,
        terminal: &FlushTaskTerminal,
        attempt_ordinal: u64,
    ) -> bool {
        let Some(checkpoint) = self.save_checkpoint_in_flight.as_mut() else {
            return false;
        };
        if checkpoint.checkpoint_generation != terminal.checkpoint_generation
            || checkpoint.state != (CheckpointAttemptState::WriteInFlight { attempt_ordinal })
        {
            return false;
        }
        checkpoint.state = CheckpointAttemptState::Indeterminate { attempt_ordinal };
        true
    }

    fn checkpoint_acknowledged_saves_if_needed(&mut self) {
        if self.automatic_save_checkpoint_blocked || self.save_checkpoint_in_flight.is_some() {
            return;
        }
        if self.save_journal.values().any(|entry| {
            matches!(
                entry.active,
                Some(
                    ActiveSaveAttempt::WriteInFlight { .. }
                        | ActiveSaveAttempt::Indeterminate { .. }
                )
            )
        }) {
            return;
        }
        let (payload_count, allocated_bytes) =
            self.save_journal
                .values()
                .fold((0usize, 0usize), |(count, bytes), entry| {
                    if entry.written_unflushed.is_some() {
                        (
                            count.saturating_add(1),
                            bytes.saturating_add(entry.acknowledged_payload_bytes()),
                        )
                    } else {
                        (count, bytes)
                    }
                });
        let successor_is_blocked = self
            .save_journal
            .values()
            .any(|entry| entry.written_unflushed.is_some() && !entry.queued_newer.is_empty());
        if !self.force_checkpoint_requested
            && !successor_is_blocked
            && payload_count < AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD
            && allocated_bytes < AUTOMATIC_SAVE_CHECKPOINT_BYTE_THRESHOLD
        {
            return;
        }

        if let Err(error) = self.begin_checkpoint(
            CheckpointOrigin::Automatic,
            MAX_AUTOMATIC_CHECKPOINT_ATTEMPTS,
            false,
        ) {
            self.save_dispatch_error = Some(error);
        }
    }

    /// Detailed diagnostics for the save entries retained after the most
    /// recent failure. The legacy [`SaveFlushError`] shape only carries count.
    pub fn last_save_failures(&self) -> Vec<UnsavedBlockSave> {
        self.last_save_failure_details()
            .into_iter()
            .map(|failure| UnsavedBlockSave {
                position_in_blocks: failure.position_in_blocks,
                lod_index: failure.lod_index,
                retry_count: failure.retry_count,
                error: failure.error,
            })
            .collect()
    }

    /// Exact revisioned diagnostics for save entries retained after the most
    /// recent failure.
    pub fn last_save_failure_details(&self) -> Vec<UnsavedBlockSaveDetails> {
        let mut blocks = self
            .save_journal
            .iter()
            .filter(|(_, entry)| entry.has_failed_attempt())
            .map(|(key, entry)| UnsavedBlockSaveDetails {
                position_in_blocks: key.position,
                lod_index: key.lod_index,
                block_revision: entry.active_meta().map_or(0, |meta| meta.block_revision),
                save_generation: entry.active_meta().map_or(0, |meta| meta.generation),
                retry_count: entry.active_meta().map_or(0, |meta| meta.retry_count),
                error: entry.active_meta().and_then(|meta| meta.last_error.clone()),
            })
            .collect::<Vec<_>>();
        blocks.sort_unstable_by_key(|block| {
            (
                block.position_in_blocks.x,
                block.position_in_blocks.y,
                block.position_in_blocks.z,
                block.lod_index,
            )
        });
        blocks
    }

    /// Most recent stream-wide checkpoint error, if acknowledged block writes
    /// could not be made durable. A later successful checkpoint clears it.
    pub fn last_save_checkpoint_error(&self) -> Option<&VoxelStreamError> {
        self.last_save_checkpoint_error.as_ref()
    }

    #[cfg(test)]
    fn journal_persistence_state_for_test(
        &self,
        operation: PersistenceOperation,
    ) -> Option<JournalPersistenceState> {
        let PersistenceOperation::Save {
            location,
            block_revision,
            save_generation,
        } = operation
        else {
            return None;
        };
        self.save_journal
            .get(&SaveKey::new(location.position, location.lod_index))?
            .state_for_identity(block_revision, save_generation)
    }

    #[cfg(test)]
    fn checkpoint_persistence_state_for_test(
        &self,
        operation: PersistenceOperation,
    ) -> Option<JournalPersistenceState> {
        let PersistenceOperation::Flush {
            checkpoint_generation,
        } = operation
        else {
            return None;
        };
        let checkpoint = self.save_checkpoint_in_flight.as_ref()?;
        if checkpoint.checkpoint_generation != checkpoint_generation {
            return None;
        }
        Some(match checkpoint.state {
            CheckpointAttemptState::Pending => JournalPersistenceState::PendingWrite,
            CheckpointAttemptState::WriteInFlight { .. } => JournalPersistenceState::WriteInFlight,
            CheckpointAttemptState::Indeterminate { .. } => JournalPersistenceState::Indeterminate,
        })
    }

    #[cfg(test)]
    fn checkpoint_retry_count_for_test(&self, operation: PersistenceOperation) -> Option<u32> {
        let PersistenceOperation::Flush {
            checkpoint_generation,
        } = operation
        else {
            return None;
        };
        let checkpoint = self.save_checkpoint_in_flight.as_ref()?;
        (checkpoint.checkpoint_generation == checkpoint_generation)
            .then_some(checkpoint.retry_count)
    }

    #[cfg(test)]
    fn journal_payload_ptr_for_test(&self, operation: PersistenceOperation) -> Option<*const u8> {
        let PersistenceOperation::Save {
            location,
            block_revision,
            save_generation,
        } = operation
        else {
            return None;
        };
        let entry = self
            .save_journal
            .get(&SaveKey::new(location.position, location.lod_index))?;
        let payload = if let Some(written) = entry.written_unflushed.as_ref().filter(|written| {
            written.block_revision == block_revision && written.generation == save_generation
        }) {
            Some(&written.payload)
        } else {
            match &entry.active {
                Some(ActiveSaveAttempt::Pending(pending))
                    if pending.meta.block_revision == block_revision
                        && pending.meta.generation == save_generation =>
                {
                    Some(&pending.payload)
                }
                _ => entry
                    .queued_newer
                    .iter()
                    .find(|queued| {
                        queued.meta.block_revision == block_revision
                            && queued.meta.generation == save_generation
                    })
                    .map(|queued| &queued.payload),
            }
        }?;
        Some(
            payload
                .channel_bytes(crate::storage::ChannelId::Type.index())
                .as_ptr(),
        )
    }

    #[cfg(test)]
    fn quarantined_save_payload_ptr_for_test(
        &self,
        operation: PersistenceOperation,
    ) -> Option<*const u8> {
        let PersistenceOperation::Save {
            location,
            block_revision,
            save_generation,
        } = operation
        else {
            return None;
        };
        self.completion_quarantine.iter().find_map(|completion| {
            let QuarantinedCompletion::Persistence {
                terminal: PersistenceTaskTerminal::Save(terminal),
                ..
            } = completion
            else {
                return None;
            };
            (terminal.location == location
                && terminal.block_revision == block_revision
                && terminal.save_generation == save_generation)
                .then(|| {
                    terminal
                        .payload
                        .channel_bytes(crate::storage::ChannelId::Type.index())
                        .as_ptr()
                })
        })
    }

    #[cfg(test)]
    fn save_error_for_test(&self, operation: PersistenceOperation) -> Option<&VoxelStreamError> {
        let PersistenceOperation::Save {
            location,
            block_revision,
            save_generation,
        } = operation
        else {
            return None;
        };
        let entry = self
            .save_journal
            .get(&SaveKey::new(location.position, location.lod_index))?;
        match &entry.active {
            Some(ActiveSaveAttempt::Pending(pending))
                if pending.meta.block_revision == block_revision
                    && pending.meta.generation == save_generation =>
            {
                pending.meta.last_error.as_ref()
            }
            _ => None,
        }
    }

    /// Most recent checked shared-data mutation failure. Paging mutations that
    /// fail before commit remain queued for retry. A caller-driven LOD-0 edit
    /// returns `false` if that edit itself fails; if only coarse propagation
    /// fails, the committed edit returns `true` and the propagation remains
    /// queued while this diagnostic stays queryable.
    pub fn last_data_mutation_error(&self) -> Option<&SharedVoxelDataMutationError> {
        self.last_data_mutation_error.as_ref()
    }

    #[allow(dead_code)] // retained as a unit-test harness after the planner cutover
    fn process_viewers(
        &mut self,
        viewers: &[ViewerUpdate],
    ) -> Result<(), VoxelTerrainRuntimeError> {
        debug_assert!(
            self.lod_count > 1,
            "fixed LOD viewer updates use the common prepared transaction"
        );
        let data_block_size = self.data_block_size();

        // Update paired viewers, recording prev_state for diffing.
        let mut seen = Vec::with_capacity(viewers.len());
        for update in viewers {
            seen.push(update.id);
            let paired = self.paired_viewers.iter_mut().find(|p| p.id == update.id);
            let horizontal = update
                .horizontal_view_distance_voxels
                .min(self.max_view_distance_voxels);
            let vertical = update
                .vertical_view_distance_voxels
                .min(self.max_view_distance_voxels);
            if let Some(paired) = paired {
                paired.prev_state = paired.state.clone();
                paired.state.local_position_voxels = update.world_position_voxels;
                paired.state.horizontal_view_distance_voxels = horizontal;
                paired.state.vertical_view_distance_voxels = vertical;
                paired.state.demand = update.demand;
            } else {
                let state = ViewerState {
                    local_position_voxels: update.world_position_voxels,
                    horizontal_view_distance_voxels: horizontal,
                    vertical_view_distance_voxels: vertical,
                    demand: update.demand,
                    ..ViewerState::default()
                };
                self.paired_viewers.push(PairedViewer {
                    id: update.id,
                    state: state.clone(),
                    prev_state: ViewerState::default(),
                });
            }
        }
        self.paired_viewers.sort_unstable_by_key(|viewer| viewer.id);

        // Compute boxes for every paired viewer (new and updated alike).
        for paired in self.paired_viewers.iter_mut() {
            if !seen.contains(&paired.id) {
                paired.prev_state = paired.state.clone();
                paired.state.data_box = Box3i::default();
                paired.state.mesh_box = Box3i::default();
                paired.state.data_box_per_lod = Vec::new();
                paired.state.mesh_box_per_lod = Vec::new();
                paired.state.demand = MeshDemand::default();
                continue;
            }
            compute_viewer_boxes_multi_lod(&mut paired.state, data_block_size, self.lod_count);
        }

        // Diff each viewer and apply view/unview operations.
        self.process_viewers_multi_lod()?;

        // Drop unpaired viewers from the list now that their boxes have
        // collapsed (matches the C++ swap-and-pop at the end of
        // process_viewers).
        self.paired_viewers.retain(|p| seen.contains(&p.id));
        Ok(())
    }

    /// Multi-LOD diff path: diff per-LOD boxes and dispatch view/unview per LOD.
    #[allow(dead_code)] // retained as a unit-test harness after the planner cutover
    fn process_viewers_multi_lod(&mut self) -> Result<(), VoxelTerrainRuntimeError> {
        let lod_count = self.lod_count as usize;
        // Collect ops per LOD (to avoid borrow conflicts).
        let mut data_unview = (0..lod_count)
            .map(|lod| std::mem::take(&mut self.data_unview_retries[lod]))
            .collect::<Vec<_>>();
        let mut data_view = (0..lod_count)
            .map(|lod| std::mem::take(&mut self.data_view_retries[lod]))
            .collect::<Vec<_>>();
        let mut mesh_unview: Vec<Vec<Vector3i>> = vec![Vec::new(); lod_count];
        let mut mesh_view: Vec<Vec<Vector3i>> = vec![Vec::new(); lod_count];

        for paired in self.paired_viewers.iter() {
            for lod in 0..lod_count {
                let prev_data = paired
                    .prev_state
                    .data_box_per_lod
                    .get(lod)
                    .copied()
                    .unwrap_or_default();
                let curr_data = paired
                    .state
                    .data_box_per_lod
                    .get(lod)
                    .copied()
                    .unwrap_or_default();
                if prev_data != curr_data {
                    for b in prev_data.difference(curr_data) {
                        data_unview[lod].push(PendingDataMutation {
                            box_in_blocks: b,
                            retry_count: 0,
                        });
                    }
                    for b in curr_data.difference(prev_data) {
                        data_view[lod].push(PendingDataMutation {
                            box_in_blocks: b,
                            retry_count: 0,
                        });
                    }
                }
                let prev_mesh = paired
                    .prev_state
                    .mesh_box_per_lod
                    .get(lod)
                    .copied()
                    .unwrap_or_default();
                let curr_mesh = paired
                    .state
                    .mesh_box_per_lod
                    .get(lod)
                    .copied()
                    .unwrap_or_default();
                if prev_mesh != curr_mesh {
                    for slab in prev_mesh.difference(curr_mesh) {
                        for pos in slab.iter_cells_zxy() {
                            mesh_unview[lod].push(pos);
                        }
                    }
                    for slab in curr_mesh.difference(prev_mesh) {
                        for pos in slab.iter_cells_zxy() {
                            mesh_view[lod].push(pos);
                        }
                    }
                }
            }
        }

        for lod in 0..lod_count {
            for pending in &data_unview[lod] {
                self.legacy_apply_data_unview_attempt(
                    pending.box_in_blocks,
                    lod,
                    pending.retry_count,
                )?;
            }
            for pending in &data_view[lod] {
                self.legacy_apply_data_view_attempt(
                    pending.box_in_blocks,
                    lod,
                    pending.retry_count,
                )?;
            }
            for pos in &mesh_unview[lod] {
                self.try_legacy_unview_mesh_block(*pos, lod)?;
            }
            for pos in &mesh_view[lod] {
                self.try_legacy_view_mesh_block(*pos, lod)?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn apply_data_view(&mut self, box_to_load: Box3i, lod: usize) {
        let _ = self.legacy_apply_data_view_attempt(box_to_load, lod, 0);
    }

    #[allow(dead_code)] // retained as a unit-test harness after the planner cutover
    fn legacy_apply_data_view_attempt(
        &mut self,
        box_to_load: Box3i,
        lod: usize,
        retry_count: u32,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        match self.try_apply_variable_data_residency_delta(box_to_load, lod, 1) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.record_legacy_data_mutation_error(&error);
                if retry_count < MAX_LOAD_RETRIES {
                    self.data_view_retries[lod].push(PendingDataMutation {
                        box_in_blocks: box_to_load,
                        retry_count: retry_count + 1,
                    });
                }
                Err(error)
            }
        }
    }

    #[cfg(test)]
    fn apply_data_unview(&mut self, box_to_unload: Box3i, lod: usize) {
        let _ = self.legacy_apply_data_unview_attempt(box_to_unload, lod, 0);
    }

    #[allow(dead_code)] // retained as a unit-test harness after the planner cutover
    fn legacy_apply_data_unview_attempt(
        &mut self,
        box_to_unload: Box3i,
        lod: usize,
        retry_count: u32,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        if let Err(error) = self.try_apply_variable_data_residency_delta(box_to_unload, lod, -1) {
            self.record_legacy_data_mutation_error(&error);
            if retry_count < MAX_LOAD_RETRIES || self.shutdown_in_progress {
                self.data_unview_retries[lod].push(PendingDataMutation {
                    box_in_blocks: box_to_unload,
                    retry_count: retry_count
                        .checked_add(1)
                        .unwrap_or(MAX_LOAD_RETRIES)
                        .min(MAX_LOAD_RETRIES),
                });
            }
            return Err(error);
        }
        Ok(())
    }

    #[allow(dead_code)] // retained as a unit-test harness after the planner cutover
    fn record_legacy_data_mutation_error(&mut self, error: &VoxelTerrainRuntimeError) {
        self.last_data_mutation_error = match error {
            VoxelTerrainRuntimeError::DataMutation(error) => Some(error.clone()),
            VoxelTerrainRuntimeError::DataRefcountOverflow {
                location,
                field: DataRefField::ResidentViewers,
            } => Some(SharedVoxelDataMutationError::ViewerCountOverflow {
                position: location.position,
                lod_index: usize::from(location.lod_index),
            }),
            VoxelTerrainRuntimeError::DataRefcountUnderflow {
                location,
                field: DataRefField::ResidentViewers,
            } => Some(SharedVoxelDataMutationError::ViewerCountUnderflow {
                position: location.position,
                lod_index: usize::from(location.lod_index),
            }),
            _ => None,
        };
    }

    /// Applies one variable-LOD viewer residency delta as a C1 storage
    /// transaction joined to the exact terrain sidecars under the publication
    /// fence. Every touched key is observed, including missing blocks, so a
    /// concurrent insertion or a stale sidecar rejects the whole box before
    /// its first live write.
    #[allow(dead_code)] // retained as a unit-test harness after the planner cutover
    fn try_apply_variable_data_residency_delta(
        &mut self,
        requested_box: Box3i,
        lod: usize,
        delta: i64,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        if self.shutdown_epoch.is_some() {
            return Err(VoxelTerrainRuntimeError::ShutdownRetryPending);
        }
        let lod_index = u8::try_from(lod)
            .map_err(|_| VoxelTerrainRuntimeError::LodMath(LodMathError::InvalidLodCount))?;
        if lod >= usize::from(self.lod_count) || lod >= self.loaded_data_residency.len() {
            return Err(VoxelTerrainRuntimeError::LodMath(
                LodMathError::InvalidLodCount,
            ));
        }
        let bounds = bounds_in_lod_blocks(self.data.bounds(), self.data_block_size(), lod_index)?;
        let blocks_box = checked_box_intersection(requested_box, bounds)?;
        if blocks_box.is_empty() {
            return Ok(());
        }
        let touched_count = usize::try_from(blocks_box.size.volume_u64())
            .map_err(|_| VoxelTerrainRuntimeError::TaskCountOverflow)?;
        let preview = self.data.begin_transaction_preview();
        let mut operations = Vec::new();
        operations
            .try_reserve_exact(touched_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut snapshots = Vec::new();
        snapshots
            .try_reserve_exact(touched_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut residency_updates = Vec::<(Vector3i, Option<DataResidencyRefs>)>::new();
        residency_updates
            .try_reserve_exact(touched_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut loading_updates = Vec::<(Vector3i, Option<LoadingBlockEntry>)>::new();
        loading_updates
            .try_reserve_exact(touched_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut removed_positions = Vec::new();
        removed_positions
            .try_reserve_exact(touched_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut prepared_removal_saves = Vec::new();
        prepared_removal_saves
            .try_reserve_exact(touched_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut save_dispatch_keys = Vec::new();
        save_dispatch_keys
            .try_reserve_exact(touched_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut next_pending_load = Vec::new();
        next_pending_load
            .try_reserve_exact(
                self.blocks_pending_load[lod]
                    .len()
                    .checked_add(touched_count)
                    .ok_or(VoxelTerrainRuntimeError::TaskCountOverflow)?,
            )
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        next_pending_load.extend(self.blocks_pending_load[lod].iter().copied());
        let mut next_request_generation = self.next_request_generation;
        let mut next_save_generation = self.next_save_generation;
        let mut next_stats = self.stats;

        for position in blocks_box.iter_cells_zxy() {
            let location = BlockLocation {
                position,
                lod_index,
            };
            let snapshot = preview.block_snapshot(location).ok_or({
                VoxelTerrainRuntimeError::DataMutation(
                    SharedVoxelDataMutationError::LodDestinationUnavailable {
                        position,
                        lod_index: lod,
                    },
                )
            })?;
            if snapshot.is_present() {
                if let Some(loading) = self.loading_blocks[lod].get(&position) {
                    return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                        location,
                        tracked_resident_viewers: Some(loading.residency.resident_viewers),
                        tracked_coverage_holds: Some(loading.residency.coverage_holds),
                        storage_viewers: snapshot.viewers(),
                    });
                }
                let current = self.validate_loaded_data_residency_snapshot(&snapshot)?;
                let next =
                    current.checked_apply_delta(location, DataRefField::ResidentViewers, delta)?;
                if next.is_empty() {
                    if snapshot.is_modified() {
                        if !snapshot.has_voxels() {
                            return Err(VoxelTerrainRuntimeError::MissingSavePayload);
                        }
                        let save_generation =
                            allocate_persistence_generation(&mut next_save_generation)?;
                        prepared_removal_saves.push(PreparedVariableRemovalSave {
                            location,
                            save_generation,
                        });
                        save_dispatch_keys.push(SaveKey::new(position, lod_index));
                    }
                    operations.push(SharedVoxelDataTransactionOperation::Remove { location });
                    residency_updates.push((position, None));
                    removed_positions.push(position);
                    next_stats.blocks_unloaded = next_stats
                        .blocks_unloaded
                        .checked_add(1)
                        .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
                } else {
                    let final_viewers = next.checked_total(location)?;
                    if final_viewers != snapshot.viewers() {
                        operations.push(SharedVoxelDataTransactionOperation::SetViewersExact {
                            location,
                            final_viewers,
                        });
                    }
                    residency_updates.push((position, Some(next)));
                }
            } else {
                if let Some(tracked) = self.loaded_data_residency[lod].get(&position).copied() {
                    return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                        location,
                        tracked_resident_viewers: Some(tracked.resident_viewers),
                        tracked_coverage_holds: Some(tracked.coverage_holds),
                        storage_viewers: 0,
                    });
                }
                let current_entry = self.loading_blocks[lod].get(&position);
                let current =
                    current_entry.map_or(DataResidencyRefs::default(), |entry| entry.residency);
                let next =
                    current.checked_apply_delta(location, DataRefField::ResidentViewers, delta)?;
                if next.is_empty() {
                    loading_updates.push((position, None));
                    next_pending_load.retain(|pending| *pending != position);
                } else if let Some(current_entry) = current_entry {
                    let mut next_entry = current_entry.clone();
                    next_entry.residency = next;
                    let needs_fresh_request = delta > 0
                        && (matches!(
                            next_entry.request_state,
                            LoadRequestState::NotFound | LoadRequestState::Exhausted
                        ) || next_entry.physical_request.is_none());
                    if needs_fresh_request {
                        let (generation, physical_request) = allocate_physical_request(
                            self.request_epoch,
                            &mut next_request_generation,
                        )?;
                        next_entry.retry_count = 0;
                        next_entry.request_generation = generation;
                        next_entry.request_state = LoadRequestState::Queued;
                        next_entry.physical_request = Some(physical_request);
                        if !next_pending_load.contains(&position) {
                            next_pending_load.push(position);
                        }
                    }
                    loading_updates.push((position, Some(next_entry)));
                } else {
                    let (generation, physical_request) = allocate_physical_request(
                        self.request_epoch,
                        &mut next_request_generation,
                    )?;
                    loading_updates.push((
                        position,
                        Some(LoadingBlockEntry {
                            residency: next,
                            retry_count: 0,
                            request_generation: generation,
                            request_state: LoadRequestState::Queued,
                            physical_request: Some(physical_request),
                        }),
                    ));
                    if !next_pending_load.contains(&position) {
                        next_pending_load.push(position);
                    }
                }
            }
            snapshots.push(snapshot);
        }

        let loading_insertions = loading_updates
            .iter()
            .filter(|(position, next)| {
                next.is_some() && !self.loading_blocks[lod].contains_key(position)
            })
            .count();
        self.loading_blocks[lod]
            .try_reserve(loading_insertions)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        self.event_outbox
            .try_reserve(removed_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        prepared_removal_saves.sort_unstable_by_key(|save| {
            (
                save.location.lod_index,
                save.location.position.x,
                save.location.position.y,
                save.location.position.z,
            )
        });
        let vacant_save_count = prepared_removal_saves
            .iter()
            .filter(|save| {
                !self.save_journal.contains_key(&SaveKey::new(
                    save.location.position,
                    save.location.lod_index,
                ))
            })
            .count();
        self.save_journal
            .try_reserve(vacant_save_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for save in &prepared_removal_saves {
            let key = SaveKey::new(save.location.position, save.location.lod_index);
            let Some(entry) = self.save_journal.get_mut(&key) else {
                continue;
            };
            if entry.active.is_some() || entry.written_unflushed.is_some() {
                entry
                    .queued_newer
                    .try_reserve(1)
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            }
        }
        let mut retired_loading = Vec::new();
        retired_loading
            .try_reserve_exact(loading_updates.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut retired_clean_removed = Vec::new();
        retired_clean_removed
            .try_reserve_exact(removed_positions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;

        let mut transaction = preview
            .prepare_transaction(operations, &snapshots)
            .map_err(|error| VoxelTerrainRuntimeError::DataMutation(error.into_parts().0))?;
        let mut fence = transaction
            .commit_holding_publication_fence()
            .map_err(VoxelTerrainRuntimeError::DataMutation)?;
        let removed_blocks = fence.take_removed_blocks();

        // Publish the exact dirty payload owner while the storage publication
        // fence is still held. All journal capacity and generations were
        // prepared above, so this transfer is infallible and no observer can
        // see the resident owner disappear before its persistence successor.
        for removed in removed_blocks {
            let location = removed.location();
            let save_key = (
                location.lod_index,
                location.position.x,
                location.position.y,
                location.position.z,
            );
            let Ok(save_index) = prepared_removal_saves.binary_search_by_key(&save_key, |save| {
                (
                    save.location.lod_index,
                    save.location.position.x,
                    save.location.position.y,
                    save.location.position.z,
                )
            }) else {
                debug_assert!(!removed.block().is_modified());
                retired_clean_removed.push(removed);
                continue;
            };
            let prepared_save = prepared_removal_saves[save_index];
            let block_revision = removed.block_revision();
            let (_, block) = removed.into_parts();
            let payload = block
                .into_voxels()
                .expect("prepared dirty removal retained its exact voxel payload");
            #[cfg(test)]
            self.record_fixed_dirty_owner_for_test(location, &payload);
            let pending = PendingSave {
                meta: SaveAttemptMeta {
                    block_revision,
                    generation: prepared_save.save_generation,
                    retry_count: 0,
                    last_error: None,
                },
                payload,
            };
            let key = SaveKey::new(location.position, location.lod_index);
            match self.save_journal.entry(key) {
                std::collections::hash_map::Entry::Vacant(vacant) => {
                    vacant.insert(SaveJournalEntry::new_pending(
                        block_revision,
                        prepared_save.save_generation,
                        pending.payload,
                    ));
                }
                std::collections::hash_map::Entry::Occupied(mut occupied) => {
                    let entry = occupied.get_mut();
                    if entry.active.is_none() && entry.written_unflushed.is_none() {
                        entry.active = Some(ActiveSaveAttempt::Pending(pending));
                    } else {
                        entry.queued_newer.push_back(pending);
                    }
                }
            }
        }
        if !prepared_removal_saves.is_empty() {
            self.next_save_generation = next_save_generation;
            self.automatic_checkpoint_satisfied_empty_flush = false;
        }

        for (position, next) in residency_updates {
            match next {
                Some(next) => {
                    let previous = self.loaded_data_residency[lod].insert(position, next);
                    debug_assert!(previous.is_some());
                }
                None => {
                    let previous = self.loaded_data_residency[lod].remove(&position);
                    debug_assert!(previous.is_some());
                }
            }
        }
        for (position, next) in loading_updates {
            match next {
                Some(next) => {
                    if let Some(previous) = self.loading_blocks[lod].insert(position, next) {
                        previous.cancel_physical_request_if_superseded_by(
                            self.loading_blocks[lod][&position]
                                .physical_request
                                .as_ref(),
                        );
                        retired_loading.push(previous);
                    }
                }
                None => {
                    if let Some(previous) = self.loading_blocks[lod].remove(&position) {
                        previous.cancel_physical_request_if_superseded_by(None);
                        retired_loading.push(previous);
                    }
                }
            }
        }
        let retired_pending_load =
            std::mem::replace(&mut self.blocks_pending_load[lod], next_pending_load);
        self.next_request_generation = next_request_generation;
        self.stats = next_stats;
        for position in &removed_positions {
            self.event_outbox
                .push_back(VoxelTerrainEvent::DataBlockUnloaded(BlockLocation {
                    position: *position,
                    lod_index,
                }));
        }
        #[cfg(test)]
        if let Some(pause) = &self.fixed_commit_pause_for_test {
            pause.commit_marker.store(true, Ordering::SeqCst);
            pause.pause_if_target(FixedCommitPausePhase::AfterTerrainPublishBeforeFenceFinish);
        }
        let outcome = fence.finish();
        for key in save_dispatch_keys {
            if self.lod_count > 1 && !self.shutdown_in_progress {
                self.dispatch_queued_save(key);
            }
        }
        drop((
            outcome,
            retired_loading,
            retired_pending_load,
            retired_clean_removed,
            transaction,
        ));
        Ok(())
    }

    #[cfg(test)]
    fn enqueue_data_save(&mut self, save: BlockToSave) {
        if let Err(failure) = self.try_enqueue_data_save(save) {
            let (error, save) = *failure;
            self.retained_save_admission_failures
                .push_back(RetainedSaveAdmissionFailure { error, save });
        }
    }

    #[cfg(test)]
    fn try_enqueue_data_save(
        &mut self,
        save: BlockToSave,
    ) -> Result<PersistenceOperation, Box<(VoxelTerrainRuntimeError, BlockToSave)>> {
        let BlockToSave {
            voxels,
            position,
            lod_index,
            block_revision,
        } = save;
        let Some(payload) = voxels else {
            return Err(Box::new((
                VoxelTerrainRuntimeError::MissingSavePayload,
                BlockToSave {
                    voxels: None,
                    position,
                    lod_index,
                    block_revision,
                },
            )));
        };
        let generation = match allocate_persistence_generation(&mut self.next_save_generation) {
            Ok(generation) => generation,
            Err(error) => {
                return Err(Box::new((
                    error,
                    BlockToSave {
                        voxels: Some(payload),
                        position,
                        lod_index,
                        block_revision,
                    },
                )));
            }
        };
        let key = SaveKey::new(position, lod_index);
        self.automatic_checkpoint_satisfied_empty_flush = false;
        let pending = PendingSave {
            meta: SaveAttemptMeta {
                block_revision,
                generation,
                retry_count: 0,
                last_error: None,
            },
            payload,
        };
        match self.save_journal.entry(key) {
            std::collections::hash_map::Entry::Vacant(vacant) => {
                vacant.insert(SaveJournalEntry::new_pending(
                    block_revision,
                    generation,
                    pending.payload,
                ));
            }
            std::collections::hash_map::Entry::Occupied(mut occupied) => {
                let entry = occupied.get_mut();
                if entry.active.is_none() && entry.written_unflushed.is_none() {
                    entry.active = Some(ActiveSaveAttempt::Pending(pending));
                } else {
                    entry.queued_newer.push_back(pending);
                }
            }
        }
        if self.lod_count > 1 && !self.shutdown_in_progress {
            self.dispatch_queued_save(key);
        }
        Ok(PersistenceOperation::Save {
            location: BlockLocation {
                position,
                lod_index,
            },
            block_revision,
            save_generation: generation,
        })
    }

    fn dispatch_queued_save(&mut self, key: SaveKey) {
        if self.save_checkpoint_in_flight.is_some() || self.automatic_save_checkpoint_blocked {
            return;
        }
        let Some(entry) = self.save_journal.get(&key) else {
            return;
        };
        let Some(ActiveSaveAttempt::Pending(pending)) = entry.active.as_ref() else {
            return;
        };
        if entry.written_unflushed.is_some()
            || (!self.shutdown_in_progress
                && pending.meta.retry_count >= MAX_AUTOMATIC_SAVE_ATTEMPTS)
        {
            return;
        }
        let attempt_ordinal = self.next_persistence_attempt_ordinal;
        let Some(next_attempt_ordinal) = attempt_ordinal.checked_add(1) else {
            self.save_dispatch_error = Some(VoxelTerrainRuntimeError::PersistenceAttemptOverflow {
                operation: PersistenceOperation::Save {
                    location: BlockLocation {
                        position: key.position,
                        lod_index: key.lod_index,
                    },
                    block_revision: pending.meta.block_revision,
                    save_generation: pending.meta.generation,
                },
            });
            return;
        };
        self.next_persistence_attempt_ordinal = next_attempt_ordinal;
        let entry = self
            .save_journal
            .get_mut(&key)
            .expect("save journal entry was checked before attempt allocation");
        let Some(ActiveSaveAttempt::Pending(pending)) = entry.active.take() else {
            unreachable!("save active state changed without releasing terrain ownership")
        };
        let PendingSave { meta, payload } = pending;
        let block_revision = meta.block_revision;
        let generation = meta.generation;
        let task = SaveBlockDataTask::new_voxels_with_generation_and_attempt_ordinal(
            BlockLocation {
                position: key.position,
                lod_index: key.lod_index,
            },
            payload,
            StreamingDependency::new(self.stream.clone()),
            None,
            block_revision,
            generation,
            attempt_ordinal,
        );
        #[cfg(test)]
        let mut task = task;
        #[cfg(test)]
        if std::mem::take(&mut self.panic_next_save_before_io_for_test) {
            task.set_panic_before_io_for_test(true);
        }
        entry.active = Some(ActiveSaveAttempt::WriteInFlight {
            meta,
            attempt_ordinal,
        });
        self.legacy_link_or_retain_task_batch(vec![ScheduledTask::new(
            Box::new(task),
            TaskLane::Serial,
        )]);
    }

    #[cfg(test)]
    fn dispatch_queued_saves_if_allowed(&mut self) {
        self.dispatch_queued_saves_except(&[]);
    }

    fn dispatch_queued_saves_except(&mut self, deferred_keys: &[SaveKey]) {
        if self.shutdown_in_progress
            || self.save_checkpoint_in_flight.is_some()
            || self.automatic_save_checkpoint_blocked
            || self.save_dispatch_error.is_some()
        {
            return;
        }
        let keys = self
            .save_journal
            .iter()
            .filter_map(|(key, entry)| {
                if deferred_keys.contains(key) {
                    return None;
                }
                matches!(
                    &entry.active,
                    Some(ActiveSaveAttempt::Pending(pending))
                        if pending.meta.retry_count < MAX_AUTOMATIC_SAVE_ATTEMPTS
                            && entry.written_unflushed.is_none()
                )
                .then_some(*key)
            })
            .collect::<Vec<_>>();
        for key in keys {
            self.dispatch_queued_save(key);
        }
    }

    fn legacy_link_or_retain_task_batch(&mut self, mut tasks: Vec<ScheduledTask>) {
        let mut combined = std::mem::take(&mut self.legacy_task_admission_retry);
        if combined.is_empty() {
            combined = tasks;
        } else {
            combined.append(&mut tasks);
        }
        if combined.is_empty() {
            return;
        }
        let prepared = match self.task_runner.try_prepare_enqueue(combined.len()) {
            Ok(prepared) => prepared,
            Err(_) => {
                self.legacy_task_admission_retry = combined;
                return;
            }
        };
        let filled = match prepared.try_fill_exact(combined) {
            Ok(filled) => filled,
            Err((_prepared, tasks)) => {
                self.legacy_task_admission_retry = tasks;
                return;
            }
        };
        let wake = self.task_runner.link_prepared(filled);
        wake.wake();
    }

    #[allow(dead_code)] // retained as a unit-test harness after the planner cutover
    fn try_legacy_view_mesh_block(
        &mut self,
        bpos: Vector3i,
        lod: usize,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        let lod_index = u8::try_from(lod)
            .map_err(|_| VoxelTerrainRuntimeError::LodMath(LodMathError::InvalidLodCount))?;
        let mesh_map = self
            .mesh_maps
            .get(lod)
            .ok_or(VoxelTerrainRuntimeError::LodMath(
                LodMathError::InvalidLodCount,
            ))?;
        let location = MeshBlockLocation::new(bpos, lod_index);
        let mut next = mesh_map.get(&bpos).map_or_else(
            || MeshBlockEntry {
                position: bpos,
                ..MeshBlockEntry::default()
            },
            MeshBlockEntry::clone_for_draft,
        );
        next.checked_apply_ref_delta(location, MeshRefField::ResidentViewers, 1)?;
        next.checked_apply_ref_delta(location, MeshRefField::VisualViewers, 1)?;
        let mut next_mesh_revision = self.next_mesh_revision;
        let mut next_request_generation = self.next_request_generation;
        let append_pending = self.prepare_legacy_mesh_schedule_entry(
            bpos,
            lod,
            &mut next,
            true,
            &mut next_mesh_revision,
            &mut next_request_generation,
        )?;
        if append_pending {
            self.blocks_pending_update[lod]
                .try_reserve(1)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        }
        let mesh_map = &mut self.mesh_maps[lod];
        if mesh_map.contains_key(&bpos) {
            let previous = std::mem::replace(
                mesh_map
                    .get_mut(&bpos)
                    .expect("checked variable mesh entry remains present"),
                next,
            );
            previous.cancel_physical_request_if_superseded_by(
                mesh_map[&bpos].physical_request.as_ref(),
            );
        } else {
            mesh_map
                .try_reserve(1)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            mesh_map.insert(bpos, next);
        }
        if append_pending {
            self.blocks_pending_update[lod].push(bpos);
        }
        self.next_mesh_revision = next_mesh_revision;
        self.next_request_generation = next_request_generation;
        Ok(())
    }

    #[cfg(test)]
    fn legacy_view_mesh_block(&mut self, bpos: Vector3i, lod: usize) {
        self.try_legacy_view_mesh_block(bpos, lod).unwrap();
    }

    #[allow(dead_code)] // retained as a unit-test harness after the planner cutover
    fn try_legacy_unview_mesh_block(
        &mut self,
        bpos: Vector3i,
        lod: usize,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        let lod_index = u8::try_from(lod)
            .map_err(|_| VoxelTerrainRuntimeError::LodMath(LodMathError::InvalidLodCount))?;
        let mesh_map = self
            .mesh_maps
            .get_mut(lod)
            .ok_or(VoxelTerrainRuntimeError::LodMath(
                LodMathError::InvalidLodCount,
            ))?;
        let Some(entry) = mesh_map.get(&bpos) else {
            return Ok(());
        };
        let location = MeshBlockLocation::new(bpos, lod_index);
        let mut next = entry.clone_for_draft();
        next.checked_apply_ref_delta(location, MeshRefField::ResidentViewers, -1)?;
        next.checked_apply_ref_delta(location, MeshRefField::VisualViewers, -1)?;
        if next.resident_refcount() == 0 {
            let had_applied_revision = next.applied_revision.is_some();
            if had_applied_revision {
                self.event_outbox
                    .try_reserve(1)
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            }
            let retired = mesh_map.remove(&bpos).map(|mut removed| {
                removed.cancel_physical_request_if_superseded_by(None);
                removed.accepted_upload.take()
            });
            self.blocks_pending_update[lod].retain(|p| *p != bpos);
            if had_applied_revision {
                self.event_outbox
                    .push_back(VoxelTerrainEvent::MeshBlockExited(location));
            }
            drop(retired);
        } else {
            *mesh_map
                .get_mut(&bpos)
                .expect("checked variable mesh entry remains present") = next;
        }
        Ok(())
    }

    #[cfg(test)]
    fn legacy_unview_mesh_block(&mut self, bpos: Vector3i, lod: usize) {
        self.try_legacy_unview_mesh_block(bpos, lod).unwrap();
    }

    /// Request a fresh mesh revision for a viewed mesh block. Multiple requests
    /// coalesce into one pending location while retaining only the newest key.
    #[cfg(test)]
    fn try_request_mesh_update(
        &mut self,
        bpos: Vector3i,
        lod: usize,
    ) -> Result<Option<MeshBlockKey>, VoxelTerrainRuntimeError> {
        if self.shutdown_epoch.is_some() {
            return Ok(None);
        }
        let Some(mesh_map) = self.mesh_maps.get(lod) else {
            return Err(VoxelTerrainRuntimeError::LodMath(
                LodMathError::InvalidLodCount,
            ));
        };
        let Some(entry) = mesh_map.get(&bpos) else {
            return Ok(None);
        };
        if entry.resident_refcount() == 0 {
            return Ok(None);
        }

        let revision = self.next_mesh_revision;
        let next_mesh_revision = revision
            .checked_add(1)
            .ok_or(VoxelTerrainRuntimeError::MeshRevisionOverflow)?;
        let request_generation = self.next_request_generation;
        let next_request_generation = request_generation
            .checked_add(1)
            .ok_or(VoxelTerrainRuntimeError::RequestGenerationOverflow)?;
        let key = MeshBlockKey {
            location: MeshBlockLocation::new(
                bpos,
                u8::try_from(lod).map_err(|_| {
                    VoxelTerrainRuntimeError::LodMath(LodMathError::InvalidLodCount)
                })?,
            ),
            revision,
        };
        if !entry.is_in_update_list {
            self.blocks_pending_update[lod]
                .try_reserve(1)
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        }
        self.next_mesh_revision = next_mesh_revision;
        self.next_request_generation = next_request_generation;
        let entry = self.mesh_maps[lod]
            .get_mut(&bpos)
            .expect("mesh block exists after the early return");
        entry.request_generation = request_generation;
        let retired_request =
            entry
                .physical_request
                .replace(PhysicalRequest::new(TaskRequestTag::new(
                    self.request_epoch,
                    request_generation,
                )));
        entry.requested_revision = Some(key.revision);
        entry.terminal_retry_count = 0;
        if !entry.is_in_update_list {
            entry.is_in_update_list = true;
            self.blocks_pending_update[lod].push(bpos);
        }
        if let Some(retired_request) = retired_request {
            retired_request.cancel();
        }
        Ok(Some(key))
    }

    #[cfg(test)]
    pub(crate) fn request_mesh_update(
        &mut self,
        bpos: Vector3i,
        lod: usize,
    ) -> Option<MeshBlockKey> {
        self.try_request_mesh_update(bpos, lod)
            .expect("test mesh request uses checked counters")
    }

    fn prepare_legacy_mesh_schedule_entry(
        &self,
        bpos: Vector3i,
        lod: usize,
        next: &mut MeshBlockEntry,
        require_data_ready: bool,
        next_mesh_revision: &mut u64,
        next_request_generation: &mut u64,
    ) -> Result<bool, VoxelTerrainRuntimeError> {
        if next.resident_refcount() == 0 {
            return Ok(false);
        }
        let lod_index = u8::try_from(lod)
            .map_err(|_| VoxelTerrainRuntimeError::LodMath(LodMathError::InvalidLodCount))?;
        if require_data_ready {
            // Readiness diagnostics are owned by the checked meshing pass.
            // Viewer admission still records demand when boundary math cannot
            // be evaluated yet; only a ready block enters generation-bearing
            // scheduling here.
            match self.meshing_data_is_ready(MeshBlockLocation::new(bpos, lod_index)) {
                Ok(true) => {}
                Ok(false) | Err(_) => return Ok(false),
            }
        }

        let has_pending_revision = next.requested_revision != next.applied_revision;
        let physical_is_current = next.physical_request.as_ref().is_some_and(|request| {
            request.tag.request_epoch == self.request_epoch
                && request.tag.request_generation == next.request_generation
                && !request.cancellation.is_cancelled()
        });
        if !has_pending_revision {
            let revision = *next_mesh_revision;
            *next_mesh_revision = revision
                .checked_add(1)
                .ok_or(VoxelTerrainRuntimeError::MeshRevisionOverflow)?;
            let (generation, physical_request) =
                allocate_physical_request(self.request_epoch, next_request_generation)?;
            next.requested_revision = Some(revision);
            next.request_generation = generation;
            next.physical_request = Some(physical_request);
            next.terminal_retry_count = 0;
        } else if !physical_is_current {
            let (generation, physical_request) =
                allocate_physical_request(self.request_epoch, next_request_generation)?;
            next.request_generation = generation;
            next.physical_request = Some(physical_request);
            next.terminal_retry_count = 0;
        }
        let append_pending = !next.is_in_update_list;
        next.is_in_update_list = true;
        Ok(append_pending)
    }

    #[cfg(test)]
    fn try_schedule_mesh_update_from_data(
        &mut self,
        voxel_box: Box3i,
        lod: usize,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        let padded = voxel_box.padded(1);
        let data_block_size = self.data_block_size();
        let mesh_box = padded.downscaled(data_block_size << lod);
        let candidate_count = usize::try_from(mesh_box.size.volume_u64())
            .map_err(|_| VoxelTerrainRuntimeError::TaskCountOverflow)?;
        let mut updates = Vec::new();
        updates
            .try_reserve_exact(candidate_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut additions = Vec::new();
        additions
            .try_reserve_exact(candidate_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut retired_entries = Vec::new();
        retired_entries
            .try_reserve_exact(candidate_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mut next_mesh_revision = self.next_mesh_revision;
        let mut next_request_generation = self.next_request_generation;
        for bpos in mesh_box.iter_cells_zxy() {
            let Some(current) = self
                .mesh_maps
                .get(lod)
                .and_then(|mesh_map| mesh_map.get(&bpos))
            else {
                continue;
            };
            if current.resident_refcount() == 0 {
                continue;
            }
            let mut next = current.clone_for_draft();
            let append_pending = self.prepare_legacy_mesh_schedule_entry(
                bpos,
                lod,
                &mut next,
                true,
                &mut next_mesh_revision,
                &mut next_request_generation,
            )?;
            updates.push((bpos, next));
            if append_pending {
                additions.push(bpos);
            }
        }
        self.blocks_pending_update[lod]
            .try_reserve(additions.len())
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for (bpos, next) in updates {
            let resident = self.mesh_maps[lod]
                .get_mut(&bpos)
                .expect("prepared data mesh schedule entry remains resident");
            let retired = std::mem::replace(resident, next);
            retired.cancel_physical_request_if_superseded_by(
                self.mesh_maps[lod][&bpos].physical_request.as_ref(),
            );
            retired_entries.push(retired);
        }
        self.blocks_pending_update[lod].extend(additions);
        self.next_mesh_revision = next_mesh_revision;
        self.next_request_generation = next_request_generation;
        drop(retired_entries);
        Ok(())
    }

    #[allow(dead_code)] // retained as a unit-test harness after the planner cutover
    fn send_data_load_requests(&mut self) {
        let data = self.data.clone();
        let stream = self.stream.clone();
        let mut all_tasks = Vec::new();
        for lod in 0..self.lod_count as usize {
            if self.blocks_pending_load[lod].is_empty() {
                continue;
            }
            let positions = std::mem::take(&mut self.blocks_pending_load[lod]);
            for bpos in positions {
                let Some(entry) = self.loading_blocks[lod].get_mut(&bpos) else {
                    continue;
                };
                if entry.request_state != LoadRequestState::Queued {
                    continue;
                }
                entry.request_state = LoadRequestState::InFlight;
                let request_generation = entry.request_generation;
                let request_control = Some(
                    entry
                        .physical_request
                        .get_or_insert_with(|| {
                            PhysicalRequest::new(TaskRequestTag::new(
                                self.request_epoch,
                                request_generation,
                            ))
                        })
                        .clone(),
                );
                let mut task = LoadBlockForTerrainTask::new(
                    bpos,
                    lod as u8,
                    request_generation,
                    data.clone(),
                    stream.clone(),
                );
                if let Some(request) = request_control {
                    task = task.with_request_control(request.tag, request.cancellation);
                }
                all_tasks.push(ScheduledTask::new(Box::new(task), TaskLane::Parallel));
            }
        }
        if !all_tasks.is_empty() || !self.legacy_task_admission_retry.is_empty() {
            self.legacy_link_or_retain_task_batch(all_tasks);
        }
    }

    fn drain_completed_tasks(&mut self) -> Result<(), VoxelTerrainRuntimeError> {
        if self.lod_count == 1 {
            return self.try_run_fixed_transaction(&[], false, false);
        }
        self.validate_meshing_bounds()?;
        let mut previous_pending = None;
        loop {
            self.try_drain_completion_work()?;
            let pending = self
                .raw_completion_inbox
                .len()
                .saturating_add(self.durable_completion_inbox.len())
                .saturating_add(self.direct_mesh_retry_inbox.len());
            if pending == 0 {
                return Ok(());
            }
            if previous_pending.is_some_and(|previous| pending >= previous) {
                return Err(VoxelTerrainRuntimeError::CompletionDrainStalled);
            }
            previous_pending = Some(pending);
        }
    }

    /// Move raw runner completions through a durable terrain-owned FIFO before
    /// touching live request state. Both queue reservations happen while the
    /// source still owns the exact task Box and payload. A checked failure can
    /// therefore be retried without reconstructing task identity or repeating
    /// external work.
    fn try_drain_completion_work(&mut self) -> Result<(), VoxelTerrainRuntimeError> {
        self.task_runner
            .try_drain_completed_into(&mut self.raw_completion_inbox)
            .map_err(CompletionDrainError::from)?;
        self.try_normalize_raw_completions()?;
        let event_count_before_apply = self.event_outbox.len();
        self.legacy_variable_apply_durable_fifo()?;
        let deferred_keys = std::mem::take(&mut self.deferred_save_dispatch_keys);
        // Recovered keys remain PendingWrite for the rest of this drain. Other
        // keys are still eligible, and a later scheduler pass can allocate a
        // fresh physical attempt for each deferred key.
        self.dispatch_queued_saves_except(&deferred_keys);
        if !std::mem::take(&mut self.deferred_checkpoint_dispatch) {
            self.dispatch_pending_checkpoint();
        }
        // Direct uploads commit last, after every other checked reservation in
        // this drain. Once a durable completion has published an event, leave
        // direct ownership queued so a later reservation failure cannot make
        // the already-published event disappear from this checked call.
        if self.event_outbox.len() == event_count_before_apply {
            self.legacy_variable_apply_direct_fifo()?;
        }
        Ok(())
    }

    fn legacy_variable_apply_direct_fifo(&mut self) -> Result<(), VoxelTerrainRuntimeError> {
        let event_count = self
            .direct_mesh_retry_inbox
            .iter()
            .filter(|completion| match completion {
                DurableCompletion::DirectMesh { upload, dropped } => {
                    self.mesh_upload_will_publish(upload, *dropped)
                }
                _ => unreachable!("direct mesh retry inbox contains only direct mesh ownership"),
            })
            .count();
        self.try_reserve_mesh_events(event_count)
            .map_err(VoxelTerrainRuntimeError::from)?;
        while self.direct_mesh_retry_inbox.front().is_some() {
            let direct = self
                .direct_mesh_retry_inbox
                .pop_front()
                .expect("direct mesh front was checked above");
            let DurableCompletion::DirectMesh { upload, dropped } = direct else {
                unreachable!("direct mesh retry inbox contains only direct mesh ownership")
            };
            if let Err(error) =
                self.legacy_variable_apply_mesh_upload(Arc::clone(&upload), dropped, None, false)
            {
                self.direct_mesh_retry_inbox
                    .push_front(DurableCompletion::DirectMesh { upload, dropped });
                return Err(error);
            }
        }
        Ok(())
    }

    fn mesh_upload_will_publish(&self, upload: &MeshUploadSnapshot, dropped: bool) -> bool {
        if dropped {
            return false;
        }
        let key = upload.key();
        let lod = usize::from(key.location.lod_index);
        self.mesh_maps.get(lod).is_some_and(|mesh_map| {
            mesh_map
                .get(&key.location.position_in_blocks)
                .is_some_and(|entry| entry.requested_revision == Some(key.revision))
        })
    }

    fn try_reserve_mesh_events(&mut self, additional: usize) -> Result<(), CompletionDrainError> {
        if additional == 0 {
            return Ok(());
        }
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_mesh_event_reservation_for_test) {
            return Err(CompletionDrainError::InjectedMeshEventReservationFailure);
        }
        self.event_outbox.try_reserve(additional)?;
        Ok(())
    }

    fn try_normalize_raw_completions(&mut self) -> Result<(), CompletionDrainError> {
        let sampled_count = self.raw_completion_inbox.len();
        if sampled_count == 0 {
            return Ok(());
        }
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_completion_normalization_for_test) {
            return Err(CompletionDrainError::InjectedNormalizationFailure);
        }
        // Reserve the complete sampled destination before moving the first
        // raw owner. Classification below is read-only; once this succeeds,
        // every conversion is an infallible move into durable ownership.
        self.durable_completion_inbox.try_reserve(sampled_count)?;
        for front in self.raw_completion_inbox.iter().take(sampled_count) {
            let kind = completion_task_kind(front);
            let _ = finished_completion_has_output(front, kind);
        }

        for _ in 0..sampled_count {
            let front = self
                .raw_completion_inbox
                .front()
                .expect("sampled raw prefix remains owned until normalization");

            let kind = completion_task_kind(front);
            let malformed_finished = front.status() == TaskCompletionStatus::Finished
                && !finished_completion_has_output(front, kind)
                && kind != CompletionTaskKind::Unknown;
            let has_persistence_terminal = match kind {
                CompletionTaskKind::Save => front
                    .task_any()
                    .downcast_ref::<SaveBlockDataTask>()
                    .is_some_and(|task| task.terminal_ref().is_some()),
                CompletionTaskKind::Flush => front
                    .task_any()
                    .downcast_ref::<FlushVoxelStreamTask>()
                    .is_some_and(|task| task.terminal_ref().is_some()),
                _ => false,
            };

            let mut completed = self
                .raw_completion_inbox
                .pop_front()
                .expect("raw completion front was checked above");
            let status = completed.status();
            let durable = match (kind, status) {
                (CompletionTaskKind::Load, TaskCompletionStatus::Finished)
                    if !malformed_finished =>
                {
                    let output = try_take_load_output(completed.task_mut())
                        .expect("finished load output was validated before raw ownership moved");
                    DurableCompletion::LoadFinished { completed, output }
                }
                (CompletionTaskKind::Load, TaskCompletionStatus::Finished) => {
                    DurableCompletion::MalformedFinished { completed, kind }
                }
                (CompletionTaskKind::Load, _) => {
                    let task = completed
                        .task_any()
                        .downcast_ref::<LoadBlockForTerrainTask>()
                        .expect("completion kind was derived from this exact task Box");
                    DurableCompletion::LoadTerminal {
                        position: task.position,
                        lod_index: task.lod_index,
                        request_generation: task.request_generation,
                        request_tag: task.request_tag(),
                        completed,
                    }
                }
                (CompletionTaskKind::Mesh, TaskCompletionStatus::Finished)
                    if !malformed_finished =>
                {
                    let output = try_take_mesh_output(completed.task_mut())
                        .expect("finished mesh output was validated before raw ownership moved");
                    DurableCompletion::MeshFinished { completed, output }
                }
                (CompletionTaskKind::Mesh, TaskCompletionStatus::Finished) => {
                    DurableCompletion::MalformedFinished { completed, kind }
                }
                (CompletionTaskKind::Mesh, _) => {
                    let task = completed
                        .task_any()
                        .downcast_ref::<MeshBlockTask>()
                        .expect("completion kind was derived from this exact task Box");
                    DurableCompletion::MeshTerminal {
                        key: task.key(),
                        request_tag: task.request_tag(),
                        completed,
                    }
                }
                (CompletionTaskKind::Save, TaskCompletionStatus::Finished)
                    if !malformed_finished =>
                {
                    let task = completed
                        .task_any_mut()
                        .downcast_mut::<SaveBlockDataTask>()
                        .expect("completion kind was derived from this exact task Box");
                    let attempt_ordinal = task.physical_attempt_ordinal();
                    let mut terminal = task
                        .take_terminal()
                        .expect("finished save terminal was validated before ownership moved");
                    terminal.task_panic_phase = None;
                    DurableCompletion::SaveAcknowledged {
                        completed,
                        terminal,
                        attempt_ordinal,
                    }
                }
                (CompletionTaskKind::Save, TaskCompletionStatus::Panicked(_))
                    if has_persistence_terminal =>
                {
                    let task = completed
                        .task_any_mut()
                        .downcast_mut::<SaveBlockDataTask>()
                        .expect("completion kind was derived from this exact task Box");
                    let attempt_ordinal = task.physical_attempt_ordinal();
                    let mut terminal = task
                        .take_terminal()
                        .expect("persistence terminal presence was checked before moving");
                    terminal.task_panic_phase = completion_panic_phase(status);
                    DurableCompletion::PersistenceTerminal {
                        completed,
                        kind: PersistenceTaskKind::Save,
                        terminal: PersistenceTaskTerminal::Save(terminal),
                        attempt_ordinal,
                    }
                }
                (CompletionTaskKind::Save, _) if has_persistence_terminal => {
                    let task = completed
                        .task_any_mut()
                        .downcast_mut::<SaveBlockDataTask>()
                        .expect("completion kind was derived from this exact task Box");
                    let attempt_ordinal = task.physical_attempt_ordinal();
                    let terminal = task
                        .take_terminal()
                        .expect("persistence terminal presence was checked before moving");
                    DurableCompletion::MalformedPersistence {
                        completed,
                        kind: PersistenceTaskKind::Save,
                        terminal: PersistenceTaskTerminal::Save(terminal),
                        attempt_ordinal,
                    }
                }
                (CompletionTaskKind::Save, _) => {
                    DurableCompletion::MalformedFinished { completed, kind }
                }
                (CompletionTaskKind::Flush, TaskCompletionStatus::Finished)
                    if !malformed_finished =>
                {
                    let task = completed
                        .task_any_mut()
                        .downcast_mut::<FlushVoxelStreamTask>()
                        .expect("completion kind was derived from this exact task Box");
                    let attempt_ordinal = task.physical_attempt_ordinal();
                    let mut terminal = task
                        .take_terminal()
                        .expect("finished flush terminal was validated before ownership moved");
                    terminal.task_panic_phase = None;
                    DurableCompletion::FlushAcknowledged {
                        completed,
                        terminal,
                        attempt_ordinal,
                    }
                }
                (CompletionTaskKind::Flush, TaskCompletionStatus::Panicked(_))
                    if has_persistence_terminal =>
                {
                    let task = completed
                        .task_any_mut()
                        .downcast_mut::<FlushVoxelStreamTask>()
                        .expect("completion kind was derived from this exact task Box");
                    let attempt_ordinal = task.physical_attempt_ordinal();
                    let mut terminal = task
                        .take_terminal()
                        .expect("persistence terminal presence was checked before moving");
                    terminal.task_panic_phase = completion_panic_phase(status);
                    DurableCompletion::PersistenceTerminal {
                        completed,
                        kind: PersistenceTaskKind::Flush,
                        terminal: PersistenceTaskTerminal::Flush(terminal),
                        attempt_ordinal,
                    }
                }
                (CompletionTaskKind::Flush, _) if has_persistence_terminal => {
                    let task = completed
                        .task_any_mut()
                        .downcast_mut::<FlushVoxelStreamTask>()
                        .expect("completion kind was derived from this exact task Box");
                    let attempt_ordinal = task.physical_attempt_ordinal();
                    let terminal = task
                        .take_terminal()
                        .expect("persistence terminal presence was checked before moving");
                    DurableCompletion::MalformedPersistence {
                        completed,
                        kind: PersistenceTaskKind::Flush,
                        terminal: PersistenceTaskTerminal::Flush(terminal),
                        attempt_ordinal,
                    }
                }
                (CompletionTaskKind::Flush, _) => {
                    DurableCompletion::MalformedFinished { completed, kind }
                }
                (CompletionTaskKind::Unknown, _) => {
                    DurableCompletion::UnknownTerminal { completed }
                }
            };
            self.durable_completion_inbox.push_back(durable);
        }
        Ok(())
    }

    fn legacy_variable_apply_durable_fifo(&mut self) -> Result<(), VoxelTerrainRuntimeError> {
        // One durable completion can publish at most one terrain event. Reserve
        // the whole batch before applying its first state transition so the
        // ordinary no-follow-up load/mesh path can drain at full throughput
        // without introducing a fallible allocation after publication.
        let completion_count = self.durable_completion_inbox.len();
        self.event_outbox
            .try_reserve(completion_count)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        let mesh_event_count = self
            .durable_completion_inbox
            .iter()
            .filter(|completion| match completion {
                DurableCompletion::MeshFinished { output, .. } => {
                    self.mesh_upload_will_publish(output.upload(), output.dropped())
                }
                _ => false,
            })
            .count();
        self.try_reserve_mesh_events(mesh_event_count)
            .map_err(VoxelTerrainRuntimeError::from)?;

        let event_count_before_apply = self.event_outbox.len();
        while let Some(front) = self.durable_completion_inbox.front() {
            let load_request = match front {
                DurableCompletion::LoadTerminal {
                    position,
                    lod_index,
                    request_generation,
                    request_tag,
                    ..
                } => Some((*position, *lod_index, *request_generation, *request_tag)),
                _ => None,
            };
            let mesh_key = match front {
                DurableCompletion::MeshTerminal {
                    key, request_tag, ..
                } => Some((*key, *request_tag)),
                _ => None,
            };
            let is_persistence_terminal = matches!(
                front,
                DurableCompletion::PersistenceTerminal { .. }
                    | DurableCompletion::MalformedPersistence { .. }
            );
            let is_malformed_or_unknown = matches!(
                front,
                DurableCompletion::MalformedFinished { .. }
                    | DurableCompletion::UnknownTerminal { .. }
            );
            let needs_checked_preflight = load_request.is_some()
                || mesh_key.is_some()
                || is_persistence_terminal
                || is_malformed_or_unknown;
            if self.event_outbox.len() > event_count_before_apply && needs_checked_preflight {
                break;
            }

            let prepared_load_generation =
                if let Some((position, lod, generation, request_tag)) = load_request {
                    self.prepare_load_terminal_recovery(position, lod, generation, request_tag)?
                } else {
                    PreparedLoadTerminalRecovery::Stale
                };
            let prepared_mesh_recovery = if let Some((key, request_tag)) = mesh_key {
                self.prepare_mesh_terminal_recovery(key, request_tag)?
            } else {
                PreparedMeshTerminalRecovery::Stale
            };
            if is_persistence_terminal || is_malformed_or_unknown {
                self.completion_quarantine
                    .try_reserve(1)
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            }
            let durable = self
                .durable_completion_inbox
                .pop_front()
                .expect("durable completion front was checked above");
            match durable {
                DurableCompletion::LoadFinished {
                    mut completed,
                    mut output,
                } => match self.try_legacy_variable_apply_load_response(&mut output) {
                    Ok(true) => self.legacy_variable_publish_followups(&mut completed),
                    Ok(false) => {}
                    Err(error) => {
                        self.durable_completion_inbox
                            .push_front(DurableCompletion::LoadFinished { completed, output });
                        return Err(error);
                    }
                },
                DurableCompletion::LoadTerminal {
                    completed,
                    position,
                    lod_index,
                    request_generation,
                    request_tag: _,
                } => {
                    self.apply_load_terminal_recovery(
                        position,
                        lod_index,
                        request_generation,
                        prepared_load_generation,
                    );
                    drop(completed);
                }
                DurableCompletion::MeshFinished {
                    mut completed,
                    output,
                } => {
                    let request_tag = output.request_tag();
                    let upload = output.upload().clone();
                    let dropped = output.dropped();
                    match self.legacy_variable_apply_mesh_upload(upload, dropped, request_tag, true)
                    {
                        Ok(true) => self.legacy_variable_publish_followups(&mut completed),
                        Ok(false) => {}
                        Err(error) => {
                            self.durable_completion_inbox
                                .push_front(DurableCompletion::MeshFinished { completed, output });
                            return Err(error);
                        }
                    }
                }
                DurableCompletion::DirectMesh { .. } => {
                    unreachable!("direct mesh completions use their dedicated retry inbox")
                }
                DurableCompletion::MeshTerminal {
                    completed,
                    key,
                    request_tag: _,
                } => {
                    self.apply_mesh_terminal_recovery(key, prepared_mesh_recovery);
                    drop(completed);
                }
                DurableCompletion::SaveAcknowledged {
                    mut completed,
                    terminal,
                    attempt_ordinal,
                } => {
                    if self.apply_save_response_for_attempt(terminal, attempt_ordinal) {
                        self.legacy_variable_publish_followups(&mut completed);
                    }
                }
                DurableCompletion::FlushAcknowledged {
                    mut completed,
                    terminal,
                    attempt_ordinal,
                } => {
                    if self.apply_save_checkpoint_response(terminal, attempt_ordinal) {
                        self.legacy_variable_publish_followups(&mut completed);
                    }
                }
                DurableCompletion::PersistenceTerminal {
                    completed,
                    kind,
                    terminal,
                    attempt_ordinal,
                } => match terminal {
                    PersistenceTaskTerminal::Save(terminal)
                        if terminal.phase == PersistenceIoPhase::BeforeIo
                            && terminal.acknowledgement.is_none() =>
                    {
                        if self.save_attempt_matches(&terminal, attempt_ordinal) {
                            let restored = self.restore_save_before_io(terminal, attempt_ordinal);
                            debug_assert!(restored);
                            drop(completed);
                        } else {
                            self.completion_quarantine.push_back(
                                QuarantinedCompletion::Persistence {
                                    kind,
                                    terminal: PersistenceTaskTerminal::Save(terminal),
                                    attempt_ordinal,
                                    completed,
                                },
                            );
                        }
                    }
                    PersistenceTaskTerminal::Save(terminal)
                        if terminal.phase == PersistenceIoPhase::Acknowledged
                            && matches!(
                                terminal.acknowledgement,
                                Some(PersistenceAcknowledgement::Save(_))
                            ) =>
                    {
                        if self.save_attempt_matches(&terminal, attempt_ordinal) {
                            let applied =
                                self.apply_save_response_for_attempt(terminal, attempt_ordinal);
                            debug_assert!(applied);
                            drop(completed);
                        } else {
                            self.completion_quarantine.push_back(
                                QuarantinedCompletion::Persistence {
                                    kind,
                                    terminal: PersistenceTaskTerminal::Save(terminal),
                                    attempt_ordinal,
                                    completed,
                                },
                            );
                        }
                    }
                    PersistenceTaskTerminal::Save(terminal)
                        if terminal.phase == PersistenceIoPhase::CallEntered
                            && terminal.acknowledgement.is_none() =>
                    {
                        let _ = self.mark_save_indeterminate(&terminal, attempt_ordinal);
                        self.completion_quarantine
                            .push_back(QuarantinedCompletion::Persistence {
                                kind,
                                terminal: PersistenceTaskTerminal::Save(terminal),
                                attempt_ordinal,
                                completed,
                            });
                    }
                    PersistenceTaskTerminal::Flush(terminal)
                        if terminal.phase == PersistenceIoPhase::BeforeIo
                            && terminal.acknowledgement.is_none() =>
                    {
                        if self.restore_checkpoint_before_io(&terminal, attempt_ordinal) {
                            drop(completed);
                        } else {
                            self.completion_quarantine.push_back(
                                QuarantinedCompletion::Persistence {
                                    kind,
                                    terminal: PersistenceTaskTerminal::Flush(terminal),
                                    attempt_ordinal,
                                    completed,
                                },
                            );
                        }
                    }
                    PersistenceTaskTerminal::Flush(terminal)
                        if terminal.phase == PersistenceIoPhase::Acknowledged
                            && matches!(
                                terminal.acknowledgement,
                                Some(PersistenceAcknowledgement::Flush(_))
                            ) =>
                    {
                        if self.checkpoint_attempt_matches(&terminal, attempt_ordinal) {
                            let applied =
                                self.apply_save_checkpoint_response(terminal, attempt_ordinal);
                            debug_assert!(applied);
                            drop(completed);
                        } else {
                            self.completion_quarantine.push_back(
                                QuarantinedCompletion::Persistence {
                                    kind,
                                    terminal: PersistenceTaskTerminal::Flush(terminal),
                                    attempt_ordinal,
                                    completed,
                                },
                            );
                        }
                    }
                    PersistenceTaskTerminal::Flush(terminal)
                        if terminal.phase == PersistenceIoPhase::CallEntered
                            && terminal.acknowledgement.is_none() =>
                    {
                        let _ = self.mark_checkpoint_indeterminate(&terminal, attempt_ordinal);
                        self.completion_quarantine
                            .push_back(QuarantinedCompletion::Persistence {
                                kind,
                                terminal: PersistenceTaskTerminal::Flush(terminal),
                                attempt_ordinal,
                                completed,
                            });
                    }
                    terminal => {
                        self.completion_quarantine
                            .push_back(QuarantinedCompletion::Persistence {
                                kind,
                                terminal,
                                attempt_ordinal,
                                completed,
                            });
                    }
                },
                DurableCompletion::MalformedPersistence {
                    completed,
                    kind,
                    terminal,
                    attempt_ordinal,
                } => {
                    self.completion_quarantine.push_back(
                        QuarantinedCompletion::MalformedPersistence {
                            kind,
                            terminal,
                            attempt_ordinal,
                            completed,
                        },
                    );
                }
                DurableCompletion::MalformedFinished { completed, kind } => {
                    self.completion_quarantine
                        .push_back(QuarantinedCompletion::Other { kind, completed });
                }
                DurableCompletion::UnknownTerminal { completed } => {
                    self.completion_quarantine
                        .push_back(QuarantinedCompletion::Other {
                            kind: CompletionTaskKind::Unknown,
                            completed,
                        });
                }
            }
        }
        Ok(())
    }

    fn prepare_load_terminal_recovery(
        &mut self,
        position: Vector3i,
        lod_index: u8,
        request_generation: u64,
        request_tag: Option<TaskRequestTag>,
    ) -> Result<PreparedLoadTerminalRecovery, VoxelTerrainRuntimeError> {
        let lod = usize::from(lod_index);
        let should_rearm = self.loading_blocks.get(lod).is_some_and(|loading| {
            loading.get(&position).is_some_and(|entry| {
                self.shutdown_epoch.is_none()
                    && !entry.residency.is_empty()
                    && entry.request_generation == request_generation
                    && entry.request_state == LoadRequestState::InFlight
                    && entry.matches_physical_request(request_tag)
                    && request_tag.is_some_and(|tag| tag.request_epoch == self.request_epoch)
            })
        });
        if !should_rearm {
            return Ok(PreparedLoadTerminalRecovery::Stale);
        }
        let Some(retry_count) = self.loading_blocks[lod][&position]
            .retry_count
            .checked_add(1)
        else {
            return Err(VoxelTerrainRuntimeError::LoadRetryCountOverflow {
                location: BlockLocation {
                    position,
                    lod_index,
                },
            });
        };
        if retry_count > MAX_LOAD_RETRIES {
            return Ok(PreparedLoadTerminalRecovery::Exhausted { retry_count });
        }
        let generation = self.next_request_generation;
        let next_generation = generation
            .checked_add(1)
            .ok_or(VoxelTerrainRuntimeError::RequestGenerationOverflow)?;
        self.blocks_pending_load[lod]
            .try_reserve(1)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        Ok(PreparedLoadTerminalRecovery::Rearm {
            retry_count,
            generation,
            next_generation,
        })
    }

    fn apply_load_terminal_recovery(
        &mut self,
        position: Vector3i,
        lod_index: u8,
        request_generation: u64,
        prepared: PreparedLoadTerminalRecovery,
    ) {
        let lod = usize::from(lod_index);
        if prepared == PreparedLoadTerminalRecovery::Stale {
            return;
        }
        let entry = self.loading_blocks[lod]
            .get_mut(&position)
            .expect("prepared terminal recovery retained exact load demand");
        debug_assert_eq!(entry.request_generation, request_generation);
        debug_assert_eq!(entry.request_state, LoadRequestState::InFlight);
        match prepared {
            PreparedLoadTerminalRecovery::Stale => unreachable!(),
            PreparedLoadTerminalRecovery::Exhausted { retry_count } => {
                entry.retry_count = retry_count;
                entry.request_state = LoadRequestState::Exhausted;
                entry.physical_request = None;
            }
            PreparedLoadTerminalRecovery::Rearm {
                retry_count,
                generation,
                next_generation,
            } => {
                self.next_request_generation = next_generation;
                entry.retry_count = retry_count;
                entry.request_generation = generation;
                entry.request_state = LoadRequestState::Queued;
                entry.physical_request = Some(PhysicalRequest::new(TaskRequestTag::new(
                    self.request_epoch,
                    generation,
                )));
                self.blocks_pending_load[lod].push(position);
            }
        }
    }

    fn prepare_mesh_terminal_recovery(
        &mut self,
        key: MeshBlockKey,
        request_tag: Option<TaskRequestTag>,
    ) -> Result<PreparedMeshTerminalRecovery, VoxelTerrainRuntimeError> {
        let lod = usize::from(key.location.lod_index);
        let position = key.location.position_in_blocks;
        let current = self.mesh_maps.get(lod).and_then(|mesh_map| {
            mesh_map
                .get(&position)
                .is_some_and(|entry| {
                    self.shutdown_epoch.is_none()
                        && entry.resident_refcount() > 0
                        && entry.requested_revision == Some(key.revision)
                        && !entry.is_in_update_list
                        && entry.matches_physical_request(request_tag)
                        && request_tag.is_some_and(|tag| tag.request_epoch == self.request_epoch)
                })
                .then(|| {
                    mesh_map
                        .get(&position)
                        .expect("current mesh entry was checked above")
                })
        });
        let Some(entry) = current else {
            return Ok(PreparedMeshTerminalRecovery::Stale);
        };
        let Some(retry_count) = entry.terminal_retry_count.checked_add(1) else {
            return Err(VoxelTerrainRuntimeError::MeshTerminalRetryCountOverflow { key });
        };
        if retry_count > MAX_MESH_TERMINAL_RETRIES {
            return Ok(PreparedMeshTerminalRecovery::Quiesce { retry_count });
        }
        let generation = self.next_request_generation;
        let next_generation = generation
            .checked_add(1)
            .ok_or(VoxelTerrainRuntimeError::RequestGenerationOverflow)?;
        self.blocks_pending_update[lod]
            .try_reserve(1)
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        Ok(PreparedMeshTerminalRecovery::Requeue {
            retry_count,
            generation,
            next_generation,
        })
    }

    fn apply_mesh_terminal_recovery(
        &mut self,
        key: MeshBlockKey,
        prepared: PreparedMeshTerminalRecovery,
    ) {
        if prepared == PreparedMeshTerminalRecovery::Stale {
            return;
        }
        let lod = usize::from(key.location.lod_index);
        let position = key.location.position_in_blocks;
        let entry = self.mesh_maps[lod]
            .get_mut(&position)
            .expect("prepared terminal recovery retained exact mesh demand");
        debug_assert_eq!(entry.requested_revision, Some(key.revision));
        debug_assert!(!entry.is_in_update_list);
        match prepared {
            PreparedMeshTerminalRecovery::Stale => unreachable!(),
            PreparedMeshTerminalRecovery::Quiesce { retry_count } => {
                entry.terminal_retry_count = retry_count;
                entry.physical_request = None;
            }
            PreparedMeshTerminalRecovery::Requeue {
                retry_count,
                generation,
                next_generation,
            } => {
                self.next_request_generation = next_generation;
                entry.terminal_retry_count = retry_count;
                entry.request_generation = generation;
                entry.physical_request = Some(PhysicalRequest::new(TaskRequestTag::new(
                    self.request_epoch,
                    generation,
                )));
                entry.is_in_update_list = true;
                self.blocks_pending_update[lod].push(position);
            }
        }
    }

    fn legacy_variable_publish_followups(&mut self, completed: &mut CompletedTask) {
        debug_assert_eq!(completed.status(), TaskCompletionStatus::Finished);
        let follow_up_tasks = completed.take_follow_up_tasks();
        if !follow_up_tasks.is_empty() {
            self.legacy_link_or_retain_task_batch(follow_up_tasks);
        }
    }

    #[allow(dead_code)] // retained as a unit-test harness after the planner cutover
    fn process_meshing(&mut self) -> Result<(), VoxelTerrainRuntimeError> {
        self.enqueue_ready_mesh_block_tasks()
    }

    fn enqueue_ready_mesh_block_tasks(&mut self) -> Result<(), VoxelTerrainRuntimeError> {
        let mut readiness_by_lod = Vec::new();
        readiness_by_lod
            .try_reserve_exact(usize::from(self.lod_count))
            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
        for lod in 0..usize::from(self.lod_count) {
            let mut readiness = Vec::new();
            readiness
                .try_reserve_exact(self.blocks_pending_update[lod].len())
                .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
            let lod_index = u8::try_from(lod)
                .map_err(|_| VoxelTerrainRuntimeError::LodMath(LodMathError::InvalidLodCount))?;
            for &position in &self.blocks_pending_update[lod] {
                readiness
                    .push(self.meshing_data_is_ready(MeshBlockLocation::new(position, lod_index))?);
            }
            readiness_by_lod.push(readiness);
        }

        let data = self.data.clone();
        let meshing_dependency = self.meshing_dependency.clone();
        let mesh_arrays_pool = self.mesh_arrays_pool.clone();
        let mut all_tasks = Vec::new();
        for (lod, readiness) in readiness_by_lod.iter_mut().enumerate() {
            if self.blocks_pending_update[lod].is_empty() {
                continue;
            }
            let positions = std::mem::take(&mut self.blocks_pending_update[lod]);
            let readiness = std::mem::take(readiness);
            debug_assert_eq!(positions.len(), readiness.len());
            for (bpos, data_is_ready) in positions.into_iter().zip(readiness) {
                if !data_is_ready {
                    if let Some(entry) = self.mesh_maps[lod].get_mut(&bpos) {
                        if entry.resident_refcount() > 0 && entry.requested_revision.is_some() {
                            entry.is_in_update_list = true;
                            self.blocks_pending_update[lod].push(bpos);
                        }
                    }
                    continue;
                }
                let Some(entry) = self.mesh_maps[lod].get_mut(&bpos) else {
                    continue;
                };
                entry.is_in_update_list = false;
                let key = MeshBlockKey {
                    location: MeshBlockLocation::new(bpos, lod as u8),
                    revision: entry
                        .requested_revision
                        .expect("queued mesh has a revision"),
                };
                let request_control = Some(
                    entry
                        .physical_request
                        .get_or_insert_with(|| {
                            PhysicalRequest::new(TaskRequestTag::new(
                                self.request_epoch,
                                entry.request_generation,
                            ))
                        })
                        .clone(),
                );
                let mut task = MeshBlockTask::new(MeshBlockTaskParams {
                    key,
                    data: data.clone(),
                    meshing_dependency: meshing_dependency.clone(),
                    collision_hint: entry.needs_collision(),
                    lod_hint: self.lod_count > 1,
                    mesh_arrays_pool: Some(mesh_arrays_pool.clone()),
                });
                if let Some(request) = request_control {
                    task = task.with_request_control(request.tag, request.cancellation);
                }
                all_tasks.push(ScheduledTask::new(Box::new(task), TaskLane::Parallel));
            }
        }
        if !all_tasks.is_empty() || !self.legacy_task_admission_retry.is_empty() {
            self.legacy_link_or_retain_task_batch(all_tasks);
        }
        Ok(())
    }

    fn try_legacy_variable_apply_load_response(
        &mut self,
        output: &mut TerrainLoadOutput,
    ) -> Result<bool, VoxelTerrainRuntimeError> {
        let bpos = output.block_data.position_in_blocks;
        let lod_index = output.block_data.lod_index;
        let lod = usize::from(lod_index);
        let Some(entry) = self
            .loading_blocks
            .get(lod)
            .and_then(|loading| loading.get(&bpos))
        else {
            return Ok(false);
        };
        let is_current = self.shutdown_epoch.is_none()
            && entry.request_generation == output.request_generation
            && entry.request_state == LoadRequestState::InFlight
            && entry.matches_physical_request(output.request_tag)
            && output
                .request_tag
                .is_some_and(|tag| tag.request_epoch == self.request_epoch);
        if !is_current {
            return Ok(false);
        }

        match output.block_data.kind {
            BlockDataOutputKind::Loaded | BlockDataOutputKind::NeedsGeneration => {
                if output.block_data.dropped {
                    let retry_count = entry.retry_count.checked_add(1).ok_or(
                        VoxelTerrainRuntimeError::LoadRetryCountOverflow {
                            location: BlockLocation {
                                position: bpos,
                                lod_index,
                            },
                        },
                    )?;
                    let replacement = if retry_count <= MAX_LOAD_RETRIES {
                        self.blocks_pending_load[lod]
                            .try_reserve(1)
                            .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                        Some(allocate_physical_request(
                            self.request_epoch,
                            &mut self.next_request_generation,
                        )?)
                    } else {
                        None
                    };
                    let entry = self.loading_blocks[lod].get_mut(&bpos).ok_or(
                        VoxelTerrainRuntimeError::DataResidencyMismatch {
                            location: BlockLocation {
                                position: bpos,
                                lod_index,
                            },
                            tracked_resident_viewers: None,
                            tracked_coverage_holds: None,
                            storage_viewers: 0,
                        },
                    )?;
                    entry.retry_count = retry_count;
                    if let Some((generation, physical_request)) = replacement {
                        entry.request_generation = generation;
                        entry.request_state = LoadRequestState::Queued;
                        entry.physical_request = Some(physical_request);
                        self.blocks_pending_load[lod].push(bpos);
                    } else {
                        entry.request_state = LoadRequestState::Exhausted;
                        entry.physical_request = None;
                    }
                    return Ok(true);
                }
                if output.block_data.voxels.is_none() {
                    return Ok(false);
                }
                let residency = entry.residency;
                if residency.is_empty() {
                    return Ok(false);
                }
                let location = BlockLocation {
                    position: bpos,
                    lod_index,
                };
                if let Some(tracked) = self.loaded_data_residency[lod].get(&bpos).copied() {
                    return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                        location,
                        tracked_resident_viewers: Some(tracked.resident_viewers),
                        tracked_coverage_holds: Some(tracked.coverage_holds),
                        storage_viewers: 0,
                    });
                }
                let final_viewers = residency.checked_total(location)?;
                let next_blocks_loaded = self
                    .stats
                    .blocks_loaded
                    .checked_add(1)
                    .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
                self.loaded_data_residency[lod]
                    .try_reserve(1)
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                self.event_outbox
                    .try_reserve(1)
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                let voxel_box = block_box_to_voxel_box(
                    Box3i::new(bpos, Vector3i::splat(1)),
                    self.data_block_size(),
                    lod,
                );
                let affected_mesh_box = voxel_box
                    .padded(1)
                    .downscaled(self.data_block_size() << lod);
                let affected_count = usize::try_from(affected_mesh_box.size.volume_u64())
                    .map_err(|_| VoxelTerrainRuntimeError::TaskCountOverflow)?;
                self.blocks_pending_update[lod]
                    .try_reserve(affected_count)
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                let mut mesh_updates = Vec::new();
                mesh_updates
                    .try_reserve_exact(affected_count)
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                let mut pending_mesh_additions = Vec::new();
                pending_mesh_additions
                    .try_reserve_exact(affected_count)
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                let mut retired_mesh_entries = Vec::new();
                retired_mesh_entries
                    .try_reserve_exact(affected_count)
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                let mut next_mesh_revision = self.next_mesh_revision;
                let mut next_request_generation = self.next_request_generation;
                for mesh_position in affected_mesh_box.iter_cells_zxy() {
                    let Some(current) = self.mesh_maps[lod].get(&mesh_position) else {
                        continue;
                    };
                    if current.resident_refcount() == 0 {
                        continue;
                    }
                    let mut next_entry = current.clone_for_draft();
                    let append_pending = self.prepare_legacy_mesh_schedule_entry(
                        mesh_position,
                        lod,
                        &mut next_entry,
                        false,
                        &mut next_mesh_revision,
                        &mut next_request_generation,
                    )?;
                    mesh_updates.push(PreparedVoxelEditMeshUpdate {
                        location: MeshBlockLocation::new(mesh_position, lod_index),
                        next_entry,
                    });
                    if append_pending {
                        pending_mesh_additions.push(mesh_position);
                    }
                }

                let preview = self.data.begin_transaction_preview();
                let snapshot = preview.block_snapshot(location).ok_or({
                    VoxelTerrainRuntimeError::DataMutation(
                        SharedVoxelDataMutationError::LodDestinationUnavailable {
                            position: bpos,
                            lod_index: lod,
                        },
                    )
                })?;
                if snapshot.is_present() {
                    return Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
                        location,
                        tracked_resident_viewers: None,
                        tracked_coverage_holds: None,
                        storage_viewers: snapshot.viewers(),
                    });
                }
                // This is the final ownership take. Every fallible capacity
                // reservation above succeeded, and both prepare/commit error
                // paths below restore this exact allocation into `output`.
                let voxels = output
                    .block_data
                    .voxels
                    .take()
                    .ok_or(VoxelTerrainRuntimeError::MissingSavePayload)?;
                let mut block = VoxelDataBlock::with_voxels(voxels, lod_index);
                block.set_edited(true);
                let operation = SharedVoxelDataTransactionOperation::Insert {
                    location,
                    block,
                    final_viewers,
                };
                let mut transaction =
                    match preview.prepare_transaction(vec![operation], &[snapshot]) {
                        Ok(transaction) => transaction,
                        Err(error) => {
                            let (storage_error, mut operations) = error.into_parts();
                            Self::restore_variable_load_insert_payload(
                                output,
                                &mut operations,
                                location,
                            );
                            return Err(VoxelTerrainRuntimeError::DataMutation(storage_error));
                        }
                    };
                let storage_error = match transaction.commit_holding_publication_fence() {
                    Ok(fence) => {
                        let retired_loading = self.loading_blocks[lod].remove(&bpos);
                        self.blocks_pending_load[lod].retain(|pending| *pending != bpos);
                        let replaced = self.loaded_data_residency[lod].insert(bpos, residency);
                        debug_assert!(replaced.is_none());
                        for update in mesh_updates {
                            let resident = self.mesh_maps[lod]
                                .get_mut(&update.location.position_in_blocks)
                                .expect("prepared load mesh entry remains resident");
                            let retired = std::mem::replace(resident, update.next_entry);
                            retired.cancel_physical_request_if_superseded_by(
                                self.mesh_maps[lod][&update.location.position_in_blocks]
                                    .physical_request
                                    .as_ref(),
                            );
                            retired_mesh_entries.push(retired);
                        }
                        self.blocks_pending_update[lod].extend(pending_mesh_additions);
                        self.next_mesh_revision = next_mesh_revision;
                        self.next_request_generation = next_request_generation;
                        self.stats.blocks_loaded = next_blocks_loaded;
                        self.event_outbox
                            .push_back(VoxelTerrainEvent::DataBlockLoaded(location));
                        let outcome = fence.finish();
                        drop((outcome, retired_loading, retired_mesh_entries));
                        return Ok(true);
                    }
                    Err(storage_error) => storage_error,
                };
                let mut operations = transaction
                    .into_operations()
                    .expect("failed variable load commit retains its insert payload");
                Self::restore_variable_load_insert_payload(output, &mut operations, location);
                return Err(VoxelTerrainRuntimeError::DataMutation(storage_error));
            }
            BlockDataOutputKind::NotFound => {
                let entry = self.loading_blocks[lod].get_mut(&bpos).ok_or(
                    VoxelTerrainRuntimeError::DataResidencyMismatch {
                        location: BlockLocation {
                            position: bpos,
                            lod_index,
                        },
                        tracked_resident_viewers: None,
                        tracked_coverage_holds: None,
                        storage_viewers: 0,
                    },
                )?;
                entry.request_state = LoadRequestState::NotFound;
                entry.physical_request = None;
                self.blocks_pending_load[lod].retain(|pending| *pending != bpos);
            }
            BlockDataOutputKind::Saved => return Ok(false),
        }
        Ok(true)
    }

    fn restore_variable_load_insert_payload(
        output: &mut TerrainLoadOutput,
        operations: &mut [SharedVoxelDataTransactionOperation],
        location: BlockLocation,
    ) {
        let operation = operations
            .iter_mut()
            .find(|operation| {
                matches!(
                    operation,
                    SharedVoxelDataTransactionOperation::Insert {
                        location: operation_location,
                        ..
                    } if *operation_location == location
                )
            })
            .expect("failed variable load transaction retains its exact insert");
        let SharedVoxelDataTransactionOperation::Insert { block, .. } = operation else {
            unreachable!("variable load escrow resolves only insert operations")
        };
        let voxels = std::mem::replace(block, VoxelDataBlock::empty(location.lod_index))
            .into_voxels()
            .expect("variable load insert retains its exact voxel payload");
        debug_assert!(output.block_data.voxels.is_none());
        output.block_data.voxels = Some(voxels);
    }

    #[cfg(test)]
    fn legacy_variable_apply_load_response(&mut self, mut output: TerrainLoadOutput) -> bool {
        let position = output.block_data.position_in_blocks;
        let lod = usize::from(output.block_data.lod_index);
        if let Some(entry) = self
            .loading_blocks
            .get_mut(lod)
            .and_then(|loading| loading.get_mut(&position))
            .filter(|entry| {
                entry.request_generation == output.request_generation
                    && output.request_tag.is_none_or(|tag| {
                        entry
                            .physical_request
                            .as_ref()
                            .is_some_and(|request| request.tag == tag)
                    })
            })
        {
            // Direct unit-test completions model a request that crossed
            // dispatch. Production completions are tagged by the worker task
            // itself and never enter through this compatibility helper.
            if entry.request_state == LoadRequestState::Queued {
                entry.request_state = LoadRequestState::InFlight;
                self.blocks_pending_load[lod].retain(|pending| *pending != position);
            }
        }
        if output.request_tag.is_none() {
            output.request_tag = self
                .loading_blocks
                .get(lod)
                .and_then(|loading| loading.get(&position))
                .filter(|entry| entry.request_generation == output.request_generation)
                .and_then(|entry| entry.physical_request.as_ref())
                .map(|request| request.tag);
        }
        self.try_legacy_variable_apply_load_response(&mut output)
            .unwrap_or(false)
    }

    #[cfg(test)]
    fn apply_save_response(&mut self, terminal: SaveTaskTerminal) -> bool {
        let key = SaveKey::new(terminal.location.position, terminal.location.lod_index);
        let Some(attempt_ordinal) =
            self.save_journal
                .get(&key)
                .and_then(|entry| match entry.active.as_ref()? {
                    ActiveSaveAttempt::WriteInFlight {
                        meta,
                        attempt_ordinal,
                    } if meta.generation == terminal.save_generation => Some(*attempt_ordinal),
                    _ => None,
                })
        else {
            return false;
        };
        self.apply_save_response_for_attempt(terminal, attempt_ordinal)
    }

    fn restore_save_before_io(&mut self, terminal: SaveTaskTerminal, attempt_ordinal: u64) -> bool {
        let key = SaveKey::new(terminal.location.position, terminal.location.lod_index);
        let Some(entry) = self.save_journal.get_mut(&key) else {
            return false;
        };
        let (meta, current_attempt) = match entry.active.take() {
            Some(ActiveSaveAttempt::WriteInFlight {
                meta,
                attempt_ordinal,
            }) => (meta, attempt_ordinal),
            other => {
                entry.active = other;
                return false;
            }
        };
        if meta.block_revision != terminal.block_revision
            || meta.generation != terminal.save_generation
            || current_attempt != attempt_ordinal
        {
            entry.active = Some(ActiveSaveAttempt::WriteInFlight {
                meta,
                attempt_ordinal: current_attempt,
            });
            return false;
        }
        let mut meta = meta;
        meta.retry_count = meta.retry_count.saturating_add(1);
        entry.active = Some(ActiveSaveAttempt::Pending(PendingSave {
            meta,
            payload: terminal.payload,
        }));
        if !self.deferred_save_dispatch_keys.contains(&key) {
            self.deferred_save_dispatch_keys.push(key);
        }
        true
    }

    fn save_attempt_matches(&self, terminal: &SaveTaskTerminal, attempt_ordinal: u64) -> bool {
        self.save_journal
            .get(&SaveKey::new(
                terminal.location.position,
                terminal.location.lod_index,
            ))
            .and_then(|entry| entry.active.as_ref())
            .is_some_and(|active| {
                matches!(
                    active,
                    ActiveSaveAttempt::WriteInFlight {
                        meta,
                        attempt_ordinal: current_attempt,
                    } if meta.generation == terminal.save_generation
                        && meta.block_revision == terminal.block_revision
                        && *current_attempt == attempt_ordinal
                )
            })
    }

    fn mark_save_indeterminate(
        &mut self,
        terminal: &SaveTaskTerminal,
        attempt_ordinal: u64,
    ) -> bool {
        let key = SaveKey::new(terminal.location.position, terminal.location.lod_index);
        let Some(entry) = self.save_journal.get_mut(&key) else {
            return false;
        };
        let (meta, current_attempt) = match entry.active.take() {
            Some(ActiveSaveAttempt::WriteInFlight {
                meta,
                attempt_ordinal,
            }) => (meta, attempt_ordinal),
            other => {
                entry.active = other;
                return false;
            }
        };
        if meta.block_revision != terminal.block_revision
            || meta.generation != terminal.save_generation
            || current_attempt != attempt_ordinal
        {
            entry.active = Some(ActiveSaveAttempt::WriteInFlight {
                meta,
                attempt_ordinal: current_attempt,
            });
            return false;
        }
        entry.active = Some(ActiveSaveAttempt::Indeterminate {
            meta,
            attempt_ordinal,
        });
        true
    }

    pub fn try_resolve_indeterminate_persistence(
        &mut self,
        operation: PersistenceOperation,
        resolution: IndeterminateIoResolution,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        if self.lod_count == 1 {
            return self.try_run_fixed_transaction_with_persistence_resolution(
                FixedPersistenceResolutionRequest {
                    operation,
                    resolution,
                },
            );
        }
        if matches!(operation, PersistenceOperation::Flush { .. }) {
            return self.resolve_indeterminate_checkpoint(operation, resolution);
        }
        let PersistenceOperation::Save {
            location,
            block_revision,
            save_generation,
        } = operation
        else {
            return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                requested: operation,
            });
        };
        let key = SaveKey::new(location.position, location.lod_index);
        let Some(entry) = self.save_journal.get(&key) else {
            return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                requested: operation,
            });
        };
        let Some(ActiveSaveAttempt::Indeterminate {
            meta,
            attempt_ordinal,
        }) = entry.active.as_ref()
        else {
            return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                requested: operation,
            });
        };
        if meta.block_revision != block_revision || meta.generation != save_generation {
            return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                requested: operation,
            });
        }
        let attempt_ordinal = *attempt_ordinal;
        let Some(quarantine_index) = self.completion_quarantine.iter().position(|completion| {
            matches!(
                completion,
                QuarantinedCompletion::Persistence {
                    kind: PersistenceTaskKind::Save,
                    terminal: PersistenceTaskTerminal::Save(terminal),
                    attempt_ordinal: terminal_attempt,
                    ..
                } if terminal.location == location
                    && terminal.block_revision == block_revision
                    && terminal.save_generation == save_generation
                    && *terminal_attempt == attempt_ordinal
                    && terminal.phase == PersistenceIoPhase::CallEntered
                    && terminal.acknowledgement.is_none()
            )
        }) else {
            return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                requested: operation,
            });
        };

        if resolution == IndeterminateIoResolution::AssumeNotWrittenAndRetry
            && self
                .next_persistence_attempt_ordinal
                .checked_add(1)
                .is_none()
        {
            return Err(VoxelTerrainRuntimeError::PersistenceAttemptOverflow { operation });
        }

        let quarantined = self
            .completion_quarantine
            .remove(quarantine_index)
            .expect("indeterminate quarantine index was checked before removal");
        let QuarantinedCompletion::Persistence {
            terminal: PersistenceTaskTerminal::Save(terminal),
            completed,
            ..
        } = quarantined
        else {
            unreachable!("indeterminate save quarantine variant was checked before removal")
        };
        let entry = self
            .save_journal
            .get_mut(&key)
            .expect("indeterminate save journal identity was checked before removal");
        let meta = match entry.active.take() {
            Some(ActiveSaveAttempt::Indeterminate { meta, .. }) => meta,
            other => {
                entry.active = other;
                self.completion_quarantine.insert(
                    quarantine_index,
                    QuarantinedCompletion::Persistence {
                        kind: PersistenceTaskKind::Save,
                        terminal: PersistenceTaskTerminal::Save(terminal),
                        attempt_ordinal,
                        completed,
                    },
                );
                return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                    requested: operation,
                });
            }
        };
        let force_checkpoint = match resolution {
            IndeterminateIoResolution::AssumeNotWrittenAndRetry => {
                entry.active = Some(ActiveSaveAttempt::Pending(PendingSave {
                    meta,
                    payload: terminal.payload,
                }));
                drop(completed);
                false
            }
            IndeterminateIoResolution::AssumeWrittenAndFlush => {
                if entry.written_unflushed.is_some() {
                    entry.active = Some(ActiveSaveAttempt::Indeterminate {
                        meta,
                        attempt_ordinal,
                    });
                    self.completion_quarantine.insert(
                        quarantine_index,
                        QuarantinedCompletion::Persistence {
                            kind: PersistenceTaskKind::Save,
                            terminal: PersistenceTaskTerminal::Save(terminal),
                            attempt_ordinal,
                            completed,
                        },
                    );
                    return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                        requested: operation,
                    });
                }
                entry.written_unflushed = Some(WrittenSave {
                    block_revision,
                    generation: save_generation,
                    payload: terminal.payload,
                });
                drop(completed);
                true
            }
        };
        if force_checkpoint {
            self.force_checkpoint_requested = true;
        }
        if self.lod_count == 1 {
            self.try_run_fixed_transaction(&[], false, true)?;
        } else if force_checkpoint {
            self.checkpoint_acknowledged_saves_if_needed();
        } else {
            self.dispatch_queued_save(key);
        }
        Ok(())
    }

    fn resolve_indeterminate_checkpoint(
        &mut self,
        operation: PersistenceOperation,
        resolution: IndeterminateIoResolution,
    ) -> Result<(), VoxelTerrainRuntimeError> {
        let PersistenceOperation::Flush {
            checkpoint_generation,
        } = operation
        else {
            unreachable!("checkpoint resolver only accepts flush operations")
        };
        let Some(checkpoint) = self.save_checkpoint_in_flight.as_ref() else {
            return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                requested: operation,
            });
        };
        let CheckpointAttemptState::Indeterminate { attempt_ordinal } = checkpoint.state else {
            return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                requested: operation,
            });
        };
        if checkpoint.checkpoint_generation != checkpoint_generation {
            return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                requested: operation,
            });
        }
        let Some(quarantine_index) = self.completion_quarantine.iter().position(|completion| {
            matches!(
                completion,
                QuarantinedCompletion::Persistence {
                    kind: PersistenceTaskKind::Flush,
                    terminal: PersistenceTaskTerminal::Flush(terminal),
                    attempt_ordinal: terminal_attempt,
                    ..
                } if terminal.checkpoint_generation == checkpoint_generation
                    && *terminal_attempt == attempt_ordinal
                    && terminal.phase == PersistenceIoPhase::CallEntered
                    && terminal.acknowledgement.is_none()
            )
        }) else {
            return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                requested: operation,
            });
        };
        if resolution == IndeterminateIoResolution::AssumeNotWrittenAndRetry
            && self
                .next_persistence_attempt_ordinal
                .checked_add(1)
                .is_none()
        {
            return Err(VoxelTerrainRuntimeError::PersistenceAttemptOverflow { operation });
        }

        let Some(quarantined) = self.completion_quarantine.remove(quarantine_index) else {
            return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                requested: operation,
            });
        };
        let QuarantinedCompletion::Persistence { completed, .. } = quarantined else {
            unreachable!("checkpoint quarantine variant was checked before removal")
        };
        match resolution {
            IndeterminateIoResolution::AssumeNotWrittenAndRetry => {
                let Some(checkpoint) = self.save_checkpoint_in_flight.as_mut() else {
                    return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                        requested: operation,
                    });
                };
                checkpoint.state = CheckpointAttemptState::Pending;
                drop(completed);
                if self.lod_count == 1 {
                    self.try_run_fixed_transaction(&[], false, true)?;
                } else {
                    self.dispatch_pending_checkpoint();
                }
            }
            IndeterminateIoResolution::AssumeWrittenAndFlush => {
                let Some(checkpoint) = self.save_checkpoint_in_flight.take() else {
                    return Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                        requested: operation,
                    });
                };
                drop(completed);
                self.last_checkpoint_outcome = Some((checkpoint_generation, Ok(())));
                let _ = self.apply_acknowledged_checkpoint_result(
                    checkpoint.acknowledged,
                    Ok(()),
                    checkpoint.record_per_block_failure,
                );
                self.automatic_checkpoint_satisfied_empty_flush = self.save_journal.is_empty();
                self.force_checkpoint_requested = false;
            }
        }
        Ok(())
    }

    fn apply_save_response_for_attempt(
        &mut self,
        terminal: SaveTaskTerminal,
        attempt_ordinal: u64,
    ) -> bool {
        let SaveTaskTerminal {
            location,
            block_revision,
            save_generation,
            payload,
            acknowledgement,
            ..
        } = terminal;
        let result = match acknowledgement {
            Some(PersistenceAcknowledgement::Save(result)) => result,
            _ => return false,
        };
        let key = SaveKey::new(location.position, location.lod_index);
        let Some(entry) = self.save_journal.get_mut(&key) else {
            return false;
        };
        let (meta, current_attempt) = match entry.active.take() {
            Some(ActiveSaveAttempt::WriteInFlight {
                meta,
                attempt_ordinal,
            }) => (meta, attempt_ordinal),
            other => {
                entry.active = other;
                return false;
            }
        };
        if meta.block_revision != block_revision
            || meta.generation != save_generation
            || current_attempt != attempt_ordinal
        {
            entry.active = Some(ActiveSaveAttempt::WriteInFlight {
                meta,
                attempt_ordinal: current_attempt,
            });
            return false;
        }

        match result {
            Ok(()) => {
                if entry.written_unflushed.is_some() {
                    entry.active = Some(ActiveSaveAttempt::WriteInFlight {
                        meta,
                        attempt_ordinal: current_attempt,
                    });
                    return false;
                }
                entry.written_unflushed = Some(WrittenSave {
                    block_revision,
                    generation: save_generation,
                    payload,
                });
            }
            Err(error) => {
                let mut meta = meta;
                meta.retry_count = meta.retry_count.saturating_add(1);
                meta.last_error = Some(error);
                entry.active = Some(ActiveSaveAttempt::Pending(PendingSave { meta, payload }));
                if !self.deferred_save_dispatch_keys.contains(&key) {
                    self.deferred_save_dispatch_keys.push(key);
                }
            }
        }
        true
    }

    pub fn try_apply_mesh_output(
        &mut self,
        output: BlockMeshOutput,
    ) -> Result<(), MeshOutputApplyError> {
        if self.shutdown_epoch.is_some() {
            return Err(MeshOutputApplyError::NotAdmitted {
                error: VoxelTerrainRuntimeError::ShutdownRetryPending,
                output,
            });
        }
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_direct_mesh_reservation_for_test) {
            return Err(MeshOutputApplyError::NotAdmitted {
                error: VoxelTerrainRuntimeError::MeshOutputAdmissionFailed,
                output,
            });
        }
        if self.direct_mesh_retry_inbox.try_reserve(1).is_err() {
            return Err(MeshOutputApplyError::NotAdmitted {
                error: VoxelTerrainRuntimeError::MeshOutputAdmissionFailed,
                output,
            });
        }
        let output = output.into_upload();
        let (upload, dropped) = output.into_parts();
        self.direct_mesh_retry_inbox
            .push_back(DurableCompletion::DirectMesh { upload, dropped });
        if self.lod_count == 1 {
            self.try_run_fixed_transaction(
                &[],
                false,
                self.automatic_loading_enabled && !self.shutdown_in_progress,
            )
            .map_err(|error| MeshOutputApplyError::Admitted { error })
        } else {
            self.legacy_variable_apply_direct_fifo()
                .map_err(|error| MeshOutputApplyError::Admitted { error })
        }
    }

    #[cfg(test)]
    fn apply_mesh_update_for_test(&mut self, output: BlockMeshOutput) {
        self.try_apply_mesh_output(output).unwrap();
    }

    fn legacy_variable_apply_mesh_upload(
        &mut self,
        upload: Arc<MeshUploadSnapshot>,
        dropped: bool,
        request_tag: Option<TaskRequestTag>,
        require_physical_request: bool,
    ) -> Result<bool, VoxelTerrainRuntimeError> {
        let key = upload.key();
        let bpos = key.location.position_in_blocks;
        let lod = key.location.lod_index as usize;
        let Some(mesh_map) = self.mesh_maps.get(lod) else {
            return Ok(false);
        };
        let is_current = mesh_map.get(&bpos).is_some_and(|entry| {
            entry.requested_revision == Some(key.revision)
                && self.shutdown_epoch.is_none()
                && (!require_physical_request
                    || entry.matches_physical_request(request_tag)
                        && request_tag.is_some_and(|tag| tag.request_epoch == self.request_epoch))
        });
        if !is_current {
            return Ok(false);
        }
        if dropped {
            let next_meshes_dropped = self
                .stats
                .meshes_dropped
                .checked_add(1)
                .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
            let should_requeue = self.mesh_maps[lod][&bpos].resident_refcount() > 0
                && !self.mesh_maps[lod][&bpos].is_in_update_list;
            let replacement = if should_requeue {
                self.blocks_pending_update[lod]
                    .try_reserve(1)
                    .map_err(|_| VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)?;
                Some(allocate_physical_request(
                    self.request_epoch,
                    &mut self.next_request_generation,
                )?)
            } else {
                None
            };
            let entry = self.mesh_maps[lod]
                .get_mut(&bpos)
                .ok_or(VoxelTerrainRuntimeError::MeshOutputApplyFailed)?;
            entry.terminal_retry_count = 0;
            self.stats.meshes_dropped = next_meshes_dropped;
            // MESH-1 parity: requeue the block if it's still viewed so it gets
            // re-meshed with the current dependency (not silently lost).
            if let Some((generation, physical_request)) = replacement {
                entry.request_generation = generation;
                entry.physical_request = Some(physical_request);
                entry.is_in_update_list = true;
                self.blocks_pending_update[lod].push(bpos);
            }
            return Ok(true);
        }
        let next_meshes_built = self
            .stats
            .meshes_built
            .checked_add(1)
            .ok_or(VoxelTerrainRuntimeError::StatsOverflow)?;
        let has_built_payload = upload.has_built_payload();
        let has_visual_geometry = upload.visual_state() == PayloadState::NonEmpty;
        let (had_built_payload, retired, retired_request) = {
            let entry = self.mesh_maps[lod]
                .get_mut(&bpos)
                .ok_or(VoxelTerrainRuntimeError::MeshOutputApplyFailed)?;
            entry.terminal_retry_count = 0;
            let had_built_payload = entry
                .accepted_upload
                .as_ref()
                .is_some_and(|accepted| accepted.has_built_payload());
            let retired = entry.accepted_upload.replace(upload.clone());
            let retired_request = entry.physical_request.take();
            entry.applied_revision = Some(key.revision);
            entry.applied_features = upload.features();
            entry.has_geometry = has_visual_geometry;
            entry.is_loaded = true;
            (had_built_payload, retired, retired_request)
        };
        if let Some(request) = &retired_request {
            request.cancel();
        }
        self.stats.meshes_built = next_meshes_built;
        let event = match (had_built_payload, has_built_payload) {
            (false, true) => VoxelTerrainEvent::MeshBlockEntered(upload),
            (true, true) => VoxelTerrainEvent::MeshBlockUpdated(upload),
            (_, false) => VoxelTerrainEvent::MeshBlockBecameEmpty(upload),
        };
        self.event_outbox.push_back(event);
        drop((retired, retired_request));
        Ok(true)
    }
}

#[allow(dead_code)] // retained as a unit-test harness after the planner cutover
trait BoxDiff {
    fn difference(self, other: Box3i) -> Vec<Box3i>;
}

impl BoxDiff for Box3i {
    fn difference(self, other: Box3i) -> Vec<Box3i> {
        // C++ Box3i::difference_to_vec produces up to 6 slabs. We need the
        // same here to enumerate cells in the removed region efficiently.
        // If `other` doesn't intersect `self`, the entire `self` is removed.
        if self.size.x <= 0 || self.size.y <= 0 || self.size.z <= 0 {
            return Vec::new();
        }
        if !self.intersects(&other) {
            return vec![self];
        }
        let clip = self.clipped(other);
        if clip.size == self.size {
            // `other` fully covers `self`: nothing remains.
            return Vec::new();
        }
        // Compute the up-to-6 surrounding slabs by subtracting the clipped
        // region from each face in turn. Order: -X, +X, -Y, +Y, -Z, +Z.
        let mut slabs = Vec::new();
        let self_min = self.position;
        let self_max = self.position + self.size;
        let clip_min = clip.position;
        let clip_max = clip.position + clip.size;

        // -X slab
        if clip_min.x > self_min.x {
            slabs.push(Box3i::new(
                Vector3i::new(self_min.x, self_min.y, self_min.z),
                Vector3i::new(clip_min.x - self_min.x, self.size.y, self.size.z),
            ));
        }
        // +X slab
        if self_max.x > clip_max.x {
            slabs.push(Box3i::new(
                Vector3i::new(clip_max.x, self_min.y, self_min.z),
                Vector3i::new(self_max.x - clip_max.x, self.size.y, self.size.z),
            ));
        }
        // -Y slab (X already clipped)
        if clip_min.y > self_min.y {
            slabs.push(Box3i::new(
                Vector3i::new(clip_min.x, self_min.y, self_min.z),
                Vector3i::new(clip.size.x, clip_min.y - self_min.y, self.size.z),
            ));
        }
        // +Y slab
        if self_max.y > clip_max.y {
            slabs.push(Box3i::new(
                Vector3i::new(clip_min.x, clip_max.y, self_min.z),
                Vector3i::new(clip.size.x, self_max.y - clip_max.y, self.size.z),
            ));
        }
        // -Z slab (X and Y already clipped)
        if clip_min.z > self_min.z {
            slabs.push(Box3i::new(
                Vector3i::new(clip_min.x, clip_min.y, self_min.z),
                Vector3i::new(clip.size.x, clip.size.y, clip_min.z - self_min.z),
            ));
        }
        // +Z slab
        if self_max.z > clip_max.z {
            slabs.push(Box3i::new(
                Vector3i::new(clip_min.x, clip_min.y, clip_max.z),
                Vector3i::new(clip.size.x, clip.size.y, self_max.z - clip_max.z),
            ));
        }
        slabs
    }
}

fn normalize_and_validate_viewer_updates(
    viewers: &[ViewerUpdate],
) -> Result<Vec<ViewerUpdate>, ViewerInputError> {
    let mut normalized = viewers.to_vec();
    normalized.sort_unstable_by_key(|viewer| viewer.id);
    if let Some(duplicate) = normalized.windows(2).find(|pair| pair[0].id == pair[1].id) {
        return Err(ViewerInputError::DuplicateId(duplicate[0].id));
    }
    if let Some(viewer) = normalized
        .iter()
        .find(|viewer| viewer.horizontal_view_distance_voxels < 0)
    {
        return Err(ViewerInputError::NegativeHorizontalDistance {
            id: viewer.id,
            value: viewer.horizontal_view_distance_voxels,
        });
    }
    if let Some(viewer) = normalized
        .iter()
        .find(|viewer| viewer.vertical_view_distance_voxels < 0)
    {
        return Err(ViewerInputError::NegativeVerticalDistance {
            id: viewer.id,
            value: viewer.vertical_view_distance_voxels,
        });
    }
    Ok(normalized)
}

/// Compute the data and mesh boxes for one viewer. Equivalent to C++
/// `process_viewers` Step E.
fn compute_viewer_boxes(state: &mut ViewerState, data_block_size: i32, mesh_block_size: i32) {
    let _ = mesh_block_size; // factor == 1 for now
    let mesh_h_blocks = ceil_div(state.horizontal_view_distance_voxels, mesh_block_size);
    let mesh_v_blocks = ceil_div(state.vertical_view_distance_voxels, mesh_block_size);
    let mesh_block_pos = floor_div_vec(state.local_position_voxels, mesh_block_size);
    state.mesh_box = Box3i::from_center_extents(
        mesh_block_pos,
        Vector3i::new(mesh_h_blocks, mesh_v_blocks, mesh_h_blocks),
    );

    // Data box is mesh box (in data-block units) padded by 1 for meshing
    // neighbours. factor == 1 here, so the conversion is identity.
    let data_h_blocks = mesh_h_blocks + 1;
    let data_v_blocks = mesh_v_blocks + 1;
    let data_block_pos = floor_div_vec(state.local_position_voxels, data_block_size);
    state.data_box = Box3i::from_center_extents(
        data_block_pos,
        Vector3i::new(data_h_blocks, data_v_blocks, data_h_blocks),
    );
}

fn ceil_div(a: i32, b: i32) -> i32 {
    (a + b - 1) / b
}

/// Compute per-LOD data/mesh boxes for a viewer. Each LOD level `N` uses a
/// block size of `data_block_size * (1 << N)`, so coarser LODs cover more world
/// space per block. The view distance (in voxels) is the same for all LODs —
/// the effect is that fewer, larger blocks are loaded at higher LODs. This is
/// the simplest multi-LOD strategy (the C++ clipbox system uses a per-LOD
/// distance falloff; this MVP uses uniform distance for simplicity).
#[allow(dead_code)] // retained as a unit-test harness after the planner cutover
fn compute_viewer_boxes_multi_lod(state: &mut ViewerState, data_block_size: i32, lod_count: u8) {
    if !state.demand.any() {
        // No meshes: just keep data resident. Only LOD 0 for simplicity.
        let h_blocks = ceil_div(state.horizontal_view_distance_voxels, data_block_size);
        let v_blocks = ceil_div(state.vertical_view_distance_voxels, data_block_size);
        let block_pos = floor_div_vec(state.local_position_voxels, data_block_size);
        state.data_box =
            Box3i::from_center_extents(block_pos, Vector3i::new(h_blocks, v_blocks, h_blocks));
        state.mesh_box = Box3i::default();
        state.data_box_per_lod = vec![state.data_box];
        state.mesh_box_per_lod = vec![Box3i::default()];
        return;
    }

    state.data_box_per_lod = Vec::with_capacity(lod_count as usize);
    state.mesh_box_per_lod = Vec::with_capacity(lod_count as usize);

    for lod in 0..lod_count as i32 {
        let lod_block_size = data_block_size << lod;
        let mesh_h = ceil_div(state.horizontal_view_distance_voxels, lod_block_size);
        let mesh_v = ceil_div(state.vertical_view_distance_voxels, lod_block_size);
        let mesh_pos = floor_div_vec(state.local_position_voxels, lod_block_size);
        let mesh_box = Box3i::from_center_extents(mesh_pos, Vector3i::new(mesh_h, mesh_v, mesh_h));

        // Data box is mesh box padded by 1 (in this LOD's block units) for
        // meshing neighbours.
        let data_h = mesh_h + 1;
        let data_v = mesh_v + 1;
        let data_pos = floor_div_vec(state.local_position_voxels, lod_block_size);
        let data_box = Box3i::from_center_extents(data_pos, Vector3i::new(data_h, data_v, data_h));

        state.data_box_per_lod.push(data_box);
        state.mesh_box_per_lod.push(mesh_box);

        // LOD-0 backward compat fields.
        if lod == 0 {
            state.data_box = data_box;
            state.mesh_box = mesh_box;
        }
    }
}

fn floor_div_vec(v: Vector3i, b: i32) -> Vector3i {
    Vector3i::new(v.x.div_euclid(b), v.y.div_euclid(b), v.z.div_euclid(b))
}

fn checked_edit_mesh_block_bounds_span(
    edit_min: Vector3i,
    edit_max: Vector3i,
    minimum_padding: u32,
    maximum_padding: u32,
    data_block_size: i32,
    lod_index: usize,
) -> Result<(Vector3i, Vector3i, usize), VoxelTerrainRuntimeError> {
    let (min_a, max_a, _) = checked_edit_mesh_block_bounds(
        edit_min,
        minimum_padding,
        maximum_padding,
        data_block_size,
        lod_index,
    )?;
    if edit_min == edit_max {
        let count = block_span_count(min_a, max_a)?;
        return Ok((min_a, max_a, count));
    }
    let (min_b, max_b, _) = checked_edit_mesh_block_bounds(
        edit_max,
        minimum_padding,
        maximum_padding,
        data_block_size,
        lod_index,
    )?;
    let minimum = Vector3i::new(
        min_a.x.min(min_b.x),
        min_a.y.min(min_b.y),
        min_a.z.min(min_b.z),
    );
    let maximum = Vector3i::new(
        max_a.x.max(max_b.x),
        max_a.y.max(max_b.y),
        max_a.z.max(max_b.z),
    );
    let count = block_span_count(minimum, maximum)?;
    Ok((minimum, maximum, count))
}

fn block_span_count(
    minimum: Vector3i,
    maximum: Vector3i,
) -> Result<usize, VoxelTerrainRuntimeError> {
    let axis_count = |min: i32, max: i32| {
        i64::from(max)
            .checked_sub(i64::from(min))?
            .checked_add(1)
            .and_then(|count| usize::try_from(count).ok())
    };
    axis_count(minimum.x, maximum.x)
        .and_then(|x| {
            axis_count(minimum.y, maximum.y).and_then(|y| {
                axis_count(minimum.z, maximum.z).and_then(|z| x.checked_mul(y)?.checked_mul(z))
            })
        })
        .ok_or(VoxelTerrainRuntimeError::CoordinateOverflow)
}

fn checked_edit_mesh_block_bounds(
    position: Vector3i,
    minimum_padding: u32,
    maximum_padding: u32,
    data_block_size: i32,
    lod_index: usize,
) -> Result<(Vector3i, Vector3i, usize), VoxelTerrainRuntimeError> {
    let shift =
        u32::try_from(lod_index).map_err(|_| VoxelTerrainRuntimeError::CoordinateOverflow)?;
    let lod_scale = 1i64
        .checked_shl(shift)
        .ok_or(VoxelTerrainRuntimeError::CoordinateOverflow)?;
    let stride = i64::from(data_block_size)
        .checked_mul(lod_scale)
        .filter(|stride| *stride > 0)
        .ok_or(VoxelTerrainRuntimeError::CoordinateOverflow)?;
    let minimum_padding = i64::from(minimum_padding)
        .checked_mul(lod_scale)
        .ok_or(VoxelTerrainRuntimeError::CoordinateOverflow)?;
    let maximum_padding = i64::from(maximum_padding)
        .checked_mul(lod_scale)
        .ok_or(VoxelTerrainRuntimeError::CoordinateOverflow)?;
    let axis = |coordinate: i32| {
        let minimum_voxel = i64::from(coordinate).checked_sub(maximum_padding)?;
        let maximum_voxel = i64::from(coordinate).checked_add(minimum_padding)?;
        let minimum_block = minimum_voxel.div_euclid(stride);
        let maximum_block = maximum_voxel.div_euclid(stride);
        Some((
            i32::try_from(minimum_block).ok()?,
            i32::try_from(maximum_block).ok()?,
        ))
    };
    let (minimum_x, maximum_x) =
        axis(position.x).ok_or(VoxelTerrainRuntimeError::CoordinateOverflow)?;
    let (minimum_y, maximum_y) =
        axis(position.y).ok_or(VoxelTerrainRuntimeError::CoordinateOverflow)?;
    let (minimum_z, maximum_z) =
        axis(position.z).ok_or(VoxelTerrainRuntimeError::CoordinateOverflow)?;
    let axis_count = |minimum: i32, maximum: i32| {
        i64::from(maximum)
            .checked_sub(i64::from(minimum))?
            .checked_add(1)
            .and_then(|count| usize::try_from(count).ok())
    };
    let count = axis_count(minimum_x, maximum_x)
        .and_then(|x| {
            axis_count(minimum_y, maximum_y).and_then(|y| {
                axis_count(minimum_z, maximum_z)
                    .and_then(|z| x.checked_mul(y).and_then(|xy| xy.checked_mul(z)))
            })
        })
        .ok_or(VoxelTerrainRuntimeError::CoordinateOverflow)?;
    Ok((
        Vector3i::new(minimum_x, minimum_y, minimum_z),
        Vector3i::new(maximum_x, maximum_y, maximum_z),
        count,
    ))
}

fn map_voxel_edit_storage_error(error: SharedVoxelDataMutationError) -> VoxelTerrainRuntimeError {
    match error {
        SharedVoxelDataMutationError::InvalidChannel { channel_index } => {
            VoxelTerrainRuntimeError::InvalidVoxelChannel { channel_index }
        }
        SharedVoxelDataMutationError::SpatialBoundsOverflow { .. } => {
            VoxelTerrainRuntimeError::CoordinateOverflow
        }
        SharedVoxelDataMutationError::SettingsRevisionOverflow => {
            VoxelTerrainRuntimeError::SettingsRevisionOverflow
        }
        SharedVoxelDataMutationError::ConcurrentSettingsMutation {
            expected_revision,
            actual_revision,
        } => VoxelTerrainRuntimeError::ConcurrentSettingsMutation {
            expected_revision,
            actual_revision,
        },
        SharedVoxelDataMutationError::ConcurrentDataMutation {
            position,
            lod_index,
            expected_revision,
            actual_revision,
        } if u8::try_from(lod_index).is_ok() => VoxelTerrainRuntimeError::ConcurrentDataMutation {
            location: BlockLocation {
                position,
                lod_index: lod_index as u8,
            },
            expected_revision,
            actual_revision,
        },
        SharedVoxelDataMutationError::KeyRevisionOverflow {
            position,
            lod_index,
        } if u8::try_from(lod_index).is_ok() => VoxelTerrainRuntimeError::BlockRevisionOverflow {
            location: BlockLocation {
                position,
                lod_index: lod_index as u8,
            },
        },
        SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed { .. }
        | SharedVoxelDataMutationError::CapacityReservationFailed => {
            VoxelTerrainRuntimeError::CapacityReservationFailed
        }
        error => VoxelTerrainRuntimeError::DataMutation(error),
    }
}

fn block_box_to_voxel_box(block_box: Box3i, block_size: i32, lod: usize) -> Box3i {
    let stride = block_size << lod;
    Box3i::new(block_box.position * stride, block_box.size * stride)
}

fn num_threads() -> usize {
    let n = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    // Cap aggressively: terrain paging is latency-bound, not throughput-bound,
    // and the test suite spawns many cores. 4 keeps turn-around short.
    n.min(4)
}

// ---------------------------------------------------------------------------
// Task plumbing: downcast helpers for load/mesh task dispatch
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TerrainLoadOutput {
    block_data: BlockDataOutput,
    request_generation: u64,
    request_tag: Option<TaskRequestTag>,
}

impl TerrainLoadOutput {
    #[cfg(test)]
    fn new(block_data: BlockDataOutput, request_generation: u64) -> Self {
        Self {
            block_data,
            request_generation,
            request_tag: None,
        }
    }

    #[cfg(test)]
    fn new_tagged(block_data: BlockDataOutput, request_tag: TaskRequestTag) -> Self {
        Self {
            block_data,
            request_generation: request_tag.request_generation,
            request_tag: Some(request_tag),
        }
    }

    fn new_optional(
        block_data: BlockDataOutput,
        request_generation: u64,
        request_tag: Option<TaskRequestTag>,
    ) -> Self {
        debug_assert!(
            request_tag.is_none_or(|tag| { tag.request_generation == request_generation })
        );
        Self {
            block_data,
            request_generation,
            request_tag,
        }
    }
}

/// Helper task the core spawns for each block load. Wraps a stream query into
/// the engine-agnostic `BlockDataOutput` shape used by terrain load handling
/// can consume it.
struct LoadBlockForTerrainTask {
    position: Vector3i,
    lod_index: u8,
    request_generation: u64,
    data: Arc<SharedVoxelData>,
    stream: Arc<dyn VoxelStream>,
    request_tag: Option<TaskRequestTag>,
    request_cancellation: Option<Arc<RequestCancellation>>,
    output: Option<TerrainLoadOutput>,
}

impl LoadBlockForTerrainTask {
    fn new(
        position: Vector3i,
        lod_index: u8,
        request_generation: u64,
        data: Arc<SharedVoxelData>,
        stream: Arc<dyn VoxelStream>,
    ) -> Self {
        Self {
            position,
            lod_index,
            request_generation,
            data,
            stream,
            request_tag: None,
            request_cancellation: None,
            output: None,
        }
    }

    pub(crate) fn with_request_control(
        mut self,
        tag: TaskRequestTag,
        cancellation: Arc<RequestCancellation>,
    ) -> Self {
        debug_assert_eq!(tag.request_generation, self.request_generation);
        self.request_tag = Some(tag);
        self.request_cancellation = Some(cancellation);
        self
    }

    pub(crate) const fn request_tag(&self) -> Option<TaskRequestTag> {
        self.request_tag
    }

    fn request_is_cancelled(&self) -> bool {
        self.request_cancellation
            .as_deref()
            .is_some_and(RequestCancellation::is_cancelled)
    }
}

impl ThreadedTask for LoadBlockForTerrainTask {
    fn run(&mut self, _ctx: crate::tasks::ThreadedTaskContext) -> crate::tasks::TaskRunStatus {
        if self.request_is_cancelled() {
            return crate::tasks::TaskRunStatus::Postponed;
        }
        // Try the stream first. If it has nothing, ask the generator.
        let settings = self.data.settings_snapshot();
        let bs = self.data.block_size() as i32;
        let format = settings.format;
        let generator = settings.generator;
        let lod = self.lod_index;
        let generation = self.request_generation;
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(bs));
        format.configure_buffer(&mut voxels);
        let query = crate::streams::VoxelLoadQuery::new(&mut voxels, self.position, lod);
        let stream_result = self.stream.load_voxel_block(query);
        if self.request_is_cancelled() {
            return crate::tasks::TaskRunStatus::Postponed;
        }
        match stream_result {
            Ok(crate::streams::LoadResult::Found) => {
                self.output = Some(TerrainLoadOutput::new_optional(
                    BlockDataOutput::loaded(self.position, lod, voxels, false),
                    generation,
                    self.request_tag,
                ));
            }
            Ok(crate::streams::LoadResult::NotFound) => {
                // Fall back to the generator if installed.
                if let Some(gen) = generator {
                    use crate::generators::base::VoxelQueryData;
                    let lod_stride = 1i32 << lod;
                    gen.generate_block(VoxelQueryData {
                        buffer: &mut voxels,
                        origin_in_voxels: self.position * bs * lod_stride,
                        lod: lod as u32,
                    });
                    if self.request_is_cancelled() {
                        return crate::tasks::TaskRunStatus::Postponed;
                    }
                    self.output = Some(TerrainLoadOutput::new_optional(
                        BlockDataOutput::loaded(self.position, lod, voxels, false),
                        generation,
                        self.request_tag,
                    ));
                } else {
                    self.output = Some(TerrainLoadOutput::new_optional(
                        BlockDataOutput::not_found(self.position, lod),
                        generation,
                        self.request_tag,
                    ));
                }
            }
            Err(_err) => {
                self.output = Some(TerrainLoadOutput::new_optional(
                    BlockDataOutput::loaded_dropped(self.position, lod),
                    generation,
                    self.request_tag,
                ));
            }
        }
        crate::tasks::TaskRunStatus::Complete {
            follow_up_tasks: Vec::new(),
        }
    }

    fn debug_name(&self) -> &'static str {
        "LoadBlockForTerrain"
    }

    fn is_cancelled(&mut self) -> bool {
        self.request_is_cancelled()
    }

    fn priority(&mut self) -> crate::tasks::TaskPriority {
        // TASK-1 parity: use TASK_PRIORITY_LOAD_BAND2 (= 10) instead of
        // inheriting the trait default TaskPriority::max() which starves mesh
        // tasks. Both load and mesh now share band 2 at the same base level.
        crate::tasks::TaskPriority::new(
            0,
            0,
            crate::constants::voxel_constants::TASK_PRIORITY_LOAD_BAND2,
            0,
        )
    }
}

/// Take typed outputs through the runner-retained task object's `Any` bound.
/// The runner never calls the user-provided `debug_name` callback while
/// dispatching or normalizing completions.
fn try_take_load_output(task: &mut dyn ThreadedTask) -> Option<TerrainLoadOutput> {
    let task = (task as &mut dyn std::any::Any).downcast_mut::<LoadBlockForTerrainTask>()?;
    task.output.take()
}

fn try_take_mesh_output(task: &mut dyn ThreadedTask) -> Option<MeshBlockTaskOutput> {
    let task = (task as &mut dyn std::any::Any).downcast_mut::<MeshBlockTask>()?;
    task.take_output()
}

const fn completion_panic_phase(
    status: TaskCompletionStatus,
) -> Option<crate::tasks::TaskPanicPhase> {
    match status {
        TaskCompletionStatus::Panicked(phase) => Some(phase),
        TaskCompletionStatus::Finished | TaskCompletionStatus::Cancelled => None,
    }
}

fn completion_task_kind(completed: &CompletedTask) -> CompletionTaskKind {
    let task = completed.task_any();
    if task.is::<LoadBlockForTerrainTask>() {
        CompletionTaskKind::Load
    } else if task.is::<MeshBlockTask>() {
        CompletionTaskKind::Mesh
    } else if task.is::<SaveBlockDataTask>() {
        CompletionTaskKind::Save
    } else if task.is::<FlushVoxelStreamTask>() {
        CompletionTaskKind::Flush
    } else {
        CompletionTaskKind::Unknown
    }
}

fn finished_completion_has_output(completed: &CompletedTask, kind: CompletionTaskKind) -> bool {
    match kind {
        CompletionTaskKind::Load => completed
            .task_any()
            .downcast_ref::<LoadBlockForTerrainTask>()
            .is_some_and(|task| task.output.is_some()),
        CompletionTaskKind::Mesh => completed
            .task_any()
            .downcast_ref::<MeshBlockTask>()
            .is_some_and(|task| task.output_ref().is_some()),
        CompletionTaskKind::Save => completed
            .task_any()
            .downcast_ref::<SaveBlockDataTask>()
            .and_then(SaveBlockDataTask::terminal_ref)
            .is_some_and(|terminal| {
                terminal.phase == crate::streams::PersistenceIoPhase::Acknowledged
                    && matches!(
                        terminal.acknowledgement,
                        Some(PersistenceAcknowledgement::Save(_))
                    )
            }),
        CompletionTaskKind::Flush => completed
            .task_any()
            .downcast_ref::<FlushVoxelStreamTask>()
            .and_then(FlushVoxelStreamTask::terminal_ref)
            .is_some_and(|terminal| {
                terminal.phase == crate::streams::PersistenceIoPhase::Acknowledged
                    && matches!(
                        terminal.acknowledgement,
                        Some(PersistenceAcknowledgement::Flush(_))
                    )
            }),
        CompletionTaskKind::Unknown => true,
    }
}

#[cfg(test)]
mod tests;

// Keep the `VoxelBuffer` import used by the load task even though tests
// don't exercise it directly.
