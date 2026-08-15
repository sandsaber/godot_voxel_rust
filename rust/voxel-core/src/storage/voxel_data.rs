//! Aggregate voxel storage over LOD maps.
//!
//! Engine-agnostic port of `storage/voxel_data.{h,cpp}`. Owns the per-LOD
//! sparse block maps plus an optional generator and stream, and exposes the
//! synchronous storage contract: LOD maps, format, bounds, block insertion,
//! direct voxel edits, modification flags, LOD cascade, copy/paste, the
//! reference-counted view/unview API and area-loaded queries. Threaded
//! streaming task integration is layered on top in later Phase 4 steps.
//! Shared task code reaches `VoxelData` through [`SharedVoxelData`], which
//! owns the scoped `SpatialLock3D` region guards used by C++ terrain workers.

use crate::constants::voxel_constants::MAX_LOD;
use crate::generators::base::{VoxelGenerator, VoxelQueryData};
use crate::math::{Box3i, BoxBounds3i, Vector3i};
use crate::storage::{
    voxel_buffer::{raw_voxel_to_real, real_to_raw_voxel, SDF_FAR_OUTSIDE},
    VoxelBuffer, VoxelDataBlock, VoxelDataMap, VoxelFormat,
};
use crate::streams::VoxelStream;
use crate::terrain::lod_clipbox::{bounds_in_lod_blocks, checked_box_intersection};
use crate::thread::{PreparedSpatialWriteBatch, SpatialLock3D, SpatialLockWriteManyGuard};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Deref;
#[cfg(test)]
use std::ops::DerefMut;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
#[cfg(test)]
use std::sync::atomic::{AtomicU8, AtomicUsize};
use std::sync::RwLockWriteGuard as StdRwLockWriteGuard;
use std::sync::{
    Arc, Mutex as StdMutex, MutexGuard as StdMutexGuard, RwLock as StdRwLock,
    RwLockReadGuard as StdRwLockReadGuard, TryLockError, Weak,
};

#[derive(Debug)]
struct VoxelDataLod {
    map: VoxelDataMap,
}

impl VoxelDataLod {
    fn new(lod_index: u8, format: VoxelFormat) -> Self {
        let mut map = VoxelDataMap::new(lod_index);
        map.set_format(format);
        Self { map }
    }
}

#[derive(Debug)]
pub struct BlockToSave {
    pub voxels: Option<VoxelBuffer>,
    pub position: Vector3i,
    pub lod_index: u8,
    pub block_revision: u64,
}

/// Position of a block affected by a LOD update pass.
/// Matches `VoxelData::BlockLocation` in C++.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLocation {
    pub position: Vector3i,
    pub lod_index: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoxelDataLodResizeError {
    InvalidLodCount { requested: usize },
    NonEmptyTruncatedLod { lod_index: usize },
    CapacityReservationFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelDataKeyRevision {
    Present(u64),
    Tombstone(u64),
}

/// Recoverable allocation class used by prepared shared-data transactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SharedVoxelDataTransactionReservation {
    OperationStorage = 1,
    SpatialPreparation = 2,
    LiveMap = 3,
    LiveKeyRevisions = 4,
    RemovedOutcome = 5,
    LiveSpatialRegistry = 6,
    PreviewSnapshotStorage = 7,
    ObservationStorage = 8,
    ResidentDirtyPayloadStorage = 9,
    ResidentDirtyPayloadCopy = 10,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SharedVoxelDataMutationError {
    /// The owning terrain has entered its final shutdown boundary. Live
    /// storage remains readable, but no further mutation can be admitted.
    MutationAdmissionClosed,
    ShutdownMutationPermitMismatch,
    InvalidChannel {
        channel_index: usize,
    },
    SettingsRevisionOverflow,
    ConcurrentSettingsMutation {
        expected_revision: u64,
        actual_revision: u64,
    },
    ConcurrentDataMutation {
        position: Vector3i,
        lod_index: usize,
        expected_revision: VoxelDataKeyRevision,
        actual_revision: VoxelDataKeyRevision,
    },
    SpatialBoundsOverflow {
        lod_index: usize,
    },
    LodDestinationUnavailable {
        position: Vector3i,
        lod_index: usize,
    },
    ViewerCountOverflow {
        position: Vector3i,
        lod_index: usize,
    },
    ViewerCountUnderflow {
        position: Vector3i,
        lod_index: usize,
    },
    KeyRevisionOverflow {
        position: Vector3i,
        lod_index: usize,
    },
    DirtyBlockMissingVoxels {
        location: BlockLocation,
    },
    DuplicatePreparedTransactionOperation {
        location: BlockLocation,
    },
    PreparedTransactionExpectedPresent {
        location: BlockLocation,
        actual_revision: VoxelDataKeyRevision,
    },
    PreparedTransactionExpectedTombstone {
        location: BlockLocation,
        actual_revision: VoxelDataKeyRevision,
    },
    PreparedTransactionConcurrentBlockState {
        location: BlockLocation,
        expected_viewers: u32,
        actual_viewers: u32,
        expected_modified: bool,
        actual_modified: bool,
        expected_edited: bool,
        actual_edited: bool,
        expected_has_voxels: bool,
        actual_has_voxels: bool,
    },
    PreparedTransactionNoop {
        location: BlockLocation,
    },
    PreparedTransactionPreviewSetMismatch,
    PreparedTransactionBlockLodMismatch {
        location: BlockLocation,
        block_lod_index: u8,
    },
    PreparedTransactionBlockSizeMismatch {
        location: BlockLocation,
        expected_size: Vector3i,
        actual_size: Vector3i,
    },
    PreparedTransactionCapacityReservationFailed {
        reservation: SharedVoxelDataTransactionReservation,
    },
    PreparedTransactionAlreadyCommitted,
    CapacityReservationFailed,
}

/// One final operation for one shared-data key.
///
/// Exact-viewer no-ops are rejected during preparation. Therefore every
/// accepted operation changes its key and advances its key revision once.
#[derive(Debug)]
#[allow(dead_code)]
// Additive C2c-C1 foundation; terrain adopts it in C2c-C2.
// Boxing the insert payload would deallocate the box while commit guards are
// held. Inline ownership keeps the no-fail move into the live map allocation-
// and drop-free.
#[allow(clippy::large_enum_variant)]
pub(crate) enum SharedVoxelDataTransactionOperation {
    SetViewersExact {
        location: BlockLocation,
        final_viewers: u32,
    },
    SetViewersExactAndClearModified {
        location: BlockLocation,
        final_viewers: u32,
    },
    Insert {
        location: BlockLocation,
        block: VoxelDataBlock,
        final_viewers: u32,
    },
    Replace {
        location: BlockLocation,
        block: VoxelDataBlock,
    },
    ClearModified {
        location: BlockLocation,
    },
    Remove {
        location: BlockLocation,
    },
}

#[allow(dead_code)]
impl SharedVoxelDataTransactionOperation {
    pub(crate) const fn location(&self) -> BlockLocation {
        match self {
            Self::SetViewersExact { location, .. }
            | Self::SetViewersExactAndClearModified { location, .. }
            | Self::Insert { location, .. }
            | Self::Replace { location, .. }
            | Self::ClearModified { location }
            | Self::Remove { location } => *location,
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct PreparedSharedVoxelDataTransactionPrepareError {
    error: SharedVoxelDataMutationError,
    operations: Vec<SharedVoxelDataTransactionOperation>,
}

#[allow(dead_code)]
impl PreparedSharedVoxelDataTransactionPrepareError {
    pub(crate) const fn error(&self) -> &SharedVoxelDataMutationError {
        &self.error
    }

    pub(crate) fn operations(&self) -> &[SharedVoxelDataTransactionOperation] {
        &self.operations
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        SharedVoxelDataMutationError,
        Vec<SharedVoxelDataTransactionOperation>,
    ) {
        (self.error, self.operations)
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct RemovedSharedVoxelDataBlock {
    location: BlockLocation,
    block: VoxelDataBlock,
    block_revision: u64,
}

#[allow(dead_code)]
impl RemovedSharedVoxelDataBlock {
    pub(crate) const fn location(&self) -> BlockLocation {
        self.location
    }

    pub(crate) const fn block(&self) -> &VoxelDataBlock {
        &self.block
    }

    pub(crate) const fn block_revision(&self) -> u64 {
        self.block_revision
    }

    pub(crate) fn into_parts(self) -> (BlockLocation, VoxelDataBlock) {
        (self.location, self.block)
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct PreparedSharedVoxelDataTransactionOutcome {
    removed_blocks: Vec<RemovedSharedVoxelDataBlock>,
}

#[allow(dead_code)]
impl PreparedSharedVoxelDataTransactionOutcome {
    pub(crate) fn removed_blocks(&self) -> &[RemovedSharedVoxelDataBlock] {
        &self.removed_blocks
    }

    pub(crate) fn into_removed_blocks(self) -> Vec<RemovedSharedVoxelDataBlock> {
        self.removed_blocks
    }
}

/// Shared generator storage. Implementations are `Send + Sync` and own any
/// internal synchronization they need, matching the C++ contract that
/// generators can be called from multiple worker threads.
pub type SharedVoxelGenerator = Arc<dyn VoxelGenerator>;

/// Shared stream storage. `VoxelStream` is already `Send + Sync`; the `Arc`
/// lets multiple task instances reach the same stream.
pub type SharedVoxelStream = Arc<dyn VoxelStream>;

/// Test-only checkpoints for transactional `SharedVoxelData` mutation paths.
///
/// Preparation checkpoints occur before the mutation gate. Commit checkpoints
/// expose the canonical gate -> spatial write -> map write order; edit-only
/// checkpoints additionally bracket voxel publication and dirty flags while
/// the same map write guard remains held.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedVoxelDataEditPhase {
    LodUpdatePreparedBeforeMutationGate,
    DirtySnapshotPreparedBeforeMutationGate,
    PreparedVoxelEditDraftedBeforeTransactionPrepare,
    MutationGateAcquiredBeforeSpatialWrite,
    SpatialWriteBatchAcquired { lod_index: usize },
    SpatialWriteAcquiredBeforeMapLock,
    VoxelWrittenBeforeDirtyFlags,
    DirtyFlagsSetBeforeMapWriteUnlock,
    PreparedTransactionValidatedBeforeFirstLiveWrite,
}

#[cfg(test)]
pub type SharedVoxelDataEditPhaseHook =
    Arc<dyn Fn(SharedVoxelDataEditPhase) + Send + Sync + 'static>;

#[derive(Debug)]
struct SharedVoxelDataLodState {
    map: VoxelDataMap,
    structural_revision: u64,
}

#[derive(Debug)]
struct SharedVoxelDataLod {
    state: StdRwLock<SharedVoxelDataLodState>,
}

struct SharedVoxelDataSettings {
    revision: u64,
    format: VoxelFormat,
    bounds_in_voxels: Box3i,
    full_load_completed: bool,
    streaming_enabled: bool,
    generator: Option<SharedVoxelGenerator>,
    stream: Option<SharedVoxelStream>,
}

#[derive(Clone)]
pub struct SharedVoxelDataSettingsSnapshot {
    pub revision: u64,
    pub format: VoxelFormat,
    pub bounds_in_voxels: Box3i,
    pub full_load_completed: bool,
    pub streaming_enabled: bool,
    pub generator: Option<SharedVoxelGenerator>,
    pub stream: Option<SharedVoxelStream>,
}

struct PreparedMarkMutation {
    position: Vector3i,
    expected_revision: VoxelDataKeyRevision,
    next_revision: u64,
    bounds: BoxBounds3i,
    set_needs_lodding: bool,
}

struct PreparedDirtyBlockMutation {
    position: Vector3i,
    lod_index: usize,
    expected_revision: VoxelDataKeyRevision,
    next_revision: u64,
    bounds: BoxBounds3i,
}

struct PreparedLodBlockMutation {
    position: Vector3i,
    lod_index: usize,
    expected_revision: VoxelDataKeyRevision,
    next_revision: u64,
    bounds: BoxBounds3i,
    block: VoxelDataBlock,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct PreparedSharedVoxelDataExpectation {
    location: BlockLocation,
    expected_revision: VoxelDataKeyRevision,
    next_revision: u64,
    expected_block_state: Option<PreparedSharedVoxelDataBlockState>,
    bounds: BoxBounds3i,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
struct PreparedSharedVoxelDataBlockState {
    viewers: u32,
    modified: bool,
    edited: bool,
    has_voxels: bool,
}

struct SharedVoxelDataTransactionPreviewLineage {
    data: Weak<SharedVoxelData>,
    settings_revision: u64,
}

/// Block metadata used by an outer terrain draft before it prepares one exact
/// shared-data transaction. Its private lineage prevents snapshots from being
/// mixed across preview tokens, even when they refer to the same data and
/// settings revision.
#[derive(Clone)]
pub(crate) struct SharedVoxelDataTransactionBlockSnapshot {
    lineage: Arc<SharedVoxelDataTransactionPreviewLineage>,
    location: BlockLocation,
    revision: VoxelDataKeyRevision,
    present: bool,
    viewers: u32,
    modified: bool,
    edited: bool,
    has_voxels: bool,
}

/// Opaque lineage token binding an outer terrain preview to one concrete
/// shared-data instance and settings revision.
///
/// Block snapshots and final C1 preparation must both flow through this
/// owner. A settings change anywhere between token creation and preparation
/// therefore rejects the whole draft without transferring any operation
/// payload.
#[must_use = "a transaction preview must be prepared or deliberately abandoned"]
pub(crate) struct SharedVoxelDataTransactionPreview {
    data: Arc<SharedVoxelData>,
    settings: SharedVoxelDataSettingsSnapshot,
    lineage: Arc<SharedVoxelDataTransactionPreviewLineage>,
    #[cfg(test)]
    after_prepare_hook: Option<Arc<dyn Fn() + Send + Sync + 'static>>,
}

/// Deep-owned save payload paired with the exact resident revision observed
/// before its matching `ClearModified` operation.
pub(crate) struct RevisionedBlockToSave {
    pub(crate) location: BlockLocation,
    pub(crate) block_revision: u64,
    pub(crate) voxels: VoxelBuffer,
}

/// Complete, non-mutating resident-dirty draft for one outer terrain
/// transaction. All three vectors use the same canonical key order.
#[must_use = "resident dirty copies must be committed or deliberately abandoned"]
pub(crate) struct PreparedResidentDirtyCopies {
    operations: Vec<SharedVoxelDataTransactionOperation>,
    snapshots: Vec<SharedVoxelDataTransactionBlockSnapshot>,
    payloads: Vec<RevisionedBlockToSave>,
}

impl PreparedResidentDirtyCopies {
    pub(crate) fn into_parts(
        self,
    ) -> (
        Vec<SharedVoxelDataTransactionOperation>,
        Vec<SharedVoxelDataTransactionBlockSnapshot>,
        Vec<RevisionedBlockToSave>,
    ) {
        (self.operations, self.snapshots, self.payloads)
    }
}

#[allow(dead_code)]
pub(crate) struct PreparedSharedVoxelDataTransaction {
    data: Arc<SharedVoxelData>,
    expected_settings_revision: u64,
    operations: Vec<SharedVoxelDataTransactionOperation>,
    expectations: Vec<PreparedSharedVoxelDataExpectation>,
    observations: Vec<PreparedSharedVoxelDataExpectation>,
    spatial_batches: [Option<PreparedSpatialWriteBatch>; MAX_LOD],
    insertion_counts: [usize; MAX_LOD],
    key_revision_counts: [usize; MAX_LOD],
    removed_blocks: Vec<RemovedSharedVoxelDataBlock>,
    shutdown_mutation_lineage: Option<Weak<SharedVoxelData>>,
    committed: bool,
    #[cfg(test)]
    spatial_batch_drop_hook: Option<Arc<dyn Fn() + Send + Sync + 'static>>,
}

/// Exact storage-lineage capability issued only when terrain begins shutdown.
///
/// Its field is private and the type is crate-private, so ordinary callers
/// cannot manufacture authorization to mutate through a closed admission
/// boundary.
pub(crate) struct SharedVoxelDataShutdownMutationPermit {
    data: Weak<SharedVoxelData>,
}

impl SharedVoxelDataShutdownMutationPermit {
    fn matches(&self, data: &Arc<SharedVoxelData>) -> bool {
        Weak::ptr_eq(&self.data, &Arc::downgrade(data))
    }
}

/// Complete storage draft for one voxel edit and its configured LOD pyramid.
///
/// The wrapped transaction owns every replacement buffer until commit. This
/// keeps preparation non-mutating and lets terrain publish matching mesh state
/// while the transaction's publication fence is still held.
#[must_use = "a prepared voxel edit must be committed or deliberately abandoned"]
#[allow(dead_code)] // Task 3 publishes the prepared edit with matching mesh state.
pub(crate) struct PreparedSharedVoxelDataEdit {
    transaction: PreparedSharedVoxelDataTransaction,
    edited_block: BlockLocation,
    block_revision: u64,
    inserted_locations: Vec<BlockLocation>,
}

/// Opaque publication fence for an already-committed shared-data transaction.
///
/// The live storage writes are installed, but every affected map and spatial
/// region plus the mutation gate remain exclusively held until this value is
/// finished or dropped. C2c-C2 will use that interval to publish matching
/// terrain state without exposing any raw storage guard.
#[must_use = "the publication fence must be finished or deliberately dropped"]
#[allow(dead_code)] // Additive C2c-C1F foundation; terrain adopts it in C2c-C2.
pub(crate) struct CommittedSharedVoxelDataTransaction<'a> {
    map_guards: Option<[Option<StdRwLockWriteGuard<'a, SharedVoxelDataLodState>>; MAX_LOD]>,
    spatial_guards: Option<[Option<SpatialLockWriteManyGuard<'a>>; MAX_LOD]>,
    mutation_gate: Option<StdMutexGuard<'a, ()>>,
    removed_blocks: Vec<RemovedSharedVoxelDataBlock>,
    #[cfg(test)]
    spatial_batch_drop_hook: Option<Arc<dyn Fn() + Send + Sync + 'static>>,
}

impl fmt::Debug for PreparedSharedVoxelDataTransaction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedSharedVoxelDataTransaction")
            .field(
                "expected_settings_revision",
                &self.expected_settings_revision,
            )
            .field("operations", &self.operations)
            .field("expectations", &self.expectations)
            .field("observations", &self.observations)
            .field("spatial_batches", &self.spatial_batches)
            .field("insertion_counts", &self.insertion_counts)
            .field("key_revision_counts", &self.key_revision_counts)
            .field("removed_blocks", &self.removed_blocks)
            .field(
                "has_shutdown_mutation_lineage",
                &self.shutdown_mutation_lineage.is_some(),
            )
            .field("committed", &self.committed)
            .finish_non_exhaustive()
    }
}

/// Shared voxel-data handle for worker tasks.
///
/// This is the migration boundary between the earlier `Arc<Mutex<VoxelData>>`
/// port and the C++ shape where terrain code passes a shared `VoxelData`
/// pointer and each method scopes its own map/region locks. Settings now live
/// behind their own lock, each LOD map has an independent lock, and the
/// per-LOD [`SpatialLock3D`] guards are taken by mesh/read and edit/write
/// regions before touching voxel data.
///
/// Raw map-write entrypoints are intentionally unavailable outside storage:
///
/// ```compile_fail
/// use voxel_core::storage::SharedVoxelData;
/// fn bypass(data: &SharedVoxelData) {
///     data.with_lod_map_mut(0, |_| {});
/// }
/// ```
///
/// ```compile_fail
/// use voxel_core::storage::SharedVoxelData;
/// fn bypass(data: &SharedVoxelData) {
///     let _guard = data.try_lock();
/// }
/// ```
///
/// ```compile_fail
/// use voxel_core::storage::SharedVoxelData;
/// fn bypass(data: &SharedVoxelData) {
///     let _guard = data.try_lod_map_write(0);
/// }
/// ```
///
/// ```compile_fail
/// use voxel_core::{math::{Box3i, Vector3i}, storage::SharedVoxelData};
/// fn invert_lock_order(data: &SharedVoxelData) {
///     let _guard = data.write_region(0, Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
/// }
/// ```
///
/// ```compile_fail
/// use voxel_core::{math::{Box3i, Vector3i}, storage::SharedVoxelData};
/// fn invert_lock_order(data: &SharedVoxelData) {
///     let _guard = data.try_write_region(0, Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
/// }
/// ```
///
/// Silent legacy mutation wrappers are not part of the production API:
///
/// ```compile_fail
/// use voxel_core::{math::Vector3i, storage::SharedVoxelData};
/// fn swallow_direct_set(data: &SharedVoxelData) {
///     let _ = data.try_set_voxel(1, Vector3i::zero(), 0);
/// }
/// ```
///
/// ```compile_fail
/// use voxel_core::{math::{Box3i, Vector3i}, storage::SharedVoxelData};
/// fn swallow_mark_error(data: &SharedVoxelData) {
///     let _ = data.mark_area_modified(Box3i::new(Vector3i::zero(), Vector3i::splat(1)), true);
/// }
/// ```
///
/// ```compile_fail
/// use voxel_core::{math::Vector3i, storage::SharedVoxelData};
/// fn swallow_lod_error(data: &SharedVoxelData) {
///     let _ = data.update_lods_from_lod0_blocks(&[Vector3i::zero()]);
/// }
/// ```
///
/// ```compile_fail
/// use voxel_core::storage::SharedVoxelData;
/// fn swallow_dirty_error(data: &SharedVoxelData) {
///     let _ = data.consume_all_modifications();
/// }
/// ```
///
/// Shutdown mutation authority is not part of the public storage API:
///
/// ```compile_fail
/// use voxel_core::storage::SharedVoxelDataShutdownMutationPermit;
/// ```
pub struct SharedVoxelData {
    lods: Vec<SharedVoxelDataLod>,
    settings: StdRwLock<SharedVoxelDataSettings>,
    mutation_gate: StdMutex<()>,
    mutation_admission_closed: AtomicBool,
    spatial_locks: Vec<SpatialLock3D>,
    #[cfg(test)]
    edit_phase_hook: StdRwLock<Option<SharedVoxelDataEditPhaseHook>>,
    #[cfg(test)]
    transaction_reservation_failpoint: AtomicU8,
    #[cfg(test)]
    resident_payload_copy_failure_countdown: AtomicUsize,
}

pub struct SharedVoxelDataLodReadGuard<'a> {
    guard: StdRwLockReadGuard<'a, SharedVoxelDataLodState>,
}

impl Deref for SharedVoxelDataLodReadGuard<'_> {
    type Target = VoxelDataMap;

    fn deref(&self) -> &Self::Target {
        &self.guard.map
    }
}

#[cfg(test)]
pub(crate) struct SharedVoxelDataLodWriteGuard<'a> {
    guard: StdRwLockWriteGuard<'a, SharedVoxelDataLodState>,
}

#[cfg(test)]
impl Deref for SharedVoxelDataLodWriteGuard<'_> {
    type Target = VoxelDataMap;

    fn deref(&self) -> &Self::Target {
        &self.guard.map
    }
}

#[cfg(test)]
impl DerefMut for SharedVoxelDataLodWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard.map
    }
}

pub(crate) struct VoxelDataMutation<'a> {
    data: &'a SharedVoxelData,
    _gate: StdMutexGuard<'a, ()>,
}

impl SharedVoxelData {
    pub fn new(data: VoxelData) -> Self {
        let VoxelData {
            lods,
            format,
            bounds_in_voxels,
            full_load_completed,
            streaming_enabled,
            generator,
            stream,
        } = data;
        Self {
            lods: lods
                .into_iter()
                .map(|lod| SharedVoxelDataLod {
                    state: StdRwLock::new(SharedVoxelDataLodState {
                        map: lod.map,
                        structural_revision: 0,
                    }),
                })
                .collect(),
            settings: StdRwLock::new(SharedVoxelDataSettings {
                revision: 0,
                format,
                bounds_in_voxels,
                full_load_completed,
                streaming_enabled,
                generator,
                stream,
            }),
            mutation_gate: StdMutex::new(()),
            mutation_admission_closed: AtomicBool::new(false),
            spatial_locks: (0..MAX_LOD).map(|_| SpatialLock3D::new()).collect(),
            #[cfg(test)]
            edit_phase_hook: StdRwLock::new(None),
            #[cfg(test)]
            transaction_reservation_failpoint: AtomicU8::new(0),
            #[cfg(test)]
            resident_payload_copy_failure_countdown: AtomicUsize::new(0),
        }
    }

    pub const fn block_size(&self) -> u32 {
        VoxelDataMap::BLOCK_SIZE
    }

    pub const fn block_size_po2(&self) -> u8 {
        VoxelDataMap::BLOCK_SIZE_PO2
    }

    pub fn lod_count(&self) -> usize {
        self.lods.len()
    }

    pub fn settings_snapshot(&self) -> SharedVoxelDataSettingsSnapshot {
        let settings = self.settings.read().unwrap_or_else(|e| e.into_inner());
        SharedVoxelDataSettingsSnapshot {
            revision: settings.revision,
            format: settings.format,
            bounds_in_voxels: settings.bounds_in_voxels,
            full_load_completed: settings.full_load_completed,
            streaming_enabled: settings.streaming_enabled,
            generator: settings.generator.clone(),
            stream: settings.stream.clone(),
        }
    }

    #[cfg(test)]
    fn with_settings<R>(&self, f: impl FnOnce(&SharedVoxelDataSettings) -> R) -> R {
        let settings = self.settings.read().unwrap_or_else(|e| e.into_inner());
        f(&settings)
    }

    /// Registers a test-only mutation lifecycle observer.
    ///
    /// This has no production build surface. Checked mutation paths call
    /// [`Self::notify_test_edit_phase`] at the ordered checkpoints documented
    /// on [`SharedVoxelDataEditPhase`].
    #[cfg(test)]
    pub fn set_test_edit_phase_hook(&self, hook: SharedVoxelDataEditPhaseHook) {
        *self
            .edit_phase_hook
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Some(hook);
    }

    #[cfg(test)]
    pub(crate) fn set_test_transaction_reservation_failpoint(
        &self,
        failpoint: Option<SharedVoxelDataTransactionReservation>,
    ) {
        self.transaction_reservation_failpoint.store(
            failpoint.map_or(0, |failpoint| failpoint as u8),
            AtomicOrdering::SeqCst,
        );
    }

    #[cfg(test)]
    pub(crate) fn fail_resident_payload_copy_for_test(&self, occurrence: usize) {
        assert!(occurrence != 0, "copy failure occurrence is one-based");
        self.resident_payload_copy_failure_countdown
            .store(occurrence, AtomicOrdering::SeqCst);
    }

    #[cfg(test)]
    fn resident_payload_copy_should_fail(&self) -> bool {
        self.resident_payload_copy_failure_countdown
            .fetch_update(
                AtomicOrdering::SeqCst,
                AtomicOrdering::SeqCst,
                |remaining| (remaining != 0).then(|| remaining - 1),
            )
            .is_ok_and(|previous| previous == 1)
    }

    #[cfg(not(test))]
    const fn resident_payload_copy_should_fail(&self) -> bool {
        false
    }

    #[cfg(test)]
    fn transaction_reservation_should_fail(
        &self,
        reservation: SharedVoxelDataTransactionReservation,
    ) -> bool {
        self.transaction_reservation_failpoint
            .load(AtomicOrdering::SeqCst)
            == reservation as u8
    }

    #[cfg(not(test))]
    #[allow(dead_code)]
    const fn transaction_reservation_should_fail(
        &self,
        _reservation: SharedVoxelDataTransactionReservation,
    ) -> bool {
        false
    }

    #[cfg(test)]
    fn notify_test_edit_phase(&self, phase: SharedVoxelDataEditPhase) {
        let hook = self
            .edit_phase_hook
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(hook) = hook {
            hook(phase);
        }
    }

    pub fn format(&self) -> VoxelFormat {
        self.settings_snapshot().format
    }

    pub fn bounds(&self) -> Box3i {
        self.settings_snapshot().bounds_in_voxels
    }

    pub fn generator(&self) -> Option<SharedVoxelGenerator> {
        self.settings_snapshot().generator
    }

    pub fn stream(&self) -> Option<SharedVoxelStream> {
        self.settings_snapshot().stream
    }

    pub fn set_generator(
        &self,
        generator: Option<SharedVoxelGenerator>,
    ) -> Result<(), SharedVoxelDataMutationError> {
        self.begin_mutation()?.set_generator(generator)
    }

    /// Starts one settings-bound outer transaction preview.
    ///
    /// The token owns the exact [`SharedVoxelData`] `Arc`, so it cannot be
    /// replayed against an independent instance which happens to have the same
    /// numeric settings revision.
    pub(crate) fn begin_transaction_preview(self: &Arc<Self>) -> SharedVoxelDataTransactionPreview {
        let settings = self.settings_snapshot();
        let lineage = Arc::new(SharedVoxelDataTransactionPreviewLineage {
            data: Arc::downgrade(self),
            settings_revision: settings.revision,
        });
        SharedVoxelDataTransactionPreview {
            data: Arc::clone(self),
            settings,
            lineage,
            #[cfg(test)]
            after_prepare_hook: None,
        }
    }

    /// Prepares one canonical, retry-ownership-safe transaction over exact
    /// shared-data keys.
    ///
    /// Preparation performs no live mutation. On every error it returns the
    /// complete operation vector, including each exact owned insert payload.
    #[allow(dead_code)] // Additive C2c-C1 foundation; terrain adopts it in C2c-C2.
    pub(crate) fn prepare_transaction(
        self: &Arc<Self>,
        operations: Vec<SharedVoxelDataTransactionOperation>,
    ) -> Result<PreparedSharedVoxelDataTransaction, PreparedSharedVoxelDataTransactionPrepareError>
    {
        let expected_settings_revision = self
            .settings
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .revision;
        self.prepare_transaction_at_settings_revision(operations, expected_settings_revision)
    }

    fn prepare_transaction_at_settings_revision(
        self: &Arc<Self>,
        mut operations: Vec<SharedVoxelDataTransactionOperation>,
        expected_settings_revision: u64,
    ) -> Result<PreparedSharedVoxelDataTransaction, PreparedSharedVoxelDataTransactionPrepareError>
    {
        operations.sort_unstable_by_key(|operation| {
            let location = operation.location();
            (
                location.lod_index,
                location.position.x,
                location.position.y,
                location.position.z,
            )
        });
        if let Some(location) = operations.windows(2).find_map(|pair| {
            (pair[0].location() == pair[1].location()).then_some(pair[0].location())
        }) {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::DuplicatePreparedTransactionOperation {
                    location,
                },
                operations,
            });
        }

        if self.transaction_reservation_should_fail(
            SharedVoxelDataTransactionReservation::OperationStorage,
        ) {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::OperationStorage,
                },
                operations,
            });
        }
        let mut expectations = Vec::new();
        if expectations.try_reserve_exact(operations.len()).is_err() {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::OperationStorage,
                },
                operations,
            });
        }
        let mut insertion_counts = [0usize; MAX_LOD];
        let mut key_revision_counts = [0usize; MAX_LOD];
        let mut removal_count = 0usize;
        for operation in &operations {
            let expectation = match self.prepare_transaction_operation(operation) {
                Ok(expectation) => expectation,
                Err(error) => {
                    return Err(PreparedSharedVoxelDataTransactionPrepareError {
                        error,
                        operations,
                    });
                }
            };
            let lod_index = usize::from(expectation.location.lod_index);
            key_revision_counts[lod_index] = match key_revision_counts[lod_index].checked_add(1) {
                Some(count) => count,
                None => {
                    return Err(PreparedSharedVoxelDataTransactionPrepareError {
                        error: SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                            reservation: SharedVoxelDataTransactionReservation::OperationStorage,
                        },
                        operations,
                    });
                }
            };
            match operation {
                SharedVoxelDataTransactionOperation::Insert { .. } => {
                    insertion_counts[lod_index] = match insertion_counts[lod_index].checked_add(1) {
                        Some(count) => count,
                        None => {
                            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                                    error: SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                                        reservation: SharedVoxelDataTransactionReservation::OperationStorage,
                                    },
                                    operations,
                                });
                        }
                    };
                }
                SharedVoxelDataTransactionOperation::Replace { .. }
                | SharedVoxelDataTransactionOperation::Remove { .. } => {
                    removal_count = match removal_count.checked_add(1) {
                        Some(count) => count,
                        None => {
                            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                                error: SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                                    reservation: SharedVoxelDataTransactionReservation::RemovedOutcome,
                                },
                                operations,
                            });
                        }
                    };
                }
                SharedVoxelDataTransactionOperation::SetViewersExact { .. }
                | SharedVoxelDataTransactionOperation::SetViewersExactAndClearModified { .. }
                | SharedVoxelDataTransactionOperation::ClearModified { .. } => {}
            }
            expectations.push(expectation);
        }

        if self.transaction_reservation_should_fail(
            SharedVoxelDataTransactionReservation::RemovedOutcome,
        ) {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::RemovedOutcome,
                },
                operations,
            });
        }
        let mut removed_blocks = Vec::new();
        if removed_blocks.try_reserve_exact(removal_count).is_err() {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::RemovedOutcome,
                },
                operations,
            });
        }

        if self.transaction_reservation_should_fail(
            SharedVoxelDataTransactionReservation::SpatialPreparation,
        ) {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::SpatialPreparation,
                },
                operations,
            });
        }
        let mut spatial_batches: [Option<PreparedSpatialWriteBatch>; MAX_LOD] =
            std::array::from_fn(|_| None);
        for (lod_index, batch) in spatial_batches.iter_mut().enumerate().take(self.lods.len()) {
            if key_revision_counts[lod_index] == 0 {
                continue;
            }
            *batch = match SpatialLock3D::try_prepare_write_many(expectations.iter().filter_map(
                |expectation| {
                    (usize::from(expectation.location.lod_index) == lod_index)
                        .then_some(expectation.bounds)
                },
            )) {
                Ok(batch) => Some(batch),
                Err(_) => {
                    return Err(PreparedSharedVoxelDataTransactionPrepareError {
                        error: SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                            reservation: SharedVoxelDataTransactionReservation::SpatialPreparation,
                        },
                        operations,
                    });
                }
            };
        }

        Ok(PreparedSharedVoxelDataTransaction {
            data: Arc::clone(self),
            expected_settings_revision,
            operations,
            expectations,
            observations: Vec::new(),
            spatial_batches,
            insertion_counts,
            key_revision_counts,
            removed_blocks,
            shutdown_mutation_lineage: None,
            committed: false,
            #[cfg(test)]
            spatial_batch_drop_hook: None,
        })
    }

    /// Joins a settings-bound outer preview to C1 preparation. This closes the
    /// read-to-prepare lost-update window before the ordinary prepared
    /// transaction revalidation protects prepare-to-commit.
    fn prepare_transaction_from_snapshots_at_settings_revision(
        self: &Arc<Self>,
        operations: Vec<SharedVoxelDataTransactionOperation>,
        snapshots: &[SharedVoxelDataTransactionBlockSnapshot],
        expected_settings_revision: u64,
    ) -> Result<PreparedSharedVoxelDataTransaction, PreparedSharedVoxelDataTransactionPrepareError>
    {
        let mut snapshots_copy = Vec::new();
        if self.transaction_reservation_should_fail(
            SharedVoxelDataTransactionReservation::PreviewSnapshotStorage,
        ) || snapshots_copy.try_reserve_exact(snapshots.len()).is_err()
        {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::PreviewSnapshotStorage,
                },
                operations,
            });
        }
        snapshots_copy.extend_from_slice(snapshots);
        let snapshots = &mut snapshots_copy;
        snapshots.sort_unstable_by_key(|snapshot| {
            (
                snapshot.location.lod_index,
                snapshot.location.position.x,
                snapshot.location.position.y,
                snapshot.location.position.z,
            )
        });
        let mut prepared = match self
            .prepare_transaction_at_settings_revision(operations, expected_settings_revision)
        {
            Ok(prepared) => prepared,
            Err(error) => {
                let (original_error, operations) = error.into_parts();
                let error = self
                    .first_transaction_snapshot_conflict(snapshots)
                    .unwrap_or(original_error);
                return Err(PreparedSharedVoxelDataTransactionPrepareError { error, operations });
            }
        };
        if !snapshots
            .windows(2)
            .all(|pair| pair[0].location != pair[1].location)
            || snapshots.len() < prepared.expectations.len()
        {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::PreparedTransactionPreviewSetMismatch,
                operations: prepared.operations,
            });
        }
        let location_key = |location: BlockLocation| {
            (
                location.lod_index,
                location.position.x,
                location.position.y,
                location.position.z,
            )
        };
        // Prove the operation-key subset with a read-only linear merge before
        // reserving the smaller observation vector. A mismatched equal-size
        // set must not accidentally push into capacity zero.
        let mut validation_index = 0usize;
        for snapshot in snapshots.iter() {
            let Some(expectation) = prepared.expectations.get(validation_index) else {
                break;
            };
            match location_key(expectation.location).cmp(&location_key(snapshot.location)) {
                std::cmp::Ordering::Less => {
                    return Err(PreparedSharedVoxelDataTransactionPrepareError {
                        error: SharedVoxelDataMutationError::PreparedTransactionPreviewSetMismatch,
                        operations: prepared.operations,
                    });
                }
                std::cmp::Ordering::Equal => validation_index += 1,
                std::cmp::Ordering::Greater => {}
            }
        }
        if validation_index != prepared.expectations.len() {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::PreparedTransactionPreviewSetMismatch,
                operations: prepared.operations,
            });
        }
        let mut observations = Vec::new();
        let observation_count = snapshots.len() - prepared.expectations.len();
        if self.transaction_reservation_should_fail(
            SharedVoxelDataTransactionReservation::ObservationStorage,
        ) || observations.try_reserve_exact(observation_count).is_err()
        {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::ObservationStorage,
                },
                operations: prepared.operations,
            });
        }
        let mut expectation_index = 0usize;
        for snapshot in snapshots.iter() {
            let snapshot_key = location_key(snapshot.location);
            let next_expectation = prepared.expectations.get(expectation_index);
            let expectation_key =
                next_expectation.map(|expectation| location_key(expectation.location));
            if expectation_key.is_some_and(|key| key < snapshot_key) {
                return Err(PreparedSharedVoxelDataTransactionPrepareError {
                    error: SharedVoxelDataMutationError::PreparedTransactionPreviewSetMismatch,
                    operations: prepared.operations,
                });
            }
            let is_operation = expectation_key == Some(snapshot_key);
            let actual = if is_operation {
                let expectation = prepared.expectations[expectation_index];
                expectation_index += 1;
                expectation
            } else {
                match self.prepare_transaction_observation(snapshot.location) {
                    Ok(expectation) => expectation,
                    Err(error) => {
                        return Err(PreparedSharedVoxelDataTransactionPrepareError {
                            error,
                            operations: prepared.operations,
                        });
                    }
                }
            };
            let expected_present = actual.expected_block_state.is_some();
            let expected_state_matches = actual.expected_block_state.is_some_and(|state| {
                state.viewers == snapshot.viewers
                    && state.modified == snapshot.modified
                    && state.edited == snapshot.edited
                    && state.has_voxels == snapshot.has_voxels
            });
            if actual.expected_revision != snapshot.revision
                || expected_present != snapshot.present
                || (snapshot.present && !expected_state_matches)
            {
                return Err(PreparedSharedVoxelDataTransactionPrepareError {
                    error: SharedVoxelDataMutationError::ConcurrentDataMutation {
                        position: snapshot.location.position,
                        lod_index: usize::from(snapshot.location.lod_index),
                        expected_revision: snapshot.revision,
                        actual_revision: actual.expected_revision,
                    },
                    operations: prepared.operations,
                });
            }
            if !is_operation {
                observations.push(actual);
            }
        }
        if expectation_index != prepared.expectations.len() {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::PreparedTransactionPreviewSetMismatch,
                operations: prepared.operations,
            });
        }
        if !observations.is_empty() {
            let mut spatial_batches: [Option<PreparedSpatialWriteBatch>; MAX_LOD] =
                std::array::from_fn(|_| None);
            for (lod_index, batch) in spatial_batches.iter_mut().enumerate().take(self.lods.len()) {
                if prepared
                    .expectations
                    .iter()
                    .chain(observations.iter())
                    .any(|expectation| usize::from(expectation.location.lod_index) == lod_index)
                {
                    *batch = match SpatialLock3D::try_prepare_write_many(
                        prepared
                            .expectations
                            .iter()
                            .chain(observations.iter())
                            .filter_map(|expectation| {
                                (usize::from(expectation.location.lod_index) == lod_index)
                                    .then_some(expectation.bounds)
                            }),
                    ) {
                        Ok(batch) => Some(batch),
                        Err(_) => {
                            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                                error: SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                                    reservation: SharedVoxelDataTransactionReservation::SpatialPreparation,
                                },
                                operations: prepared.operations,
                            });
                        }
                    };
                }
            }
            prepared.spatial_batches = spatial_batches;
        }
        prepared.observations = observations;
        Ok(prepared)
    }

    fn first_transaction_snapshot_conflict(
        &self,
        snapshots: &[SharedVoxelDataTransactionBlockSnapshot],
    ) -> Option<SharedVoxelDataMutationError> {
        for expected in snapshots {
            let actual =
                self.transaction_block_snapshot(expected.location, Arc::clone(&expected.lineage))?;
            let state_matches = actual.present == expected.present
                && (!expected.present
                    || (actual.viewers == expected.viewers
                        && actual.modified == expected.modified
                        && actual.edited == expected.edited
                        && actual.has_voxels == expected.has_voxels));
            if actual.revision != expected.revision || !state_matches {
                return Some(SharedVoxelDataMutationError::ConcurrentDataMutation {
                    position: expected.location.position,
                    lod_index: usize::from(expected.location.lod_index),
                    expected_revision: expected.revision,
                    actual_revision: actual.revision,
                });
            }
        }
        None
    }

    fn prepare_transaction_observation(
        &self,
        location: BlockLocation,
    ) -> Result<PreparedSharedVoxelDataExpectation, SharedVoxelDataMutationError> {
        let lod_index = usize::from(location.lod_index);
        let Some(lod) = self.lods.get(lod_index) else {
            return Err(SharedVoxelDataMutationError::LodDestinationUnavailable {
                position: location.position,
                lod_index,
            });
        };
        let bounds = checked_lod_block_bounds(
            Box3i::new(location.position, Vector3i::splat(1)),
            self.block_size(),
            lod_index,
        )
        .ok_or(SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index })?;
        let state = lod.state.read().unwrap_or_else(|error| error.into_inner());
        let expected_revision = current_key_revision(&state.map, location.position);
        let expected_block_state =
            state
                .map
                .get_block(location.position)
                .map(|block| PreparedSharedVoxelDataBlockState {
                    viewers: block.viewers.get(),
                    modified: block.is_modified(),
                    edited: block.is_edited(),
                    has_voxels: block.has_voxels(),
                });
        Ok(PreparedSharedVoxelDataExpectation {
            location,
            expected_revision,
            next_revision: key_revision_value(expected_revision),
            expected_block_state,
            bounds,
        })
    }

    #[allow(dead_code)]
    fn prepare_transaction_operation(
        &self,
        operation: &SharedVoxelDataTransactionOperation,
    ) -> Result<PreparedSharedVoxelDataExpectation, SharedVoxelDataMutationError> {
        let location = operation.location();
        let lod_index = usize::from(location.lod_index);
        let Some(lod) = self.lods.get(lod_index) else {
            return Err(SharedVoxelDataMutationError::LodDestinationUnavailable {
                position: location.position,
                lod_index,
            });
        };
        if let SharedVoxelDataTransactionOperation::Insert { block, .. }
        | SharedVoxelDataTransactionOperation::Replace { block, .. } = operation
        {
            if block.lod_index() != location.lod_index {
                return Err(
                    SharedVoxelDataMutationError::PreparedTransactionBlockLodMismatch {
                        location,
                        block_lod_index: block.lod_index(),
                    },
                );
            }
            if block.has_voxels() {
                let expected_size = Vector3i::splat(self.block_size() as i32);
                let actual_size = block.voxels().size();
                if actual_size != expected_size {
                    return Err(
                        SharedVoxelDataMutationError::PreparedTransactionBlockSizeMismatch {
                            location,
                            expected_size,
                            actual_size,
                        },
                    );
                }
            }
        }
        let bounds = checked_lod_block_bounds(
            Box3i::new(location.position, Vector3i::splat(1)),
            self.block_size(),
            lod_index,
        )
        .ok_or(SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index })?;
        let state = lod.state.read().unwrap_or_else(|error| error.into_inner());
        let expected_revision = current_key_revision(&state.map, location.position);
        let next_revision = key_revision_value(expected_revision).checked_add(1).ok_or(
            SharedVoxelDataMutationError::KeyRevisionOverflow {
                position: location.position,
                lod_index,
            },
        )?;
        let expected_block_state = match operation {
            SharedVoxelDataTransactionOperation::SetViewersExact { final_viewers, .. } => {
                let Some(block) = state.map.get_block(location.position) else {
                    return Err(
                        SharedVoxelDataMutationError::PreparedTransactionExpectedPresent {
                            location,
                            actual_revision: expected_revision,
                        },
                    );
                };
                if block.viewers.get() == *final_viewers {
                    return Err(SharedVoxelDataMutationError::PreparedTransactionNoop { location });
                }
                Some(PreparedSharedVoxelDataBlockState {
                    viewers: block.viewers.get(),
                    modified: block.is_modified(),
                    edited: block.is_edited(),
                    has_voxels: block.has_voxels(),
                })
            }
            SharedVoxelDataTransactionOperation::SetViewersExactAndClearModified { .. } => {
                let Some(block) = state.map.get_block(location.position) else {
                    return Err(
                        SharedVoxelDataMutationError::PreparedTransactionExpectedPresent {
                            location,
                            actual_revision: expected_revision,
                        },
                    );
                };
                if !block.is_modified() {
                    return Err(SharedVoxelDataMutationError::PreparedTransactionNoop { location });
                }
                Some(PreparedSharedVoxelDataBlockState {
                    viewers: block.viewers.get(),
                    modified: true,
                    edited: block.is_edited(),
                    has_voxels: block.has_voxels(),
                })
            }
            SharedVoxelDataTransactionOperation::Insert { .. } => {
                if state.map.has_block(location.position) {
                    return Err(
                        SharedVoxelDataMutationError::PreparedTransactionExpectedTombstone {
                            location,
                            actual_revision: expected_revision,
                        },
                    );
                }
                None
            }
            SharedVoxelDataTransactionOperation::Replace { .. } => {
                let Some(block) = state.map.get_block(location.position) else {
                    return Err(
                        SharedVoxelDataMutationError::PreparedTransactionExpectedPresent {
                            location,
                            actual_revision: expected_revision,
                        },
                    );
                };
                Some(PreparedSharedVoxelDataBlockState {
                    viewers: block.viewers.get(),
                    modified: block.is_modified(),
                    edited: block.is_edited(),
                    has_voxels: block.has_voxels(),
                })
            }
            SharedVoxelDataTransactionOperation::ClearModified { .. } => {
                let Some(block) = state.map.get_block(location.position) else {
                    return Err(
                        SharedVoxelDataMutationError::PreparedTransactionExpectedPresent {
                            location,
                            actual_revision: expected_revision,
                        },
                    );
                };
                if !block.is_modified() {
                    return Err(SharedVoxelDataMutationError::PreparedTransactionNoop { location });
                }
                Some(PreparedSharedVoxelDataBlockState {
                    viewers: block.viewers.get(),
                    modified: true,
                    edited: block.is_edited(),
                    has_voxels: block.has_voxels(),
                })
            }
            SharedVoxelDataTransactionOperation::Remove { .. } => {
                let Some(block) = state.map.get_block(location.position) else {
                    return Err(
                        SharedVoxelDataMutationError::PreparedTransactionExpectedPresent {
                            location,
                            actual_revision: expected_revision,
                        },
                    );
                };
                Some(PreparedSharedVoxelDataBlockState {
                    viewers: block.viewers.get(),
                    modified: block.is_modified(),
                    edited: block.is_edited(),
                    has_voxels: block.has_voxels(),
                })
            }
        };
        Ok(PreparedSharedVoxelDataExpectation {
            location,
            expected_revision,
            next_revision,
            expected_block_state,
            bounds,
        })
    }

    pub(crate) fn begin_mutation(
        &self,
    ) -> Result<VoxelDataMutation<'_>, SharedVoxelDataMutationError> {
        let gate = self.mutation_gate.lock().unwrap_or_else(|e| e.into_inner());
        if self.mutation_admission_closed.load(AtomicOrdering::Acquire) {
            return Err(SharedVoxelDataMutationError::MutationAdmissionClosed);
        }
        Ok(VoxelDataMutation {
            data: self,
            _gate: gate,
        })
    }

    /// Atomically closes ordinary mutation admission and returns the exact
    /// storage-lineage capability needed by shutdown-owned prepared writes.
    pub(crate) fn close_mutation_admission_for_shutdown(
        self: &Arc<Self>,
    ) -> SharedVoxelDataShutdownMutationPermit {
        let _gate = self.mutation_gate.lock().unwrap_or_else(|e| e.into_inner());
        self.mutation_admission_closed
            .store(true, AtomicOrdering::Release);
        SharedVoxelDataShutdownMutationPermit {
            data: Arc::downgrade(self),
        }
    }

    /// Validates shutdown authority before an outer transaction starts
    /// moving payloads into escrow. This is deliberately separate from
    /// `PreparedSharedVoxelDataTransaction::authorize_shutdown_mutation`: the
    /// latter remains the commit-time capability handoff, while this early
    /// check guarantees a mismatched terrain/storage lineage cannot disturb
    /// any caller-owned draft state first.
    pub(crate) fn validate_shutdown_mutation_permit(
        self: &Arc<Self>,
        permit: &SharedVoxelDataShutdownMutationPermit,
    ) -> Result<(), SharedVoxelDataMutationError> {
        if permit.matches(self) {
            Ok(())
        } else {
            Err(SharedVoxelDataMutationError::ShutdownMutationPermitMismatch)
        }
    }

    /// Reopens mutation admission after a failed shutdown attempt. The gate
    /// serializes this transition with every writer that may already be
    /// waiting on the closed boundary.
    pub(crate) fn reopen_mutation_admission(&self) {
        let _gate = self.mutation_gate.lock().unwrap_or_else(|e| e.into_inner());
        self.mutation_admission_closed
            .store(false, AtomicOrdering::Release);
    }

    #[cfg(test)]
    pub(crate) fn is_mutation_admission_closed(&self) -> bool {
        self.mutation_admission_closed.load(AtomicOrdering::Acquire)
    }

    #[cfg(test)]
    pub(crate) fn set_test_settings_revision(&self, revision: u64) {
        self.begin_mutation()
            .expect("test settings mutation is admitted")
            .set_settings_revision_for_test(revision);
    }

    #[cfg(test)]
    pub(crate) fn increment_test_settings_revision(&self) {
        let mutation = self
            .begin_mutation()
            .expect("test settings mutation is admitted");
        let current = mutation
            .data
            .settings
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .revision;
        mutation.set_settings_revision_for_test(
            current
                .checked_add(1)
                .expect("test settings-conflict hook requires revision headroom"),
        );
    }

    #[cfg(test)]
    pub(crate) fn set_test_transaction_live_spatial_registry_fail_lod(
        &self,
        fail_lod: Option<usize>,
    ) {
        for (lod_index, spatial_lock) in self.spatial_locks.iter().enumerate() {
            spatial_lock.set_test_write_registry_reservation_failure(fail_lod == Some(lod_index));
        }
    }

    #[cfg(test)]
    fn lod_structural_revision(&self, lod_index: usize) -> Option<u64> {
        let lod = self.lods.get(lod_index)?;
        let state = lod.state.read().unwrap_or_else(|e| e.into_inner());
        Some(state.structural_revision)
    }

    #[cfg(test)]
    pub(crate) fn key_revision(
        &self,
        block_pos: Vector3i,
        lod_index: usize,
    ) -> Option<VoxelDataKeyRevision> {
        let lod = self.lods.get(lod_index)?;
        let state = lod.state.read().unwrap_or_else(|e| e.into_inner());
        let revision = state.map.key_revision(block_pos);
        Some(if state.map.has_block(block_pos) {
            VoxelDataKeyRevision::Present(revision)
        } else {
            VoxelDataKeyRevision::Tombstone(revision)
        })
    }

    pub fn with_lod_map<R>(&self, lod_index: usize, f: impl FnOnce(&VoxelDataMap) -> R) -> R {
        let lod = self
            .lods
            .get(lod_index)
            .expect("LOD index is outside the loaded range");
        let state = lod.state.read().unwrap_or_else(|e| e.into_inner());
        f(&state.map)
    }

    #[cfg(test)]
    pub(crate) fn with_lod_map_mut<R>(
        &self,
        lod_index: usize,
        f: impl FnOnce(&mut VoxelDataMap) -> R,
    ) -> R {
        let lod = self
            .lods
            .get(lod_index)
            .expect("LOD index is outside the loaded range");
        let mut state = lod.state.write().unwrap_or_else(|e| e.into_inner());
        f(&mut state.map)
    }

    #[cfg(test)]
    pub(crate) fn try_lock(&self) -> Option<SharedVoxelDataLodWriteGuard<'_>> {
        self.try_lod_map_write(0)
    }

    #[cfg(test)]
    pub(crate) fn try_lod_map_write(
        &self,
        lod_index: usize,
    ) -> Option<SharedVoxelDataLodWriteGuard<'_>> {
        let lod = self.lods.get(lod_index)?;
        match lod.state.try_write() {
            Ok(guard) => Some(SharedVoxelDataLodWriteGuard { guard }),
            Err(TryLockError::Poisoned(e)) => Some(SharedVoxelDataLodWriteGuard {
                guard: e.into_inner(),
            }),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    pub fn try_lod_map_read(&self, lod_index: usize) -> Option<SharedVoxelDataLodReadGuard<'_>> {
        let lod = self.lods.get(lod_index)?;
        match lod.state.try_read() {
            Ok(guard) => Some(SharedVoxelDataLodReadGuard { guard }),
            Err(TryLockError::Poisoned(e)) => Some(SharedVoxelDataLodReadGuard {
                guard: e.into_inner(),
            }),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    pub fn has_all_blocks_in_area(&self, blocks_box: Box3i, lod_index: usize) -> bool {
        if lod_index >= self.lods.len() {
            return false;
        }
        self.with_lod_map(lod_index, |map| {
            blocks_box.all_cells_match(|pos| map.has_block(pos))
        })
    }

    pub fn try_set_block(
        &self,
        block_pos: Vector3i,
        block: VoxelDataBlock,
    ) -> Result<bool, SharedVoxelDataMutationError> {
        let lod_index = usize::from(block.lod_index());
        assert!(lod_index < self.lods.len(), "block LOD is not loaded");
        if block.has_voxels() {
            assert_eq!(
                block.voxels().size(),
                Vector3i::splat(self.block_size() as i32),
                "block voxels must match VoxelData block size"
            );
        }
        self.begin_mutation()?
            .try_insert_block(block_pos, block, lod_index)
    }

    pub fn view_area(
        &self,
        mut blocks_box: Box3i,
        lod_index: usize,
        missing_blocks: Option<&mut Vec<Vector3i>>,
        found_blocks_positions: Option<&mut Vec<Vector3i>>,
        found_blocks: Option<&mut Vec<VoxelDataBlock>>,
    ) -> Result<(), SharedVoxelDataMutationError> {
        let Some(bounds_in_blocks) =
            try_bounds_in_lod_blocks(self.bounds(), self.block_size() as i32, lod_index)
        else {
            if let Some(out) = missing_blocks {
                try_extend_missing_blocks_if_representable(out, blocks_box)?;
            }
            return Ok(());
        };
        let Ok(clipped_blocks_box) = checked_box_intersection(blocks_box, bounds_in_blocks) else {
            return Ok(());
        };
        blocks_box = clipped_blocks_box;

        if lod_index >= self.lods.len() {
            if let Some(out) = missing_blocks {
                try_extend_missing_blocks_if_representable(out, blocks_box)?;
            }
            return Ok(());
        }

        self.begin_mutation()?.view_area(
            blocks_box,
            lod_index,
            missing_blocks,
            found_blocks_positions,
            found_blocks,
        )
    }

    pub fn unview_area(
        &self,
        mut blocks_box: Box3i,
        lod_index: usize,
        removed_blocks: Option<&mut Vec<Vector3i>>,
        missing_blocks: Option<&mut Vec<Vector3i>>,
        to_save: Option<&mut Vec<BlockToSave>>,
    ) -> Result<(), SharedVoxelDataMutationError> {
        let Some(bounds_in_blocks) =
            try_bounds_in_lod_blocks(self.bounds(), self.block_size() as i32, lod_index)
        else {
            if let Some(out) = missing_blocks {
                try_extend_missing_blocks_if_representable(out, blocks_box)?;
            }
            return Ok(());
        };
        let Ok(clipped_blocks_box) = checked_box_intersection(blocks_box, bounds_in_blocks) else {
            return Ok(());
        };
        blocks_box = clipped_blocks_box;

        if lod_index >= self.lods.len() {
            if let Some(out) = missing_blocks {
                try_extend_missing_blocks_if_representable(out, blocks_box)?;
            }
            return Ok(());
        }

        self.begin_mutation()?.unview_area(
            blocks_box,
            lod_index,
            removed_blocks,
            missing_blocks,
            to_save,
        )
    }

    #[cfg(test)]
    pub(crate) fn try_edit_voxel(&self, value: u64, pos: Vector3i, channel_index: usize) -> bool {
        self.try_edit_voxel_checked(value, pos, channel_index)
            .unwrap_or(false)
    }

    /// Prepares one voxel edit and every configured parent LOD as one exact
    /// shared-data transaction without changing live storage.
    ///
    /// Generator callbacks and all fallible allocation happen before commit.
    /// `Ok(None)` is reserved for out-of-bounds coordinates or blocks which
    /// the current streaming policy does not permit this path to materialize.
    #[allow(dead_code)] // Task 3 adopts this prepared storage boundary.
    pub(crate) fn prepare_voxel_edit(
        self: &Arc<Self>,
        value: u64,
        position: Vector3i,
        channel_index: usize,
    ) -> Result<Option<PreparedSharedVoxelDataEdit>, SharedVoxelDataMutationError> {
        if channel_index >= crate::storage::voxel_buffer::MAX_CHANNELS {
            return Err(SharedVoxelDataMutationError::InvalidChannel { channel_index });
        }

        let preview = self.begin_transaction_preview();
        let settings = preview.settings();
        let validated_bounds =
            checked_box_intersection(settings.bounds_in_voxels, settings.bounds_in_voxels)
                .map_err(|_| SharedVoxelDataMutationError::SpatialBoundsOverflow {
                    lod_index: 0,
                })?;
        if !validated_bounds.contains_point(position) {
            return Ok(None);
        }

        struct EditDraft {
            snapshot: SharedVoxelDataTransactionBlockSnapshot,
            block: VoxelDataBlock,
        }

        let block_size = preview.block_size();
        let block_size_i32 = block_size as i32;
        let lod_count = self.lod_count();
        let mut drafts = Vec::new();
        drafts
            .try_reserve_exact(lod_count)
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;

        let lod0_location = BlockLocation {
            position: VoxelDataMap::voxel_to_block_b(position, self.block_size_po2()),
            lod_index: 0,
        };
        let lod0_bounds = checked_lod_block_bounds(
            Box3i::new(lod0_location.position, Vector3i::splat(1)),
            block_size,
            0,
        )
        .ok_or(SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index: 0 })?;
        let (lod0_snapshot, lod0_resident) = preview.block_draft(lod0_location).ok_or(
            SharedVoxelDataMutationError::LodDestinationUnavailable {
                position: lod0_location.position,
                lod_index: 0,
            },
        )?;
        let lod0_needs_materialization = lod0_resident
            .as_ref()
            .is_none_or(|block| !block.has_voxels());
        if lod0_needs_materialization
            && (settings.streaming_enabled || !settings.full_load_completed)
        {
            return Ok(None);
        }
        let mut lod0_block =
            lod0_resident.unwrap_or_else(|| VoxelDataBlock::empty(lod0_location.lod_index));
        if lod0_needs_materialization {
            let mut voxels = create_block_buffer(block_size_i32, settings.format);
            if let Some(generator) = settings.generator.as_deref() {
                generator.generate_block(VoxelQueryData {
                    buffer: &mut voxels,
                    origin_in_voxels: lod0_bounds.min_pos,
                    lod: 0,
                });
            }
            lod0_block.set_voxels(voxels);
        }
        let local_position = position - lod0_bounds.min_pos;
        lod0_block.voxels_mut().set_voxel(
            value,
            local_position.x,
            local_position.y,
            local_position.z,
            channel_index,
        );
        lod0_block.set_modified(true);
        lod0_block.set_edited(true);
        lod0_block.set_needs_lodding(false);
        drafts.push(EditDraft {
            snapshot: lod0_snapshot,
            block: lod0_block,
        });

        let half_block_size = block_size_i32 >> 1;
        for lod_index in 1..lod_count {
            let source_location = drafts
                .last()
                .expect("LOD0 edit draft was prepared")
                .snapshot
                .location();
            let destination_location = BlockLocation {
                position: source_location.position >> 1,
                lod_index: lod_index as u8,
            };
            let destination_bounds = checked_lod_block_bounds(
                Box3i::new(destination_location.position, Vector3i::splat(1)),
                block_size,
                lod_index,
            )
            .ok_or(SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index })?;
            let (snapshot, resident) = preview.block_draft(destination_location).ok_or(
                SharedVoxelDataMutationError::LodDestinationUnavailable {
                    position: destination_location.position,
                    lod_index,
                },
            )?;
            let needs_materialization = resident.as_ref().is_none_or(|block| !block.has_voxels());
            if needs_materialization
                && (settings.streaming_enabled || !settings.full_load_completed)
            {
                return Ok(None);
            }
            let mut destination =
                resident.unwrap_or_else(|| VoxelDataBlock::empty(destination_location.lod_index));
            if needs_materialization {
                let mut voxels = create_block_buffer(block_size_i32, settings.format);
                if let Some(generator) = settings.generator.as_deref() {
                    generator.generate_block(VoxelQueryData {
                        buffer: &mut voxels,
                        origin_in_voxels: destination_bounds.min_pos,
                        lod: lod_index as u32,
                    });
                }
                destination.set_voxels(voxels);
            }
            let parity = Vector3i::new(
                source_location.position.x.rem_euclid(2),
                source_location.position.y.rem_euclid(2),
                source_location.position.z.rem_euclid(2),
            );
            let destination_offset = parity * half_block_size;
            let source = &drafts
                .last()
                .expect("previous LOD edit draft was prepared")
                .block;
            source.voxels().downscale_to(
                destination.voxels_mut(),
                Vector3i::zero(),
                source.voxels().size(),
                destination_offset,
            );
            destination.set_modified(true);
            destination.set_needs_lodding(false);
            drafts.push(EditDraft {
                snapshot,
                block: destination,
            });
        }

        let mut block_revision = None;
        for draft in &drafts {
            let location = draft.snapshot.location();
            let next_revision = key_revision_value(draft.snapshot.revision())
                .checked_add(1)
                .ok_or(SharedVoxelDataMutationError::KeyRevisionOverflow {
                    position: location.position,
                    lod_index: usize::from(location.lod_index),
                })?;
            if location == lod0_location {
                block_revision = Some(next_revision);
            }
        }
        let block_revision = block_revision.expect("prepared LOD0 edit has one next revision");

        let mut operations = Vec::new();
        let mut snapshots = Vec::new();
        let mut inserted_locations = Vec::new();
        operations
            .try_reserve_exact(drafts.len())
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        snapshots
            .try_reserve_exact(drafts.len())
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        inserted_locations
            .try_reserve_exact(drafts.len())
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        for draft in drafts {
            let location = draft.snapshot.location();
            let operation = if draft.snapshot.is_present() {
                SharedVoxelDataTransactionOperation::Replace {
                    location,
                    block: draft.block,
                }
            } else {
                inserted_locations.push(location);
                SharedVoxelDataTransactionOperation::Insert {
                    location,
                    block: draft.block,
                    final_viewers: 0,
                }
            };
            snapshots.push(draft.snapshot);
            operations.push(operation);
        }
        #[cfg(test)]
        self.notify_test_edit_phase(
            SharedVoxelDataEditPhase::PreparedVoxelEditDraftedBeforeTransactionPrepare,
        );
        let transaction = preview
            .prepare_transaction(operations, &snapshots)
            .map_err(|error| error.into_parts().0)?;
        Ok(Some(PreparedSharedVoxelDataEdit {
            transaction,
            edited_block: lod0_location,
            block_revision,
            inserted_locations,
        }))
    }

    pub fn try_edit_voxel_checked(
        &self,
        value: u64,
        pos: Vector3i,
        channel_index: usize,
    ) -> Result<bool, SharedVoxelDataMutationError> {
        if channel_index >= crate::storage::voxel_buffer::MAX_CHANNELS {
            return Err(SharedVoxelDataMutationError::InvalidChannel { channel_index });
        }
        let settings = self.settings_snapshot();
        if !settings.bounds_in_voxels.contains_point(pos) {
            return Ok(false);
        }

        let block_size = self.block_size() as i32;
        let block_pos = VoxelDataMap::voxel_to_block_b(pos, self.block_size_po2());
        let (expected_key_revision, needs_materialization) = self.with_lod_map(0, |map| {
            (
                current_key_revision(map, block_pos),
                map.get_block(block_pos)
                    .is_none_or(|block| !block.has_voxels()),
            )
        });
        if needs_materialization && (settings.streaming_enabled || !settings.full_load_completed) {
            return Ok(false);
        }

        let prepared = needs_materialization.then(|| {
            let mut voxels = create_block_buffer(block_size, settings.format);
            if let Some(generator) = settings.generator.as_deref() {
                generator.generate_block(VoxelQueryData {
                    buffer: &mut voxels,
                    origin_in_voxels: block_pos * block_size,
                    lod: 0,
                });
            }
            voxels
        });

        self.begin_mutation()?.commit_edit_voxel(
            settings.revision,
            expected_key_revision,
            prepared,
            value,
            pos,
            block_pos,
            channel_index,
        )?;
        Ok(true)
    }

    /// Propagate modified LOD0 blocks into the shared higher-LOD maps.
    ///
    /// This is the lock-aware counterpart of [`VoxelData::update_lods`]. It
    /// prepares source and destination drafts without commit locks, then
    /// acquires all affected spatial batches and LOD maps in ascending LOD
    /// order for one atomic revision-checked commit.
    pub fn update_lods_from_lod0_blocks_checked(
        &self,
        modified_lod0_blocks: &[Vector3i],
    ) -> Result<Vec<BlockLocation>, SharedVoxelDataMutationError> {
        let lod_count = self.lods.len();
        if modified_lod0_blocks.is_empty() || lod_count == 0 {
            return Ok(Vec::new());
        }

        let settings = self.settings_snapshot();
        let block_size = self.block_size() as i32;
        let half_block_size = block_size >> 1;
        let last_lod = lod_count - 1;
        let mut worklists = vec![Vec::new(); lod_count];
        let mut queued = (0..lod_count)
            .map(|_| HashSet::new())
            .collect::<Vec<HashSet<Vector3i>>>();
        let mut drafts =
            HashMap::<(usize, Vector3i), (VoxelDataKeyRevision, VoxelDataBlock)>::new();

        let mut lod0_sources = modified_lod0_blocks.to_vec();
        lod0_sources.sort_unstable_by_key(|position| (position.z, position.x, position.y));
        lod0_sources.dedup();
        for block_pos in lod0_sources {
            let snapshot = self.with_lod_map(0, |map| {
                let block = map.get_block(block_pos)?;
                block
                    .has_voxels()
                    .then(|| (current_key_revision(map, block_pos), clone_block(block)))
            });
            let Some((revision, mut block)) = snapshot else {
                continue;
            };
            block.set_needs_lodding(false);
            drafts.insert((0, block_pos), (revision, block));
            queued[0].insert(block_pos);
            worklists[0].push(block_pos);
        }

        for dst_lod in 1..lod_count {
            let src_lod = dst_lod - 1;
            let sources = std::mem::take(&mut worklists[src_lod]);
            for src_pos in sources {
                let Some((_, source_block)) = drafts.get_mut(&(src_lod, src_pos)) else {
                    continue;
                };
                source_block.set_needs_lodding(false);
                if !source_block.has_voxels() {
                    continue;
                }
                let source = source_block.voxels().copy_to_owned();
                let dst_pos = src_pos >> 1;

                if let std::collections::hash_map::Entry::Vacant(entry) =
                    drafts.entry((dst_lod, dst_pos))
                {
                    let (expected_revision, snapshot) = self.with_lod_map(dst_lod, |map| {
                        (
                            current_key_revision(map, dst_pos),
                            map.get_block(dst_pos).map(clone_block),
                        )
                    });
                    let destination_needs_voxels =
                        snapshot.as_ref().is_none_or(|block| !block.has_voxels());
                    if destination_needs_voxels && settings.streaming_enabled {
                        return Err(SharedVoxelDataMutationError::LodDestinationUnavailable {
                            position: dst_pos,
                            lod_index: dst_lod,
                        });
                    }

                    let block = if destination_needs_voxels {
                        let bounds = checked_lod_block_bounds(
                            Box3i::new(dst_pos, Vector3i::splat(1)),
                            self.block_size(),
                            dst_lod,
                        )
                        .ok_or(
                            SharedVoxelDataMutationError::SpatialBoundsOverflow {
                                lod_index: dst_lod,
                            },
                        )?;
                        let mut voxels = create_block_buffer(block_size, settings.format);
                        if let Some(generator) = settings.generator.as_deref() {
                            generator.generate_block(VoxelQueryData {
                                buffer: &mut voxels,
                                origin_in_voxels: bounds.min_pos,
                                lod: dst_lod as u32,
                            });
                        }
                        if let Some(mut block) = snapshot {
                            block.set_voxels(voxels);
                            block
                        } else {
                            VoxelDataBlock::with_voxels(voxels, dst_lod as u8)
                        }
                    } else {
                        snapshot.expect("voxel-bearing destination snapshot exists")
                    };
                    entry.insert((expected_revision, block));
                }

                let (_, destination) = drafts
                    .get_mut(&(dst_lod, dst_pos))
                    .expect("destination draft was prepared");
                destination.set_modified(true);
                let dst_offset = (src_pos - (dst_pos << 1)) * half_block_size;
                source.downscale_to(
                    destination.voxels_mut(),
                    Vector3i::zero(),
                    source.size(),
                    dst_offset,
                );
                if dst_lod != last_lod && queued[dst_lod].insert(dst_pos) {
                    destination.set_needs_lodding(true);
                    worklists[dst_lod].push(dst_pos);
                }
            }
        }

        let mut changes = drafts
            .into_iter()
            .map(|((lod_index, position), (expected_revision, block))| {
                let next_revision = key_revision_value(expected_revision).checked_add(1).ok_or(
                    SharedVoxelDataMutationError::KeyRevisionOverflow {
                        position,
                        lod_index,
                    },
                )?;
                let bounds = checked_lod_block_bounds(
                    Box3i::new(position, Vector3i::splat(1)),
                    self.block_size(),
                    lod_index,
                )
                .ok_or(SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index })?;
                Ok(PreparedLodBlockMutation {
                    position,
                    lod_index,
                    expected_revision,
                    next_revision,
                    bounds,
                    block,
                })
            })
            .collect::<Result<Vec<_>, SharedVoxelDataMutationError>>()?;
        changes.sort_unstable_by_key(|change| {
            (
                change.lod_index,
                change.position.z,
                change.position.x,
                change.position.y,
            )
        });
        let updated = changes
            .iter()
            .map(|change| BlockLocation {
                position: change.position,
                lod_index: change.lod_index as u8,
            })
            .collect::<Vec<_>>();
        let mut spatial_batches: [Option<PreparedSpatialWriteBatch>; MAX_LOD] =
            std::array::from_fn(|_| None);
        for (lod_index, batch) in spatial_batches.iter_mut().enumerate().take(lod_count) {
            if changes.iter().any(|change| change.lod_index == lod_index) {
                *batch = Some(SpatialLock3D::prepare_write_many(
                    changes.iter().filter_map(|change| {
                        (change.lod_index == lod_index).then_some(change.bounds)
                    }),
                ));
            }
        }
        let mut retired = Vec::new();
        retired
            .try_reserve_exact(changes.len())
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        #[cfg(test)]
        self.notify_test_edit_phase(SharedVoxelDataEditPhase::LodUpdatePreparedBeforeMutationGate);
        self.begin_mutation()?.commit_lod_updates(
            settings.revision,
            changes,
            spatial_batches,
            retired,
        )?;
        Ok(updated)
    }

    pub fn try_set_voxel_checked(
        &self,
        value: u64,
        pos: Vector3i,
        channel_index: usize,
    ) -> Result<bool, SharedVoxelDataMutationError> {
        if channel_index >= crate::storage::voxel_buffer::MAX_CHANNELS {
            return Err(SharedVoxelDataMutationError::InvalidChannel { channel_index });
        }
        let settings = self.settings_snapshot();
        if !settings.bounds_in_voxels.contains_point(pos) {
            return Ok(false);
        }
        let block_pos = VoxelDataMap::voxel_to_block_b(pos, self.block_size_po2());
        let block_size = self.block_size() as i32;
        let (expected_key_revision, block_state) = self.with_lod_map(0, |map| {
            (
                current_key_revision(map, block_pos),
                map.get_block(block_pos).map(VoxelDataBlock::has_voxels),
            )
        });
        if block_state.is_none() && (settings.streaming_enabled || !settings.full_load_completed) {
            return Ok(false);
        }
        let prepared_voxels = (!matches!(block_state, Some(true)))
            .then(|| create_block_buffer(block_size, settings.format));
        let bounds = checked_lod_block_bounds(
            Box3i::new(block_pos, Vector3i::splat(1)),
            self.block_size(),
            0,
        )
        .ok_or(SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index: 0 })?;
        let spatial_batch = SpatialLock3D::prepare_write_many([bounds]);

        self.begin_mutation()?.commit_set_voxel(
            settings.revision,
            expected_key_revision,
            prepared_voxels,
            value,
            pos,
            block_pos,
            channel_index,
            spatial_batch,
        )?;
        Ok(true)
    }

    pub fn mark_area_modified_checked(
        &self,
        voxel_box: Box3i,
        require_lod_updates: bool,
    ) -> Result<Vec<Vector3i>, SharedVoxelDataMutationError> {
        let voxel_box = checked_box_intersection(voxel_box, voxel_box)
            .map_err(|_| SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index: 0 })?;
        if voxel_box.is_empty() {
            return Ok(Vec::new());
        }
        let blocks_box = voxel_box.downscaled(self.block_size() as i32);
        let mut prepared = self.with_lod_map(0, |map| {
            map.block_positions()
                .filter(|position| blocks_box.contains_point(*position))
                .filter_map(|position| {
                    let block = map.get_block(position)?;
                    if !block.has_voxels() {
                        return None;
                    }
                    let set_needs_lodding = require_lod_updates && !block.needs_lodding();
                    if block.is_modified() && block.is_edited() && !set_needs_lodding {
                        return None;
                    }
                    Some((|| {
                        let expected_revision = current_key_revision(map, position);
                        let next_revision = key_revision_value(expected_revision)
                            .checked_add(1)
                            .ok_or(SharedVoxelDataMutationError::KeyRevisionOverflow {
                                position,
                                lod_index: 0,
                            })?;
                        let bounds = checked_lod_block_bounds(
                            Box3i::new(position, Vector3i::splat(1)),
                            self.block_size(),
                            0,
                        )
                        .ok_or(
                            SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index: 0 },
                        )?;
                        Ok(PreparedMarkMutation {
                            position,
                            expected_revision,
                            next_revision,
                            bounds,
                            set_needs_lodding,
                        })
                    })())
                })
                .collect::<Result<Vec<_>, SharedVoxelDataMutationError>>()
        })?;
        prepared.sort_unstable_by_key(|change| {
            (change.position.z, change.position.x, change.position.y)
        });
        let newly_needing_lod = prepared
            .iter()
            .filter(|change| change.set_needs_lodding)
            .map(|change| change.position)
            .collect::<Vec<_>>();
        let spatial_batch =
            SpatialLock3D::prepare_write_many(prepared.iter().map(|change| change.bounds));
        self.begin_mutation()?
            .commit_mark_area_modified(&prepared, spatial_batch)?;
        Ok(newly_needing_lod)
    }

    pub fn block_snapshot(&self, block_pos: Vector3i, lod_index: usize) -> Option<VoxelDataBlock> {
        if lod_index >= self.lods.len() {
            return None;
        }
        self.with_lod_map(lod_index, |map| map.get_block(block_pos).map(clone_block))
    }

    fn transaction_block_snapshot(
        &self,
        location: BlockLocation,
        lineage: Arc<SharedVoxelDataTransactionPreviewLineage>,
    ) -> Option<SharedVoxelDataTransactionBlockSnapshot> {
        let lod_index = usize::from(location.lod_index);
        if lod_index >= self.lods.len() {
            return None;
        }
        self.with_lod_map(lod_index, |map| {
            let revision = current_key_revision(map, location.position);
            match map.get_block(location.position) {
                Some(block) => SharedVoxelDataTransactionBlockSnapshot {
                    lineage: Arc::clone(&lineage),
                    location,
                    revision,
                    present: true,
                    viewers: block.viewers.get(),
                    modified: block.is_modified(),
                    edited: block.is_edited(),
                    has_voxels: block.has_voxels(),
                },
                None => SharedVoxelDataTransactionBlockSnapshot {
                    lineage,
                    location,
                    revision,
                    present: false,
                    viewers: 0,
                    modified: false,
                    edited: false,
                    has_voxels: false,
                },
            }
            .into()
        })
    }

    #[allow(dead_code)] // Used by the Task 3 terrain edit preparation path.
    fn transaction_block_draft(
        &self,
        location: BlockLocation,
        lineage: Arc<SharedVoxelDataTransactionPreviewLineage>,
    ) -> Option<(
        SharedVoxelDataTransactionBlockSnapshot,
        Option<VoxelDataBlock>,
    )> {
        let lod_index = usize::from(location.lod_index);
        if lod_index >= self.lods.len() {
            return None;
        }
        self.with_lod_map(lod_index, |map| {
            let revision = current_key_revision(map, location.position);
            let block = map.get_block(location.position);
            let snapshot = match block {
                Some(block) => SharedVoxelDataTransactionBlockSnapshot {
                    lineage,
                    location,
                    revision,
                    present: true,
                    viewers: block.viewers.get(),
                    modified: block.is_modified(),
                    edited: block.is_edited(),
                    has_voxels: block.has_voxels(),
                },
                None => SharedVoxelDataTransactionBlockSnapshot {
                    lineage,
                    location,
                    revision,
                    present: false,
                    viewers: 0,
                    modified: false,
                    edited: false,
                    has_voxels: false,
                },
            };
            (snapshot, block.map(clone_block))
        })
        .into()
    }

    pub fn block_count(&self) -> usize {
        self.lods
            .iter()
            .map(|lod| {
                lod.state
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .map
                    .block_count()
            })
            .sum()
    }

    /// Copies every dirty resident block into save payloads and clears its
    /// dirty flag. Callers retain the returned payloads until persistence
    /// succeeds, so a failed save remains retryable without keeping the
    /// resident block dirty.
    pub fn consume_all_modifications_checked(
        &self,
    ) -> Result<Vec<BlockToSave>, SharedVoxelDataMutationError> {
        let mut changes = Vec::new();
        let mut saves = Vec::new();
        for lod_index in 0..self.lods.len() {
            let mut lod_changes = self.with_lod_map(lod_index, |map| {
                map.block_positions()
                    .filter_map(|position| {
                        let block = map.get_block(position)?;
                        block.is_modified().then(|| {
                            let expected_revision = current_key_revision(map, position);
                            let next_revision = key_revision_value(expected_revision)
                                .checked_add(1)
                                .ok_or(SharedVoxelDataMutationError::KeyRevisionOverflow {
                                    position,
                                    lod_index,
                                })?;
                            let bounds = checked_lod_block_bounds(
                                Box3i::new(position, Vector3i::splat(1)),
                                self.block_size(),
                                lod_index,
                            )
                            .ok_or(
                                SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index },
                            )?;
                            let voxels = block.has_voxels().then(|| block.voxels().copy_to_owned());
                            Ok((
                                PreparedDirtyBlockMutation {
                                    position,
                                    lod_index,
                                    expected_revision,
                                    next_revision,
                                    bounds,
                                },
                                BlockToSave {
                                    voxels,
                                    position,
                                    lod_index: lod_index as u8,
                                    block_revision: match expected_revision {
                                        VoxelDataKeyRevision::Present(revision) => revision,
                                        VoxelDataKeyRevision::Tombstone(_) => unreachable!(
                                            "a dirty resident block has a present revision"
                                        ),
                                    },
                                },
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, SharedVoxelDataMutationError>>()
            })?;
            lod_changes.sort_unstable_by_key(|(change, _)| {
                (change.position.z, change.position.x, change.position.y)
            });
            for (change, save) in lod_changes {
                changes.push(change);
                saves.push(save);
            }
        }
        let mut spatial_batches: [Option<PreparedSpatialWriteBatch>; MAX_LOD] =
            std::array::from_fn(|_| None);
        for (lod_index, batch) in spatial_batches.iter_mut().enumerate().take(self.lods.len()) {
            if changes.iter().any(|change| change.lod_index == lod_index) {
                *batch = Some(SpatialLock3D::prepare_write_many(
                    changes.iter().filter_map(|change| {
                        (change.lod_index == lod_index).then_some(change.bounds)
                    }),
                ));
            }
        }
        #[cfg(test)]
        self.notify_test_edit_phase(
            SharedVoxelDataEditPhase::DirtySnapshotPreparedBeforeMutationGate,
        );
        self.begin_mutation()?
            .commit_dirty_consumption(&changes, spatial_batches)?;
        Ok(saves)
    }

    pub(crate) fn read_region(
        &self,
        lod_index: usize,
        voxel_box: Box3i,
    ) -> SharedVoxelDataReadRegion<'_> {
        let bounds = bounds_from_box(voxel_box);
        let lock = self.spatial_lock(lod_index);
        lock.lock_read(bounds);
        SharedVoxelDataReadRegion { lock, bounds }
    }

    #[cfg(test)]
    pub(crate) fn try_read_region(
        &self,
        lod_index: usize,
        voxel_box: Box3i,
    ) -> Option<SharedVoxelDataReadRegion<'_>> {
        let bounds = bounds_from_box(voxel_box);
        let lock = self.spatial_lock(lod_index);
        if lock.try_lock_read(bounds) {
            Some(SharedVoxelDataReadRegion { lock, bounds })
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(crate) fn try_write_region(
        &self,
        lod_index: usize,
        voxel_box: Box3i,
    ) -> Option<SharedVoxelDataWriteRegion<'_>> {
        let bounds = bounds_from_box(voxel_box);
        let lock = self.spatial_lock(lod_index);
        if lock.try_lock_write(bounds) {
            Some(SharedVoxelDataWriteRegion { lock, bounds })
        } else {
            None
        }
    }

    pub fn locked_region_count(&self, lod_index: usize) -> usize {
        self.spatial_lock(lod_index).locked_boxes_count()
    }

    fn spatial_lock(&self, lod_index: usize) -> &SpatialLock3D {
        self.spatial_locks
            .get(lod_index)
            .expect("LOD index is outside the supported range")
    }
}

impl SharedVoxelDataTransactionPreview {
    /// Returns the immutable settings snapshot which defines this preview.
    #[allow(dead_code)] // Task 3 consumes this through voxel-edit preparation.
    pub(crate) const fn settings(&self) -> &SharedVoxelDataSettingsSnapshot {
        &self.settings
    }

    /// Returns the block size used to derive spatial bounds for this preview.
    pub(crate) fn block_size(&self) -> u32 {
        self.data.block_size()
    }

    /// Samples one block through this preview's concrete shared-data lineage.
    pub(crate) fn block_snapshot(
        &self,
        location: BlockLocation,
    ) -> Option<SharedVoxelDataTransactionBlockSnapshot> {
        self.data
            .transaction_block_snapshot(location, Arc::clone(&self.lineage))
    }

    /// Samples transaction metadata and the matching deep-owned block through
    /// one map read, closing the snapshot-to-clone lost-update window.
    #[allow(dead_code)] // Used by the Task 3 terrain edit preparation path.
    fn block_draft(
        &self,
        location: BlockLocation,
    ) -> Option<(
        SharedVoxelDataTransactionBlockSnapshot,
        Option<VoxelDataBlock>,
    )> {
        self.data
            .transaction_block_draft(location, Arc::clone(&self.lineage))
    }

    /// Copies every dirty resident payload and captures its opaque transaction
    /// snapshot during the same map-read observation.
    ///
    /// No live state is changed here. The returned `ClearModified` operations
    /// must be joined to the caller's one outer transaction; preparation or
    /// commit conflicts therefore preserve both a newer edit and its dirty
    /// flag before the first storage write.
    pub(crate) fn prepare_resident_dirty_copies(
        &self,
    ) -> Result<PreparedResidentDirtyCopies, SharedVoxelDataMutationError> {
        let map_guards: [Option<StdRwLockReadGuard<'_, SharedVoxelDataLodState>>; MAX_LOD] =
            std::array::from_fn(|lod_index| {
                self.data
                    .lods
                    .get(lod_index)
                    .map(|lod| lod.state.read().unwrap_or_else(|error| error.into_inner()))
            });
        let dirty_count = map_guards
            .iter()
            .flatten()
            .map(|state| {
                state
                    .map
                    .block_positions()
                    .filter(|position| {
                        state
                            .map
                            .get_block(*position)
                            .is_some_and(VoxelDataBlock::is_modified)
                    })
                    .count()
            })
            .try_fold(0usize, usize::checked_add)
            .ok_or(SharedVoxelDataMutationError::CapacityReservationFailed)?;

        let mut operations = Vec::<SharedVoxelDataTransactionOperation>::new();
        if self.data.transaction_reservation_should_fail(
            SharedVoxelDataTransactionReservation::OperationStorage,
        ) || operations.try_reserve_exact(dirty_count).is_err()
        {
            return Err(
                SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::OperationStorage,
                },
            );
        }
        let mut snapshots = Vec::<SharedVoxelDataTransactionBlockSnapshot>::new();
        if self.data.transaction_reservation_should_fail(
            SharedVoxelDataTransactionReservation::PreviewSnapshotStorage,
        ) || snapshots.try_reserve_exact(dirty_count).is_err()
        {
            return Err(
                SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::PreviewSnapshotStorage,
                },
            );
        }
        let mut payloads = Vec::<RevisionedBlockToSave>::new();
        if self.data.transaction_reservation_should_fail(
            SharedVoxelDataTransactionReservation::ResidentDirtyPayloadStorage,
        ) || payloads.try_reserve_exact(dirty_count).is_err()
        {
            return Err(
                SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::ResidentDirtyPayloadStorage,
                },
            );
        }

        for (lod_index, state) in map_guards.iter().enumerate() {
            let Some(state) = state else {
                continue;
            };
            for position in state.map.block_positions() {
                let block = state
                    .map
                    .get_block(position)
                    .expect("a position yielded by the resident map remains present");
                if !block.is_modified() {
                    continue;
                }
                let location = BlockLocation {
                    position,
                    lod_index: lod_index as u8,
                };
                if !block.has_voxels() {
                    return Err(SharedVoxelDataMutationError::DirtyBlockMissingVoxels { location });
                }
                let revision = current_key_revision(&state.map, position);
                let VoxelDataKeyRevision::Present(block_revision) = revision else {
                    unreachable!("a resident block has a present key revision")
                };
                block_revision.checked_add(1).ok_or(
                    SharedVoxelDataMutationError::KeyRevisionOverflow {
                        position,
                        lod_index,
                    },
                )?;
                snapshots.push(SharedVoxelDataTransactionBlockSnapshot {
                    lineage: Arc::clone(&self.lineage),
                    location,
                    revision,
                    present: true,
                    viewers: block.viewers.get(),
                    modified: true,
                    edited: block.is_edited(),
                    has_voxels: true,
                });
                operations.push(SharedVoxelDataTransactionOperation::ClearModified { location });
                if self.data.transaction_reservation_should_fail(
                    SharedVoxelDataTransactionReservation::ResidentDirtyPayloadCopy,
                ) || self.data.resident_payload_copy_should_fail()
                {
                    return Err(
                        SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                            reservation:
                                SharedVoxelDataTransactionReservation::ResidentDirtyPayloadCopy,
                        },
                    );
                }
                let voxels = block.voxels().try_copy_to_owned().map_err(|_| {
                    SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                        reservation:
                            SharedVoxelDataTransactionReservation::ResidentDirtyPayloadCopy,
                    }
                })?;
                payloads.push(RevisionedBlockToSave {
                    location,
                    block_revision,
                    voxels,
                });
            }
        }

        let key = |location: BlockLocation| {
            (
                location.lod_index,
                location.position.x,
                location.position.y,
                location.position.z,
            )
        };
        operations.sort_unstable_by_key(|operation| key(operation.location()));
        snapshots.sort_unstable_by_key(|snapshot| key(snapshot.location));
        payloads.sort_unstable_by_key(|payload| key(payload.location));
        debug_assert_eq!(operations.len(), dirty_count);
        debug_assert_eq!(snapshots.len(), dirty_count);
        debug_assert_eq!(payloads.len(), dirty_count);
        Ok(PreparedResidentDirtyCopies {
            operations,
            snapshots,
            payloads,
        })
    }

    #[cfg(test)]
    fn set_test_after_prepare_hook(&mut self, hook: Arc<dyn Fn() + Send + Sync + 'static>) {
        self.after_prepare_hook = Some(hook);
    }

    /// Prepares one exact C1 transaction only if the preview's settings remain
    /// current. A conflict is detected before ordinary operation preparation,
    /// returning every exact operation owner unchanged.
    pub(crate) fn prepare_transaction(
        self,
        operations: Vec<SharedVoxelDataTransactionOperation>,
        snapshots: &[SharedVoxelDataTransactionBlockSnapshot],
    ) -> Result<PreparedSharedVoxelDataTransaction, PreparedSharedVoxelDataTransactionPrepareError>
    {
        let actual_settings_revision = self
            .data
            .settings
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .revision;
        let expected_settings_revision = self.settings.revision;
        if actual_settings_revision != expected_settings_revision {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                    expected_revision: expected_settings_revision,
                    actual_revision: actual_settings_revision,
                },
                operations,
            });
        }
        let lineage_matches = std::ptr::eq(self.lineage.data.as_ptr(), Arc::as_ptr(&self.data))
            && self.lineage.settings_revision == expected_settings_revision
            && snapshots
                .iter()
                .all(|snapshot| Arc::ptr_eq(&snapshot.lineage, &self.lineage));
        if !lineage_matches {
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::PreparedTransactionPreviewSetMismatch,
                operations,
            });
        }
        let prepared = self
            .data
            .prepare_transaction_from_snapshots_at_settings_revision(
                operations,
                snapshots,
                expected_settings_revision,
            );
        #[cfg(test)]
        if let Some(hook) = &self.after_prepare_hook {
            hook();
        }
        let actual_settings_revision = self
            .data
            .settings
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .revision;
        if actual_settings_revision != expected_settings_revision {
            let operations = match prepared {
                Ok(prepared) => prepared.into_uncommitted_operations(),
                Err(error) => error.into_parts().1,
            };
            return Err(PreparedSharedVoxelDataTransactionPrepareError {
                error: SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                    expected_revision: expected_settings_revision,
                    actual_revision: actual_settings_revision,
                },
                operations,
            });
        }
        prepared
    }
}

impl SharedVoxelDataTransactionBlockSnapshot {
    pub(crate) const fn location(&self) -> BlockLocation {
        self.location
    }

    pub(crate) const fn is_present(&self) -> bool {
        self.present
    }

    pub(crate) const fn viewers(&self) -> u32 {
        self.viewers
    }

    #[allow(dead_code)] // Consumed by the C2D edit and resident-save drafts.
    pub(crate) const fn revision(&self) -> VoxelDataKeyRevision {
        self.revision
    }

    pub(crate) const fn is_modified(&self) -> bool {
        self.modified
    }

    #[allow(dead_code)] // Consumed by the C2D edit and resident-save drafts.
    pub(crate) const fn is_edited(&self) -> bool {
        self.edited
    }

    pub(crate) const fn has_voxels(&self) -> bool {
        self.has_voxels
    }
}

#[allow(dead_code)] // Task 3 adopts every accessor for terrain publication.
impl PreparedSharedVoxelDataEdit {
    pub(crate) const fn edited_block(&self) -> BlockLocation {
        self.edited_block
    }

    pub(crate) const fn block_revision(&self) -> u64 {
        self.block_revision
    }

    pub(crate) fn inserted_locations(&self) -> &[BlockLocation] {
        &self.inserted_locations
    }

    /// Transfers the outer terrain owner's exact pre-existing residency into
    /// a newly materialized edit block before the storage transaction is
    /// committed.
    pub(crate) fn set_insert_final_viewers(
        &mut self,
        location: BlockLocation,
        final_viewers: u32,
    ) -> bool {
        self.transaction
            .operations
            .iter_mut()
            .find_map(|operation| match operation {
                SharedVoxelDataTransactionOperation::Insert {
                    location: operation_location,
                    final_viewers: operation_final_viewers,
                    ..
                } if *operation_location == location => Some(operation_final_viewers),
                _ => None,
            })
            .is_some_and(|operation_final_viewers| {
                *operation_final_viewers = final_viewers;
                true
            })
    }

    pub(crate) fn transaction_mut(&mut self) -> &mut PreparedSharedVoxelDataTransaction {
        &mut self.transaction
    }

    pub(crate) fn into_transaction(self) -> PreparedSharedVoxelDataTransaction {
        self.transaction
    }
}

#[allow(dead_code)]
impl PreparedSharedVoxelDataTransaction {
    /// Authorizes this prepared transaction to commit through a closed
    /// shutdown boundary for its exact storage lineage.
    pub(crate) fn authorize_shutdown_mutation(
        &mut self,
        permit: &SharedVoxelDataShutdownMutationPermit,
    ) -> Result<(), SharedVoxelDataMutationError> {
        if !permit.matches(&self.data) {
            return Err(SharedVoxelDataMutationError::ShutdownMutationPermitMismatch);
        }
        self.shutdown_mutation_lineage = Some(permit.data.clone());
        Ok(())
    }

    fn into_uncommitted_operations(self) -> Vec<SharedVoxelDataTransactionOperation> {
        debug_assert!(!self.committed);
        self.operations
    }

    #[cfg(test)]
    fn set_test_spatial_batch_drop_hook(&mut self, hook: Arc<dyn Fn() + Send + Sync + 'static>) {
        self.spatial_batch_drop_hook = Some(hook);
    }

    /// Returns the retained insert payload for test/outer-draft inspection.
    pub(crate) fn inserted_block(&self, location: BlockLocation) -> Option<&VoxelDataBlock> {
        self.operations
            .iter()
            .find_map(|operation| match operation {
                SharedVoxelDataTransactionOperation::Insert {
                    location: operation_location,
                    block,
                    ..
                } if *operation_location == location => Some(block),
                _ => None,
            })
    }

    /// Returns an exact block payload still owned by an uncommitted insert or
    /// replacement operation.
    pub(crate) fn owned_block(&self, location: BlockLocation) -> Option<&VoxelDataBlock> {
        self.operations
            .iter()
            .find_map(|operation| match operation {
                SharedVoxelDataTransactionOperation::Insert {
                    location: operation_location,
                    block,
                    ..
                }
                | SharedVoxelDataTransactionOperation::Replace {
                    location: operation_location,
                    block,
                } if *operation_location == location => Some(block),
                _ => None,
            })
    }

    /// Recovers every original operation after a failed commit so the caller
    /// can prepare against fresh expectations without rebuilding payloads.
    pub(crate) fn into_operations(
        self,
    ) -> Result<Vec<SharedVoxelDataTransactionOperation>, SharedVoxelDataMutationError> {
        if self.committed {
            return Err(SharedVoxelDataMutationError::PreparedTransactionAlreadyCommitted);
        }
        Ok(self.operations)
    }

    /// Commits the prepared key set exactly once.
    ///
    /// Every recoverable error occurs before the first live write. On those
    /// paths the exact spatial batches and owned insert payloads remain in
    /// `self`, allowing either a retry or [`Self::into_operations`].
    pub(crate) fn commit(
        &mut self,
    ) -> Result<PreparedSharedVoxelDataTransactionOutcome, SharedVoxelDataMutationError> {
        Ok(self.commit_holding_publication_fence()?.finish())
    }

    /// Installs every storage write while retaining the complete publication
    /// guard set for a matching outer-state commit.
    ///
    /// Every recoverable error still occurs before the first live write and
    /// restores the prepared transaction exactly for retry.
    pub(crate) fn commit_holding_publication_fence<'a>(
        &'a mut self,
    ) -> Result<CommittedSharedVoxelDataTransaction<'a>, SharedVoxelDataMutationError> {
        let Self {
            data,
            expected_settings_revision,
            operations,
            expectations,
            observations,
            spatial_batches,
            insertion_counts,
            key_revision_counts,
            removed_blocks,
            shutdown_mutation_lineage,
            committed,
            #[cfg(test)]
            spatial_batch_drop_hook,
        } = self;

        if *committed {
            return Err(SharedVoxelDataMutationError::PreparedTransactionAlreadyCommitted);
        }

        let data_arc: &Arc<SharedVoxelData> = &*data;
        let data: &'a SharedVoxelData = Arc::as_ref(data_arc);
        let gate = data
            .mutation_gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if data.mutation_admission_closed.load(AtomicOrdering::Acquire) && !operations.is_empty() {
            match shutdown_mutation_lineage {
                Some(lineage) if Weak::ptr_eq(lineage, &Arc::downgrade(data_arc)) => {}
                Some(_) => {
                    return Err(SharedVoxelDataMutationError::ShutdownMutationPermitMismatch);
                }
                None => return Err(SharedVoxelDataMutationError::MutationAdmissionClosed),
            }
        }
        let actual_settings_revision = {
            data.settings
                .read()
                .unwrap_or_else(|error| error.into_inner())
                .revision
        };
        if actual_settings_revision != *expected_settings_revision {
            return Err(SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                expected_revision: *expected_settings_revision,
                actual_revision: actual_settings_revision,
            });
        }

        let mut spatial_guards: [Option<SpatialLockWriteManyGuard<'_>>; MAX_LOD] =
            std::array::from_fn(|_| None);
        for (lod_index, guard) in spatial_guards.iter_mut().enumerate().take(data.lods.len()) {
            let Some(batch) = spatial_batches[lod_index].take() else {
                continue;
            };
            match data.spatial_lock(lod_index).write_prepared_fallible(batch) {
                Ok(acquired) => *guard = Some(acquired),
                Err(batch) => {
                    spatial_batches[lod_index] = Some(batch);
                    for (acquired_lod, acquired) in spatial_guards.iter_mut().enumerate() {
                        if let Some(acquired) = acquired.take() {
                            spatial_batches[acquired_lod] = Some(acquired.release_prepared());
                        }
                    }
                    drop(gate);
                    return Err(
                        SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                            reservation:
                                SharedVoxelDataTransactionReservation::LiveSpatialRegistry,
                        },
                    );
                }
            }
        }

        let mut map_guards: [Option<StdRwLockWriteGuard<'_, SharedVoxelDataLodState>>; MAX_LOD] =
            std::array::from_fn(|_| None);
        for (lod_index, guard) in map_guards.iter_mut().enumerate().take(data.lods.len()) {
            if spatial_guards[lod_index].is_none() {
                continue;
            }
            *guard = Some(
                data.lods[lod_index]
                    .state
                    .write()
                    .unwrap_or_else(|error| error.into_inner()),
            );
        }

        let validation = (|| {
            for expectation in observations.iter() {
                let lod_index = usize::from(expectation.location.lod_index);
                let state = map_guards[lod_index]
                    .as_ref()
                    .expect("prepared observation owns the affected LOD map guard");
                let actual_revision =
                    current_key_revision(&state.map, expectation.location.position);
                if actual_revision != expectation.expected_revision {
                    return Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                        position: expectation.location.position,
                        lod_index,
                        expected_revision: expectation.expected_revision,
                        actual_revision,
                    });
                }
                match expectation.expected_block_state {
                    Some(expected) => {
                        let Some(block) = state.map.get_block(expectation.location.position) else {
                            return Err(
                                SharedVoxelDataMutationError::PreparedTransactionExpectedPresent {
                                    location: expectation.location,
                                    actual_revision,
                                },
                            );
                        };
                        let actual = PreparedSharedVoxelDataBlockState {
                            viewers: block.viewers.get(),
                            modified: block.is_modified(),
                            edited: block.is_edited(),
                            has_voxels: block.has_voxels(),
                        };
                        if actual != expected {
                            return Err(SharedVoxelDataMutationError::PreparedTransactionConcurrentBlockState {
                                location: expectation.location,
                                expected_viewers: expected.viewers,
                                actual_viewers: actual.viewers,
                                expected_modified: expected.modified,
                                actual_modified: actual.modified,
                                expected_edited: expected.edited,
                                actual_edited: actual.edited,
                                expected_has_voxels: expected.has_voxels,
                                actual_has_voxels: actual.has_voxels,
                            });
                        }
                    }
                    None => {
                        if state.map.has_block(expectation.location.position) {
                            return Err(SharedVoxelDataMutationError::PreparedTransactionExpectedTombstone {
                                location: expectation.location,
                                actual_revision,
                            });
                        }
                    }
                }
            }
            for (operation, expectation) in operations.iter().zip(expectations.iter()) {
                debug_assert_eq!(operation.location(), expectation.location);
                let lod_index = usize::from(expectation.location.lod_index);
                let state = map_guards[lod_index]
                    .as_ref()
                    .expect("prepared transaction owns the affected LOD map guard");
                let actual_revision =
                    current_key_revision(&state.map, expectation.location.position);
                if actual_revision != expectation.expected_revision {
                    return Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                        position: expectation.location.position,
                        lod_index,
                        expected_revision: expectation.expected_revision,
                        actual_revision,
                    });
                }
                let Some(actual_next_revision) = key_revision_value(actual_revision).checked_add(1)
                else {
                    return Err(SharedVoxelDataMutationError::KeyRevisionOverflow {
                        position: expectation.location.position,
                        lod_index,
                    });
                };
                if actual_next_revision != expectation.next_revision {
                    return Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                        position: expectation.location.position,
                        lod_index,
                        expected_revision: expectation.expected_revision,
                        actual_revision,
                    });
                }

                match expectation.expected_block_state {
                    Some(expected) => {
                        let Some(block) = state.map.get_block(expectation.location.position) else {
                            return Err(
                                SharedVoxelDataMutationError::PreparedTransactionExpectedPresent {
                                    location: expectation.location,
                                    actual_revision,
                                },
                            );
                        };
                        let actual = PreparedSharedVoxelDataBlockState {
                            viewers: block.viewers.get(),
                            modified: block.is_modified(),
                            edited: block.is_edited(),
                            has_voxels: block.has_voxels(),
                        };
                        if actual != expected {
                            return Err(SharedVoxelDataMutationError::PreparedTransactionConcurrentBlockState {
                                location: expectation.location,
                                expected_viewers: expected.viewers,
                                actual_viewers: actual.viewers,
                                expected_modified: expected.modified,
                                actual_modified: actual.modified,
                                expected_edited: expected.edited,
                                actual_edited: actual.edited,
                                expected_has_voxels: expected.has_voxels,
                                actual_has_voxels: actual.has_voxels,
                            });
                        }
                    }
                    None => {
                        if state.map.has_block(expectation.location.position) {
                            return Err(SharedVoxelDataMutationError::PreparedTransactionExpectedTombstone {
                                location: expectation.location,
                                actual_revision,
                            });
                        }
                    }
                }
            }

            let removal_count = operations
                .iter()
                .filter(|operation| {
                    matches!(
                        operation,
                        SharedVoxelDataTransactionOperation::Replace { .. }
                            | SharedVoxelDataTransactionOperation::Remove { .. }
                    )
                })
                .count();
            if removed_blocks.capacity() - removed_blocks.len() < removal_count {
                return Err(
                    SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                        reservation: SharedVoxelDataTransactionReservation::RemovedOutcome,
                    },
                );
            }

            for (lod_index, state) in map_guards.iter_mut().enumerate().take(data.lods.len()) {
                let Some(state) = state.as_mut() else {
                    continue;
                };
                if insertion_counts[lod_index] == 0 {
                    continue;
                }
                if data.transaction_reservation_should_fail(
                    SharedVoxelDataTransactionReservation::LiveMap,
                ) {
                    return Err(
                        SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                            reservation: SharedVoxelDataTransactionReservation::LiveMap,
                        },
                    );
                }
                state
                    .map
                    .try_reserve(insertion_counts[lod_index])
                    .map_err(|_| {
                        SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                            reservation: SharedVoxelDataTransactionReservation::LiveMap,
                        }
                    })?;
            }
            for (lod_index, state) in map_guards.iter_mut().enumerate().take(data.lods.len()) {
                let Some(state) = state.as_mut() else {
                    continue;
                };
                if key_revision_counts[lod_index] == 0 {
                    continue;
                }
                if data.transaction_reservation_should_fail(
                    SharedVoxelDataTransactionReservation::LiveKeyRevisions,
                ) {
                    return Err(
                        SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                            reservation: SharedVoxelDataTransactionReservation::LiveKeyRevisions,
                        },
                    );
                }
                state
                    .map
                    .try_reserve_key_revisions(key_revision_counts[lod_index])
                    .map_err(|_| {
                        SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                            reservation: SharedVoxelDataTransactionReservation::LiveKeyRevisions,
                        }
                    })?;
            }
            Ok(())
        })();

        if let Err(error) = validation {
            drop(map_guards);
            for (lod_index, guard) in spatial_guards.iter_mut().enumerate() {
                if let Some(guard) = guard.take() {
                    spatial_batches[lod_index] = Some(guard.release_prepared());
                }
            }
            drop(gate);
            return Err(error);
        }

        #[cfg(test)]
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            data.notify_test_edit_phase(
                SharedVoxelDataEditPhase::PreparedTransactionValidatedBeforeFirstLiveWrite,
            );
        })) {
            drop(map_guards);
            for (lod_index, guard) in spatial_guards.iter_mut().enumerate() {
                if let Some(guard) = guard.take() {
                    spatial_batches[lod_index] = Some(guard.release_prepared());
                }
            }
            drop(gate);
            std::panic::resume_unwind(payload);
        }

        for (operation, expectation) in operations.iter_mut().zip(expectations.iter()) {
            let lod_index = usize::from(expectation.location.lod_index);
            let state = map_guards[lod_index]
                .as_mut()
                .expect("prepared transaction owns the affected LOD map guard");
            match operation {
                SharedVoxelDataTransactionOperation::SetViewersExact { final_viewers, .. } => {
                    state
                        .map
                        .get_block_mut(expectation.location.position)
                        .expect("validated exact-viewer block remains present")
                        .viewers
                        .set_exact(*final_viewers);
                }
                SharedVoxelDataTransactionOperation::SetViewersExactAndClearModified {
                    final_viewers,
                    ..
                } => {
                    let block = state
                        .map
                        .get_block_mut(expectation.location.position)
                        .expect("validated dirty exact-viewer block remains present");
                    block.viewers.set_exact(*final_viewers);
                    block.set_modified(false);
                }
                SharedVoxelDataTransactionOperation::Insert {
                    block,
                    final_viewers,
                    ..
                } => {
                    let mut adopted = std::mem::replace(
                        block,
                        VoxelDataBlock::empty(expectation.location.lod_index),
                    );
                    adopted.viewers.set_exact(*final_viewers);
                    state
                        .map
                        .set_block(expectation.location.position, adopted, false);
                }
                SharedVoxelDataTransactionOperation::Replace { block, .. } => {
                    let replacement = std::mem::replace(
                        block,
                        VoxelDataBlock::empty(expectation.location.lod_index),
                    );
                    let resident = state
                        .map
                        .get_block_mut(expectation.location.position)
                        .expect("validated replacement block remains present");
                    let retired = std::mem::replace(resident, replacement);
                    removed_blocks.push(RemovedSharedVoxelDataBlock {
                        location: expectation.location,
                        block: retired,
                        block_revision: key_revision_value(expectation.expected_revision),
                    });
                }
                SharedVoxelDataTransactionOperation::ClearModified { .. } => {
                    state
                        .map
                        .get_block_mut(expectation.location.position)
                        .expect("validated dirty block remains present")
                        .set_modified(false);
                }
                SharedVoxelDataTransactionOperation::Remove { .. } => {
                    let block = state
                        .map
                        .remove_block(expectation.location.position)
                        .expect("validated removed block remains present");
                    removed_blocks.push(RemovedSharedVoxelDataBlock {
                        location: expectation.location,
                        block,
                        block_revision: key_revision_value(expectation.expected_revision),
                    });
                }
            }
            state
                .map
                .commit_key_revision(expectation.location.position, expectation.next_revision);
        }

        *committed = true;
        Ok(CommittedSharedVoxelDataTransaction {
            map_guards: Some(map_guards),
            spatial_guards: Some(spatial_guards),
            mutation_gate: Some(gate),
            removed_blocks: std::mem::take(removed_blocks),
            #[cfg(test)]
            spatial_batch_drop_hook: spatial_batch_drop_hook.take(),
        })
    }
}

#[allow(dead_code)] // Additive C2c-C1F foundation; terrain adopts it in C2c-C2.
impl CommittedSharedVoxelDataTransaction<'_> {
    /// Returns the exact removed owners while the publication fence is held.
    pub(crate) fn removed_blocks(&self) -> &[RemovedSharedVoxelDataBlock] {
        &self.removed_blocks
    }

    /// Moves the already-allocated removed-owner vector out without reserving
    /// or cloning any buffer.
    pub(crate) fn take_removed_blocks(&mut self) -> Vec<RemovedSharedVoxelDataBlock> {
        std::mem::take(&mut self.removed_blocks)
    }

    /// Releases the publication guards in their required order and returns
    /// any removed owners not already taken by the caller.
    pub(crate) fn finish(mut self) -> PreparedSharedVoxelDataTransactionOutcome {
        self.release_publication_fence();
        PreparedSharedVoxelDataTransactionOutcome {
            removed_blocks: std::mem::take(&mut self.removed_blocks),
        }
    }

    /// Idempotently releases map, spatial, and mutation guards before any
    /// prepared bounds, removed owner, or test retirement probe can retire.
    fn release_publication_fence(&mut self) {
        drop(self.map_guards.take());

        let mut retired_spatial_batches: [Option<PreparedSpatialWriteBatch>; MAX_LOD] =
            std::array::from_fn(|_| None);
        if let Some(mut spatial_guards) = self.spatial_guards.take() {
            for (lod_index, guard) in spatial_guards.iter_mut().enumerate() {
                if let Some(guard) = guard.take() {
                    retired_spatial_batches[lod_index] = Some(guard.release_prepared());
                }
            }
            drop(spatial_guards);
        }

        drop(self.mutation_gate.take());

        #[cfg(test)]
        let had_retired_spatial_batches = retired_spatial_batches.iter().any(Option::is_some);
        drop(retired_spatial_batches);
        #[cfg(test)]
        if had_retired_spatial_batches {
            if let Some(hook) = &self.spatial_batch_drop_hook {
                hook();
            }
        }
    }
}

impl Drop for CommittedSharedVoxelDataTransaction<'_> {
    fn drop(&mut self) {
        self.release_publication_fence();
    }
}

impl VoxelDataMutation<'_> {
    fn set_generator(
        &self,
        generator: Option<SharedVoxelGenerator>,
    ) -> Result<(), SharedVoxelDataMutationError> {
        let mut settings = self
            .data
            .settings
            .write()
            .unwrap_or_else(|e| e.into_inner());
        let unchanged = match (&settings.generator, &generator) {
            (None, None) => true,
            (Some(current), Some(next)) => Arc::ptr_eq(current, next),
            _ => false,
        };
        if unchanged {
            return Ok(());
        }
        let next_revision = settings
            .revision
            .checked_add(1)
            .ok_or(SharedVoxelDataMutationError::SettingsRevisionOverflow)?;
        settings.generator = generator;
        settings.revision = next_revision;
        Ok(())
    }

    #[cfg(test)]
    fn set_settings_revision_for_test(&self, revision: u64) {
        self.data
            .settings
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .revision = revision;
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_edit_voxel(
        &self,
        expected_settings_revision: u64,
        expected_key_revision: VoxelDataKeyRevision,
        mut prepared_voxels: Option<VoxelBuffer>,
        value: u64,
        position: Vector3i,
        block_position: Vector3i,
        channel_index: usize,
    ) -> Result<(), SharedVoxelDataMutationError> {
        #[cfg(test)]
        self.data.notify_test_edit_phase(
            SharedVoxelDataEditPhase::MutationGateAcquiredBeforeSpatialWrite,
        );
        let actual_settings_revision = self
            .data
            .settings
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .revision;
        if actual_settings_revision != expected_settings_revision {
            return Err(SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                expected_revision: expected_settings_revision,
                actual_revision: actual_settings_revision,
            });
        }

        let block_box = Box3i::new(block_position, Vector3i::splat(1));
        let bounds = checked_lod_block_bounds(block_box, self.data.block_size(), 0)
            .ok_or(SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index: 0 })?;
        let _spatial = self.data.spatial_lock(0).write_many([bounds]);
        #[cfg(test)]
        self.data
            .notify_test_edit_phase(SharedVoxelDataEditPhase::SpatialWriteAcquiredBeforeMapLock);
        let lod = self
            .data
            .lods
            .first()
            .expect("SharedVoxelData always owns LOD 0");
        let mut state = lod.state.write().unwrap_or_else(|e| e.into_inner());
        let actual_key_revision = current_key_revision(&state.map, block_position);
        if actual_key_revision != expected_key_revision {
            return Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: block_position,
                lod_index: 0,
                expected_revision: expected_key_revision,
                actual_revision: actual_key_revision,
            });
        }
        let next_revision = key_revision_value(actual_key_revision)
            .checked_add(1)
            .ok_or(SharedVoxelDataMutationError::KeyRevisionOverflow {
                position: block_position,
                lod_index: 0,
            })?;
        let needs_materialization = state
            .map
            .get_block(block_position)
            .is_none_or(|block| !block.has_voxels());
        // A concurrent writer may have changed whether this key needs
        // materialization after generator preparation. Reject that stale
        // draft instead of installing it over live data.
        if needs_materialization && prepared_voxels.is_none() {
            return Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: block_position,
                lod_index: 0,
                expected_revision: expected_key_revision,
                actual_revision: actual_key_revision,
            });
        }
        if !state.map.has_block(block_position) {
            state
                .map
                .try_reserve(1)
                .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        }
        state
            .map
            .try_reserve_key_revisions(1)
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;

        if needs_materialization {
            state.map.set_block_buffer(
                block_position,
                prepared_voxels
                    .take()
                    .expect("missing resident data was prepared before commit"),
                true,
            );
        }
        state.map.set_voxel(value, position, channel_index);
        #[cfg(test)]
        self.data
            .notify_test_edit_phase(SharedVoxelDataEditPhase::VoxelWrittenBeforeDirtyFlags);
        let block = state
            .map
            .get_block_mut(block_position)
            .expect("edited block exists after materialization");
        block.set_modified(true);
        block.set_edited(true);
        #[cfg(test)]
        self.data
            .notify_test_edit_phase(SharedVoxelDataEditPhase::DirtyFlagsSetBeforeMapWriteUnlock);
        state.map.commit_key_revision(block_position, next_revision);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_set_voxel(
        &self,
        expected_settings_revision: u64,
        expected_key_revision: VoxelDataKeyRevision,
        mut prepared_voxels: Option<VoxelBuffer>,
        value: u64,
        position: Vector3i,
        block_position: Vector3i,
        channel_index: usize,
        spatial_batch: PreparedSpatialWriteBatch,
    ) -> Result<(), SharedVoxelDataMutationError> {
        #[cfg(test)]
        self.data.notify_test_edit_phase(
            SharedVoxelDataEditPhase::MutationGateAcquiredBeforeSpatialWrite,
        );
        let actual_settings_revision = self
            .data
            .settings
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .revision;
        if actual_settings_revision != expected_settings_revision {
            return Err(SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                expected_revision: expected_settings_revision,
                actual_revision: actual_settings_revision,
            });
        }

        let _spatial = self.data.spatial_lock(0).write_prepared(spatial_batch);
        #[cfg(test)]
        self.data
            .notify_test_edit_phase(SharedVoxelDataEditPhase::SpatialWriteAcquiredBeforeMapLock);
        let lod = self
            .data
            .lods
            .first()
            .expect("SharedVoxelData always owns LOD 0");
        let mut state = lod.state.write().unwrap_or_else(|e| e.into_inner());
        let actual_key_revision = current_key_revision(&state.map, block_position);
        if actual_key_revision != expected_key_revision {
            return Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: block_position,
                lod_index: 0,
                expected_revision: expected_key_revision,
                actual_revision: actual_key_revision,
            });
        }
        let next_revision = key_revision_value(actual_key_revision)
            .checked_add(1)
            .ok_or(SharedVoxelDataMutationError::KeyRevisionOverflow {
                position: block_position,
                lod_index: 0,
            })?;
        let needs_materialization = state
            .map
            .get_block(block_position)
            .is_none_or(|block| !block.has_voxels());
        if needs_materialization && prepared_voxels.is_none() {
            return Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: block_position,
                lod_index: 0,
                expected_revision: expected_key_revision,
                actual_revision: actual_key_revision,
            });
        }
        if !state.map.has_block(block_position) {
            state
                .map
                .try_reserve(1)
                .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        }
        state
            .map
            .try_reserve_key_revisions(1)
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        if needs_materialization {
            state.map.set_block_buffer(
                block_position,
                prepared_voxels
                    .take()
                    .expect("direct-set materialization was prepared before commit"),
                true,
            );
        }
        state.map.set_voxel(value, position, channel_index);
        state.map.commit_key_revision(block_position, next_revision);
        Ok(())
    }

    fn commit_lod_updates(
        self,
        expected_settings_revision: u64,
        mut changes: Vec<PreparedLodBlockMutation>,
        mut spatial_batches: [Option<PreparedSpatialWriteBatch>; MAX_LOD],
        mut retired: Vec<VoxelDataBlock>,
    ) -> Result<(), SharedVoxelDataMutationError> {
        if changes.is_empty() {
            return Ok(());
        }
        let data = self.data;
        #[cfg(test)]
        data.notify_test_edit_phase(
            SharedVoxelDataEditPhase::MutationGateAcquiredBeforeSpatialWrite,
        );
        let actual_settings_revision = data
            .settings
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .revision;
        if actual_settings_revision != expected_settings_revision {
            return Err(SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                expected_revision: expected_settings_revision,
                actual_revision: actual_settings_revision,
            });
        }

        let mut spatial_guards: [Option<SpatialLockWriteManyGuard<'_>>; MAX_LOD] =
            std::array::from_fn(|_| None);
        for lod_index in 0..data.lods.len() {
            let Some(batch) = spatial_batches[lod_index].take() else {
                continue;
            };
            spatial_guards[lod_index] = Some(data.spatial_lock(lod_index).write_prepared(batch));
            #[cfg(test)]
            data.notify_test_edit_phase(SharedVoxelDataEditPhase::SpatialWriteBatchAcquired {
                lod_index,
            });
        }
        #[cfg(test)]
        data.notify_test_edit_phase(SharedVoxelDataEditPhase::SpatialWriteAcquiredBeforeMapLock);
        let mut map_guards: [Option<StdRwLockWriteGuard<'_, SharedVoxelDataLodState>>; MAX_LOD] =
            std::array::from_fn(|_| None);
        for lod_index in 0..data.lods.len() {
            if spatial_guards[lod_index].is_some() {
                map_guards[lod_index] = Some(
                    data.lods[lod_index]
                        .state
                        .write()
                        .unwrap_or_else(|e| e.into_inner()),
                );
            }
        }

        for (lod_index, state) in map_guards.iter_mut().enumerate().take(data.lods.len()) {
            let Some(state) = state.as_mut() else {
                continue;
            };
            let mut insertion_count = 0usize;
            for change in changes
                .iter()
                .filter(|change| change.lod_index == lod_index)
            {
                let actual_revision = current_key_revision(&state.map, change.position);
                if actual_revision != change.expected_revision {
                    return Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                        position: change.position,
                        lod_index,
                        expected_revision: change.expected_revision,
                        actual_revision,
                    });
                }
                if !state.map.has_block(change.position) {
                    insertion_count = insertion_count
                        .checked_add(1)
                        .ok_or(SharedVoxelDataMutationError::CapacityReservationFailed)?;
                }
            }
            state
                .map
                .try_reserve(insertion_count)
                .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
            let change_count = changes
                .iter()
                .filter(|change| change.lod_index == lod_index)
                .count();
            state
                .map
                .try_reserve_key_revisions(change_count)
                .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        }

        for (lod_index, state) in map_guards.iter_mut().enumerate().take(data.lods.len()) {
            let Some(state) = state.as_mut() else {
                continue;
            };
            for change in changes
                .iter_mut()
                .filter(|change| change.lod_index == lod_index)
            {
                let replacement =
                    std::mem::replace(&mut change.block, VoxelDataBlock::empty(lod_index as u8));
                if let Some(resident) = state.map.get_block_mut(change.position) {
                    retired.push(std::mem::replace(resident, replacement));
                } else {
                    state.map.set_block(change.position, replacement, false);
                }
                state
                    .map
                    .commit_key_revision(change.position, change.next_revision);
            }
        }

        drop(map_guards);
        drop(spatial_guards);
        drop(self);
        drop(retired);
        Ok(())
    }

    fn commit_mark_area_modified(
        &self,
        changes: &[PreparedMarkMutation],
        spatial_batch: PreparedSpatialWriteBatch,
    ) -> Result<(), SharedVoxelDataMutationError> {
        if changes.is_empty() {
            return Ok(());
        }
        #[cfg(test)]
        self.data.notify_test_edit_phase(
            SharedVoxelDataEditPhase::MutationGateAcquiredBeforeSpatialWrite,
        );
        let _spatial = self.data.spatial_lock(0).write_prepared(spatial_batch);
        #[cfg(test)]
        self.data
            .notify_test_edit_phase(SharedVoxelDataEditPhase::SpatialWriteAcquiredBeforeMapLock);
        let lod = self
            .data
            .lods
            .first()
            .expect("SharedVoxelData always owns LOD 0");
        let mut state = lod.state.write().unwrap_or_else(|e| e.into_inner());

        for change in changes {
            let actual_revision = current_key_revision(&state.map, change.position);
            if actual_revision != change.expected_revision {
                return Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                    position: change.position,
                    lod_index: 0,
                    expected_revision: change.expected_revision,
                    actual_revision,
                });
            }
        }
        state
            .map
            .try_reserve_key_revisions(changes.len())
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;

        for change in changes {
            let block = state
                .map
                .get_block_mut(change.position)
                .expect("prepared marked block remains resident");
            block.set_modified(true);
            block.set_edited(true);
            if change.set_needs_lodding {
                block.set_needs_lodding(true);
            }
            state
                .map
                .commit_key_revision(change.position, change.next_revision);
        }
        Ok(())
    }

    fn commit_dirty_consumption(
        &self,
        changes: &[PreparedDirtyBlockMutation],
        mut spatial_batches: [Option<PreparedSpatialWriteBatch>; MAX_LOD],
    ) -> Result<(), SharedVoxelDataMutationError> {
        if changes.is_empty() {
            return Ok(());
        }
        #[cfg(test)]
        self.data.notify_test_edit_phase(
            SharedVoxelDataEditPhase::MutationGateAcquiredBeforeSpatialWrite,
        );

        let mut spatial_guards: [Option<SpatialLockWriteManyGuard<'_>>; MAX_LOD] =
            std::array::from_fn(|_| None);
        for lod_index in 0..self.data.lods.len() {
            let Some(batch) = spatial_batches[lod_index].take() else {
                continue;
            };
            spatial_guards[lod_index] =
                Some(self.data.spatial_lock(lod_index).write_prepared(batch));
        }

        let mut map_guards: [Option<StdRwLockWriteGuard<'_, SharedVoxelDataLodState>>; MAX_LOD] =
            std::array::from_fn(|_| None);
        for lod_index in 0..self.data.lods.len() {
            if spatial_guards[lod_index].is_some() {
                map_guards[lod_index] = Some(
                    self.data.lods[lod_index]
                        .state
                        .write()
                        .unwrap_or_else(|e| e.into_inner()),
                );
            }
        }

        for (lod_index, state) in map_guards.iter_mut().enumerate().take(self.data.lods.len()) {
            let Some(state) = state.as_mut() else {
                continue;
            };
            for change in changes
                .iter()
                .filter(|change| change.lod_index == lod_index)
            {
                let actual_revision = current_key_revision(&state.map, change.position);
                if actual_revision != change.expected_revision {
                    return Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                        position: change.position,
                        lod_index,
                        expected_revision: change.expected_revision,
                        actual_revision,
                    });
                }
                if !state
                    .map
                    .get_block(change.position)
                    .is_some_and(VoxelDataBlock::is_modified)
                {
                    return Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                        position: change.position,
                        lod_index,
                        expected_revision: change.expected_revision,
                        actual_revision,
                    });
                }
            }
            let change_count = changes
                .iter()
                .filter(|change| change.lod_index == lod_index)
                .count();
            state
                .map
                .try_reserve_key_revisions(change_count)
                .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        }

        for (lod_index, state) in map_guards.iter_mut().enumerate().take(self.data.lods.len()) {
            let Some(state) = state.as_mut() else {
                continue;
            };
            for change in changes
                .iter()
                .filter(|change| change.lod_index == lod_index)
            {
                state
                    .map
                    .get_block_mut(change.position)
                    .expect("prepared dirty block remains resident")
                    .set_modified(false);
                state
                    .map
                    .commit_key_revision(change.position, change.next_revision);
            }
        }
        drop(map_guards);
        drop(spatial_guards);
        Ok(())
    }

    fn view_area(
        &self,
        blocks_box: Box3i,
        lod_index: usize,
        mut missing_blocks: Option<&mut Vec<Vector3i>>,
        mut found_blocks_positions: Option<&mut Vec<Vector3i>>,
        mut found_blocks: Option<&mut Vec<VoxelDataBlock>>,
    ) -> Result<(), SharedVoxelDataMutationError> {
        #[cfg(test)]
        self.data.notify_test_edit_phase(
            SharedVoxelDataEditPhase::MutationGateAcquiredBeforeSpatialWrite,
        );
        let bounds = checked_lod_block_bounds(blocks_box, self.data.block_size(), lod_index)
            .ok_or(SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index })?;
        let _spatial = self.data.spatial_lock(lod_index).write_many([bounds]);
        let lod = self
            .data
            .lods
            .get(lod_index)
            .expect("LOD index is outside the loaded range");
        let mut state = lod.state.write().unwrap_or_else(|e| e.into_inner());
        let cell_count = checked_box_cell_count(blocks_box)
            .ok_or(SharedVoxelDataMutationError::CapacityReservationFailed)?;
        let mut changes = Vec::new();
        changes
            .try_reserve_exact(cell_count)
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        let mut missing_local = Vec::new();
        if missing_blocks.is_some() {
            missing_local
                .try_reserve_exact(cell_count)
                .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        }
        let mut found_positions_local = Vec::new();
        if found_blocks_positions.is_some() {
            found_positions_local
                .try_reserve_exact(cell_count)
                .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        }
        let mut found_blocks_local = Vec::new();
        if found_blocks.is_some() {
            found_blocks_local
                .try_reserve_exact(cell_count)
                .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        }

        for position in blocks_box.iter_cells_zxy() {
            let Some(block) = state.map.get_block(position) else {
                if missing_blocks.is_some() {
                    missing_local.push(position);
                }
                continue;
            };
            let next_viewers = block.viewers.get().checked_add(1).ok_or(
                SharedVoxelDataMutationError::ViewerCountOverflow {
                    position,
                    lod_index,
                },
            )?;
            let next_revision = state.map.key_revision(position).checked_add(1).ok_or(
                SharedVoxelDataMutationError::KeyRevisionOverflow {
                    position,
                    lod_index,
                },
            )?;
            if found_blocks_positions.is_some() {
                found_positions_local.push(position);
            }
            if found_blocks.is_some() {
                let mut snapshot = clone_block(block);
                snapshot.viewers.set_exact(next_viewers);
                found_blocks_local.push(snapshot);
            }
            changes.push((position, next_viewers, next_revision));
        }

        state
            .map
            .try_reserve_key_revisions(changes.len())
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        reserve_output(&mut missing_blocks, missing_local.len())?;
        reserve_output(&mut found_blocks_positions, found_positions_local.len())?;
        reserve_output(&mut found_blocks, found_blocks_local.len())?;

        for (position, next_viewers, next_revision) in changes {
            state
                .map
                .get_block_mut(position)
                .expect("prepared viewed block remains present")
                .viewers
                .set_exact(next_viewers);
            state.map.commit_key_revision(position, next_revision);
        }
        if let Some(out) = missing_blocks {
            out.extend(missing_local);
        }
        if let Some(out) = found_blocks_positions {
            out.extend(found_positions_local);
        }
        if let Some(out) = found_blocks {
            out.extend(found_blocks_local);
        }
        Ok(())
    }

    fn unview_area(
        &self,
        blocks_box: Box3i,
        lod_index: usize,
        mut removed_blocks: Option<&mut Vec<Vector3i>>,
        mut missing_blocks: Option<&mut Vec<Vector3i>>,
        mut to_save: Option<&mut Vec<BlockToSave>>,
    ) -> Result<(), SharedVoxelDataMutationError> {
        #[cfg(test)]
        self.data.notify_test_edit_phase(
            SharedVoxelDataEditPhase::MutationGateAcquiredBeforeSpatialWrite,
        );
        let bounds = checked_lod_block_bounds(blocks_box, self.data.block_size(), lod_index)
            .ok_or(SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index })?;
        let _spatial = self.data.spatial_lock(lod_index).write_many([bounds]);
        let lod = self
            .data
            .lods
            .get(lod_index)
            .expect("LOD index is outside the loaded range");
        let mut state = lod.state.write().unwrap_or_else(|e| e.into_inner());
        let cell_count = checked_box_cell_count(blocks_box)
            .ok_or(SharedVoxelDataMutationError::CapacityReservationFailed)?;
        let mut changes = Vec::new();
        changes
            .try_reserve_exact(cell_count)
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        let mut removed_local = Vec::new();
        if removed_blocks.is_some() {
            removed_local
                .try_reserve_exact(cell_count)
                .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        }
        let mut missing_local = Vec::new();
        if missing_blocks.is_some() {
            missing_local
                .try_reserve_exact(cell_count)
                .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        }
        let mut save_count = 0usize;

        for position in blocks_box.iter_cells_zxy() {
            let Some(block) = state.map.get_block(position) else {
                if missing_blocks.is_some() {
                    missing_local.push(position);
                }
                continue;
            };
            let next_viewers = block.viewers.get().checked_sub(1).ok_or(
                SharedVoxelDataMutationError::ViewerCountUnderflow {
                    position,
                    lod_index,
                },
            )?;
            let should_save = next_viewers == 0 && block.is_modified() && to_save.is_some();
            let should_remove = next_viewers == 0 && (!block.is_modified() || should_save);
            let next_revision = state.map.key_revision(position).checked_add(1).ok_or(
                SharedVoxelDataMutationError::KeyRevisionOverflow {
                    position,
                    lod_index,
                },
            )?;
            if should_remove && removed_blocks.is_some() {
                removed_local.push(position);
            }
            if should_save {
                save_count = save_count
                    .checked_add(1)
                    .ok_or(SharedVoxelDataMutationError::CapacityReservationFailed)?;
            }
            changes.push((
                position,
                next_viewers,
                next_revision,
                should_remove,
                should_save,
            ));
        }

        state
            .map
            .try_reserve_key_revisions(changes.len())
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        reserve_output(&mut removed_blocks, removed_local.len())?;
        reserve_output(&mut missing_blocks, missing_local.len())?;
        reserve_output(&mut to_save, save_count)?;

        for (position, next_viewers, next_revision, should_remove, should_save) in changes {
            if should_remove {
                let block = state
                    .map
                    .remove_block(position)
                    .expect("prepared unviewed block remains present");
                if should_save {
                    to_save
                        .as_deref_mut()
                        .expect("dirty removal prepared a save destination")
                        .push(BlockToSave {
                            voxels: block.into_voxels(),
                            position,
                            lod_index: lod_index as u8,
                            block_revision: next_revision - 1,
                        });
                }
            } else {
                state
                    .map
                    .get_block_mut(position)
                    .expect("prepared retained block remains present")
                    .viewers
                    .set_exact(next_viewers);
            }
            state.map.commit_key_revision(position, next_revision);
        }
        if let Some(out) = removed_blocks {
            out.extend(removed_local);
        }
        if let Some(out) = missing_blocks {
            out.extend(missing_local);
        }
        Ok(())
    }

    fn try_insert_block(
        &self,
        block_pos: Vector3i,
        block: VoxelDataBlock,
        lod_index: usize,
    ) -> Result<bool, SharedVoxelDataMutationError> {
        #[cfg(test)]
        self.data.notify_test_edit_phase(
            SharedVoxelDataEditPhase::MutationGateAcquiredBeforeSpatialWrite,
        );
        let block_box = Box3i::new(block_pos, Vector3i::splat(1));
        let bounds = checked_lod_block_bounds(block_box, self.data.block_size(), lod_index)
            .ok_or(SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index })?;
        let _spatial = self.data.spatial_lock(lod_index).write_many([bounds]);
        let lod = self
            .data
            .lods
            .get(lod_index)
            .expect("LOD index is outside the loaded range");
        let mut state = lod.state.write().unwrap_or_else(|e| e.into_inner());
        if state.map.has_block(block_pos) {
            return Ok(false);
        }
        let next_revision = state.map.key_revision(block_pos).checked_add(1).ok_or(
            SharedVoxelDataMutationError::KeyRevisionOverflow {
                position: block_pos,
                lod_index,
            },
        )?;
        state
            .map
            .try_reserve(1)
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        state
            .map
            .try_reserve_key_revisions(1)
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
        state.map.set_block(block_pos, block, false);
        state.map.commit_key_revision(block_pos, next_revision);
        Ok(true)
    }
}

impl fmt::Debug for SharedVoxelData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let settings = self.settings.read().unwrap_or_else(|e| e.into_inner());
        let lod_structural_revisions = self
            .lods
            .iter()
            .map(|lod| {
                lod.state
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .structural_revision
            })
            .collect::<Vec<_>>();
        f.debug_struct("SharedVoxelData")
            .field("lod_count", &self.lods.len())
            .field("lod_structural_revisions", &lod_structural_revisions)
            .field("format", &settings.format)
            .field("bounds_in_voxels", &settings.bounds_in_voxels)
            .field("streaming_enabled", &settings.streaming_enabled)
            .field("full_load_completed", &settings.full_load_completed)
            .field("has_generator", &settings.generator.is_some())
            .field("has_stream", &settings.stream.is_some())
            .field("settings_revision", &settings.revision)
            .field("spatial_lock_count", &self.spatial_locks.len())
            .finish()
    }
}

#[derive(Debug)]
pub(crate) struct SharedVoxelDataReadRegion<'a> {
    lock: &'a SpatialLock3D,
    bounds: BoxBounds3i,
}

impl Drop for SharedVoxelDataReadRegion<'_> {
    fn drop(&mut self) {
        self.lock.unlock_read(self.bounds);
    }
}

#[derive(Debug)]
#[cfg(test)]
pub(crate) struct SharedVoxelDataWriteRegion<'a> {
    lock: &'a SpatialLock3D,
    bounds: BoxBounds3i,
}

#[cfg(test)]
impl Drop for SharedVoxelDataWriteRegion<'_> {
    fn drop(&mut self) {
        self.lock.unlock_write(self.bounds);
    }
}

fn bounds_from_box(voxel_box: Box3i) -> BoxBounds3i {
    BoxBounds3i::from_box(voxel_box.position, voxel_box.size)
}

fn create_block_buffer(block_size: i32, format: VoxelFormat) -> VoxelBuffer {
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(block_size));
    format.configure_buffer(&mut voxels);
    voxels
}

/// Aggregate voxel storage.
///
/// Locking invariant for task code using [`SharedVoxelData`]: clone shared
/// generator/stream handles and copy cheap settings while holding the data
/// lock, then release the lock before calling generator, mesher or stream
/// methods. This mirrors the C++ contract where those shared resources are
/// thread-safe and not protected by the voxel-data map lock.
pub struct VoxelData {
    lods: Vec<VoxelDataLod>,
    format: VoxelFormat,
    bounds_in_voxels: Box3i,
    full_load_completed: bool,
    streaming_enabled: bool,
    generator: Option<SharedVoxelGenerator>,
    stream: Option<SharedVoxelStream>,
}

impl fmt::Debug for VoxelData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VoxelData")
            .field("lod_count", &self.lods.len())
            .field("format", &self.format)
            .field("bounds_in_voxels", &self.bounds_in_voxels)
            .field("streaming_enabled", &self.streaming_enabled)
            .field("full_load_completed", &self.full_load_completed)
            .field("has_generator", &self.generator.is_some())
            .field("has_stream", &self.stream.is_some())
            .finish()
    }
}

impl Default for VoxelData {
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelData {
    pub fn new() -> Self {
        let format = VoxelFormat::new();
        Self {
            lods: vec![VoxelDataLod::new(0, format)],
            format,
            bounds_in_voxels: Box3i::default(),
            full_load_completed: false,
            streaming_enabled: true,
            generator: None,
            stream: None,
        }
    }

    pub const fn block_size(&self) -> u32 {
        VoxelDataMap::BLOCK_SIZE
    }

    pub const fn block_size_po2(&self) -> u8 {
        VoxelDataMap::BLOCK_SIZE_PO2
    }

    pub fn voxel_to_block(&self, pos: Vector3i) -> Vector3i {
        VoxelDataMap::voxel_to_block_b(pos, self.block_size_po2())
    }

    pub fn block_to_voxel(&self, pos: Vector3i) -> Vector3i {
        pos * self.block_size() as i32
    }

    pub fn lod_count(&self) -> usize {
        self.lods.len()
    }

    pub fn set_lod_count(&mut self, lod_count: usize) -> Result<(), VoxelDataLodResizeError> {
        self.try_resize_lods_preserving(lod_count)
    }

    pub fn try_resize_lods_preserving(
        &mut self,
        lod_count: usize,
    ) -> Result<(), VoxelDataLodResizeError> {
        if !(1..MAX_LOD).contains(&lod_count) {
            return Err(VoxelDataLodResizeError::InvalidLodCount {
                requested: lod_count,
            });
        }

        let current_lod_count = self.lods.len();
        if lod_count == current_lod_count {
            return Ok(());
        }

        if lod_count < current_lod_count {
            if let Some((offset, _)) = self.lods[lod_count..]
                .iter()
                .enumerate()
                .find(|(_, lod)| lod.map.block_count() != 0)
            {
                return Err(VoxelDataLodResizeError::NonEmptyTruncatedLod {
                    lod_index: lod_count + offset,
                });
            }
            self.lods.truncate(lod_count);
            return Ok(());
        }

        let additional = lod_count - current_lod_count;
        self.lods
            .try_reserve_exact(additional)
            .map_err(|_| VoxelDataLodResizeError::CapacityReservationFailed)?;
        let mut appended = Vec::new();
        appended
            .try_reserve_exact(additional)
            .map_err(|_| VoxelDataLodResizeError::CapacityReservationFailed)?;
        for lod_index in current_lod_count..lod_count {
            appended.push(VoxelDataLod::new(lod_index as u8, self.format));
        }
        self.lods.append(&mut appended);
        Ok(())
    }

    pub fn reset_maps(&mut self) {
        for (lod_index, lod) in self.lods.iter_mut().enumerate() {
            lod.map.create(lod_index as u8);
            lod.map.set_format(self.format);
        }
    }

    pub const fn bounds(&self) -> Box3i {
        self.bounds_in_voxels
    }

    pub const fn set_bounds(&mut self, bounds: Box3i) {
        self.bounds_in_voxels = bounds;
    }

    pub const fn format(&self) -> VoxelFormat {
        self.format
    }

    pub fn set_format(&mut self, format: VoxelFormat) {
        if self.format == format {
            return;
        }
        self.format = format;
        self.reset_maps();
    }

    pub const fn is_streaming_enabled(&self) -> bool {
        self.streaming_enabled
    }

    pub const fn set_streaming_enabled(&mut self, enabled: bool) {
        self.streaming_enabled = enabled;
    }

    pub const fn is_full_load_completed(&self) -> bool {
        self.full_load_completed
    }

    pub const fn set_full_load_completed(&mut self, complete: bool) {
        self.full_load_completed = complete;
    }

    /// Returns a clone of the shared generator handle, if any. Cheap (one Arc
    /// refcount bump). Matches `VoxelData::get_generator` in C++.
    pub fn generator(&self) -> Option<SharedVoxelGenerator> {
        self.generator.clone()
    }

    /// Installs a shared generator. Matches `VoxelData::set_generator`.
    /// Pass `None` to detach. The handle can be safely cloned into worker
    /// tasks later; generators own any internal synchronization they need.
    pub fn set_generator(&mut self, generator: Option<SharedVoxelGenerator>) {
        self.generator = generator;
    }

    /// Runs `f` against the installed generator. Returns
    /// `None` when no generator is set. Used by `pre_generate_box` /
    /// `update_lods` when the caller doesn't pass an explicit generator.
    pub fn with_generator<R>(&self, f: impl FnOnce(&dyn VoxelGenerator) -> R) -> Option<R> {
        self.generator.as_ref().map(|gen| f(gen.as_ref()))
    }

    /// Returns a clone of the shared stream handle, if any. Matches
    /// `VoxelData::get_stream` in C++.
    pub fn stream(&self) -> Option<SharedVoxelStream> {
        self.stream.clone()
    }

    /// Installs a shared stream. Matches `VoxelData::set_stream`.
    pub fn set_stream(&mut self, stream: Option<SharedVoxelStream>) {
        self.stream = stream;
    }

    pub const fn has_generator(&self) -> bool {
        self.generator.is_some()
    }

    pub const fn has_stream(&self) -> bool {
        self.stream.is_some()
    }

    /// Copies voxel data in a box from LOD0 into `dst_buffer`. Ports
    /// `VoxelData::copy`. `channels_mask` selects which channels are read;
    /// missing blocks produce the format default. When a generator is
    /// installed and `generate_missing` is true, missing blocks inside
    /// bounds are generated on the fly instead of falling back to defaults
    /// (mirrors the C++ generator callback path).
    pub fn copy(
        &self,
        min_pos: Vector3i,
        dst_buffer: &mut VoxelBuffer,
        channels_mask: u32,
        generate_missing: bool,
    ) {
        if channels_mask == 0 {
            return;
        }
        // Match C++: configure the destination buffer with our format first.
        self.format.configure_buffer(dst_buffer);

        let dst_size = dst_buffer.size();
        if dst_size.x <= 0 || dst_size.y <= 0 || dst_size.z <= 0 {
            return;
        }

        let block_size = self.block_size() as i32;
        let max_pos = min_pos + dst_size;
        let min_block_pos = VoxelDataMap::voxel_to_block_b(min_pos, self.block_size_po2());
        let max_block_pos =
            VoxelDataMap::voxel_to_block_b(max_pos - Vector3i::splat(1), self.block_size_po2())
                + Vector3i::splat(1);

        let channels: Vec<usize> = (0..8u32)
            .filter(|ci| (channels_mask & (1u32 << ci)) != 0)
            .map(|ci| ci as usize)
            .collect();

        for block_pos in Box3i::from_min_max(min_block_pos, max_block_pos).iter_cells_zxy() {
            let src_block_origin = block_pos * block_size;
            let dst_offset = src_block_origin - min_pos;

            // Loaded edited block: copy directly from its voxel buffer.
            if let Some(block) = self.lods[0].map.get_block(block_pos) {
                if block.has_voxels() {
                    for &channel_index in &channels {
                        dst_buffer.copy_channel_from_area(
                            block.voxels(),
                            Vector3i::zero(),
                            block.voxels().size(),
                            dst_offset,
                            channel_index,
                        );
                    }
                    continue;
                }
            }

            // Missing block: generate on the fly if a generator is available
            // and the area is inside bounds; otherwise leave the default.
            if generate_missing
                && self.generator.is_some()
                && self.bounds_in_voxels.contains_point(src_block_origin)
            {
                let mut scratch = self.create_block_buffer();
                self.with_generator(|gen| {
                    gen.generate_block(VoxelQueryData {
                        buffer: &mut scratch,
                        origin_in_voxels: src_block_origin,
                        lod: 0,
                    });
                });
                for &channel_index in &channels {
                    dst_buffer.copy_channel_from_area(
                        &scratch,
                        Vector3i::zero(),
                        scratch.size(),
                        dst_offset,
                        channel_index,
                    );
                }
            }
        }
    }

    /// Pastes `src_buffer` into LOD0 at `min_pos`. Ports `VoxelData::paste`.
    /// `channels_mask` selects which channels are written.
    /// `create_new_blocks` controls whether missing destination blocks are
    /// materialised (as formatted empty buffers) before writing.
    pub fn paste(
        &mut self,
        min_pos: Vector3i,
        src_buffer: &VoxelBuffer,
        channels_mask: u32,
        create_new_blocks: bool,
    ) {
        self.lods[0]
            .map
            .paste(min_pos, src_buffer, channels_mask, create_new_blocks);
    }

    /// Pastes `src_buffer` into LOD0 with a source mask. Ports
    /// `VoxelData::paste_masked`. Voxels of `src_buffer` whose
    /// `src_mask_channel` equals `src_mask_value` are skipped.
    pub fn paste_masked(
        &mut self,
        min_pos: Vector3i,
        src_buffer: &VoxelBuffer,
        channels_mask: u32,
        src_mask_channel: usize,
        src_mask_value: u64,
        create_new_blocks: bool,
    ) {
        self.lods[0].map.paste_masked(
            min_pos,
            src_buffer,
            channels_mask,
            src_mask_channel,
            src_mask_value,
            create_new_blocks,
        );
    }

    /// Pastes `src_buffer` into LOD0 with a source mask and a destination
    /// writable-values list. Ports `VoxelData::paste_masked_writable_list`.
    /// Voxels of `src_buffer` whose `src_mask_channel` equals `src_mask_value`
    /// are skipped; voxels of the destination whose `dst_mask_channel` value
    /// is not in `dst_writable_values` are also skipped.
    #[allow(clippy::too_many_arguments)]
    pub fn paste_masked_with_destination_mask(
        &mut self,
        min_pos: Vector3i,
        src_buffer: &VoxelBuffer,
        channels_mask: u32,
        src_mask_channel: usize,
        src_mask_value: u64,
        dst_mask_channel: usize,
        dst_writable_values: &[u64],
        create_new_blocks: bool,
    ) {
        self.lods[0].map.paste_masked_with_destination_mask(
            min_pos,
            src_buffer,
            channels_mask,
            src_mask_channel,
            src_mask_value,
            dst_mask_channel,
            dst_writable_values,
            create_new_blocks,
        );
    }

    /// Tests whether every block intersecting the given voxel box at LOD0 is
    /// loaded. Ports `VoxelData::is_area_loaded`. The C++ version also
    /// short-circuits to false when streaming is enabled and the area
    /// extends outside the bounds (we replicate that here).
    pub fn is_area_loaded(&self, voxel_box: Box3i) -> bool {
        if self.streaming_enabled && !self.bounds_in_voxels.contains_box(voxel_box) {
            return false;
        }
        self.lods[0].map.is_area_fully_loaded(voxel_box)
    }

    /// Tests if all blocks in the given block-coord box at `lod_index` are
    /// loaded, accounting for data boundaries. Ports
    /// `VoxelData::has_all_blocks_in_area`.
    pub fn has_all_blocks_in_area(&self, blocks_box: Box3i, lod_index: usize) -> bool {
        let Some(lod) = self.lods.get(lod_index) else {
            return false;
        };
        blocks_box.all_cells_match(|pos| lod.map.has_block(pos))
    }

    /// Appends block positions inside `blocks_box` at `lod_index` that are
    /// not loaded. Ports `VoxelData::get_missing_blocks` (the box overload).
    pub fn get_missing_blocks(
        &self,
        blocks_box: Box3i,
        lod_index: usize,
        out_missing: &mut Vec<Vector3i>,
    ) {
        let Some(lod) = self.lods.get(lod_index) else {
            out_missing.extend(blocks_box.iter_cells_zxy());
            return;
        };
        for pos in blocks_box.iter_cells_zxy() {
            if !lod.map.has_block(pos) {
                out_missing.push(pos);
            }
        }
    }

    /// Returns references to the voxel buffers of every block with voxel data
    /// in `blocks_box` at `lod_index`, indexed into a flat ZXY grid covering
    /// the box. Missing or empty entries are left as `None`. Ports
    /// `VoxelData::get_blocks_with_voxel_data`.
    pub fn get_blocks_with_voxel_data(
        &self,
        blocks_box: Box3i,
        lod_index: usize,
    ) -> Vec<Option<&VoxelBuffer>> {
        let mut out = Vec::new();
        let Some(lod) = self.lods.get(lod_index) else {
            return out;
        };
        let size = blocks_box.size;
        out.reserve_exact((size.x as usize) * (size.y as usize) * (size.z as usize));
        for pos in blocks_box.iter_cells_zxy() {
            let buffer = lod
                .map
                .get_block(pos)
                .filter(|block| block.has_voxels())
                .map(|block| block.voxels());
            out.push(buffer);
        }
        out
    }

    pub fn get_voxel(&self, pos: Vector3i, channel_index: usize, defval: u64) -> u64 {
        if !self.bounds_in_voxels.contains_point(pos) {
            return defval;
        }
        if !self.streaming_enabled && !self.full_load_completed {
            return defval;
        }

        if !self.streaming_enabled {
            // Non-streaming: every block is expected to be loaded. If a block
            // or its voxels are missing, fall back to the generator (single
            // voxel query) — mirrors the C++ branch at voxel_data.cpp:182-200.
            let block_pos = self.voxel_to_block(pos);
            if let Some(block) = self.lods[0].map.get_block(block_pos) {
                if block.has_voxels() {
                    let local_pos = self.lods[0].map.to_local(pos);
                    return block.voxels().get_voxel(
                        local_pos.x,
                        local_pos.y,
                        local_pos.z,
                        channel_index,
                    );
                }
            }
            return self
                .with_generator(|gen| gen.generate_single(pos, channel_index).as_raw())
                .unwrap_or(defval);
        }

        // Streaming mode: probe LODs from finest to coarsest, falling back to
        // a lower LOD when the finer one isn't resident. If none is resident
        // and a generator is available, query it directly (matches the C++
        // behaviour at voxel_data.cpp:209-254).
        let mut block_pos = self.voxel_to_block(pos);
        let mut voxel_pos = pos;
        for lod_index in 0..self.lods.len() {
            if let Some(block) = self.lods[lod_index].map.get_block(block_pos) {
                if block.has_voxels() {
                    let local_pos = self.lods[lod_index].map.to_local(voxel_pos);
                    return block.voxels().get_voxel(
                        local_pos.x,
                        local_pos.y,
                        local_pos.z,
                        channel_index,
                    );
                }
            }
            block_pos = block_pos >> 1;
            voxel_pos = voxel_pos >> 1;
        }
        self.with_generator(|gen| gen.generate_single(pos, channel_index).as_raw())
            .unwrap_or(defval)
    }

    pub fn get_voxel_f(&self, pos: Vector3i, channel_index: usize) -> f32 {
        let raw = self.get_voxel(
            pos,
            channel_index,
            real_to_raw_voxel(SDF_FAR_OUTSIDE, self.format.depths[channel_index]),
        );
        raw_voxel_to_real(raw, self.format.depths[channel_index])
    }

    pub fn try_set_voxel(&mut self, value: u64, pos: Vector3i, channel_index: usize) -> bool {
        if !self.bounds_in_voxels.contains_point(pos) {
            return false;
        }
        let block_pos = self.voxel_to_block(pos);
        let block_state = self.lods[0]
            .map
            .get_block(block_pos)
            .map(|block| block.has_voxels());

        match block_state {
            Some(true) => {}
            Some(false) => {
                let voxels = self.create_block_buffer();
                self.lods[0].map.set_block_buffer(block_pos, voxels, true);
            }
            None => {
                if self.streaming_enabled || !self.full_load_completed {
                    return false;
                }
                let voxels = self.create_block_buffer();
                self.lods[0].map.set_block_buffer(block_pos, voxels, true);
            }
        }

        self.lods[0].map.set_voxel(value, pos, channel_index);
        true
    }

    fn create_block_buffer(&self) -> VoxelBuffer {
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(self.block_size() as i32));
        self.format.configure_buffer(&mut voxels);
        voxels
    }

    pub fn try_get_block_voxels(&self, block_pos: Vector3i) -> Option<&VoxelBuffer> {
        self.get_block(block_pos, 0).and_then(|block| {
            if block.has_voxels() {
                Some(block.voxels())
            } else {
                None
            }
        })
    }

    pub fn try_set_voxel_f(&mut self, value: f32, pos: Vector3i, channel_index: usize) -> bool {
        let raw = real_to_raw_voxel(value, self.format.depths[channel_index]);
        self.try_set_voxel(raw, pos, channel_index)
    }

    pub fn try_set_block(&mut self, block_pos: Vector3i, block: VoxelDataBlock) -> bool {
        let lod_index = usize::from(block.lod_index());
        assert!(lod_index < self.lods.len(), "block LOD is not loaded");
        if block.has_voxels() {
            assert_eq!(
                block.voxels().size(),
                Vector3i::splat(self.block_size() as i32),
                "block voxels must match VoxelData block size"
            );
        }
        if self.lods[lod_index].map.has_block(block_pos) {
            return false;
        }
        self.lods[lod_index].map.set_block(block_pos, block, false);
        true
    }

    pub fn has_block(&self, block_pos: Vector3i, lod_index: usize) -> bool {
        self.lods
            .get(lod_index)
            .is_some_and(|lod| lod.map.has_block(block_pos))
    }

    pub fn block_count(&self) -> usize {
        self.lods.iter().map(|lod| lod.map.block_count()).sum()
    }

    pub fn mark_area_modified(
        &mut self,
        voxel_box: Box3i,
        require_lod_updates: bool,
    ) -> Vec<Vector3i> {
        let blocks_box = voxel_box.downscaled(self.block_size() as i32);
        let mut newly_needing_lod = Vec::new();
        for block_pos in blocks_box.iter_cells_zxy() {
            let Some(block) = self.lods[0].map.get_block_mut(block_pos) else {
                continue;
            };
            if !block.has_voxels() {
                continue;
            }
            block.set_modified(true);
            block.set_edited(true);
            if require_lod_updates && !block.needs_lodding() {
                block.set_needs_lodding(true);
                newly_needing_lod.push(block_pos);
            }
        }
        newly_needing_lod
    }

    /// Propagates LOD0 edits to higher LODs by 2:1 downscaling.
    ///
    /// Ports `VoxelData::update_lods`. The caller passes the LOD0 blocks that
    /// were marked as needing LOD updates (typically the result of
    /// [`mark_area_modified`]). The function walks up the LOD chain in pairs:
    /// for each source (lower-LOD) block it finds or generates the destination
    /// (higher-LOD) block, marks it modified, and downscales the source
    /// voxels into the matching sub-region of the destination.
    ///
    /// When `generator` is `Some`, missing or empty destination blocks in
    /// non-streaming mode are filled by the generator before downscaling
    /// (matching the C++ `L::generate_voxels` path). In streaming mode the
    /// destination is expected to already be resident; if not, the function
    /// logs the discrepancy and skips that pair (the C++ branch prints an
    /// error and continues).
    ///
    /// If `out_updated_blocks` is `Some`, every block touched at every LOD is
    /// appended (LOD0 first, then progressively higher LODs). This mirrors
    /// the C++ `StdVector<BlockLocation> *out_updated_blocks` parameter.
    pub fn update_lods(
        &mut self,
        modified_lod0_blocks: &[Vector3i],
        generator: Option<&dyn VoxelGenerator>,
        mut out_updated_blocks: Option<&mut Vec<BlockLocation>>,
    ) {
        let lod_count = self.lods.len();
        if lod_count < 2 && modified_lod0_blocks.is_empty() {
            // Single-LOD case still needs to clear the needs_lodding flag so
            // the caller doesn't see stale state; handled below.
        }

        // Per-LOD worklists. Index 0 is seeded from the caller's input; each
        // successive LOD is filled by the cascade. Using a small fixed-size
        // `Vec<Vec<_>>` mirrors the C++ `thread_local FixedArray<...,MAX_LOD>`.
        let mut blocks_to_process_per_lod: Vec<Vec<Vector3i>> = (0..lod_count)
            .map(|i| {
                if i == 0 {
                    modified_lod0_blocks.to_vec()
                } else {
                    Vec::new()
                }
            })
            .collect();

        // LOD0 phase: clear needs_lodding and record updates.
        for &block_pos in &blocks_to_process_per_lod[0] {
            let Some(block) = self.lods[0].map.get_block_mut(block_pos) else {
                // C++ uses ERR_CONTINUE; we just skip the missing block.
                continue;
            };
            block.set_needs_lodding(false);
            if let Some(out) = out_updated_blocks.as_deref_mut() {
                out.push(BlockLocation {
                    position: block_pos,
                    lod_index: 0,
                });
            }
        }

        let half_bs = (self.block_size() as i32) >> 1;
        let last_lod_index = lod_count - 1;

        // Cascade upwards in pairs of consecutive LODs.
        for dst_lod_index in 1..lod_count {
            let src_lod_index = dst_lod_index - 1;
            // Snapshot the src worklist so we can borrow `self` mutably inside
            // the loop without holding the borrow across iterations.
            let src_worklist = std::mem::take(&mut blocks_to_process_per_lod[src_lod_index]);

            for src_bpos in src_worklist {
                let dst_bpos = src_bpos >> 1;

                // Resolve the source block. C++ asserts non-null; the input
                // contract guarantees the block exists (it came from a
                // `needs_lodding` flag set by mark_area_modified).
                let src_has_voxels = self.lods[src_lod_index]
                    .map
                    .get_block_mut(src_bpos)
                    .is_some_and(|block| {
                        block.set_needs_lodding(false);
                        block.has_voxels()
                    });
                if !src_has_voxels {
                    // Source block missing or empty — nothing to downscale.
                    continue;
                }

                // Resolve (or generate) the destination block.
                let dst_exists = self.lods[dst_lod_index].map.has_block(dst_bpos);
                if !dst_exists {
                    if !self.streaming_enabled {
                        // Generate an empty destination block and fill it via
                        // the generator before downscaling. Matches C++.
                        let mut voxels = self.create_block_buffer();
                        if let Some(generator) = generator {
                            let lod_block_size = (self.block_size() as i32) << dst_lod_index;
                            generator.generate_block(VoxelQueryData {
                                buffer: &mut voxels,
                                origin_in_voxels: dst_bpos * lod_block_size,
                                lod: dst_lod_index as u32,
                            });
                        }
                        self.lods[dst_lod_index]
                            .map
                            .set_block_buffer(dst_bpos, voxels, true);
                    } else {
                        // Streaming mode expects parents to be resident. The
                        // C++ branch prints an error and `continue`s.
                        // TODO: route via the project logger once integrated.
                        continue;
                    }
                }

                // The destination may still have no voxel buffer (loaded but
                // uncached). Generate on the fly like C++.
                let dst_has_voxels = self.lods[dst_lod_index]
                    .map
                    .get_block(dst_bpos)
                    .is_some_and(|block| block.has_voxels());
                if !dst_has_voxels {
                    let mut voxels = self.create_block_buffer();
                    if let Some(generator) = generator {
                        let lod_block_size = (self.block_size() as i32) << dst_lod_index;
                        generator.generate_block(VoxelQueryData {
                            buffer: &mut voxels,
                            origin_in_voxels: dst_bpos * lod_block_size,
                            lod: dst_lod_index as u32,
                        });
                    }
                    if let Some(block) = self.lods[dst_lod_index].map.get_block_mut(dst_bpos) {
                        block.set_voxels(voxels);
                    }
                }

                // Mark modified and enqueue for the next LOD pass if needed.
                let mut enqueue_next = false;
                if let Some(block) = self.lods[dst_lod_index].map.get_block_mut(dst_bpos) {
                    block.set_modified(true);
                    if dst_lod_index != last_lod_index && !block.needs_lodding() {
                        block.set_needs_lodding(true);
                        enqueue_next = true;
                    }
                }
                if enqueue_next {
                    blocks_to_process_per_lod[dst_lod_index].push(dst_bpos);
                }

                if let Some(out) = out_updated_blocks.as_deref_mut() {
                    out.push(BlockLocation {
                        position: dst_bpos,
                        lod_index: dst_lod_index as u8,
                    });
                }

                // Downscale source into the matching sub-region of the dst.
                // `rel = src_bpos - (dst_bpos << 1)` selects one of the 2×2×2
                // octants of the destination block; scaled by `half_bs` it
                // gives the destination-local offset of that octant.
                let rel = src_bpos - (dst_bpos << 1);
                let dst_offset = rel * half_bs;

                // Borrow src and dst blocks independently. `src_lod_index` is
                // always less than `dst_lod_index`, so we split the LOD slice
                // to convince the borrow checker the two borrows are disjoint.
                let (src_lods, dst_lods) = self.lods.split_at_mut(dst_lod_index);
                let Some(src_block) = src_lods[src_lod_index].map.get_block(src_bpos) else {
                    continue;
                };
                let Some(dst_block) = dst_lods[0].map.get_block_mut(dst_bpos) else {
                    continue;
                };

                // Copy the source voxels into a temporary so we don't hold a
                // borrow of `src_block` while mutating `dst_block` (the two
                // live in different LOD maps but share the same `&mut self`).
                // `downscale_to` takes `&self` and `&mut dst`, and our two
                // references come from disjoint LOD slices, so this is sound.
                let src_size = src_block.voxels().size();
                let dst_voxels = dst_block.voxels_mut();
                src_block
                    .voxels()
                    .downscale_to(dst_voxels, Vector3i::zero(), src_size, dst_offset);
            }
        }
    }

    pub fn pre_generate_box(
        &mut self,
        voxel_box: Box3i,
        generator: Option<&dyn VoxelGenerator>,
    ) -> usize {
        let mut generated_count = 0;
        let data_block_size = self.block_size() as i32;
        for lod_index in 0..self.lods.len() {
            let lod_block_size = data_block_size << lod_index;
            let block_box = voxel_box.downscaled(lod_block_size);
            for block_pos in block_box.iter_cells_zxy() {
                let should_generate = match self.lods[lod_index].map.get_block(block_pos) {
                    Some(block) => !block.has_voxels(),
                    None => !self.streaming_enabled,
                };
                if !should_generate {
                    continue;
                }

                let mut voxels = self.create_block_buffer();
                if let Some(generator) = generator {
                    generator.generate_block(VoxelQueryData {
                        buffer: &mut voxels,
                        origin_in_voxels: block_pos * lod_block_size,
                        lod: lod_index as u32,
                    });
                }

                if self.lods[lod_index]
                    .map
                    .get_block(block_pos)
                    .is_some_and(|block| block.has_voxels())
                {
                    continue;
                }

                self.lods[lod_index]
                    .map
                    .set_block_buffer(block_pos, voxels, true);
                generated_count += 1;
            }
        }
        generated_count
    }

    pub fn consume_block_modifications(&mut self, block_pos: Vector3i) -> Option<BlockToSave> {
        self.consume_block_modifications_at(block_pos, 0)
    }

    pub fn consume_all_modifications(&mut self) -> Vec<BlockToSave> {
        let mut saves = Vec::new();
        for lod_index in 0..self.lods.len() {
            let block_positions: Vec<_> = self.lods[lod_index].map.block_positions().collect();
            for block_pos in block_positions {
                if let Some(save) = self.consume_block_modifications_at(block_pos, lod_index) {
                    saves.push(save);
                }
            }
        }
        saves
    }

    fn consume_block_modifications_at(
        &mut self,
        block_pos: Vector3i,
        lod_index: usize,
    ) -> Option<BlockToSave> {
        let lod = self.lods.get_mut(lod_index)?;
        let block = lod.map.get_block_mut(block_pos)?;
        if !block.is_modified() {
            return None;
        }
        let voxels = if block.has_voxels() {
            Some(block.voxels().copy_to_owned())
        } else {
            None
        };
        block.set_modified(false);
        Some(BlockToSave {
            voxels,
            position: block_pos,
            lod_index: lod_index as u8,
            block_revision: 0,
        })
    }

    pub fn unload_blocks(
        &mut self,
        blocks_box: Box3i,
        lod_index: usize,
        collect_modified: bool,
    ) -> Vec<BlockToSave> {
        let Some(lod) = self.lods.get_mut(lod_index) else {
            return Vec::new();
        };
        let mut saves = Vec::new();
        for block_pos in blocks_box.iter_cells_zxy() {
            let Some(block) = lod.map.remove_block(block_pos) else {
                continue;
            };
            if collect_modified && block.is_modified() {
                saves.push(BlockToSave {
                    voxels: block.into_voxels(),
                    position: block_pos,
                    lod_index: lod_index as u8,
                    block_revision: 0,
                });
            }
        }
        saves
    }

    pub fn get_block(&self, block_pos: Vector3i, lod_index: usize) -> Option<&VoxelDataBlock> {
        self.lods
            .get(lod_index)
            .and_then(|lod| lod.map.get_block(block_pos))
    }

    /// Increases the reference count of every loaded block in `blocks_box` at
    /// `lod_index`, returning the positions of the missing (not-loaded) ones
    /// and optionally shallow copies of the found blocks / their positions.
    ///
    /// Ports `VoxelData::view_area`. The C++ method is used by mesh block
    /// tasks to pin blocks they will read while the mesher runs on a worker
    /// thread. `unview_area` is the matching release.
    pub fn view_area(
        &mut self,
        mut blocks_box: Box3i,
        lod_index: usize,
        missing_blocks: Option<&mut Vec<Vector3i>>,
        found_blocks_positions: Option<&mut Vec<Vector3i>>,
        found_blocks: Option<&mut Vec<VoxelDataBlock>>,
    ) {
        let Some(bounds_in_blocks) =
            try_bounds_in_lod_blocks(self.bounds_in_voxels, self.block_size() as i32, lod_index)
        else {
            if let Some(out) = missing_blocks {
                extend_missing_blocks_if_representable(out, blocks_box);
            }
            return;
        };
        let Ok(clipped_blocks_box) = checked_box_intersection(blocks_box, bounds_in_blocks) else {
            return;
        };
        blocks_box = clipped_blocks_box;

        let Some(lod) = self.lods.get_mut(lod_index) else {
            return;
        };

        let mut missing_local = Vec::new();
        let mut found_positions_local = Vec::new();
        let mut found_blocks_local: Vec<VoxelDataBlock> = Vec::new();

        for bpos in blocks_box.iter_cells_zxy() {
            match lod.map.get_block_mut(bpos) {
                Some(block) => {
                    block.viewers.add();
                    if found_blocks.is_some() {
                        // Shallow copy: voxels are deep, but the C++ path also
                        // returns a full copy of the `VoxelDataBlock` value.
                        found_blocks_local.push(clone_block(block));
                    }
                    if found_blocks_positions.is_some() {
                        found_positions_local.push(bpos);
                    }
                }
                None => {
                    if missing_blocks.is_some() {
                        missing_local.push(bpos);
                    }
                }
            }
        }

        if let Some(out) = missing_blocks {
            out.extend(missing_local);
        }
        if let Some(out) = found_blocks_positions {
            out.extend(found_positions_local);
        }
        if let Some(out) = found_blocks {
            out.extend(found_blocks_local);
        }
    }

    /// Decreases the reference count of every loaded block in `blocks_box` at
    /// `lod_index`. Blocks reaching zero viewers are removed; if they were
    /// modified and `to_save` is provided, their voxels are returned for the
    /// caller to persist. Ports `VoxelData::unview_area`.
    pub fn unview_area(
        &mut self,
        mut blocks_box: Box3i,
        lod_index: usize,
        removed_blocks: Option<&mut Vec<Vector3i>>,
        missing_blocks: Option<&mut Vec<Vector3i>>,
        mut to_save: Option<&mut Vec<BlockToSave>>,
    ) {
        let Some(bounds_in_blocks) =
            try_bounds_in_lod_blocks(self.bounds_in_voxels, self.block_size() as i32, lod_index)
        else {
            if let Some(out) = missing_blocks {
                extend_missing_blocks_if_representable(out, blocks_box);
            }
            return;
        };
        let Ok(clipped_blocks_box) = checked_box_intersection(blocks_box, bounds_in_blocks) else {
            return;
        };
        blocks_box = clipped_blocks_box;

        let Some(lod) = self.lods.get_mut(lod_index) else {
            // Still report every block as missing to mirror C++ behaviour.
            if let Some(out) = missing_blocks {
                out.extend(blocks_box.iter_cells_zxy());
            }
            return;
        };

        let mut removed_local = Vec::new();
        let mut missing_local = Vec::new();
        let saves_local: Vec<BlockToSave> = Vec::new();

        for bpos in blocks_box.iter_cells_zxy() {
            // Borrow, decrement, and decide whether to remove. We do this in
            // two steps because removing the block invalidates any outstanding
            // borrow of the map.
            let should_remove = match lod.map.get_block_mut(bpos) {
                Some(block) => {
                    block.viewers.remove();
                    block.viewers.get() == 0
                }
                None => {
                    missing_local.push(bpos);
                    continue;
                }
            };

            if should_remove {
                if let Some(block) = lod.map.remove_block(bpos) {
                    if let Some(out) = to_save.as_deref_mut() {
                        if block.is_modified() {
                            out.push(BlockToSave {
                                voxels: block.into_voxels(),
                                position: bpos,
                                lod_index: lod_index as u8,
                                block_revision: 0,
                            });
                        }
                    }
                    removed_local.push(bpos);
                }
            }
        }

        if let Some(out) = removed_blocks {
            out.extend(removed_local);
        }
        if let Some(out) = missing_blocks {
            out.extend(missing_local);
        }
        if let Some(out) = to_save {
            out.extend(saves_local);
        }
    }
}

/// Copy a `VoxelDataBlock` for `view_area`'s found-blocks return. The C++
/// implementation returns a full value copy of the block; we do the same,
/// deep-copying the underlying `VoxelBuffer`. The refcount is also copied
/// (post-increment) so the snapshot reflects the live count.
fn clone_block(block: &VoxelDataBlock) -> VoxelDataBlock {
    let mut copy = match block.has_voxels() {
        true => VoxelDataBlock::with_voxels(block.voxels().copy_to_owned(), block.lod_index()),
        false => VoxelDataBlock::empty(block.lod_index()),
    };
    copy.set_modified(block.is_modified());
    copy.set_edited(block.is_edited());
    copy.set_needs_lodding(block.needs_lodding());
    copy.viewers = block.viewers;
    copy
}

fn try_bounds_in_lod_blocks(
    bounds_voxels: Box3i,
    block_size: i32,
    lod_index: usize,
) -> Option<Box3i> {
    let lod_index = u8::try_from(lod_index).ok()?;
    bounds_in_lod_blocks(bounds_voxels, block_size, lod_index).ok()
}

fn checked_lod_block_bounds(
    blocks_box: Box3i,
    block_size: u32,
    lod_index: usize,
) -> Option<BoxBounds3i> {
    let shift = u32::try_from(lod_index).ok()?;
    let stride = i64::from(block_size).checked_shl(shift)?;
    let scaled_axis = |position: i32, size: i32| {
        let min = i64::from(position).checked_mul(stride)?;
        let end = i64::from(position).checked_add(i64::from(size))?;
        let max = end.checked_mul(stride)?;
        Some((i32::try_from(min).ok()?, i32::try_from(max).ok()?))
    };
    let (min_x, max_x) = scaled_axis(blocks_box.position.x, blocks_box.size.x)?;
    let (min_y, max_y) = scaled_axis(blocks_box.position.y, blocks_box.size.y)?;
    let (min_z, max_z) = scaled_axis(blocks_box.position.z, blocks_box.size.z)?;
    Some(BoxBounds3i::new(
        Vector3i::new(min_x, min_y, min_z),
        Vector3i::new(max_x, max_y, max_z),
    ))
}

fn current_key_revision(map: &VoxelDataMap, position: Vector3i) -> VoxelDataKeyRevision {
    let revision = map.key_revision(position);
    if map.has_block(position) {
        VoxelDataKeyRevision::Present(revision)
    } else {
        VoxelDataKeyRevision::Tombstone(revision)
    }
}

const fn key_revision_value(revision: VoxelDataKeyRevision) -> u64 {
    match revision {
        VoxelDataKeyRevision::Present(value) | VoxelDataKeyRevision::Tombstone(value) => value,
    }
}

fn checked_box_cell_count(box_: Box3i) -> Option<usize> {
    if box_.size.x <= 0 || box_.size.y <= 0 || box_.size.z <= 0 {
        return Some(0);
    }
    [box_.size.x, box_.size.y, box_.size.z]
        .into_iter()
        .try_fold(1usize, |count, axis| {
            count.checked_mul(usize::try_from(axis).ok()?)
        })
}

fn reserve_output<T>(
    output: &mut Option<&mut Vec<T>>,
    additional: usize,
) -> Result<(), SharedVoxelDataMutationError> {
    if let Some(output) = output.as_deref_mut() {
        output
            .try_reserve_exact(additional)
            .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
    }
    Ok(())
}

fn try_extend_missing_blocks_if_representable(
    out: &mut Vec<Vector3i>,
    blocks_box: Box3i,
) -> Result<(), SharedVoxelDataMutationError> {
    let Ok(validated) = checked_box_intersection(blocks_box, blocks_box) else {
        return Ok(());
    };
    let count = checked_box_cell_count(validated)
        .ok_or(SharedVoxelDataMutationError::CapacityReservationFailed)?;
    out.try_reserve_exact(count)
        .map_err(|_| SharedVoxelDataMutationError::CapacityReservationFailed)?;
    out.extend(validated.iter_cells_zxy());
    Ok(())
}

fn extend_missing_blocks_if_representable(out: &mut Vec<Vector3i>, blocks_box: Box3i) {
    if let Ok(validated) = checked_box_intersection(blocks_box, blocks_box) {
        out.extend(validated.iter_cells_zxy());
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BlockLocation, SharedVoxelData, SharedVoxelDataEditPhase, SharedVoxelDataMutationError,
        SharedVoxelDataTransactionOperation, SharedVoxelDataTransactionReservation,
        SharedVoxelGenerator, VoxelData, VoxelDataKeyRevision, VoxelDataLodResizeError,
    };
    use crate::constants::voxel_constants::MAX_LOD;
    use crate::generators::base::{GenResult, VoxelGenerator, VoxelQueryData};
    use crate::math::{Box3i, Vector3i};
    use crate::storage::{
        ChannelDepth, ChannelId, VoxelBuffer, VoxelDataBlock, VoxelFormat, VoxelMemoryPool,
    };
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    #[derive(Default)]
    struct RecordingGenerator {
        calls: Mutex<Vec<(Vector3i, u32)>>,
    }

    impl VoxelGenerator for RecordingGenerator {
        fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
            self.calls
                .lock()
                .unwrap()
                .push((input.origin_in_voxels, input.lod));
            let value = 10 + input.lod as u64 + input.origin_in_voxels.x as u64;
            input.buffer.fill(value, ChannelId::Type.index());
            GenResult::default()
        }

        fn used_channels_mask(&self) -> u32 {
            1 << ChannelId::Type.index()
        }
    }

    #[test]
    fn lod_count_resizes_maps_and_reset_preserves_settings() {
        let mut data = VoxelData::new();
        assert_eq!(data.lod_count(), 1);

        data.set_lod_count(3).unwrap();
        assert_eq!(data.lod_count(), 3);
        assert_eq!(data.block_count(), 0);

        let bounds = Box3i::new(Vector3i::new(-16, -16, -16), Vector3i::new(32, 32, 32));
        data.set_bounds(bounds);
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let channel = ChannelId::Type.index();
        assert!(data.try_set_voxel(11, Vector3i::zero(), channel));
        assert_eq!(data.block_count(), 1);

        data.reset_maps();

        assert_eq!(data.lod_count(), 3);
        assert_eq!(data.bounds(), bounds);
        assert!(data.is_full_load_completed());
        assert_eq!(data.block_count(), 0);
    }

    fn make_tagged_lod_block(value: u64, lod_index: u8) -> VoxelDataBlock {
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(
            crate::storage::VoxelDataMap::BLOCK_SIZE as i32,
        ));
        voxels.set_voxel(value, 0, 0, 0, ChannelId::Type.index());
        VoxelDataBlock::with_voxels(voxels, lod_index)
    }

    fn make_shared_data_with_lods(lod_count: usize) -> Arc<SharedVoxelData> {
        let mut data = VoxelData::new();
        data.set_lod_count(lod_count).unwrap();
        data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048)));
        Arc::new(SharedVoxelData::new(data))
    }

    fn make_atomic_edit_shared_data(lod_count: usize) -> Arc<SharedVoxelData> {
        let mut data = VoxelData::new();
        data.set_lod_count(lod_count).unwrap();
        data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        Arc::new(SharedVoxelData::new(data))
    }

    fn block_buffer_identity(block: &VoxelDataBlock) -> *const u8 {
        let bytes = block.voxels().channel_bytes(ChannelId::Type.index());
        assert!(
            !bytes.is_empty(),
            "tagged test buffers must be materialized"
        );
        bytes.as_ptr()
    }

    fn location(position: Vector3i, lod_index: u8) -> BlockLocation {
        BlockLocation {
            position,
            lod_index,
        }
    }

    fn sample_atomic_edit_lods(shared: &SharedVoxelData) -> [(u64, VoxelDataKeyRevision); 3] {
        let regions = (0..3)
            .map(|lod_index| {
                let extent = (shared.block_size() as i32) << lod_index;
                shared.read_region(
                    lod_index,
                    Box3i::new(Vector3i::zero(), Vector3i::splat(extent)),
                )
            })
            .collect::<Vec<_>>();
        let sample = std::array::from_fn(|lod_index| {
            (
                shared.with_lod_map(lod_index, |map| {
                    map.get_block(Vector3i::zero()).unwrap().voxels().get_voxel(
                        0,
                        0,
                        0,
                        ChannelId::Type.index(),
                    )
                }),
                shared.key_revision(Vector3i::zero(), lod_index).unwrap(),
            )
        });
        drop(regions);
        sample
    }

    fn assert_insert_operation_identity(
        operations: &[SharedVoxelDataTransactionOperation],
        location: BlockLocation,
        identity: *const u8,
    ) {
        let block = operations.iter().find_map(|operation| match operation {
            SharedVoxelDataTransactionOperation::Insert {
                location: operation_location,
                block,
                ..
            } if *operation_location == location => Some(block),
            _ => None,
        });
        assert_eq!(block.map(block_buffer_identity), Some(identity));
    }

    #[test]
    fn prepared_edit_materializes_and_propagates_all_lods_in_one_revisioned_batch() {
        let shared = make_atomic_edit_shared_data(3);
        let channel = ChannelId::Type.index();
        let edited_voxel = Vector3i::new(20, 4, 4);
        let mut prepared = shared
            .prepare_voxel_edit(77, edited_voxel, channel)
            .unwrap()
            .expect("non-streaming storage must materialize the complete edit pyramid");

        assert_eq!(prepared.edited_block(), location(Vector3i::new(1, 0, 0), 0));
        assert_eq!(prepared.block_revision(), 1);
        assert_eq!(shared.block_count(), 0, "preparation must not publish data");
        let draft_identities = (0..3)
            .map(|lod_index| {
                let position = if lod_index == 0 {
                    Vector3i::new(1, 0, 0)
                } else {
                    Vector3i::zero()
                };
                block_buffer_identity(
                    prepared
                        .transaction_mut()
                        .owned_block(location(position, lod_index as u8))
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let outcome = prepared.transaction_mut().commit().unwrap();
        assert!(outcome.removed_blocks().is_empty());

        let expected = [
            (Vector3i::new(1, 0, 0), Vector3i::new(4, 4, 4)),
            (Vector3i::zero(), Vector3i::new(10, 2, 2)),
            (Vector3i::zero(), Vector3i::new(5, 1, 1)),
        ];
        for (lod_index, (block_position, local_position)) in expected.into_iter().enumerate() {
            let block = shared.block_snapshot(block_position, lod_index).unwrap();
            assert_eq!(
                shared.with_lod_map(lod_index, |map| {
                    block_buffer_identity(map.get_block(block_position).unwrap())
                }),
                draft_identities[lod_index]
            );
            assert_eq!(
                block.voxels().get_voxel(
                    local_position.x,
                    local_position.y,
                    local_position.z,
                    channel,
                ),
                77
            );
            assert!(block.is_modified());
            assert_eq!(block.is_edited(), lod_index == 0);
            assert!(!block.needs_lodding());
            assert_eq!(block.viewers.get(), 0);
            assert_eq!(
                shared.key_revision(block_position, lod_index),
                Some(VoxelDataKeyRevision::Present(1))
            );
        }
    }

    #[test]
    fn prepared_edit_invalid_channel_and_unavailable_block_are_non_mutating() {
        let shared = make_atomic_edit_shared_data(3);
        assert!(matches!(
            shared.prepare_voxel_edit(
                1,
                Vector3i::zero(),
                crate::storage::voxel_buffer::MAX_CHANNELS,
            ),
            Err(SharedVoxelDataMutationError::InvalidChannel {
                channel_index: crate::storage::voxel_buffer::MAX_CHANNELS,
            })
        ));
        assert_eq!(shared.block_count(), 0);

        let streaming = make_shared_data_with_lods(3);
        assert!(streaming
            .prepare_voxel_edit(2, Vector3i::zero(), ChannelId::Type.index())
            .unwrap()
            .is_none());
        assert_eq!(streaming.block_count(), 0);
        for lod_index in 0..3 {
            assert_eq!(
                streaming.key_revision(Vector3i::zero(), lod_index),
                Some(VoxelDataKeyRevision::Tombstone(0))
            );
        }

        let mut resident_lod0 = make_tagged_lod_block(7, 0);
        resident_lod0.viewers.set_exact(4);
        resident_lod0.set_modified(true);
        resident_lod0.set_edited(true);
        resident_lod0.set_needs_lodding(true);
        let resident_identity = block_buffer_identity(&resident_lod0);
        assert!(streaming
            .try_set_block(Vector3i::zero(), resident_lod0)
            .unwrap());
        assert!(streaming
            .prepare_voxel_edit(8, Vector3i::zero(), ChannelId::Type.index())
            .unwrap()
            .is_none());
        streaming.with_lod_map(0, |map| {
            let resident = map.get_block(Vector3i::zero()).unwrap();
            assert_eq!(block_buffer_identity(resident), resident_identity);
            assert_eq!(resident.viewers.get(), 4);
            assert!(resident.is_modified());
            assert!(resident.is_edited());
            assert!(resident.needs_lodding());
            assert_eq!(
                resident
                    .voxels()
                    .get_voxel(0, 0, 0, ChannelId::Type.index()),
                7
            );
        });
        assert_eq!(
            streaming.key_revision(Vector3i::zero(), 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        for lod_index in 1..3 {
            assert_eq!(
                streaming.key_revision(Vector3i::zero(), lod_index),
                Some(VoxelDataKeyRevision::Tombstone(0))
            );
        }
    }

    #[test]
    fn prepared_edit_late_lod_conflict_returns_every_replacement_owner() {
        let shared = make_atomic_edit_shared_data(3);
        let positions = [Vector3i::zero(); 3];
        let mut live_identities = Vec::new();
        for (lod_index, position) in positions.into_iter().enumerate() {
            assert!(shared
                .try_set_block(
                    position,
                    make_tagged_lod_block(10 + lod_index as u64, lod_index as u8)
                )
                .unwrap());
            live_identities.push(shared.with_lod_map(lod_index, |map| {
                block_buffer_identity(map.get_block(position).unwrap())
            }));
        }
        let mut prepared = shared
            .prepare_voxel_edit(99, Vector3i::zero(), ChannelId::Type.index())
            .unwrap()
            .unwrap();
        let replacement_identities = (0..3)
            .map(|lod_index| {
                block_buffer_identity(
                    prepared
                        .transaction_mut()
                        .owned_block(location(Vector3i::zero(), lod_index as u8))
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();

        let mut conflict = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::SetViewersExact {
                location: location(Vector3i::zero(), 2),
                final_viewers: 77,
            }])
            .unwrap();
        assert!(conflict.commit().unwrap().removed_blocks().is_empty());
        assert_eq!(
            prepared.transaction_mut().commit().unwrap_err(),
            SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: Vector3i::zero(),
                lod_index: 2,
                expected_revision: VoxelDataKeyRevision::Present(1),
                actual_revision: VoxelDataKeyRevision::Present(2),
            }
        );

        for lod_index in 0..3 {
            let location = location(Vector3i::zero(), lod_index as u8);
            assert_eq!(
                block_buffer_identity(prepared.transaction_mut().owned_block(location).unwrap()),
                replacement_identities[lod_index]
            );
            shared.with_lod_map(lod_index, |map| {
                let live = map.get_block(Vector3i::zero()).unwrap();
                assert_eq!(block_buffer_identity(live), live_identities[lod_index]);
                assert_eq!(live.viewers.get(), if lod_index == 2 { 77 } else { 0 });
                assert_eq!(
                    live.voxels().get_voxel(0, 0, 0, ChannelId::Type.index()),
                    10 + lod_index as u64
                );
            });
        }
        assert_eq!(
            shared.key_revision(Vector3i::zero(), 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(
            shared.key_revision(Vector3i::zero(), 1),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(
            shared.key_revision(Vector3i::zero(), 2),
            Some(VoxelDataKeyRevision::Present(2))
        );
        assert!(prepared.transaction.removed_blocks.is_empty());
        let operations = prepared.into_transaction().into_operations().unwrap();
        assert_eq!(operations.len(), 3);
        for (lod_index, expected_identity) in replacement_identities.into_iter().enumerate() {
            let actual = operations.iter().find_map(|operation| match operation {
                SharedVoxelDataTransactionOperation::Replace {
                    location: operation_location,
                    block,
                } if *operation_location == location(Vector3i::zero(), lod_index as u8) => {
                    assert_eq!(block.viewers.get(), 0);
                    assert!(block.is_modified());
                    assert_eq!(block.is_edited(), lod_index == 0);
                    assert!(!block.needs_lodding());
                    Some(block_buffer_identity(block))
                }
                _ => None,
            });
            assert_eq!(actual, Some(expected_identity));
        }
    }

    #[test]
    fn prepared_edit_snapshot_shape_race_reports_exact_concurrent_revision() {
        let shared = make_atomic_edit_shared_data(2);
        for lod_index in 0..2 {
            assert!(shared
                .try_set_block(
                    Vector3i::zero(),
                    make_tagged_lod_block(50 + lod_index as u64, lod_index as u8),
                )
                .unwrap());
        }
        let weak = Arc::downgrade(&shared);
        shared.set_test_edit_phase_hook(Arc::new(move |phase| {
            if phase != SharedVoxelDataEditPhase::PreparedVoxelEditDraftedBeforeTransactionPrepare {
                return;
            }
            let shared = weak.upgrade().unwrap();
            let mut conflict = shared
                .prepare_transaction(vec![SharedVoxelDataTransactionOperation::Remove {
                    location: location(Vector3i::zero(), 0),
                }])
                .unwrap();
            conflict.commit().unwrap();
        }));

        assert!(matches!(
            shared.prepare_voxel_edit(99, Vector3i::zero(), ChannelId::Type.index()),
            Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                position,
                lod_index: 0,
                expected_revision: VoxelDataKeyRevision::Present(1),
                actual_revision: VoxelDataKeyRevision::Tombstone(2),
            }) if position == Vector3i::zero()
        ));
        assert!(!shared.with_lod_map(0, |map| map.has_block(Vector3i::zero())));
        assert_eq!(
            shared.key_revision(Vector3i::zero(), 0),
            Some(VoxelDataKeyRevision::Tombstone(2))
        );
        assert_eq!(
            shared.key_revision(Vector3i::zero(), 1),
            Some(VoxelDataKeyRevision::Present(1))
        );
    }

    #[test]
    fn prepared_edit_two_writers_use_canonical_order_without_deadlock() {
        let shared = make_atomic_edit_shared_data(3);
        let original_identities: [usize; 3] = std::array::from_fn(|lod_index| {
            let mut block = make_tagged_lod_block(20 + lod_index as u64, lod_index as u8);
            block.viewers.set_exact((lod_index + 1) as u32);
            let identity = block_buffer_identity(&block) as usize;
            assert!(shared.try_set_block(Vector3i::zero(), block).unwrap());
            identity
        });
        let mut prepared_a = shared
            .prepare_voxel_edit(31, Vector3i::zero(), ChannelId::Type.index())
            .unwrap()
            .unwrap();
        let mut prepared_b = shared
            .prepare_voxel_edit(32, Vector3i::zero(), ChannelId::Type.index())
            .unwrap()
            .unwrap();
        let draft_a: [usize; 3] = std::array::from_fn(|lod_index| {
            block_buffer_identity(
                prepared_a
                    .transaction_mut()
                    .owned_block(location(Vector3i::zero(), lod_index as u8))
                    .unwrap(),
            ) as usize
        });
        let draft_b: [usize; 3] = std::array::from_fn(|lod_index| {
            block_buffer_identity(
                prepared_b
                    .transaction_mut()
                    .owned_block(location(Vector3i::zero(), lod_index as u8))
                    .unwrap(),
            ) as usize
        });
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let (sent, received) = std::sync::mpsc::channel();
        let mut workers = Vec::new();
        for (value, draft_identities, mut prepared, barrier) in [
            (31, draft_a, prepared_a, Arc::clone(&barrier)),
            (32, draft_b, prepared_b, Arc::clone(&barrier)),
        ] {
            let sent = sent.clone();
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                let result = prepared.transaction_mut().commit();
                sent.send((value, draft_identities, result, prepared))
                    .unwrap();
            }));
        }
        barrier.wait();
        let mut results = vec![
            received.recv_timeout(Duration::from_secs(2)).unwrap(),
            received.recv_timeout(Duration::from_secs(2)).unwrap(),
        ];
        for worker in workers {
            worker.join().unwrap();
        }
        let winner_index = results
            .iter()
            .position(|(_, _, result, _)| result.is_ok())
            .expect("one prepared writer must win");
        let winner = results.remove(winner_index);
        let loser = results.pop().expect("the other prepared writer must lose");
        let (winner_value, winner_drafts, winner_result, _) = winner;
        let (_, loser_drafts, loser_result, loser_prepared) = loser;
        assert_eq!(
            loser_result.unwrap_err(),
            SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: Vector3i::zero(),
                lod_index: 0,
                expected_revision: VoxelDataKeyRevision::Present(1),
                actual_revision: VoxelDataKeyRevision::Present(2),
            }
        );
        let winner_outcome = winner_result.unwrap();
        assert_eq!(winner_outcome.removed_blocks().len(), 3);
        for lod_index in 0..3 {
            let location = location(Vector3i::zero(), lod_index as u8);
            let retired = winner_outcome
                .removed_blocks()
                .iter()
                .find(|removed| removed.location() == location)
                .unwrap();
            assert_eq!(
                block_buffer_identity(retired.block()) as usize,
                original_identities[lod_index]
            );
            shared.with_lod_map(lod_index, |map| {
                let resident = map.get_block(Vector3i::zero()).unwrap();
                assert_eq!(
                    block_buffer_identity(resident) as usize,
                    winner_drafts[lod_index]
                );
                assert_eq!(
                    resident
                        .voxels()
                        .get_voxel(0, 0, 0, ChannelId::Type.index()),
                    winner_value
                );
            });
            assert_eq!(
                shared.key_revision(Vector3i::zero(), lod_index),
                Some(VoxelDataKeyRevision::Present(2))
            );
            assert_eq!(
                block_buffer_identity(loser_prepared.transaction.owned_block(location).unwrap())
                    as usize,
                loser_drafts[lod_index]
            );
        }
        assert!(loser_prepared.transaction.removed_blocks.is_empty());
    }

    #[test]
    fn prepared_edit_observer_sees_only_complete_old_or_complete_new_lods() {
        let shared = make_atomic_edit_shared_data(3);
        for lod_index in 0..3 {
            assert!(shared
                .try_set_block(
                    Vector3i::zero(),
                    make_tagged_lod_block(40 + lod_index as u64, lod_index as u8),
                )
                .unwrap());
        }
        let (request_sent, request_received) = std::sync::mpsc::channel();
        let (sample_sent, sample_received) = std::sync::mpsc::channel();
        let observer_data = Arc::clone(&shared);
        let observer = std::thread::spawn(move || {
            for _ in 0..2 {
                request_received.recv().unwrap();
                sample_sent
                    .send(sample_atomic_edit_lods(&observer_data))
                    .unwrap();
            }
        });
        request_sent.send(()).unwrap();
        let old = sample_received
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert_eq!(
            old,
            [
                (40, VoxelDataKeyRevision::Present(1)),
                (41, VoxelDataKeyRevision::Present(1)),
                (42, VoxelDataKeyRevision::Present(1)),
            ]
        );

        let mut prepared = shared
            .prepare_voxel_edit(88, Vector3i::zero(), ChannelId::Type.index())
            .unwrap()
            .unwrap();
        let pause = Arc::new((Mutex::new(false), Condvar::new()));
        let pause_hook = Arc::clone(&pause);
        let (phase_sent, phase_received) = std::sync::mpsc::channel();
        shared.set_test_edit_phase_hook(Arc::new(move |phase| {
            if phase != SharedVoxelDataEditPhase::PreparedTransactionValidatedBeforeFirstLiveWrite {
                return;
            }
            phase_sent.send(()).unwrap();
            let (lock, cvar) = &*pause_hook;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = cvar.wait(released).unwrap();
            }
        }));
        let commit_worker = std::thread::spawn(move || prepared.transaction_mut().commit());
        phase_received.recv_timeout(Duration::from_secs(1)).unwrap();
        request_sent.send(()).unwrap();
        assert!(
            sample_received
                .recv_timeout(Duration::from_millis(50))
                .is_err(),
            "the atomic observer must remain behind the multi-LOD publication fence"
        );
        for lod_index in 0..3 {
            assert!(
                shared.try_lod_map_read(lod_index).is_none(),
                "no observer may sample LOD {lod_index} inside the publication fence"
            );
            let extent = (shared.block_size() as i32) << lod_index;
            assert!(shared
                .try_read_region(
                    lod_index,
                    Box3i::new(Vector3i::zero(), Vector3i::splat(extent)),
                )
                .is_none());
        }
        let (lock, cvar) = &*pause;
        *lock.lock().unwrap() = true;
        cvar.notify_all();
        let outcome = commit_worker.join().unwrap().unwrap();
        assert_eq!(outcome.removed_blocks().len(), 3);

        let new = sample_received
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        observer.join().unwrap();
        assert_eq!(
            new,
            [
                (88, VoxelDataKeyRevision::Present(2)),
                (88, VoxelDataKeyRevision::Present(2)),
                (88, VoxelDataKeyRevision::Present(2)),
            ]
        );
        assert_ne!(old, new);
    }

    #[test]
    fn transaction_preview_settings_conflict_preserves_insert_owner_for_fresh_preview_retry() {
        let shared = make_shared_data_with_lods(1);
        let observed = location(Vector3i::zero(), 0);
        let inserted = location(Vector3i::new(1, 0, 0), 0);
        assert!(shared
            .try_set_block(observed.position, make_tagged_lod_block(91, 0))
            .unwrap());
        let preview = shared.begin_transaction_preview();
        assert_eq!(preview.settings().revision, 0);
        assert_eq!(preview.block_size(), shared.block_size());
        let snapshots = [
            preview.block_snapshot(observed).unwrap(),
            preview.block_snapshot(inserted).unwrap(),
        ];
        let block = make_tagged_lod_block(92, 0);
        let identity = block_buffer_identity(&block);
        let operations = vec![SharedVoxelDataTransactionOperation::Insert {
            location: inserted,
            block,
            final_viewers: 7,
        }];

        let generator: SharedVoxelGenerator = Arc::new(RecordingGenerator::default());
        shared.set_generator(Some(generator)).unwrap();

        let error = preview
            .prepare_transaction(operations, &snapshots)
            .unwrap_err();
        assert_eq!(
            error.error(),
            &SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                expected_revision: 0,
                actual_revision: 1,
            }
        );
        let (_, operations) = error.into_parts();
        assert_insert_operation_identity(&operations, inserted, identity);
        assert!(!shared.with_lod_map(0, |map| map.has_block(inserted.position)));

        let retry_preview = shared.begin_transaction_preview();
        assert_eq!(retry_preview.settings().revision, 1);
        let retry_snapshots = [
            retry_preview.block_snapshot(observed).unwrap(),
            retry_preview.block_snapshot(inserted).unwrap(),
        ];
        let mut retry = retry_preview
            .prepare_transaction(operations, &retry_snapshots)
            .unwrap();
        assert_eq!(
            block_buffer_identity(retry.inserted_block(inserted).unwrap()),
            identity
        );
        retry.commit().unwrap();
        assert_eq!(
            shared.with_lod_map(0, |map| {
                block_buffer_identity(map.get_block(inserted.position).unwrap())
            }),
            identity
        );
    }

    #[test]
    fn observation_only_transaction_preview_rejects_settings_change_before_prepare() {
        let shared = make_shared_data_with_lods(1);
        let observed = location(Vector3i::new(2, 0, 0), 0);
        assert!(shared
            .try_set_block(observed.position, make_tagged_lod_block(93, 0))
            .unwrap());
        let revision_before = shared.key_revision(observed.position, 0).unwrap();
        let preview = shared.begin_transaction_preview();
        let snapshots = [preview.block_snapshot(observed).unwrap()];

        let generator: SharedVoxelGenerator = Arc::new(RecordingGenerator::default());
        shared.set_generator(Some(generator)).unwrap();

        let error = preview
            .prepare_transaction(Vec::new(), &snapshots)
            .unwrap_err();
        assert_eq!(
            error.error(),
            &SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                expected_revision: 0,
                actual_revision: 1,
            }
        );
        assert!(error.operations().is_empty());
        assert_eq!(
            shared.key_revision(observed.position, 0),
            Some(revision_before)
        );
        assert_eq!(
            shared.with_lod_map(0, |map| map
                .get_block(observed.position)
                .unwrap()
                .viewers
                .get()),
            0
        );
    }

    #[test]
    fn transaction_preview_rejects_snapshots_from_another_preview_or_data_lineage() {
        for cross_data in [false, true] {
            let target = make_shared_data_with_lods(1);
            let source = if cross_data {
                make_shared_data_with_lods(1)
            } else {
                Arc::clone(&target)
            };
            let inserted = location(Vector3i::new(3, 0, 0), 0);
            let source_preview = source.begin_transaction_preview();
            let foreign_snapshots = [source_preview.block_snapshot(inserted).unwrap()];
            let target_preview = target.begin_transaction_preview();
            let block = make_tagged_lod_block(94 + u64::from(cross_data), 0);
            let identity = block_buffer_identity(&block);

            let error = target_preview
                .prepare_transaction(
                    vec![SharedVoxelDataTransactionOperation::Insert {
                        location: inserted,
                        block,
                        final_viewers: 1,
                    }],
                    &foreign_snapshots,
                )
                .unwrap_err();

            assert_eq!(
                error.error(),
                &SharedVoxelDataMutationError::PreparedTransactionPreviewSetMismatch
            );
            let (_, operations) = error.into_parts();
            assert_insert_operation_identity(&operations, inserted, identity);
            assert!(!target.with_lod_map(0, |map| map.has_block(inserted.position)));
        }
    }

    #[test]
    fn transaction_preview_rechecks_settings_after_ordinary_preparation() {
        let shared = make_shared_data_with_lods(1);
        let inserted = location(Vector3i::new(4, 0, 0), 0);
        let mut preview = shared.begin_transaction_preview();
        let snapshots = [preview.block_snapshot(inserted).unwrap()];
        let block = make_tagged_lod_block(96, 0);
        let identity = block_buffer_identity(&block);
        let generator: SharedVoxelGenerator = Arc::new(RecordingGenerator::default());
        let shared_for_hook = Arc::clone(&shared);
        preview.set_test_after_prepare_hook(Arc::new(move || {
            shared_for_hook
                .set_generator(Some(Arc::clone(&generator)))
                .unwrap();
        }));

        let error = preview
            .prepare_transaction(
                vec![SharedVoxelDataTransactionOperation::Insert {
                    location: inserted,
                    block,
                    final_viewers: 1,
                }],
                &snapshots,
            )
            .unwrap_err();

        assert_eq!(
            error.error(),
            &SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                expected_revision: 0,
                actual_revision: 1,
            }
        );
        let (_, operations) = error.into_parts();
        assert_insert_operation_identity(&operations, inserted, identity);
        assert!(!shared.with_lod_map(0, |map| map.has_block(inserted.position)));

        let retry_preview = shared.begin_transaction_preview();
        let retry_snapshots = [retry_preview.block_snapshot(inserted).unwrap()];
        let mut retry = retry_preview
            .prepare_transaction(operations, &retry_snapshots)
            .unwrap();
        assert_eq!(
            block_buffer_identity(retry.inserted_block(inserted).unwrap()),
            identity
        );
        retry.commit().unwrap();
        assert_eq!(
            shared.with_lod_map(0, |map| {
                block_buffer_identity(map.get_block(inserted.position).unwrap())
            }),
            identity
        );
    }

    #[test]
    fn snapshot_transaction_preview_and_observation_reservations_are_typed_and_preserve_insert_owner(
    ) {
        for reservation in [
            SharedVoxelDataTransactionReservation::PreviewSnapshotStorage,
            SharedVoxelDataTransactionReservation::ObservationStorage,
        ] {
            let shared = make_shared_data_with_lods(1);
            let observed = location(Vector3i::zero(), 0);
            let inserted = location(Vector3i::new(1, 0, 0), 0);
            assert!(shared
                .try_set_block(observed.position, make_tagged_lod_block(101, 0))
                .unwrap());
            let preview = shared.begin_transaction_preview();
            let snapshots = [
                preview.block_snapshot(observed).unwrap(),
                preview.block_snapshot(inserted).unwrap(),
            ];
            let block = make_tagged_lod_block(102, 0);
            let identity = block_buffer_identity(&block);
            shared.set_test_transaction_reservation_failpoint(Some(reservation));

            let error = preview
                .prepare_transaction(
                    vec![SharedVoxelDataTransactionOperation::Insert {
                        location: inserted,
                        block,
                        final_viewers: 7,
                    }],
                    &snapshots,
                )
                .unwrap_err();

            assert_eq!(
                error.error(),
                &SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation,
                }
            );
            let (_, operations) = error.into_parts();
            assert_insert_operation_identity(&operations, inserted, identity);
            assert!(!shared.with_lod_map(0, |map| map.has_block(inserted.position)));
            assert_eq!(
                shared.key_revision(inserted.position, 0),
                Some(VoxelDataKeyRevision::Tombstone(0))
            );
        }
    }

    #[test]
    fn snapshot_transaction_requires_the_exact_mutating_and_observed_key_set() {
        let cases = [
            // A duplicate observed key is invalid even though the mutating key
            // is present.
            (vec![0, 1, 0], vec![1]),
            // The observed keys interleave one mutating key while another
            // mutating key is missing.
            (vec![0, 1, 2], vec![1, 3]),
            // Equal cardinality must not be mistaken for equal key sets.
            (vec![0, 2], vec![1, 3]),
        ];

        for (snapshot_xs, operation_xs) in cases {
            let shared = make_shared_data_with_lods(1);
            let preview = shared.begin_transaction_preview();
            let mut identities = Vec::new();
            let mut operations = Vec::new();
            for x in operation_xs {
                let location = location(Vector3i::new(x, 0, 0), 0);
                let block = make_tagged_lod_block(110 + x as u64, 0);
                identities.push((location, block_buffer_identity(&block)));
                operations.push(SharedVoxelDataTransactionOperation::Insert {
                    location,
                    block,
                    final_viewers: 1,
                });
            }
            let snapshots = snapshot_xs
                .into_iter()
                .map(|x| {
                    preview
                        .block_snapshot(location(Vector3i::new(x, 0, 0), 0))
                        .unwrap()
                })
                .collect::<Vec<_>>();

            let error = preview
                .prepare_transaction(operations, &snapshots)
                .unwrap_err();

            assert_eq!(
                error.error(),
                &SharedVoxelDataMutationError::PreparedTransactionPreviewSetMismatch
            );
            let (_, operations) = error.into_parts();
            assert_eq!(operations.len(), identities.len());
            for (location, identity) in identities {
                assert_insert_operation_identity(&operations, location, identity);
                assert!(!shared.with_lod_map(0, |map| map.has_block(location.position)));
                assert_eq!(
                    shared.key_revision(location.position, 0),
                    Some(VoxelDataKeyRevision::Tombstone(0))
                );
            }
        }
    }

    #[test]
    fn snapshot_to_prepare_present_and_tombstone_mutations_do_not_write_and_retry_exact_owner() {
        for initially_present in [true, false] {
            let shared = make_shared_data_with_lods(1);
            let observed = location(Vector3i::zero(), 0);
            let inserted = location(Vector3i::new(10, 0, 0), 0);
            if initially_present {
                assert!(shared
                    .try_set_block(observed.position, make_tagged_lod_block(121, 0))
                    .unwrap());
            }
            let preview = shared.begin_transaction_preview();
            let snapshots = [
                preview.block_snapshot(observed).unwrap(),
                preview.block_snapshot(inserted).unwrap(),
            ];
            let block = make_tagged_lod_block(122, 0);
            let identity = block_buffer_identity(&block);
            let operations = vec![SharedVoxelDataTransactionOperation::Insert {
                location: inserted,
                block,
                final_viewers: 4,
            }];

            if initially_present {
                assert!(shared
                    .try_set_voxel_checked(123, Vector3i::zero(), ChannelId::Type.index())
                    .unwrap());
            } else {
                assert!(shared
                    .try_set_block(observed.position, make_tagged_lod_block(124, 0))
                    .unwrap());
            }

            let error = preview
                .prepare_transaction(operations, &snapshots)
                .unwrap_err();
            assert!(matches!(
                error.error(),
                SharedVoxelDataMutationError::ConcurrentDataMutation {
                    position,
                    lod_index: 0,
                    ..
                } if *position == observed.position
            ));
            let (_, operations) = error.into_parts();
            assert_insert_operation_identity(&operations, inserted, identity);
            assert!(!shared.with_lod_map(0, |map| map.has_block(inserted.position)));
            assert_eq!(
                shared.key_revision(inserted.position, 0),
                Some(VoxelDataKeyRevision::Tombstone(0))
            );

            let retry_preview = shared.begin_transaction_preview();
            let retry_snapshots = [
                retry_preview.block_snapshot(observed).unwrap(),
                retry_preview.block_snapshot(inserted).unwrap(),
            ];
            let mut retry = retry_preview
                .prepare_transaction(operations, &retry_snapshots)
                .unwrap();
            assert_eq!(
                block_buffer_identity(retry.inserted_block(inserted).unwrap()),
                identity
            );
            retry.commit().unwrap();
            assert_eq!(
                shared.with_lod_map(0, |map| {
                    block_buffer_identity(map.get_block(inserted.position).unwrap())
                }),
                identity
            );
        }
    }

    #[test]
    fn prepare_to_commit_present_and_tombstone_mutations_do_not_write_and_retry_exact_owner() {
        for initially_present in [true, false] {
            let shared = make_shared_data_with_lods(1);
            let observed = location(Vector3i::zero(), 0);
            let inserted = location(Vector3i::new(10, 0, 0), 0);
            if initially_present {
                assert!(shared
                    .try_set_block(observed.position, make_tagged_lod_block(131, 0))
                    .unwrap());
            }
            let preview = shared.begin_transaction_preview();
            let snapshots = [
                preview.block_snapshot(observed).unwrap(),
                preview.block_snapshot(inserted).unwrap(),
            ];
            let block = make_tagged_lod_block(132, 0);
            let identity = block_buffer_identity(&block);
            let mut prepared = preview
                .prepare_transaction(
                    vec![SharedVoxelDataTransactionOperation::Insert {
                        location: inserted,
                        block,
                        final_viewers: 5,
                    }],
                    &snapshots,
                )
                .unwrap();

            if initially_present {
                assert!(shared
                    .try_set_voxel_checked(133, Vector3i::zero(), ChannelId::Type.index())
                    .unwrap());
            } else {
                assert!(shared
                    .try_set_block(observed.position, make_tagged_lod_block(134, 0))
                    .unwrap());
            }

            assert!(matches!(
                prepared.commit(),
                Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                    position,
                    lod_index: 0,
                    ..
                }) if position == observed.position
            ));
            assert_eq!(
                block_buffer_identity(prepared.inserted_block(inserted).unwrap()),
                identity
            );
            assert!(!shared.with_lod_map(0, |map| map.has_block(inserted.position)));
            assert_eq!(
                shared.key_revision(inserted.position, 0),
                Some(VoxelDataKeyRevision::Tombstone(0))
            );

            let operations = prepared.into_operations().unwrap();
            assert_insert_operation_identity(&operations, inserted, identity);
            let retry_preview = shared.begin_transaction_preview();
            let retry_snapshots = [
                retry_preview.block_snapshot(observed).unwrap(),
                retry_preview.block_snapshot(inserted).unwrap(),
            ];
            let mut retry = retry_preview
                .prepare_transaction(operations, &retry_snapshots)
                .unwrap();
            retry.commit().unwrap();
            assert_eq!(
                shared.with_lod_map(0, |map| {
                    block_buffer_identity(map.get_block(inserted.position).unwrap())
                }),
                identity
            );
        }
    }

    #[test]
    fn observation_only_publication_fence_blocks_region_and_map_without_advancing_revision() {
        let shared = make_shared_data_with_lods(1);
        let observed = location(Vector3i::new(2, 0, 0), 0);
        assert!(shared
            .try_set_block(observed.position, make_tagged_lod_block(141, 0))
            .unwrap());
        let revision_before = shared.key_revision(observed.position, 0).unwrap();
        let structural_before = shared.lod_structural_revision(0).unwrap();
        let preview = shared.begin_transaction_preview();
        let snapshots = [preview.block_snapshot(observed).unwrap()];
        let mut prepared = preview.prepare_transaction(Vec::new(), &snapshots).unwrap();

        let fence = prepared.commit_holding_publication_fence().unwrap();

        let guarded_state = fence.map_guards.as_ref().unwrap()[0].as_ref().unwrap();
        assert!(guarded_state.map.has_block(observed.position));
        let voxel_box = Box3i::new(observed.position * 16, Vector3i::splat(16));
        assert!(shared.try_read_region(0, voxel_box).is_none());
        assert!(shared.try_lod_map_read(0).is_none());
        assert!(shared.try_lod_map_write(0).is_none());
        assert!(matches!(
            shared.mutation_gate.try_lock(),
            Err(std::sync::TryLockError::WouldBlock)
        ));

        let outcome = fence.finish();

        assert!(outcome.removed_blocks().is_empty());
        assert_eq!(
            shared.key_revision(observed.position, 0),
            Some(revision_before)
        );
        assert_eq!(shared.lod_structural_revision(0), Some(structural_before));
        assert!(shared.try_read_region(0, voxel_box).is_some());
        assert!(shared.try_lod_map_read(0).is_some());
        assert!(shared.try_lod_map_write(0).is_some());
        assert!(shared.mutation_gate.try_lock().is_ok());
    }

    #[test]
    fn observation_only_multi_lod_live_spatial_failure_restores_every_batch_for_retry() {
        let shared = make_shared_data_with_lods(2);
        let lod0 = location(Vector3i::new(1, 0, 0), 0);
        let lod1 = location(Vector3i::new(2, 0, 0), 1);
        assert!(shared
            .try_set_block(lod0.position, make_tagged_lod_block(151, 0))
            .unwrap());
        assert!(shared
            .try_set_block(lod1.position, make_tagged_lod_block(152, 1))
            .unwrap());
        let revisions_before = [
            shared.key_revision(lod0.position, 0).unwrap(),
            shared.key_revision(lod1.position, 1).unwrap(),
        ];
        let preview = shared.begin_transaction_preview();
        let snapshots = [
            preview.block_snapshot(lod1).unwrap(),
            preview.block_snapshot(lod0).unwrap(),
        ];
        let mut prepared = preview.prepare_transaction(Vec::new(), &snapshots).unwrap();
        shared.set_test_transaction_live_spatial_registry_fail_lod(Some(1));

        let error = match prepared.commit_holding_publication_fence() {
            Ok(_) => panic!("injected LOD 1 live-spatial failure must reject the commit"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                reservation: SharedVoxelDataTransactionReservation::LiveSpatialRegistry,
            }
        );
        assert_eq!(shared.locked_region_count(0), 0);
        assert_eq!(shared.locked_region_count(1), 0);
        assert!(shared.try_lod_map_write(0).is_some());
        assert!(shared.try_lod_map_write(1).is_some());
        assert_eq!(
            shared.key_revision(lod0.position, 0),
            Some(revisions_before[0])
        );
        assert_eq!(
            shared.key_revision(lod1.position, 1),
            Some(revisions_before[1])
        );

        shared.set_test_transaction_live_spatial_registry_fail_lod(None);
        let fence = prepared.commit_holding_publication_fence().unwrap();
        assert_eq!(shared.locked_region_count(0), 1);
        assert_eq!(shared.locked_region_count(1), 1);
        assert!(shared.try_lod_map_read(0).is_none());
        assert!(shared.try_lod_map_read(1).is_none());
        fence.finish();

        assert_eq!(shared.locked_region_count(0), 0);
        assert_eq!(shared.locked_region_count(1), 0);
        assert_eq!(
            shared.key_revision(lod0.position, 0),
            Some(revisions_before[0])
        );
        assert_eq!(
            shared.key_revision(lod1.position, 1),
            Some(revisions_before[1])
        );
    }

    #[test]
    fn observation_revalidation_rejects_every_same_revision_raw_metadata_change() {
        #[derive(Clone, Copy)]
        enum RawMutation {
            Viewers,
            Modified,
            Edited,
            HasVoxels,
        }

        for mutation in [
            RawMutation::Viewers,
            RawMutation::Modified,
            RawMutation::Edited,
            RawMutation::HasVoxels,
        ] {
            let shared = make_shared_data_with_lods(1);
            let observed = location(Vector3i::zero(), 0);
            let inserted = location(Vector3i::new(10, 0, 0), 0);
            assert!(shared
                .try_set_block(observed.position, make_tagged_lod_block(161, 0))
                .unwrap());
            let revision_before = shared.key_revision(observed.position, 0).unwrap();
            let preview = shared.begin_transaction_preview();
            let snapshots = [
                preview.block_snapshot(observed).unwrap(),
                preview.block_snapshot(inserted).unwrap(),
            ];
            let block = make_tagged_lod_block(162, 0);
            let identity = block_buffer_identity(&block);
            let mut prepared = preview
                .prepare_transaction(
                    vec![SharedVoxelDataTransactionOperation::Insert {
                        location: inserted,
                        block,
                        final_viewers: 6,
                    }],
                    &snapshots,
                )
                .unwrap();

            shared.with_lod_map_mut(0, |map| {
                let block = map.get_block_mut(observed.position).unwrap();
                match mutation {
                    RawMutation::Viewers => block.viewers.set_exact(1),
                    RawMutation::Modified => block.set_modified(true),
                    RawMutation::Edited => block.set_edited(true),
                    RawMutation::HasVoxels => block.clear_voxels(),
                }
            });
            assert_eq!(
                shared.key_revision(observed.position, 0),
                Some(revision_before),
                "the raw test mutation must exercise metadata revalidation, not revision revalidation"
            );

            let (actual_viewers, actual_modified, actual_edited, actual_has_voxels) = match mutation
            {
                RawMutation::Viewers => (1, false, false, true),
                RawMutation::Modified => (0, true, false, true),
                RawMutation::Edited => (0, false, true, true),
                RawMutation::HasVoxels => (0, false, false, false),
            };
            assert_eq!(
                prepared.commit().unwrap_err(),
                SharedVoxelDataMutationError::PreparedTransactionConcurrentBlockState {
                    location: observed,
                    expected_viewers: 0,
                    actual_viewers,
                    expected_modified: false,
                    actual_modified,
                    expected_edited: false,
                    actual_edited,
                    expected_has_voxels: true,
                    actual_has_voxels,
                }
            );
            assert_eq!(
                block_buffer_identity(prepared.inserted_block(inserted).unwrap()),
                identity
            );
            assert!(!shared.with_lod_map(0, |map| map.has_block(inserted.position)));

            let operations = prepared.into_operations().unwrap();
            assert_insert_operation_identity(&operations, inserted, identity);
            let retry_preview = shared.begin_transaction_preview();
            let retry_snapshots = [
                retry_preview.block_snapshot(observed).unwrap(),
                retry_preview.block_snapshot(inserted).unwrap(),
            ];
            let mut retry = retry_preview
                .prepare_transaction(operations, &retry_snapshots)
                .unwrap();
            retry.commit().unwrap();
            assert_eq!(
                shared.with_lod_map(0, |map| {
                    block_buffer_identity(map.get_block(inserted.position).unwrap())
                }),
                identity
            );
        }
    }

    #[test]
    fn preview_revalidation_rejects_every_same_revision_raw_metadata_change() {
        for mutation_index in 0..4 {
            let shared = make_shared_data_with_lods(1);
            let observed = location(Vector3i::zero(), 0);
            let inserted = location(Vector3i::new(10, 0, 0), 0);
            assert!(shared
                .try_set_block(observed.position, make_tagged_lod_block(166, 0))
                .unwrap());
            let revision_before = shared.key_revision(observed.position, 0).unwrap();
            let preview = shared.begin_transaction_preview();
            let snapshots = [
                preview.block_snapshot(observed).unwrap(),
                preview.block_snapshot(inserted).unwrap(),
            ];
            let block = make_tagged_lod_block(167, 0);
            let identity = block_buffer_identity(&block);
            let operations = vec![SharedVoxelDataTransactionOperation::Insert {
                location: inserted,
                block,
                final_viewers: 8,
            }];

            shared.with_lod_map_mut(0, |map| {
                let block = map.get_block_mut(observed.position).unwrap();
                match mutation_index {
                    0 => block.viewers.set_exact(1),
                    1 => block.set_modified(true),
                    2 => block.set_edited(true),
                    3 => block.clear_voxels(),
                    _ => unreachable!(),
                }
            });
            assert_eq!(
                shared.key_revision(observed.position, 0),
                Some(revision_before)
            );

            let error = preview
                .prepare_transaction(operations, &snapshots)
                .unwrap_err();
            assert_eq!(
                error.error(),
                &SharedVoxelDataMutationError::ConcurrentDataMutation {
                    position: observed.position,
                    lod_index: 0,
                    expected_revision: revision_before,
                    actual_revision: revision_before,
                }
            );
            let (_, operations) = error.into_parts();
            assert_insert_operation_identity(&operations, inserted, identity);
            assert!(!shared.with_lod_map(0, |map| map.has_block(inserted.position)));

            let retry_preview = shared.begin_transaction_preview();
            let retry_snapshots = [
                retry_preview.block_snapshot(observed).unwrap(),
                retry_preview.block_snapshot(inserted).unwrap(),
            ];
            let mut retry = retry_preview
                .prepare_transaction(operations, &retry_snapshots)
                .unwrap();
            retry.commit().unwrap();
            assert_eq!(
                shared.with_lod_map(0, |map| {
                    block_buffer_identity(map.get_block(inserted.position).unwrap())
                }),
                identity
            );
        }
    }

    #[test]
    fn observed_only_finish_and_abandon_release_all_guards_before_retiring_bounds() {
        for abandon in [false, true] {
            let shared = make_shared_data_with_lods(2);
            let lod0 = location(Vector3i::zero(), 0);
            let lod1 = location(Vector3i::new(1, 0, 0), 1);
            assert!(shared
                .try_set_block(lod0.position, make_tagged_lod_block(171, 0))
                .unwrap());
            assert!(shared
                .try_set_block(lod1.position, make_tagged_lod_block(172, 1))
                .unwrap());
            let revisions_before = [
                shared.key_revision(lod0.position, 0).unwrap(),
                shared.key_revision(lod1.position, 1).unwrap(),
            ];
            let preview = shared.begin_transaction_preview();
            let snapshots = [
                preview.block_snapshot(lod0).unwrap(),
                preview.block_snapshot(lod1).unwrap(),
            ];
            let mut prepared = preview.prepare_transaction(Vec::new(), &snapshots).unwrap();
            let hook_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let hook_saw_release = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let calls = hook_calls.clone();
            let saw_release = hook_saw_release.clone();
            let weak = Arc::downgrade(&shared);
            prepared.set_test_spatial_batch_drop_hook(Arc::new(move || {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let shared = weak.upgrade().unwrap();
                let maps_available =
                    shared.try_lod_map_write(0).is_some() && shared.try_lod_map_write(1).is_some();
                let spatial_available = shared
                    .try_write_region(0, Box3i::new(lod0.position * 16, Vector3i::splat(16)))
                    .is_some()
                    && shared
                        .try_write_region(1, Box3i::new(lod1.position * 32, Vector3i::splat(32)))
                        .is_some();
                saw_release.store(
                    maps_available && spatial_available && shared.mutation_gate.try_lock().is_ok(),
                    std::sync::atomic::Ordering::SeqCst,
                );
            }));

            let fence = prepared.commit_holding_publication_fence().unwrap();
            assert!(shared.try_lod_map_read(0).is_none());
            assert!(shared.try_lod_map_read(1).is_none());
            if abandon {
                drop(fence);
            } else {
                fence.finish();
            }

            assert_eq!(hook_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
            assert!(hook_saw_release.load(std::sync::atomic::Ordering::SeqCst));
            assert_eq!(shared.locked_region_count(0), 0);
            assert_eq!(shared.locked_region_count(1), 0);
            assert_eq!(
                shared.key_revision(lod0.position, 0),
                Some(revisions_before[0])
            );
            assert_eq!(
                shared.key_revision(lod1.position, 1),
                Some(revisions_before[1])
            );
        }
    }

    #[test]
    fn publication_fence_keeps_region_exclusive_until_matching_state_marker() {
        let shared = make_shared_data_with_lods(1);
        let first = location(Vector3i::new(0, 0, 0), 0);
        let second = location(Vector3i::new(1, 0, 0), 0);
        assert!(shared
            .try_set_block(first.position, make_tagged_lod_block(111, 0))
            .unwrap());
        assert!(shared
            .try_set_block(second.position, make_tagged_lod_block(112, 0))
            .unwrap());
        let mut prepared = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: second,
                    final_viewers: 22,
                },
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: first,
                    final_viewers: 11,
                },
            ])
            .unwrap();

        let fence = prepared.commit_holding_publication_fence().unwrap();
        let guarded_state = fence.map_guards.as_ref().unwrap()[0].as_ref().unwrap();
        assert_eq!(
            (
                guarded_state
                    .map
                    .get_block(first.position)
                    .unwrap()
                    .viewers
                    .get(),
                guarded_state
                    .map
                    .get_block(second.position)
                    .unwrap()
                    .viewers
                    .get(),
            ),
            (11, 22)
        );
        let voxel_box = Box3i::new(Vector3i::zero(), Vector3i::new(32, 16, 16));
        assert!(shared.try_read_region(0, voxel_box).is_none());

        let matching_state_marker = std::sync::atomic::AtomicBool::new(false);
        matching_state_marker.store(true, std::sync::atomic::Ordering::SeqCst);
        fence.finish();

        let _reader = shared.read_region(0, voxel_box);
        assert!(matching_state_marker.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(
            shared.with_lod_map(0, |map| (
                map.get_block(first.position).unwrap().viewers.get(),
                map.get_block(second.position).unwrap().viewers.get(),
            )),
            (11, 22)
        );
    }

    #[test]
    fn publication_fence_finish_returns_exact_clean_and_dirty_owners_after_release() {
        let shared = make_shared_data_with_lods(2);
        let clean = location(Vector3i::new(0, 0, 0), 0);
        let dirty = location(Vector3i::new(1, 0, 0), 1);
        let clean_block = make_tagged_lod_block(121, 0);
        let clean_identity = block_buffer_identity(&clean_block);
        let mut dirty_block = make_tagged_lod_block(122, 1);
        dirty_block.set_modified(true);
        dirty_block.set_edited(true);
        dirty_block.set_needs_lodding(true);
        let dirty_identity = block_buffer_identity(&dirty_block);
        assert!(shared.try_set_block(clean.position, clean_block).unwrap());
        assert!(shared.try_set_block(dirty.position, dirty_block).unwrap());
        let mut prepared = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::Remove { location: dirty },
                SharedVoxelDataTransactionOperation::Remove { location: clean },
            ])
            .unwrap();

        let fence = prepared.commit_holding_publication_fence().unwrap();
        assert_eq!(fence.removed_blocks().len(), 2);
        let outcome = fence.finish();

        assert!(shared.try_lod_map_write(0).is_some());
        assert!(shared.try_lod_map_write(1).is_some());
        assert!(shared
            .try_write_region(0, Box3i::new(clean.position * 16, Vector3i::splat(16)))
            .is_some());
        assert!(shared
            .try_write_region(1, Box3i::new(dirty.position * 32, Vector3i::splat(32)))
            .is_some());
        assert!(shared.mutation_gate.try_lock().is_ok());
        assert_eq!(outcome.removed_blocks().len(), 2);
        let clean_removed = outcome
            .removed_blocks()
            .iter()
            .find(|removed| removed.location() == clean)
            .unwrap();
        assert_eq!(block_buffer_identity(clean_removed.block()), clean_identity);
        assert_eq!(
            clean_removed
                .block()
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            121
        );
        assert!(!clean_removed.block().is_modified());
        assert!(!clean_removed.block().is_edited());
        let dirty_removed = outcome
            .removed_blocks()
            .iter()
            .find(|removed| removed.location() == dirty)
            .unwrap();
        assert_eq!(block_buffer_identity(dirty_removed.block()), dirty_identity);
        assert_eq!(
            dirty_removed
                .block()
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            122
        );
        assert!(dirty_removed.block().is_modified());
        assert!(dirty_removed.block().is_edited());
        assert!(dirty_removed.block().needs_lodding());
    }

    #[test]
    fn publication_fence_take_removed_blocks_is_allocation_free_and_finish_is_empty() {
        let shared = make_shared_data_with_lods(1);
        let removed_location = location(Vector3i::zero(), 0);
        let block = make_tagged_lod_block(131, 0);
        let buffer_identity = block_buffer_identity(&block);
        assert!(shared
            .try_set_block(removed_location.position, block)
            .unwrap());
        let mut prepared = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::Remove {
                location: removed_location,
            }])
            .unwrap();

        let mut fence = prepared.commit_holding_publication_fence().unwrap();
        let vector_identity = fence.removed_blocks().as_ptr();
        let vector_capacity = fence.removed_blocks.capacity();
        let removed = fence.take_removed_blocks();
        assert_eq!(removed.as_ptr(), vector_identity);
        assert_eq!(removed.capacity(), vector_capacity);
        assert_eq!(block_buffer_identity(removed[0].block()), buffer_identity);
        let outcome = fence.finish();

        assert!(outcome.removed_blocks().is_empty());
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].location(), removed_location);
    }

    #[test]
    fn publication_fence_drop_releases_guards_before_bounds_and_removed_owners() {
        let shared = make_shared_data_with_lods(2);
        let first = location(Vector3i::zero(), 0);
        let second = location(Vector3i::new(1, 0, 0), 1);
        let first_pool = Arc::new(VoxelMemoryPool::new());
        let second_pool = Arc::new(VoxelMemoryPool::new());
        let mut first_voxels =
            VoxelBuffer::with_size(Vector3i::splat(16)).with_pool(first_pool.clone());
        first_voxels.set_voxel(141, 0, 0, 0, ChannelId::Type.index());
        let mut second_voxels =
            VoxelBuffer::with_size(Vector3i::splat(16)).with_pool(second_pool.clone());
        second_voxels.set_voxel(142, 0, 0, 0, ChannelId::Type.index());
        assert!(shared
            .try_set_block(
                first.position,
                VoxelDataBlock::with_voxels(first_voxels, first.lod_index),
            )
            .unwrap());
        assert!(shared
            .try_set_block(
                second.position,
                VoxelDataBlock::with_voxels(second_voxels, second.lod_index),
            )
            .unwrap());
        let mut prepared = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::Remove { location: second },
                SharedVoxelDataTransactionOperation::Remove { location: first },
            ])
            .unwrap();
        let probe_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let probe_saw_all_guards_released = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let probe_saw_removed_owner_alive = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let count = probe_count.clone();
        let guards_released = probe_saw_all_guards_released.clone();
        let owner_alive = probe_saw_removed_owner_alive.clone();
        let weak_shared = Arc::downgrade(&shared);
        let weak_first_pool = Arc::downgrade(&first_pool);
        let weak_second_pool = Arc::downgrade(&second_pool);
        prepared.set_test_spatial_batch_drop_hook(Arc::new(move || {
            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            owner_alive.store(
                weak_first_pool.strong_count() == 2 && weak_second_pool.strong_count() == 2,
                std::sync::atomic::Ordering::SeqCst,
            );
            let shared = weak_shared.upgrade().unwrap();
            let maps_available =
                shared.try_lod_map_write(0).is_some() && shared.try_lod_map_write(1).is_some();
            let spatial_available = shared
                .try_write_region(0, Box3i::new(Vector3i::zero(), Vector3i::splat(16)))
                .is_some()
                && shared
                    .try_write_region(1, Box3i::new(second.position * 32, Vector3i::splat(32)))
                    .is_some();
            let gate_available = shared.mutation_gate.try_lock().is_ok();
            guards_released.store(
                maps_available && spatial_available && gate_available,
                std::sync::atomic::Ordering::SeqCst,
            );
        }));

        let fence = prepared.commit_holding_publication_fence().unwrap();
        assert!(shared.try_lod_map_write(0).is_none());
        assert!(shared.try_lod_map_write(1).is_none());
        assert!(shared
            .try_write_region(0, Box3i::new(Vector3i::zero(), Vector3i::splat(16)))
            .is_none());
        assert!(shared
            .try_write_region(1, Box3i::new(second.position * 32, Vector3i::splat(32)),)
            .is_none());
        assert!(matches!(
            shared.mutation_gate.try_lock(),
            Err(std::sync::TryLockError::WouldBlock)
        ));
        drop(fence);

        assert_eq!(probe_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(probe_saw_all_guards_released.load(std::sync::atomic::Ordering::SeqCst));
        assert!(probe_saw_removed_owner_alive.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(Arc::strong_count(&first_pool), 1);
        assert_eq!(Arc::strong_count(&second_pool), 1);
        assert_eq!(shared.locked_region_count(0), 0);
        assert_eq!(shared.locked_region_count(1), 0);
        assert!(shared.try_lod_map_write(0).is_some());
        assert!(shared.try_lod_map_write(1).is_some());
        assert!(shared.mutation_gate.try_lock().is_ok());
    }

    #[test]
    fn publication_fence_preserves_prewrite_failure_retry_ownership() {
        #[derive(Clone, Copy)]
        enum Failure {
            Settings,
            Key,
            LiveSpatialRegistry,
            Reservation(SharedVoxelDataTransactionReservation),
        }

        for failure in [
            Failure::Settings,
            Failure::Key,
            Failure::LiveSpatialRegistry,
            Failure::Reservation(SharedVoxelDataTransactionReservation::LiveMap),
            Failure::Reservation(SharedVoxelDataTransactionReservation::LiveKeyRevisions),
        ] {
            let shared = make_shared_data_with_lods(1);
            let resident = location(Vector3i::zero(), 0);
            let inserted = location(Vector3i::new(1, 0, 0), 0);
            assert!(shared
                .try_set_block(resident.position, make_tagged_lod_block(151, 0))
                .unwrap());
            let insert = make_tagged_lod_block(152, 0);
            let insert_identity = block_buffer_identity(&insert);
            let mut prepared = shared
                .prepare_transaction(vec![
                    SharedVoxelDataTransactionOperation::Remove { location: resident },
                    SharedVoxelDataTransactionOperation::Insert {
                        location: inserted,
                        block: insert,
                        final_viewers: 3,
                    },
                ])
                .unwrap();

            let expected_error = match failure {
                Failure::Settings => {
                    shared.set_test_settings_revision(1);
                    SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                        expected_revision: 0,
                        actual_revision: 1,
                    }
                }
                Failure::Key => {
                    shared
                        .view_area(
                            Box3i::new(resident.position, Vector3i::splat(1)),
                            0,
                            None,
                            None,
                            None,
                        )
                        .unwrap();
                    SharedVoxelDataMutationError::ConcurrentDataMutation {
                        position: resident.position,
                        lod_index: 0,
                        expected_revision: VoxelDataKeyRevision::Present(1),
                        actual_revision: VoxelDataKeyRevision::Present(2),
                    }
                }
                Failure::LiveSpatialRegistry => {
                    shared.set_test_transaction_live_spatial_registry_fail_lod(Some(0));
                    SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                        reservation: SharedVoxelDataTransactionReservation::LiveSpatialRegistry,
                    }
                }
                Failure::Reservation(reservation) => {
                    shared.set_test_transaction_reservation_failpoint(Some(reservation));
                    SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                        reservation,
                    }
                }
            };

            let actual_error = match prepared.commit_holding_publication_fence() {
                Ok(_) => panic!("prewrite failure unexpectedly returned a publication fence"),
                Err(error) => error,
            };
            assert_eq!(actual_error, expected_error);
            assert_eq!(
                block_buffer_identity(prepared.inserted_block(inserted).unwrap()),
                insert_identity
            );
            assert!(prepared.spatial_batches[0].is_some());
            assert!(prepared.removed_blocks.is_empty());
            assert!(shared.with_lod_map(0, |map| map.has_block(resident.position)));
            assert!(!shared.with_lod_map(0, |map| map.has_block(inserted.position)));

            match failure {
                Failure::Settings => shared.set_test_settings_revision(0),
                Failure::Key => shared.with_lod_map_mut(0, |map| {
                    map.get_block_mut(resident.position)
                        .unwrap()
                        .viewers
                        .set_exact(0);
                    map.set_key_revision_for_test(resident.position, 1);
                }),
                Failure::LiveSpatialRegistry => {
                    shared.set_test_transaction_live_spatial_registry_fail_lod(None);
                }
                Failure::Reservation(_) => {
                    shared.set_test_transaction_reservation_failpoint(None);
                }
            }
            prepared
                .commit_holding_publication_fence()
                .unwrap()
                .finish();
            assert_eq!(
                shared.with_lod_map(0, |map| {
                    block_buffer_identity(map.get_block(inserted.position).unwrap())
                }),
                insert_identity
            );
        }
    }

    #[test]
    fn legacy_commit_delegates_to_fence_without_behavior_change() {
        let shared = make_shared_data_with_lods(1);
        let removed_location = location(Vector3i::zero(), 0);
        let block = make_tagged_lod_block(161, 0);
        let identity = block_buffer_identity(&block);
        assert!(shared
            .try_set_block(removed_location.position, block)
            .unwrap());
        let mut prepared = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::Remove {
                location: removed_location,
            }])
            .unwrap();

        let outcome = prepared.commit().unwrap();
        assert_eq!(outcome.removed_blocks().len(), 1);
        assert_eq!(outcome.removed_blocks()[0].location(), removed_location);
        assert_eq!(
            block_buffer_identity(outcome.removed_blocks()[0].block()),
            identity
        );
        assert_eq!(
            prepared.commit().unwrap_err(),
            SharedVoxelDataMutationError::PreparedTransactionAlreadyCommitted
        );
    }

    #[test]
    fn publication_fence_allows_split_borrow_outer_state_mutation() {
        struct PreparedOuterStateFixture {
            data_transaction: super::PreparedSharedVoxelDataTransaction,
            matching_state_marker: Option<BlockLocation>,
        }

        let shared = make_shared_data_with_lods(1);
        let changed = location(Vector3i::zero(), 0);
        assert!(shared
            .try_set_block(changed.position, make_tagged_lod_block(171, 0))
            .unwrap());
        let mut outer = PreparedOuterStateFixture {
            data_transaction: shared
                .prepare_transaction(vec![SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: changed,
                    final_viewers: 17,
                }])
                .unwrap(),
            matching_state_marker: None,
        };

        let fence = outer
            .data_transaction
            .commit_holding_publication_fence()
            .unwrap();
        outer.matching_state_marker = Some(changed);
        assert!(fence.removed_blocks().is_empty());
        fence.finish();

        assert_eq!(outer.matching_state_marker, Some(changed));
        assert_eq!(
            shared.with_lod_map(0, |map| map
                .get_block(changed.position)
                .unwrap()
                .viewers
                .get()),
            17
        );
    }

    #[test]
    fn prepared_batch_sets_exact_viewer_totals_across_lods_atomically() {
        let shared = make_shared_data_with_lods(3);
        let locations = [
            location(Vector3i::new(5, 0, 0), 2),
            location(Vector3i::new(-2, 1, 0), 0),
            location(Vector3i::new(1, -3, 4), 1),
        ];
        for (index, location) in locations.iter().enumerate() {
            assert!(shared
                .try_set_block(
                    location.position,
                    make_tagged_lod_block(10 + index as u64, location.lod_index),
                )
                .unwrap());
        }
        let before_structural = [
            shared.lod_structural_revision(0),
            shared.lod_structural_revision(1),
            shared.lod_structural_revision(2),
        ];

        let operations = vec![
            SharedVoxelDataTransactionOperation::SetViewersExact {
                location: locations[0],
                final_viewers: 7,
            },
            SharedVoxelDataTransactionOperation::SetViewersExact {
                location: locations[2],
                final_viewers: 3,
            },
            SharedVoxelDataTransactionOperation::SetViewersExact {
                location: locations[1],
                final_viewers: 5,
            },
        ];
        let mut prepared = shared.prepare_transaction(operations).unwrap();
        let outcome = prepared.commit().unwrap();

        assert!(outcome.removed_blocks().is_empty());
        for (location, expected_viewers) in locations.into_iter().zip([7, 5, 3]) {
            assert_eq!(
                shared.with_lod_map(usize::from(location.lod_index), |map| map
                    .get_block(location.position)
                    .unwrap()
                    .viewers
                    .get()),
                expected_viewers
            );
            assert_eq!(
                shared.key_revision(location.position, usize::from(location.lod_index)),
                Some(VoxelDataKeyRevision::Present(2))
            );
        }
        assert_eq!(
            [
                shared.lod_structural_revision(0),
                shared.lod_structural_revision(1),
                shared.lod_structural_revision(2),
            ],
            before_structural
        );
        assert_eq!(
            prepared.commit().unwrap_err(),
            SharedVoxelDataMutationError::PreparedTransactionAlreadyCommitted
        );
    }

    #[test]
    fn prepared_batch_same_key_conflict_rolls_back_all_and_retains_inputs() {
        let shared = make_shared_data_with_lods(2);
        let conflicted = location(Vector3i::new(0, 0, 0), 0);
        let untouched = location(Vector3i::new(1, 0, 0), 0);
        let inserted = location(Vector3i::new(2, 0, 0), 1);
        assert!(shared
            .try_set_block(conflicted.position, make_tagged_lod_block(11, 0))
            .unwrap());
        assert!(shared
            .try_set_block(untouched.position, make_tagged_lod_block(12, 0))
            .unwrap());
        let insert_block = make_tagged_lod_block(99, 1);
        let insert_identity = block_buffer_identity(&insert_block);
        let operations = vec![
            SharedVoxelDataTransactionOperation::SetViewersExact {
                location: untouched,
                final_viewers: 6,
            },
            SharedVoxelDataTransactionOperation::Insert {
                location: inserted,
                block: insert_block,
                final_viewers: 4,
            },
            SharedVoxelDataTransactionOperation::SetViewersExact {
                location: conflicted,
                final_viewers: 5,
            },
        ];
        let mut prepared = shared.prepare_transaction(operations).unwrap();

        shared
            .view_area(
                Box3i::new(conflicted.position, Vector3i::splat(1)),
                0,
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            prepared.commit().unwrap_err(),
            SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: conflicted.position,
                lod_index: 0,
                expected_revision: VoxelDataKeyRevision::Present(1),
                actual_revision: VoxelDataKeyRevision::Present(2),
            }
        );
        assert_eq!(
            shared.with_lod_map(0, |map| map
                .get_block(conflicted.position)
                .unwrap()
                .viewers
                .get()),
            1
        );
        assert_eq!(
            shared.with_lod_map(0, |map| map
                .get_block(untouched.position)
                .unwrap()
                .viewers
                .get()),
            0
        );
        assert!(!shared.with_lod_map(1, |map| map.has_block(inserted.position)));
        assert_eq!(
            block_buffer_identity(prepared.inserted_block(inserted).unwrap()),
            insert_identity
        );

        let operations = prepared.into_operations().unwrap();
        let mut fresh = shared.prepare_transaction(operations).unwrap();
        fresh.commit().unwrap();
        assert_eq!(
            shared.with_lod_map(0, |map| map
                .get_block(conflicted.position)
                .unwrap()
                .viewers
                .get()),
            5
        );
        assert_eq!(
            shared.with_lod_map(0, |map| map
                .get_block(untouched.position)
                .unwrap()
                .viewers
                .get()),
            6
        );
        assert_eq!(
            shared.with_lod_map(1, |map| {
                let block = map.get_block(inserted.position).unwrap();
                assert_eq!(block.viewers.get(), 4);
                block_buffer_identity(block)
            }),
            insert_identity
        );
        assert_eq!(
            fresh.commit().unwrap_err(),
            SharedVoxelDataMutationError::PreparedTransactionAlreadyCommitted
        );
    }

    #[test]
    fn prepared_batch_disjoint_key_write_does_not_false_conflict() {
        let shared = make_shared_data_with_lods(1);
        let prepared_location = location(Vector3i::new(0, 0, 0), 0);
        let disjoint_location = location(Vector3i::new(4, 0, 0), 0);
        assert!(shared
            .try_set_block(prepared_location.position, make_tagged_lod_block(1, 0))
            .unwrap());
        assert!(shared
            .try_set_block(disjoint_location.position, make_tagged_lod_block(2, 0))
            .unwrap());
        let mut prepared = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::SetViewersExact {
                location: prepared_location,
                final_viewers: 8,
            }])
            .unwrap();

        shared
            .view_area(
                Box3i::new(disjoint_location.position, Vector3i::splat(1)),
                0,
                None,
                None,
                None,
            )
            .unwrap();
        prepared.commit().unwrap();

        assert_eq!(
            shared.with_lod_map(0, |map| map
                .get_block(prepared_location.position)
                .unwrap()
                .viewers
                .get()),
            8
        );
        assert_eq!(
            shared.with_lod_map(0, |map| map
                .get_block(disjoint_location.position)
                .unwrap()
                .viewers
                .get()),
            1
        );
        assert_eq!(
            shared.key_revision(prepared_location.position, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
        assert_eq!(
            shared.key_revision(disjoint_location.position, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
    }

    #[test]
    fn prepared_batch_settings_conflict_and_key_overflow_are_non_mutating() {
        let shared = make_shared_data_with_lods(1);
        let present = location(Vector3i::zero(), 0);
        assert!(shared
            .try_set_block(present.position, make_tagged_lod_block(7, 0))
            .unwrap());
        let mut prepared = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::SetViewersExact {
                location: present,
                final_viewers: 9,
            }])
            .unwrap();
        let generator: SharedVoxelGenerator = Arc::new(RecordingGenerator::default());
        shared.set_generator(Some(generator)).unwrap();
        assert_eq!(
            prepared.commit().unwrap_err(),
            SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                expected_revision: 0,
                actual_revision: 1,
            }
        );
        assert_eq!(
            shared.with_lod_map(0, |map| map
                .get_block(present.position)
                .unwrap()
                .viewers
                .get()),
            0
        );
        assert_eq!(
            shared.key_revision(present.position, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );

        let overflowed = location(Vector3i::new(3, 0, 0), 0);
        shared.with_lod_map_mut(0, |map| {
            map.set_key_revision_for_test(overflowed.position, u64::MAX);
        });
        let insert = make_tagged_lod_block(55, 0);
        let identity = block_buffer_identity(&insert);
        let error = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::Insert {
                location: overflowed,
                block: insert,
                final_viewers: 1,
            }])
            .unwrap_err();
        assert_eq!(
            error.error(),
            &SharedVoxelDataMutationError::KeyRevisionOverflow {
                position: overflowed.position,
                lod_index: 0,
            }
        );
        let (_, operations) = error.into_parts();
        let SharedVoxelDataTransactionOperation::Insert { block, .. } = &operations[0] else {
            panic!("insert operation must be retained");
        };
        assert_eq!(block_buffer_identity(block), identity);
        assert!(!shared.with_lod_map(0, |map| map.has_block(overflowed.position)));
        assert_eq!(
            shared.key_revision(overflowed.position, 0),
            Some(VoxelDataKeyRevision::Tombstone(u64::MAX))
        );
    }

    #[test]
    fn prepared_insert_adopts_exact_block_and_preserves_pointer_across_retry() {
        let shared = make_shared_data_with_lods(1);
        let inserted = location(Vector3i::new(3, 2, 1), 0);
        let block = make_tagged_lod_block(71, 0);
        let identity = block_buffer_identity(&block);
        let mut prepared = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::Insert {
                location: inserted,
                block,
                final_viewers: 12,
            }])
            .unwrap();
        shared.set_test_transaction_reservation_failpoint(Some(
            SharedVoxelDataTransactionReservation::LiveMap,
        ));
        assert_eq!(
            prepared.commit().unwrap_err(),
            SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                reservation: SharedVoxelDataTransactionReservation::LiveMap,
            }
        );
        assert_eq!(
            block_buffer_identity(prepared.inserted_block(inserted).unwrap()),
            identity
        );
        assert!(!shared.with_lod_map(0, |map| map.has_block(inserted.position)));

        shared.set_test_transaction_reservation_failpoint(None);
        prepared.commit().unwrap();
        shared.with_lod_map(0, |map| {
            let resident = map.get_block(inserted.position).unwrap();
            assert_eq!(resident.viewers.get(), 12);
            assert_eq!(block_buffer_identity(resident), identity);
            assert_eq!(
                resident
                    .voxels()
                    .get_voxel(0, 0, 0, ChannelId::Type.index()),
                71
            );
        });
        assert_eq!(
            prepared.commit().unwrap_err(),
            SharedVoxelDataMutationError::PreparedTransactionAlreadyCommitted
        );
    }

    #[test]
    fn prepared_remove_returns_clean_and_dirty_blocks_after_guards_release() {
        let shared = make_shared_data_with_lods(2);
        let clean = location(Vector3i::new(0, 0, 0), 0);
        let dirty = location(Vector3i::new(1, 0, 0), 1);
        let clean_block = make_tagged_lod_block(21, 0);
        let clean_identity = block_buffer_identity(&clean_block);
        let mut dirty_block = make_tagged_lod_block(22, 1);
        dirty_block.set_modified(true);
        dirty_block.set_edited(true);
        dirty_block.set_needs_lodding(true);
        let dirty_identity = block_buffer_identity(&dirty_block);
        assert!(shared.try_set_block(clean.position, clean_block).unwrap());
        assert!(shared.try_set_block(dirty.position, dirty_block).unwrap());

        let mut prepared = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::Remove { location: dirty },
                SharedVoxelDataTransactionOperation::Remove { location: clean },
            ])
            .unwrap();
        let outcome = prepared.commit().unwrap();

        assert!(shared.try_lod_map_write(0).is_some());
        assert!(shared.try_lod_map_write(1).is_some());
        assert!(!shared.with_lod_map(0, |map| map.has_block(clean.position)));
        assert!(!shared.with_lod_map(1, |map| map.has_block(dirty.position)));
        assert_eq!(outcome.removed_blocks().len(), 2);
        let clean_removed = outcome
            .removed_blocks()
            .iter()
            .find(|removed| removed.location() == clean)
            .unwrap();
        assert!(!clean_removed.block().is_modified());
        assert!(!clean_removed.block().is_edited());
        assert_eq!(block_buffer_identity(clean_removed.block()), clean_identity);
        let dirty_removed = outcome
            .removed_blocks()
            .iter()
            .find(|removed| removed.location() == dirty)
            .unwrap();
        assert!(dirty_removed.block().is_modified());
        assert!(dirty_removed.block().is_edited());
        assert!(dirty_removed.block().needs_lodding());
        assert_eq!(block_buffer_identity(dirty_removed.block()), dirty_identity);
        assert_eq!(
            shared.key_revision(clean.position, 0),
            Some(VoxelDataKeyRevision::Tombstone(2))
        );
        assert_eq!(
            shared.key_revision(dirty.position, 1),
            Some(VoxelDataKeyRevision::Tombstone(2))
        );
    }

    #[test]
    fn prepared_replace_updates_one_key_once_and_retires_the_exact_old_owner() {
        let shared = make_shared_data_with_lods(1);
        let location = location(Vector3i::new(2, 0, 0), 0);
        let mut original = make_tagged_lod_block(201, 0);
        original.viewers.set_exact(3);
        original.set_modified(true);
        original.set_needs_lodding(true);
        let original_identity = block_buffer_identity(&original);
        assert!(shared.try_set_block(location.position, original).unwrap());
        let mut replacement = make_tagged_lod_block(202, 0);
        replacement.viewers.set_exact(7);
        replacement.set_edited(true);
        let replacement_identity = block_buffer_identity(&replacement);

        let mut prepared = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::Replace {
                location,
                block: replacement,
            }])
            .unwrap();
        let outcome = prepared.commit().unwrap();

        assert_eq!(
            shared.key_revision(location.position, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
        shared.with_lod_map(0, |map| {
            let resident = map.get_block(location.position).unwrap();
            assert_eq!(block_buffer_identity(resident), replacement_identity);
            assert_eq!(resident.viewers.get(), 7);
            assert!(!resident.is_modified());
            assert!(resident.is_edited());
            assert!(!resident.needs_lodding());
            assert_eq!(
                resident
                    .voxels()
                    .get_voxel(0, 0, 0, ChannelId::Type.index()),
                202
            );
        });
        assert_eq!(outcome.removed_blocks().len(), 1);
        assert_eq!(outcome.removed_blocks()[0].location(), location);
        assert_eq!(outcome.removed_blocks()[0].block().viewers.get(), 3);
        assert!(outcome.removed_blocks()[0].block().is_modified());
        assert!(!outcome.removed_blocks()[0].block().is_edited());
        assert!(outcome.removed_blocks()[0].block().needs_lodding());
        assert_eq!(
            block_buffer_identity(outcome.removed_blocks()[0].block()),
            original_identity
        );
    }

    #[test]
    fn prepared_replace_conflict_preserves_live_and_replacement_payload_identity() {
        let shared = make_shared_data_with_lods(1);
        let location = location(Vector3i::zero(), 0);
        let mut original = make_tagged_lod_block(211, 0);
        original.viewers.set_exact(4);
        original.set_modified(true);
        original.set_edited(true);
        let original_identity = block_buffer_identity(&original);
        assert!(shared.try_set_block(location.position, original).unwrap());
        let replacement = make_tagged_lod_block(212, 0);
        let replacement_identity = block_buffer_identity(&replacement);
        let mut prepared = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::Replace {
                location,
                block: replacement,
            }])
            .unwrap();

        assert!(shared
            .try_set_voxel_checked(213, Vector3i::new(1, 1, 1), ChannelId::Type.index(),)
            .unwrap());
        assert_eq!(
            prepared.commit().unwrap_err(),
            SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: location.position,
                lod_index: 0,
                expected_revision: VoxelDataKeyRevision::Present(1),
                actual_revision: VoxelDataKeyRevision::Present(2),
            }
        );
        assert_eq!(
            block_buffer_identity(prepared.owned_block(location).unwrap()),
            replacement_identity
        );
        assert!(prepared.removed_blocks.is_empty());
        assert_eq!(
            shared.key_revision(location.position, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
        shared.with_lod_map(0, |map| {
            let resident = map.get_block(location.position).unwrap();
            assert_eq!(block_buffer_identity(resident), original_identity);
            assert_eq!(resident.viewers.get(), 4);
            assert!(resident.is_modified());
            assert!(resident.is_edited());
            assert_eq!(
                resident
                    .voxels()
                    .get_voxel(1, 1, 1, ChannelId::Type.index()),
                213
            );
        });
    }

    #[test]
    fn prepared_clear_modified_advances_revision_without_changing_voxels() {
        let shared = make_shared_data_with_lods(1);
        let location = location(Vector3i::new(-2, 0, 0), 0);
        let mut dirty = make_tagged_lod_block(221, 0);
        dirty.set_modified(true);
        dirty.set_edited(true);
        dirty.viewers.set_exact(5);
        let identity = block_buffer_identity(&dirty);
        assert!(shared.try_set_block(location.position, dirty).unwrap());
        let preview = shared.begin_transaction_preview();
        let snapshot = preview.block_snapshot(location).unwrap();
        assert_eq!(snapshot.revision(), VoxelDataKeyRevision::Present(1));
        assert!(snapshot.is_edited());

        let mut prepared = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::ClearModified {
                location,
            }])
            .unwrap();
        let outcome = prepared.commit().unwrap();

        assert!(outcome.removed_blocks().is_empty());
        assert_eq!(
            shared.key_revision(location.position, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
        shared.with_lod_map(0, |map| {
            let resident = map.get_block(location.position).unwrap();
            assert_eq!(block_buffer_identity(resident), identity);
            assert!(!resident.is_modified());
            assert!(resident.is_edited());
            assert_eq!(resident.viewers.get(), 5);
            assert_eq!(
                resident
                    .voxels()
                    .get_voxel(0, 0, 0, ChannelId::Type.index()),
                221
            );
        });
    }

    #[test]
    fn prepared_clear_modified_rejects_same_key_edit_after_snapshot() {
        let shared = make_shared_data_with_lods(1);
        let location = location(Vector3i::zero(), 0);
        let mut dirty = make_tagged_lod_block(231, 0);
        dirty.set_modified(true);
        dirty.set_edited(true);
        dirty.viewers.set_exact(6);
        let identity = block_buffer_identity(&dirty);
        assert!(shared.try_set_block(location.position, dirty).unwrap());
        let mut prepared = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::ClearModified {
                location,
            }])
            .unwrap();

        assert!(shared
            .try_set_voxel_checked(232, Vector3i::new(1, 1, 1), ChannelId::Type.index(),)
            .unwrap());
        assert_eq!(
            prepared.commit().unwrap_err(),
            SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: location.position,
                lod_index: 0,
                expected_revision: VoxelDataKeyRevision::Present(1),
                actual_revision: VoxelDataKeyRevision::Present(2),
            }
        );
        assert!(prepared.removed_blocks.is_empty());
        assert_eq!(
            shared.key_revision(location.position, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
        shared.with_lod_map(0, |map| {
            let resident = map.get_block(location.position).unwrap();
            assert_eq!(block_buffer_identity(resident), identity);
            assert!(resident.is_modified());
            assert!(resident.is_edited());
            assert_eq!(resident.viewers.get(), 6);
            assert_eq!(
                resident
                    .voxels()
                    .get_voxel(1, 1, 1, ChannelId::Type.index()),
                232
            );
        });
    }

    #[test]
    fn resident_dirty_snapshot_and_clear_commit_with_one_exact_block_revision() {
        let shared = make_shared_data_with_lods(1);
        let location = location(Vector3i::new(3, -1, 2), 0);
        let mut dirty = make_tagged_lod_block(233, 0);
        dirty.set_modified(true);
        dirty.set_edited(true);
        dirty.viewers.set_exact(7);
        let resident_identity = block_buffer_identity(&dirty);
        assert!(shared.try_set_block(location.position, dirty).unwrap());

        let preview = shared.begin_transaction_preview();
        let copies = preview.prepare_resident_dirty_copies().unwrap();
        let (operations, snapshots, payloads) = copies.into_parts();
        assert_eq!(operations.len(), 1);
        assert_eq!(snapshots.len(), 1);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].location, location);
        assert_eq!(payloads[0].block_revision, 1);
        assert_ne!(
            payloads[0].voxels.channel_bytes(0).as_ptr(),
            resident_identity
        );
        assert_eq!(
            payloads[0]
                .voxels
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            233
        );

        let mut prepared = preview.prepare_transaction(operations, &snapshots).unwrap();
        prepared.commit().unwrap();

        assert_eq!(
            shared.key_revision(location.position, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
        shared.with_lod_map(0, |map| {
            let resident = map.get_block(location.position).unwrap();
            assert_eq!(block_buffer_identity(resident), resident_identity);
            assert!(!resident.is_modified());
            assert!(resident.is_edited());
            assert_eq!(resident.viewers.get(), 7);
        });
    }

    #[test]
    fn resident_dirty_copy_vs_edit_conflict_preserves_new_edit_dirty() {
        let shared = make_shared_data_with_lods(1);
        let location = location(Vector3i::zero(), 0);
        let mut dirty = make_tagged_lod_block(234, 0);
        dirty.set_modified(true);
        dirty.set_edited(true);
        assert!(shared.try_set_block(location.position, dirty).unwrap());

        let preview = shared.begin_transaction_preview();
        let (operations, snapshots, payloads) = preview
            .prepare_resident_dirty_copies()
            .unwrap()
            .into_parts();
        assert_eq!(payloads[0].block_revision, 1);
        assert!(shared
            .try_set_voxel_checked(235, Vector3i::new(1, 1, 1), ChannelId::Type.index(),)
            .unwrap());

        let error = preview
            .prepare_transaction(operations, &snapshots)
            .unwrap_err();
        assert_eq!(
            error.error(),
            &SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: location.position,
                lod_index: 0,
                expected_revision: VoxelDataKeyRevision::Present(1),
                actual_revision: VoxelDataKeyRevision::Present(2),
            }
        );
        assert_eq!(
            payloads[0]
                .voxels
                .get_voxel(1, 1, 1, ChannelId::Type.index()),
            0
        );
        let resident = shared.block_snapshot(location.position, 0).unwrap();
        assert!(resident.is_modified());
        assert_eq!(
            resident
                .voxels()
                .get_voxel(1, 1, 1, ChannelId::Type.index()),
            235
        );
    }

    #[test]
    fn resident_edit_after_snapshot_commit_creates_a_superseding_revision() {
        let shared = make_shared_data_with_lods(1);
        let location = location(Vector3i::zero(), 0);
        let mut dirty = make_tagged_lod_block(236, 0);
        dirty.set_modified(true);
        assert!(shared.try_set_block(location.position, dirty).unwrap());

        let preview = shared.begin_transaction_preview();
        let (operations, snapshots, first_payloads) = preview
            .prepare_resident_dirty_copies()
            .unwrap()
            .into_parts();
        let mut prepared = preview.prepare_transaction(operations, &snapshots).unwrap();
        prepared.commit().unwrap();
        assert_eq!(first_payloads[0].block_revision, 1);

        assert!(shared
            .try_edit_voxel_checked(237, Vector3i::new(2, 2, 2), ChannelId::Type.index(),)
            .unwrap());
        let second_preview = shared.begin_transaction_preview();
        let (_, _, second_payloads) = second_preview
            .prepare_resident_dirty_copies()
            .unwrap()
            .into_parts();
        assert_eq!(second_payloads[0].block_revision, 3);
        assert_eq!(
            second_payloads[0]
                .voxels
                .get_voxel(2, 2, 2, ChannelId::Type.index()),
            237
        );
        assert!(shared
            .block_snapshot(location.position, 0)
            .unwrap()
            .is_modified());
    }

    #[test]
    fn resident_snapshot_capacity_failure_preserves_every_dirty_key() {
        for reservation in [
            SharedVoxelDataTransactionReservation::OperationStorage,
            SharedVoxelDataTransactionReservation::PreviewSnapshotStorage,
            SharedVoxelDataTransactionReservation::ResidentDirtyPayloadStorage,
            SharedVoxelDataTransactionReservation::ResidentDirtyPayloadCopy,
        ] {
            let shared = make_shared_data_with_lods(2);
            let locations = [
                location(Vector3i::new(-1, 0, 0), 0),
                location(Vector3i::new(1, 0, 0), 1),
            ];
            let mut identities = Vec::new();
            for (tag, location) in [241_u64, 242].into_iter().zip(locations) {
                let mut dirty = make_tagged_lod_block(tag, location.lod_index);
                dirty.set_modified(true);
                identities.push(block_buffer_identity(&dirty));
                assert!(shared.try_set_block(location.position, dirty).unwrap());
            }
            shared.set_test_transaction_reservation_failpoint(Some(reservation));

            assert_eq!(
                shared
                    .begin_transaction_preview()
                    .prepare_resident_dirty_copies()
                    .err()
                    .unwrap(),
                SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation,
                }
            );
            for (index, location) in locations.into_iter().enumerate() {
                assert_eq!(
                    shared.key_revision(location.position, usize::from(location.lod_index)),
                    Some(VoxelDataKeyRevision::Present(1))
                );
                shared.with_lod_map(usize::from(location.lod_index), |map| {
                    let resident = map.get_block(location.position).unwrap();
                    assert!(resident.is_modified());
                    assert_eq!(block_buffer_identity(resident), identities[index]);
                });
            }
        }
    }

    #[test]
    fn resident_second_payload_copy_failure_preserves_every_dirty_owner() {
        let shared = make_shared_data_with_lods(1);
        let locations = [
            location(Vector3i::new(-1, 0, 0), 0),
            location(Vector3i::zero(), 0),
            location(Vector3i::new(1, 0, 0), 0),
        ];
        let mut identities = Vec::new();
        for (tag, location) in [251_u64, 252, 253].into_iter().zip(locations) {
            let mut dirty = make_tagged_lod_block(tag, 0);
            dirty.set_modified(true);
            identities.push(block_buffer_identity(&dirty));
            assert!(shared.try_set_block(location.position, dirty).unwrap());
        }
        shared.fail_resident_payload_copy_for_test(2);

        assert_eq!(
            shared
                .begin_transaction_preview()
                .prepare_resident_dirty_copies()
                .err()
                .unwrap(),
            SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                reservation: SharedVoxelDataTransactionReservation::ResidentDirtyPayloadCopy,
            }
        );
        for (index, location) in locations.into_iter().enumerate() {
            assert_eq!(
                shared.key_revision(location.position, 0),
                Some(VoxelDataKeyRevision::Present(1))
            );
            shared.with_lod_map(0, |map| {
                let resident = map.get_block(location.position).unwrap();
                assert!(resident.is_modified());
                assert_eq!(block_buffer_identity(resident), identities[index]);
            });
        }
    }

    #[test]
    fn resident_dirty_without_voxels_is_typed_and_remains_dirty() {
        let shared = make_shared_data_with_lods(1);
        let location = location(Vector3i::zero(), 0);
        let mut dirty = VoxelDataBlock::empty(0);
        dirty.set_modified(true);
        assert!(shared.try_set_block(location.position, dirty).unwrap());

        assert_eq!(
            shared
                .begin_transaction_preview()
                .prepare_resident_dirty_copies()
                .err()
                .unwrap(),
            SharedVoxelDataMutationError::DirtyBlockMissingVoxels { location }
        );
        assert!(shared
            .block_snapshot(location.position, 0)
            .unwrap()
            .is_modified());
        assert_eq!(
            shared.key_revision(location.position, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
    }

    #[test]
    fn closed_mutation_admission_rejects_direct_and_prepared_writers_until_reopened() {
        let shared = make_shared_data_with_lods(1);
        let location = location(Vector3i::zero(), 0);
        assert!(shared
            .try_set_block(location.position, make_tagged_lod_block(261, 0))
            .unwrap());
        let mut stale = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::SetViewersExact {
                location,
                final_viewers: 3,
            }])
            .unwrap();
        let _permit = shared.close_mutation_admission_for_shutdown();

        assert!(shared.is_mutation_admission_closed());
        assert_eq!(
            shared.try_edit_voxel_checked(262, Vector3i::zero(), ChannelId::Type.index()),
            Err(SharedVoxelDataMutationError::MutationAdmissionClosed)
        );
        assert_eq!(
            stale.commit().unwrap_err(),
            SharedVoxelDataMutationError::MutationAdmissionClosed
        );
        shared.reopen_mutation_admission();
        stale.commit().unwrap();
        assert_eq!(
            shared
                .block_snapshot(location.position, 0)
                .unwrap()
                .viewers
                .get(),
            3
        );
    }

    #[test]
    fn shutdown_mutation_permit_is_exact_lineage_authority_for_closed_admission() {
        let first = make_shared_data_with_lods(1);
        let second = make_shared_data_with_lods(1);
        let position = Vector3i::zero();
        for shared in [&first, &second] {
            assert!(shared
                .try_set_block(position, make_tagged_lod_block(271, 0))
                .unwrap());
        }
        let location = location(position, 0);
        let mut first_transaction = first
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::SetViewersExact {
                location,
                final_viewers: 4,
            }])
            .unwrap();
        let mut second_transaction = second
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::SetViewersExact {
                location,
                final_viewers: 5,
            }])
            .unwrap();

        let first_permit = first.close_mutation_admission_for_shutdown();
        let second_revision_before = second.key_revision(position, 0);
        assert_eq!(
            second_transaction.authorize_shutdown_mutation(&first_permit),
            Err(SharedVoxelDataMutationError::ShutdownMutationPermitMismatch)
        );
        assert_eq!(second.key_revision(position, 0), second_revision_before);
        second_transaction.commit().unwrap();
        assert_eq!(second.block_snapshot(position, 0).unwrap().viewers.get(), 5);

        assert_eq!(
            first_transaction.commit().unwrap_err(),
            SharedVoxelDataMutationError::MutationAdmissionClosed
        );
        first_transaction
            .authorize_shutdown_mutation(&first_permit)
            .unwrap();
        first_transaction.commit().unwrap();
        assert_eq!(first.block_snapshot(position, 0).unwrap().viewers.get(), 4);
        assert!(first.is_mutation_admission_closed());
    }

    #[test]
    fn prepared_mixed_lod_replace_revision_overflow_is_preparation_atomic() {
        let shared = make_shared_data_with_lods(2);
        let lod0 = location(Vector3i::zero(), 0);
        let lod1 = location(Vector3i::zero(), 1);
        let mut lod0_original = make_tagged_lod_block(241, 0);
        lod0_original.viewers.set_exact(8);
        let lod0_original_identity = block_buffer_identity(&lod0_original);
        assert!(shared.try_set_block(lod0.position, lod0_original).unwrap());
        let mut lod1_original = make_tagged_lod_block(242, 1);
        lod1_original.set_modified(true);
        let lod1_original_identity = block_buffer_identity(&lod1_original);
        assert!(shared.try_set_block(lod1.position, lod1_original).unwrap());
        shared.with_lod_map_mut(1, |map| {
            map.set_key_revision_for_test(lod1.position, u64::MAX);
        });
        let lod0_replacement = make_tagged_lod_block(243, 0);
        let lod0_identity = block_buffer_identity(&lod0_replacement);
        let lod1_replacement = make_tagged_lod_block(244, 1);
        let lod1_identity = block_buffer_identity(&lod1_replacement);

        let error = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::Replace {
                    location: lod1,
                    block: lod1_replacement,
                },
                SharedVoxelDataTransactionOperation::Replace {
                    location: lod0,
                    block: lod0_replacement,
                },
            ])
            .unwrap_err();

        assert_eq!(
            error.error(),
            &SharedVoxelDataMutationError::KeyRevisionOverflow {
                position: lod1.position,
                lod_index: 1,
            }
        );
        let (_, operations) = error.into_parts();
        let retained_identity = |location| {
            operations
                .iter()
                .find_map(|operation| match operation {
                    SharedVoxelDataTransactionOperation::Replace {
                        location: candidate,
                        block,
                    } if *candidate == location => Some(block_buffer_identity(block)),
                    _ => None,
                })
                .unwrap()
        };
        assert_eq!(retained_identity(lod0), lod0_identity);
        assert_eq!(retained_identity(lod1), lod1_identity);
        assert_eq!(
            shared.key_revision(lod0.position, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(
            shared.key_revision(lod1.position, 1),
            Some(VoxelDataKeyRevision::Present(u64::MAX))
        );
        shared.with_lod_map(0, |map| {
            let resident = map.get_block(lod0.position).unwrap();
            assert_eq!(block_buffer_identity(resident), lod0_original_identity);
            assert_eq!(resident.viewers.get(), 8);
            assert!(!resident.is_modified());
            assert!(!resident.is_edited());
            assert_eq!(
                resident
                    .voxels()
                    .get_voxel(0, 0, 0, ChannelId::Type.index()),
                241
            );
        });
        shared.with_lod_map(1, |map| {
            let resident = map.get_block(lod1.position).unwrap();
            assert_eq!(block_buffer_identity(resident), lod1_original_identity);
            assert!(resident.is_modified());
            assert!(!resident.is_edited());
            assert_eq!(resident.viewers.get(), 0);
            assert_eq!(
                resident
                    .voxels()
                    .get_voxel(0, 0, 0, ChannelId::Type.index()),
                242
            );
        });
    }

    #[test]
    fn prepared_mixed_clear_and_replace_late_conflict_rolls_back_then_retries() {
        let shared = make_shared_data_with_lods(2);
        let cleared = location(Vector3i::zero(), 0);
        let replaced = location(Vector3i::zero(), 1);
        let mut dirty = make_tagged_lod_block(261, 0);
        dirty.set_modified(true);
        dirty.set_edited(true);
        let dirty_identity = block_buffer_identity(&dirty);
        assert!(shared.try_set_block(cleared.position, dirty).unwrap());
        assert!(shared
            .try_set_block(replaced.position, make_tagged_lod_block(262, 1))
            .unwrap());
        let replacement = make_tagged_lod_block(263, 1);
        let replacement_identity = block_buffer_identity(&replacement);
        let mut prepared = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::Replace {
                    location: replaced,
                    block: replacement,
                },
                SharedVoxelDataTransactionOperation::ClearModified { location: cleared },
            ])
            .unwrap();

        let concurrent = make_tagged_lod_block(264, 1);
        let concurrent_identity = block_buffer_identity(&concurrent);
        shared.with_lod_map_mut(1, |map| {
            map.set_block(replaced.position, concurrent, true);
            map.set_key_revision_for_test(replaced.position, 2);
        });
        assert_eq!(
            prepared.commit().unwrap_err(),
            SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: replaced.position,
                lod_index: 1,
                expected_revision: VoxelDataKeyRevision::Present(1),
                actual_revision: VoxelDataKeyRevision::Present(2),
            }
        );
        assert!(prepared.removed_blocks.is_empty());
        assert_eq!(
            block_buffer_identity(prepared.owned_block(replaced).unwrap()),
            replacement_identity
        );
        shared.with_lod_map(0, |map| {
            let resident = map.get_block(cleared.position).unwrap();
            assert_eq!(block_buffer_identity(resident), dirty_identity);
            assert!(resident.is_modified());
        });
        assert_eq!(
            shared.key_revision(cleared.position, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        shared.with_lod_map(1, |map| {
            let resident = map.get_block(replaced.position).unwrap();
            assert_eq!(block_buffer_identity(resident), concurrent_identity);
            assert_eq!(
                resident
                    .voxels()
                    .get_voxel(0, 0, 0, ChannelId::Type.index()),
                264
            );
        });

        let operations = prepared.into_operations().unwrap();
        let mut retry = shared.prepare_transaction(operations).unwrap();
        let outcome = retry.commit().unwrap();
        assert!(!shared
            .block_snapshot(cleared.position, 0)
            .unwrap()
            .is_modified());
        shared.with_lod_map(1, |map| {
            assert_eq!(
                block_buffer_identity(
                    map.get_block(replaced.position)
                        .expect("retry installs replacement")
                ),
                replacement_identity
            );
        });
        assert_eq!(outcome.removed_blocks().len(), 1);
        assert_eq!(
            block_buffer_identity(outcome.removed_blocks()[0].block()),
            concurrent_identity
        );
    }

    #[test]
    fn prepared_replace_and_clear_failpoints_preserve_owned_and_resident_state() {
        let shared = make_shared_data_with_lods(1);
        let replaced = location(Vector3i::zero(), 0);
        let cleared = location(Vector3i::new(1, 0, 0), 0);
        assert!(shared
            .try_set_block(replaced.position, make_tagged_lod_block(251, 0))
            .unwrap());
        let mut dirty = make_tagged_lod_block(252, 0);
        dirty.set_modified(true);
        let dirty_identity = block_buffer_identity(&dirty);
        assert!(shared.try_set_block(cleared.position, dirty).unwrap());
        let replacement = make_tagged_lod_block(253, 0);
        let replacement_identity = block_buffer_identity(&replacement);

        shared.set_test_transaction_reservation_failpoint(Some(
            SharedVoxelDataTransactionReservation::RemovedOutcome,
        ));
        let error = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::Replace {
                location: replaced,
                block: replacement,
            }])
            .unwrap_err();
        assert_eq!(
            error.error(),
            &SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                reservation: SharedVoxelDataTransactionReservation::RemovedOutcome,
            }
        );
        let (_, operations) = error.into_parts();
        assert_eq!(
            operations
                .iter()
                .find_map(|operation| match operation {
                    SharedVoxelDataTransactionOperation::Replace { block, .. } => {
                        Some(block_buffer_identity(block))
                    }
                    _ => None,
                })
                .unwrap(),
            replacement_identity
        );
        assert_eq!(
            shared.with_lod_map(0, |map| map
                .get_block(replaced.position)
                .unwrap()
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index())),
            251
        );

        shared.set_test_transaction_reservation_failpoint(None);
        let mut prepared = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::ClearModified {
                location: cleared,
            }])
            .unwrap();
        shared.set_test_transaction_reservation_failpoint(Some(
            SharedVoxelDataTransactionReservation::LiveKeyRevisions,
        ));
        assert_eq!(
            prepared.commit().unwrap_err(),
            SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                reservation: SharedVoxelDataTransactionReservation::LiveKeyRevisions,
            }
        );
        assert_eq!(
            shared.key_revision(cleared.position, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        shared.with_lod_map(0, |map| {
            let resident = map.get_block(cleared.position).unwrap();
            assert_eq!(block_buffer_identity(resident), dirty_identity);
            assert!(resident.is_modified());
        });

        shared.set_test_transaction_reservation_failpoint(None);
        prepared.commit().unwrap();
        assert!(!shared
            .block_snapshot(cleared.position, 0)
            .unwrap()
            .is_modified());
    }

    #[test]
    fn prepared_batch_capacity_failpoints_cover_every_destination_without_state_change() {
        let preparation_failpoints = [
            SharedVoxelDataTransactionReservation::OperationStorage,
            SharedVoxelDataTransactionReservation::SpatialPreparation,
            SharedVoxelDataTransactionReservation::RemovedOutcome,
        ];
        for failpoint in preparation_failpoints {
            let shared = make_shared_data_with_lods(1);
            let resident = location(Vector3i::zero(), 0);
            let inserted = location(Vector3i::new(1, 0, 0), 0);
            assert!(shared
                .try_set_block(resident.position, make_tagged_lod_block(31, 0))
                .unwrap());
            let insert = make_tagged_lod_block(32, 0);
            let identity = block_buffer_identity(&insert);
            shared.set_test_transaction_reservation_failpoint(Some(failpoint));
            let error = shared
                .prepare_transaction(vec![
                    SharedVoxelDataTransactionOperation::Remove { location: resident },
                    SharedVoxelDataTransactionOperation::Insert {
                        location: inserted,
                        block: insert,
                        final_viewers: 2,
                    },
                ])
                .unwrap_err();
            assert_eq!(
                error.error(),
                &SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: failpoint,
                }
            );
            let (_, operations) = error.into_parts();
            let retained = operations
                .iter()
                .find_map(|operation| match operation {
                    SharedVoxelDataTransactionOperation::Insert { block, .. } => Some(block),
                    _ => None,
                })
                .unwrap();
            assert_eq!(block_buffer_identity(retained), identity);
            assert!(shared.with_lod_map(0, |map| map.has_block(resident.position)));
            assert!(!shared.with_lod_map(0, |map| map.has_block(inserted.position)));
            assert_eq!(
                shared.key_revision(resident.position, 0),
                Some(VoxelDataKeyRevision::Present(1))
            );
        }

        for failpoint in [
            SharedVoxelDataTransactionReservation::LiveMap,
            SharedVoxelDataTransactionReservation::LiveKeyRevisions,
        ] {
            let shared = make_shared_data_with_lods(1);
            let resident = location(Vector3i::zero(), 0);
            let inserted = location(Vector3i::new(1, 0, 0), 0);
            assert!(shared
                .try_set_block(resident.position, make_tagged_lod_block(41, 0))
                .unwrap());
            let insert = make_tagged_lod_block(42, 0);
            let identity = block_buffer_identity(&insert);
            let mut prepared = shared
                .prepare_transaction(vec![
                    SharedVoxelDataTransactionOperation::Remove { location: resident },
                    SharedVoxelDataTransactionOperation::Insert {
                        location: inserted,
                        block: insert,
                        final_viewers: 2,
                    },
                ])
                .unwrap();
            shared.set_test_transaction_reservation_failpoint(Some(failpoint));
            assert_eq!(
                prepared.commit().unwrap_err(),
                SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: failpoint,
                }
            );
            assert_eq!(
                block_buffer_identity(prepared.inserted_block(inserted).unwrap()),
                identity
            );
            assert!(shared.with_lod_map(0, |map| map.has_block(resident.position)));
            assert!(!shared.with_lod_map(0, |map| map.has_block(inserted.position)));
            assert_eq!(
                shared.key_revision(resident.position, 0),
                Some(VoxelDataKeyRevision::Present(1))
            );
            assert_eq!(
                shared.key_revision(inserted.position, 0),
                Some(VoxelDataKeyRevision::Tombstone(0))
            );
        }
    }

    #[test]
    fn prepared_batch_live_spatial_registry_failure_restores_all_batches_and_inputs() {
        let shared = make_shared_data_with_lods(2);
        let inserted = location(Vector3i::new(0, 0, 0), 0);
        let existing = location(Vector3i::new(2, 0, 0), 1);
        assert!(shared
            .try_set_block(existing.position, make_tagged_lod_block(51, 1))
            .unwrap());
        let insert = make_tagged_lod_block(52, 0);
        let insert_identity = block_buffer_identity(&insert);
        let mut prepared = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: existing,
                    final_viewers: 7,
                },
                SharedVoxelDataTransactionOperation::Insert {
                    location: inserted,
                    block: insert,
                    final_viewers: 3,
                },
            ])
            .unwrap();

        shared.set_test_transaction_live_spatial_registry_fail_lod(Some(1));
        assert_eq!(
            prepared.commit().unwrap_err(),
            SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                reservation: SharedVoxelDataTransactionReservation::LiveSpatialRegistry,
            }
        );
        assert_eq!(shared.locked_region_count(0), 0);
        assert_eq!(shared.locked_region_count(1), 0);
        assert!(!shared.with_lod_map(0, |map| map.has_block(inserted.position)));
        assert_eq!(
            shared.with_lod_map(1, |map| map
                .get_block(existing.position)
                .unwrap()
                .viewers
                .get()),
            0
        );
        assert_eq!(
            shared.key_revision(inserted.position, 0),
            Some(VoxelDataKeyRevision::Tombstone(0))
        );
        assert_eq!(
            shared.key_revision(existing.position, 1),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(
            block_buffer_identity(prepared.inserted_block(inserted).unwrap()),
            insert_identity
        );

        shared.set_test_transaction_live_spatial_registry_fail_lod(None);
        prepared.commit().unwrap();
        assert_eq!(
            shared.with_lod_map(0, |map| {
                let block = map.get_block(inserted.position).unwrap();
                assert_eq!(block.viewers.get(), 3);
                block_buffer_identity(block)
            }),
            insert_identity
        );
        assert_eq!(
            shared.with_lod_map(1, |map| map
                .get_block(existing.position)
                .unwrap()
                .viewers
                .get()),
            7
        );
    }

    #[test]
    fn prepared_batch_retires_spatial_bounds_after_mutation_gate_release() {
        let shared = make_shared_data_with_lods(2);
        let first = location(Vector3i::new(0, 0, 0), 0);
        let second = location(Vector3i::new(1, 0, 0), 1);
        assert!(shared
            .try_set_block(first.position, make_tagged_lod_block(53, 0))
            .unwrap());
        assert!(shared
            .try_set_block(second.position, make_tagged_lod_block(54, 1))
            .unwrap());
        let mut prepared = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: first,
                    final_viewers: 2,
                },
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: second,
                    final_viewers: 4,
                },
            ])
            .unwrap();
        let drop_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let drop_saw_gate_released = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let count = drop_count.clone();
        let released = drop_saw_gate_released.clone();
        let weak = Arc::downgrade(&shared);
        prepared.set_test_spatial_batch_drop_hook(Arc::new(move || {
            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let Some(shared) = weak.upgrade() else {
                released.store(false, std::sync::atomic::Ordering::SeqCst);
                return;
            };
            let gate_is_available = match shared.mutation_gate.try_lock() {
                Ok(_) | Err(std::sync::TryLockError::Poisoned(_)) => true,
                Err(std::sync::TryLockError::WouldBlock) => false,
            };
            if !gate_is_available {
                released.store(false, std::sync::atomic::Ordering::SeqCst);
            }
        }));

        prepared.commit().unwrap();
        assert_eq!(drop_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(drop_saw_gate_released.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn prepared_batch_duplicate_or_presence_mismatch_is_rejected_before_mutation() {
        let shared = make_shared_data_with_lods(1);
        let present = location(Vector3i::zero(), 0);
        let absent = location(Vector3i::new(1, 0, 0), 0);
        assert!(shared
            .try_set_block(present.position, make_tagged_lod_block(61, 0))
            .unwrap());

        let duplicate = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: present,
                    final_viewers: 1,
                },
                SharedVoxelDataTransactionOperation::Remove { location: present },
            ])
            .unwrap_err();
        assert_eq!(
            duplicate.error(),
            &SharedVoxelDataMutationError::DuplicatePreparedTransactionOperation {
                location: present,
            }
        );
        assert_eq!(duplicate.operations().len(), 2);

        let missing = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::SetViewersExact {
                location: absent,
                final_viewers: 1,
            }])
            .unwrap_err();
        assert_eq!(
            missing.error(),
            &SharedVoxelDataMutationError::PreparedTransactionExpectedPresent {
                location: absent,
                actual_revision: VoxelDataKeyRevision::Tombstone(0),
            }
        );

        let insert = make_tagged_lod_block(62, 0);
        let identity = block_buffer_identity(&insert);
        let occupied = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::Insert {
                location: present,
                block: insert,
                final_viewers: 1,
            }])
            .unwrap_err();
        assert_eq!(
            occupied.error(),
            &SharedVoxelDataMutationError::PreparedTransactionExpectedTombstone {
                location: present,
                actual_revision: VoxelDataKeyRevision::Present(1),
            }
        );
        let (_, operations) = occupied.into_parts();
        let SharedVoxelDataTransactionOperation::Insert { block, .. } = &operations[0] else {
            panic!("insert operation must be retained");
        };
        assert_eq!(block_buffer_identity(block), identity);

        let no_op = shared
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::SetViewersExact {
                location: present,
                final_viewers: 0,
            }])
            .unwrap_err();
        assert_eq!(
            no_op.error(),
            &SharedVoxelDataMutationError::PreparedTransactionNoop { location: present }
        );
        assert_eq!(
            shared.key_revision(present.position, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(
            shared.with_lod_map(0, |map| map
                .get_block(present.position)
                .unwrap()
                .viewers
                .get()),
            0
        );
    }

    #[test]
    fn prepared_batch_two_writers_use_canonical_order_without_deadlock() {
        let shared = make_shared_data_with_lods(2);
        let first = location(Vector3i::new(-1, 0, 0), 0);
        let second = location(Vector3i::new(2, 0, 0), 1);
        assert!(shared
            .try_set_block(first.position, make_tagged_lod_block(81, 0))
            .unwrap());
        assert!(shared
            .try_set_block(second.position, make_tagged_lod_block(82, 1))
            .unwrap());
        let prepared_a = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: second,
                    final_viewers: 3,
                },
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: first,
                    final_viewers: 2,
                },
            ])
            .unwrap();
        let prepared_b = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: first,
                    final_viewers: 4,
                },
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: second,
                    final_viewers: 5,
                },
            ])
            .unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let (sent, received) = std::sync::mpsc::channel();
        let mut workers = Vec::new();
        for (mut prepared, barrier) in
            [(prepared_a, barrier.clone()), (prepared_b, barrier.clone())]
        {
            let sent = sent.clone();
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                sent.send(prepared.commit()).unwrap();
            }));
        }
        barrier.wait();
        let results = [
            received.recv_timeout(Duration::from_secs(2)).unwrap(),
            received.recv_timeout(Duration::from_secs(2)).unwrap(),
        ];
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);
        let final_totals = shared.with_lod_map(0, |map| {
            let first_total = map.get_block(first.position).unwrap().viewers.get();
            let second_total = shared.with_lod_map(1, |map| {
                map.get_block(second.position).unwrap().viewers.get()
            });
            (first_total, second_total)
        });
        assert!(matches!(final_totals, (2, 3) | (4, 5)));
        assert_eq!(
            shared.key_revision(first.position, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
        assert_eq!(
            shared.key_revision(second.position, 1),
            Some(VoxelDataKeyRevision::Present(2))
        );
    }

    #[test]
    fn storage_publication_fence_excludes_observer_until_finish() {
        let shared = make_shared_data_with_lods(1);
        let first = location(Vector3i::new(0, 0, 0), 0);
        let second = location(Vector3i::new(1, 0, 0), 0);
        assert!(shared
            .try_set_block(first.position, make_tagged_lod_block(91, 0))
            .unwrap());
        assert!(shared
            .try_set_block(second.position, make_tagged_lod_block(92, 0))
            .unwrap());
        let mut prepared = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: second,
                    final_viewers: 20,
                },
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: first,
                    final_viewers: 10,
                },
            ])
            .unwrap();
        let pause = Arc::new((Mutex::new(false), Condvar::new()));
        let pause_hook = pause.clone();
        let (phase_sent, phase_received) = std::sync::mpsc::channel();
        shared.set_test_edit_phase_hook(Arc::new(move |phase| {
            if phase != SharedVoxelDataEditPhase::PreparedTransactionValidatedBeforeFirstLiveWrite {
                return;
            }
            phase_sent.send(()).unwrap();
            let (lock, cvar) = &*pause_hook;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = cvar.wait(released).unwrap();
            }
        }));
        let commit_worker = std::thread::spawn(move || prepared.commit());
        phase_received.recv_timeout(Duration::from_secs(1)).unwrap();

        let voxel_box = Box3i::new(Vector3i::zero(), Vector3i::new(32, 16, 16));
        assert!(
            shared.try_read_region(0, voxel_box).is_none(),
            "the observer's explicit read attempt must be excluded by the prepared spatial batch"
        );
        let (lock, cvar) = &*pause;
        *lock.lock().unwrap() = true;
        cvar.notify_all();
        assert!(commit_worker.join().unwrap().is_ok());
        let _read = shared.read_region(0, voxel_box);
        assert_eq!(
            shared.with_lod_map(0, |map| {
                (
                    map.get_block(first.position).unwrap().viewers.get(),
                    map.get_block(second.position).unwrap().viewers.get(),
                )
            }),
            (10, 20)
        );
    }

    #[test]
    fn prepared_batch_pre_first_write_hook_panic_is_non_mutating_and_retryable() {
        let shared = make_shared_data_with_lods(1);
        let first = location(Vector3i::new(0, 0, 0), 0);
        let second = location(Vector3i::new(1, 0, 0), 0);
        assert!(shared
            .try_set_block(first.position, make_tagged_lod_block(93, 0))
            .unwrap());
        assert!(shared
            .try_set_block(second.position, make_tagged_lod_block(94, 0))
            .unwrap());
        let mut prepared = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: first,
                    final_viewers: 11,
                },
                SharedVoxelDataTransactionOperation::SetViewersExact {
                    location: second,
                    final_viewers: 22,
                },
            ])
            .unwrap();
        shared.set_test_edit_phase_hook(Arc::new(|phase| {
            if phase == SharedVoxelDataEditPhase::PreparedTransactionValidatedBeforeFirstLiveWrite {
                panic!("injected pre-first-write hook panic");
            }
        }));

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prepared.commit()));
        assert!(panic.is_err());
        assert_eq!(shared.locked_region_count(0), 0);
        shared.with_lod_map(0, |map| {
            assert_eq!(map.get_block(first.position).unwrap().viewers.get(), 0);
            assert_eq!(map.get_block(second.position).unwrap().viewers.get(), 0);
        });
        assert_eq!(
            shared.key_revision(first.position, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(
            shared.key_revision(second.position, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );

        shared.set_test_edit_phase_hook(Arc::new(|_| {}));
        prepared.commit().unwrap();
        shared.with_lod_map(0, |map| {
            assert_eq!(map.get_block(first.position).unwrap().viewers.get(), 11);
            assert_eq!(map.get_block(second.position).unwrap().viewers.get(), 22);
        });
    }

    #[test]
    fn prepared_batch_has_no_recoverable_failure_after_first_live_write() {
        let shared = make_shared_data_with_lods(1);
        let resident = location(Vector3i::zero(), 0);
        let inserted = location(Vector3i::new(1, 0, 0), 0);
        assert!(shared
            .try_set_block(resident.position, make_tagged_lod_block(101, 0))
            .unwrap());
        let first_write_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let first_write_hook = first_write_count.clone();
        shared.set_test_edit_phase_hook(Arc::new(move |phase| {
            if phase == SharedVoxelDataEditPhase::PreparedTransactionValidatedBeforeFirstLiveWrite {
                first_write_hook.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }));
        for failpoint in [
            SharedVoxelDataTransactionReservation::LiveMap,
            SharedVoxelDataTransactionReservation::LiveKeyRevisions,
        ] {
            let mut prepared = shared
                .prepare_transaction(vec![
                    SharedVoxelDataTransactionOperation::Remove { location: resident },
                    SharedVoxelDataTransactionOperation::Insert {
                        location: inserted,
                        block: make_tagged_lod_block(102, 0),
                        final_viewers: 6,
                    },
                ])
                .unwrap();
            shared.set_test_transaction_reservation_failpoint(Some(failpoint));
            assert!(prepared.commit().is_err());
            assert_eq!(
                first_write_count.load(std::sync::atomic::Ordering::SeqCst),
                0
            );
        }
        shared.set_test_transaction_reservation_failpoint(None);
        let mut prepared = shared
            .prepare_transaction(vec![
                SharedVoxelDataTransactionOperation::Remove { location: resident },
                SharedVoxelDataTransactionOperation::Insert {
                    location: inserted,
                    block: make_tagged_lod_block(103, 0),
                    final_viewers: 6,
                },
            ])
            .unwrap();
        prepared.commit().unwrap();
        assert_eq!(
            first_write_count.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[test]
    fn preserving_lod_resize_extends_without_replacing_existing_blocks() {
        let mut data = VoxelData::new();
        let negative = Vector3i::new(-5, 2, 9);
        assert!(data.try_set_block(negative, make_tagged_lod_block(41, 0)));
        let original_buffer = data.get_block(negative, 0).unwrap().voxels() as *const VoxelBuffer;

        data.try_resize_lods_preserving(3).unwrap();

        assert_eq!(data.lod_count(), 3);
        let preserved = data.get_block(negative, 0).unwrap();
        assert_eq!(preserved.voxels() as *const VoxelBuffer, original_buffer);
        assert_eq!(
            preserved
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            41
        );
        assert!(data.try_set_block(Vector3i::new(7, -4, 5), make_tagged_lod_block(73, 2),));
    }

    #[test]
    fn preserving_lod_resize_equal_and_invalid_counts_are_non_mutating() {
        let mut data = VoxelData::new();
        let position = Vector3i::new(-1, -2, -3);
        assert!(data.try_set_block(position, make_tagged_lod_block(17, 0)));
        let original_buffer = data.get_block(position, 0).unwrap().voxels() as *const VoxelBuffer;

        data.try_resize_lods_preserving(1).unwrap();
        assert_eq!(
            data.get_block(position, 0).unwrap().voxels() as *const VoxelBuffer,
            original_buffer
        );

        for requested in [0, MAX_LOD, usize::MAX] {
            assert_eq!(
                data.try_resize_lods_preserving(requested),
                Err(VoxelDataLodResizeError::InvalidLodCount { requested })
            );
            assert_eq!(data.lod_count(), 1);
            assert_eq!(
                data.get_block(position, 0).unwrap().voxels() as *const VoxelBuffer,
                original_buffer
            );
        }
    }

    #[test]
    fn preserving_lod_resize_rejects_nonempty_truncation_without_data_loss() {
        let mut data = VoxelData::new();
        data.try_resize_lods_preserving(3).unwrap();
        let location = Vector3i::new(-5, 2, 9);
        assert!(data.try_set_block(location, make_tagged_lod_block(41, 2)));
        let original_buffer = data.get_block(location, 2).unwrap().voxels() as *const VoxelBuffer;

        assert_eq!(
            data.try_resize_lods_preserving(1),
            Err(VoxelDataLodResizeError::NonEmptyTruncatedLod { lod_index: 2 })
        );
        assert_eq!(data.lod_count(), 3);
        let preserved = data.get_block(location, 2).unwrap();
        assert_eq!(preserved.voxels() as *const VoxelBuffer, original_buffer);
        assert_eq!(
            preserved
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            41
        );
    }

    #[test]
    fn preserving_lod_resize_truncates_only_an_empty_suffix() {
        let mut data = VoxelData::new();
        data.try_resize_lods_preserving(4).unwrap();
        let location = Vector3i::new(3, -7, 11);
        assert!(data.try_set_block(location, make_tagged_lod_block(29, 1)));

        data.try_resize_lods_preserving(2).unwrap();

        assert_eq!(data.lod_count(), 2);
        assert_eq!(
            data.get_block(location, 1).unwrap().voxels().get_voxel(
                0,
                0,
                0,
                ChannelId::Type.index()
            ),
            29
        );
    }

    #[test]
    fn shared_settings_revision_advances_only_for_a_real_generator_change() {
        let shared = SharedVoxelData::new(VoxelData::new());
        assert_eq!(shared.settings_snapshot().revision, 0);
        let generator: SharedVoxelGenerator = Arc::new(RecordingGenerator::default());

        shared.set_generator(Some(generator.clone())).unwrap();
        assert_eq!(shared.settings_snapshot().revision, 1);
        shared.set_generator(Some(generator)).unwrap();
        assert_eq!(shared.settings_snapshot().revision, 1);
        shared.set_generator(None).unwrap();
        assert_eq!(shared.settings_snapshot().revision, 2);
    }

    #[test]
    fn shared_single_key_insert_advances_only_that_key_revision() {
        let shared = SharedVoxelData::new(VoxelData::new());
        let first = Vector3i::new(-3, 2, 9);
        let second = Vector3i::new(7, -4, 5);
        assert_eq!(shared.lod_structural_revision(0), Some(0));
        assert_eq!(
            shared.key_revision(first, 0),
            Some(VoxelDataKeyRevision::Tombstone(0))
        );

        assert!(shared
            .try_set_block(first, VoxelDataBlock::empty(0))
            .unwrap());
        assert_eq!(
            shared.key_revision(first, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert!(!shared
            .try_set_block(first, VoxelDataBlock::empty(0))
            .unwrap());
        assert_eq!(
            shared.key_revision(first, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );

        assert!(shared
            .try_set_block(second, VoxelDataBlock::empty(0))
            .unwrap());
        assert_eq!(
            shared.key_revision(first, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(
            shared.key_revision(second, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(shared.lod_structural_revision(0), Some(0));
    }

    #[test]
    fn shared_settings_writer_waits_for_the_mutation_gate() {
        let shared = Arc::new(SharedVoxelData::new(VoxelData::new()));
        let held = shared.begin_mutation().unwrap();
        let worker_data = shared.clone();
        let (sent, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let generator: SharedVoxelGenerator = Arc::new(RecordingGenerator::default());
            let result = worker_data.set_generator(Some(generator));
            sent.send(result).unwrap();
        });

        assert!(received.recv_timeout(Duration::from_millis(50)).is_err());
        drop(held);
        assert_eq!(
            received.recv_timeout(Duration::from_secs(1)).unwrap(),
            Ok(())
        );
        worker.join().unwrap();
        assert_eq!(shared.settings_snapshot().revision, 1);
    }

    #[test]
    fn shared_block_writer_waits_for_the_mutation_gate() {
        let shared = Arc::new(SharedVoxelData::new(VoxelData::new()));
        let held = shared.begin_mutation().unwrap();
        let worker_data = shared.clone();
        let position = Vector3i::new(4, -2, 7);
        let (sent, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            sent.send(worker_data.try_set_block(position, VoxelDataBlock::empty(0)))
                .unwrap();
        });

        assert!(received.recv_timeout(Duration::from_millis(50)).is_err());
        drop(held);
        assert_eq!(
            received.recv_timeout(Duration::from_secs(1)).unwrap(),
            Ok(true)
        );
        worker.join().unwrap();
        assert_eq!(
            shared.key_revision(position, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
    }

    #[test]
    fn shared_settings_revision_overflow_is_typed_and_non_mutating() {
        let shared = SharedVoxelData::new(VoxelData::new());
        shared.set_test_settings_revision(u64::MAX);
        let generator: SharedVoxelGenerator = Arc::new(RecordingGenerator::default());

        assert_eq!(
            shared.set_generator(Some(generator)),
            Err(SharedVoxelDataMutationError::SettingsRevisionOverflow)
        );
        let snapshot = shared.settings_snapshot();
        assert_eq!(snapshot.revision, u64::MAX);
        assert!(snapshot.generator.is_none());
    }

    #[test]
    fn shared_key_revision_overflow_rejects_insert_without_a_block() {
        let shared = SharedVoxelData::new(VoxelData::new());
        let position = Vector3i::new(-8, 3, 11);
        shared.with_lod_map_mut(0, |map| {
            map.set_key_revision_for_test(position, u64::MAX);
        });

        assert_eq!(
            shared.try_set_block(position, VoxelDataBlock::empty(0)),
            Err(SharedVoxelDataMutationError::KeyRevisionOverflow {
                position,
                lod_index: 0,
            })
        );
        assert!(!shared.with_lod_map(0, |map| map.has_block(position)));
        assert_eq!(
            shared.key_revision(position, 0),
            Some(VoxelDataKeyRevision::Tombstone(u64::MAX))
        );
        assert_eq!(shared.lod_structural_revision(0), Some(0));
    }

    #[test]
    fn shared_view_and_unview_advance_only_touched_key_revisions() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(128)));
        let shared = SharedVoxelData::new(data);
        let first = Vector3i::new(0, 0, 0);
        let second = Vector3i::new(1, 0, 0);
        let untouched = Vector3i::new(4, 0, 0);
        assert!(shared
            .try_set_block(first, VoxelDataBlock::empty(0))
            .unwrap());
        assert!(shared
            .try_set_block(second, VoxelDataBlock::empty(0))
            .unwrap());
        assert!(shared
            .try_set_block(untouched, VoxelDataBlock::empty(0))
            .unwrap());

        shared
            .view_area(
                Box3i::new(first, Vector3i::new(2, 1, 1)),
                0,
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            shared.key_revision(first, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
        assert_eq!(
            shared.key_revision(second, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
        assert_eq!(
            shared.key_revision(untouched, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );

        shared
            .unview_area(
                Box3i::new(first, Vector3i::new(2, 1, 1)),
                0,
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            shared.key_revision(first, 0),
            Some(VoxelDataKeyRevision::Tombstone(3))
        );
        assert_eq!(
            shared.key_revision(second, 0),
            Some(VoxelDataKeyRevision::Tombstone(3))
        );
        assert_eq!(
            shared.key_revision(untouched, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(shared.lod_structural_revision(0), Some(0));
    }

    #[test]
    fn shared_view_revision_overflow_rolls_back_the_entire_batch() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        let shared = SharedVoxelData::new(data);
        let first = Vector3i::new(0, 0, 0);
        let second = Vector3i::new(1, 0, 0);
        assert!(shared
            .try_set_block(first, VoxelDataBlock::empty(0))
            .unwrap());
        assert!(shared
            .try_set_block(second, VoxelDataBlock::empty(0))
            .unwrap());
        shared.with_lod_map_mut(0, |map| {
            map.set_key_revision_for_test(second, u64::MAX);
        });

        let mut found_positions = Vec::new();
        assert_eq!(
            shared.view_area(
                Box3i::new(first, Vector3i::new(2, 1, 1)),
                0,
                None,
                Some(&mut found_positions),
                None,
            ),
            Err(SharedVoxelDataMutationError::KeyRevisionOverflow {
                position: second,
                lod_index: 0,
            })
        );
        assert!(found_positions.is_empty());
        shared.with_lod_map(0, |map| {
            assert_eq!(map.get_block(first).unwrap().viewers.get(), 0);
            assert_eq!(map.get_block(second).unwrap().viewers.get(), 0);
        });
        assert_eq!(
            shared.key_revision(first, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(
            shared.key_revision(second, 0),
            Some(VoxelDataKeyRevision::Present(u64::MAX))
        );
    }

    #[test]
    fn shared_viewer_count_overflow_is_typed_and_non_mutating() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        let shared = SharedVoxelData::new(data);
        let position = Vector3i::zero();
        assert!(shared
            .try_set_block(position, VoxelDataBlock::empty(0))
            .unwrap());
        shared.with_lod_map_mut(0, |map| {
            map.get_block_mut(position)
                .unwrap()
                .viewers
                .set_exact(u32::MAX);
        });

        assert_eq!(
            shared.view_area(
                Box3i::new(position, Vector3i::splat(1)),
                0,
                None,
                None,
                None,
            ),
            Err(SharedVoxelDataMutationError::ViewerCountOverflow {
                position,
                lod_index: 0,
            })
        );
        assert_eq!(
            shared.with_lod_map(0, |map| map.get_block(position).unwrap().viewers.get()),
            u32::MAX
        );
        assert_eq!(
            shared.key_revision(position, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
    }

    #[test]
    fn shared_unview_revision_overflow_preserves_dirty_data_and_outputs() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        let shared = SharedVoxelData::new(data);
        let position = Vector3i::zero();
        assert!(shared
            .try_set_block(position, VoxelDataBlock::empty(0))
            .unwrap());
        shared.with_lod_map_mut(0, |map| {
            let block = map.get_block_mut(position).unwrap();
            block.viewers.set_exact(1);
            block.set_modified(true);
            map.set_key_revision_for_test(position, u64::MAX);
        });
        let mut removed = Vec::new();
        let mut saves = Vec::new();

        assert_eq!(
            shared.unview_area(
                Box3i::new(position, Vector3i::splat(1)),
                0,
                Some(&mut removed),
                None,
                Some(&mut saves),
            ),
            Err(SharedVoxelDataMutationError::KeyRevisionOverflow {
                position,
                lod_index: 0,
            })
        );
        assert!(removed.is_empty());
        assert!(saves.is_empty());
        shared.with_lod_map(0, |map| {
            let block = map.get_block(position).unwrap();
            assert_eq!(block.viewers.get(), 1);
            assert!(block.is_modified());
        });
        assert_eq!(
            shared.key_revision(position, 0),
            Some(VoxelDataKeyRevision::Present(u64::MAX))
        );
    }

    #[test]
    fn shared_unview_viewer_underflow_is_typed_and_non_mutating() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        let shared = SharedVoxelData::new(data);
        let position = Vector3i::zero();
        assert!(shared
            .try_set_block(position, VoxelDataBlock::empty(0))
            .unwrap());
        let mut removed = Vec::new();

        assert_eq!(
            shared.unview_area(
                Box3i::new(position, Vector3i::splat(1)),
                0,
                Some(&mut removed),
                None,
                None,
            ),
            Err(SharedVoxelDataMutationError::ViewerCountUnderflow {
                position,
                lod_index: 0,
            })
        );
        assert!(removed.is_empty());
        assert!(shared.with_lod_map(0, |map| map.has_block(position)));
        assert_eq!(
            shared.key_revision(position, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
    }

    #[test]
    fn shared_paging_writers_wait_for_overlapping_spatial_readers() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        let shared = Arc::new(SharedVoxelData::new(data));
        let block_pos = Vector3i::zero();
        assert!(shared
            .try_set_block(block_pos, VoxelDataBlock::empty(0))
            .unwrap());
        let (phase_sent, phase_received) = std::sync::mpsc::channel();
        shared.set_test_edit_phase_hook(Arc::new(move |phase| {
            if phase == SharedVoxelDataEditPhase::MutationGateAcquiredBeforeSpatialWrite {
                phase_sent.send(()).unwrap();
            }
        }));
        let voxel_box = Box3i::new(Vector3i::zero(), Vector3i::splat(16));
        let read = shared.read_region(0, voxel_box);

        let (view_sent, view_received) = std::sync::mpsc::channel();
        let view_data = shared.clone();
        let view_worker = std::thread::spawn(move || {
            let result = view_data.view_area(
                Box3i::new(block_pos, Vector3i::splat(1)),
                0,
                None,
                None,
                None,
            );
            view_sent.send(result).unwrap();
        });
        phase_received
            .recv_timeout(Duration::from_secs(1))
            .expect("view writer did not acquire the mutation gate");
        assert!(matches!(
            view_received.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        let (settings_sent, settings_received) = std::sync::mpsc::channel();
        let settings_data = shared.clone();
        let settings_worker = std::thread::spawn(move || {
            let generator: SharedVoxelGenerator = Arc::new(RecordingGenerator::default());
            settings_sent
                .send(settings_data.set_generator(Some(generator)))
                .unwrap();
        });
        drop(read);
        assert_eq!(
            view_received.recv_timeout(Duration::from_secs(1)).unwrap(),
            Ok(())
        );
        view_worker.join().unwrap();
        assert_eq!(
            settings_received
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            Ok(())
        );
        settings_worker.join().unwrap();

        shared
            .unview_area(
                Box3i::new(block_pos, Vector3i::splat(1)),
                0,
                None,
                None,
                None,
            )
            .unwrap();
        phase_received
            .recv_timeout(Duration::from_secs(1))
            .expect("unview writer did not acquire the mutation gate");
        let read = shared.read_region(0, voxel_box);
        let (insert_sent, insert_received) = std::sync::mpsc::channel();
        let insert_data = shared.clone();
        let insert_worker = std::thread::spawn(move || {
            insert_sent
                .send(insert_data.try_set_block(block_pos, VoxelDataBlock::empty(0)))
                .unwrap();
        });
        phase_received
            .recv_timeout(Duration::from_secs(1))
            .expect("insert writer did not acquire the mutation gate");
        assert!(matches!(
            insert_received.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        drop(read);
        assert_eq!(
            insert_received
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            Ok(true)
        );
        insert_worker.join().unwrap();
    }

    #[test]
    fn set_format_resets_maps_and_configures_new_blocks() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        assert!(data.try_set_voxel(1, Vector3i::zero(), ChannelId::Type.index()));
        assert_eq!(data.block_count(), 1);

        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        data.set_format(format);

        assert_eq!(data.block_count(), 0);
        assert_eq!(data.format(), format);
        assert!(data.try_set_voxel_f(-3.25, Vector3i::zero(), ChannelId::Sdf.index()));
        let block = data.get_block(Vector3i::zero(), 0).unwrap();
        assert_eq!(
            block.voxels().channel_depth(ChannelId::Sdf.index()),
            ChannelDepth::Bit32
        );
    }

    #[test]
    fn try_set_voxel_requires_bounds_and_known_loaded_data() {
        let mut data = VoxelData::new();
        let channel = ChannelId::Type.index();
        let inside = Vector3i::new(1, 1, 1);
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::new(4, 4, 4)));

        assert!(data.is_streaming_enabled());
        assert!(!data.try_set_voxel(5, inside, channel));
        assert_eq!(data.get_voxel(inside, channel, 99), 99);

        data.set_full_load_completed(true);
        assert!(!data.try_set_voxel(5, inside, channel));

        data.set_streaming_enabled(false);

        assert!(data.try_set_voxel(5, inside, channel));
        assert_eq!(data.get_voxel(inside, channel, 99), 5);
        assert!(!data.try_set_voxel(6, Vector3i::new(8, 1, 1), channel));
        assert_eq!(data.get_voxel(Vector3i::new(8, 1, 1), channel, 99), 99);
    }

    #[test]
    fn try_set_block_inserts_once_and_tracks_lod() {
        let mut data = VoxelData::new();
        data.set_lod_count(2).unwrap();
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(data.block_size() as i32));
        voxels.set_voxel(7, 0, 0, 0, ChannelId::Type.index());
        let block = VoxelDataBlock::with_voxels(voxels, 1);
        let block_pos = Vector3i::new(3, 0, -2);

        assert!(data.try_set_block(block_pos, block));
        assert!(data.has_block(block_pos, 1));
        assert_eq!(data.block_count(), 1);

        let duplicate = VoxelDataBlock::empty(1);
        assert!(!data.try_set_block(block_pos, duplicate));
        assert_eq!(data.block_count(), 1);
    }

    #[test]
    fn streaming_try_set_voxel_requires_existing_block() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::new(32, 16, 16)));
        let channel = ChannelId::Type.index();
        let pos = Vector3i::new(1, 1, 1);

        assert!(!data.try_set_voxel(3, pos, channel));

        let voxels = VoxelBuffer::with_size(Vector3i::splat(data.block_size() as i32));
        assert!(data.try_set_block(Vector3i::zero(), VoxelDataBlock::with_voxels(voxels, 0)));

        assert!(data.try_set_voxel(3, pos, channel));
        assert_eq!(data.get_voxel(pos, channel, 99), 3);
        assert!(data.try_get_block_voxels(Vector3i::zero()).is_some());
    }

    #[test]
    fn mark_area_modified_sets_block_flags_once() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::new(64, 16, 16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        assert!(data.try_set_voxel(1, Vector3i::new(1, 1, 1), ChannelId::Type.index()));
        assert!(data.try_set_voxel(2, Vector3i::new(20, 1, 1), ChannelId::Type.index()));

        let changed = data.mark_area_modified(
            Box3i::new(Vector3i::zero(), Vector3i::new(32, 16, 16)),
            true,
        );

        assert_eq!(
            changed,
            vec![Vector3i::new(0, 0, 0), Vector3i::new(1, 0, 0)]
        );
        for block_pos in changed {
            let block = data.get_block(block_pos, 0).unwrap();
            assert!(block.is_modified());
            assert!(block.is_edited());
            assert!(block.needs_lodding());
        }

        let second = data.mark_area_modified(
            Box3i::new(Vector3i::zero(), Vector3i::new(32, 16, 16)),
            true,
        );
        assert!(second.is_empty());
    }

    #[test]
    fn pre_generate_box_non_streaming_generates_missing_lod_blocks() {
        let mut data = VoxelData::new();
        data.set_lod_count(2).unwrap();
        data.set_streaming_enabled(false);
        let generator = RecordingGenerator::default();

        let generated = data.pre_generate_box(
            Box3i::new(Vector3i::zero(), Vector3i::new(32, 16, 16)),
            Some(&generator),
        );

        assert_eq!(generated, 3);
        assert_eq!(
            *generator.calls.lock().unwrap(),
            vec![
                (Vector3i::new(0, 0, 0), 0),
                (Vector3i::new(16, 0, 0), 0),
                (Vector3i::new(0, 0, 0), 1),
            ]
        );
        assert_eq!(
            data.get_block(Vector3i::new(1, 0, 0), 0)
                .unwrap()
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            26
        );
        assert_eq!(
            data.get_block(Vector3i::zero(), 1)
                .unwrap()
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            11
        );
    }

    #[test]
    fn pre_generate_box_streaming_only_fills_existing_empty_blocks() {
        let mut data = VoxelData::new();
        let block_pos = Vector3i::zero();
        assert!(data.try_set_block(block_pos, VoxelDataBlock::empty(0)));
        let generator = RecordingGenerator::default();

        let generated = data.pre_generate_box(
            Box3i::new(Vector3i::zero(), Vector3i::new(32, 16, 16)),
            Some(&generator),
        );

        assert_eq!(generated, 1);
        assert!(data.try_get_block_voxels(block_pos).is_some());
        assert!(!data.has_block(Vector3i::new(1, 0, 0), 0));
    }

    #[test]
    fn consume_block_modifications_copies_voxels_and_clears_modified_flag() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::new(16, 16, 16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let channel = ChannelId::Type.index();
        assert!(data.try_set_voxel(7, Vector3i::new(1, 1, 1), channel));
        data.mark_area_modified(
            Box3i::new(Vector3i::zero(), Vector3i::new(16, 16, 16)),
            false,
        );

        let mut save = data
            .consume_block_modifications(Vector3i::zero())
            .expect("modified block should be consumed");

        assert_eq!(save.position, Vector3i::zero());
        assert_eq!(save.lod_index, 0);
        assert_eq!(save.voxels.as_ref().unwrap().get_voxel(1, 1, 1, channel), 7);
        save.voxels.as_mut().unwrap().set_voxel(9, 1, 1, 1, channel);
        assert_eq!(data.get_voxel(Vector3i::new(1, 1, 1), channel, 99), 7);
        assert!(!data.get_block(Vector3i::zero(), 0).unwrap().is_modified());
        assert!(data.consume_block_modifications(Vector3i::zero()).is_none());
    }

    #[test]
    fn consume_all_modifications_collects_all_lods() {
        let mut data = VoxelData::new();
        data.set_lod_count(2).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::new(16, 16, 16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        assert!(data.try_set_voxel(3, Vector3i::new(1, 1, 1), ChannelId::Type.index()));
        data.mark_area_modified(
            Box3i::new(Vector3i::zero(), Vector3i::new(16, 16, 16)),
            false,
        );

        let mut lod1_voxels = VoxelBuffer::with_size(Vector3i::splat(data.block_size() as i32));
        lod1_voxels.set_voxel(4, 0, 0, 0, ChannelId::Type.index());
        let mut lod1_block = VoxelDataBlock::with_voxels(lod1_voxels, 1);
        lod1_block.set_modified(true);
        assert!(data.try_set_block(Vector3i::new(2, 0, 0), lod1_block));

        let saves = data.consume_all_modifications();

        assert_eq!(saves.len(), 2);
        assert!(saves
            .iter()
            .any(|save| save.position == Vector3i::zero() && save.lod_index == 0));
        assert!(saves
            .iter()
            .any(|save| save.position == Vector3i::new(2, 0, 0) && save.lod_index == 1));
        assert!(!data.get_block(Vector3i::zero(), 0).unwrap().is_modified());
        assert!(!data
            .get_block(Vector3i::new(2, 0, 0), 1)
            .unwrap()
            .is_modified());
    }

    #[test]
    fn unload_blocks_removes_blocks_and_returns_modified_voxels_to_save() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::new(32, 16, 16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        assert!(data.try_set_voxel(5, Vector3i::new(1, 1, 1), ChannelId::Type.index()));
        assert!(data.try_set_voxel(6, Vector3i::new(20, 1, 1), ChannelId::Type.index()));
        data.mark_area_modified(
            Box3i::new(Vector3i::zero(), Vector3i::new(16, 16, 16)),
            false,
        );

        let saves = data.unload_blocks(
            Box3i::new(Vector3i::zero(), Vector3i::new(2, 1, 1)),
            0,
            true,
        );

        assert_eq!(saves.len(), 1);
        assert_eq!(saves[0].position, Vector3i::zero());
        assert!(saves[0].voxels.is_some());
        assert!(!data.has_block(Vector3i::zero(), 0));
        assert!(!data.has_block(Vector3i::new(1, 0, 0), 0));
    }

    #[test]
    fn update_lods_clears_needs_lodding_and_reports_touched_blocks() {
        let mut data = VoxelData::new();
        data.set_lod_count(2).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);

        // Two LOD0 blocks need LOD updates.
        let channel = ChannelId::Type.index();
        assert!(data.try_set_voxel(1, Vector3i::new(1, 1, 1), channel));
        assert!(data.try_set_voxel(2, Vector3i::new(20, 1, 1), channel));
        let modified = data.mark_area_modified(
            Box3i::new(Vector3i::zero(), Vector3i::new(32, 16, 16)),
            true,
        );
        assert_eq!(modified.len(), 2);

        let mut updated = Vec::new();
        data.update_lods(&modified, None, Some(&mut updated));

        // LOD0 blocks: needs_lodding cleared and reported.
        for &lod0_pos in &modified {
            assert!(!data.get_block(lod0_pos, 0).unwrap().needs_lodding());
        }
        // Both LOD0 positions map to the same LOD1 block (0,0,0).
        assert!(updated.contains(&BlockLocation {
            position: Vector3i::zero(),
            lod_index: 0,
        }));
        assert!(updated.contains(&BlockLocation {
            position: Vector3i::new(1, 0, 0),
            lod_index: 0,
        }));
        assert!(updated.contains(&BlockLocation {
            position: Vector3i::zero(),
            lod_index: 1,
        }));
        // The destination LOD1 block is now modified.
        assert!(data.get_block(Vector3i::zero(), 1).unwrap().is_modified());
    }

    #[test]
    fn update_lods_downscales_lod0_edits_into_lod1_octants() {
        let mut data = VoxelData::new();
        data.set_lod_count(2).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let channel = ChannelId::Type.index();

        // Edit a single LOD0 voxel inside block (1,0,0). This block maps to
        // the +X octant of LOD1 block (0,0,0). Local coords (4,4,6) are chosen
        // so the 2:1 nearest-neighbor sample lands at LOD1 (10,2,3).
        let edited_pos = Vector3i::new(20, 4, 6);
        assert!(data.try_set_voxel(7, edited_pos, channel));
        let modified = data.mark_area_modified(
            Box3i::new(edited_pos, edited_pos + Vector3i::splat(1)),
            true,
        );
        assert_eq!(modified, vec![Vector3i::new(1, 0, 0)]);

        // Pre-create the destination LOD1 block so downscaling lands in it
        // (matches the streaming-pyramid invariant that parents are resident).
        let lod1_voxels = VoxelBuffer::with_size(Vector3i::splat(data.block_size() as i32));
        assert!(data.try_set_block(
            Vector3i::zero(),
            VoxelDataBlock::with_voxels(lod1_voxels, 1),
        ));

        data.update_lods(&modified, None, None);

        // The edited LOD0 voxel (20,4,6) maps to LOD1 (10,2,3) via 2:1 nearest.
        // In LOD1 block-local coords (block_size 16) that is (10,2,3).
        let lod1_block = data.get_block(Vector3i::zero(), 1).unwrap();
        assert_eq!(lod1_block.voxels().get_voxel(10, 2, 3, channel), 7);
        // A voxel outside the downscaled octant stays at the default.
        assert_eq!(lod1_block.voxels().get_voxel(0, 0, 0, channel), 0);
    }

    #[test]
    fn update_lods_generates_missing_destination_in_non_streaming_mode() {
        let mut data = VoxelData::new();
        data.set_lod_count(2).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let channel = ChannelId::Type.index();

        assert!(data.try_set_voxel(11, Vector3i::new(1, 1, 1), channel));
        let modified = data.mark_area_modified(
            Box3i::new(Vector3i::zero(), Vector3i::new(16, 16, 16)),
            true,
        );

        // The destination LOD1 block doesn't exist; the generator must fill it
        // before the downscale runs. The recorder lets us observe the call.
        let generator = RecordingGenerator::default();
        data.update_lods(&modified, Some(&generator), None);

        // LOD1 block (0,0,0) was generated on demand and is now present.
        assert!(data.has_block(Vector3i::zero(), 1));
        assert!(generator
            .calls
            .lock()
            .unwrap()
            .iter()
            .any(|(origin, lod)| { *lod == 1 && origin.x == 0 && origin.y == 0 && origin.z == 0 }));
    }

    #[test]
    fn update_lods_rearms_intermediate_blocks_for_repeated_edit_cascades() {
        let mut data = VoxelData::new();
        data.set_lod_count(3).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(64)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let channel = ChannelId::Type.index();
        let first_voxel = Vector3i::zero();
        let sibling_voxel = Vector3i::new(data.block_size() as i32, 0, 0);
        let edited_area = Box3i::new(Vector3i::zero(), Vector3i::new(32, 1, 1));

        assert!(data.try_set_voxel(11, first_voxel, channel));
        assert!(data.try_set_voxel(12, sibling_voxel, channel));
        let modified = data.mark_area_modified(edited_area, true);
        assert_eq!(modified, vec![Vector3i::zero(), Vector3i::new(1, 0, 0)]);

        let mut first_updated = Vec::new();
        data.update_lods(&modified, None, Some(&mut first_updated));
        assert_eq!(
            first_updated
                .iter()
                .filter(|location| location.lod_index == 2)
                .count(),
            1,
            "sibling LOD0 blocks sharing one LOD1 parent must enqueue that parent once"
        );
        assert_eq!(
            data.get_block(Vector3i::zero(), 2)
                .unwrap()
                .voxels()
                .get_voxel(0, 0, 0, channel),
            11
        );
        assert_eq!(
            data.get_block(Vector3i::zero(), 2)
                .unwrap()
                .voxels()
                .get_voxel(4, 0, 0, channel),
            12
        );

        assert!(data.try_set_voxel(21, first_voxel, channel));
        assert!(data.try_set_voxel(22, sibling_voxel, channel));
        let modified = data.mark_area_modified(edited_area, true);
        assert_eq!(modified, vec![Vector3i::zero(), Vector3i::new(1, 0, 0)]);

        let mut second_updated = Vec::new();
        data.update_lods(&modified, None, Some(&mut second_updated));
        assert_eq!(
            second_updated
                .iter()
                .filter(|location| location.lod_index == 2)
                .count(),
            1,
            "a later edit must re-enqueue the already-used LOD1 parent exactly once"
        );
        let lod2 = data.get_block(Vector3i::zero(), 2).unwrap();
        assert_eq!(lod2.voxels().get_voxel(0, 0, 0, channel), 21);
        assert_eq!(lod2.voxels().get_voxel(4, 0, 0, channel), 22);
        assert!(
            !data.get_block(Vector3i::zero(), 1).unwrap().needs_lodding(),
            "processed intermediate blocks must be ready for the next cascade"
        );
    }

    #[test]
    fn view_area_increments_viewers_and_reports_found_and_missing_blocks() {
        let mut data = VoxelData::new();
        // Bounds cover a 4×4×4 block region so view queries can probe blocks
        // that exist alongside ones that don't, without being clipped out.
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(64)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let channel = ChannelId::Type.index();

        // Two loaded blocks; (2,0,0) is left empty within the queried area.
        assert!(data.try_set_voxel(1, Vector3i::new(1, 1, 1), channel));
        assert!(data.try_set_voxel(2, Vector3i::new(20, 1, 1), channel));

        let mut missing = Vec::new();
        let mut found_positions = Vec::new();
        let mut found_blocks: Vec<VoxelDataBlock> = Vec::new();
        data.view_area(
            Box3i::new(Vector3i::zero(), Vector3i::new(3, 1, 1)),
            0,
            Some(&mut missing),
            Some(&mut found_positions),
            Some(&mut found_blocks),
        );

        assert_eq!(
            found_positions,
            vec![Vector3i::zero(), Vector3i::new(1, 0, 0)]
        );
        assert_eq!(missing, vec![Vector3i::new(2, 0, 0)]);
        assert_eq!(found_blocks.len(), 2);
        // Viewers were incremented on the live blocks.
        assert_eq!(
            data.get_block(Vector3i::zero(), 0).unwrap().viewers.get(),
            1
        );
        assert_eq!(
            data.get_block(Vector3i::new(1, 0, 0), 0)
                .unwrap()
                .viewers
                .get(),
            1
        );
    }

    #[test]
    fn owned_view_area_uses_lod_scaled_bounds() {
        let mut data = VoxelData::new();
        data.set_lod_count(3).unwrap();
        data.set_bounds(Box3i::new(
            Vector3i::new(1024, 0, 0),
            Vector3i::new(256, 256, 256),
        ));

        let lod_index = 2;
        let edge_position = Vector3i::new(19, 0, 0);
        let old_base_size_bounds = data.bounds().downscaled(data.block_size() as i32);
        assert!(!old_base_size_bounds.contains_point(edge_position));
        assert!(data.try_set_block(edge_position, VoxelDataBlock::empty(lod_index)));

        data.view_area(
            Box3i::new(edge_position, Vector3i::splat(1)),
            lod_index as usize,
            None,
            None,
            None,
        );
        assert_eq!(
            data.get_block(edge_position, lod_index as usize)
                .unwrap()
                .viewers
                .get(),
            1
        );

        let mut removed = Vec::new();
        data.unview_area(
            Box3i::new(edge_position, Vector3i::splat(1)),
            lod_index as usize,
            Some(&mut removed),
            None,
            None,
        );
        assert_eq!(removed, vec![edge_position]);
        assert!(!data.has_block(edge_position, lod_index as usize));
    }

    #[test]
    fn shared_view_area_uses_lod_scaled_bounds() {
        let mut data = VoxelData::new();
        data.set_lod_count(3).unwrap();
        data.set_bounds(Box3i::new(
            Vector3i::new(1024, 0, 0),
            Vector3i::new(256, 256, 256),
        ));
        let shared = SharedVoxelData::new(data);

        let lod_index = 2;
        let edge_position = Vector3i::new(19, 0, 0);
        let old_base_size_bounds = shared.bounds().downscaled(shared.block_size() as i32);
        assert!(!old_base_size_bounds.contains_point(edge_position));
        assert!(shared
            .try_set_block(edge_position, VoxelDataBlock::empty(lod_index))
            .unwrap());

        shared
            .view_area(
                Box3i::new(edge_position, Vector3i::splat(1)),
                lod_index as usize,
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            shared.with_lod_map(lod_index as usize, |map| map
                .get_block(edge_position)
                .unwrap()
                .viewers
                .get()),
            1
        );

        let mut removed = Vec::new();
        shared
            .unview_area(
                Box3i::new(edge_position, Vector3i::splat(1)),
                lod_index as usize,
                Some(&mut removed),
                None,
                None,
            )
            .unwrap();
        assert_eq!(removed, vec![edge_position]);
        assert!(!shared.with_lod_map(lod_index as usize, |map| map.has_block(edge_position)));
    }

    #[test]
    fn view_and_unview_reject_unrepresentable_lod_index_without_wrapping() {
        let bounds = Box3i::new(Vector3i::zero(), Vector3i::splat(64));
        let requested = Box3i::new(Vector3i::zero(), Vector3i::new(2, 1, 1));
        let expected: Vec<_> = requested.iter_cells_zxy().collect();
        let invalid_lod_index = usize::from(u8::MAX) + 1;

        let mut owned = VoxelData::new();
        owned.set_bounds(bounds);
        let mut missing = Vec::new();
        owned.view_area(requested, invalid_lod_index, Some(&mut missing), None, None);
        assert_eq!(missing, expected);
        missing.clear();
        owned.unview_area(requested, invalid_lod_index, None, Some(&mut missing), None);
        assert_eq!(missing, expected);

        let mut data = VoxelData::new();
        data.set_bounds(bounds);
        let shared = SharedVoxelData::new(data);
        missing.clear();
        shared
            .view_area(requested, invalid_lod_index, Some(&mut missing), None, None)
            .unwrap();
        assert_eq!(missing, expected);
        missing.clear();
        shared
            .unview_area(requested, invalid_lod_index, None, Some(&mut missing), None)
            .unwrap();
        assert_eq!(missing, expected);
    }

    #[test]
    fn view_and_unview_handle_unrepresentable_request_without_panicking() {
        let bounds = Box3i::new(Vector3i::zero(), Vector3i::splat(64));
        let requested = Box3i::new(Vector3i::splat(i32::MAX), Vector3i::splat(1));

        let mut owned = VoxelData::new();
        owned.set_bounds(bounds);
        let mut missing = Vec::new();
        owned.view_area(requested, 0, Some(&mut missing), None, None);
        assert!(missing.is_empty());
        owned.unview_area(requested, 0, None, Some(&mut missing), None);
        assert!(missing.is_empty());

        let mut data = VoxelData::new();
        data.set_bounds(bounds);
        let shared = SharedVoxelData::new(data);
        shared
            .view_area(requested, 0, Some(&mut missing), None, None)
            .unwrap();
        assert!(missing.is_empty());
        shared
            .unview_area(requested, 0, None, Some(&mut missing), None)
            .unwrap();
        assert!(missing.is_empty());
    }

    #[test]
    fn unview_area_releases_viewers_and_removes_blocks_reaching_zero() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let channel = ChannelId::Type.index();

        // Block A is unmodified; block B is modified and should be returned
        // for saving when it is unloaded by the unview.
        assert!(data.try_set_voxel(1, Vector3i::new(1, 1, 1), channel));
        assert!(data.try_set_voxel(2, Vector3i::new(20, 1, 1), channel));
        data.mark_area_modified(
            Box3i::new(Vector3i::new(16, 0, 0), Vector3i::new(32, 16, 16)),
            false,
        );

        // Pin both blocks, then release them.
        data.view_area(
            Box3i::new(Vector3i::zero(), Vector3i::new(2, 1, 1)),
            0,
            None,
            None,
            None,
        );
        let mut removed = Vec::new();
        let mut saves = Vec::new();
        data.unview_area(
            Box3i::new(Vector3i::zero(), Vector3i::new(2, 1, 1)),
            0,
            Some(&mut removed),
            None,
            Some(&mut saves),
        );

        assert_eq!(removed, vec![Vector3i::zero(), Vector3i::new(1, 0, 0)]);
        assert!(!data.has_block(Vector3i::zero(), 0));
        assert!(!data.has_block(Vector3i::new(1, 0, 0), 0));
        assert_eq!(saves.len(), 1);
        assert_eq!(saves[0].position, Vector3i::new(1, 0, 0));
        assert!(saves[0].voxels.is_some());
    }

    #[test]
    fn unview_area_keeps_blocks_with_remaining_viewers() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let channel = ChannelId::Type.index();
        assert!(data.try_set_voxel(1, Vector3i::new(1, 1, 1), channel));

        // View the same block twice; a single unview should leave it pinned.
        data.view_area(
            Box3i::new(Vector3i::zero(), Vector3i::splat(1)),
            0,
            None,
            None,
            None,
        );
        data.view_area(
            Box3i::new(Vector3i::zero(), Vector3i::splat(1)),
            0,
            None,
            None,
            None,
        );
        assert_eq!(
            data.get_block(Vector3i::zero(), 0).unwrap().viewers.get(),
            2
        );

        let mut removed = Vec::new();
        data.unview_area(
            Box3i::new(Vector3i::zero(), Vector3i::splat(1)),
            0,
            Some(&mut removed),
            None,
            None,
        );

        assert!(removed.is_empty());
        assert!(data.has_block(Vector3i::zero(), 0));
        assert_eq!(
            data.get_block(Vector3i::zero(), 0).unwrap().viewers.get(),
            1
        );
    }

    #[test]
    fn set_generator_attaches_a_shared_handle_round_trippable_via_with_generator() {
        let mut data = VoxelData::new();
        assert!(!data.has_generator());

        let generator: SharedVoxelGenerator = Arc::new(RecordingGenerator::default());
        data.set_generator(Some(generator.clone()));
        assert!(data.has_generator());

        let mut probed = Vec::new();
        data.with_generator(|gen| {
            // Touch the generator under its lock; the recorder stores nothing
            // externally but we can confirm it runs.
            let _ = gen.used_channels_mask();
            probed.push(());
        });
        assert_eq!(probed.len(), 1);
        assert!(Arc::ptr_eq(data.generator().as_ref().unwrap(), &generator));
    }

    #[test]
    fn shared_voxel_data_region_locks_follow_voxel_data_contract() {
        let shared = SharedVoxelData::new(VoxelData::new());
        let area = Box3i::new(Vector3i::zero(), Vector3i::splat(16));
        let overlap = Box3i::new(Vector3i::splat(8), Vector3i::splat(16));
        let disjoint = Box3i::new(Vector3i::splat(64), Vector3i::splat(16));

        let read = shared.read_region(0, area);
        let overlap_read = shared
            .try_read_region(0, overlap)
            .expect("overlapping mesh/read regions may coexist");
        assert!(
            shared.try_write_region(0, overlap).is_none(),
            "overlapping edit/write region must wait for readers"
        );
        let disjoint_write = shared
            .try_write_region(0, disjoint)
            .expect("disjoint edit/write region can proceed");
        assert_eq!(shared.locked_region_count(0), 3);

        drop(disjoint_write);
        drop(overlap_read);
        drop(read);

        let write = shared
            .try_write_region(0, overlap)
            .expect("write should acquire after readers drop");
        assert_eq!(shared.locked_region_count(0), 1);
        drop(write);
        assert_eq!(shared.locked_region_count(0), 0);
    }

    #[test]
    fn shared_edit_voxel_materializes_procedural_block_and_marks_it_dirty() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        data.set_generator(Some(Arc::new(RecordingGenerator::default())));
        let shared = SharedVoxelData::new(data);
        let channel = ChannelId::Type.index();

        assert!(shared.try_edit_voxel(99, Vector3i::new(1, 1, 1), channel));

        let block = shared.block_snapshot(Vector3i::zero(), 0).unwrap();
        assert_eq!(block.voxels().get_voxel(1, 1, 1, channel), 99);
        assert_eq!(block.voxels().get_voxel(2, 1, 1, channel), 10);
        assert!(block.is_modified());
        assert!(block.is_edited());
        assert_eq!(
            shared.key_revision(Vector3i::zero(), 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
    }

    #[test]
    fn shared_update_lods_rearms_intermediate_blocks_for_repeated_edit_cascades() {
        let mut data = VoxelData::new();
        data.set_lod_count(3).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(64)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = SharedVoxelData::new(data);
        let channel = ChannelId::Type.index();
        let first_voxel = Vector3i::zero();
        let sibling_voxel = Vector3i::new(shared.block_size() as i32, 0, 0);
        let edited_area = Box3i::new(Vector3i::zero(), Vector3i::new(32, 1, 1));

        assert!(shared
            .try_set_voxel_checked(11, first_voxel, channel)
            .unwrap());
        assert!(shared
            .try_set_voxel_checked(12, sibling_voxel, channel)
            .unwrap());
        let modified = shared
            .mark_area_modified_checked(edited_area, true)
            .unwrap();
        assert_eq!(modified, vec![Vector3i::zero(), Vector3i::new(1, 0, 0)]);

        let first_updated = shared
            .update_lods_from_lod0_blocks_checked(&modified)
            .unwrap();
        assert_eq!(
            first_updated
                .iter()
                .filter(|location| location.lod_index == 2)
                .count(),
            1,
            "sibling LOD0 blocks sharing one LOD1 parent must enqueue that parent once"
        );
        let lod2 = shared.block_snapshot(Vector3i::zero(), 2).unwrap();
        assert_eq!(lod2.voxels().get_voxel(0, 0, 0, channel), 11);
        assert_eq!(lod2.voxels().get_voxel(4, 0, 0, channel), 12);

        assert!(shared
            .try_set_voxel_checked(21, first_voxel, channel)
            .unwrap());
        assert!(shared
            .try_set_voxel_checked(22, sibling_voxel, channel)
            .unwrap());
        let modified = shared
            .mark_area_modified_checked(edited_area, true)
            .unwrap();
        assert_eq!(modified, vec![Vector3i::zero(), Vector3i::new(1, 0, 0)]);

        let second_updated = shared
            .update_lods_from_lod0_blocks_checked(&modified)
            .unwrap();
        assert_eq!(
            second_updated
                .iter()
                .filter(|location| location.lod_index == 2)
                .count(),
            1,
            "a later edit must re-enqueue the already-used LOD1 parent exactly once"
        );
        let lod2 = shared.block_snapshot(Vector3i::zero(), 2).unwrap();
        assert_eq!(lod2.voxels().get_voxel(0, 0, 0, channel), 21);
        assert_eq!(lod2.voxels().get_voxel(4, 0, 0, channel), 22);
        assert!(
            !shared
                .block_snapshot(Vector3i::zero(), 1)
                .unwrap()
                .needs_lodding(),
            "processed intermediate blocks must be ready for the next cascade"
        );
    }

    #[test]
    fn shared_checked_set_voxel_advances_only_its_key_revision() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = SharedVoxelData::new(data);
        let channel = ChannelId::Type.index();
        let first = Vector3i::new(1, 1, 1);
        let second_block = Vector3i::new(1, 0, 0);

        assert!(shared.try_set_voxel_checked(7, first, channel).unwrap());
        assert_eq!(
            shared.key_revision(Vector3i::zero(), 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(
            shared.key_revision(second_block, 0),
            Some(VoxelDataKeyRevision::Tombstone(0))
        );

        assert!(shared.try_set_voxel_checked(8, first, channel).unwrap());
        assert_eq!(
            shared.key_revision(Vector3i::zero(), 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
        assert_eq!(
            shared.try_set_voxel_checked(9, first, crate::storage::voxel_buffer::MAX_CHANNELS),
            Err(SharedVoxelDataMutationError::InvalidChannel {
                channel_index: crate::storage::voxel_buffer::MAX_CHANNELS,
            })
        );
        assert_eq!(
            shared.key_revision(Vector3i::zero(), 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
    }

    #[test]
    fn shared_checked_set_voxel_revision_overflow_preserves_an_empty_block() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        data.set_streaming_enabled(true);
        let shared = SharedVoxelData::new(data);
        let position = Vector3i::zero();
        assert!(shared
            .try_set_block(position, VoxelDataBlock::empty(0))
            .unwrap());
        shared.with_lod_map_mut(0, |map| {
            map.set_key_revision_for_test(position, u64::MAX);
        });

        assert_eq!(
            shared.try_set_voxel_checked(7, Vector3i::new(1, 1, 1), ChannelId::Type.index(),),
            Err(SharedVoxelDataMutationError::KeyRevisionOverflow {
                position,
                lod_index: 0,
            })
        );
        assert!(!shared.block_snapshot(position, 0).unwrap().has_voxels());
    }

    #[test]
    fn shared_mark_area_revision_overflow_rolls_back_the_complete_batch() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::new(32, 16, 16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = SharedVoxelData::new(data);
        let channel = ChannelId::Type.index();
        let first = Vector3i::zero();
        let second = Vector3i::new(1, 0, 0);

        assert!(shared
            .try_set_voxel_checked(1, Vector3i::new(1, 1, 1), channel)
            .unwrap());
        assert!(shared
            .try_set_voxel_checked(2, Vector3i::new(17, 1, 1), channel)
            .unwrap());
        shared.with_lod_map_mut(0, |map| {
            map.set_key_revision_for_test(second, u64::MAX);
        });

        assert_eq!(
            shared.mark_area_modified_checked(
                Box3i::new(Vector3i::zero(), Vector3i::new(32, 1, 1)),
                true,
            ),
            Err(SharedVoxelDataMutationError::KeyRevisionOverflow {
                position: second,
                lod_index: 0,
            })
        );
        for position in [first, second] {
            let block = shared.block_snapshot(position, 0).unwrap();
            assert!(!block.is_modified());
            assert!(!block.is_edited());
            assert!(!block.needs_lodding());
        }
        assert_eq!(
            shared.key_revision(first, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
    }

    #[test]
    fn shared_mark_area_rejects_overflowing_bounds_without_mutation() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = SharedVoxelData::new(data);
        let position = Vector3i::new(1, 1, 1);
        assert!(shared
            .try_set_voxel_checked(7, position, ChannelId::Type.index())
            .unwrap());

        assert_eq!(
            shared.mark_area_modified_checked(
                Box3i::new(Vector3i::splat(i32::MAX - 1), Vector3i::splat(4)),
                true,
            ),
            Err(SharedVoxelDataMutationError::SpatialBoundsOverflow { lod_index: 0 })
        );
        assert!(
            shared
                .mark_area_modified_checked(
                    Box3i::new(Vector3i::zero(), Vector3i::new(-1, 1, 1)),
                    true,
                )
                .unwrap()
                .is_empty()
        );
        let block = shared.block_snapshot(Vector3i::zero(), 0).unwrap();
        assert!(!block.is_modified());
        assert!(!block.is_edited());
        assert!(!block.needs_lodding());
        assert_eq!(
            shared.key_revision(Vector3i::zero(), 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
    }

    #[test]
    fn shared_mark_area_and_dirty_consumption_advance_each_changed_key_once() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::new(32, 16, 16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = SharedVoxelData::new(data);
        let channel = ChannelId::Type.index();
        let first = Vector3i::zero();
        let second = Vector3i::new(1, 0, 0);

        assert!(shared
            .try_set_voxel_checked(31, Vector3i::new(1, 1, 1), channel)
            .unwrap());
        assert!(shared
            .try_set_voxel_checked(32, Vector3i::new(17, 1, 1), channel)
            .unwrap());
        assert_eq!(
            shared
                .mark_area_modified_checked(
                    Box3i::new(Vector3i::zero(), Vector3i::new(32, 1, 1)),
                    true,
                )
                .unwrap(),
            vec![first, second]
        );
        for position in [first, second] {
            assert_eq!(
                shared.key_revision(position, 0),
                Some(VoxelDataKeyRevision::Present(2))
            );
        }

        assert!(
            shared
                .mark_area_modified_checked(
                    Box3i::new(Vector3i::zero(), Vector3i::new(32, 1, 1)),
                    true,
                )
                .unwrap()
                .is_empty()
        );
        for position in [first, second] {
            assert_eq!(
                shared.key_revision(position, 0),
                Some(VoxelDataKeyRevision::Present(2))
            );
        }

        let saves = shared.consume_all_modifications_checked().unwrap();
        assert_eq!(
            saves
                .iter()
                .map(|save| (save.lod_index, save.position))
                .collect::<Vec<_>>(),
            vec![(0, first), (0, second)]
        );
        assert_eq!(
            saves[0]
                .voxels
                .as_ref()
                .unwrap()
                .get_voxel(1, 1, 1, channel),
            31
        );
        assert_eq!(
            saves[1]
                .voxels
                .as_ref()
                .unwrap()
                .get_voxel(1, 1, 1, channel),
            32
        );
        for position in [first, second] {
            assert!(!shared.block_snapshot(position, 0).unwrap().is_modified());
            assert_eq!(
                shared.key_revision(position, 0),
                Some(VoxelDataKeyRevision::Present(3))
            );
        }
    }

    #[test]
    fn shared_dirty_consumption_revision_overflow_keeps_every_block_dirty() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::new(32, 16, 16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = SharedVoxelData::new(data);
        let channel = ChannelId::Type.index();
        let first = Vector3i::zero();
        let second = Vector3i::new(1, 0, 0);

        assert!(shared
            .try_set_voxel_checked(1, Vector3i::new(1, 1, 1), channel)
            .unwrap());
        assert!(shared
            .try_set_voxel_checked(2, Vector3i::new(17, 1, 1), channel)
            .unwrap());
        shared
            .mark_area_modified_checked(
                Box3i::new(Vector3i::zero(), Vector3i::new(32, 1, 1)),
                false,
            )
            .unwrap();
        shared.with_lod_map_mut(0, |map| {
            map.set_key_revision_for_test(second, u64::MAX);
        });

        assert!(matches!(
            shared.consume_all_modifications_checked(),
            Err(SharedVoxelDataMutationError::KeyRevisionOverflow {
                position,
                lod_index: 0,
            }) if position == second
        ));
        assert!(shared.block_snapshot(first, 0).unwrap().is_modified());
        assert!(shared.block_snapshot(second, 0).unwrap().is_modified());
        assert_eq!(
            shared.key_revision(first, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
    }

    #[test]
    fn shared_lod_update_revision_overflow_rolls_back_every_lod() {
        let mut data = VoxelData::new();
        data.set_lod_count(2).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = SharedVoxelData::new(data);
        let source = Vector3i::zero();
        let destination = Vector3i::zero();

        assert!(shared
            .try_set_voxel_checked(3, Vector3i::new(1, 1, 1), ChannelId::Type.index())
            .unwrap());
        let modified = shared
            .mark_area_modified_checked(Box3i::new(Vector3i::zero(), Vector3i::splat(1)), true)
            .unwrap();
        assert!(shared
            .try_set_block(destination, VoxelDataBlock::empty(1))
            .unwrap());
        shared.with_lod_map_mut(1, |map| {
            map.set_key_revision_for_test(destination, u64::MAX);
        });

        assert_eq!(
            shared.update_lods_from_lod0_blocks_checked(&modified),
            Err(SharedVoxelDataMutationError::KeyRevisionOverflow {
                position: destination,
                lod_index: 1,
            })
        );
        assert!(shared.block_snapshot(source, 0).unwrap().needs_lodding());
        assert!(!shared.block_snapshot(destination, 1).unwrap().has_voxels());
        assert_eq!(
            shared.key_revision(source, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
    }

    #[test]
    fn shared_lod_update_retains_source_until_streamed_parent_is_resident() {
        let mut data = VoxelData::new();
        data.set_lod_count(2).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(true);
        data.set_full_load_completed(true);
        let shared = SharedVoxelData::new(data);
        let source = Vector3i::zero();
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(16));
        voxels.set_voxel(29, 2, 2, 2, ChannelId::Type.index());
        assert!(shared
            .try_set_block(source, VoxelDataBlock::with_voxels(voxels, 0))
            .unwrap());
        let modified = shared
            .mark_area_modified_checked(
                Box3i::new(Vector3i::new(2, 2, 2), Vector3i::splat(1)),
                true,
            )
            .unwrap();

        assert_eq!(
            shared.update_lods_from_lod0_blocks_checked(&modified),
            Err(SharedVoxelDataMutationError::LodDestinationUnavailable {
                position: source,
                lod_index: 1,
            })
        );
        assert!(shared.block_snapshot(source, 0).unwrap().needs_lodding());
        assert!(shared.block_snapshot(source, 1).is_none());
        assert_eq!(
            shared.key_revision(source, 0),
            Some(VoxelDataKeyRevision::Present(2))
        );

        assert!(shared
            .try_set_block(
                source,
                VoxelDataBlock::with_voxels(VoxelBuffer::with_size(Vector3i::splat(16)), 1),
            )
            .unwrap());
        assert!(shared
            .update_lods_from_lod0_blocks_checked(&modified)
            .is_ok());
        assert!(!shared.block_snapshot(source, 0).unwrap().needs_lodding());
        assert_eq!(
            shared
                .block_snapshot(source, 1)
                .unwrap()
                .voxels()
                .get_voxel(1, 1, 1, ChannelId::Type.index()),
            29
        );
    }

    #[test]
    fn shared_lod_update_mutates_each_unique_key_once_across_all_lods() {
        let mut data = VoxelData::new();
        data.set_lod_count(3).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(64)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = SharedVoxelData::new(data);
        let channel = ChannelId::Type.index();
        let first = Vector3i::zero();
        let sibling = Vector3i::new(1, 0, 0);

        assert!(shared
            .try_set_voxel_checked(41, Vector3i::new(1, 1, 1), channel)
            .unwrap());
        assert!(shared
            .try_set_voxel_checked(42, Vector3i::new(17, 1, 1), channel)
            .unwrap());
        let modified = shared
            .mark_area_modified_checked(Box3i::new(Vector3i::zero(), Vector3i::new(32, 1, 1)), true)
            .unwrap();
        let updated = shared
            .update_lods_from_lod0_blocks_checked(&modified)
            .unwrap();

        assert_eq!(
            updated,
            vec![
                BlockLocation {
                    position: first,
                    lod_index: 0,
                },
                BlockLocation {
                    position: sibling,
                    lod_index: 0,
                },
                BlockLocation {
                    position: first,
                    lod_index: 1,
                },
                BlockLocation {
                    position: first,
                    lod_index: 2,
                },
            ]
        );
        for position in [first, sibling] {
            assert_eq!(
                shared.key_revision(position, 0),
                Some(VoxelDataKeyRevision::Present(3))
            );
        }
        assert_eq!(
            shared.key_revision(first, 1),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(
            shared.key_revision(first, 2),
            Some(VoxelDataKeyRevision::Present(1))
        );
        assert_eq!(shared.lod_structural_revision(0), Some(0));
        assert_eq!(shared.lod_structural_revision(1), Some(0));
        assert_eq!(shared.lod_structural_revision(2), Some(0));
    }

    #[test]
    fn shared_lod_update_propagates_through_an_already_pending_intermediate_block() {
        let mut data = VoxelData::new();
        data.set_lod_count(3).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(64)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = SharedVoxelData::new(data);
        let source = Vector3i::zero();
        assert!(shared
            .try_set_voxel_checked(51, Vector3i::new(1, 1, 1), ChannelId::Type.index())
            .unwrap());
        let modified = shared
            .mark_area_modified_checked(Box3i::new(Vector3i::zero(), Vector3i::splat(1)), true)
            .unwrap();
        let mut intermediate =
            VoxelDataBlock::with_voxels(VoxelBuffer::with_size(Vector3i::splat(16)), 1);
        intermediate.set_needs_lodding(true);
        assert!(shared.try_set_block(source, intermediate).unwrap());

        let updated = shared
            .update_lods_from_lod0_blocks_checked(&modified)
            .unwrap();

        assert!(updated.contains(&BlockLocation {
            position: source,
            lod_index: 2,
        }));
        assert!(shared.block_snapshot(source, 2).unwrap().has_voxels());
        assert!(!shared.block_snapshot(source, 1).unwrap().needs_lodding());
    }

    #[test]
    fn shared_dirty_consumption_rejects_a_same_key_edit_after_snapshot_and_retries() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = Arc::new(SharedVoxelData::new(data));
        let channel = ChannelId::Type.index();
        let voxel = Vector3i::new(1, 1, 1);
        let block = Vector3i::zero();
        assert_eq!(shared.try_edit_voxel_checked(1, voxel, channel), Ok(true));

        let prepared = Arc::new((Mutex::new(false), Condvar::new()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        shared.set_test_edit_phase_hook(Arc::new({
            let prepared = prepared.clone();
            let release = release.clone();
            move |phase| {
                if phase != SharedVoxelDataEditPhase::DirtySnapshotPreparedBeforeMutationGate {
                    return;
                }
                *prepared.0.lock().unwrap() = true;
                prepared.1.notify_one();
                let mut released = release.0.lock().unwrap();
                while !*released {
                    released = release.1.wait(released).unwrap();
                }
            }
        }));
        let consumer_data = shared.clone();
        let consumer =
            std::thread::spawn(move || consumer_data.consume_all_modifications_checked());

        let mut reached = prepared.0.lock().unwrap();
        while !*reached {
            let (next, timeout) = prepared
                .1
                .wait_timeout(reached, Duration::from_secs(1))
                .unwrap();
            reached = next;
            assert!(!timeout.timed_out(), "dirty snapshot was not prepared");
        }
        drop(reached);
        assert_eq!(shared.try_edit_voxel_checked(2, voxel, channel), Ok(true));
        *release.0.lock().unwrap() = true;
        release.1.notify_one();

        assert_eq!(
            consumer.join().unwrap().unwrap_err(),
            SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: block,
                lod_index: 0,
                expected_revision: VoxelDataKeyRevision::Present(1),
                actual_revision: VoxelDataKeyRevision::Present(2),
            }
        );
        let resident = shared.block_snapshot(block, 0).unwrap();
        assert_eq!(resident.voxels().get_voxel(1, 1, 1, channel), 2);
        assert!(resident.is_modified());

        let saves = shared.consume_all_modifications_checked().unwrap();
        assert_eq!(saves.len(), 1);
        assert_eq!(
            saves[0]
                .voxels
                .as_ref()
                .unwrap()
                .get_voxel(1, 1, 1, channel),
            2
        );
        assert!(!shared.block_snapshot(block, 0).unwrap().is_modified());
    }

    #[test]
    fn shared_lod_update_rejects_a_same_key_edit_after_preparation_and_retries() {
        let mut data = VoxelData::new();
        data.set_lod_count(2).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = Arc::new(SharedVoxelData::new(data));
        let channel = ChannelId::Type.index();
        let source = Vector3i::zero();
        let voxel = Vector3i::new(1, 1, 1);
        assert!(shared.try_set_voxel_checked(3, voxel, channel).unwrap());
        let modified = shared
            .mark_area_modified_checked(Box3i::new(voxel, Vector3i::splat(1)), true)
            .unwrap();

        let prepared = Arc::new((Mutex::new(false), Condvar::new()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        shared.set_test_edit_phase_hook(Arc::new({
            let prepared = prepared.clone();
            let release = release.clone();
            move |phase| {
                if phase != SharedVoxelDataEditPhase::LodUpdatePreparedBeforeMutationGate {
                    return;
                }
                *prepared.0.lock().unwrap() = true;
                prepared.1.notify_one();
                let mut released = release.0.lock().unwrap();
                while !*released {
                    released = release.1.wait(released).unwrap();
                }
            }
        }));
        let updater_data = shared.clone();
        let updater_modified = modified.clone();
        let updater = std::thread::spawn(move || {
            updater_data.update_lods_from_lod0_blocks_checked(&updater_modified)
        });

        let mut reached = prepared.0.lock().unwrap();
        while !*reached {
            let (next, timeout) = prepared
                .1
                .wait_timeout(reached, Duration::from_secs(1))
                .unwrap();
            reached = next;
            assert!(!timeout.timed_out(), "LOD drafts were not prepared");
        }
        drop(reached);
        assert_eq!(shared.try_edit_voxel_checked(9, voxel, channel), Ok(true));
        *release.0.lock().unwrap() = true;
        release.1.notify_one();

        assert_eq!(
            updater.join().unwrap(),
            Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: source,
                lod_index: 0,
                expected_revision: VoxelDataKeyRevision::Present(2),
                actual_revision: VoxelDataKeyRevision::Present(3),
            })
        );
        assert!(shared.block_snapshot(source, 1).is_none());
        assert!(shared.block_snapshot(source, 0).unwrap().needs_lodding());

        assert!(shared
            .update_lods_from_lod0_blocks_checked(&modified)
            .is_ok());
        assert!(shared.block_snapshot(source, 1).unwrap().has_voxels());
        assert!(!shared.block_snapshot(source, 0).unwrap().needs_lodding());
    }

    #[test]
    fn shared_lod_update_rejects_generator_output_after_settings_change() {
        let mut data = VoxelData::new();
        data.set_lod_count(2).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let entered = Arc::new((Mutex::new(false), Condvar::new()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let generation_may_finish = Arc::new((Mutex::new(false), Condvar::new()));
        data.set_generator(Some(Arc::new(BlockingGenerator {
            entered: entered.clone(),
            release: release.clone(),
            resident_inserted: generation_may_finish.clone(),
        })));
        let shared = Arc::new(SharedVoxelData::new(data));
        assert!(shared
            .try_set_voxel_checked(5, Vector3i::new(1, 1, 1), ChannelId::Type.index(),)
            .unwrap());
        let modified = shared
            .mark_area_modified_checked(Box3i::new(Vector3i::zero(), Vector3i::splat(1)), true)
            .unwrap();
        let worker_data = shared.clone();
        let worker =
            std::thread::spawn(move || worker_data.update_lods_from_lod0_blocks_checked(&modified));

        let (entered_lock, entered_cvar) = &*entered;
        let mut started = entered_lock.lock().unwrap();
        while !*started {
            let (next, timeout) = entered_cvar
                .wait_timeout(started, Duration::from_secs(1))
                .unwrap();
            started = next;
            assert!(!timeout.timed_out(), "LOD generator callback did not start");
        }
        drop(started);
        shared.set_generator(None).unwrap();
        *release.0.lock().unwrap() = true;
        release.1.notify_one();
        *generation_may_finish.0.lock().unwrap() = true;
        generation_may_finish.1.notify_one();

        assert_eq!(
            worker.join().unwrap(),
            Err(SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                expected_revision: 0,
                actual_revision: 1,
            })
        );
        assert!(shared
            .block_snapshot(Vector3i::zero(), 0)
            .unwrap()
            .needs_lodding());
        assert!(shared.block_snapshot(Vector3i::zero(), 1).is_none());
    }

    #[test]
    fn shared_lod_update_acquires_every_spatial_batch_before_any_map_write() {
        let mut data = VoxelData::new();
        data.set_lod_count(2).unwrap();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = Arc::new(SharedVoxelData::new(data));
        assert!(shared
            .try_set_voxel_checked(9, Vector3i::new(1, 1, 1), ChannelId::Type.index(),)
            .unwrap());
        assert!(shared
            .try_set_block(
                Vector3i::zero(),
                VoxelDataBlock::with_voxels(VoxelBuffer::with_size(Vector3i::splat(16)), 1),
            )
            .unwrap());
        let modified = shared
            .mark_area_modified_checked(Box3i::new(Vector3i::zero(), Vector3i::splat(1)), true)
            .unwrap();
        let lod0_spatial_reached = Arc::new((Mutex::new(false), Condvar::new()));
        shared.set_test_edit_phase_hook(Arc::new({
            let lod0_spatial_reached = lod0_spatial_reached.clone();
            move |phase| {
                if matches!(
                    phase,
                    SharedVoxelDataEditPhase::SpatialWriteBatchAcquired { lod_index: 0 }
                ) {
                    *lod0_spatial_reached.0.lock().unwrap() = true;
                    lod0_spatial_reached.1.notify_one();
                }
            }
        }));
        let higher_lod_reader =
            shared.read_region(1, Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        let writer_data = shared.clone();
        let writer =
            std::thread::spawn(move || writer_data.update_lods_from_lod0_blocks_checked(&modified));

        let mut reached = lod0_spatial_reached.0.lock().unwrap();
        while !*reached {
            let (next, timeout) = lod0_spatial_reached
                .1
                .wait_timeout(reached, Duration::from_secs(1))
                .unwrap();
            reached = next;
            assert!(
                !timeout.timed_out(),
                "LOD writer did not acquire LOD0 spatial batch"
            );
        }
        drop(reached);
        assert!(
            shared.try_lod_map_write(0).is_some(),
            "LOD0 map must remain unlocked while the writer waits for a later spatial batch"
        );
        assert!(
            shared
                .try_write_region(0, Box3i::new(Vector3i::zero(), Vector3i::splat(16)),)
                .is_none(),
            "the earlier LOD spatial batch must already be held"
        );
        drop(higher_lod_reader);
        assert!(writer.join().unwrap().is_ok());
    }

    #[test]
    fn shared_edit_voxel_does_not_materialize_unavailable_blocks() {
        let channel = ChannelId::Type.index();

        for &(streaming_enabled, full_load_completed) in &[(true, true), (false, false)] {
            for empty_block in [false, true] {
                let mut data = VoxelData::new();
                data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
                data.set_streaming_enabled(streaming_enabled);
                data.set_full_load_completed(full_load_completed);
                let generator = Arc::new(RecordingGenerator::default());
                data.set_generator(Some(generator.clone()));
                let shared = SharedVoxelData::new(data);

                if empty_block {
                    assert!(shared
                        .try_set_block(Vector3i::zero(), VoxelDataBlock::empty(0))
                        .unwrap());
                }

                assert!(
                    !shared.try_edit_voxel(99, Vector3i::new(1, 1, 1), channel),
                    "streaming={streaming_enabled}, full_load_completed={full_load_completed}, empty_block={empty_block}"
                );
                assert!(generator.calls.lock().unwrap().is_empty());

                match shared.block_snapshot(Vector3i::zero(), 0) {
                    Some(block) if empty_block => assert!(!block.has_voxels()),
                    None if !empty_block => {}
                    _ => panic!(
                        "unavailable block was materialized: streaming={streaming_enabled}, full_load_completed={full_load_completed}, empty_block={empty_block}"
                    ),
                }
            }
        }
    }

    struct SpatialLockProbeGenerator {
        data: std::sync::Weak<SharedVoxelData>,
    }

    impl VoxelGenerator for SpatialLockProbeGenerator {
        fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
            let data = self
                .data
                .upgrade()
                .expect("shared data survives generation");
            let block_box = Box3i::new(input.origin_in_voxels, input.buffer.size());
            drop(
                data.try_write_region(0, block_box)
                    .expect("generator must run without a spatial write guard"),
            );
            input.buffer.fill(42, ChannelId::Type.index());
            GenResult::default()
        }
    }

    #[test]
    fn shared_edit_voxel_runs_generator_without_a_spatial_write_guard() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = Arc::new(SharedVoxelData::new(data));
        shared
            .set_generator(Some(Arc::new(SpatialLockProbeGenerator {
                data: Arc::downgrade(&shared),
            })))
            .unwrap();
        let channel = ChannelId::Type.index();

        assert!(shared.try_edit_voxel(99, Vector3i::new(1, 1, 1), channel));
        assert_eq!(
            shared
                .block_snapshot(Vector3i::zero(), 0)
                .unwrap()
                .voxels()
                .get_voxel(2, 1, 1, channel),
            42
        );
    }

    #[test]
    fn shared_edit_voxel_signals_spatial_lock_before_waiting_for_map_lock() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = Arc::new(SharedVoxelData::new(data));
        let spatial_phase = Arc::new((Mutex::new(false), Condvar::new()));
        shared.set_test_edit_phase_hook(Arc::new({
            let spatial_phase = spatial_phase.clone();
            move |phase| {
                if phase == SharedVoxelDataEditPhase::SpatialWriteAcquiredBeforeMapLock {
                    let (lock, cvar) = &*spatial_phase;
                    *lock.lock().unwrap() = true;
                    cvar.notify_one();
                }
            }
        }));

        let held_map = shared
            .try_lod_map_read(0)
            .expect("test must hold a LOD map read lock");
        let edit_data = shared.clone();
        let channel = ChannelId::Type.index();
        let edit = std::thread::spawn(move || {
            edit_data.try_edit_voxel(99, Vector3i::new(1, 1, 1), channel)
        });

        let (phase_lock, phase_cvar) = &*spatial_phase;
        let mut signalled = phase_lock.lock().unwrap();
        let reached_before_map_unlock = loop {
            if *signalled {
                break true;
            }
            let (next, timeout) = phase_cvar
                .wait_timeout(signalled, Duration::from_secs(1))
                .unwrap();
            signalled = next;
            if timeout.timed_out() && !*signalled {
                break false;
            }
        };
        let spatial_lock_held = reached_before_map_unlock
            && shared
                .try_write_region(0, Box3i::new(Vector3i::zero(), Vector3i::splat(16)))
                .is_none();
        drop(signalled);
        drop(held_map);

        assert!(edit.join().unwrap());
        assert!(
            reached_before_map_unlock,
            "try_edit_voxel did not signal the spatial phase before the blocked map lock"
        );
        assert!(
            spatial_lock_held,
            "the target spatial write region was not held while the spatial phase was signalled"
        );
    }

    #[test]
    fn shared_edit_voxel_keeps_map_write_lock_until_dirty_flags_are_set() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = Arc::new(SharedVoxelData::new(data));
        assert!(shared
            .try_set_block(
                Vector3i::zero(),
                VoxelDataBlock::with_voxels(VoxelBuffer::with_size(Vector3i::splat(16)), 0),
            )
            .unwrap());
        let before_dirty = Arc::new((Mutex::new(false), Condvar::new()));
        let release_before_dirty = Arc::new((Mutex::new(false), Condvar::new()));
        let after_dirty = Arc::new((Mutex::new(false), Condvar::new()));
        let release_after_dirty = Arc::new((Mutex::new(false), Condvar::new()));
        shared.set_test_edit_phase_hook(Arc::new({
            let before_dirty = before_dirty.clone();
            let release_before_dirty = release_before_dirty.clone();
            let after_dirty = after_dirty.clone();
            let release_after_dirty = release_after_dirty.clone();
            move |phase| match phase {
                SharedVoxelDataEditPhase::VoxelWrittenBeforeDirtyFlags => {
                    let (entered_lock, entered_cvar) = &*before_dirty;
                    *entered_lock.lock().unwrap() = true;
                    entered_cvar.notify_one();

                    let (release_lock, release_cvar) = &*release_before_dirty;
                    let mut released = release_lock.lock().unwrap();
                    while !*released {
                        let (next, timeout) = release_cvar
                            .wait_timeout(released, Duration::from_secs(1))
                            .unwrap();
                        released = next;
                        assert!(
                            !timeout.timed_out(),
                            "edit phase timed out before dirty flags; map write lock may have escaped its closure"
                        );
                    }
                }
                SharedVoxelDataEditPhase::DirtyFlagsSetBeforeMapWriteUnlock => {
                    let (entered_lock, entered_cvar) = &*after_dirty;
                    *entered_lock.lock().unwrap() = true;
                    entered_cvar.notify_one();

                    let (release_lock, release_cvar) = &*release_after_dirty;
                    let mut released = release_lock.lock().unwrap();
                    while !*released {
                        let (next, timeout) = release_cvar
                            .wait_timeout(released, Duration::from_secs(1))
                            .unwrap();
                        released = next;
                        assert!(
                            !timeout.timed_out(),
                            "edit phase timed out after dirty flags; map write lock may have escaped its closure"
                        );
                    }
                }
                SharedVoxelDataEditPhase::LodUpdatePreparedBeforeMutationGate
                | SharedVoxelDataEditPhase::DirtySnapshotPreparedBeforeMutationGate
                | SharedVoxelDataEditPhase::PreparedVoxelEditDraftedBeforeTransactionPrepare
                | SharedVoxelDataEditPhase::MutationGateAcquiredBeforeSpatialWrite
                | SharedVoxelDataEditPhase::SpatialWriteBatchAcquired { .. }
                | SharedVoxelDataEditPhase::SpatialWriteAcquiredBeforeMapLock
                | SharedVoxelDataEditPhase::PreparedTransactionValidatedBeforeFirstLiveWrite => {}
            }
        }));

        let edit_data = shared.clone();
        let channel = ChannelId::Type.index();
        let edit = std::thread::spawn(move || {
            edit_data.try_edit_voxel(99, Vector3i::new(1, 1, 1), channel)
        });

        let (entered_lock, entered_cvar) = &*before_dirty;
        let mut entered = entered_lock.lock().unwrap();
        let reached_before_dirty = loop {
            if *entered {
                break true;
            }
            let (next, timeout) = entered_cvar
                .wait_timeout(entered, Duration::from_secs(1))
                .unwrap();
            entered = next;
            if timeout.timed_out() && !*entered {
                break false;
            }
        };
        let map_still_write_locked = reached_before_dirty && shared.try_lod_map_read(0).is_none();
        drop(entered);
        let (release_lock, release_cvar) = &*release_before_dirty;
        *release_lock.lock().unwrap() = true;
        release_cvar.notify_one();

        let (after_dirty_lock, after_dirty_cvar) = &*after_dirty;
        let mut after_dirty_entered = after_dirty_lock.lock().unwrap();
        let reached_after_dirty = loop {
            if *after_dirty_entered {
                break true;
            }
            let (next, timeout) = after_dirty_cvar
                .wait_timeout(after_dirty_entered, Duration::from_secs(1))
                .unwrap();
            after_dirty_entered = next;
            if timeout.timed_out() && !*after_dirty_entered {
                break false;
            }
        };
        let map_still_write_locked_after_dirty =
            reached_after_dirty && shared.try_lod_map_read(0).is_none();
        drop(after_dirty_entered);
        let (release_lock, release_cvar) = &*release_after_dirty;
        *release_lock.lock().unwrap() = true;
        release_cvar.notify_one();

        assert!(edit.join().unwrap());
        assert!(
            reached_before_dirty,
            "try_edit_voxel did not expose the pre-dirty phase"
        );
        assert!(
            map_still_write_locked,
            "try_edit_voxel released its map write lock between voxel mutation and dirty flags"
        );
        assert!(
            reached_after_dirty,
            "try_edit_voxel did not expose the fully dirty phase"
        );
        assert!(
            map_still_write_locked_after_dirty,
            "try_edit_voxel released its map write lock after dirty flags but before leaving the map write closure"
        );
    }

    #[test]
    fn shared_edit_voxel_is_dirty_before_immediate_unview() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = SharedVoxelData::new(data);
        let channel = ChannelId::Type.index();

        assert!(shared.try_edit_voxel(77, Vector3i::new(1, 1, 1), channel));
        let mut saves = Vec::new();
        let area = Box3i::new(Vector3i::zero(), Vector3i::splat(1));
        shared.view_area(area, 0, None, None, None).unwrap();
        shared
            .unview_area(area, 0, None, None, Some(&mut saves))
            .unwrap();

        assert_eq!(saves.len(), 1);
        assert_eq!(saves[0].position, Vector3i::zero());
        assert_eq!(
            saves[0]
                .voxels
                .as_ref()
                .unwrap()
                .get_voxel(1, 1, 1, channel),
            77
        );
    }

    struct BlockingGenerator {
        entered: Arc<(Mutex<bool>, Condvar)>,
        release: Arc<(Mutex<bool>, Condvar)>,
        resident_inserted: Arc<(Mutex<bool>, Condvar)>,
    }

    impl VoxelGenerator for BlockingGenerator {
        fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
            let (entered_lock, entered_cvar) = &*self.entered;
            *entered_lock.lock().unwrap() = true;
            entered_cvar.notify_one();
            let (release_lock, release_cvar) = &*self.release;
            let mut released = release_lock.lock().unwrap();
            while !*released {
                let (next, timeout) = release_cvar
                    .wait_timeout(released, Duration::from_secs(1))
                    .unwrap();
                released = next;
                assert!(
                    !timeout.timed_out(),
                    "blocking generator timed out waiting for release"
                );
            }
            drop(released);

            let (resident_lock, resident_cvar) = &*self.resident_inserted;
            let mut inserted = resident_lock.lock().unwrap();
            while !*inserted {
                let (next, timeout) = resident_cvar
                    .wait_timeout(inserted, Duration::from_secs(1))
                    .unwrap();
                inserted = next;
                assert!(
                    !timeout.timed_out(),
                    "blocking generator timed out waiting for resident insertion; map lock may be held during generation"
                );
            }
            input.buffer.fill(10, ChannelId::Type.index());
            GenResult::default()
        }
    }

    #[test]
    fn shared_edit_voxel_rejects_a_stale_preparation_without_overwriting_resident_data() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let entered = Arc::new((Mutex::new(false), Condvar::new()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let resident_inserted = Arc::new((Mutex::new(false), Condvar::new()));
        data.set_generator(Some(Arc::new(BlockingGenerator {
            entered: entered.clone(),
            release: release.clone(),
            resident_inserted: resident_inserted.clone(),
        })));
        let shared = Arc::new(SharedVoxelData::new(data));
        let channel = ChannelId::Type.index();
        let edit_data = shared.clone();
        let edit = std::thread::spawn(move || {
            edit_data.try_edit_voxel_checked(99, Vector3i::new(1, 1, 1), channel)
        });

        let (entered_lock, entered_cvar) = &*entered;
        let mut started = entered_lock.lock().unwrap();
        while !*started {
            let (next, timeout) = entered_cvar
                .wait_timeout(started, Duration::from_secs(1))
                .unwrap();
            started = next;
            assert!(
                !timeout.timed_out(),
                "try_edit_voxel never entered procedural materialization"
            );
        }
        drop(started);
        let (release_lock, release_cvar) = &*release;
        *release_lock.lock().unwrap() = true;
        release_cvar.notify_one();
        let mut resident = VoxelBuffer::with_size(Vector3i::splat(16));
        resident.set_voxel(33, 2, 1, 1, channel);
        assert!(shared
            .try_set_block(Vector3i::zero(), VoxelDataBlock::with_voxels(resident, 0))
            .unwrap());
        let (resident_lock, resident_cvar) = &*resident_inserted;
        *resident_lock.lock().unwrap() = true;
        resident_cvar.notify_one();
        assert_eq!(
            edit.join().unwrap(),
            Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: Vector3i::zero(),
                lod_index: 0,
                expected_revision: VoxelDataKeyRevision::Tombstone(0),
                actual_revision: VoxelDataKeyRevision::Present(1),
            })
        );

        let block = shared.block_snapshot(Vector3i::zero(), 0).unwrap();
        assert_eq!(block.voxels().get_voxel(1, 1, 1, channel), 0);
        assert_eq!(block.voxels().get_voxel(2, 1, 1, channel), 33);
    }

    #[test]
    fn shared_edit_voxel_conflicts_with_checked_direct_materialization_and_retries_cleanly() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        assert!(data.try_set_block(Vector3i::zero(), VoxelDataBlock::empty(0)));
        let entered = Arc::new((Mutex::new(false), Condvar::new()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let materialized = Arc::new((Mutex::new(false), Condvar::new()));
        data.set_generator(Some(Arc::new(BlockingGenerator {
            entered: entered.clone(),
            release: release.clone(),
            resident_inserted: materialized.clone(),
        })));
        let shared = Arc::new(SharedVoxelData::new(data));
        let channel = ChannelId::Type.index();
        let edit_data = shared.clone();
        let edit = std::thread::spawn(move || {
            edit_data.try_edit_voxel_checked(99, Vector3i::new(1, 1, 1), channel)
        });

        let (entered_lock, entered_cvar) = &*entered;
        let mut started = entered_lock.lock().unwrap();
        while !*started {
            let (next, timeout) = entered_cvar
                .wait_timeout(started, Duration::from_secs(1))
                .unwrap();
            started = next;
            assert!(!timeout.timed_out(), "generator callback did not start");
        }
        drop(started);
        assert!(shared
            .try_set_voxel_checked(33, Vector3i::new(2, 1, 1), channel)
            .unwrap());
        let (release_lock, release_cvar) = &*release;
        *release_lock.lock().unwrap() = true;
        release_cvar.notify_one();
        let (materialized_lock, materialized_cvar) = &*materialized;
        *materialized_lock.lock().unwrap() = true;
        materialized_cvar.notify_one();

        assert_eq!(
            edit.join().unwrap(),
            Err(SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: Vector3i::zero(),
                lod_index: 0,
                expected_revision: VoxelDataKeyRevision::Present(0),
                actual_revision: VoxelDataKeyRevision::Present(1),
            })
        );
        let block = shared.block_snapshot(Vector3i::zero(), 0).unwrap();
        assert_eq!(block.voxels().get_voxel(1, 1, 1, channel), 0);
        assert_eq!(block.voxels().get_voxel(2, 1, 1, channel), 33);
        assert_eq!(block.voxels().get_voxel(3, 1, 1, channel), 0);

        assert_eq!(
            shared.try_edit_voxel_checked(99, Vector3i::new(1, 1, 1), channel),
            Ok(true)
        );
        let block = shared.block_snapshot(Vector3i::zero(), 0).unwrap();
        assert_eq!(block.voxels().get_voxel(1, 1, 1, channel), 99);
        assert_eq!(block.voxels().get_voxel(2, 1, 1, channel), 33);
        assert_eq!(
            shared.key_revision(Vector3i::zero(), 0),
            Some(VoxelDataKeyRevision::Present(2))
        );
    }

    #[test]
    fn shared_edit_voxel_rejects_generator_output_after_a_settings_change() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let entered = Arc::new((Mutex::new(false), Condvar::new()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let generation_may_finish = Arc::new((Mutex::new(false), Condvar::new()));
        data.set_generator(Some(Arc::new(BlockingGenerator {
            entered: entered.clone(),
            release: release.clone(),
            resident_inserted: generation_may_finish.clone(),
        })));
        let shared = Arc::new(SharedVoxelData::new(data));
        let edit_data = shared.clone();
        let edit = std::thread::spawn(move || {
            edit_data.try_edit_voxel_checked(99, Vector3i::new(1, 1, 1), ChannelId::Type.index())
        });

        let (entered_lock, entered_cvar) = &*entered;
        let mut started = entered_lock.lock().unwrap();
        while !*started {
            let (next, timeout) = entered_cvar
                .wait_timeout(started, Duration::from_secs(1))
                .unwrap();
            started = next;
            assert!(!timeout.timed_out(), "generator callback did not start");
        }
        drop(started);
        shared.set_generator(None).unwrap();
        let (release_lock, release_cvar) = &*release;
        *release_lock.lock().unwrap() = true;
        release_cvar.notify_one();
        let (finish_lock, finish_cvar) = &*generation_may_finish;
        *finish_lock.lock().unwrap() = true;
        finish_cvar.notify_one();

        assert_eq!(
            edit.join().unwrap(),
            Err(SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                expected_revision: 0,
                actual_revision: 1,
            })
        );
        assert!(shared.block_snapshot(Vector3i::zero(), 0).is_none());
    }

    struct MapUnlockProbeGenerator {
        data: std::sync::Weak<SharedVoxelData>,
    }

    impl VoxelGenerator for MapUnlockProbeGenerator {
        fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
            let data = self
                .data
                .upgrade()
                .expect("shared data survives generation");
            drop(
                data.try_lod_map_write(0)
                    .expect("generator must run without map write lock"),
            );
            input.buffer.fill(42, ChannelId::Type.index());
            GenResult::default()
        }
    }

    #[test]
    fn shared_edit_voxel_runs_generator_without_map_lock() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(16)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let shared = Arc::new(SharedVoxelData::new(data));
        shared
            .set_generator(Some(Arc::new(MapUnlockProbeGenerator {
                data: Arc::downgrade(&shared),
            })))
            .unwrap();
        let channel = ChannelId::Type.index();

        assert!(shared.try_edit_voxel(99, Vector3i::new(1, 1, 1), channel));
        assert_eq!(
            shared
                .block_snapshot(Vector3i::zero(), 0)
                .unwrap()
                .voxels()
                .get_voxel(2, 1, 1, channel),
            42
        );
    }

    #[test]
    fn shared_voxel_data_allows_parallel_read_snapshots() {
        let shared = Arc::new(SharedVoxelData::new(VoxelData::new()));
        let entered = Arc::new((Mutex::new(0usize), Condvar::new()));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let shared = shared.clone();
                let entered = entered.clone();
                std::thread::spawn(move || {
                    shared.with_settings(|_| {
                        let (lock, cvar) = &*entered;
                        let mut count = lock.lock().unwrap();
                        *count += 1;
                        cvar.notify_all();
                        while *count < 2 {
                            let (next, timeout) =
                                cvar.wait_timeout(count, Duration::from_secs(1)).unwrap();
                            count = next;
                            if timeout.timed_out() && *count < 2 {
                                return false;
                            }
                        }
                        true
                    })
                })
            })
            .collect();

        for handle in handles {
            assert!(
                handle.join().unwrap(),
                "SharedVoxelData read snapshots should overlap"
            );
        }
    }

    #[test]
    fn shared_voxel_data_allows_parallel_lod_map_writes() {
        let mut data = VoxelData::new();
        data.set_lod_count(2).unwrap();
        let shared = Arc::new(SharedVoxelData::new(data));
        let entered = Arc::new((Mutex::new(0usize), Condvar::new()));

        let handles: Vec<_> = (0..2)
            .map(|lod_index| {
                let shared = shared.clone();
                let entered = entered.clone();
                std::thread::spawn(move || {
                    shared.with_lod_map_mut(lod_index, |_| {
                        let (lock, cvar) = &*entered;
                        let mut count = lock.lock().unwrap();
                        *count += 1;
                        cvar.notify_all();
                        while *count < 2 {
                            let (next, timeout) =
                                cvar.wait_timeout(count, Duration::from_secs(1)).unwrap();
                            count = next;
                            if timeout.timed_out() && *count < 2 {
                                return false;
                            }
                        }
                        true
                    })
                })
            })
            .collect();

        for handle in handles {
            assert!(
                handle.join().unwrap(),
                "SharedVoxelData writes to different LOD maps should overlap"
            );
        }
    }

    #[test]
    fn copy_round_trips_through_lod0_with_generator_filling_missing_blocks() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        let channel = ChannelId::Type.index();
        let generator: SharedVoxelGenerator = Arc::new(RecordingGenerator::default());
        data.set_generator(Some(generator));

        // No blocks loaded yet. Copy must invoke the generator for the area.
        let mut dst = VoxelBuffer::with_size(Vector3i::new(16, 16, 16));
        data.copy(Vector3i::zero(), &mut dst, 1u32 << channel, true);

        // RecordingGenerator writes `10 + lod + origin.x`; for block (0,0,0)
        // and lod 0 that is 10. The generator is invoked once per block here.
        assert_eq!(dst.get_voxel(0, 0, 0, channel), 10);
    }

    #[test]
    fn copy_without_generator_returns_defaults_for_missing_blocks() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        let channel = ChannelId::Type.index();
        let mut dst = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        data.copy(Vector3i::zero(), &mut dst, 1u32 << channel, true);

        // No generator and no blocks: dst stays at default (0 for Type).
        assert_eq!(dst.get_voxel(0, 0, 0, channel), 0);
    }

    #[test]
    fn paste_and_paste_masked_route_into_lod0_map() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let channel = ChannelId::Type.index();
        let mask = 1u32 << channel;

        let mut source = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        source.fill(7, channel);
        data.paste(Vector3i::zero(), &source, mask, true);
        assert_eq!(data.get_voxel(Vector3i::new(1, 1, 1), channel, 0), 7);

        // Masked paste: skip voxels equal to the mask sentinel.
        let mut masked_source = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        masked_source.fill(9, channel);
        data.paste_masked(
            Vector3i::zero(),
            &masked_source,
            mask,
            channel,
            9, // skip everything → no writes
            true,
        );
        // Unchanged because every source voxel matched the mask sentinel.
        assert_eq!(data.get_voxel(Vector3i::zero(), channel, 0), 7);
    }

    #[test]
    fn is_area_loaded_reflects_block_residency_and_streaming_bounds() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);

        let area = Box3i::new(Vector3i::zero(), Vector3i::splat(16));
        assert!(!data.is_area_loaded(area));

        assert!(data.try_set_voxel(1, Vector3i::new(1, 1, 1), ChannelId::Type.index()));
        assert!(data.is_area_loaded(area));

        // Streaming-mode short-circuit: area outside bounds returns false.
        data.set_streaming_enabled(true);
        assert!(!data.is_area_loaded(Box3i::new(Vector3i::new(100, 0, 0), Vector3i::splat(16),)));
    }

    #[test]
    fn has_all_blocks_and_get_missing_blocks_agree() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(64)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        assert!(data.try_set_voxel(1, Vector3i::new(1, 1, 1), ChannelId::Type.index()));
        // Block (1,0,0) is intentionally left empty.

        let area = Box3i::new(Vector3i::zero(), Vector3i::new(2, 1, 1));
        assert!(!data.has_all_blocks_in_area(area, 0));

        let mut missing = Vec::new();
        data.get_missing_blocks(area, 0, &mut missing);
        assert_eq!(missing, vec![Vector3i::new(1, 0, 0)]);

        assert!(data.try_set_voxel(2, Vector3i::new(20, 1, 1), ChannelId::Type.index()));
        assert!(data.has_all_blocks_in_area(area, 0));
    }

    #[test]
    fn get_blocks_with_voxel_data_returns_grid_with_empty_slots_for_missing() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(64)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        assert!(data.try_set_voxel(1, Vector3i::new(1, 1, 1), ChannelId::Type.index()));
        // Add an empty (no-voxels) block alongside.
        assert!(data.try_set_block(Vector3i::new(1, 0, 0), VoxelDataBlock::empty(0)));

        let blocks = data
            .get_blocks_with_voxel_data(Box3i::new(Vector3i::zero(), Vector3i::new(2, 1, 1)), 0);
        // ZXY layout: (0,0,0) is index 0, (1,0,0) is index 1.
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].is_some());
        assert!(blocks[1].is_none()); // empty block has no voxel data
    }

    #[test]
    fn get_voxel_falls_back_to_generator_when_block_is_missing_non_streaming() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(64)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let channel = ChannelId::Type.index();

        // RecordingGenerator writes `10 + lod + origin.x`. The default
        // `generate_single` impl passes the queried voxel position as the
        // 1×1×1 block's origin, so for voxel (20,5,5) the result is 10+0+20=30.
        let generator: SharedVoxelGenerator = Arc::new(RecordingGenerator::default());
        data.set_generator(Some(generator));

        let value = data.get_voxel(Vector3i::new(20, 5, 5), channel, 0);
        assert_eq!(value, 30);
    }

    #[test]
    fn get_voxel_returns_defval_when_no_generator_and_block_missing_non_streaming() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(64)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);

        // No generator: the fallback returns the caller-provided default.
        assert_eq!(
            data.get_voxel(Vector3i::new(20, 5, 5), ChannelId::Type.index(), 99),
            99
        );
    }
}
