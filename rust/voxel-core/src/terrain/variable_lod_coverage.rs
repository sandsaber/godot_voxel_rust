//! Persistent, transactional coverage state for variable-LOD terrain.
pub use super::clipbox_coordinator::CoverageFeature;
use super::clipbox_coordinator::DemandCounts;
use crate::constants::voxel_constants::MAX_LOD;
use crate::math::Vector3i;
use crate::meshers::MeshBlockLocation;
use crate::storage::BlockLocation;
use crate::tasks::{TaskPanicPhase, TaskRequestTag};
use imbl::{OrdMap, OrdSet};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptedFeatureSnapshot {
    pub revision: u64,
    pub visuals: bool,
    pub collisions: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FeatureReadiness {
    pub visual_accepted_revision: Option<u64>,
    pub collision_accepted_revision: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoverageInput {
    SetDemand {
        location: MeshBlockLocation,
        counts: DemandCounts,
    },
    Accept {
        location: MeshBlockLocation,
        snapshot: AcceptedFeatureSnapshot,
    },
    Evict {
        location: MeshBlockLocation,
    },
    SetJoinTargetState {
        id: CoverageHoldId,
        state: PendingJoinTargetState,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageInputKind {
    SetDemand,
    Accept,
    Evict,
    SetJoinTargetState,
}

/// Stable identity of one entry-owned load or mesh request.
///
/// This is defined next to coverage temporarily so the typed pending-join
/// state can be integrated without widening the current one-file Task 6
/// slice. The terrain core becomes the producer in the following slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalRequestId {
    Load {
        location: BlockLocation,
        tag: TaskRequestTag,
    },
    Mesh {
        location: MeshBlockLocation,
        tag: TaskRequestTag,
    },
}

impl PhysicalRequestId {
    pub const fn tag(self) -> TaskRequestTag {
        match self {
            Self::Load { tag, .. } | Self::Mesh { tag, .. } => tag,
        }
    }
}

impl Hash for PhysicalRequestId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match *self {
            Self::Load { location, tag } => {
                0_u8.hash(state);
                location.lod_index.hash(state);
                location.position.hash(state);
                tag.hash(state);
            }
            Self::Mesh { location, tag } => {
                1_u8.hash(state);
                location.hash(state);
                tag.hash(state);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestFailureKind {
    Cancelled,
    Panicked(TaskPanicPhase),
    Exhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TransitionFace {
    NegativeX = 0,
    PositiveX = 1,
    NegativeY = 2,
    PositiveY = 3,
    NegativeZ = 4,
    PositiveZ = 5,
}

impl TransitionFace {
    pub const ALL: [Self; 6] = [
        Self::NegativeX,
        Self::PositiveX,
        Self::NegativeY,
        Self::PositiveY,
        Self::NegativeZ,
        Self::PositiveZ,
    ];

    pub const fn normal(self) -> Vector3i {
        match self {
            Self::NegativeX => Vector3i::new(-1, 0, 0),
            Self::PositiveX => Vector3i::new(1, 0, 0),
            Self::NegativeY => Vector3i::new(0, -1, 0),
            Self::PositiveY => Vector3i::new(0, 1, 0),
            Self::NegativeZ => Vector3i::new(0, 0, -1),
            Self::PositiveZ => Vector3i::new(0, 0, 1),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransitionMask(u8);

impl TransitionMask {
    pub const NONE: Self = Self(0);

    pub const fn from_bits(bits: u8) -> Self {
        Self(bits & 0b11_1111)
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn contains(self, face: TransitionFace) -> bool {
        self.0 & (1 << face as u8) != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyOperation {
    RootActivate,
    FrontierActivate,
    Split,
    Join,
    RootDeactivate,
    FrontierDeactivate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureTopologyGroup {
    pub feature: CoverageFeature,
    pub operation: TopologyOperation,
    pub anchor: MeshBlockLocation,
    pub activate: Vec<MeshBlockLocation>,
    pub deactivate: Vec<MeshBlockLocation>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RenderTopologyBatch {
    pub revision: u64,
    pub groups: Vec<FeatureTopologyGroup>,
    pub transition_masks: Vec<(MeshBlockLocation, TransitionMask)>,
}

impl RenderTopologyBatch {
    pub fn visual_activations(&self) -> Vec<MeshBlockLocation> {
        self.feature_locations(CoverageFeature::Visual, true)
    }

    pub fn collision_activations(&self) -> Vec<MeshBlockLocation> {
        self.feature_locations(CoverageFeature::Collision, true)
    }

    pub fn visual_deactivations(&self) -> Vec<MeshBlockLocation> {
        self.feature_locations(CoverageFeature::Visual, false)
    }

    pub fn collision_deactivations(&self) -> Vec<MeshBlockLocation> {
        self.feature_locations(CoverageFeature::Collision, false)
    }

    fn feature_locations(
        &self,
        feature: CoverageFeature,
        activations: bool,
    ) -> Vec<MeshBlockLocation> {
        self.groups
            .iter()
            .filter(|group| group.feature == feature)
            .flat_map(|group| {
                if activations {
                    group.activate.iter()
                } else {
                    group.deactivate.iter()
                }
            })
            .copied()
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoverageHoldId {
    pub join_parent: MeshBlockLocation,
    pub feature: CoverageFeature,
    pub transition_generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageHoldIntentManifest {
    pub id: CoverageHoldId,
    pub target: MeshBlockLocation,
    pub fallbacks: Vec<MeshBlockLocation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageHoldOwnerDelta {
    pub old: Option<CoverageHoldIntentManifest>,
    pub before_topology: Option<CoverageHoldIntentManifest>,
    pub after_topology: Option<CoverageHoldIntentManifest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PendingJoinKey {
    feature_rank: u8,
    lod: u8,
    x: i32,
    y: i32,
    z: i32,
}

impl PendingJoinKey {
    const fn new(target: MeshBlockLocation, feature: CoverageFeature) -> Self {
        Self {
            feature_rank: feature_rank(feature),
            lod: target.lod_index,
            x: target.position_in_blocks.x,
            y: target.position_in_blocks.y,
            z: target.position_in_blocks.z,
        }
    }

    const fn target(self) -> MeshBlockLocation {
        MeshBlockLocation::new(Vector3i::new(self.x, self.y, self.z), self.lod)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Constructed by the terrain-core Task 6 integration seam.
pub(super) enum PendingRequestFailure {
    Retryable {
        request: PhysicalRequestId,
        failure: RequestFailureKind,
        attempt: u32,
    },
    Exhausted {
        request: PhysicalRequestId,
        failure: RequestFailureKind,
    },
}

impl PendingRequestFailure {
    const fn request(self) -> PhysicalRequestId {
        match self {
            Self::Retryable { request, .. } | Self::Exhausted { request, .. } => request,
        }
    }
}

/// Canonical ordered adapter for `BlockLocation`, which intentionally remains
/// non-`Ord`. It is crate-private so physical/core integration can construct a
/// complete snapshot without leaking the adapter into the public storage API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct PendingJoinDataKey {
    lod_index: u8,
    x: i32,
    y: i32,
    z: i32,
}

impl PendingJoinDataKey {
    pub(super) const fn from_location(location: BlockLocation) -> Self {
        Self {
            lod_index: location.lod_index,
            x: location.position.x,
            y: location.position.y,
            z: location.position.z,
        }
    }

    pub(super) const fn to_location(self) -> BlockLocation {
        BlockLocation {
            position: Vector3i::new(self.x, self.y, self.z),
            lod_index: self.lod_index,
        }
    }
}

// Task 6 core integration consumes this crate-private adapter in the next
// bounded slice; keep its construction contract compiled in normal builds.
const _: fn(BlockLocation) -> PendingJoinDataKey = PendingJoinDataKey::from_location;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
enum PendingJoinMemberUniverse {
    #[default]
    Uninitialized,
    CoreValidated(OrdSet<PendingJoinDataKey>),
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct PendingJoinDataState {
    member_universe: PendingJoinMemberUniverse,
    pub(super) ready: OrdSet<PendingJoinDataKey>,
    pub(super) outstanding: OrdMap<PendingJoinDataKey, PhysicalRequestId>,
    pub(super) failures: OrdMap<PendingJoinDataKey, PendingRequestFailure>,
}

impl PendingJoinDataState {
    #[allow(dead_code)] // Terrain-core query seam; unit tests exercise it now.
    pub(super) fn ready_locations(&self) -> Vec<BlockLocation> {
        self.ready
            .iter()
            .copied()
            .map(PendingJoinDataKey::to_location)
            .collect()
    }

    #[allow(dead_code)] // Terrain-core query seam; unit tests exercise it now.
    pub(super) fn all_locations(&self) -> Vec<BlockLocation> {
        self.all_keys()
            .into_iter()
            .map(PendingJoinDataKey::to_location)
            .collect()
    }

    fn all_keys(&self) -> OrdSet<PendingJoinDataKey> {
        self.ready
            .iter()
            .copied()
            .chain(self.outstanding.keys().copied())
            .chain(self.failures.keys().copied())
            .collect()
    }

    fn request_at(&self, key: PendingJoinDataKey) -> Option<PhysicalRequestId> {
        self.failures
            .get(&key)
            .copied()
            .map(PendingRequestFailure::request)
            .or_else(|| self.outstanding.get(&key).copied())
    }

    fn first_request(&self) -> Option<PhysicalRequestId> {
        self.failures
            .iter()
            .next()
            .map(|(_, state)| state.request())
            .or_else(|| self.outstanding.iter().next().map(|(_, request)| *request))
    }

    fn expected_keys(&self) -> Option<&OrdSet<PendingJoinDataKey>> {
        match &self.member_universe {
            PendingJoinMemberUniverse::Uninitialized => None,
            PendingJoinMemberUniverse::CoreValidated(keys) => Some(keys),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Constructed by the terrain-core Task 6 integration seam.
pub(super) enum PendingJoinMeshState {
    AwaitingData,
    AwaitingRequest,
    Meshing {
        request: PhysicalRequestId,
    },
    RetryableFailure {
        request: PhysicalRequestId,
        failure: RequestFailureKind,
        attempt: u32,
    },
    Exhausted {
        request: PhysicalRequestId,
        failure: RequestFailureKind,
    },
    Ready,
}

impl PendingJoinMeshState {
    const fn request(self) -> Option<PhysicalRequestId> {
        match self {
            Self::Meshing { request }
            | Self::RetryableFailure { request, .. }
            | Self::Exhausted { request, .. } => Some(request),
            Self::AwaitingData | Self::AwaitingRequest | Self::Ready => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingJoinTargetState {
    pub(super) data: PendingJoinDataState,
    pub(super) mesh: PendingJoinMeshState,
}

impl Default for PendingJoinTargetState {
    fn default() -> Self {
        Self {
            data: PendingJoinDataState::default(),
            mesh: PendingJoinMeshState::AwaitingData,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Returned by the terrain-core pending-join query seam.
pub(super) enum PendingJoinBlocker {
    DataLoading {
        location: BlockLocation,
        request: PhysicalRequestId,
    },
    DataFailure {
        location: BlockLocation,
        state: PendingRequestFailure,
    },
    Mesh(PendingJoinMeshState),
}

impl PendingJoinTargetState {
    /// Establishes the exact member universe only after the terrain core has
    /// compared it with the resolved hold-ledger halo. Coverage subsequently
    /// proves that every snapshot is a full disjoint partition of this set.
    pub(super) fn from_core_validated_parts(
        expected: OrdSet<PendingJoinDataKey>,
        ready: OrdSet<PendingJoinDataKey>,
        outstanding: OrdMap<PendingJoinDataKey, PhysicalRequestId>,
        failures: OrdMap<PendingJoinDataKey, PendingRequestFailure>,
        mesh: PendingJoinMeshState,
    ) -> Self {
        Self {
            data: PendingJoinDataState {
                member_universe: PendingJoinMemberUniverse::CoreValidated(expected),
                ready,
                outstanding,
                failures,
            },
            mesh,
        }
    }

    #[allow(dead_code)] // Terrain-core query seam; unit tests exercise it now.
    pub(super) fn canonical_blocker(&self) -> Option<PendingJoinBlocker> {
        if let Some((&key, &state)) = self.data.failures.iter().next() {
            return Some(PendingJoinBlocker::DataFailure {
                location: key.to_location(),
                state,
            });
        }
        if let Some((&key, &request)) = self.data.outstanding.iter().next() {
            return Some(PendingJoinBlocker::DataLoading {
                location: key.to_location(),
                request,
            });
        }
        (self.mesh != PendingJoinMeshState::Ready).then_some(PendingJoinBlocker::Mesh(self.mesh))
    }

    fn first_request(&self) -> Option<PhysicalRequestId> {
        self.data.first_request().or_else(|| self.mesh.request())
    }
}

type CoreValidatedTargetStateConstructor = fn(
    OrdSet<PendingJoinDataKey>,
    OrdSet<PendingJoinDataKey>,
    OrdMap<PendingJoinDataKey, PhysicalRequestId>,
    OrdMap<PendingJoinDataKey, PendingRequestFailure>,
    PendingJoinMeshState,
) -> PendingJoinTargetState;
const _: CoreValidatedTargetStateConstructor = PendingJoinTargetState::from_core_validated_parts;

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingJoinIntent {
    id: CoverageHoldId,
    target: MeshBlockLocation,
    feature: CoverageFeature,
    current_manifest: CoverageHoldIntentManifest,
    target_state: PendingJoinTargetState,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CoverageReconcileResult {
    pub topology: RenderTopologyBatch,
    pub hold_deltas: Vec<CoverageHoldOwnerDelta>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CoverageNode {
    demand: DemandCounts,
    // Incremental subtree summaries. A descendant can justify keeping a ready
    // ancestor active even after the ancestor's node-local demand reaches zero.
    branch_resident_nodes: u64,
    branch_visual_nodes: u64,
    branch_collision_nodes: u64,
    // Exact active-node summaries make local topology reconciliation depend on
    // touched ancestor paths rather than every resident node below a root.
    branch_visual_active_nodes: u64,
    branch_collision_active_nodes: u64,
    visual_coverage_owners: u64,
    collision_coverage_owners: u64,
    accepted_snapshot: Option<AcceptedFeatureSnapshot>,
    visual_active: bool,
    collision_active: bool,
    transition_mask: TransitionMask,
}

impl Default for CoverageNode {
    fn default() -> Self {
        Self {
            demand: DemandCounts::default(),
            branch_resident_nodes: 0,
            branch_visual_nodes: 0,
            branch_collision_nodes: 0,
            branch_visual_active_nodes: 0,
            branch_collision_active_nodes: 0,
            visual_coverage_owners: 0,
            collision_coverage_owners: 0,
            accepted_snapshot: None,
            visual_active: false,
            collision_active: false,
            transition_mask: TransitionMask::NONE,
        }
    }
}

impl CoverageNode {
    fn is_active(&self, feature: CoverageFeature) -> bool {
        match feature {
            CoverageFeature::Visual => self.visual_active,
            CoverageFeature::Collision => self.collision_active,
        }
    }

    fn set_active(&mut self, feature: CoverageFeature, active: bool) {
        match feature {
            CoverageFeature::Visual => self.visual_active = active,
            CoverageFeature::Collision => self.collision_active = active,
        }
    }

    fn is_ready(&self, feature: CoverageFeature) -> bool {
        self.accepted_snapshot
            .is_some_and(|snapshot| match feature {
                CoverageFeature::Visual => snapshot.visuals,
                CoverageFeature::Collision => snapshot.collisions,
            })
    }

    fn is_demanded(&self, feature: CoverageFeature) -> bool {
        match feature {
            CoverageFeature::Visual => self.demand.visuals > 0,
            CoverageFeature::Collision => self.demand.collisions > 0,
        }
    }

    fn is_split_demanded(&self, feature: CoverageFeature) -> bool {
        match feature {
            CoverageFeature::Visual => self.demand.visual_splits > 0,
            CoverageFeature::Collision => self.demand.collision_splits > 0,
        }
    }

    fn branch_is_resident(&self) -> bool {
        self.branch_resident_nodes > 0
    }

    fn branch_is_demanded(&self, feature: CoverageFeature) -> bool {
        match feature {
            CoverageFeature::Visual => self.branch_visual_nodes > 0,
            CoverageFeature::Collision => self.branch_collision_nodes > 0,
        }
    }

    fn branch_active_nodes(&self, feature: CoverageFeature) -> u64 {
        match feature {
            CoverageFeature::Visual => self.branch_visual_active_nodes,
            CoverageFeature::Collision => self.branch_collision_active_nodes,
        }
    }

    fn set_branch_active_nodes(&mut self, feature: CoverageFeature, count: u64) {
        match feature {
            CoverageFeature::Visual => self.branch_visual_active_nodes = count,
            CoverageFeature::Collision => self.branch_collision_active_nodes = count,
        }
    }

    fn coverage_owners(&self, feature: CoverageFeature) -> u64 {
        match feature {
            CoverageFeature::Visual => self.visual_coverage_owners,
            CoverageFeature::Collision => self.collision_coverage_owners,
        }
    }

    fn set_coverage_owners(&mut self, feature: CoverageFeature, count: u64) {
        match feature {
            CoverageFeature::Visual => self.visual_coverage_owners = count,
            CoverageFeature::Collision => self.collision_coverage_owners = count,
        }
    }

    fn readiness(&self) -> FeatureReadiness {
        self.accepted_snapshot
            .map_or_else(FeatureReadiness::default, |snapshot| FeatureReadiness {
                visual_accepted_revision: snapshot.visuals.then_some(snapshot.revision),
                collision_accepted_revision: snapshot.collisions.then_some(snapshot.revision),
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CoverageKey {
    lod: u8,
    x: i32,
    y: i32,
    z: i32,
}

impl CoverageKey {
    const fn from_location(location: MeshBlockLocation) -> Self {
        Self {
            lod: location.lod_index,
            x: location.position_in_blocks.x,
            y: location.position_in_blocks.y,
            z: location.position_in_blocks.z,
        }
    }

    const fn location(self) -> MeshBlockLocation {
        MeshBlockLocation::new(Vector3i::new(self.x, self.y, self.z), self.lod)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CoverageState {
    revision: u64,
    next_transition_generation: u64,
    nodes: OrdMap<CoverageKey, CoverageNode>,
    // Persistent canonical frontiers let release previews replay topology from
    // the exact old active sets without scanning every coverage node.
    visual_active: OrdSet<CoverageKey>,
    collision_active: OrdSet<CoverageKey>,
    pending_joins: OrdMap<PendingJoinKey, PendingJoinIntent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariableLodCoverage {
    lod_count: u8,
    state: Arc<CoverageState>,
}

#[derive(Debug)]
pub(super) struct CoveragePreview {
    base: Arc<CoverageState>,
    next: Arc<CoverageState>,
    result: CoverageReconcileResult,
    work_counters: CoverageWorkCounters,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CoverageWorkCounters {
    pub nodes_visited: usize,
    pub ancestor_links_checked: usize,
    pub groups_validated: usize,
    pub full_state_iterations: usize,
    pub pending_owners_examined: usize,
}

impl CoveragePreview {
    pub const fn result(&self) -> &CoverageReconcileResult {
        &self.result
    }

    pub fn base_revision(&self) -> u64 {
        self.base.revision
    }

    pub fn next_revision(&self) -> u64 {
        self.next.revision
    }

    /// Returns the exact accepted feature snapshot in the projected state.
    pub(super) fn next_accepted_snapshot(
        &self,
        location: MeshBlockLocation,
    ) -> Option<AcceptedFeatureSnapshot> {
        self.next
            .nodes
            .get(&CoverageKey::from_location(location))
            .and_then(|node| node.accepted_snapshot)
    }

    /// Returns whether one feature is active at `location` in the projected state.
    pub(super) fn next_is_active(
        &self,
        location: MeshBlockLocation,
        feature: CoverageFeature,
    ) -> bool {
        active_set(&self.next, feature).contains(&CoverageKey::from_location(location))
    }

    #[cfg(test)]
    fn next_active_locations(&self, feature: CoverageFeature) -> Vec<MeshBlockLocation> {
        active_locations_in(&self.next, feature)
    }

    pub(super) const fn work_counters(&self) -> CoverageWorkCounters {
        self.work_counters
    }

    fn validate_base_identity(
        &self,
        coverage: &VariableLodCoverage,
    ) -> Result<(), CoverageInvariantError> {
        if !Arc::ptr_eq(&coverage.state, &self.base) {
            return Err(CoverageInvariantError::StalePreviewIdentity);
        }
        Ok(())
    }

    /// Consumes this preview after proving that it still targets the exact
    /// live coverage snapshot. The returned token is safe to retain across a
    /// wider preflight, but must pass
    /// [`ValidatedCoveragePreview::revalidate_for`] immediately before the
    /// transaction enters its publication fence.
    pub(super) fn validate_for(
        self,
        coverage: &VariableLodCoverage,
    ) -> Result<ValidatedCoveragePreview, CoverageInvariantError> {
        self.validate_base_identity(coverage)?;
        Ok(ValidatedCoveragePreview(self))
    }
}

/// Identity-validated coverage preview ready for infallible publication.
#[derive(Debug)]
pub(super) struct ValidatedCoveragePreview(CoveragePreview);

impl ValidatedCoveragePreview {
    pub(super) const fn result(&self) -> &CoverageReconcileResult {
        self.0.result()
    }

    pub(super) fn next_accepted_snapshot(
        &self,
        location: MeshBlockLocation,
    ) -> Option<AcceptedFeatureSnapshot> {
        self.0.next_accepted_snapshot(location)
    }

    pub(super) fn next_is_active(
        &self,
        location: MeshBlockLocation,
        feature: CoverageFeature,
    ) -> bool {
        self.0.next_is_active(location, feature)
    }

    pub(super) const fn work_counters(&self) -> CoverageWorkCounters {
        self.0.work_counters()
    }

    /// Rechecks exact base identity without consuming the validated token.
    /// This is the release-safe final gate immediately before a wider storage
    /// transaction enters its publication fence.
    pub(super) fn revalidate_for(
        &self,
        coverage: &VariableLodCoverage,
    ) -> Result<(), CoverageInvariantError> {
        self.0.validate_base_identity(coverage)
    }

    /// Moves the next state into the live coverage and returns all old/base
    /// Arc owners for retirement after the enclosing publication fence. The
    /// caller must have successfully called [`Self::revalidate_for`] as its
    /// immediately preceding fallible identity gate and must not mutate the
    /// coverage before this infallible publication step.
    pub(super) fn publish(self, coverage: &mut VariableLodCoverage) -> PublishedCoveragePreview {
        let CoveragePreview {
            base,
            next,
            result,
            work_counters: _,
        } = self.0;
        debug_assert!(Arc::ptr_eq(&coverage.state, &base));
        let live = std::mem::replace(&mut coverage.state, next);
        PublishedCoveragePreview {
            result,
            retirement: CoverageStateRetirement {
                _live: live,
                _prepared_base: base,
            },
        }
    }
}

const _: for<'a> fn(&'a ValidatedCoveragePreview) -> &'a CoverageReconcileResult =
    ValidatedCoveragePreview::result;
const _: fn(&ValidatedCoveragePreview, MeshBlockLocation) -> Option<AcceptedFeatureSnapshot> =
    ValidatedCoveragePreview::next_accepted_snapshot;
const _: fn(&ValidatedCoveragePreview, MeshBlockLocation, CoverageFeature) -> bool =
    ValidatedCoveragePreview::next_is_active;
const _: fn(&ValidatedCoveragePreview) -> CoverageWorkCounters =
    ValidatedCoveragePreview::work_counters;
const _: fn(&ValidatedCoveragePreview, &VariableLodCoverage) -> Result<(), CoverageInvariantError> =
    ValidatedCoveragePreview::revalidate_for;

#[derive(Debug)]
pub(super) struct PublishedCoveragePreview {
    result: CoverageReconcileResult,
    retirement: CoverageStateRetirement,
}

impl PublishedCoveragePreview {
    pub(super) fn into_parts(self) -> (CoverageReconcileResult, CoverageStateRetirement) {
        (self.result, self.retirement)
    }
}

/// Opaque ownership that must outlive any enclosing publication fence.
#[derive(Debug)]
pub(super) struct CoverageStateRetirement {
    _live: Arc<CoverageState>,
    _prepared_base: Arc<CoverageState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoverageInvariantError {
    InvalidLodCount,
    InvalidLodIndex {
        location: MeshBlockLocation,
    },
    InvalidDemand {
        location: MeshBlockLocation,
        counts: DemandCounts,
    },
    DuplicateInput {
        location: MeshBlockLocation,
        kind: CoverageInputKind,
    },
    InvalidEviction {
        location: MeshBlockLocation,
    },
    UnknownAcceptedLocation(MeshBlockLocation),
    InvalidAcceptedSnapshot {
        location: MeshBlockLocation,
    },
    CoordinateOverflow {
        location: MeshBlockLocation,
    },
    RevisionOverflow,
    HoldGenerationOverflow,
    StalePreviewIdentity,
    InvalidPartition,
    InvalidTopologyGroup {
        group_index: usize,
    },
    InvalidHoldOwner(CoverageHoldId),
    MissingJoinTargetMember {
        id: CoverageHoldId,
        location: BlockLocation,
    },
    StaleJoinTargetRequest {
        id: CoverageHoldId,
        request: PhysicalRequestId,
    },
}

#[derive(Default)]
struct NormalizedInputs {
    demands: BTreeMap<CoverageKey, DemandCounts>,
    acceptances: BTreeMap<CoverageKey, AcceptedFeatureSnapshot>,
    evictions: BTreeSet<CoverageKey>,
    join_target_states: BTreeMap<PendingJoinKey, (CoverageHoldId, PendingJoinTargetState)>,
}

impl VariableLodCoverage {
    pub fn try_new(lod_count: u8) -> Result<Self, CoverageInvariantError> {
        if lod_count == 0 || usize::from(lod_count) >= MAX_LOD {
            return Err(CoverageInvariantError::InvalidLodCount);
        }
        Ok(Self {
            lod_count,
            state: Arc::new(CoverageState {
                revision: 0,
                next_transition_generation: 0,
                nodes: OrdMap::new(),
                visual_active: OrdSet::new(),
                collision_active: OrdSet::new(),
                pending_joins: OrdMap::new(),
            }),
        })
    }

    pub const fn lod_count(&self) -> u8 {
        self.lod_count
    }

    pub fn revision(&self) -> u64 {
        self.state.revision
    }

    #[cfg(test)]
    pub(super) fn state_identity_for_test(&self) -> usize {
        Arc::as_ptr(&self.state) as usize
    }

    pub fn contains_node(&self, location: MeshBlockLocation) -> bool {
        self.state
            .nodes
            .contains_key(&CoverageKey::from_location(location))
    }

    pub fn is_resident(&self, location: MeshBlockLocation) -> bool {
        self.state
            .nodes
            .get(&CoverageKey::from_location(location))
            .is_some_and(CoverageNode::branch_is_resident)
    }

    pub fn is_active(&self, location: MeshBlockLocation, feature: CoverageFeature) -> bool {
        self.state
            .nodes
            .get(&CoverageKey::from_location(location))
            .is_some_and(|node| node.is_active(feature))
    }

    pub fn readiness(&self, location: MeshBlockLocation) -> FeatureReadiness {
        self.state
            .nodes
            .get(&CoverageKey::from_location(location))
            .map_or_else(FeatureReadiness::default, CoverageNode::readiness)
    }

    pub fn active_locations(&self, feature: CoverageFeature) -> Vec<MeshBlockLocation> {
        active_locations_in(&self.state, feature)
    }

    pub fn pending_join_id(
        &self,
        target: MeshBlockLocation,
        feature: CoverageFeature,
    ) -> Option<CoverageHoldId> {
        self.state
            .pending_joins
            .get(&PendingJoinKey::new(target, feature))
            .map(|intent| intent.id)
    }

    pub fn pending_join_manifest(
        &self,
        target: MeshBlockLocation,
        feature: CoverageFeature,
    ) -> Option<&CoverageHoldIntentManifest> {
        self.state
            .pending_joins
            .get(&PendingJoinKey::new(target, feature))
            .map(|intent| &intent.current_manifest)
    }

    pub(super) fn pending_join_target_state(
        &self,
        target: MeshBlockLocation,
        feature: CoverageFeature,
    ) -> Option<&PendingJoinTargetState> {
        self.state
            .pending_joins
            .get(&PendingJoinKey::new(target, feature))
            .map(|intent| &intent.target_state)
    }

    pub fn validate_partition(
        &self,
        feature: CoverageFeature,
    ) -> Result<(), CoverageInvariantError> {
        validate_partition_in(self.lod_count, &self.state, feature, true)
    }

    pub(super) fn preview_reconcile(
        &self,
        inputs: &[CoverageInput],
    ) -> Result<CoveragePreview, CoverageInvariantError> {
        let normalized = normalize_inputs(self.lod_count, inputs)?;
        let mut work = CoverageWorkCounters::default();
        let (input_pending_joins, target_state_changed) =
            apply_join_target_state_inputs(&self.state, &normalized.join_target_states, &mut work)?;
        // Target-state inputs alter only the persistent owner map. Keeping a
        // separate persistent draft lets later manifest replacement reuse the
        // exact state while topology replay continues from the live base.
        let owner_input_state = CoverageState {
            revision: self.state.revision,
            next_transition_generation: self.state.next_transition_generation,
            nodes: self.state.nodes.clone(),
            visual_active: self.state.visual_active.clone(),
            collision_active: self.state.collision_active.clone(),
            pending_joins: input_pending_joins,
        };
        let mut next_nodes = self.state.nodes.clone();
        let mut logical_change = target_state_changed;
        let mut affected_roots = BTreeSet::new();
        let mut affected_split_anchors = BTreeSet::new();

        for (&key, &counts) in &normalized.demands {
            affected_roots.insert(coarsest_root(key, self.lod_count, &mut work)?);
            collect_ancestor_keys(key, self.lod_count, &mut affected_split_anchors, &mut work)?;
            let old_demand = next_nodes
                .get(&key)
                .map_or_else(DemandCounts::default, |node| node.demand);
            if old_demand != counts {
                apply_demand_update(
                    &mut next_nodes,
                    key,
                    old_demand,
                    counts,
                    self.lod_count,
                    &mut work,
                )?;
                logical_change = true;
            }
        }

        validate_known_acceptance_locations(&next_nodes, &normalized.acceptances)?;
        validate_accepted_snapshots(&next_nodes, &normalized.acceptances)?;
        for (&key, &snapshot) in &normalized.acceptances {
            work.nodes_visited += 1;
            affected_roots.insert(coarsest_root(key, self.lod_count, &mut work)?);
            collect_ancestor_keys(key, self.lod_count, &mut affected_split_anchors, &mut work)?;
            let Some(mut node) = next_nodes.get(&key).cloned() else {
                return Err(CoverageInvariantError::UnknownAcceptedLocation(
                    key.location(),
                ));
            };
            if node.accepted_snapshot != Some(snapshot) {
                node.accepted_snapshot = Some(snapshot);
                next_nodes.insert(key, node);
                logical_change = true;
            }
        }

        work.nodes_visited += normalized.evictions.len();

        #[cfg(test)]
        if inputs.is_empty() {
            for (&key, _) in &next_nodes {
                if key.lod + 1 == self.lod_count {
                    affected_roots.insert(key);
                }
            }
        }

        let mut groups = Vec::new();
        for &root in &affected_roots {
            for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
                let Some(node) = next_nodes.get(&root) else {
                    continue;
                };
                let active = active_set(&self.state, feature);
                let root_is_active = active.contains(&root);
                let branch_active_nodes = node.branch_active_nodes(feature);
                if root_is_active && branch_active_nodes != 1 {
                    return Err(CoverageInvariantError::InvalidPartition);
                }
                let root_is_eligible = node_is_eligible(node, feature);
                if branch_active_nodes == 0 {
                    if root_is_eligible {
                        groups.push(root_group(feature, root.location(), true));
                    }
                } else if root_is_active {
                    if !root_is_eligible {
                        groups.push(root_group(feature, root.location(), false));
                    }
                } else if !node.branch_is_resident() || !node.branch_is_demanded(feature) {
                    // A split frontier with no remaining viewer is still the
                    // only valid visual/collision fallback while its parent
                    // join target is unready. Task 6 turns that state into a
                    // persistent logical owner below instead of creating a
                    // hole by deactivating the frontier here.
                }
            }
        }

        let mut projected_nodes = next_nodes.clone();
        let mut projected_visual_active = self.state.visual_active.clone();
        let mut projected_collision_active = self.state.collision_active.clone();
        project_topology_groups(
            &mut projected_nodes,
            &groups,
            &mut projected_visual_active,
            &mut projected_collision_active,
            self.lod_count,
            &mut work,
        )?;
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let active = match feature {
                CoverageFeature::Visual => &mut projected_visual_active,
                CoverageFeature::Collision => &mut projected_collision_active,
            };
            for &anchor in &affected_split_anchors {
                if active.contains(&anchor) {
                    continue;
                }
                let Some(node) = projected_nodes.get(&anchor) else {
                    return Err(CoverageInvariantError::InvalidPartition);
                };
                if node.branch_active_nodes(feature) == 0 {
                    if node_is_eligible(node, feature)
                        && sparse_gap_can_activate(
                            &self.state,
                            active,
                            anchor,
                            feature,
                            self.lod_count,
                            &mut work,
                        )?
                    {
                        let group = frontier_activate_group(feature, anchor.location());
                        project_topology_group(
                            &mut projected_nodes,
                            active,
                            &group,
                            self.lod_count,
                            &mut work,
                        )?;
                        groups.push(group);
                    }
                    continue;
                }
                if anchor.lod == 0 || !node.is_ready(feature) {
                    continue;
                }
                if !node.is_split_demanded(feature) {
                    let mut draft_nodes = projected_nodes.clone();
                    let mut draft_active = active.clone();
                    let mut join_groups = Vec::new();
                    if !prepare_exact_join_chain(
                        &mut draft_nodes,
                        &mut draft_active,
                        anchor,
                        feature,
                        self.lod_count,
                        &mut join_groups,
                        &mut work,
                    )? {
                        continue;
                    }
                    projected_nodes = draft_nodes;
                    *active = draft_active;
                    groups.extend(join_groups);
                    continue;
                }
                let Some(children) = exact_active_children(
                    &projected_nodes,
                    active,
                    anchor,
                    feature,
                    self.lod_count,
                    &mut work,
                )?
                else {
                    continue;
                };
                if children.iter().all(|location| {
                    projected_nodes
                        .get(&CoverageKey::from_location(*location))
                        .is_some_and(|child| node_is_eligible(child, feature))
                }) {
                    continue;
                }
                let group = join_group(feature, anchor.location(), children);
                project_topology_group(
                    &mut projected_nodes,
                    active,
                    &group,
                    self.lod_count,
                    &mut work,
                )?;
                groups.push(group);
            }

            let mut split_anchors = affected_split_anchors.clone();
            loop {
                let mut split_groups = Vec::new();
                for &anchor in &split_anchors {
                    if !active.contains(&anchor) {
                        continue;
                    }
                    if let Some(group) = ready_split_group(
                        &projected_nodes,
                        active,
                        anchor,
                        feature,
                        self.lod_count,
                        &mut work,
                    )? {
                        split_groups.push(group);
                    }
                }
                if split_groups.is_empty() {
                    break;
                }
                split_groups.sort_by_key(group_sort_key);
                for group in &split_groups {
                    project_topology_group(
                        &mut projected_nodes,
                        active,
                        group,
                        self.lod_count,
                        &mut work,
                    )?;
                    for child in group.activate.iter().copied() {
                        let child = CoverageKey::from_location(child);
                        // A child that was already fully ready before its
                        // parent split becomes a new active split candidate.
                        // Expanding only through the eight activated children
                        // preserves incremental work without a tree-wide scan.
                        split_anchors.insert(child);
                    }
                }
                groups.extend(split_groups);
            }
        }

        // If a zero-viewer join reached an active coarsest target in this
        // preview, retire that target only after its parent-first Join group.
        // The join owner installed below protects the full sequence.
        for &root in &affected_roots {
            for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
                let active = match feature {
                    CoverageFeature::Visual => &mut projected_visual_active,
                    CoverageFeature::Collision => &mut projected_collision_active,
                };
                let Some(node) = projected_nodes.get(&root) else {
                    continue;
                };
                if active.contains(&root) && !node.branch_is_demanded(feature) {
                    let group = root_group(feature, root.location(), false);
                    project_topology_group(
                        &mut projected_nodes,
                        active,
                        &group,
                        self.lod_count,
                        &mut work,
                    )?;
                    groups.push(group);
                }
            }
        }

        groups.sort_by_key(group_sort_key);
        let owner_preparation = prepare_join_owners(
            JoinOwnerPreparationInput {
                state: &owner_input_state,
                projected_nodes: &projected_nodes,
                projected_visual_active: &projected_visual_active,
                projected_collision_active: &projected_collision_active,
                affected_anchors: &affected_split_anchors,
                groups: &groups,
                lod_count: self.lod_count,
            },
            &mut work,
        )?;
        let mut before_topology_nodes = next_nodes;
        apply_owner_phase(&mut before_topology_nodes, &owner_preparation.deltas, true)?;
        let (mut next_nodes, next_visual_active, next_collision_active) = replay_topology_groups(
            self.lod_count,
            &self.state,
            before_topology_nodes,
            &groups,
            &mut work,
        )?;
        apply_owner_phase(&mut next_nodes, &owner_preparation.deltas, false)?;
        logical_change |= !groups.is_empty() || !owner_preparation.deltas.is_empty();

        for &key in &normalized.evictions {
            let Some(node) = next_nodes.get(&key) else {
                return Err(CoverageInvariantError::InvalidEviction {
                    location: key.location(),
                });
            };
            if node.demand.resident != 0
                || node.branch_is_resident()
                || node.visual_active
                || node.collision_active
                || node.visual_coverage_owners != 0
                || node.collision_coverage_owners != 0
            {
                return Err(CoverageInvariantError::InvalidEviction {
                    location: key.location(),
                });
            }
        }
        for key in normalized.evictions {
            next_nodes.remove(&key);
            logical_change = true;
        }

        let next_revision = if logical_change {
            self.state
                .revision
                .checked_add(1)
                .ok_or(CoverageInvariantError::RevisionOverflow)?
        } else {
            self.state.revision
        };
        let next = if logical_change {
            Arc::new(CoverageState {
                revision: next_revision,
                next_transition_generation: owner_preparation.next_transition_generation,
                nodes: next_nodes,
                visual_active: next_visual_active,
                collision_active: next_collision_active,
                pending_joins: owner_preparation.pending_joins,
            })
        } else {
            Arc::clone(&self.state)
        };
        validate_after_owner_endpoints(
            self.lod_count,
            &next,
            &owner_preparation.deltas,
            &mut work,
        )?;

        #[cfg(debug_assertions)]
        if !groups.is_empty() || !owner_preparation.deltas.is_empty() {
            let draft = Self {
                lod_count: self.lod_count,
                state: Arc::clone(&next),
            };
            draft.validate_partition(CoverageFeature::Visual)?;
            draft.validate_partition(CoverageFeature::Collision)?;
        }

        Ok(CoveragePreview {
            base: Arc::clone(&self.state),
            next,
            result: CoverageReconcileResult {
                topology: RenderTopologyBatch {
                    revision: 0,
                    groups,
                    transition_masks: Vec::new(),
                },
                hold_deltas: owner_preparation.deltas,
            },
            work_counters: work,
        })
    }

    /// Builds the coverage half of an owner-only terminal join failure.
    ///
    /// The terrain core must call this only after it has validated the matching
    /// physical target state. Coverage proves the logical half here: the
    /// anchored branch has no remaining feature demand, the deactivation is
    /// its exact active antichain, successor owners are reduced canonically,
    /// and every owner remains installed through the topology phase.
    pub(super) fn preview_terminal_pending_join_failure(
        &self,
        id: CoverageHoldId,
    ) -> Result<CoveragePreview, CoverageInvariantError> {
        let key = PendingJoinKey::new(id.join_parent, id.feature);
        let Some(intent) = self.state.pending_joins.get(&key) else {
            return Err(CoverageInvariantError::InvalidHoldOwner(id));
        };
        if intent.id != id {
            return Err(CoverageInvariantError::InvalidHoldOwner(id));
        }
        let anchor = CoverageKey::from_location(intent.target);
        let Some(anchor_node) = self.state.nodes.get(&anchor) else {
            return Err(CoverageInvariantError::InvalidHoldOwner(id));
        };
        // `valid_demand` enforces split <= feature demand at every node, so a
        // zero feature-demand subtree summary is also an exact zero split-
        // demand proof for the complete anchored branch.
        if anchor_node.branch_is_demanded(id.feature) {
            return Err(CoverageInvariantError::InvalidHoldOwner(id));
        }

        let mut work = CoverageWorkCounters::default();
        validate_owner_endpoint(self.lod_count, &self.state, intent, &mut work)?;
        let frontier = canonical_active_frontier_under(
            &self.state.nodes,
            active_set(&self.state, id.feature),
            anchor,
            id.feature,
            self.lod_count,
            &mut work,
        )?;
        if frontier != intent.current_manifest.fallbacks {
            return Err(CoverageInvariantError::InvalidHoldOwner(id));
        }
        let group = frontier_deactivate_group(id.feature, intent.target, frontier.clone());

        let mut pending_joins = self.state.pending_joins.clone();
        pending_joins.remove(&key);
        let mut deltas = vec![CoverageHoldOwnerDelta {
            old: Some(intent.current_manifest.clone()),
            before_topology: Some(intent.current_manifest.clone()),
            after_topology: None,
        }];

        let mut ancestor = checked_parent(intent.target, self.lod_count)?;
        while let Some(location) = ancestor {
            work.ancestor_links_checked += 1;
            let owner_key = PendingJoinKey::new(location, id.feature);
            work.pending_owners_examined += 1;
            if let Some(owner) = self.state.pending_joins.get(&owner_key) {
                validate_owner_endpoint(self.lod_count, &self.state, owner, &mut work)?;
                let mut after_fallbacks = Vec::new();
                for &fallback in &owner.current_manifest.fallbacks {
                    if !location_is_under(fallback, intent.target, self.lod_count)? {
                        after_fallbacks.push(fallback);
                    }
                }
                let after = (!after_fallbacks.is_empty()).then_some(CoverageHoldIntentManifest {
                    id: owner.id,
                    target: owner.target,
                    fallbacks: after_fallbacks,
                });
                if after.as_ref() != Some(&owner.current_manifest) {
                    deltas.push(CoverageHoldOwnerDelta {
                        old: Some(owner.current_manifest.clone()),
                        before_topology: Some(owner.current_manifest.clone()),
                        after_topology: after.clone(),
                    });
                    if let Some(after) = after {
                        pending_joins.insert(
                            owner_key,
                            PendingJoinIntent {
                                id: owner.id,
                                target: owner.target,
                                feature: owner.feature,
                                current_manifest: after,
                                target_state: owner.target_state.clone(),
                            },
                        );
                    } else {
                        pending_joins.remove(&owner_key);
                    }
                }
            }
            ancestor = checked_parent(location, self.lod_count)?;
        }
        deltas.sort_by_key(owner_delta_sort_key);

        let mut before_nodes = self.state.nodes.clone();
        apply_owner_phase(&mut before_nodes, &deltas, true)?;
        let (mut next_nodes, next_visual_active, next_collision_active) = replay_topology_groups(
            self.lod_count,
            &self.state,
            before_nodes,
            std::slice::from_ref(&group),
            &mut work,
        )?;
        apply_owner_phase(&mut next_nodes, &deltas, false)?;
        let next_revision = self
            .state
            .revision
            .checked_add(1)
            .ok_or(CoverageInvariantError::RevisionOverflow)?;
        let next = Arc::new(CoverageState {
            revision: next_revision,
            next_transition_generation: self.state.next_transition_generation,
            nodes: next_nodes,
            visual_active: next_visual_active,
            collision_active: next_collision_active,
            pending_joins,
        });
        validate_after_owner_endpoints(self.lod_count, &next, &deltas, &mut work)?;

        #[cfg(debug_assertions)]
        {
            let draft = Self {
                lod_count: self.lod_count,
                state: Arc::clone(&next),
            };
            draft.validate_partition(CoverageFeature::Visual)?;
            draft.validate_partition(CoverageFeature::Collision)?;
        }

        Ok(CoveragePreview {
            base: Arc::clone(&self.state),
            next,
            result: CoverageReconcileResult {
                topology: RenderTopologyBatch {
                    revision: 0,
                    groups: vec![group],
                    transition_masks: Vec::new(),
                },
                hold_deltas: deltas,
            },
            work_counters: work,
        })
    }

    pub(super) fn apply_preview(
        &mut self,
        preview: CoveragePreview,
    ) -> Result<CoverageReconcileResult, CoverageInvariantError> {
        let validated = preview.validate_for(self)?;
        let (result, retirement) = validated.publish(self).into_parts();
        drop(retirement);
        Ok(result)
    }

    #[cfg(test)]
    fn at_revision_for_test(lod_count: u8, revision: u64) -> Self {
        let mut coverage = Self::try_new(lod_count).unwrap();
        coverage.state = Arc::new(CoverageState {
            revision,
            next_transition_generation: 0,
            nodes: OrdMap::new(),
            visual_active: OrdSet::new(),
            collision_active: OrdSet::new(),
            pending_joins: OrdMap::new(),
        });
        coverage
    }

    #[cfg(test)]
    fn force_active_for_test(
        &mut self,
        location: MeshBlockLocation,
        feature: CoverageFeature,
        active: bool,
    ) {
        let key = CoverageKey::from_location(location);
        let mut nodes = self.state.nodes.clone();
        let mut visual_active = self.state.visual_active.clone();
        let mut collision_active = self.state.collision_active.clone();
        let active_set = match feature {
            CoverageFeature::Visual => &mut visual_active,
            CoverageFeature::Collision => &mut collision_active,
        };
        set_topology_active(
            &mut nodes,
            active_set,
            key,
            feature,
            active,
            self.lod_count,
            &mut CoverageWorkCounters::default(),
        )
        .unwrap();
        self.state = Arc::new(CoverageState {
            revision: self.state.revision,
            next_transition_generation: self.state.next_transition_generation,
            nodes,
            visual_active,
            collision_active,
            pending_joins: self.state.pending_joins.clone(),
        });
    }

    #[cfg(test)]
    fn force_demand_for_test(&mut self, location: MeshBlockLocation, demand: DemandCounts) {
        let key = CoverageKey::from_location(location);
        let mut nodes = self.state.nodes.clone();
        let mut node = nodes.get(&key).cloned().unwrap();
        node.demand = demand;
        node.branch_resident_nodes = u64::from(demand.resident > 0);
        node.branch_visual_nodes = u64::from(demand.visuals > 0);
        node.branch_collision_nodes = u64::from(demand.collisions > 0);
        nodes.insert(key, node);
        self.state = Arc::new(CoverageState {
            revision: self.state.revision,
            next_transition_generation: self.state.next_transition_generation,
            nodes,
            visual_active: self.state.visual_active.clone(),
            collision_active: self.state.collision_active.clone(),
            pending_joins: self.state.pending_joins.clone(),
        });
    }
}

const _: fn(
    &VariableLodCoverage,
    CoverageHoldId,
) -> Result<CoveragePreview, CoverageInvariantError> =
    VariableLodCoverage::preview_terminal_pending_join_failure;
const _: for<'a> fn(
    &'a VariableLodCoverage,
    MeshBlockLocation,
    CoverageFeature,
) -> Option<&'a PendingJoinTargetState> = VariableLodCoverage::pending_join_target_state;

fn normalize_inputs(
    lod_count: u8,
    inputs: &[CoverageInput],
) -> Result<NormalizedInputs, CoverageInvariantError> {
    let invalid_lod = inputs
        .iter()
        .map(input_location)
        .filter(|location| location.lod_index >= lod_count)
        .map(CoverageKey::from_location)
        .min();
    if let Some(key) = invalid_lod {
        return Err(CoverageInvariantError::InvalidLodIndex {
            location: key.location(),
        });
    }

    let mut normalized = NormalizedInputs::default();
    let mut duplicate_demands = BTreeSet::new();
    let mut duplicate_acceptances = BTreeSet::new();
    let mut duplicate_evictions = BTreeSet::new();
    let mut duplicate_join_target_states = BTreeSet::new();
    for input in inputs {
        match input {
            CoverageInput::SetDemand { location, counts } => {
                let key = CoverageKey::from_location(*location);
                if normalized.demands.insert(key, *counts).is_some() {
                    duplicate_demands.insert(key);
                }
            }
            CoverageInput::Accept { location, snapshot } => {
                let key = CoverageKey::from_location(*location);
                if normalized.acceptances.insert(key, *snapshot).is_some() {
                    duplicate_acceptances.insert(key);
                }
            }
            CoverageInput::Evict { location } => {
                let key = CoverageKey::from_location(*location);
                if !normalized.evictions.insert(key) {
                    duplicate_evictions.insert(key);
                }
            }
            CoverageInput::SetJoinTargetState { id, state } => {
                let key = PendingJoinKey::new(id.join_parent, id.feature);
                if normalized
                    .join_target_states
                    .insert(key, (*id, state.clone()))
                    .is_some()
                {
                    duplicate_join_target_states.insert(key);
                }
            }
        }
    }
    if let Some(key) = duplicate_demands.into_iter().next() {
        return Err(CoverageInvariantError::DuplicateInput {
            location: key.location(),
            kind: CoverageInputKind::SetDemand,
        });
    }
    if let Some(key) = duplicate_acceptances.into_iter().next() {
        return Err(CoverageInvariantError::DuplicateInput {
            location: key.location(),
            kind: CoverageInputKind::Accept,
        });
    }
    if let Some(key) = duplicate_evictions.into_iter().next() {
        return Err(CoverageInvariantError::DuplicateInput {
            location: key.location(),
            kind: CoverageInputKind::Evict,
        });
    }
    if let Some(key) = duplicate_join_target_states.into_iter().next() {
        return Err(CoverageInvariantError::DuplicateInput {
            location: key.target(),
            kind: CoverageInputKind::SetJoinTargetState,
        });
    }
    if let Some((&key, &counts)) = normalized
        .demands
        .iter()
        .find(|(_, counts)| !valid_demand(**counts))
    {
        return Err(CoverageInvariantError::InvalidDemand {
            location: key.location(),
            counts,
        });
    }
    Ok(normalized)
}

const fn input_location(input: &CoverageInput) -> MeshBlockLocation {
    match input {
        CoverageInput::SetDemand { location, .. }
        | CoverageInput::Accept { location, .. }
        | CoverageInput::Evict { location } => *location,
        CoverageInput::SetJoinTargetState { id, .. } => id.join_parent,
    }
}

fn apply_join_target_state_inputs(
    state: &CoverageState,
    inputs: &BTreeMap<PendingJoinKey, (CoverageHoldId, PendingJoinTargetState)>,
    work: &mut CoverageWorkCounters,
) -> Result<(OrdMap<PendingJoinKey, PendingJoinIntent>, bool), CoverageInvariantError> {
    let mut pending_joins = state.pending_joins.clone();
    let mut changed = false;
    for (&key, (id, target_state)) in inputs {
        work.pending_owners_examined += 1;
        let Some(intent) = state.pending_joins.get(&key) else {
            return Err(stale_owner_or_request(*id, target_state));
        };
        if intent.id != *id
            || intent.target != id.join_parent
            || intent.feature != id.feature
            || key != PendingJoinKey::new(intent.target, intent.feature)
        {
            return Err(stale_owner_or_request(*id, target_state));
        }
        validate_join_target_state(*id, intent, target_state)?;
        if intent.target_state != *target_state {
            let mut next_intent = intent.clone();
            next_intent.target_state = target_state.clone();
            pending_joins.insert(key, next_intent);
            changed = true;
        }
    }
    Ok((pending_joins, changed))
}

fn stale_owner_or_request(
    id: CoverageHoldId,
    state: &PendingJoinTargetState,
) -> CoverageInvariantError {
    state
        .first_request()
        .map_or(CoverageInvariantError::InvalidHoldOwner(id), |request| {
            CoverageInvariantError::StaleJoinTargetRequest { id, request }
        })
}

fn validate_join_target_state(
    id: CoverageHoldId,
    intent: &PendingJoinIntent,
    next: &PendingJoinTargetState,
) -> Result<(), CoverageInvariantError> {
    if let Err(error) = validate_join_target_snapshot(id, intent.target, next, false) {
        if matches!(
            error,
            CoverageInvariantError::InvalidHoldOwner(actual_id)
                | CoverageInvariantError::MissingJoinTargetMember { id: actual_id, .. }
                if actual_id == id
        ) {
            validate_join_target_transition(id, &intent.target_state, next)?;
        }
        return Err(error);
    }
    validate_join_target_transition(id, &intent.target_state, next)
}

fn validate_join_target_snapshot(
    id: CoverageHoldId,
    target: MeshBlockLocation,
    state: &PendingJoinTargetState,
    allow_uninitialized: bool,
) -> Result<(), CoverageInvariantError> {
    let Some(expected) = state.data.expected_keys() else {
        if allow_uninitialized
            && state.data.ready.is_empty()
            && state.data.outstanding.is_empty()
            && state.data.failures.is_empty()
            && state.mesh == PendingJoinMeshState::AwaitingData
        {
            return Ok(());
        }
        return Err(CoverageInvariantError::InvalidHoldOwner(id));
    };
    let actual = state.data.all_keys();
    if let Some(key) = actual.iter().copied().find(|key| {
        usize::from(state.data.ready.contains(key))
            + usize::from(state.data.outstanding.contains_key(key))
            + usize::from(state.data.failures.contains_key(key))
            != 1
    }) {
        return Err(stale_member_or_owner(id, state, key, None));
    }
    for (&key, &request) in &state.data.outstanding {
        validate_data_request(id, key, request)?;
    }
    for (&key, &failure) in &state.data.failures {
        validate_data_request(id, key, failure.request())?;
    }
    if let Some(request) = state.mesh.request() {
        validate_mesh_request(id, target, request)?;
    }
    if expected != &actual {
        let changed_key = expected
            .iter()
            .chain(actual.iter())
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .find(|key| expected.contains(key) != actual.contains(key));
        if let Some(key) = changed_key {
            return Err(stale_member_or_owner(id, state, key, None));
        }
        return Err(CoverageInvariantError::InvalidHoldOwner(id));
    }
    let all_data_ready = state.data.ready.len() == expected.len();
    if !all_data_ready && state.mesh != PendingJoinMeshState::AwaitingData {
        return Err(stale_state_or_owner(id, state, None));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingJoinDataMemberState {
    Ready,
    Outstanding(PhysicalRequestId),
    Failure(PendingRequestFailure),
}

fn data_member_state(
    state: &PendingJoinDataState,
    key: PendingJoinDataKey,
) -> Option<PendingJoinDataMemberState> {
    if state.ready.contains(&key) {
        return Some(PendingJoinDataMemberState::Ready);
    }
    state
        .outstanding
        .get(&key)
        .copied()
        .map(PendingJoinDataMemberState::Outstanding)
        .or_else(|| {
            state
                .failures
                .get(&key)
                .copied()
                .map(PendingJoinDataMemberState::Failure)
        })
}

fn validate_join_target_transition(
    id: CoverageHoldId,
    old: &PendingJoinTargetState,
    next: &PendingJoinTargetState,
) -> Result<(), CoverageInvariantError> {
    validate_join_target_snapshot(id, id.join_parent, old, true)?;
    match (old.data.expected_keys(), next.data.expected_keys()) {
        (None, Some(_)) => {}
        (Some(old_expected), Some(next_expected)) if old_expected == next_expected => {
            for &key in old_expected {
                let old_member = data_member_state(&old.data, key)
                    .ok_or(CoverageInvariantError::InvalidHoldOwner(id))?;
                let next_member = data_member_state(&next.data, key).ok_or_else(|| {
                    stale_member_or_owner(id, next, key, member_request(old_member))
                })?;
                let valid = match (old_member, next_member) {
                    (PendingJoinDataMemberState::Ready, PendingJoinDataMemberState::Ready) => true,
                    (
                        PendingJoinDataMemberState::Outstanding(old_request),
                        PendingJoinDataMemberState::Outstanding(next_request),
                    ) => old_request == next_request,
                    (
                        PendingJoinDataMemberState::Outstanding(_),
                        PendingJoinDataMemberState::Ready,
                    ) => true,
                    (
                        PendingJoinDataMemberState::Outstanding(old_request),
                        PendingJoinDataMemberState::Failure(next_failure),
                    ) => old_request == next_failure.request(),
                    (
                        PendingJoinDataMemberState::Failure(old_failure),
                        PendingJoinDataMemberState::Failure(next_failure),
                    ) => old_failure == next_failure,
                    (
                        PendingJoinDataMemberState::Failure(old_failure),
                        PendingJoinDataMemberState::Outstanding(next_request),
                    ) => is_fresh_request_successor(old_failure.request(), next_request),
                    _ => false,
                };
                if !valid {
                    return Err(stale_member_or_owner(
                        id,
                        next,
                        key,
                        member_request(old_member),
                    ));
                }
            }
        }
        _ => return Err(stale_state_or_owner(id, next, old.first_request())),
    }
    validate_mesh_transition(id, old.mesh, next.mesh, next)
}

fn member_request(state: PendingJoinDataMemberState) -> Option<PhysicalRequestId> {
    match state {
        PendingJoinDataMemberState::Ready => None,
        PendingJoinDataMemberState::Outstanding(request) => Some(request),
        PendingJoinDataMemberState::Failure(failure) => Some(failure.request()),
    }
}

fn validate_mesh_transition(
    id: CoverageHoldId,
    old: PendingJoinMeshState,
    next: PendingJoinMeshState,
    next_state: &PendingJoinTargetState,
) -> Result<(), CoverageInvariantError> {
    let valid = match (old, next) {
        (
            PendingJoinMeshState::AwaitingData,
            PendingJoinMeshState::AwaitingData
            | PendingJoinMeshState::AwaitingRequest
            | PendingJoinMeshState::Meshing { .. },
        ) => true,
        (
            PendingJoinMeshState::AwaitingRequest,
            PendingJoinMeshState::AwaitingRequest | PendingJoinMeshState::Meshing { .. },
        ) => true,
        (
            PendingJoinMeshState::Meshing {
                request: old_request,
            },
            PendingJoinMeshState::Meshing {
                request: next_request,
            },
        ) => old_request == next_request,
        (
            PendingJoinMeshState::Meshing {
                request: old_request,
            },
            PendingJoinMeshState::RetryableFailure {
                request: next_request,
                ..
            }
            | PendingJoinMeshState::Exhausted {
                request: next_request,
                ..
            },
        ) => old_request == next_request,
        (PendingJoinMeshState::Meshing { .. }, PendingJoinMeshState::Ready) => true,
        (
            old_terminal @ (PendingJoinMeshState::RetryableFailure { .. }
            | PendingJoinMeshState::Exhausted { .. }),
            next_terminal @ (PendingJoinMeshState::RetryableFailure { .. }
            | PendingJoinMeshState::Exhausted { .. }),
        ) => old_terminal == next_terminal,
        (
            PendingJoinMeshState::RetryableFailure {
                request: old_request,
                ..
            }
            | PendingJoinMeshState::Exhausted {
                request: old_request,
                ..
            },
            PendingJoinMeshState::Meshing {
                request: next_request,
            },
        ) => is_fresh_request_successor(old_request, next_request),
        (PendingJoinMeshState::Ready, PendingJoinMeshState::Ready) => true,
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(stale_state_or_owner(id, next_state, old.request()))
    }
}

fn stale_member_or_owner(
    id: CoverageHoldId,
    state: &PendingJoinTargetState,
    key: PendingJoinDataKey,
    fallback: Option<PhysicalRequestId>,
) -> CoverageInvariantError {
    state.data.request_at(key).or(fallback).map_or_else(
        || CoverageInvariantError::MissingJoinTargetMember {
            id,
            location: key.to_location(),
        },
        |request| CoverageInvariantError::StaleJoinTargetRequest { id, request },
    )
}

fn stale_state_or_owner(
    id: CoverageHoldId,
    state: &PendingJoinTargetState,
    fallback: Option<PhysicalRequestId>,
) -> CoverageInvariantError {
    state
        .first_request()
        .or(fallback)
        .map_or(CoverageInvariantError::InvalidHoldOwner(id), |request| {
            CoverageInvariantError::StaleJoinTargetRequest { id, request }
        })
}

fn is_fresh_request_successor(old: PhysicalRequestId, next: PhysicalRequestId) -> bool {
    let old_tag = old.tag();
    let next_tag = next.tag();
    old_tag.request_epoch == next_tag.request_epoch
        && old_tag.request_generation < next_tag.request_generation
}

fn validate_data_request(
    id: CoverageHoldId,
    key: PendingJoinDataKey,
    request: PhysicalRequestId,
) -> Result<(), CoverageInvariantError> {
    if !matches!(
        request,
        PhysicalRequestId::Load { location, .. } if location == key.to_location()
    ) {
        return Err(CoverageInvariantError::StaleJoinTargetRequest { id, request });
    }
    Ok(())
}

fn validate_mesh_request(
    id: CoverageHoldId,
    target: MeshBlockLocation,
    request: PhysicalRequestId,
) -> Result<(), CoverageInvariantError> {
    if !matches!(
        request,
        PhysicalRequestId::Mesh { location, .. } if location == target
    ) {
        return Err(CoverageInvariantError::StaleJoinTargetRequest { id, request });
    }
    Ok(())
}

const fn valid_demand(counts: DemandCounts) -> bool {
    counts.visual_splits <= counts.visuals
        && counts.visuals <= counts.resident
        && counts.collision_splits <= counts.collisions
        && counts.collisions <= counts.resident
}

fn apply_demand_update(
    nodes: &mut OrdMap<CoverageKey, CoverageNode>,
    key: CoverageKey,
    old: DemandCounts,
    new: DemandCounts,
    lod_count: u8,
    work: &mut CoverageWorkCounters,
) -> Result<(), CoverageInvariantError> {
    let mut current = Some(key.location());
    while let Some(location) = current {
        work.nodes_visited += 1;
        let current_key = CoverageKey::from_location(location);
        let mut node = nodes.get(&current_key).cloned().unwrap_or_default();
        if current_key == key {
            node.demand = new;
        }
        node.branch_resident_nodes = adjust_branch_counter(
            node.branch_resident_nodes,
            old.resident > 0,
            new.resident > 0,
        )?;
        node.branch_visual_nodes =
            adjust_branch_counter(node.branch_visual_nodes, old.visuals > 0, new.visuals > 0)?;
        node.branch_collision_nodes = adjust_branch_counter(
            node.branch_collision_nodes,
            old.collisions > 0,
            new.collisions > 0,
        )?;
        nodes.insert(current_key, node);
        current = checked_parent(location, lod_count)?;
        if current.is_some() {
            work.ancestor_links_checked += 1;
        }
    }
    Ok(())
}

fn adjust_branch_counter(
    current: u64,
    old_present: bool,
    new_present: bool,
) -> Result<u64, CoverageInvariantError> {
    match (old_present, new_present) {
        (false, true) => current
            .checked_add(1)
            .ok_or(CoverageInvariantError::InvalidPartition),
        (true, false) => current
            .checked_sub(1)
            .ok_or(CoverageInvariantError::InvalidPartition),
        _ => Ok(current),
    }
}

fn validate_known_acceptance_locations(
    nodes: &OrdMap<CoverageKey, CoverageNode>,
    acceptances: &BTreeMap<CoverageKey, AcceptedFeatureSnapshot>,
) -> Result<(), CoverageInvariantError> {
    for &key in acceptances.keys() {
        if !nodes.contains_key(&key) {
            return Err(CoverageInvariantError::UnknownAcceptedLocation(
                key.location(),
            ));
        }
    }
    Ok(())
}

fn validate_accepted_snapshots(
    nodes: &OrdMap<CoverageKey, CoverageNode>,
    acceptances: &BTreeMap<CoverageKey, AcceptedFeatureSnapshot>,
) -> Result<(), CoverageInvariantError> {
    for (&key, &snapshot) in acceptances {
        let node = nodes
            .get(&key)
            .ok_or(CoverageInvariantError::UnknownAcceptedLocation(
                key.location(),
            ))?;
        let invalid = (!snapshot.visuals && !snapshot.collisions)
            || node.accepted_snapshot.is_some_and(|current| {
                snapshot.revision < current.revision
                    || (snapshot.revision == current.revision && snapshot != current)
            });
        if invalid {
            return Err(CoverageInvariantError::InvalidAcceptedSnapshot {
                location: key.location(),
            });
        }
    }
    Ok(())
}

fn root_group(
    feature: CoverageFeature,
    anchor: MeshBlockLocation,
    activating: bool,
) -> FeatureTopologyGroup {
    FeatureTopologyGroup {
        feature,
        operation: if activating {
            TopologyOperation::RootActivate
        } else {
            TopologyOperation::RootDeactivate
        },
        anchor,
        activate: activating.then_some(anchor).into_iter().collect(),
        deactivate: (!activating).then_some(anchor).into_iter().collect(),
    }
}

fn frontier_activate_group(
    feature: CoverageFeature,
    anchor: MeshBlockLocation,
) -> FeatureTopologyGroup {
    FeatureTopologyGroup {
        feature,
        operation: TopologyOperation::FrontierActivate,
        anchor,
        activate: vec![anchor],
        deactivate: Vec::new(),
    }
}

fn join_group(
    feature: CoverageFeature,
    anchor: MeshBlockLocation,
    frontier: Vec<MeshBlockLocation>,
) -> FeatureTopologyGroup {
    FeatureTopologyGroup {
        feature,
        operation: TopologyOperation::Join,
        anchor,
        activate: vec![anchor],
        deactivate: frontier,
    }
}

fn frontier_deactivate_group(
    feature: CoverageFeature,
    anchor: MeshBlockLocation,
    frontier: Vec<MeshBlockLocation>,
) -> FeatureTopologyGroup {
    FeatureTopologyGroup {
        feature,
        operation: TopologyOperation::FrontierDeactivate,
        anchor,
        activate: Vec::new(),
        deactivate: frontier,
    }
}

#[derive(Default)]
struct JoinOwnerDraft {
    before_fallbacks: BTreeSet<CoverageKey>,
    after_fallbacks: Option<Vec<MeshBlockLocation>>,
}

struct PreparedJoinOwners {
    next_transition_generation: u64,
    pending_joins: OrdMap<PendingJoinKey, PendingJoinIntent>,
    deltas: Vec<CoverageHoldOwnerDelta>,
}

struct JoinOwnerPreparationInput<'a> {
    state: &'a CoverageState,
    projected_nodes: &'a OrdMap<CoverageKey, CoverageNode>,
    projected_visual_active: &'a OrdSet<CoverageKey>,
    projected_collision_active: &'a OrdSet<CoverageKey>,
    affected_anchors: &'a BTreeSet<CoverageKey>,
    groups: &'a [FeatureTopologyGroup],
    lod_count: u8,
}

fn prepare_join_owners(
    input: JoinOwnerPreparationInput<'_>,
    work: &mut CoverageWorkCounters,
) -> Result<PreparedJoinOwners, CoverageInvariantError> {
    let JoinOwnerPreparationInput {
        state,
        projected_nodes,
        projected_visual_active,
        projected_collision_active,
        affected_anchors,
        groups,
        lod_count,
    } = input;
    let mut drafts = BTreeMap::<PendingJoinKey, JoinOwnerDraft>::new();
    let mut joining = BTreeSet::new();

    for group in groups {
        if !matches!(
            group.operation,
            TopologyOperation::Join | TopologyOperation::FrontierDeactivate
        ) {
            continue;
        }
        let key = PendingJoinKey::new(group.anchor, group.feature);
        if group.operation == TopologyOperation::Join {
            joining.insert(key);
        }
        let draft = drafts.entry(key).or_default();
        draft.before_fallbacks.extend(
            group
                .deactivate
                .iter()
                .copied()
                .map(CoverageKey::from_location),
        );
        draft.after_fallbacks = None;
    }

    // Only touched ancestor paths can create, update, or cancel an owner.
    // Canonical BTreeSet iteration makes generation assignment independent of
    // caller input order without scanning unrelated pending joins.
    for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
        let active = match feature {
            CoverageFeature::Visual => projected_visual_active,
            CoverageFeature::Collision => projected_collision_active,
        };
        for &anchor in affected_anchors {
            let key = PendingJoinKey::new(anchor.location(), feature);
            let old = state.pending_joins.get(&key);
            work.pending_owners_examined += usize::from(old.is_some());
            if let Some(old) = old {
                validate_owner_endpoint(lod_count, state, old, work)?;
            }
            let Some(node) = projected_nodes.get(&anchor) else {
                continue;
            };

            if node.is_split_demanded(feature) {
                if let Some(old) = old {
                    let draft = drafts.entry(key).or_default();
                    draft.before_fallbacks.extend(
                        old.current_manifest
                            .fallbacks
                            .iter()
                            .copied()
                            .map(CoverageKey::from_location),
                    );
                    draft.after_fallbacks = None;
                }
                continue;
            }

            if active.contains(&anchor) || node.branch_active_nodes(feature) == 0 {
                continue;
            }
            if joining.contains(&key) {
                continue;
            }

            let frontier = canonical_active_frontier_under(
                projected_nodes,
                active,
                anchor,
                feature,
                lod_count,
                work,
            )?;
            let draft = drafts.entry(key).or_default();
            draft
                .before_fallbacks
                .extend(frontier.iter().copied().map(CoverageKey::from_location));
            for group in groups.iter().filter(|group| group.feature == feature) {
                for &location in &group.activate {
                    if location != anchor.location()
                        && location_is_under(location, anchor.location(), lod_count)?
                    {
                        draft
                            .before_fallbacks
                            .insert(CoverageKey::from_location(location));
                    }
                }
            }
            draft.after_fallbacks = Some(frontier);
        }
    }

    let mut next_transition_generation = state.next_transition_generation;
    let mut pending_joins = state.pending_joins.clone();
    let mut deltas = Vec::with_capacity(drafts.len());
    for (key, mut draft) in drafts {
        let old_intent = state.pending_joins.get(&key);
        let id = if let Some(old) = old_intent {
            draft.before_fallbacks.extend(
                old.current_manifest
                    .fallbacks
                    .iter()
                    .copied()
                    .map(CoverageKey::from_location),
            );
            old.id
        } else {
            next_transition_generation = next_transition_generation
                .checked_add(1)
                .ok_or(CoverageInvariantError::HoldGenerationOverflow)?;
            CoverageHoldId {
                join_parent: key.target(),
                feature: match key.feature_rank {
                    0 => CoverageFeature::Visual,
                    1 => CoverageFeature::Collision,
                    _ => return Err(CoverageInvariantError::InvalidPartition),
                },
                transition_generation: next_transition_generation,
            }
        };
        let target = key.target();
        let feature = id.feature;
        let before = CoverageHoldIntentManifest {
            id,
            target,
            fallbacks: draft
                .before_fallbacks
                .into_iter()
                .map(CoverageKey::location)
                .collect(),
        };
        let after = draft
            .after_fallbacks
            .map(|fallbacks| CoverageHoldIntentManifest {
                id,
                target,
                fallbacks,
            });
        let old = old_intent.map(|intent| intent.current_manifest.clone());

        if let Some(after) = &after {
            pending_joins.insert(
                key,
                PendingJoinIntent {
                    id,
                    target,
                    feature,
                    current_manifest: after.clone(),
                    target_state: old_intent
                        .map_or_else(PendingJoinTargetState::default, |intent| {
                            intent.target_state.clone()
                        }),
                },
            );
        } else {
            pending_joins.remove(&key);
        }

        if old.as_ref() != Some(&before) || after.as_ref() != Some(&before) {
            deltas.push(CoverageHoldOwnerDelta {
                old,
                before_topology: Some(before),
                after_topology: after,
            });
        }
    }

    Ok(PreparedJoinOwners {
        next_transition_generation,
        pending_joins,
        deltas,
    })
}

fn apply_owner_phase(
    nodes: &mut OrdMap<CoverageKey, CoverageNode>,
    deltas: &[CoverageHoldOwnerDelta],
    before_topology: bool,
) -> Result<(), CoverageInvariantError> {
    for delta in deltas {
        let old = if before_topology {
            delta.old.as_ref()
        } else {
            delta.before_topology.as_ref()
        };
        let next = if before_topology {
            delta.before_topology.as_ref()
        } else {
            delta.after_topology.as_ref()
        };
        apply_one_owner_change(nodes, old, next)?;
    }
    Ok(())
}

fn apply_one_owner_change(
    nodes: &mut OrdMap<CoverageKey, CoverageNode>,
    old: Option<&CoverageHoldIntentManifest>,
    next: Option<&CoverageHoldIntentManifest>,
) -> Result<(), CoverageInvariantError> {
    let old_resources = manifest_resources(old);
    let next_resources = manifest_resources(next);
    let id = old
        .or(next)
        .map(|manifest| manifest.id)
        .ok_or(CoverageInvariantError::InvalidPartition)?;
    for key in old_resources.difference(&next_resources).copied() {
        adjust_owner_count(nodes, key, id.feature, false, id)?;
    }
    for key in next_resources.difference(&old_resources).copied() {
        adjust_owner_count(nodes, key, id.feature, true, id)?;
    }
    Ok(())
}

fn manifest_resources(manifest: Option<&CoverageHoldIntentManifest>) -> BTreeSet<CoverageKey> {
    manifest.map_or_else(BTreeSet::new, |manifest| {
        std::iter::once(manifest.target)
            .chain(manifest.fallbacks.iter().copied())
            .map(CoverageKey::from_location)
            .collect()
    })
}

fn validate_owner_endpoint(
    lod_count: u8,
    state: &CoverageState,
    intent: &PendingJoinIntent,
    work: &mut CoverageWorkCounters,
) -> Result<(), CoverageInvariantError> {
    let id = intent.id;
    let manifest = &intent.current_manifest;
    let active = active_set(state, intent.feature);
    if id.join_parent != intent.target
        || id.feature != intent.feature
        || manifest.id != id
        || manifest.target != intent.target
        || manifest.fallbacks.is_empty()
        || !state
            .nodes
            .contains_key(&CoverageKey::from_location(intent.target))
        || active.contains(&CoverageKey::from_location(intent.target))
        || manifest.fallbacks.contains(&intent.target)
        || manifest
            .fallbacks
            .windows(2)
            .any(|pair| CoverageKey::from_location(pair[0]) >= CoverageKey::from_location(pair[1]))
    {
        return Err(CoverageInvariantError::InvalidHoldOwner(id));
    }
    for &fallback in &manifest.fallbacks {
        let key = CoverageKey::from_location(fallback);
        if !location_is_under(fallback, intent.target, lod_count)? || !active.contains(&key) {
            return Err(CoverageInvariantError::InvalidHoldOwner(id));
        }
    }
    let canonical = canonical_active_frontier_under(
        &state.nodes,
        active,
        CoverageKey::from_location(intent.target),
        intent.feature,
        lod_count,
        work,
    )
    .map_err(|_| CoverageInvariantError::InvalidHoldOwner(id))?;
    if manifest.fallbacks != canonical {
        return Err(CoverageInvariantError::InvalidHoldOwner(id));
    }
    validate_join_target_snapshot(id, intent.target, &intent.target_state, true)?;
    Ok(())
}

fn validate_after_owner_endpoints(
    lod_count: u8,
    state: &CoverageState,
    deltas: &[CoverageHoldOwnerDelta],
    work: &mut CoverageWorkCounters,
) -> Result<(), CoverageInvariantError> {
    for delta in deltas {
        let manifest = delta
            .after_topology
            .as_ref()
            .or(delta.before_topology.as_ref())
            .or(delta.old.as_ref())
            .ok_or(CoverageInvariantError::InvalidPartition)?;
        let key = PendingJoinKey::new(manifest.target, manifest.id.feature);
        match &delta.after_topology {
            Some(expected) => {
                let intent = state
                    .pending_joins
                    .get(&key)
                    .ok_or(CoverageInvariantError::InvalidHoldOwner(expected.id))?;
                if &intent.current_manifest != expected {
                    return Err(CoverageInvariantError::InvalidHoldOwner(expected.id));
                }
                validate_owner_endpoint(lod_count, state, intent, work)?;
            }
            None => {
                if state.pending_joins.contains_key(&key) {
                    return Err(CoverageInvariantError::InvalidHoldOwner(manifest.id));
                }
            }
        }
    }
    Ok(())
}

fn adjust_owner_count(
    nodes: &mut OrdMap<CoverageKey, CoverageNode>,
    key: CoverageKey,
    feature: CoverageFeature,
    adding: bool,
    id: CoverageHoldId,
) -> Result<(), CoverageInvariantError> {
    let Some(mut node) = nodes.get(&key).cloned() else {
        return Err(CoverageInvariantError::InvalidHoldOwner(id));
    };
    let count = if adding {
        node.coverage_owners(feature).checked_add(1)
    } else {
        node.coverage_owners(feature).checked_sub(1)
    }
    .ok_or(CoverageInvariantError::InvalidHoldOwner(id))?;
    node.set_coverage_owners(feature, count);
    nodes.insert(key, node);
    Ok(())
}

fn project_topology_groups(
    nodes: &mut OrdMap<CoverageKey, CoverageNode>,
    groups: &[FeatureTopologyGroup],
    visual_active: &mut OrdSet<CoverageKey>,
    collision_active: &mut OrdSet<CoverageKey>,
    lod_count: u8,
    work: &mut CoverageWorkCounters,
) -> Result<(), CoverageInvariantError> {
    for group in groups {
        let active = match group.feature {
            CoverageFeature::Visual => &mut *visual_active,
            CoverageFeature::Collision => &mut *collision_active,
        };
        project_topology_group(nodes, active, group, lod_count, work)?;
    }
    Ok(())
}

fn project_topology_group(
    nodes: &mut OrdMap<CoverageKey, CoverageNode>,
    active: &mut OrdSet<CoverageKey>,
    group: &FeatureTopologyGroup,
    lod_count: u8,
    work: &mut CoverageWorkCounters,
) -> Result<(), CoverageInvariantError> {
    for location in &group.deactivate {
        set_topology_active(
            nodes,
            active,
            CoverageKey::from_location(*location),
            group.feature,
            false,
            lod_count,
            work,
        )?;
    }
    for location in &group.activate {
        set_topology_active(
            nodes,
            active,
            CoverageKey::from_location(*location),
            group.feature,
            true,
            lod_count,
            work,
        )?;
    }
    Ok(())
}

fn set_topology_active(
    nodes: &mut OrdMap<CoverageKey, CoverageNode>,
    active: &mut OrdSet<CoverageKey>,
    key: CoverageKey,
    feature: CoverageFeature,
    new_active: bool,
    lod_count: u8,
    work: &mut CoverageWorkCounters,
) -> Result<(), CoverageInvariantError> {
    let old_active = active.contains(&key);
    if old_active == new_active {
        return Err(CoverageInvariantError::InvalidPartition);
    }
    let mut current = Some(key.location());
    while let Some(location) = current {
        work.nodes_visited += 1;
        let current_key = CoverageKey::from_location(location);
        let mut node = nodes
            .get(&current_key)
            .cloned()
            .ok_or(CoverageInvariantError::InvalidPartition)?;
        if current_key == key {
            if node.is_active(feature) != old_active {
                return Err(CoverageInvariantError::InvalidPartition);
            }
            node.set_active(feature, new_active);
        }
        let count =
            adjust_branch_counter(node.branch_active_nodes(feature), old_active, new_active)?;
        node.set_branch_active_nodes(feature, count);
        nodes.insert(current_key, node);
        current = checked_parent(location, lod_count)?;
        if current.is_some() {
            work.ancestor_links_checked += 1;
        }
    }
    if new_active {
        active.insert(key);
    } else {
        active.remove(&key);
    }
    Ok(())
}

fn node_is_eligible(node: &CoverageNode, feature: CoverageFeature) -> bool {
    node.is_ready(feature)
        && ((node.branch_is_resident() && node.branch_is_demanded(feature))
            || node.coverage_owners(feature) > 0)
}

fn sparse_gap_can_activate(
    state: &CoverageState,
    active: &OrdSet<CoverageKey>,
    anchor: CoverageKey,
    feature: CoverageFeature,
    lod_count: u8,
    work: &mut CoverageWorkCounters,
) -> Result<bool, CoverageInvariantError> {
    let mut has_pending_ancestor = false;
    let mut ancestor = checked_parent(anchor.location(), lod_count)?;
    while let Some(location) = ancestor {
        work.ancestor_links_checked += 1;
        let key = CoverageKey::from_location(location);
        if active.contains(&key) {
            return Ok(false);
        }
        work.pending_owners_examined += 1;
        has_pending_ancestor |= state
            .pending_joins
            .contains_key(&PendingJoinKey::new(location, feature));
        ancestor = checked_parent(location, lod_count)?;
    }
    Ok(has_pending_ancestor)
}

fn canonical_active_frontier_under(
    nodes: &OrdMap<CoverageKey, CoverageNode>,
    active: &OrdSet<CoverageKey>,
    anchor: CoverageKey,
    feature: CoverageFeature,
    lod_count: u8,
    work: &mut CoverageWorkCounters,
) -> Result<Vec<MeshBlockLocation>, CoverageInvariantError> {
    let mut frontier = Vec::new();
    collect_active_frontier_under(
        nodes,
        active,
        anchor,
        feature,
        lod_count,
        &mut frontier,
        work,
    )?;
    if frontier.is_empty() {
        return Err(CoverageInvariantError::InvalidPartition);
    }
    frontier.sort_by_key(|location| CoverageKey::from_location(*location));
    Ok(frontier)
}

fn collect_active_frontier_under(
    nodes: &OrdMap<CoverageKey, CoverageNode>,
    active: &OrdSet<CoverageKey>,
    anchor: CoverageKey,
    feature: CoverageFeature,
    lod_count: u8,
    output: &mut Vec<MeshBlockLocation>,
    work: &mut CoverageWorkCounters,
) -> Result<(), CoverageInvariantError> {
    work.nodes_visited += 1;
    let node = nodes
        .get(&anchor)
        .ok_or(CoverageInvariantError::InvalidPartition)?;
    let branch_active_nodes = node.branch_active_nodes(feature);
    if branch_active_nodes == 0 {
        return Ok(());
    }
    if active.contains(&anchor) {
        if branch_active_nodes != 1 || !node.is_active(feature) {
            return Err(CoverageInvariantError::InvalidPartition);
        }
        output.push(anchor.location());
        return Ok(());
    }
    if node.is_active(feature) || anchor.lod == 0 {
        return Err(CoverageInvariantError::InvalidPartition);
    }

    let mut child_active_nodes = 0_u64;
    for child in checked_children(anchor.location(), lod_count)? {
        let child_key = CoverageKey::from_location(child);
        // A terminal descendant failure can leave a valid sparse antichain.
        // Missing and inactive child branches contribute zero; the exact
        // parent summary check below still rejects stale or lost activity.
        let Some(child_node) = nodes.get(&child_key) else {
            continue;
        };
        let child_count = child_node.branch_active_nodes(feature);
        if child_count == 0 {
            continue;
        }
        child_active_nodes = child_active_nodes
            .checked_add(child_count)
            .ok_or(CoverageInvariantError::InvalidPartition)?;
        collect_active_frontier_under(nodes, active, child_key, feature, lod_count, output, work)?;
    }
    if child_active_nodes != branch_active_nodes {
        return Err(CoverageInvariantError::InvalidPartition);
    }
    Ok(())
}

fn exact_active_children(
    nodes: &OrdMap<CoverageKey, CoverageNode>,
    active: &OrdSet<CoverageKey>,
    anchor: CoverageKey,
    feature: CoverageFeature,
    lod_count: u8,
    work: &mut CoverageWorkCounters,
) -> Result<Option<Vec<MeshBlockLocation>>, CoverageInvariantError> {
    let node = nodes
        .get(&anchor)
        .ok_or(CoverageInvariantError::InvalidPartition)?;
    if active.contains(&anchor) || node.branch_active_nodes(feature) == 0 || anchor.lod == 0 {
        return Ok(None);
    }
    let children = checked_children(anchor.location(), lod_count)?;
    let mut all_exact = true;
    let mut child_active_nodes = 0_u64;
    for child in &children {
        work.nodes_visited += 1;
        let child_key = CoverageKey::from_location(*child);
        // Sparse frontiers are valid owner endpoints, but not exact joins.
        let Some(child_node) = nodes.get(&child_key) else {
            all_exact = false;
            continue;
        };
        let child_count = child_node.branch_active_nodes(feature);
        if child_count == 0 {
            all_exact = false;
            continue;
        }
        child_active_nodes = child_active_nodes
            .checked_add(child_count)
            .ok_or(CoverageInvariantError::InvalidPartition)?;
        all_exact &= active.contains(&child_key) && child_count == 1;
    }
    if child_active_nodes != node.branch_active_nodes(feature) {
        return Err(CoverageInvariantError::InvalidPartition);
    }
    Ok(all_exact.then_some(children))
}

fn prepare_exact_join_chain(
    nodes: &mut OrdMap<CoverageKey, CoverageNode>,
    active: &mut OrdSet<CoverageKey>,
    anchor: CoverageKey,
    feature: CoverageFeature,
    lod_count: u8,
    groups: &mut Vec<FeatureTopologyGroup>,
    work: &mut CoverageWorkCounters,
) -> Result<bool, CoverageInvariantError> {
    let Some(anchor_node) = nodes.get(&anchor) else {
        return Err(CoverageInvariantError::InvalidPartition);
    };
    if active.contains(&anchor) {
        return Ok(anchor_node.branch_active_nodes(feature) == 1);
    }
    if anchor.lod == 0
        || anchor_node.branch_active_nodes(feature) == 0
        || !anchor_node.is_ready(feature)
    {
        return Ok(false);
    }
    let children = checked_children(anchor.location(), lod_count)?;
    for child in &children {
        let child_key = CoverageKey::from_location(*child);
        // An exact join waits until all eight child branches are present.
        let Some(child_node) = nodes.get(&child_key) else {
            return Ok(false);
        };
        if child_node.branch_active_nodes(feature) == 0 {
            return Ok(false);
        }
        if !active.contains(&child_key)
            && !prepare_exact_join_chain(
                nodes, active, child_key, feature, lod_count, groups, work,
            )?
        {
            return Ok(false);
        }
    }
    let Some(exact_children) =
        exact_active_children(nodes, active, anchor, feature, lod_count, work)?
    else {
        return Err(CoverageInvariantError::InvalidPartition);
    };
    let group = join_group(feature, anchor.location(), exact_children);
    project_topology_group(nodes, active, &group, lod_count, work)?;
    groups.push(group);
    Ok(true)
}

fn ready_split_group(
    nodes: &OrdMap<CoverageKey, CoverageNode>,
    active: &OrdSet<CoverageKey>,
    anchor: CoverageKey,
    feature: CoverageFeature,
    lod_count: u8,
    work: &mut CoverageWorkCounters,
) -> Result<Option<FeatureTopologyGroup>, CoverageInvariantError> {
    work.nodes_visited += 1;
    let Some(parent) = nodes.get(&anchor) else {
        return Err(CoverageInvariantError::InvalidPartition);
    };
    if anchor.lod == 0
        || !parent.is_split_demanded(feature)
        || parent.branch_active_nodes(feature) != 1
    {
        return Ok(None);
    }
    let children = checked_children(anchor.location(), lod_count)?;
    for child in &children {
        work.nodes_visited += 1;
        let child_key = CoverageKey::from_location(*child);
        let Some(child_node) = nodes.get(&child_key) else {
            return Ok(None);
        };
        if active.contains(&child_key)
            || child_node.branch_active_nodes(feature) != 0
            || !child_node.branch_is_resident()
            || !child_node.branch_is_demanded(feature)
            || !child_node.is_ready(feature)
        {
            return Ok(None);
        }
    }
    Ok(Some(FeatureTopologyGroup {
        feature,
        operation: TopologyOperation::Split,
        anchor: anchor.location(),
        activate: children,
        deactivate: vec![anchor.location()],
    }))
}

const fn feature_rank(feature: CoverageFeature) -> u8 {
    match feature {
        CoverageFeature::Visual => 0,
        CoverageFeature::Collision => 1,
    }
}

fn owner_delta_sort_key(delta: &CoverageHoldOwnerDelta) -> Option<(u8, u8, i32, i32, i32, u64)> {
    let id = delta
        .old
        .as_ref()
        .or(delta.before_topology.as_ref())
        .or(delta.after_topology.as_ref())
        .map(|manifest| manifest.id)?;
    let location = id.join_parent;
    Some((
        feature_rank(id.feature),
        location.lod_index,
        location.position_in_blocks.x,
        location.position_in_blocks.y,
        location.position_in_blocks.z,
        id.transition_generation,
    ))
}

fn location_is_under(
    location: MeshBlockLocation,
    anchor: MeshBlockLocation,
    lod_count: u8,
) -> Result<bool, CoverageInvariantError> {
    validate_location(location, lod_count)?;
    validate_location(anchor, lod_count)?;
    if location.lod_index > anchor.lod_index {
        return Ok(false);
    }
    let mut current = location;
    while current.lod_index < anchor.lod_index {
        let Some(parent) = checked_parent(current, lod_count)? else {
            return Ok(false);
        };
        current = parent;
    }
    Ok(current == anchor)
}

fn operation_rank(operation: TopologyOperation) -> u8 {
    match operation {
        TopologyOperation::RootActivate => 0,
        TopologyOperation::FrontierActivate => 1,
        TopologyOperation::Split => 2,
        TopologyOperation::Join => 3,
        TopologyOperation::RootDeactivate => 4,
        TopologyOperation::FrontierDeactivate => 5,
    }
}

fn group_sort_key(group: &FeatureTopologyGroup) -> (u8, u8, u8, i32, i32, i32) {
    let location = group.anchor;
    let dependency_lod = match group.operation {
        TopologyOperation::Split => u8::MAX - location.lod_index,
        TopologyOperation::RootActivate
        | TopologyOperation::FrontierActivate
        | TopologyOperation::Join
        | TopologyOperation::RootDeactivate
        | TopologyOperation::FrontierDeactivate => location.lod_index,
    };
    (
        feature_rank(group.feature),
        operation_rank(group.operation),
        dependency_lod,
        location.position_in_blocks.x,
        location.position_in_blocks.y,
        location.position_in_blocks.z,
    )
}

pub(super) fn checked_parent(
    location: MeshBlockLocation,
    lod_count: u8,
) -> Result<Option<MeshBlockLocation>, CoverageInvariantError> {
    validate_location(location, lod_count)?;
    if location.lod_index + 1 == lod_count {
        return Ok(None);
    }
    let position = location.position_in_blocks;
    Ok(Some(MeshBlockLocation::new(
        Vector3i::new(
            position.x.div_euclid(2),
            position.y.div_euclid(2),
            position.z.div_euclid(2),
        ),
        location.lod_index + 1,
    )))
}

pub(super) fn checked_children(
    parent: MeshBlockLocation,
    lod_count: u8,
) -> Result<Vec<MeshBlockLocation>, CoverageInvariantError> {
    validate_location(parent, lod_count)?;
    let child_lod = parent
        .lod_index
        .checked_sub(1)
        .ok_or(CoverageInvariantError::InvalidLodIndex { location: parent })?;
    let position = parent.position_in_blocks;
    let mut children = Vec::with_capacity(8);
    for index in 0..8_i64 {
        let child_x = i64::from(position.x)
            .checked_mul(2)
            .and_then(|value| value.checked_add(index & 1))
            .and_then(|value| i32::try_from(value).ok());
        let child_y = i64::from(position.y)
            .checked_mul(2)
            .and_then(|value| value.checked_add((index >> 1) & 1))
            .and_then(|value| i32::try_from(value).ok());
        let child_z = i64::from(position.z)
            .checked_mul(2)
            .and_then(|value| value.checked_add((index >> 2) & 1))
            .and_then(|value| i32::try_from(value).ok());
        let (Some(x), Some(y), Some(z)) = (child_x, child_y, child_z) else {
            return Err(CoverageInvariantError::CoordinateOverflow { location: parent });
        };
        children.push(MeshBlockLocation::new(Vector3i::new(x, y, z), child_lod));
    }
    Ok(children)
}

fn validate_location(
    location: MeshBlockLocation,
    lod_count: u8,
) -> Result<(), CoverageInvariantError> {
    if location.lod_index >= lod_count {
        return Err(CoverageInvariantError::InvalidLodIndex { location });
    }
    Ok(())
}

fn coarsest_root(
    key: CoverageKey,
    lod_count: u8,
    work: &mut CoverageWorkCounters,
) -> Result<CoverageKey, CoverageInvariantError> {
    let mut current = key.location();
    while let Some(parent) = checked_parent(current, lod_count)? {
        work.ancestor_links_checked += 1;
        current = parent;
    }
    Ok(CoverageKey::from_location(current))
}

fn collect_ancestor_keys(
    key: CoverageKey,
    lod_count: u8,
    output: &mut BTreeSet<CoverageKey>,
    work: &mut CoverageWorkCounters,
) -> Result<(), CoverageInvariantError> {
    let mut current = Some(key.location());
    while let Some(location) = current {
        output.insert(CoverageKey::from_location(location));
        current = checked_parent(location, lod_count)?;
        if current.is_some() {
            work.ancestor_links_checked += 1;
        }
    }
    Ok(())
}

fn active_locations_in(state: &CoverageState, feature: CoverageFeature) -> Vec<MeshBlockLocation> {
    active_set(state, feature)
        .iter()
        .map(|key| key.location())
        .collect()
}

fn active_set(state: &CoverageState, feature: CoverageFeature) -> &OrdSet<CoverageKey> {
    match feature {
        CoverageFeature::Visual => &state.visual_active,
        CoverageFeature::Collision => &state.collision_active,
    }
}

type ReplayedTopologyState = (
    OrdMap<CoverageKey, CoverageNode>,
    OrdSet<CoverageKey>,
    OrdSet<CoverageKey>,
);

fn replay_topology_groups(
    lod_count: u8,
    old_state: &CoverageState,
    mut next_nodes: OrdMap<CoverageKey, CoverageNode>,
    groups: &[FeatureTopologyGroup],
    work: &mut CoverageWorkCounters,
) -> Result<ReplayedTopologyState, CoverageInvariantError> {
    let mut visual_active = old_state.visual_active.clone();
    let mut collision_active = old_state.collision_active.clone();
    let join_dependencies = groups
        .iter()
        .filter(|group| group.operation == TopologyOperation::Join)
        .flat_map(|group| {
            group.deactivate.iter().copied().map(|location| {
                (
                    feature_rank(group.feature),
                    CoverageKey::from_location(location),
                )
            })
        })
        .collect::<BTreeSet<_>>();
    #[cfg(debug_assertions)]
    let mut debug_nodes = {
        // Pending groups observe their old topology until their own atomic
        // replay, while the complete before-topology owner phase is already
        // installed for every group in the batch.
        let mut nodes = next_nodes.clone();
        for group in groups {
            for location in std::iter::once(&group.anchor)
                .chain(&group.activate)
                .chain(&group.deactivate)
            {
                let key = CoverageKey::from_location(*location);
                if let Some(old_node) = old_state.nodes.get(&key) {
                    nodes.insert(key, old_node.clone());
                } else {
                    nodes.remove(&key);
                }
            }
        }
        for (&key, next_node) in &next_nodes {
            let Some(mut node) = nodes.get(&key).cloned() else {
                continue;
            };
            node.visual_coverage_owners = next_node.visual_coverage_owners;
            node.collision_coverage_owners = next_node.collision_coverage_owners;
            nodes.insert(key, node);
        }
        nodes
    };
    for (group_index, group) in groups.iter().enumerate() {
        work.groups_validated += 1;
        let key = CoverageKey::from_location(group.anchor);
        let active = match group.feature {
            CoverageFeature::Visual => &mut visual_active,
            CoverageFeature::Collision => &mut collision_active,
        };
        let anchor_node = next_nodes
            .get(&key)
            .ok_or(CoverageInvariantError::InvalidTopologyGroup { group_index })?;
        let branch_active_nodes = anchor_node.branch_active_nodes(group.feature);
        let valid_shape = match group.operation {
            TopologyOperation::RootActivate => {
                group.anchor.lod_index + 1 == lod_count
                    && group.activate.as_slice() == [group.anchor]
                    && group.deactivate.is_empty()
                    && !active.contains(&key)
                    && branch_active_nodes == 0
                    && node_is_eligible(anchor_node, group.feature)
            }
            TopologyOperation::FrontierActivate => {
                group.anchor.lod_index + 1 < lod_count
                    && sparse_gap_can_activate(
                        old_state,
                        active,
                        key,
                        group.feature,
                        lod_count,
                        work,
                    )?
                    && group.activate.as_slice() == [group.anchor]
                    && group.deactivate.is_empty()
                    && !active.contains(&key)
                    && branch_active_nodes == 0
                    && node_is_eligible(anchor_node, group.feature)
            }
            TopologyOperation::RootDeactivate => {
                group.anchor.lod_index + 1 == lod_count
                    && group.deactivate.as_slice() == [group.anchor]
                    && group.activate.is_empty()
                    && active.contains(&key)
                    && branch_active_nodes == 1
                    && (!anchor_node.branch_is_demanded(group.feature)
                        || !anchor_node.is_ready(group.feature))
            }
            TopologyOperation::Split => {
                let expected_children = checked_children(group.anchor, lod_count)?;
                group.anchor.lod_index > 0
                    && group.activate == expected_children
                    && group.deactivate.as_slice() == [group.anchor]
                    && active.contains(&key)
                    && branch_active_nodes == 1
                    && anchor_node.is_split_demanded(group.feature)
                    && group.activate.iter().all(|child| {
                        let child_key = CoverageKey::from_location(*child);
                        next_nodes.get(&child_key).is_some_and(|node| {
                            node.branch_active_nodes(group.feature) == 0
                                && node_is_eligible(node, group.feature)
                        })
                    })
            }
            TopologyOperation::Join => {
                let expected_children = checked_children(group.anchor, lod_count)?;
                let children_are_exact = expected_children.iter().all(|location| {
                    let child_key = CoverageKey::from_location(*location);
                    next_nodes.get(&child_key).is_some_and(|node| {
                        active.contains(&child_key) && node.branch_active_nodes(group.feature) == 1
                    })
                });
                let child_is_invalid = expected_children.iter().any(|location| {
                    next_nodes
                        .get(&CoverageKey::from_location(*location))
                        .is_none_or(|node| {
                            !node.branch_is_resident()
                                || !node.branch_is_demanded(group.feature)
                                || !node.is_ready(group.feature)
                        })
                });
                group.anchor.lod_index > 0
                    && group.activate.as_slice() == [group.anchor]
                    && group.deactivate == expected_children
                    && !active.contains(&key)
                    && branch_active_nodes == 8
                    && children_are_exact
                    && node_is_eligible(anchor_node, group.feature)
                    && (!anchor_node.is_split_demanded(group.feature)
                        || child_is_invalid
                        || join_dependencies.contains(&(feature_rank(group.feature), key)))
            }
            TopologyOperation::FrontierDeactivate => {
                let frontier = canonical_active_frontier_under(
                    &next_nodes,
                    active,
                    key,
                    group.feature,
                    lod_count,
                    work,
                )
                .map_err(|_| CoverageInvariantError::InvalidTopologyGroup { group_index })?;
                let owner = old_state
                    .pending_joins
                    .get(&PendingJoinKey::new(group.anchor, group.feature));
                owner.is_some_and(|intent| {
                    intent.current_manifest.fallbacks == frontier && intent.target == group.anchor
                }) && !active.contains(&key)
                    && group.activate.is_empty()
                    && group.deactivate == frontier
                    && !anchor_node.branch_is_demanded(group.feature)
            }
        };
        if !valid_shape {
            return Err(CoverageInvariantError::InvalidTopologyGroup { group_index });
        }

        project_topology_group(&mut next_nodes, active, group, lod_count, work)
            .map_err(|_| CoverageInvariantError::InvalidTopologyGroup { group_index })?;

        #[cfg(debug_assertions)]
        {
            for location in std::iter::once(&group.anchor)
                .chain(&group.activate)
                .chain(&group.deactivate)
            {
                let location_key = CoverageKey::from_location(*location);
                if let Some(node) = next_nodes.get(&location_key) {
                    debug_nodes.insert(location_key, node.clone());
                } else {
                    debug_nodes.remove(&location_key);
                }
            }
            let debug_state = CoverageState {
                revision: old_state.revision,
                next_transition_generation: old_state.next_transition_generation,
                nodes: debug_nodes.clone(),
                visual_active: visual_active.clone(),
                collision_active: collision_active.clone(),
                // `next_nodes` carries the complete before-topology owner
                // counts. Transitional manifests may intentionally be a
                // union larger than either persistent endpoint, so the
                // intermediate oracle validates eligibility from those exact
                // counts and leaves endpoint-manifest validation to the final
                // draft below.
                pending_joins: OrdMap::new(),
            };
            validate_partition_in(lod_count, &debug_state, group.feature, false)
                .map_err(|_| CoverageInvariantError::InvalidTopologyGroup { group_index })?;
        }
    }
    Ok((next_nodes, visual_active, collision_active))
}

fn validate_partition_in(
    lod_count: u8,
    state: &CoverageState,
    feature: CoverageFeature,
    validate_owner_endpoints: bool,
) -> Result<(), CoverageInvariantError> {
    let nodes = &state.nodes;
    let active = active_set(state, feature);
    for &active_key in active {
        let Some(node) = nodes.get(&active_key) else {
            return Err(CoverageInvariantError::InvalidPartition);
        };
        if !node_is_eligible(node, feature) || !node.is_active(feature) {
            return Err(CoverageInvariantError::InvalidPartition);
        }
        let mut current = active_key.location();
        while let Some(parent) = checked_parent(current, lod_count)? {
            if active.contains(&CoverageKey::from_location(parent)) {
                return Err(CoverageInvariantError::InvalidPartition);
            }
            current = parent;
        }
    }

    for (&key, node) in nodes {
        if node.is_active(feature) != active.contains(&key) {
            return Err(CoverageInvariantError::InvalidPartition);
        }
        if !node.is_demanded(feature) {
            continue;
        }
        let mut current = Some(key.location());
        let mut has_ready_eligible = false;
        let mut covering_active = 0_u8;
        while let Some(location) = current {
            let ancestor_key = CoverageKey::from_location(location);
            if let Some(ancestor) = nodes.get(&ancestor_key) {
                has_ready_eligible |= ancestor.branch_is_resident()
                    && ancestor.branch_is_demanded(feature)
                    && ancestor.is_ready(feature);
            }
            covering_active += u8::from(active.contains(&ancestor_key));
            current = checked_parent(location, lod_count)?;
        }
        if has_ready_eligible {
            let covered_by_descendant_frontier =
                covering_active == 0 && has_complete_active_frontier_under(key, active, lod_count)?;
            if covering_active != 1 && !covered_by_descendant_frontier {
                return Err(CoverageInvariantError::InvalidPartition);
            }
        }
    }

    if !validate_owner_endpoints {
        return Ok(());
    }

    let mut expected_owner_counts = BTreeMap::<CoverageKey, u64>::new();
    let mut endpoint_work = CoverageWorkCounters::default();
    for intent in state
        .pending_joins
        .values()
        .filter(|intent| intent.feature == feature)
    {
        validate_owner_endpoint(lod_count, state, intent, &mut endpoint_work)?;
        for location in
            std::iter::once(intent.target).chain(intent.current_manifest.fallbacks.iter().copied())
        {
            let key = CoverageKey::from_location(location);
            let count = expected_owner_counts.entry(key).or_default();
            *count = count
                .checked_add(1)
                .ok_or(CoverageInvariantError::InvalidHoldOwner(intent.id))?;
        }
    }
    for (&key, node) in nodes {
        if node.coverage_owners(feature)
            != expected_owner_counts.get(&key).copied().unwrap_or_default()
        {
            return Err(CoverageInvariantError::InvalidPartition);
        }
    }
    Ok(())
}

fn has_complete_active_frontier_under(
    key: CoverageKey,
    active: &OrdSet<CoverageKey>,
    lod_count: u8,
) -> Result<bool, CoverageInvariantError> {
    if active.contains(&key) {
        return Ok(true);
    }
    if key.lod == 0 {
        return Ok(false);
    }
    for child in checked_children(key.location(), lod_count)? {
        if !has_complete_active_frontier_under(
            CoverageKey::from_location(child),
            active,
            lod_count,
        )? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vector3i;
    use crate::meshers::MeshBlockLocation;
    use crate::terrain::clipbox_coordinator::DemandCounts;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn loc(position_in_blocks: Vector3i, lod_index: u8) -> MeshBlockLocation {
        MeshBlockLocation::new(position_in_blocks, lod_index)
    }

    fn counts(
        resident: u32,
        visuals: u32,
        collisions: u32,
        visual_splits: u32,
        collision_splits: u32,
    ) -> DemandCounts {
        DemandCounts {
            resident,
            visuals,
            collisions,
            visual_splits,
            collision_splits,
        }
    }

    fn snapshot(revision: u64, visuals: bool, collisions: bool) -> AcceptedFeatureSnapshot {
        AcceptedFeatureSnapshot {
            revision,
            visuals,
            collisions,
        }
    }

    fn demand(location: MeshBlockLocation, counts: DemandCounts) -> CoverageInput {
        CoverageInput::SetDemand { location, counts }
    }

    fn accepted(location: MeshBlockLocation, snapshot: AcceptedFeatureSnapshot) -> CoverageInput {
        CoverageInput::Accept { location, snapshot }
    }

    fn commit(
        coverage: &mut VariableLodCoverage,
        inputs: &[CoverageInput],
    ) -> CoverageReconcileResult {
        let preview = coverage.preview_reconcile(inputs).unwrap();
        coverage.apply_preview(preview).unwrap()
    }

    fn only_root() -> MeshBlockLocation {
        loc(Vector3i::zero(), 2)
    }

    fn first_root() -> MeshBlockLocation {
        loc(Vector3i::new(-2, 0, 0), 2)
    }

    fn second_root() -> MeshBlockLocation {
        loc(Vector3i::new(2, 0, 0), 2)
    }

    fn first_root_inputs() -> Vec<CoverageInput> {
        vec![
            demand(first_root(), counts(1, 1, 0, 0, 0)),
            accepted(first_root(), snapshot(1, true, false)),
        ]
    }

    fn second_root_inputs() -> Vec<CoverageInput> {
        vec![
            demand(second_root(), counts(1, 1, 0, 0, 0)),
            accepted(second_root(), snapshot(1, true, false)),
        ]
    }

    fn parent() -> MeshBlockLocation {
        only_root()
    }

    fn children(parent: MeshBlockLocation) -> Vec<MeshBlockLocation> {
        checked_children(parent, 3).unwrap()
    }

    fn assert_partition(coverage: &VariableLodCoverage, feature: CoverageFeature) {
        coverage.validate_partition(feature).unwrap();

        let mut expected_branch_counts = BTreeMap::<CoverageKey, u64>::new();
        for key in active_set(&coverage.state, feature) {
            let mut current = Some(key.location());
            while let Some(location) = current {
                *expected_branch_counts
                    .entry(CoverageKey::from_location(location))
                    .or_default() += 1;
                current = checked_parent(location, coverage.lod_count).unwrap();
            }
        }
        for (key, node) in &coverage.state.nodes {
            assert_eq!(
                node.branch_active_nodes(feature),
                expected_branch_counts.get(key).copied().unwrap_or(0),
                "stale {feature:?} active-branch summary at {:?}",
                key.location()
            );
        }
    }

    fn converged_root_with_snapshot(snapshot: AcceptedFeatureSnapshot) -> VariableLodCoverage {
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        let location = only_root();
        let demand_counts = counts(
            1,
            u32::from(snapshot.visuals),
            u32::from(snapshot.collisions),
            0,
            0,
        );
        commit(
            &mut coverage,
            &[
                demand(location, demand_counts),
                accepted(location, snapshot),
            ],
        );
        coverage
    }

    fn converged_root_with_both_features() -> VariableLodCoverage {
        converged_root_with_snapshot(snapshot(8, true, true))
    }

    #[test]
    fn preview_point_queries_observe_projected_state_without_publishing() {
        let coverage = VariableLodCoverage::try_new(3).unwrap();
        let location = only_root();
        let accepted_snapshot = snapshot(37, true, false);
        let preview = coverage
            .preview_reconcile(&[
                demand(location, counts(1, 1, 0, 0, 0)),
                accepted(location, accepted_snapshot),
            ])
            .unwrap();

        assert!(!coverage.contains_node(location));
        assert!(!coverage.is_active(location, CoverageFeature::Visual));
        assert_eq!(
            preview.next_accepted_snapshot(location),
            Some(accepted_snapshot)
        );
        assert!(preview.next_is_active(location, CoverageFeature::Visual));
        assert!(!preview.next_is_active(location, CoverageFeature::Collision));

        let validated = preview.validate_for(&coverage).unwrap();
        assert_eq!(
            validated.next_accepted_snapshot(location),
            Some(accepted_snapshot)
        );
        assert!(validated.next_is_active(location, CoverageFeature::Visual));
        assert!(!coverage.contains_node(location));
    }

    #[test]
    fn preview_point_queries_return_absent_values_for_unknown_location() {
        let coverage = VariableLodCoverage::try_new(3).unwrap();
        let preview = coverage.preview_reconcile(&[]).unwrap();
        let unknown = loc(Vector3i::new(99, -7, 3), 1);

        assert_eq!(preview.next_accepted_snapshot(unknown), None);
        assert!(!preview.next_is_active(unknown, CoverageFeature::Visual));
        assert!(!preview.next_is_active(unknown, CoverageFeature::Collision));
    }

    fn demanded_parent_and_children_with_ready_parent() -> VariableLodCoverage {
        let mut coverage = converged_root_with_snapshot(snapshot(1, true, false));
        let child_inputs = children(parent())
            .into_iter()
            .map(|child| demand(child, counts(1, 1, 0, 0, 0)))
            .collect::<Vec<_>>();
        commit(&mut coverage, &child_inputs);
        coverage.force_active_for_test(parent(), CoverageFeature::Visual, false);
        coverage
    }

    fn feature_counts(
        feature: CoverageFeature,
        resident: u32,
        demanded: u32,
        splits: u32,
    ) -> DemandCounts {
        match feature {
            CoverageFeature::Visual => counts(resident, demanded, 0, splits, 0),
            CoverageFeature::Collision => counts(resident, 0, demanded, 0, splits),
        }
    }

    fn feature_snapshot(revision: u64, feature: CoverageFeature) -> AcceptedFeatureSnapshot {
        match feature {
            CoverageFeature::Visual => snapshot(revision, true, false),
            CoverageFeature::Collision => snapshot(revision, false, true),
        }
    }

    fn split_fixture(feature: CoverageFeature) -> VariableLodCoverage {
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        let mut inputs = vec![
            demand(parent(), feature_counts(feature, 1, 1, 1)),
            accepted(parent(), feature_snapshot(1, feature)),
        ];
        inputs.extend(
            children(parent())
                .into_iter()
                .map(|child| demand(child, feature_counts(feature, 1, 1, 0))),
        );
        commit(&mut coverage, &inputs);
        coverage
    }

    fn seven_ready_children_fixture(feature: CoverageFeature) -> VariableLodCoverage {
        let mut coverage = split_fixture(feature);
        let inputs = children(parent())
            .into_iter()
            .take(7)
            .enumerate()
            .map(|(index, child)| accepted(child, feature_snapshot(10 + index as u64, feature)))
            .collect::<Vec<_>>();
        commit(&mut coverage, &inputs);
        coverage
    }

    fn active_children_with_unready_parent(feature: CoverageFeature) -> VariableLodCoverage {
        let mut coverage = seven_ready_children_fixture(feature);
        commit(
            &mut coverage,
            &[accepted(
                children(parent())[7],
                feature_snapshot(18, feature),
            )],
        );

        let key = CoverageKey::from_location(parent());
        let mut nodes = coverage.state.nodes.clone();
        let mut node = nodes.get(&key).cloned().unwrap();
        node.accepted_snapshot = None;
        nodes.insert(key, node);
        coverage.state = Arc::new(CoverageState {
            revision: coverage.state.revision,
            next_transition_generation: coverage.state.next_transition_generation,
            nodes,
            visual_active: coverage.state.visual_active.clone(),
            collision_active: coverage.state.collision_active.clone(),
            pending_joins: coverage.state.pending_joins.clone(),
        });
        coverage
    }

    fn exit_split_frontier_inputs() -> Vec<CoverageInput> {
        let mut inputs = vec![demand(parent(), DemandCounts::default())];
        inputs.extend(
            children(parent())
                .into_iter()
                .map(|child| demand(child, DemandCounts::default())),
        );
        inputs
    }

    fn pending_join_fixture(feature: CoverageFeature) -> VariableLodCoverage {
        let mut coverage = active_children_with_unready_parent(feature);
        commit(&mut coverage, &exit_split_frontier_inputs());
        assert!(coverage.pending_join_id(parent(), feature).is_some());
        coverage
    }

    fn active_children_for_both_features_with_unready_parent() -> VariableLodCoverage {
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        let mut inputs = vec![
            demand(parent(), counts(1, 1, 1, 1, 1)),
            accepted(parent(), snapshot(1, true, true)),
        ];
        for (index, child) in children(parent()).into_iter().enumerate() {
            inputs.push(demand(child, counts(1, 1, 1, 0, 0)));
            inputs.push(accepted(child, snapshot(10 + index as u64, true, true)));
        }
        commit(&mut coverage, &inputs);

        let key = CoverageKey::from_location(parent());
        let mut nodes = coverage.state.nodes.clone();
        let mut node = nodes.get(&key).cloned().unwrap();
        node.accepted_snapshot = None;
        nodes.insert(key, node);
        coverage.state = Arc::new(CoverageState {
            revision: coverage.state.revision,
            next_transition_generation: coverage.state.next_transition_generation,
            nodes,
            visual_active: coverage.state.visual_active.clone(),
            collision_active: coverage.state.collision_active.clone(),
            pending_joins: coverage.state.pending_joins.clone(),
        });
        coverage
    }

    fn many_pending_owner_fixture(count: usize) -> VariableLodCoverage {
        let template = pending_join_fixture(CoverageFeature::Visual);
        let template_root = CoverageKey::from_location(parent());
        let template_root_node = template.state.nodes.get(&template_root).unwrap().clone();
        let template_children = children(parent())
            .into_iter()
            .map(|child| {
                (
                    CoverageKey::from_location(child),
                    template
                        .state
                        .nodes
                        .get(&CoverageKey::from_location(child))
                        .unwrap()
                        .clone(),
                )
            })
            .collect::<Vec<_>>();
        let mut nodes = OrdMap::new();
        let mut visual_active = OrdSet::new();
        let mut pending_joins = OrdMap::new();
        for index in 0..count {
            let x = i32::try_from(index).unwrap();
            let target = loc(Vector3i::new(x, 0, 0), 2);
            let target_key = CoverageKey::from_location(target);
            nodes.insert(target_key, template_root_node.clone());
            let mut fallbacks = checked_children(target, 3).unwrap();
            fallbacks.sort_by_key(|location| location_key(*location));
            for (child, (_, child_node)) in checked_children(target, 3)
                .unwrap()
                .into_iter()
                .zip(&template_children)
            {
                let key = CoverageKey::from_location(child);
                nodes.insert(key, child_node.clone());
                visual_active.insert(key);
            }
            let generation = u64::try_from(index + 1).unwrap();
            let id = CoverageHoldId {
                join_parent: target,
                feature: CoverageFeature::Visual,
                transition_generation: generation,
            };
            let manifest = CoverageHoldIntentManifest {
                id,
                target,
                fallbacks,
            };
            pending_joins.insert(
                PendingJoinKey::new(target, CoverageFeature::Visual),
                PendingJoinIntent {
                    id,
                    target,
                    feature: CoverageFeature::Visual,
                    current_manifest: manifest,
                    target_state: PendingJoinTargetState::default(),
                },
            );
        }
        VariableLodCoverage {
            lod_count: 3,
            state: Arc::new(CoverageState {
                revision: 1,
                next_transition_generation: u64::try_from(count).unwrap(),
                nodes,
                visual_active,
                collision_active: OrdSet::new(),
                pending_joins,
            }),
        }
    }

    fn return_first_pending_owner_inputs() -> Vec<CoverageInput> {
        let target = parent();
        let mut inputs = vec![demand(
            target,
            feature_counts(CoverageFeature::Visual, 1, 1, 1),
        )];
        inputs.extend(
            children(target)
                .into_iter()
                .map(|child| demand(child, feature_counts(CoverageFeature::Visual, 1, 1, 0))),
        );
        inputs
    }

    fn arbitrary_pending_branch_with_demanded_sibling(
        feature: CoverageFeature,
    ) -> (VariableLodCoverage, MeshBlockLocation, MeshBlockLocation) {
        let root = parent();
        let branch = children(root)[0];
        let sibling = children(root)[1];
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        commit(&mut coverage, &nested_ready_split_inputs(feature));

        let key = CoverageKey::from_location(branch);
        let mut nodes = coverage.state.nodes.clone();
        let mut node = nodes.get(&key).cloned().unwrap();
        node.accepted_snapshot = None;
        nodes.insert(key, node);
        coverage.state = Arc::new(CoverageState {
            revision: coverage.state.revision,
            next_transition_generation: coverage.state.next_transition_generation,
            nodes,
            visual_active: coverage.state.visual_active.clone(),
            collision_active: coverage.state.collision_active.clone(),
            pending_joins: coverage.state.pending_joins.clone(),
        });

        let mut inputs = vec![
            demand(root, DemandCounts::default()),
            demand(branch, DemandCounts::default()),
        ];
        inputs.extend(
            children(branch)
                .into_iter()
                .map(|child| demand(child, DemandCounts::default())),
        );
        commit(&mut coverage, &inputs);
        assert!(coverage.pending_join_id(branch, feature).is_some());
        assert!(coverage.pending_join_id(root, feature).is_some());
        (coverage, branch, sibling)
    }

    fn nested_ready_split_inputs(feature: CoverageFeature) -> Vec<CoverageInput> {
        let root = parent();
        let direct_children = children(root);
        let nested_parent = direct_children[0];
        let mut inputs = vec![
            demand(root, feature_counts(feature, 1, 1, 1)),
            accepted(root, feature_snapshot(1, feature)),
        ];
        for (index, child) in direct_children.into_iter().enumerate() {
            inputs.push(demand(
                child,
                feature_counts(feature, 1, 1, u32::from(child == nested_parent)),
            ));
            inputs.push(accepted(
                child,
                feature_snapshot(10 + index as u64, feature),
            ));
        }
        for (index, grandchild) in children(nested_parent).into_iter().enumerate() {
            inputs.push(demand(grandchild, feature_counts(feature, 1, 1, 0)));
            inputs.push(accepted(
                grandchild,
                feature_snapshot(20 + index as u64, feature),
            ));
        }
        inputs
    }

    fn active_zero_viewer_fixture() -> VariableLodCoverage {
        let mut coverage = converged_root_with_snapshot(snapshot(9, true, true));
        coverage.force_demand_for_test(only_root(), DemandCounts::default());
        coverage
    }

    fn independent_root_inputs() -> Vec<CoverageInput> {
        vec![
            accepted(second_root(), snapshot(2, false, true)),
            demand(first_root(), counts(1, 1, 0, 0, 0)),
            accepted(first_root(), snapshot(1, true, false)),
            demand(second_root(), counts(1, 0, 1, 0, 0)),
        ]
    }

    fn deterministic_permutations<T: Clone>(values: Vec<T>) -> Vec<Vec<T>> {
        fn recurse<T: Clone>(prefix: &mut Vec<T>, rest: &mut Vec<T>, output: &mut Vec<Vec<T>>) {
            if rest.is_empty() {
                output.push(prefix.clone());
                return;
            }
            for index in 0..rest.len() {
                let value = rest.remove(index);
                prefix.push(value.clone());
                recurse(prefix, rest, output);
                prefix.pop();
                rest.insert(index, value);
            }
        }

        let mut output = Vec::new();
        recurse(&mut Vec::new(), &mut values.clone(), &mut output);
        output
    }

    fn coverage_at_revision_for_test(revision: u64) -> VariableLodCoverage {
        VariableLodCoverage::at_revision_for_test(3, revision)
    }

    fn valid_new_root_input() -> CoverageInput {
        demand(only_root(), counts(1, 1, 0, 0, 0))
    }

    fn canonical_invalid_lod_location() -> MeshBlockLocation {
        loc(Vector3i::new(-2, 0, 0), 3)
    }

    fn mixed_invalid_inputs() -> Vec<CoverageInput> {
        vec![
            demand(only_root(), counts(0, 1, 0, 0, 0)),
            demand(canonical_invalid_lod_location(), counts(1, 1, 0, 0, 0)),
            accepted(second_root(), snapshot(1, true, false)),
        ]
    }

    fn independent_root_activation_and_deactivation_fixture() -> VariableLodCoverage {
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        commit(&mut coverage, &independent_root_inputs());
        coverage
    }

    fn third_root() -> MeshBlockLocation {
        loc(Vector3i::new(5, -1, 1), 2)
    }

    fn root_change_inputs() -> Vec<CoverageInput> {
        vec![
            demand(first_root(), DemandCounts::default()),
            demand(third_root(), counts(1, 1, 0, 0, 0)),
            accepted(third_root(), snapshot(3, true, false)),
        ]
    }

    fn location_key(location: MeshBlockLocation) -> (u8, i32, i32, i32) {
        (
            location.lod_index,
            location.position_in_blocks.x,
            location.position_in_blocks.y,
            location.position_in_blocks.z,
        )
    }

    fn replay_and_validate_groups(
        lod_count: u8,
        old_visual: Vec<MeshBlockLocation>,
        old_collision: Vec<MeshBlockLocation>,
        groups: &[FeatureTopologyGroup],
    ) -> Result<(Vec<MeshBlockLocation>, Vec<MeshBlockLocation>), CoverageInvariantError> {
        let mut visual = old_visual
            .into_iter()
            .map(|location| (location_key(location), location))
            .collect::<BTreeMap<_, _>>();
        let mut collision = old_collision
            .into_iter()
            .map(|location| (location_key(location), location))
            .collect::<BTreeMap<_, _>>();
        for (group_index, group) in groups.iter().enumerate() {
            let active = match group.feature {
                CoverageFeature::Visual => &mut visual,
                CoverageFeature::Collision => &mut collision,
            };
            match group.operation {
                TopologyOperation::RootActivate
                    if group.anchor.lod_index + 1 == lod_count
                        && group.activate == vec![group.anchor]
                        && group.deactivate.is_empty() =>
                {
                    if active
                        .insert(location_key(group.anchor), group.anchor)
                        .is_some()
                    {
                        return Err(CoverageInvariantError::InvalidTopologyGroup { group_index });
                    }
                }
                TopologyOperation::FrontierActivate
                    if group.anchor.lod_index + 1 < lod_count
                        && group.activate == vec![group.anchor]
                        && group.deactivate.is_empty() =>
                {
                    if active
                        .insert(location_key(group.anchor), group.anchor)
                        .is_some()
                    {
                        return Err(CoverageInvariantError::InvalidTopologyGroup { group_index });
                    }
                }
                TopologyOperation::RootDeactivate
                    if group.anchor.lod_index + 1 == lod_count
                        && group.deactivate == vec![group.anchor]
                        && group.activate.is_empty() =>
                {
                    if active.remove(&location_key(group.anchor)).is_none() {
                        return Err(CoverageInvariantError::InvalidTopologyGroup { group_index });
                    }
                }
                TopologyOperation::Split
                    if group.anchor.lod_index > 0
                        && group.activate == checked_children(group.anchor, lod_count)?
                        && group.deactivate == vec![group.anchor] =>
                {
                    if active.remove(&location_key(group.anchor)).is_none() {
                        return Err(CoverageInvariantError::InvalidTopologyGroup { group_index });
                    }
                    for child in &group.activate {
                        if active.insert(location_key(*child), *child).is_some() {
                            return Err(CoverageInvariantError::InvalidTopologyGroup {
                                group_index,
                            });
                        }
                    }
                }
                TopologyOperation::Join
                    if group.anchor.lod_index > 0
                        && group.activate == vec![group.anchor]
                        && group.deactivate == checked_children(group.anchor, lod_count)? =>
                {
                    if active
                        .insert(location_key(group.anchor), group.anchor)
                        .is_some()
                    {
                        return Err(CoverageInvariantError::InvalidTopologyGroup { group_index });
                    }
                    for child in &group.deactivate {
                        if active.remove(&location_key(*child)).is_none() {
                            return Err(CoverageInvariantError::InvalidTopologyGroup {
                                group_index,
                            });
                        }
                    }
                }
                TopologyOperation::FrontierDeactivate
                    if group.anchor.lod_index + 1 == lod_count
                        && group.activate.is_empty()
                        && !group.deactivate.is_empty()
                        && group
                            .deactivate
                            .windows(2)
                            .all(|pair| location_key(pair[0]) < location_key(pair[1])) =>
                {
                    for child in &group.deactivate {
                        if active.remove(&location_key(*child)).is_none() {
                            return Err(CoverageInvariantError::InvalidTopologyGroup {
                                group_index,
                            });
                        }
                    }
                }
                _ => {
                    return Err(CoverageInvariantError::InvalidTopologyGroup { group_index });
                }
            }
        }
        let visual = visual.into_values().collect::<Vec<_>>();
        let collision = collision.into_values().collect::<Vec<_>>();
        Ok((visual, collision))
    }

    fn many_independent_roots_fixture(count: i32) -> VariableLodCoverage {
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        let mut inputs = Vec::new();
        for x in 0..count {
            let location = loc(Vector3i::new(x, 0, 0), 2);
            inputs.push(demand(location, counts(1, 1, 0, 0, 0)));
            inputs.push(accepted(location, snapshot(1, true, false)));
        }
        commit(&mut coverage, &inputs);
        coverage
    }

    fn same_root_descendant_leaves_fixture(count: usize) -> VariableLodCoverage {
        let mut coverage = VariableLodCoverage::try_new(5).unwrap();
        let root = loc(Vector3i::zero(), 4);
        let mut inputs = vec![
            demand(root, counts(1, 1, 0, 0, 0)),
            accepted(root, snapshot(1, true, false)),
        ];
        for index in 0..count {
            let x = i32::try_from(index % 16).unwrap();
            let y = i32::try_from((index / 16) % 16).unwrap();
            let z = i32::try_from(index / 256).unwrap();
            inputs.push(demand(
                loc(Vector3i::new(x, y, z), 0),
                counts(1, 1, 0, 0, 0),
            ));
        }
        commit(&mut coverage, &inputs);
        coverage
    }

    fn refresh_first_same_root_leaf(coverage: &VariableLodCoverage) -> CoveragePreview {
        coverage
            .preview_reconcile(&[accepted(loc(Vector3i::zero(), 0), snapshot(2, true, false))])
            .unwrap()
    }

    fn change_first_root() -> Vec<CoverageInput> {
        let location = loc(Vector3i::zero(), 2);
        vec![
            demand(location, counts(1, 0, 1, 0, 0)),
            accepted(location, snapshot(2, false, true)),
        ]
    }

    #[test]
    fn negative_and_non_origin_coarsest_cells_are_exact_independent_roots() {
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        let negative = loc(Vector3i::new(-1, 0, 0), 2);
        let positive = loc(Vector3i::new(3, -2, 1), 2);
        let preview = coverage
            .preview_reconcile(&[
                demand(positive, counts(1, 1, 0, 0, 0)),
                accepted(negative, snapshot(1, true, false)),
                demand(negative, counts(1, 1, 0, 0, 0)),
                accepted(positive, snapshot(2, true, false)),
            ])
            .unwrap();

        assert!(coverage
            .active_locations(CoverageFeature::Visual)
            .is_empty());
        assert_eq!(
            preview.result().topology.visual_activations(),
            vec![negative, positive],
        );
        coverage.apply_preview(preview).unwrap();
        assert_eq!(
            coverage.active_locations(CoverageFeature::Visual),
            vec![negative, positive],
        );
        assert_eq!(
            checked_parent(negative, coverage.lod_count()).unwrap(),
            None
        );

        commit(&mut coverage, &[demand(negative, DemandCounts::default())]);
        assert_eq!(
            coverage.active_locations(CoverageFeature::Visual),
            vec![positive],
        );
        assert!(coverage.is_resident(positive));
    }

    #[test]
    fn preview_is_non_mutating_and_resident_overlap_keeps_only_ready_fallback_active() {
        let coverage = demanded_parent_and_children_with_ready_parent();
        let before = coverage.clone();
        let preview = coverage.preview_reconcile(&[]).unwrap();
        assert_eq!(coverage, before);
        assert!(preview
            .result()
            .topology
            .visual_activations()
            .contains(&parent()));

        let mut coverage = coverage;
        coverage.apply_preview(preview).unwrap();
        assert!(coverage.is_resident(parent()));
        for child in children(parent()) {
            assert!(coverage.is_resident(child));
            assert!(!coverage.is_active(child, CoverageFeature::Visual));
        }
        assert_partition(&coverage, CoverageFeature::Visual);
    }

    #[test]
    fn parent_stays_active_until_all_eight_children_are_ready() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            for ready_count in 0..8 {
                let mut coverage = split_fixture(feature);
                let inputs = children(parent())
                    .into_iter()
                    .take(ready_count)
                    .enumerate()
                    .map(|(index, child)| {
                        accepted(child, feature_snapshot(10 + index as u64, feature))
                    })
                    .collect::<Vec<_>>();

                let batch = commit(&mut coverage, &inputs).topology;

                assert!(coverage.is_active(parent(), feature));
                assert_eq!(coverage.active_locations(feature), vec![parent()]);
                assert!(batch
                    .groups
                    .iter()
                    .filter(|group| group.feature == feature)
                    .all(|group| group.deactivate.is_empty()));
                assert_partition(&coverage, feature);
            }
        }
    }

    #[test]
    fn eighth_ready_child_commits_one_atomic_split_batch() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let mut coverage = seven_ready_children_fixture(feature);
            let old_visual = coverage.active_locations(CoverageFeature::Visual);
            let old_collision = coverage.active_locations(CoverageFeature::Collision);

            let batch = commit(
                &mut coverage,
                &[accepted(
                    children(parent())[7],
                    feature_snapshot(18, feature),
                )],
            )
            .topology;

            assert_eq!(
                batch.groups,
                vec![FeatureTopologyGroup {
                    feature,
                    operation: TopologyOperation::Split,
                    anchor: parent(),
                    activate: children(parent()),
                    deactivate: vec![parent()],
                }]
            );
            let (replayed_visual, replayed_collision) = replay_and_validate_groups(
                coverage.lod_count(),
                old_visual,
                old_collision,
                &batch.groups,
            )
            .unwrap();
            assert_eq!(
                replayed_visual,
                coverage.active_locations(CoverageFeature::Visual)
            );
            assert_eq!(
                replayed_collision,
                coverage.active_locations(CoverageFeature::Collision)
            );
            assert_partition(&coverage, feature);
        }
    }

    #[test]
    fn nested_ready_splits_are_dependency_ordered_coarse_to_fine() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let mut coverage = VariableLodCoverage::try_new(3).unwrap();
            let result = commit(&mut coverage, &nested_ready_split_inputs(feature));
            let split_anchors = result
                .topology
                .groups
                .iter()
                .filter(|group| {
                    group.feature == feature && group.operation == TopologyOperation::Split
                })
                .map(|group| group.anchor)
                .collect::<Vec<_>>();

            assert_eq!(split_anchors, vec![parent(), children(parent())[0]]);
            assert_partition(&coverage, feature);
        }
    }

    #[test]
    fn parent_split_revisits_previously_ready_nested_branch() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let root = parent();
            let direct_children = children(root);
            let nested_parent = direct_children[0];
            let delayed_child = direct_children[7];
            let mut coverage = VariableLodCoverage::try_new(3).unwrap();
            let mut initial = vec![
                demand(root, feature_counts(feature, 1, 1, 1)),
                accepted(root, feature_snapshot(1, feature)),
            ];
            for (index, child) in direct_children.iter().copied().enumerate() {
                initial.push(demand(
                    child,
                    feature_counts(feature, 1, 1, u32::from(child == nested_parent)),
                ));
                if child != delayed_child {
                    initial.push(accepted(
                        child,
                        feature_snapshot(10 + index as u64, feature),
                    ));
                }
            }
            for (index, grandchild) in children(nested_parent).into_iter().enumerate() {
                initial.push(demand(grandchild, feature_counts(feature, 1, 1, 0)));
                initial.push(accepted(
                    grandchild,
                    feature_snapshot(20 + index as u64, feature),
                ));
            }
            commit(&mut coverage, &initial);
            assert_eq!(coverage.active_locations(feature), vec![root]);

            let result = commit(
                &mut coverage,
                &[accepted(delayed_child, feature_snapshot(18, feature))],
            );
            let split_anchors = result
                .topology
                .groups
                .iter()
                .filter(|group| {
                    group.feature == feature && group.operation == TopologyOperation::Split
                })
                .map(|group| group.anchor)
                .collect::<Vec<_>>();

            assert_eq!(split_anchors, vec![root, nested_parent]);
            assert_partition(&coverage, feature);
        }
    }

    #[test]
    fn split_frontier_survives_child_refresh_and_repeated_demand_without_root_reactivation() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let mut coverage = seven_ready_children_fixture(feature);
            commit(
                &mut coverage,
                &[accepted(
                    children(parent())[7],
                    feature_snapshot(18, feature),
                )],
            );
            let child = children(parent())[0];

            let result = commit(
                &mut coverage,
                &[
                    demand(child, feature_counts(feature, 1, 1, 0)),
                    accepted(child, feature_snapshot(100, feature)),
                ],
            );

            assert!(result.topology.groups.is_empty());
            assert!(!coverage.is_active(parent(), feature));
            let mut expected = children(parent());
            expected.sort_by_key(|location| location_key(*location));
            assert_eq!(coverage.active_locations(feature), expected);
            assert_partition(&coverage, feature);
        }
    }

    #[test]
    fn removing_split_demand_commits_one_atomic_join_to_the_ready_parent() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let mut coverage = seven_ready_children_fixture(feature);
            commit(
                &mut coverage,
                &[accepted(
                    children(parent())[7],
                    feature_snapshot(18, feature),
                )],
            );
            let old_visual = coverage.active_locations(CoverageFeature::Visual);
            let old_collision = coverage.active_locations(CoverageFeature::Collision);

            let result = commit(
                &mut coverage,
                &[demand(parent(), feature_counts(feature, 1, 1, 0))],
            );

            assert_eq!(
                result.topology.groups,
                vec![FeatureTopologyGroup {
                    feature,
                    operation: TopologyOperation::Join,
                    anchor: parent(),
                    activate: vec![parent()],
                    deactivate: children(parent()),
                }]
            );
            let (visual, collision) = replay_and_validate_groups(
                coverage.lod_count(),
                old_visual,
                old_collision,
                &result.topology.groups,
            )
            .unwrap();
            assert_eq!(visual, coverage.active_locations(CoverageFeature::Visual));
            assert_eq!(
                collision,
                coverage.active_locations(CoverageFeature::Collision)
            );
            assert_eq!(coverage.active_locations(feature), vec![parent()]);
            assert_partition(&coverage, feature);
        }
    }

    #[test]
    fn ready_zero_viewer_join_is_parent_first_then_root_deactivate() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let mut coverage = seven_ready_children_fixture(feature);
            commit(
                &mut coverage,
                &[accepted(
                    children(parent())[7],
                    feature_snapshot(18, feature),
                )],
            );
            let mut inputs = vec![demand(parent(), DemandCounts::default())];
            inputs.extend(
                children(parent())
                    .into_iter()
                    .map(|child| demand(child, DemandCounts::default())),
            );

            let result = commit(&mut coverage, &inputs);
            assert_eq!(
                result.topology.groups,
                vec![
                    FeatureTopologyGroup {
                        feature,
                        operation: TopologyOperation::Join,
                        anchor: parent(),
                        activate: vec![parent()],
                        deactivate: children(parent()),
                    },
                    FeatureTopologyGroup {
                        feature,
                        operation: TopologyOperation::RootDeactivate,
                        anchor: parent(),
                        activate: Vec::new(),
                        deactivate: vec![parent()],
                    },
                ]
            );
            assert_eq!(result.hold_deltas.len(), 1);
            assert_eq!(result.hold_deltas[0].old, None);
            assert!(result.hold_deltas[0].before_topology.is_some());
            assert_eq!(result.hold_deltas[0].after_topology, None);
            assert!(coverage.active_locations(feature).is_empty());
            assert_partition(&coverage, feature);
        }
    }

    #[test]
    fn ready_nested_zero_viewer_join_collapses_fine_to_coarse_then_deactivates() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let root = parent();
            let direct_children = children(root);
            let nested_parents = [direct_children[0], direct_children[7]];
            let mut initial = vec![
                demand(root, feature_counts(feature, 1, 1, 1)),
                accepted(root, feature_snapshot(1, feature)),
            ];
            for (index, child) in direct_children.iter().copied().enumerate() {
                initial.push(demand(
                    child,
                    feature_counts(feature, 1, 1, u32::from(nested_parents.contains(&child))),
                ));
                initial.push(accepted(
                    child,
                    feature_snapshot(10 + index as u64, feature),
                ));
            }
            for (parent_index, nested_parent) in nested_parents.into_iter().enumerate() {
                for (child_index, grandchild) in children(nested_parent).into_iter().enumerate() {
                    initial.push(demand(grandchild, feature_counts(feature, 1, 1, 0)));
                    initial.push(accepted(
                        grandchild,
                        feature_snapshot(100 + (parent_index * 8 + child_index) as u64, feature),
                    ));
                }
            }
            let mut coverage = VariableLodCoverage::try_new(3).unwrap();
            commit(&mut coverage, &initial);
            let mut remove = vec![demand(root, DemandCounts::default())];
            remove.extend(
                direct_children
                    .iter()
                    .copied()
                    .map(|child| demand(child, DemandCounts::default())),
            );
            for nested_parent in nested_parents {
                remove.extend(
                    children(nested_parent)
                        .into_iter()
                        .map(|child| demand(child, DemandCounts::default())),
                );
            }

            let result = commit(&mut coverage, &remove);

            assert_eq!(result.topology.groups.len(), 4);
            assert_eq!(
                result
                    .topology
                    .groups
                    .iter()
                    .map(|group| (group.operation, group.anchor))
                    .collect::<Vec<_>>(),
                vec![
                    (TopologyOperation::Join, nested_parents[0]),
                    (TopologyOperation::Join, nested_parents[1]),
                    (TopologyOperation::Join, root),
                    (TopologyOperation::RootDeactivate, root),
                ],
            );
            assert_eq!(result.hold_deltas.len(), 3);
            assert!(result
                .hold_deltas
                .iter()
                .all(|delta| delta.before_topology.is_some() && delta.after_topology.is_none()));
            assert!(coverage.active_locations(feature).is_empty());
            assert_partition(&coverage, feature);
        }
    }

    #[test]
    fn nested_joins_replay_fine_to_coarse_to_the_exact_parent_frontier() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let root = parent();
            let nested_parent = children(root)[0];
            let mut coverage = VariableLodCoverage::try_new(3).unwrap();
            commit(&mut coverage, &nested_ready_split_inputs(feature));
            let old_visual = coverage.active_locations(CoverageFeature::Visual);
            let old_collision = coverage.active_locations(CoverageFeature::Collision);

            let result = commit(
                &mut coverage,
                &[
                    demand(nested_parent, feature_counts(feature, 1, 1, 0)),
                    demand(root, feature_counts(feature, 1, 1, 0)),
                ],
            );
            let feature_groups = result
                .topology
                .groups
                .iter()
                .filter(|group| group.feature == feature)
                .cloned()
                .collect::<Vec<_>>();

            assert_eq!(feature_groups.len(), 2);
            assert_eq!(feature_groups[0].operation, TopologyOperation::Join);
            assert_eq!(feature_groups[0].anchor, nested_parent);
            assert_eq!(feature_groups[0].activate, vec![nested_parent]);
            assert_eq!(feature_groups[0].deactivate, children(nested_parent));
            assert_eq!(feature_groups[1].operation, TopologyOperation::Join);
            assert_eq!(feature_groups[1].anchor, root);
            assert_eq!(feature_groups[1].activate, vec![root]);
            assert_eq!(feature_groups[1].deactivate, children(root));
            let (visual, collision) = replay_and_validate_groups(
                coverage.lod_count(),
                old_visual,
                old_collision,
                &result.topology.groups,
            )
            .unwrap();
            assert_eq!(visual, coverage.active_locations(CoverageFeature::Visual));
            assert_eq!(
                collision,
                coverage.active_locations(CoverageFeature::Collision)
            );
            assert_eq!(coverage.active_locations(feature), vec![root]);
            assert_partition(&coverage, feature);
        }
    }

    #[test]
    fn outer_join_synthesizes_exact_inner_join_before_exact_parent_join() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let root = parent();
            let direct_children = children(root);
            let nested_parent = direct_children[0];
            let mut coverage = VariableLodCoverage::try_new(3).unwrap();
            commit(&mut coverage, &nested_ready_split_inputs(feature));

            let result = commit(
                &mut coverage,
                &[demand(root, feature_counts(feature, 1, 1, 0))],
            );

            assert_eq!(
                result.topology.groups,
                vec![
                    FeatureTopologyGroup {
                        feature,
                        operation: TopologyOperation::Join,
                        anchor: nested_parent,
                        activate: vec![nested_parent],
                        deactivate: children(nested_parent),
                    },
                    FeatureTopologyGroup {
                        feature,
                        operation: TopologyOperation::Join,
                        anchor: root,
                        activate: vec![root],
                        deactivate: direct_children,
                    },
                ]
            );
            assert_eq!(coverage.active_locations(feature), vec![root]);
            assert_partition(&coverage, feature);
        }
    }

    #[test]
    fn post_split_child_readiness_or_demand_loss_joins_to_ready_fallback() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            for lose_readiness in [false, true] {
                let mut coverage = seven_ready_children_fixture(feature);
                commit(
                    &mut coverage,
                    &[accepted(
                        children(parent())[7],
                        feature_snapshot(18, feature),
                    )],
                );
                let child = children(parent())[0];
                let input = if lose_readiness {
                    accepted(
                        child,
                        match feature {
                            CoverageFeature::Visual => snapshot(100, false, true),
                            CoverageFeature::Collision => snapshot(100, true, false),
                        },
                    )
                } else {
                    demand(child, DemandCounts::default())
                };

                let result = commit(&mut coverage, &[input]);

                assert_eq!(result.topology.groups.len(), 1);
                assert_eq!(result.topology.groups[0].feature, feature);
                assert_eq!(result.topology.groups[0].operation, TopologyOperation::Join);
                assert_eq!(result.topology.groups[0].anchor, parent());
                assert_eq!(result.topology.groups[0].activate, vec![parent()]);
                assert_eq!(result.topology.groups[0].deactivate, children(parent()));
                assert_eq!(coverage.active_locations(feature), vec![parent()]);
                assert_partition(&coverage, feature);
            }
        }
    }

    #[test]
    fn root_activate_shape_rejects_an_existing_descendant_frontier() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let mut coverage = seven_ready_children_fixture(feature);
            commit(
                &mut coverage,
                &[accepted(
                    children(parent())[7],
                    feature_snapshot(18, feature),
                )],
            );
            let invalid = FeatureTopologyGroup {
                feature,
                operation: TopologyOperation::RootActivate,
                anchor: parent(),
                activate: vec![parent()],
                deactivate: Vec::new(),
            };

            assert_eq!(
                replay_topology_groups(
                    coverage.lod_count,
                    &coverage.state,
                    coverage.state.nodes.clone(),
                    &[invalid],
                    &mut CoverageWorkCounters::default(),
                ),
                Err(CoverageInvariantError::InvalidTopologyGroup { group_index: 0 }),
            );
        }
    }

    #[test]
    fn non_coarsest_root_activate_rejects_a_sparse_owned_gap() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let (mut coverage, branch, _) = arbitrary_pending_branch_with_demanded_sibling(feature);
            let terminal_id = coverage.pending_join_id(branch, feature).unwrap();
            let terminal = coverage
                .preview_terminal_pending_join_failure(terminal_id)
                .unwrap();
            coverage.apply_preview(terminal).unwrap();
            let gap = children(branch)[0];
            let gap_key = CoverageKey::from_location(gap);
            let mut next_nodes = coverage.state.nodes.clone();
            let old_demand = next_nodes.get(&gap_key).unwrap().demand;
            apply_demand_update(
                &mut next_nodes,
                gap_key,
                old_demand,
                feature_counts(feature, 1, 1, 0),
                coverage.lod_count,
                &mut CoverageWorkCounters::default(),
            )
            .unwrap();
            let mut gap_node = next_nodes.get(&gap_key).unwrap().clone();
            gap_node.accepted_snapshot = Some(feature_snapshot(100, feature));
            next_nodes.insert(gap_key, gap_node);
            let invalid = FeatureTopologyGroup {
                feature,
                operation: TopologyOperation::RootActivate,
                anchor: gap,
                activate: vec![gap],
                deactivate: Vec::new(),
            };

            assert_eq!(
                replay_topology_groups(
                    coverage.lod_count,
                    &coverage.state,
                    next_nodes,
                    &[invalid],
                    &mut CoverageWorkCounters::default(),
                ),
                Err(CoverageInvariantError::InvalidTopologyGroup { group_index: 0 }),
            );
        }
    }

    #[test]
    fn join_shape_rejects_a_partial_descendant_frontier() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let mut coverage = seven_ready_children_fixture(feature);
            commit(
                &mut coverage,
                &[accepted(
                    children(parent())[7],
                    feature_snapshot(18, feature),
                )],
            );
            let mut next_nodes = coverage.state.nodes.clone();
            let key = CoverageKey::from_location(parent());
            let mut node = next_nodes.get(&key).cloned().unwrap();
            node.demand = feature_counts(feature, 1, 1, 0);
            next_nodes.insert(key, node);
            let invalid = FeatureTopologyGroup {
                feature,
                operation: TopologyOperation::Join,
                anchor: parent(),
                activate: vec![parent()],
                deactivate: children(parent()).into_iter().take(7).collect(),
            };

            assert_eq!(
                replay_topology_groups(
                    coverage.lod_count,
                    &coverage.state,
                    next_nodes,
                    &[invalid],
                    &mut CoverageWorkCounters::default(),
                ),
                Err(CoverageInvariantError::InvalidTopologyGroup { group_index: 0 }),
            );
        }
    }

    #[test]
    fn extreme_coarse_root_without_split_demand_activates_and_refreshes() {
        for coordinate in [i32::MIN, i32::MAX] {
            let root = loc(Vector3i::new(coordinate, 0, 0), 2);
            let mut coverage = VariableLodCoverage::try_new(3).unwrap();

            let activated = commit(
                &mut coverage,
                &[
                    demand(root, counts(1, 1, 0, 0, 0)),
                    accepted(root, snapshot(1, true, false)),
                ],
            );
            assert_eq!(activated.topology.visual_activations(), vec![root]);

            let refreshed = commit(&mut coverage, &[accepted(root, snapshot(2, true, false))]);
            assert!(refreshed.topology.groups.is_empty());
            assert_eq!(
                coverage.active_locations(CoverageFeature::Visual),
                vec![root]
            );
            assert_partition(&coverage, CoverageFeature::Visual);
        }
    }

    #[test]
    fn retain_only_and_feature_specific_demand_never_alias() {
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        let visual = loc(Vector3i::new(-1, 0, 0), 2);
        let collision = loc(Vector3i::new(0, 0, 0), 2);
        let retained = loc(Vector3i::new(1, 0, 0), 2);
        commit(
            &mut coverage,
            &[
                demand(visual, counts(1, 1, 0, 0, 0)),
                accepted(visual, snapshot(1, true, false)),
                demand(collision, counts(1, 0, 1, 0, 0)),
                accepted(collision, snapshot(2, false, true)),
                demand(retained, counts(1, 0, 0, 0, 0)),
            ],
        );
        assert_eq!(
            coverage.active_locations(CoverageFeature::Visual),
            vec![visual]
        );
        assert_eq!(
            coverage.active_locations(CoverageFeature::Collision),
            vec![collision],
        );
        assert!(coverage.is_resident(retained));
        assert!(!coverage.is_active(retained, CoverageFeature::Visual));
        assert!(!coverage.is_active(retained, CoverageFeature::Collision));
    }

    #[test]
    fn accepted_snapshot_replaces_both_feature_slots_with_one_revision() {
        let mut coverage = converged_root_with_both_features();
        let location = only_root();
        let old = coverage.readiness(location);
        let in_flight = coverage.preview_reconcile(&[]).unwrap();
        assert_eq!(coverage.readiness(location), old);
        drop(in_flight);

        commit(
            &mut coverage,
            &[accepted(location, snapshot(9, true, false))],
        );
        assert_eq!(
            coverage.readiness(location),
            FeatureReadiness {
                visual_accepted_revision: Some(9),
                collision_accepted_revision: None,
            },
        );
    }

    #[test]
    fn evict_prunes_only_an_unowned_inactive_zero_resident_node() {
        let mut coverage = converged_root_with_snapshot(snapshot(9, true, true));
        let location = only_root();
        commit(&mut coverage, &[demand(location, DemandCounts::default())]);
        commit(&mut coverage, &[CoverageInput::Evict { location }]);
        assert_eq!(coverage.readiness(location), FeatureReadiness::default());
        assert!(!coverage.contains_node(location));

        // Task 3 has no hold-owner state. Genuine current/projected owner
        // eviction protection is added and tested with the join model in Task 6.
        let protected = active_zero_viewer_fixture();
        let before = protected.clone();
        assert!(matches!(
            protected.preview_reconcile(&[CoverageInput::Evict { location }]),
            Err(CoverageInvariantError::InvalidEviction { .. }),
        ));
        assert_eq!(protected, before);
    }

    #[test]
    fn demanded_unready_descendant_keeps_ready_root_active_when_root_demand_drops() {
        let mut coverage = converged_root_with_snapshot(snapshot(1, true, false));
        let child = children(only_root())[0];
        commit(&mut coverage, &[demand(child, counts(1, 1, 0, 0, 0))]);

        let result = commit(
            &mut coverage,
            &[demand(only_root(), DemandCounts::default())],
        );

        assert!(result.topology.visual_deactivations().is_empty());
        assert!(coverage.is_active(only_root(), CoverageFeature::Visual));
        assert!(coverage.is_resident(only_root()));
        assert_partition(&coverage, CoverageFeature::Visual);
    }

    #[test]
    fn exit_then_reenter_is_unready_until_a_fresh_accept() {
        let mut coverage = converged_root_with_snapshot(snapshot(9, true, false));
        let location = only_root();
        commit(&mut coverage, &[demand(location, DemandCounts::default())]);
        commit(&mut coverage, &[CoverageInput::Evict { location }]);
        commit(&mut coverage, &[demand(location, counts(1, 1, 0, 0, 0))]);
        assert_eq!(coverage.readiness(location), FeatureReadiness::default());
        assert!(!coverage.is_active(location, CoverageFeature::Visual));
        commit(
            &mut coverage,
            &[accepted(location, snapshot(10, true, false))],
        );
        assert!(coverage.is_active(location, CoverageFeature::Visual));
    }

    #[test]
    fn equivalent_input_permutations_produce_identical_preview_and_state() {
        let inputs = independent_root_inputs();
        let mut expected = None;
        for permutation in deterministic_permutations(inputs) {
            let mut coverage = VariableLodCoverage::try_new(3).unwrap();
            let preview = coverage.preview_reconcile(&permutation).unwrap();
            let observable = preview.result().clone();
            coverage.apply_preview(preview).unwrap();
            let actual = (
                observable,
                coverage.active_locations(CoverageFeature::Visual),
                coverage.active_locations(CoverageFeature::Collision),
            );
            assert_eq!(expected.get_or_insert_with(|| actual.clone()), &actual);
        }
    }

    #[test]
    fn invalid_lod_is_exact_and_non_mutating() {
        assert!(matches!(
            VariableLodCoverage::try_new(0),
            Err(CoverageInvariantError::InvalidLodCount),
        ));
        let coverage = VariableLodCoverage::try_new(3).unwrap();
        let before = coverage.clone();
        assert!(matches!(
            coverage.preview_reconcile(&[demand(
                loc(Vector3i::zero(), 3),
                counts(1, 1, 0, 0, 0),
            )]),
            Err(CoverageInvariantError::InvalidLodIndex {
                location,
            }) if location == loc(Vector3i::zero(), 3)
        ));
        assert_eq!(coverage, before);
    }

    #[test]
    fn invalid_demand_relationships_are_exact_and_non_mutating() {
        let coverage = VariableLodCoverage::try_new(3).unwrap();
        for invalid in [
            counts(0, 1, 0, 0, 0),
            counts(1, 0, 0, 1, 0),
            counts(1, 0, 0, 0, 1),
            counts(2, 1, 0, 2, 0),
            counts(2, 0, 1, 0, 2),
        ] {
            let before = coverage.clone();
            assert!(matches!(
                coverage.preview_reconcile(&[demand(only_root(), invalid)]),
                Err(CoverageInvariantError::InvalidDemand {
                    location,
                    counts,
                }) if location == only_root() && counts == invalid
            ));
            assert_eq!(coverage, before);
        }
    }

    #[test]
    fn child_coordinate_overflow_is_exact_and_non_mutating() {
        let coverage = VariableLodCoverage::try_new(3).unwrap();
        let before = coverage.clone();
        assert!(matches!(
            checked_children(loc(Vector3i::new(i32::MAX, 0, 0), 1), 3),
            Err(CoverageInvariantError::CoordinateOverflow { .. }),
        ));
        assert_eq!(coverage, before);
    }

    #[test]
    fn coverage_revision_overflow_leaves_state_unchanged() {
        let coverage = coverage_at_revision_for_test(u64::MAX);
        let before = coverage.clone();
        assert!(matches!(
            coverage.preview_reconcile(&[valid_new_root_input()]),
            Err(CoverageInvariantError::RevisionOverflow),
        ));
        assert_eq!(coverage, before);
    }

    #[test]
    fn false_false_and_old_or_conflicting_snapshots_are_rejected() {
        let coverage = converged_root_with_snapshot(snapshot(9, true, true));
        for invalid in [
            snapshot(10, false, false),
            snapshot(8, true, true),
            snapshot(9, true, false),
        ] {
            let before = coverage.clone();
            assert!(matches!(
                coverage.preview_reconcile(&[accepted(only_root(), invalid)]),
                Err(CoverageInvariantError::InvalidAcceptedSnapshot { .. }),
            ));
            assert_eq!(coverage, before);
        }
    }

    #[test]
    fn unknown_acceptance_precedes_invalid_snapshot_for_every_input_permutation() {
        let coverage = converged_root_with_snapshot(snapshot(9, true, true));
        let unknown = second_root();
        let inputs = vec![
            accepted(only_root(), snapshot(10, false, false)),
            accepted(unknown, snapshot(1, true, false)),
        ];

        for permutation in deterministic_permutations(inputs) {
            assert!(matches!(
                coverage.preview_reconcile(&permutation),
                Err(CoverageInvariantError::UnknownAcceptedLocation(location))
                    if location == unknown
            ));
        }
    }

    #[test]
    fn duplicate_inputs_are_rejected_by_kind_without_mutation() {
        let mut accepted_coverage = VariableLodCoverage::try_new(3).unwrap();
        commit(
            &mut accepted_coverage,
            &[demand(only_root(), counts(1, 1, 0, 0, 0))],
        );
        for (coverage, inputs, kind) in [
            (
                VariableLodCoverage::try_new(3).unwrap(),
                vec![
                    demand(only_root(), counts(1, 1, 0, 0, 0)),
                    demand(only_root(), counts(1, 1, 0, 0, 0)),
                ],
                CoverageInputKind::SetDemand,
            ),
            (
                accepted_coverage.clone(),
                vec![
                    accepted(only_root(), snapshot(1, true, false)),
                    accepted(only_root(), snapshot(1, true, false)),
                ],
                CoverageInputKind::Accept,
            ),
            (
                accepted_coverage,
                vec![
                    CoverageInput::Evict {
                        location: only_root(),
                    },
                    CoverageInput::Evict {
                        location: only_root(),
                    },
                ],
                CoverageInputKind::Evict,
            ),
        ] {
            let before = coverage.clone();
            assert!(matches!(
                coverage.preview_reconcile(&inputs),
                Err(CoverageInvariantError::DuplicateInput {
                    location,
                    kind: actual_kind,
                }) if location == only_root() && actual_kind == kind
            ));
            assert_eq!(coverage, before);
        }
    }

    #[test]
    fn deterministic_validation_priority_is_independent_of_input_order() {
        let invalid = mixed_invalid_inputs();
        let expected = CoverageInvariantError::InvalidLodIndex {
            location: canonical_invalid_lod_location(),
        };
        for inputs in deterministic_permutations(invalid) {
            let coverage = VariableLodCoverage::try_new(3).unwrap();
            let error = coverage.preview_reconcile(&inputs).unwrap_err();
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn two_previews_from_one_base_reject_the_second_after_first_apply() {
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        let first = coverage.preview_reconcile(&first_root_inputs()).unwrap();
        let stale = coverage.preview_reconcile(&second_root_inputs()).unwrap();
        coverage.apply_preview(first).unwrap();
        let before = coverage.clone();
        assert!(matches!(
            coverage.apply_preview(stale),
            Err(CoverageInvariantError::StalePreviewIdentity),
        ));
        assert_eq!(coverage, before);
    }

    #[test]
    fn cross_instance_same_revision_preview_is_rejected_without_mutation() {
        let source = VariableLodCoverage::try_new(3).unwrap();
        let preview = source.preview_reconcile(&first_root_inputs()).unwrap();
        let mut other = VariableLodCoverage::try_new(3).unwrap();
        assert_eq!(source.revision(), other.revision());
        let before = other.clone();
        assert_eq!(
            other.apply_preview(preview),
            Err(CoverageInvariantError::StalePreviewIdentity),
        );
        assert_eq!(other, before);
    }

    #[test]
    fn divergent_clone_rejects_a_preview_from_its_old_shared_base() {
        let original = VariableLodCoverage::try_new(3).unwrap();
        let mut branch = original.clone();
        let stale = branch.preview_reconcile(&first_root_inputs()).unwrap();
        let diverging = branch.preview_reconcile(&second_root_inputs()).unwrap();
        branch.apply_preview(diverging).unwrap();
        let before = branch.clone();
        assert_eq!(
            branch.apply_preview(stale),
            Err(CoverageInvariantError::StalePreviewIdentity),
        );
        assert_eq!(branch, before);
    }

    #[test]
    fn undiverged_clone_accepts_preview_from_the_exact_shared_lineage() {
        let source = VariableLodCoverage::try_new(3).unwrap();
        let preview = source.preview_reconcile(&first_root_inputs()).unwrap();
        let mut draft = source.clone();
        let result = draft.apply_preview(preview).unwrap();
        assert_eq!(result.topology.visual_activations(), vec![first_root()]);
        assert!(draft.is_active(first_root(), CoverageFeature::Visual));
        assert!(source.active_locations(CoverageFeature::Visual).is_empty());
    }

    #[test]
    fn validated_coverage_publication_retires_old_and_base_arcs_after_publish() {
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        let old_state = Arc::downgrade(&coverage.state);
        let preview = coverage.preview_reconcile(&first_root_inputs()).unwrap();
        let validated = preview.validate_for(&coverage).unwrap();
        assert_eq!(
            validated.result().topology.visual_activations(),
            vec![first_root()]
        );

        let published = validated.publish(&mut coverage);
        assert!(coverage.is_active(first_root(), CoverageFeature::Visual));
        assert!(old_state.upgrade().is_some());
        let (result, retirement) = published.into_parts();
        assert_eq!(result.topology.visual_activations(), vec![first_root()]);
        assert!(old_state.upgrade().is_some());

        drop(retirement);
        assert!(old_state.upgrade().is_none());
    }

    #[test]
    fn coverage_validation_rejects_stale_base_without_mutation() {
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        let stale = coverage.preview_reconcile(&first_root_inputs()).unwrap();
        commit(&mut coverage, &second_root_inputs());
        let before = coverage.clone();

        assert!(matches!(
            stale.validate_for(&coverage),
            Err(CoverageInvariantError::StalePreviewIdentity)
        ));
        assert_eq!(coverage, before);
    }

    #[test]
    fn coverage_revalidation_rejects_state_changed_after_validation_without_mutation() {
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        let validated = coverage
            .preview_reconcile(&first_root_inputs())
            .unwrap()
            .validate_for(&coverage)
            .unwrap();
        commit(&mut coverage, &second_root_inputs());
        let before = coverage.clone();
        let before_identity = coverage.state_identity_for_test();

        assert_eq!(
            validated.revalidate_for(&coverage),
            Err(CoverageInvariantError::StalePreviewIdentity)
        );
        assert_eq!(coverage, before);
        assert_eq!(coverage.state_identity_for_test(), before_identity);
    }

    #[test]
    fn task3_root_groups_replay_to_the_exact_valid_next_partition() {
        let coverage = independent_root_activation_and_deactivation_fixture();
        let old_visual = coverage.active_locations(CoverageFeature::Visual);
        let old_collision = coverage.active_locations(CoverageFeature::Collision);
        let preview = coverage.preview_reconcile(&root_change_inputs()).unwrap();
        let (replayed_visual, replayed_collision) = replay_and_validate_groups(
            coverage.lod_count(),
            old_visual,
            old_collision,
            &preview.result().topology.groups,
        )
        .unwrap();
        assert_eq!(
            replayed_visual,
            preview.next_active_locations(CoverageFeature::Visual)
        );
        assert_eq!(
            replayed_collision,
            preview.next_active_locations(CoverageFeature::Collision),
        );
    }

    #[test]
    fn missing_demanded_active_root_is_an_invalid_partition() {
        let mut coverage = converged_root_with_snapshot(snapshot(1, true, false));
        coverage.force_active_for_test(only_root(), CoverageFeature::Visual, false);
        assert_eq!(
            coverage.validate_partition(CoverageFeature::Visual),
            Err(CoverageInvariantError::InvalidPartition),
        );
    }

    #[test]
    fn unrelated_roots_do_not_increase_incremental_coverage_work() {
        let small = many_independent_roots_fixture(2);
        let large = many_independent_roots_fixture(2_000);
        let a = small.preview_reconcile(&change_first_root()).unwrap();
        let b = large.preview_reconcile(&change_first_root()).unwrap();
        assert_eq!(a.work_counters(), b.work_counters());
        assert_eq!(a.work_counters().full_state_iterations, 0);
        assert_eq!(b.work_counters().full_state_iterations, 0);
    }

    #[test]
    fn two_vs_two_thousand_descendant_leaves_in_one_root_have_equal_local_work() {
        let small = same_root_descendant_leaves_fixture(2);
        let large = same_root_descendant_leaves_fixture(2_000);

        let a = refresh_first_same_root_leaf(&small);
        let b = refresh_first_same_root_leaf(&large);

        assert!(a.result().topology.groups.is_empty());
        assert!(b.result().topology.groups.is_empty());
        assert_eq!(a.work_counters(), b.work_counters());
        assert_eq!(a.work_counters().full_state_iterations, 0);
    }

    #[test]
    fn transition_mask_sanitizes_bits_and_faces_have_exact_normals() {
        let mask = TransitionMask::from_bits(u8::MAX);
        assert_eq!(mask.bits(), 0b11_1111);
        for face in TransitionFace::ALL {
            assert!(mask.contains(face));
        }
        assert_eq!(TransitionFace::NegativeX.normal(), Vector3i::new(-1, 0, 0));
        assert_eq!(TransitionFace::PositiveX.normal(), Vector3i::new(1, 0, 0));
        assert_eq!(TransitionFace::NegativeY.normal(), Vector3i::new(0, -1, 0));
        assert_eq!(TransitionFace::PositiveY.normal(), Vector3i::new(0, 1, 0));
        assert_eq!(TransitionFace::NegativeZ.normal(), Vector3i::new(0, 0, -1));
        assert_eq!(TransitionFace::PositiveZ.normal(), Vector3i::new(0, 0, 1));
    }

    #[test]
    fn logical_noop_preview_preserves_lineage_and_numeric_revision() {
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        let preview = coverage.preview_reconcile(&[]).unwrap();
        assert_eq!(preview.base_revision(), 0);
        assert_eq!(preview.next_revision(), 0);
        assert!(Arc::ptr_eq(&preview.base, &preview.next));
        coverage.apply_preview(preview).unwrap();
        assert_eq!(coverage.revision(), 0);
    }

    #[test]
    fn topology_batches_are_unpublished() {
        let coverage = VariableLodCoverage::try_new(3).unwrap();
        let preview = coverage.preview_reconcile(&first_root_inputs()).unwrap();
        assert_eq!(preview.result().topology.revision, 0);
    }

    #[test]
    fn immediate_ready_join_uses_ephemeral_complete_manifest() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let mut coverage = seven_ready_children_fixture(feature);
            commit(
                &mut coverage,
                &[accepted(
                    children(parent())[7],
                    feature_snapshot(18, feature),
                )],
            );
            let result = commit(
                &mut coverage,
                &[demand(parent(), feature_counts(feature, 1, 1, 0))],
            );

            assert_eq!(result.hold_deltas.len(), 1);
            let delta = &result.hold_deltas[0];
            let before = delta.before_topology.as_ref().unwrap();
            assert_eq!(delta.old, None);
            assert_eq!(delta.after_topology, None);
            assert_eq!(before.target, parent());
            let mut expected = children(parent());
            expected.sort_by_key(|location| location_key(*location));
            assert_eq!(before.fallbacks, expected);
        }
    }

    #[test]
    fn last_viewer_exit_creates_logical_pending_join_and_preserves_children() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let mut coverage = active_children_with_unready_parent(feature);
            let leaves = coverage.active_locations(feature);
            let mut exit = vec![demand(parent(), DemandCounts::default())];
            exit.extend(
                children(parent())
                    .into_iter()
                    .map(|child| demand(child, DemandCounts::default())),
            );

            let result = commit(&mut coverage, &exit);

            assert!(result.topology.groups.is_empty());
            assert_eq!(coverage.active_locations(feature), leaves);
            assert_eq!(result.hold_deltas.len(), 1);
            let delta = &result.hold_deltas[0];
            assert_eq!(delta.old, None);
            assert_eq!(delta.before_topology, delta.after_topology);
            let manifest = delta.after_topology.as_ref().unwrap();
            assert_eq!(manifest.target, parent());
            let mut expected = children(parent());
            expected.sort_by_key(|location| location_key(*location));
            assert_eq!(manifest.fallbacks, expected);
            coverage.validate_partition(feature).unwrap();
        }
    }

    #[test]
    fn nested_join_acquires_projected_lod1_fallbacks_before_lower_owner_release() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let root = parent();
            let inner = children(root)[0];
            let mut coverage = VariableLodCoverage::try_new(3).unwrap();
            commit(&mut coverage, &nested_ready_split_inputs(feature));

            let key = CoverageKey::from_location(root);
            let mut nodes = coverage.state.nodes.clone();
            let mut node = nodes.get(&key).cloned().unwrap();
            node.accepted_snapshot = None;
            nodes.insert(key, node);
            coverage.state = Arc::new(CoverageState {
                revision: coverage.state.revision,
                next_transition_generation: coverage.state.next_transition_generation,
                nodes,
                visual_active: coverage.state.visual_active.clone(),
                collision_active: coverage.state.collision_active.clone(),
                pending_joins: coverage.state.pending_joins.clone(),
            });

            let result = commit(
                &mut coverage,
                &[
                    demand(inner, feature_counts(feature, 1, 1, 0)),
                    demand(root, feature_counts(feature, 1, 1, 0)),
                ],
            );

            assert_eq!(result.topology.groups.len(), 1);
            assert_eq!(result.topology.groups[0].anchor, inner);
            assert_eq!(result.hold_deltas.len(), 2);
            let inner_delta = result
                .hold_deltas
                .iter()
                .find(|delta| {
                    delta
                        .before_topology
                        .as_ref()
                        .is_some_and(|manifest| manifest.target == inner)
                })
                .unwrap();
            assert_eq!(inner_delta.after_topology, None);
            let outer_delta = result
                .hold_deltas
                .iter()
                .find(|delta| {
                    delta
                        .before_topology
                        .as_ref()
                        .is_some_and(|manifest| manifest.target == root)
                })
                .unwrap();
            assert_eq!(outer_delta.old, None);
            assert_eq!(outer_delta.before_topology, outer_delta.after_topology);
            let mut expected = children(root);
            expected.sort_by_key(|location| location_key(*location));
            assert_eq!(
                outer_delta.after_topology.as_ref().unwrap().fallbacks,
                expected,
            );
        }
    }

    #[test]
    fn outer_before_manifest_contains_every_projected_intermediate_leaf() {
        let feature = CoverageFeature::Visual;
        let root = loc(Vector3i::zero(), 3);
        let lod2_children = checked_children(root, 4).unwrap();
        let lod2_inner = lod2_children[0];
        let lod1_children = checked_children(lod2_inner, 4).unwrap();
        let lod1_inner = lod1_children[0];
        let lod0_children = checked_children(lod1_inner, 4).unwrap();
        let mut inputs = vec![
            demand(root, feature_counts(feature, 1, 1, 1)),
            accepted(root, feature_snapshot(1, feature)),
        ];
        for (index, location) in lod2_children.iter().copied().enumerate() {
            inputs.push(demand(
                location,
                feature_counts(feature, 1, 1, u32::from(location == lod2_inner)),
            ));
            inputs.push(accepted(
                location,
                feature_snapshot(10 + index as u64, feature),
            ));
        }
        for (index, location) in lod1_children.iter().copied().enumerate() {
            inputs.push(demand(
                location,
                feature_counts(feature, 1, 1, u32::from(location == lod1_inner)),
            ));
            inputs.push(accepted(
                location,
                feature_snapshot(30 + index as u64, feature),
            ));
        }
        for (index, location) in lod0_children.iter().copied().enumerate() {
            inputs.push(demand(location, feature_counts(feature, 1, 1, 0)));
            inputs.push(accepted(
                location,
                feature_snapshot(50 + index as u64, feature),
            ));
        }
        let mut coverage = VariableLodCoverage::try_new(4).unwrap();
        commit(&mut coverage, &inputs);

        let key = CoverageKey::from_location(root);
        let mut nodes = coverage.state.nodes.clone();
        let mut node = nodes.get(&key).cloned().unwrap();
        node.accepted_snapshot = None;
        nodes.insert(key, node);
        coverage.state = Arc::new(CoverageState {
            revision: coverage.state.revision,
            next_transition_generation: coverage.state.next_transition_generation,
            nodes,
            visual_active: coverage.state.visual_active.clone(),
            collision_active: coverage.state.collision_active.clone(),
            pending_joins: coverage.state.pending_joins.clone(),
        });

        let result = commit(
            &mut coverage,
            &[
                demand(root, feature_counts(feature, 1, 1, 0)),
                demand(lod2_inner, feature_counts(feature, 1, 1, 0)),
                demand(lod1_inner, feature_counts(feature, 1, 1, 0)),
            ],
        );

        assert_eq!(
            result
                .topology
                .groups
                .iter()
                .map(|group| group.anchor)
                .collect::<Vec<_>>(),
            vec![lod1_inner, lod2_inner],
        );
        let outer = result
            .hold_deltas
            .iter()
            .find(|delta| {
                delta
                    .before_topology
                    .as_ref()
                    .is_some_and(|manifest| manifest.target == root)
            })
            .unwrap();
        let before = &outer.before_topology.as_ref().unwrap().fallbacks;
        let after = &outer.after_topology.as_ref().unwrap().fallbacks;
        let mut expected_after = lod2_children;
        expected_after.sort_by_key(|location| location_key(*location));
        assert_eq!(after, &expected_after);
        assert!(before.contains(&lod1_inner));
        assert!(before.contains(&lod2_inner));
        assert!(after.contains(&lod2_inner));
        assert!(!after.contains(&lod1_inner));
        assert!(before.len() > after.len());
    }

    #[test]
    fn pending_visual_and_collision_join_intents_are_distinct_and_idempotent() {
        let mut coverage = active_children_for_both_features_with_unready_parent();
        let mut exit = vec![demand(parent(), DemandCounts::default())];
        exit.extend(
            children(parent())
                .into_iter()
                .map(|child| demand(child, DemandCounts::default())),
        );

        let first = commit(&mut coverage, &exit);

        assert_eq!(first.hold_deltas.len(), 2);
        assert_eq!(
            first
                .hold_deltas
                .iter()
                .map(|delta| delta.before_topology.as_ref().unwrap().id.feature)
                .collect::<Vec<_>>(),
            vec![CoverageFeature::Visual, CoverageFeature::Collision],
        );
        let visual = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let collision = coverage
            .pending_join_id(parent(), CoverageFeature::Collision)
            .unwrap();
        assert_ne!(visual, collision);
        assert!(visual.transition_generation < collision.transition_generation);

        let repeated = commit(&mut coverage, &[]);
        assert!(repeated.hold_deltas.is_empty());
        assert_eq!(
            coverage.pending_join_id(parent(), CoverageFeature::Visual),
            Some(visual),
        );
        assert_eq!(
            coverage.pending_join_id(parent(), CoverageFeature::Collision),
            Some(collision),
        );
    }

    #[test]
    fn returning_split_demand_cancels_then_restart_uses_fresh_generation() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let mut coverage = pending_join_fixture(feature);
            let old_id = coverage.pending_join_id(parent(), feature).unwrap();
            let old_manifest = coverage
                .pending_join_manifest(parent(), feature)
                .unwrap()
                .clone();
            let leaves = coverage.active_locations(feature);
            let mut returning = vec![demand(parent(), feature_counts(feature, 1, 1, 1))];
            returning.extend(
                children(parent())
                    .into_iter()
                    .map(|child| demand(child, feature_counts(feature, 1, 1, 0))),
            );

            let cancelled = commit(&mut coverage, &returning);

            assert!(cancelled.topology.groups.is_empty());
            assert_eq!(cancelled.hold_deltas.len(), 1);
            assert_eq!(cancelled.hold_deltas[0].old, Some(old_manifest.clone()));
            assert_eq!(
                cancelled.hold_deltas[0].before_topology,
                Some(old_manifest.clone()),
            );
            assert_eq!(cancelled.hold_deltas[0].after_topology, None);
            assert_eq!(coverage.pending_join_id(parent(), feature), None);
            assert_eq!(coverage.active_locations(feature), leaves);

            let restarted = commit(&mut coverage, &exit_split_frontier_inputs());
            let new_id = coverage.pending_join_id(parent(), feature).unwrap();
            assert!(new_id.transition_generation > old_id.transition_generation);
            assert_eq!(restarted.hold_deltas.len(), 1);
            assert_eq!(restarted.hold_deltas[0].old, None);
            assert_eq!(
                restarted.hold_deltas[0].before_topology,
                restarted.hold_deltas[0].after_topology,
            );
        }
    }

    #[test]
    fn delayed_join_target_readiness_completes_existing_owner_then_releases() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let mut coverage = pending_join_fixture(feature);
            let old = coverage
                .pending_join_manifest(parent(), feature)
                .unwrap()
                .clone();

            let result = commit(
                &mut coverage,
                &[accepted(parent(), feature_snapshot(100, feature))],
            );

            assert_eq!(
                result
                    .topology
                    .groups
                    .iter()
                    .map(|group| group.operation)
                    .collect::<Vec<_>>(),
                vec![TopologyOperation::Join, TopologyOperation::RootDeactivate],
            );
            assert_eq!(result.hold_deltas.len(), 1);
            assert_eq!(result.hold_deltas[0].old, Some(old.clone()));
            assert_eq!(result.hold_deltas[0].before_topology, Some(old));
            assert_eq!(result.hold_deltas[0].after_topology, None);
            assert_eq!(coverage.pending_join_id(parent(), feature), None);
            assert!(coverage.active_locations(feature).is_empty());
        }
    }

    #[test]
    fn hold_generation_overflow_is_typed_and_non_mutating() {
        let mut coverage = active_children_with_unready_parent(CoverageFeature::Visual);
        coverage.state = Arc::new(CoverageState {
            revision: coverage.state.revision,
            next_transition_generation: u64::MAX,
            nodes: coverage.state.nodes.clone(),
            visual_active: coverage.state.visual_active.clone(),
            collision_active: coverage.state.collision_active.clone(),
            pending_joins: coverage.state.pending_joins.clone(),
        });
        let before = coverage.clone();

        assert!(matches!(
            coverage.preview_reconcile(&exit_split_frontier_inputs()),
            Err(CoverageInvariantError::HoldGenerationOverflow),
        ));
        assert_eq!(coverage, before);
    }

    #[test]
    fn nested_preview_order_is_independent_of_demand_input_order() {
        let feature = CoverageFeature::Visual;
        let root = parent();
        let inner = children(root)[0];
        let mut coverage = VariableLodCoverage::try_new(3).unwrap();
        commit(&mut coverage, &nested_ready_split_inputs(feature));
        let mut nodes = coverage.state.nodes.clone();
        for location in [root, inner] {
            let key = CoverageKey::from_location(location);
            let mut node = nodes.get(&key).cloned().unwrap();
            node.accepted_snapshot = None;
            nodes.insert(key, node);
        }
        coverage.state = Arc::new(CoverageState {
            revision: coverage.state.revision,
            next_transition_generation: coverage.state.next_transition_generation,
            nodes,
            visual_active: coverage.state.visual_active.clone(),
            collision_active: coverage.state.collision_active.clone(),
            pending_joins: coverage.state.pending_joins.clone(),
        });
        let inputs = vec![
            demand(root, feature_counts(feature, 1, 1, 0)),
            demand(inner, feature_counts(feature, 1, 1, 0)),
        ];
        let reversed = vec![inputs[1].clone(), inputs[0].clone()];

        let expected = coverage.preview_reconcile(&inputs).unwrap();
        let actual = coverage.preview_reconcile(&reversed).unwrap();

        assert_eq!(actual.result(), expected.result());
        assert_eq!(actual.next, expected.next);
        assert_eq!(
            expected
                .result()
                .hold_deltas
                .iter()
                .map(|delta| delta.before_topology.as_ref().unwrap().id.join_parent)
                .collect::<Vec<_>>(),
            vec![inner, root],
        );
    }

    #[test]
    fn pending_owner_rejects_target_and_fallback_eviction_without_mutation() {
        let coverage = pending_join_fixture(CoverageFeature::Visual);
        for location in [parent(), children(parent())[0]] {
            let before = coverage.clone();
            assert!(matches!(
                coverage.preview_reconcile(&[CoverageInput::Evict { location }]),
                Err(CoverageInvariantError::InvalidEviction { location: rejected })
                    if rejected == location
            ));
            assert_eq!(coverage, before);
        }
    }

    #[test]
    fn pending_owner_endpoint_validation_rejects_a_missing_target_node() {
        let mut coverage = pending_join_fixture(CoverageFeature::Visual);
        let key = CoverageKey::from_location(parent());
        let mut nodes = coverage.state.nodes.clone();
        nodes.remove(&key);
        coverage.state = Arc::new(CoverageState {
            revision: coverage.state.revision,
            next_transition_generation: coverage.state.next_transition_generation,
            nodes,
            visual_active: coverage.state.visual_active.clone(),
            collision_active: coverage.state.collision_active.clone(),
            pending_joins: coverage.state.pending_joins.clone(),
        });

        assert!(matches!(
            coverage.validate_partition(CoverageFeature::Visual),
            Err(CoverageInvariantError::InvalidHoldOwner(_)),
        ));
    }

    #[test]
    fn arbitrary_pending_join_frontier_deactivate_preserves_demanded_sibling() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let (mut coverage, branch, sibling) =
                arbitrary_pending_branch_with_demanded_sibling(feature);
            let id = coverage.pending_join_id(branch, feature).unwrap();
            let old_outer = coverage
                .pending_join_manifest(parent(), feature)
                .unwrap()
                .clone();
            let expected = coverage
                .pending_join_manifest(branch, feature)
                .unwrap()
                .fallbacks
                .clone();

            let preview = coverage.preview_terminal_pending_join_failure(id).unwrap();
            let result = coverage.apply_preview(preview).unwrap();

            assert_eq!(
                result.topology.groups,
                vec![FeatureTopologyGroup {
                    feature,
                    operation: TopologyOperation::FrontierDeactivate,
                    anchor: branch,
                    activate: Vec::new(),
                    deactivate: expected,
                }],
            );
            assert!(coverage.is_active(sibling, feature));
            assert_eq!(coverage.pending_join_id(branch, feature), None);
            let outer = coverage.pending_join_manifest(parent(), feature).unwrap();
            assert!(outer.fallbacks.iter().all(|fallback| {
                !location_is_under(*fallback, branch, coverage.lod_count()).unwrap()
            }));
            let outer_delta = result
                .hold_deltas
                .iter()
                .find(|delta| delta.old.as_ref() == Some(&old_outer))
                .unwrap();
            assert_eq!(outer_delta.before_topology, Some(old_outer));
            assert_eq!(outer_delta.after_topology.as_ref(), Some(outer));
            coverage.validate_partition(feature).unwrap();
        }
    }

    #[test]
    fn terminal_inner_failure_allows_demanded_sibling_reconciliation() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let (mut coverage, branch, sibling) =
                arbitrary_pending_branch_with_demanded_sibling(feature);
            let id = coverage.pending_join_id(branch, feature).unwrap();
            let terminal = coverage.preview_terminal_pending_join_failure(id).unwrap();
            coverage.apply_preview(terminal).unwrap();
            let old_outer = coverage
                .pending_join_manifest(parent(), feature)
                .unwrap()
                .clone();

            let preview = coverage
                .preview_reconcile(&[demand(sibling, feature_counts(feature, 2, 2, 0))])
                .unwrap();
            assert!(preview.result().topology.groups.is_empty());
            assert!(preview.result().hold_deltas.is_empty());
            coverage.apply_preview(preview).unwrap();

            assert!(coverage.is_active(sibling, feature));
            assert_eq!(
                coverage.pending_join_manifest(parent(), feature),
                Some(&old_outer),
            );
            coverage.validate_partition(feature).unwrap();
        }
    }

    #[test]
    fn terminal_sparse_branch_reentry_reactivates_gap_and_reanchors_owners() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let (mut coverage, branch, sibling) =
                arbitrary_pending_branch_with_demanded_sibling(feature);
            let terminal_id = coverage.pending_join_id(branch, feature).unwrap();
            let terminal = coverage
                .preview_terminal_pending_join_failure(terminal_id)
                .unwrap();
            coverage.apply_preview(terminal).unwrap();
            let old_outer = coverage
                .pending_join_manifest(parent(), feature)
                .unwrap()
                .clone();
            let surviving_frontier = coverage.active_locations(feature);
            let reentered = children(branch)[0];

            let preview = coverage
                .preview_reconcile(&[
                    demand(reentered, feature_counts(feature, 1, 1, 0)),
                    accepted(reentered, feature_snapshot(100, feature)),
                ])
                .unwrap();
            assert_eq!(
                preview.result().topology.groups,
                vec![FeatureTopologyGroup {
                    feature,
                    operation: TopologyOperation::FrontierActivate,
                    anchor: reentered,
                    activate: vec![reentered],
                    deactivate: Vec::new(),
                }],
            );

            let mut complete_frontier = surviving_frontier.clone();
            complete_frontier.push(reentered);
            complete_frontier.sort_by_key(|location| CoverageKey::from_location(*location));
            assert_eq!(preview.result().hold_deltas.len(), 2);
            let inner_delta = &preview.result().hold_deltas[0];
            assert_eq!(inner_delta.old, None);
            assert_eq!(
                inner_delta
                    .before_topology
                    .as_ref()
                    .map(|manifest| (manifest.target, manifest.fallbacks.as_slice())),
                Some((branch, [reentered].as_slice())),
            );
            assert_eq!(inner_delta.after_topology, inner_delta.before_topology);
            let outer_delta = &preview.result().hold_deltas[1];
            assert_eq!(outer_delta.old, Some(old_outer));
            assert_eq!(
                outer_delta
                    .before_topology
                    .as_ref()
                    .map(|manifest| manifest.fallbacks.as_slice()),
                Some(complete_frontier.as_slice()),
            );
            assert_eq!(outer_delta.after_topology, outer_delta.before_topology);

            coverage.apply_preview(preview).unwrap();
            assert!(coverage.is_active(reentered, feature));
            assert!(coverage.is_active(sibling, feature));
            assert!(surviving_frontier
                .iter()
                .all(|location| coverage.is_active(*location, feature)));
            assert_eq!(coverage.active_locations(feature), complete_frontier);
            coverage.validate_partition(feature).unwrap();
        }
    }

    #[test]
    fn release_preview_rejects_partial_sparse_owner_before_healing() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let (mut coverage, branch, _) = arbitrary_pending_branch_with_demanded_sibling(feature);
            let terminal_id = coverage.pending_join_id(branch, feature).unwrap();
            let terminal = coverage
                .preview_terminal_pending_join_failure(terminal_id)
                .unwrap();
            coverage.apply_preview(terminal).unwrap();

            let owner_key = PendingJoinKey::new(parent(), feature);
            let mut pending_joins = coverage.state.pending_joins.clone();
            let mut owner = pending_joins.get(&owner_key).unwrap().clone();
            let omitted = owner.current_manifest.fallbacks.remove(0);
            let owner_id = owner.id;
            pending_joins.insert(owner_key, owner);
            let omitted_key = CoverageKey::from_location(omitted);
            let mut nodes = coverage.state.nodes.clone();
            let mut omitted_node = nodes.get(&omitted_key).unwrap().clone();
            omitted_node.set_coverage_owners(
                feature,
                omitted_node
                    .coverage_owners(feature)
                    .checked_sub(1)
                    .unwrap(),
            );
            assert!(omitted_node.is_demanded(feature));
            nodes.insert(omitted_key, omitted_node);
            coverage.state = Arc::new(CoverageState {
                revision: coverage.state.revision,
                next_transition_generation: coverage.state.next_transition_generation,
                nodes,
                visual_active: coverage.state.visual_active.clone(),
                collision_active: coverage.state.collision_active.clone(),
                pending_joins,
            });
            let before = coverage.clone();

            assert_eq!(
                coverage
                    .preview_reconcile(&[demand(omitted, feature_counts(feature, 2, 2, 0),)])
                    .unwrap_err(),
                CoverageInvariantError::InvalidHoldOwner(owner_id),
            );
            assert_eq!(coverage, before);
        }
    }

    #[test]
    fn terminal_pending_join_rejects_any_descendant_viewer_or_split_demand() {
        for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
            let mut coverage = active_children_with_unready_parent(feature);
            commit(
                &mut coverage,
                &[demand(parent(), feature_counts(feature, 1, 1, 0))],
            );
            let id = coverage.pending_join_id(parent(), feature).unwrap();
            let before = coverage.clone();

            assert_eq!(
                coverage
                    .preview_terminal_pending_join_failure(id)
                    .unwrap_err(),
                CoverageInvariantError::InvalidHoldOwner(id),
            );
            assert_eq!(coverage, before);
        }
    }

    #[test]
    fn one_pending_owner_has_constant_work_with_unrelated_owners() {
        let small = many_pending_owner_fixture(2);
        let large = many_pending_owner_fixture(2_000);

        let a = small
            .preview_reconcile(&return_first_pending_owner_inputs())
            .unwrap()
            .work_counters();
        let b = large
            .preview_reconcile(&return_first_pending_owner_inputs())
            .unwrap()
            .work_counters();

        assert_eq!(a, b);
        assert_eq!(a.pending_owners_examined, 1);
    }

    #[test]
    fn checked_parent_uses_euclidean_division_for_negative_cells() {
        let child = loc(Vector3i::new(-1, -3, 5), 0);
        assert_eq!(
            checked_parent(child, 3).unwrap(),
            Some(loc(Vector3i::new(-1, -2, 2), 1)),
        );
    }

    #[test]
    fn accepted_snapshot_for_unknown_location_is_exact() {
        let coverage = VariableLodCoverage::try_new(3).unwrap();
        assert!(matches!(
            coverage.preview_reconcile(&[accepted(only_root(), snapshot(1, true, false))]),
            Err(CoverageInvariantError::UnknownAcceptedLocation(location))
                if location == only_root()
        ));
    }

    #[test]
    fn preview_is_not_cloneable() {
        fn assert_send<T: Send>() {}
        assert_send::<CoveragePreview>();
        let coverage = VariableLodCoverage::try_new(3).unwrap();
        let preview = coverage.preview_reconcile(&[]).unwrap();
        assert!(Arc::strong_count(&preview.base) >= 2);
    }

    fn data_location(x: i32) -> BlockLocation {
        BlockLocation {
            position: Vector3i::new(x, -2, 3),
            lod_index: 0,
        }
    }

    fn load_request(location: BlockLocation, generation: u64) -> PhysicalRequestId {
        PhysicalRequestId::Load {
            location,
            tag: TaskRequestTag::new(7, generation),
        }
    }

    fn mesh_request(target: MeshBlockLocation, generation: u64) -> PhysicalRequestId {
        PhysicalRequestId::Mesh {
            location: target,
            tag: TaskRequestTag::new(7, generation),
        }
    }

    fn outstanding_target_state(
        members: &[(BlockLocation, PhysicalRequestId)],
    ) -> PendingJoinTargetState {
        PendingJoinTargetState::from_core_validated_parts(
            members
                .iter()
                .map(|(location, _)| PendingJoinDataKey::from_location(*location))
                .collect(),
            OrdSet::new(),
            members
                .iter()
                .map(|&(location, request)| (PendingJoinDataKey::from_location(location), request))
                .collect(),
            OrdMap::new(),
            PendingJoinMeshState::AwaitingData,
        )
    }

    fn ready_target_state(
        locations: &[BlockLocation],
        mesh: PendingJoinMeshState,
    ) -> PendingJoinTargetState {
        let expected = locations
            .iter()
            .copied()
            .map(PendingJoinDataKey::from_location)
            .collect::<OrdSet<_>>();
        PendingJoinTargetState::from_core_validated_parts(
            expected.clone(),
            expected,
            OrdMap::new(),
            OrdMap::new(),
            mesh,
        )
    }

    fn core_validated_target_state(
        expected: &[BlockLocation],
        ready: &[BlockLocation],
        outstanding: &[(BlockLocation, PhysicalRequestId)],
        failures: &[(BlockLocation, PendingRequestFailure)],
        mesh: PendingJoinMeshState,
    ) -> PendingJoinTargetState {
        PendingJoinTargetState::from_core_validated_parts(
            expected
                .iter()
                .copied()
                .map(PendingJoinDataKey::from_location)
                .collect(),
            ready
                .iter()
                .copied()
                .map(PendingJoinDataKey::from_location)
                .collect(),
            outstanding
                .iter()
                .map(|&(location, request)| (PendingJoinDataKey::from_location(location), request))
                .collect(),
            failures
                .iter()
                .map(|&(location, failure)| (PendingJoinDataKey::from_location(location), failure))
                .collect(),
            mesh,
        )
    }

    fn force_target_state_for_test(
        coverage: &mut VariableLodCoverage,
        target: MeshBlockLocation,
        feature: CoverageFeature,
        target_state: PendingJoinTargetState,
    ) {
        let key = PendingJoinKey::new(target, feature);
        let mut pending_joins = coverage.state.pending_joins.clone();
        let mut intent = pending_joins.get(&key).unwrap().clone();
        intent.target_state = target_state;
        pending_joins.insert(key, intent);
        coverage.state = Arc::new(CoverageState {
            revision: coverage.state.revision,
            next_transition_generation: coverage.state.next_transition_generation,
            nodes: coverage.state.nodes.clone(),
            visual_active: coverage.state.visual_active.clone(),
            collision_active: coverage.state.collision_active.clone(),
            pending_joins,
        });
    }

    fn set_join_target_state(id: CoverageHoldId, state: PendingJoinTargetState) -> CoverageInput {
        CoverageInput::SetJoinTargetState { id, state }
    }

    #[test]
    fn new_pending_join_starts_with_conservative_empty_target_state() {
        let coverage = pending_join_fixture(CoverageFeature::Visual);
        let state = coverage
            .pending_join_target_state(parent(), CoverageFeature::Visual)
            .unwrap();

        assert!(state.data.ready.is_empty());
        assert!(state.data.outstanding.is_empty());
        assert!(state.data.failures.is_empty());
        assert_eq!(state.mesh, PendingJoinMeshState::AwaitingData);
        assert_eq!(
            state.canonical_blocker(),
            Some(PendingJoinBlocker::Mesh(PendingJoinMeshState::AwaitingData)),
        );
    }

    #[test]
    fn join_target_partition_rejects_missing_extra_and_duplicate_members() {
        let mut coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let locations = [data_location(3), data_location(-1), data_location(8)];
        let requests = [
            load_request(locations[0], 10),
            load_request(locations[1], 11),
            load_request(locations[2], 12),
        ];
        let baseline = outstanding_target_state(&[
            (locations[0], requests[0]),
            (locations[1], requests[1]),
            (locations[2], requests[2]),
        ]);
        commit(
            &mut coverage,
            &[set_join_target_state(id, baseline.clone())],
        );

        let mut missing = baseline.clone();
        missing
            .data
            .outstanding
            .remove(&PendingJoinDataKey::from_location(locations[1]));
        let mut extra = baseline.clone();
        let extra_location = data_location(20);
        let extra_request = load_request(extra_location, 13);
        extra.data.outstanding.insert(
            PendingJoinDataKey::from_location(extra_location),
            extra_request,
        );
        let mut duplicate = baseline.clone();
        duplicate
            .data
            .ready
            .insert(PendingJoinDataKey::from_location(locations[0]));

        for (state, expected_request) in [
            (missing, requests[1]),
            (extra, extra_request),
            (duplicate, requests[0]),
        ] {
            let before = coverage.clone();
            assert_eq!(
                coverage
                    .preview_reconcile(&[set_join_target_state(id, state)])
                    .unwrap_err(),
                CoverageInvariantError::StaleJoinTargetRequest {
                    id,
                    request: expected_request,
                },
            );
            assert_eq!(coverage, before);
        }
    }

    #[test]
    fn malformed_first_snapshot_reports_the_exact_missing_member_without_mutation() {
        let coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let missing = data_location(41);
        let malformed = core_validated_target_state(
            &[missing],
            &[],
            &[],
            &[],
            PendingJoinMeshState::AwaitingData,
        );
        let before = coverage.clone();

        assert_eq!(
            coverage
                .preview_reconcile(&[set_join_target_state(id, malformed)])
                .unwrap_err(),
            CoverageInvariantError::MissingJoinTargetMember {
                id,
                location: missing,
            },
        );
        assert_eq!(coverage, before);
    }

    #[test]
    fn omitting_a_previously_ready_member_reports_its_location_without_mutation() {
        let mut coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let missing = data_location(42);
        commit(
            &mut coverage,
            &[set_join_target_state(
                id,
                ready_target_state(&[missing], PendingJoinMeshState::AwaitingRequest),
            )],
        );
        let malformed = core_validated_target_state(
            &[missing],
            &[],
            &[],
            &[],
            PendingJoinMeshState::AwaitingData,
        );
        let before = coverage.clone();

        assert_eq!(
            coverage
                .preview_reconcile(&[set_join_target_state(id, malformed)])
                .unwrap_err(),
            CoverageInvariantError::MissingJoinTargetMember {
                id,
                location: missing,
            },
        );
        assert_eq!(coverage, before);
    }

    #[test]
    fn join_target_state_rejects_stale_owner_and_request_identity() {
        let mut coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let location = data_location(1);
        let request = load_request(location, 30);
        let baseline = outstanding_target_state(&[(location, request)]);

        let stale_id = CoverageHoldId {
            transition_generation: id.transition_generation + 1,
            ..id
        };
        assert_eq!(
            coverage
                .preview_reconcile(&[set_join_target_state(stale_id, baseline.clone())])
                .unwrap_err(),
            CoverageInvariantError::StaleJoinTargetRequest {
                id: stale_id,
                request,
            },
        );

        let wrong_location = data_location(2);
        let wrong_request = load_request(wrong_location, 31);
        let malformed = PendingJoinTargetState {
            data: PendingJoinDataState {
                member_universe: PendingJoinMemberUniverse::CoreValidated(
                    [PendingJoinDataKey::from_location(location)]
                        .into_iter()
                        .collect(),
                ),
                ready: OrdSet::new(),
                outstanding: [(PendingJoinDataKey::from_location(location), wrong_request)]
                    .into_iter()
                    .collect(),
                failures: OrdMap::new(),
            },
            mesh: PendingJoinMeshState::AwaitingData,
        };
        assert_eq!(
            coverage
                .preview_reconcile(&[set_join_target_state(id, malformed)])
                .unwrap_err(),
            CoverageInvariantError::StaleJoinTargetRequest {
                id,
                request: wrong_request,
            },
        );

        commit(
            &mut coverage,
            &[set_join_target_state(id, baseline.clone())],
        );
        let replacement = load_request(location, 32);
        let replaced = outstanding_target_state(&[(location, replacement)]);
        assert_eq!(
            coverage
                .preview_reconcile(&[set_join_target_state(id, replaced)])
                .unwrap_err(),
            CoverageInvariantError::StaleJoinTargetRequest {
                id,
                request: replacement,
            },
        );
    }

    #[test]
    fn late_outstanding_snapshot_cannot_overwrite_a_ready_member() {
        let mut coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let location = data_location(2);
        let request = load_request(location, 33);
        let outstanding = outstanding_target_state(&[(location, request)]);
        commit(
            &mut coverage,
            &[set_join_target_state(id, outstanding.clone())],
        );
        let ready = PendingJoinTargetState {
            data: PendingJoinDataState {
                member_universe: PendingJoinMemberUniverse::CoreValidated(
                    [PendingJoinDataKey::from_location(location)]
                        .into_iter()
                        .collect(),
                ),
                ready: [PendingJoinDataKey::from_location(location)]
                    .into_iter()
                    .collect(),
                outstanding: OrdMap::new(),
                failures: OrdMap::new(),
            },
            mesh: PendingJoinMeshState::AwaitingData,
        };
        commit(&mut coverage, &[set_join_target_state(id, ready)]);

        assert_eq!(
            coverage
                .preview_reconcile(&[set_join_target_state(id, outstanding)])
                .unwrap_err(),
            CoverageInvariantError::StaleJoinTargetRequest { id, request },
        );
    }

    #[test]
    fn failed_member_retry_requires_a_fresh_same_epoch_generation() {
        let mut coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let location = data_location(3);
        let request = load_request(location, 34);
        let outstanding = outstanding_target_state(&[(location, request)]);
        commit(&mut coverage, &[set_join_target_state(id, outstanding)]);
        let failure = PendingJoinTargetState {
            data: PendingJoinDataState {
                member_universe: PendingJoinMemberUniverse::CoreValidated(
                    [PendingJoinDataKey::from_location(location)]
                        .into_iter()
                        .collect(),
                ),
                ready: OrdSet::new(),
                outstanding: OrdMap::new(),
                failures: [(
                    PendingJoinDataKey::from_location(location),
                    PendingRequestFailure::Retryable {
                        request,
                        failure: RequestFailureKind::Cancelled,
                        attempt: 1,
                    },
                )]
                .into_iter()
                .collect(),
            },
            mesh: PendingJoinMeshState::AwaitingData,
        };
        commit(&mut coverage, &[set_join_target_state(id, failure)]);

        for stale in [
            request,
            PhysicalRequestId::Load {
                location,
                tag: TaskRequestTag::new(request.tag().request_epoch + 1, 35),
            },
        ] {
            let state = outstanding_target_state(&[(location, stale)]);
            assert_eq!(
                coverage
                    .preview_reconcile(&[set_join_target_state(id, state)])
                    .unwrap_err(),
                CoverageInvariantError::StaleJoinTargetRequest { id, request: stale },
            );
        }

        let fresh = load_request(location, 35);
        coverage
            .preview_reconcile(&[set_join_target_state(
                id,
                outstanding_target_state(&[(location, fresh)]),
            )])
            .unwrap();
    }

    #[test]
    fn duplicate_join_target_inputs_are_canonical_and_non_mutating() {
        let coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let location = data_location(4);
        let state = outstanding_target_state(&[(location, load_request(location, 36))]);
        let inputs = vec![
            set_join_target_state(id, state.clone()),
            set_join_target_state(id, state),
        ];

        for permutation in deterministic_permutations(inputs) {
            let before = coverage.clone();
            assert_eq!(
                coverage.preview_reconcile(&permutation).unwrap_err(),
                CoverageInvariantError::DuplicateInput {
                    location: parent(),
                    kind: CoverageInputKind::SetJoinTargetState,
                },
            );
            assert_eq!(coverage, before);
        }
    }

    #[test]
    fn member_completion_order_converges_to_identical_join_target_state() {
        let fixture = pending_join_fixture(CoverageFeature::Visual);
        let id = fixture
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let locations = [data_location(-3), data_location(4), data_location(9)];
        let requests = [
            load_request(locations[0], 40),
            load_request(locations[1], 41),
            load_request(locations[2], 42),
        ];
        let baseline = outstanding_target_state(&[
            (locations[0], requests[0]),
            (locations[1], requests[1]),
            (locations[2], requests[2]),
        ]);
        let mut ready_first = baseline.clone();
        ready_first
            .data
            .outstanding
            .remove(&PendingJoinDataKey::from_location(locations[0]));
        ready_first
            .data
            .ready
            .insert(PendingJoinDataKey::from_location(locations[0]));
        let mut failure_first = baseline.clone();
        failure_first
            .data
            .outstanding
            .remove(&PendingJoinDataKey::from_location(locations[1]));
        failure_first.data.failures.insert(
            PendingJoinDataKey::from_location(locations[1]),
            PendingRequestFailure::Retryable {
                request: requests[1],
                failure: RequestFailureKind::Cancelled,
                attempt: 1,
            },
        );
        let mut final_state = ready_first.clone();
        final_state
            .data
            .outstanding
            .remove(&PendingJoinDataKey::from_location(locations[1]));
        final_state.data.failures.insert(
            PendingJoinDataKey::from_location(locations[1]),
            PendingRequestFailure::Retryable {
                request: requests[1],
                failure: RequestFailureKind::Cancelled,
                attempt: 1,
            },
        );

        let mut a = fixture.clone();
        commit(&mut a, &[set_join_target_state(id, baseline.clone())]);
        commit(&mut a, &[set_join_target_state(id, ready_first)]);
        commit(&mut a, &[set_join_target_state(id, final_state.clone())]);

        let mut b = fixture;
        commit(&mut b, &[set_join_target_state(id, baseline)]);
        commit(&mut b, &[set_join_target_state(id, failure_first)]);
        commit(&mut b, &[set_join_target_state(id, final_state.clone())]);

        assert_eq!(a, b);
        assert_eq!(
            a.pending_join_target_state(parent(), CoverageFeature::Visual),
            Some(&final_state),
        );
        assert_eq!(final_state.data.ready_locations(), vec![locations[0]],);
        let mut expected_locations = locations.to_vec();
        expected_locations.sort_by_key(|location| PendingJoinDataKey::from_location(*location));
        assert_eq!(final_state.data.all_locations(), expected_locations);
    }

    #[test]
    fn structurally_equal_join_target_update_preserves_identity_and_revision() {
        let mut coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let location = data_location(5);
        let state = outstanding_target_state(&[(location, load_request(location, 50))]);
        commit(&mut coverage, &[set_join_target_state(id, state.clone())]);
        let revision = coverage.revision();

        let preview = coverage
            .preview_reconcile(&[set_join_target_state(id, state)])
            .unwrap();

        assert_eq!(preview.base_revision(), revision);
        assert_eq!(preview.next_revision(), revision);
        assert!(Arc::ptr_eq(&preview.base, &preview.next));
        assert!(preview.result().topology.groups.is_empty());
        assert!(preview.result().hold_deltas.is_empty());
    }

    #[test]
    fn one_join_target_update_has_bounded_work_with_unrelated_owners() {
        let small = many_pending_owner_fixture(2);
        let large = many_pending_owner_fixture(2_000);
        let target = parent();
        let small_id = small
            .pending_join_id(target, CoverageFeature::Visual)
            .unwrap();
        let large_id = large
            .pending_join_id(target, CoverageFeature::Visual)
            .unwrap();
        let location = data_location(6);
        let state = outstanding_target_state(&[(location, load_request(location, 60))]);

        let a = small
            .preview_reconcile(&[set_join_target_state(small_id, state.clone())])
            .unwrap();
        let b = large
            .preview_reconcile(&[set_join_target_state(large_id, state)])
            .unwrap();

        assert_eq!(a.work_counters(), b.work_counters());
        assert_eq!(a.work_counters().pending_owners_examined, 1);
        let untouched = loc(Vector3i::new(1, 0, 0), 2);
        assert_eq!(
            a.next
                .pending_joins
                .get(&PendingJoinKey::new(untouched, CoverageFeature::Visual)),
            small
                .state
                .pending_joins
                .get(&PendingJoinKey::new(untouched, CoverageFeature::Visual)),
        );
    }

    #[test]
    fn join_target_state_survives_ancestor_manifest_replacement() {
        let (mut coverage, branch, _) =
            arbitrary_pending_branch_with_demanded_sibling(CoverageFeature::Visual);
        let outer_id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let inner_id = coverage
            .pending_join_id(branch, CoverageFeature::Visual)
            .unwrap();
        let location = data_location(7);
        let state = outstanding_target_state(&[(location, load_request(location, 70))]);
        commit(
            &mut coverage,
            &[set_join_target_state(outer_id, state.clone())],
        );
        let old_manifest = coverage
            .pending_join_manifest(parent(), CoverageFeature::Visual)
            .unwrap()
            .clone();

        let terminal = coverage
            .preview_terminal_pending_join_failure(inner_id)
            .unwrap();
        coverage.apply_preview(terminal).unwrap();

        assert_ne!(
            coverage
                .pending_join_manifest(parent(), CoverageFeature::Visual)
                .unwrap(),
            &old_manifest,
        );
        assert_eq!(
            coverage.pending_join_target_state(parent(), CoverageFeature::Visual),
            Some(&state),
        );
    }

    #[test]
    fn canonical_blocker_prefers_first_failure_then_outstanding_then_mesh() {
        let low = data_location(-8);
        let high = data_location(12);
        let low_request = load_request(low, 80);
        let high_request = load_request(high, 81);
        let failure = PendingRequestFailure::Exhausted {
            request: high_request,
            failure: RequestFailureKind::Exhausted,
        };
        let mut state = PendingJoinTargetState {
            data: PendingJoinDataState {
                member_universe: PendingJoinMemberUniverse::CoreValidated(
                    [
                        PendingJoinDataKey::from_location(low),
                        PendingJoinDataKey::from_location(high),
                    ]
                    .into_iter()
                    .collect(),
                ),
                ready: OrdSet::new(),
                outstanding: [(PendingJoinDataKey::from_location(low), low_request)]
                    .into_iter()
                    .collect(),
                failures: [(PendingJoinDataKey::from_location(high), failure)]
                    .into_iter()
                    .collect(),
            },
            mesh: PendingJoinMeshState::Meshing {
                request: mesh_request(parent(), 82),
            },
        };

        assert_eq!(
            state.canonical_blocker(),
            Some(PendingJoinBlocker::DataFailure {
                location: high,
                state: failure,
            }),
        );
        state.data.failures.clear();
        assert_eq!(
            state.canonical_blocker(),
            Some(PendingJoinBlocker::DataLoading {
                location: low,
                request: low_request,
            }),
        );
        state.data.outstanding.clear();
        assert_eq!(
            state.canonical_blocker(),
            Some(PendingJoinBlocker::Mesh(state.mesh)),
        );
        state.mesh = PendingJoinMeshState::Ready;
        assert_eq!(state.canonical_blocker(), None);
    }

    #[test]
    fn meshing_cannot_regress_to_an_awaiting_state() {
        let mut coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let location = data_location(30);
        let key = PendingJoinDataKey::from_location(location);
        let awaiting = PendingJoinTargetState {
            data: PendingJoinDataState {
                member_universe: PendingJoinMemberUniverse::CoreValidated(
                    [key].into_iter().collect(),
                ),
                ready: [key].into_iter().collect(),
                outstanding: OrdMap::new(),
                failures: OrdMap::new(),
            },
            mesh: PendingJoinMeshState::AwaitingRequest,
        };
        commit(&mut coverage, &[set_join_target_state(id, awaiting)]);
        let request = mesh_request(parent(), 90);
        let meshing = PendingJoinTargetState {
            data: PendingJoinDataState {
                member_universe: PendingJoinMemberUniverse::CoreValidated(
                    [key].into_iter().collect(),
                ),
                ready: [key].into_iter().collect(),
                outstanding: OrdMap::new(),
                failures: OrdMap::new(),
            },
            mesh: PendingJoinMeshState::Meshing { request },
        };
        commit(&mut coverage, &[set_join_target_state(id, meshing.clone())]);

        for mesh in [
            PendingJoinMeshState::AwaitingData,
            PendingJoinMeshState::AwaitingRequest,
        ] {
            let mut stale = meshing.clone();
            stale.mesh = mesh;
            assert_eq!(
                coverage
                    .preview_reconcile(&[set_join_target_state(id, stale)])
                    .unwrap_err(),
                CoverageInvariantError::StaleJoinTargetRequest { id, request },
            );
        }
    }

    #[test]
    fn data_failure_is_stable_until_a_fresh_retry_is_published() {
        let mut coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let location = data_location(31);
        let key = PendingJoinDataKey::from_location(location);
        let request = load_request(location, 91);
        commit(
            &mut coverage,
            &[set_join_target_state(
                id,
                outstanding_target_state(&[(location, request)]),
            )],
        );
        let terminal_record = PendingRequestFailure::Retryable {
            request,
            failure: RequestFailureKind::Cancelled,
            attempt: 2,
        };
        let terminal = PendingJoinTargetState {
            data: PendingJoinDataState {
                member_universe: PendingJoinMemberUniverse::CoreValidated(
                    [key].into_iter().collect(),
                ),
                ready: OrdSet::new(),
                outstanding: OrdMap::new(),
                failures: [(key, terminal_record)].into_iter().collect(),
            },
            mesh: PendingJoinMeshState::AwaitingData,
        };
        commit(
            &mut coverage,
            &[set_join_target_state(id, terminal.clone())],
        );

        let mut ready = terminal.clone();
        ready.data.failures.clear();
        ready.data.ready.insert(key);
        assert!(coverage
            .preview_reconcile(&[set_join_target_state(id, ready)])
            .is_err());

        for changed_record in [
            PendingRequestFailure::Retryable {
                request,
                failure: RequestFailureKind::Cancelled,
                attempt: 3,
            },
            PendingRequestFailure::Retryable {
                request,
                failure: RequestFailureKind::Exhausted,
                attempt: 2,
            },
            PendingRequestFailure::Exhausted {
                request,
                failure: RequestFailureKind::Cancelled,
            },
        ] {
            let mut changed = terminal.clone();
            changed.data.failures.insert(key, changed_record);
            assert!(coverage
                .preview_reconcile(&[set_join_target_state(id, changed)])
                .is_err());
        }

        for stale in [
            request,
            load_request(location, 90),
            PhysicalRequestId::Load {
                location,
                tag: TaskRequestTag::new(request.tag().request_epoch + 1, 92),
            },
        ] {
            let retry = outstanding_target_state(&[(location, stale)]);
            assert_eq!(
                coverage
                    .preview_reconcile(&[set_join_target_state(id, retry)])
                    .unwrap_err(),
                CoverageInvariantError::StaleJoinTargetRequest { id, request: stale },
            );
        }

        let fresh = load_request(location, 92);
        commit(
            &mut coverage,
            &[set_join_target_state(
                id,
                outstanding_target_state(&[(location, fresh)]),
            )],
        );
        commit(
            &mut coverage,
            &[set_join_target_state(
                id,
                ready_target_state(&[location], PendingJoinMeshState::AwaitingData),
            )],
        );
        assert_eq!(
            coverage
                .pending_join_target_state(parent(), CoverageFeature::Visual)
                .unwrap()
                .data
                .ready_locations(),
            vec![location],
        );
    }

    #[test]
    fn first_snapshot_requires_the_exact_core_validated_member_universe() {
        let coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let expected = [data_location(40), data_location(41), data_location(42)];
        let requests = [
            load_request(expected[0], 100),
            load_request(expected[1], 101),
            load_request(expected[2], 102),
        ];
        let extra = data_location(43);
        let malformed = [
            core_validated_target_state(
                &expected,
                &[],
                &[(expected[0], requests[0]), (expected[1], requests[1])],
                &[],
                PendingJoinMeshState::AwaitingData,
            ),
            core_validated_target_state(
                &expected,
                &[],
                &[
                    (expected[0], requests[0]),
                    (expected[1], requests[1]),
                    (expected[2], requests[2]),
                    (extra, load_request(extra, 103)),
                ],
                &[],
                PendingJoinMeshState::AwaitingData,
            ),
            core_validated_target_state(
                &expected,
                &expected[..2],
                &[],
                &[],
                PendingJoinMeshState::AwaitingData,
            ),
        ];

        for state in malformed {
            let before = coverage.clone();
            assert!(coverage
                .preview_reconcile(&[set_join_target_state(id, state)])
                .is_err());
            assert_eq!(coverage, before);
        }
    }

    #[test]
    fn data_key_diagnostics_order_by_lod_then_xyz() {
        let locations = [
            BlockLocation {
                position: Vector3i::new(0, 0, -1),
                lod_index: 2,
            },
            BlockLocation {
                position: Vector3i::new(4, -1, 7),
                lod_index: 0,
            },
            BlockLocation {
                position: Vector3i::new(-3, 9, 2),
                lod_index: 1,
            },
            BlockLocation {
                position: Vector3i::new(-3, 8, 9),
                lod_index: 1,
            },
        ];
        let state = ready_target_state(&locations, PendingJoinMeshState::Ready);

        assert_eq!(
            state.data.all_locations(),
            vec![locations[1], locations[3], locations[2], locations[0]],
        );
    }

    #[test]
    fn mesh_snapshot_rejects_wrong_identity_and_data_barrier_bypass() {
        let coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let locations = [data_location(50), data_location(51)];
        let wrong_kind = core_validated_target_state(
            &locations,
            &locations,
            &[],
            &[],
            PendingJoinMeshState::Meshing {
                request: load_request(locations[0], 110),
            },
        );
        let wrong_target_request = mesh_request(second_root(), 111);
        let wrong_target = core_validated_target_state(
            &locations,
            &locations,
            &[],
            &[],
            PendingJoinMeshState::Meshing {
                request: wrong_target_request,
            },
        );
        let outstanding = [
            (locations[0], load_request(locations[0], 112)),
            (locations[1], load_request(locations[1], 113)),
        ];
        let barrier_bypass = core_validated_target_state(
            &locations,
            &[],
            &outstanding,
            &[],
            PendingJoinMeshState::AwaitingRequest,
        );

        for state in [wrong_kind, wrong_target, barrier_bypass] {
            assert!(coverage
                .preview_reconcile(&[set_join_target_state(id, state)])
                .is_err());
        }
    }

    #[test]
    fn mesh_terminal_requires_fresh_same_epoch_meshing_retry() {
        let mut coverage = pending_join_fixture(CoverageFeature::Visual);
        let id = coverage
            .pending_join_id(parent(), CoverageFeature::Visual)
            .unwrap();
        let location = data_location(60);
        commit(
            &mut coverage,
            &[set_join_target_state(
                id,
                ready_target_state(&[location], PendingJoinMeshState::AwaitingRequest),
            )],
        );
        let request = mesh_request(parent(), 120);
        commit(
            &mut coverage,
            &[set_join_target_state(
                id,
                ready_target_state(&[location], PendingJoinMeshState::Meshing { request }),
            )],
        );
        let terminal_mesh = PendingJoinMeshState::RetryableFailure {
            request,
            failure: RequestFailureKind::Cancelled,
            attempt: 1,
        };
        commit(
            &mut coverage,
            &[set_join_target_state(
                id,
                ready_target_state(&[location], terminal_mesh),
            )],
        );

        assert!(coverage
            .preview_reconcile(&[set_join_target_state(
                id,
                ready_target_state(&[location], PendingJoinMeshState::AwaitingRequest),
            )])
            .is_err());
        let reused = request;
        let lower = mesh_request(parent(), 119);
        let other_epoch = PhysicalRequestId::Mesh {
            location: parent(),
            tag: TaskRequestTag::new(request.tag().request_epoch + 1, 121),
        };
        for stale in [reused, lower, other_epoch] {
            assert_eq!(
                coverage
                    .preview_reconcile(&[set_join_target_state(
                        id,
                        ready_target_state(
                            &[location],
                            PendingJoinMeshState::Meshing { request: stale },
                        ),
                    )])
                    .unwrap_err(),
                CoverageInvariantError::StaleJoinTargetRequest { id, request: stale },
            );
        }

        let fresh = mesh_request(parent(), 121);
        coverage
            .preview_reconcile(&[set_join_target_state(
                id,
                ready_target_state(
                    &[location],
                    PendingJoinMeshState::Meshing { request: fresh },
                ),
            )])
            .unwrap();
    }

    #[test]
    fn validate_partition_rejects_corrupt_stored_target_snapshots() {
        let location = data_location(70);
        let other = data_location(71);
        let key = PendingJoinDataKey::from_location(location);
        let request = load_request(location, 130);
        let malformed = [
            PendingJoinTargetState::from_core_validated_parts(
                [key].into_iter().collect(),
                [key].into_iter().collect(),
                [(key, request)].into_iter().collect(),
                OrdMap::new(),
                PendingJoinMeshState::AwaitingData,
            ),
            core_validated_target_state(
                &[location],
                &[],
                &[(location, load_request(other, 131))],
                &[],
                PendingJoinMeshState::AwaitingData,
            ),
            core_validated_target_state(
                &[location],
                &[],
                &[(location, request)],
                &[],
                PendingJoinMeshState::AwaitingRequest,
            ),
            core_validated_target_state(
                &[location, other],
                &[location],
                &[],
                &[],
                PendingJoinMeshState::AwaitingData,
            ),
        ];

        for state in malformed {
            let mut coverage = pending_join_fixture(CoverageFeature::Visual);
            force_target_state_for_test(&mut coverage, parent(), CoverageFeature::Visual, state);
            assert!(coverage
                .validate_partition(CoverageFeature::Visual)
                .is_err());
        }

        let coverage = pending_join_fixture(CoverageFeature::Visual);
        coverage
            .validate_partition(CoverageFeature::Visual)
            .unwrap();
    }
}
