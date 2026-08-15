//! Persistent two-phase ownership ledger for variable-LOD topology holds.
//!
//! A reconcile batch may need resources both before and after its topology
//! mutation. `prepare_phases` resolves only a newly installed owner's target
//! data halo; fallback resources are mesh-only holds. It returns immutable
//! before/after ledgers plus coalesced refcount compare-and-set updates.
//! Existing records always replay their saved target resolution, even if
//! bounds change later.

use super::lod_clipbox::{clipped_meshing_data_box, LodClipboxSettings};
use super::variable_lod_coverage::{
    CoverageFeature, CoverageHoldId, CoverageHoldIntentManifest, CoverageHoldOwnerDelta,
};
use crate::math::{Box3i, Vector3i};
use crate::meshers::MeshBlockLocation;
use crate::storage::BlockLocation;
use imbl::{OrdMap, OrdSet};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoverageMeshResource {
    location: MeshBlockLocation,
    feature: CoverageFeature,
}

impl CoverageMeshResource {
    pub(super) const fn new(location: MeshBlockLocation, feature: CoverageFeature) -> Self {
        Self { location, feature }
    }

    pub(super) const fn location(self) -> MeshBlockLocation {
        self.location
    }

    pub(super) const fn feature(self) -> CoverageFeature {
        self.feature
    }

    const fn sort_key(self) -> (u8, u8, i32, i32, i32) {
        let position = self.location.position_in_blocks;
        (
            feature_rank(self.feature),
            self.location.lod_index,
            position.x,
            position.y,
            position.z,
        )
    }
}

impl PartialOrd for CoverageMeshResource {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CoverageMeshResource {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct CoverageDataResource {
    position_in_blocks: Vector3i,
    lod_index: u8,
}

impl CoverageDataResource {
    pub(super) const fn new(position_in_blocks: Vector3i, lod_index: u8) -> Self {
        Self {
            position_in_blocks,
            lod_index,
        }
    }

    pub(super) const fn position_in_blocks(self) -> Vector3i {
        self.position_in_blocks
    }

    pub(super) const fn lod_index(self) -> u8 {
        self.lod_index
    }

    const fn sort_key(self) -> (u8, i32, i32, i32) {
        (
            self.lod_index,
            self.position_in_blocks.x,
            self.position_in_blocks.y,
            self.position_in_blocks.z,
        )
    }
}

impl PartialOrd for CoverageDataResource {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CoverageDataResource {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct CoverageDataResourceState {
    refcount: u32,
    ready: bool,
}

impl CoverageDataResourceState {
    pub(super) const fn new(refcount: u32, ready: bool) -> Self {
        Self { refcount, ready }
    }

    pub(super) const fn refcount(self) -> u32 {
        self.refcount
    }

    pub(super) const fn ready(self) -> bool {
        self.ready
    }
}

/// On-demand resource state used while binding resolved coverage owners to
/// live terrain refcounts. Implementations must be read-only: preparation can
/// query a resource, but it never acquires or releases it through this view.
pub(super) trait CoverageHoldResourceView {
    fn mesh_refcount(&self, resource: CoverageMeshResource) -> u32;
    fn data_state(&self, resource: CoverageDataResource) -> CoverageDataResourceState;
}

/// Read-only resource state consumed by one ledger preparation.
///
/// Missing resources have refcount zero and are not ready. Callers populate
/// only the resources that can be touched by the supplied owner deltas; the
/// ledger never scans this snapshot globally.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CoverageHoldResourceSnapshot {
    mesh_refcounts: BTreeMap<CoverageMeshResource, u32>,
    data_resources: BTreeMap<CoverageDataResource, CoverageDataResourceState>,
}

impl CoverageHoldResourceSnapshot {
    pub(super) fn mesh_refcount(&self, resource: CoverageMeshResource) -> u32 {
        self.mesh_refcounts.get(&resource).copied().unwrap_or(0)
    }

    pub(super) fn data_refcount(&self, resource: CoverageDataResource) -> u32 {
        self.data_resources
            .get(&resource)
            .map_or(0, |state| state.refcount)
    }

    pub(super) fn data_is_ready(&self, resource: CoverageDataResource) -> bool {
        self.data_resources
            .get(&resource)
            .is_some_and(|state| state.ready)
    }

    pub(super) fn set_mesh_refcount(&mut self, resource: CoverageMeshResource, refcount: u32) {
        if refcount == 0 {
            self.mesh_refcounts.remove(&resource);
        } else {
            self.mesh_refcounts.insert(resource, refcount);
        }
    }

    pub(super) fn set_data_resource(
        &mut self,
        resource: CoverageDataResource,
        refcount: u32,
        ready: bool,
    ) {
        if refcount == 0 && !ready {
            self.data_resources.remove(&resource);
        } else {
            self.data_resources
                .insert(resource, CoverageDataResourceState::new(refcount, ready));
        }
    }
}

impl CoverageHoldResourceView for CoverageHoldResourceSnapshot {
    fn mesh_refcount(&self, resource: CoverageMeshResource) -> u32 {
        CoverageHoldResourceSnapshot::mesh_refcount(self, resource)
    }

    fn data_state(&self, resource: CoverageDataResource) -> CoverageDataResourceState {
        self.data_resources
            .get(&resource)
            .copied()
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CoverageHoldLedgerErrorKind {
    OldManifestMismatch,
    NonCanonicalManifest,
    TargetMismatch,
    InvalidPhaseRelation,
    MeshRefcountOverflow,
    MeshRefcountUnderflow,
    DataRefcountOverflow,
    DataRefcountUnderflow,
    CoordinateOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CoverageHoldLedgerError {
    OldManifestMismatch { id: CoverageHoldId },
    NonCanonicalManifest { id: CoverageHoldId },
    TargetMismatch { id: CoverageHoldId },
    InvalidPhaseRelation { id: Option<CoverageHoldId> },
    MeshRefcountOverflow { resource: CoverageMeshResource },
    MeshRefcountUnderflow { resource: CoverageMeshResource },
    DataRefcountOverflow { resource: CoverageDataResource },
    DataRefcountUnderflow { resource: CoverageDataResource },
    CoordinateOverflow { location: MeshBlockLocation },
}

impl CoverageHoldLedgerError {
    pub(super) const fn kind(&self) -> CoverageHoldLedgerErrorKind {
        match self {
            Self::OldManifestMismatch { .. } => CoverageHoldLedgerErrorKind::OldManifestMismatch,
            Self::NonCanonicalManifest { .. } => CoverageHoldLedgerErrorKind::NonCanonicalManifest,
            Self::TargetMismatch { .. } => CoverageHoldLedgerErrorKind::TargetMismatch,
            Self::InvalidPhaseRelation { .. } => CoverageHoldLedgerErrorKind::InvalidPhaseRelation,
            Self::MeshRefcountOverflow { .. } => CoverageHoldLedgerErrorKind::MeshRefcountOverflow,
            Self::MeshRefcountUnderflow { .. } => {
                CoverageHoldLedgerErrorKind::MeshRefcountUnderflow
            }
            Self::DataRefcountOverflow { .. } => CoverageHoldLedgerErrorKind::DataRefcountOverflow,
            Self::DataRefcountUnderflow { .. } => {
                CoverageHoldLedgerErrorKind::DataRefcountUnderflow
            }
            Self::CoordinateOverflow { .. } => CoverageHoldLedgerErrorKind::CoordinateOverflow,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CoverageHoldKey {
    feature_rank: u8,
    lod: u8,
    x: i32,
    y: i32,
    z: i32,
    generation: u64,
}

impl CoverageHoldKey {
    const fn from_id(id: CoverageHoldId) -> Self {
        let position = id.join_parent.position_in_blocks;
        Self {
            feature_rank: feature_rank(id.feature),
            lod: id.join_parent.lod_index,
            x: position.x,
            y: position.y,
            z: position.z,
            generation: id.transition_generation,
        }
    }

    const fn to_id(self) -> CoverageHoldId {
        CoverageHoldId {
            join_parent: MeshBlockLocation::new(Vector3i::new(self.x, self.y, self.z), self.lod),
            feature: match self.feature_rank {
                0 => CoverageFeature::Visual,
                1 => CoverageFeature::Collision,
                _ => unreachable!(),
            },
            transition_generation: self.generation,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CoverageLocationKey {
    lod: u8,
    x: i32,
    y: i32,
    z: i32,
}

impl CoverageLocationKey {
    const fn from_mesh_location(location: MeshBlockLocation) -> Self {
        Self {
            lod: location.lod_index,
            x: location.position_in_blocks.x,
            y: location.position_in_blocks.y,
            z: location.position_in_blocks.z,
        }
    }
}

const fn feature_rank(feature: CoverageFeature) -> u8 {
    match feature {
        CoverageFeature::Visual => 0,
        CoverageFeature::Collision => 1,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CoverageHoldRecord {
    intent: CoverageHoldIntentManifest,
    // Target first, followed by canonical manifest fallback order.
    mesh_resources: Vec<CoverageMeshResource>,
    // Only `intent.target` owns data. Fallbacks are mesh-only coverage holds.
    // Order is the exact clipped target box's `iter_cells_zxy` order.
    data_halo: Vec<CoverageDataResource>,
}

impl CoverageHoldRecord {
    pub(super) const fn intent(&self) -> &CoverageHoldIntentManifest {
        &self.intent
    }

    pub(super) fn mesh_resources(&self) -> &[CoverageMeshResource] {
        &self.mesh_resources
    }

    pub(super) fn data_halo(&self) -> &[CoverageDataResource] {
        &self.data_halo
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CoverageHoldLedger {
    by_id: OrdMap<CoverageHoldKey, CoverageHoldRecord>,
    data_owners: OrdMap<CoverageDataResource, OrdSet<CoverageHoldKey>>,
    target_owners: OrdMap<CoverageLocationKey, OrdSet<CoverageHoldKey>>,
}

impl CoverageHoldLedger {
    pub(super) fn len(&self) -> usize {
        self.by_id.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub(super) fn record(&self, id: CoverageHoldId) -> Option<&CoverageHoldRecord> {
        self.by_id.get(&CoverageHoldKey::from_id(id))
    }

    pub(super) fn data_owner_ids(
        &self,
        location: BlockLocation,
    ) -> impl Iterator<Item = CoverageHoldId> + '_ {
        let resource = CoverageDataResource::new(location.position, location.lod_index);
        self.data_owners
            .get(&resource)
            .into_iter()
            .flat_map(OrdSet::iter)
            .copied()
            .map(CoverageHoldKey::to_id)
    }

    pub(super) fn target_owner_ids(
        &self,
        location: MeshBlockLocation,
    ) -> impl Iterator<Item = CoverageHoldId> + '_ {
        let location = CoverageLocationKey::from_mesh_location(location);
        self.target_owners
            .get(&location)
            .into_iter()
            .flat_map(OrdSet::iter)
            .copied()
            .map(CoverageHoldKey::to_id)
    }

    #[cfg(test)]
    fn data_owner_ids_with_work<'a>(
        &'a self,
        location: BlockLocation,
        work: &'a mut CoverageHoldReverseIndexWorkCounters,
    ) -> impl Iterator<Item = CoverageHoldId> + 'a {
        work.index_lookups += 1;
        self.data_owner_ids(location).inspect(move |_| {
            work.owner_keys_visited += 1;
        })
    }

    pub(super) fn prepare_phases(
        &self,
        deltas: &[CoverageHoldOwnerDelta],
        settings: LodClipboxSettings,
        bounds_voxels: Box3i,
        resources: &dyn CoverageHoldResourceView,
    ) -> Result<PreparedCoverageHoldPhases, CoverageHoldLedgerError> {
        self.prepare_resolution(deltas, settings, bounds_voxels)?
            .bind(resources)
    }

    pub(super) fn prepare_resolution(
        &self,
        deltas: &[CoverageHoldOwnerDelta],
        settings: LodClipboxSettings,
        bounds_voxels: Box3i,
    ) -> Result<PreparedCoverageHoldResolution, CoverageHoldLedgerError> {
        let mut work = CoverageHoldLedgerWorkCounters::default();
        let normalized = self.validate_deltas(deltas, &mut work)?;

        let mut before_ledger = self.clone();
        let mut before_changes = Vec::with_capacity(normalized.len());
        for owner in &normalized {
            work.owner_records_read += 1;
            let old_record = self.by_id.get(&owner.key).cloned();
            let next_record = match &owner.before_topology {
                Some(intent) => Some(resolve_record(
                    intent,
                    old_record.as_ref(),
                    settings,
                    bounds_voxels,
                    &mut work,
                )?),
                None => None,
            };
            replace_record(&mut before_ledger, owner.key, next_record.clone());
            before_changes.push(OwnerRecordChange {
                old: old_record,
                next: next_record,
            });
        }

        let mut after_ledger = before_ledger.clone();
        let mut after_changes = Vec::with_capacity(normalized.len());
        for owner in &normalized {
            work.owner_records_read += 1;
            let old_record = before_ledger.by_id.get(&owner.key).cloned();
            let next_record = match &owner.after_topology {
                Some(intent) => Some(resolve_record(
                    intent,
                    old_record.as_ref(),
                    settings,
                    bounds_voxels,
                    &mut work,
                )?),
                None => None,
            };
            replace_record(&mut after_ledger, owner.key, next_record.clone());
            after_changes.push(OwnerRecordChange {
                old: old_record,
                next: next_record,
            });
        }

        let before_deltas = collect_resource_deltas(&before_changes, &mut work)?;
        let after_deltas = collect_resource_deltas(&after_changes, &mut work)?;

        Ok(PreparedCoverageHoldResolution {
            before_ledger,
            after_ledger,
            before_deltas,
            after_deltas,
            work_counters: work,
        })
    }

    fn validate_deltas(
        &self,
        deltas: &[CoverageHoldOwnerDelta],
        work: &mut CoverageHoldLedgerWorkCounters,
    ) -> Result<Vec<NormalizedOwnerDelta>, CoverageHoldLedgerError> {
        let mut by_key = BTreeMap::new();
        for delta in deltas {
            work.owners_visited += 1;
            for intent in [
                delta.old.as_ref(),
                delta.before_topology.as_ref(),
                delta.after_topology.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                validate_manifest(intent)?;
                work.manifests_validated += 1;
            }

            let Some(id) = delta
                .old
                .as_ref()
                .or(delta.before_topology.as_ref())
                .or(delta.after_topology.as_ref())
                .map(|intent| intent.id)
            else {
                return Err(CoverageHoldLedgerError::InvalidPhaseRelation { id: None });
            };
            if [
                delta.old.as_ref(),
                delta.before_topology.as_ref(),
                delta.after_topology.as_ref(),
            ]
            .into_iter()
            .flatten()
            .any(|intent| intent.id != id)
            {
                return Err(CoverageHoldLedgerError::InvalidPhaseRelation { id: Some(id) });
            }

            let key = CoverageHoldKey::from_id(id);
            work.owner_records_read += 1;
            let saved = self.by_id.get(&key).map(|record| &record.intent);
            if saved != delta.old.as_ref() {
                return Err(CoverageHoldLedgerError::OldManifestMismatch { id });
            }
            if !manifest_is_subset(delta.old.as_ref(), delta.before_topology.as_ref())
                || !manifest_is_subset(
                    delta.after_topology.as_ref(),
                    delta.before_topology.as_ref(),
                )
            {
                return Err(CoverageHoldLedgerError::InvalidPhaseRelation { id: Some(id) });
            }

            let normalized = NormalizedOwnerDelta {
                key,
                before_topology: delta.before_topology.clone(),
                after_topology: delta.after_topology.clone(),
            };
            if by_key.insert(key, normalized).is_some() {
                return Err(CoverageHoldLedgerError::InvalidPhaseRelation { id: Some(id) });
            }
        }
        Ok(by_key.into_values().collect())
    }
}

#[derive(Debug)]
struct NormalizedOwnerDelta {
    key: CoverageHoldKey,
    before_topology: Option<CoverageHoldIntentManifest>,
    after_topology: Option<CoverageHoldIntentManifest>,
}

#[derive(Debug, Clone)]
struct OwnerRecordChange {
    old: Option<CoverageHoldRecord>,
    next: Option<CoverageHoldRecord>,
}

fn replace_record(
    ledger: &mut CoverageHoldLedger,
    key: CoverageHoldKey,
    record: Option<CoverageHoldRecord>,
) {
    if let Some(old) = ledger.by_id.remove(&key) {
        remove_record_from_reverse_indices(ledger, key, &old);
    }
    if let Some(record) = record {
        add_record_to_reverse_indices(ledger, key, &record);
        ledger.by_id.insert(key, record);
    }
}

fn add_record_to_reverse_indices(
    ledger: &mut CoverageHoldLedger,
    key: CoverageHoldKey,
    record: &CoverageHoldRecord,
) {
    for resource in &record.data_halo {
        add_reverse_owner(&mut ledger.data_owners, *resource, key);
    }
    add_reverse_owner(
        &mut ledger.target_owners,
        CoverageLocationKey::from_mesh_location(record.intent.target),
        key,
    );
}

fn remove_record_from_reverse_indices(
    ledger: &mut CoverageHoldLedger,
    key: CoverageHoldKey,
    record: &CoverageHoldRecord,
) {
    for resource in &record.data_halo {
        remove_reverse_owner(&mut ledger.data_owners, resource, key);
    }
    remove_reverse_owner(
        &mut ledger.target_owners,
        &CoverageLocationKey::from_mesh_location(record.intent.target),
        key,
    );
}

fn add_reverse_owner<K>(
    index: &mut OrdMap<K, OrdSet<CoverageHoldKey>>,
    resource: K,
    key: CoverageHoldKey,
) where
    K: Clone + Ord,
{
    let mut owners = index.get(&resource).cloned().unwrap_or_default();
    owners.insert(key);
    index.insert(resource, owners);
}

fn remove_reverse_owner<K>(
    index: &mut OrdMap<K, OrdSet<CoverageHoldKey>>,
    resource: &K,
    key: CoverageHoldKey,
) where
    K: Clone + Ord,
{
    let Some(mut owners) = index.get(resource).cloned() else {
        return;
    };
    owners.remove(&key);
    if owners.is_empty() {
        index.remove(resource);
    } else {
        index.insert(resource.clone(), owners);
    }
}

fn validate_manifest(intent: &CoverageHoldIntentManifest) -> Result<(), CoverageHoldLedgerError> {
    if intent.target != intent.id.join_parent {
        return Err(CoverageHoldLedgerError::TargetMismatch { id: intent.id });
    }
    let mut previous = None;
    for fallback in &intent.fallbacks {
        if *fallback == intent.target {
            return Err(CoverageHoldLedgerError::NonCanonicalManifest { id: intent.id });
        }
        let key = location_sort_key(*fallback);
        if previous.is_some_and(|previous| previous >= key) {
            return Err(CoverageHoldLedgerError::NonCanonicalManifest { id: intent.id });
        }
        previous = Some(key);
    }
    Ok(())
}

const fn location_sort_key(location: MeshBlockLocation) -> (u8, i32, i32, i32) {
    let position = location.position_in_blocks;
    (location.lod_index, position.x, position.y, position.z)
}

fn manifest_is_subset(
    subset: Option<&CoverageHoldIntentManifest>,
    superset: Option<&CoverageHoldIntentManifest>,
) -> bool {
    match (subset, superset) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(subset), Some(superset)) => {
            subset.id == superset.id
                && subset.target == superset.target
                && subset.fallbacks.iter().all(|fallback| {
                    superset
                        .fallbacks
                        .binary_search_by_key(&location_sort_key(*fallback), |candidate| {
                            location_sort_key(*candidate)
                        })
                        .is_ok()
                })
        }
    }
}

fn resolve_record(
    intent: &CoverageHoldIntentManifest,
    previous: Option<&CoverageHoldRecord>,
    settings: LodClipboxSettings,
    bounds_voxels: Box3i,
    work: &mut CoverageHoldLedgerWorkCounters,
) -> Result<CoverageHoldRecord, CoverageHoldLedgerError> {
    match previous {
        Some(previous) if previous.intent == *intent => return Ok(previous.clone()),
        Some(_) | None => {}
    }

    let mut mesh_resources = Vec::with_capacity(intent.fallbacks.len() + 1);
    mesh_resources.push(CoverageMeshResource::new(intent.target, intent.id.feature));
    mesh_resources.extend(
        intent
            .fallbacks
            .iter()
            .copied()
            .map(|location| CoverageMeshResource::new(location, intent.id.feature)),
    );

    let data_halo = match previous {
        // A key fixes `intent.target == id.join_parent`, so fallback-only
        // manifest changes must replay the exact target resolution saved at
        // first install even if settings or bounds have since changed.
        Some(previous) => previous.data_halo.clone(),
        None => resolve_target_data_halo(intent.target, settings, bounds_voxels, work)?,
    };

    Ok(CoverageHoldRecord {
        intent: intent.clone(),
        mesh_resources,
        data_halo,
    })
}

fn resolve_target_data_halo(
    location: MeshBlockLocation,
    settings: LodClipboxSettings,
    bounds_voxels: Box3i,
    work: &mut CoverageHoldLedgerWorkCounters,
) -> Result<Vec<CoverageDataResource>, CoverageHoldLedgerError> {
    if settings.lod_count == 0
        || location.lod_index >= settings.lod_count
        || settings.mesh_block_size <= 0
    {
        return Err(CoverageHoldLedgerError::CoordinateOverflow { location });
    }
    let clipped = clipped_meshing_data_box(location, settings.mesh_block_size, 1, bounds_voxels)
        .map_err(|_| CoverageHoldLedgerError::CoordinateOverflow { location })?;
    work.mesh_resources_resolved += 1;
    let mut halo = Vec::new();
    for position in clipped.iter_cells_zxy() {
        work.data_cells_resolved += 1;
        halo.push(CoverageDataResource::new(position, location.lod_index));
    }
    Ok(halo)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct CoverageHoldLedgerWorkCounters {
    pub(super) owners_visited: usize,
    pub(super) owner_records_read: usize,
    pub(super) manifests_validated: usize,
    pub(super) mesh_resources_resolved: usize,
    pub(super) data_cells_resolved: usize,
    pub(super) resource_differences_visited: usize,
    pub(super) full_ledger_iterations: usize,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CoverageHoldReverseIndexWorkCounters {
    index_lookups: usize,
    owner_keys_visited: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CoverageMeshRefcountUpdate {
    resource: CoverageMeshResource,
    expected: u32,
    next: u32,
}

impl CoverageMeshRefcountUpdate {
    pub(super) const fn resource(self) -> CoverageMeshResource {
        self.resource
    }

    pub(super) const fn expected(self) -> u32 {
        self.expected
    }

    pub(super) const fn next(self) -> u32 {
        self.next
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CoverageDataRefcountUpdate {
    resource: CoverageDataResource,
    expected: u32,
    next: u32,
    ready: bool,
}

impl CoverageDataRefcountUpdate {
    pub(super) const fn resource(self) -> CoverageDataResource {
        self.resource
    }

    pub(super) const fn expected(self) -> u32 {
        self.expected
    }

    pub(super) const fn next(self) -> u32 {
        self.next
    }

    pub(super) const fn ready(self) -> bool {
        self.ready
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreparedCoverageHoldPhase {
    ledger: CoverageHoldLedger,
    mesh_refcount_updates: Vec<CoverageMeshRefcountUpdate>,
    data_refcount_updates: Vec<CoverageDataRefcountUpdate>,
    all_required_data_ready: bool,
}

impl PreparedCoverageHoldPhase {
    pub(super) const fn ledger(&self) -> &CoverageHoldLedger {
        &self.ledger
    }

    pub(super) fn mesh_refcount_updates(&self) -> &[CoverageMeshRefcountUpdate] {
        &self.mesh_refcount_updates
    }

    pub(super) fn data_refcount_updates(&self) -> &[CoverageDataRefcountUpdate] {
        &self.data_refcount_updates
    }

    /// Whether every saved data-halo resource required by the changed owners
    /// is ready in the input snapshot. This deliberately rechecks unchanged
    /// owners in the delta so a deferred topology attempt cannot mistake a
    /// previously acquired but still-loading halo for ready data.
    pub(super) const fn all_required_data_ready(&self) -> bool {
        self.all_required_data_ready
    }

    /// Consumes this phase so a wider transaction can move its ledger and
    /// update owners without cloning or dropping persistent roots under a
    /// storage publication fence.
    pub(super) fn into_parts(
        self,
    ) -> (
        CoverageHoldLedger,
        Vec<CoverageMeshRefcountUpdate>,
        Vec<CoverageDataRefcountUpdate>,
        bool,
    ) {
        (
            self.ledger,
            self.mesh_refcount_updates,
            self.data_refcount_updates,
            self.all_required_data_ready,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreparedCoverageHoldPhases {
    before_topology: PreparedCoverageHoldPhase,
    after_topology: PreparedCoverageHoldPhase,
    work_counters: CoverageHoldLedgerWorkCounters,
}

/// Structurally resolved coverage-owner transition awaiting a read-only
/// snapshot of exactly the resources exposed by `mesh_resources` and
/// `data_resources`.
///
/// This value is deliberately non-clone: it owns the two ordered phase deltas
/// and is consumed exactly once by `bind`.
#[derive(Debug)]
pub(super) struct PreparedCoverageHoldResolution {
    before_ledger: CoverageHoldLedger,
    after_ledger: CoverageHoldLedger,
    before_deltas: ResourceDeltas,
    after_deltas: ResourceDeltas,
    work_counters: CoverageHoldLedgerWorkCounters,
}

impl PreparedCoverageHoldResolution {
    pub(super) const fn before_topology_ledger(&self) -> &CoverageHoldLedger {
        &self.before_ledger
    }

    pub(super) const fn after_topology_ledger(&self) -> &CoverageHoldLedger {
        &self.after_ledger
    }

    pub(super) fn mesh_resources(&self) -> impl Iterator<Item = CoverageMeshResource> + '_ {
        sorted_union(
            self.before_deltas.mesh.keys().copied(),
            self.after_deltas.mesh.keys().copied(),
        )
    }

    pub(super) fn data_resources(&self) -> impl Iterator<Item = CoverageDataResource> + '_ {
        let before = sorted_union(
            self.before_deltas.data.keys().copied(),
            self.before_deltas.required_data.iter().copied(),
        );
        let after = sorted_union(
            self.after_deltas.data.keys().copied(),
            self.after_deltas.required_data.iter().copied(),
        );
        sorted_union(before, after)
    }

    pub(super) const fn work_counters(&self) -> CoverageHoldLedgerWorkCounters {
        self.work_counters
    }

    pub(super) fn bind(
        self,
        resources: &dyn CoverageHoldResourceView,
    ) -> Result<PreparedCoverageHoldPhases, CoverageHoldLedgerError> {
        let Self {
            before_ledger,
            after_ledger,
            before_deltas,
            after_deltas,
            work_counters,
        } = self;
        let mut shadow = ResourceCountShadow::default();
        let before_topology = prepare_phase(before_ledger, before_deltas, resources, &mut shadow)?;
        let after_topology = prepare_phase(after_ledger, after_deltas, resources, &mut shadow)?;
        Ok(PreparedCoverageHoldPhases {
            before_topology,
            after_topology,
            work_counters,
        })
    }
}

impl PreparedCoverageHoldPhases {
    pub(super) const fn before_topology(&self) -> &PreparedCoverageHoldPhase {
        &self.before_topology
    }

    pub(super) const fn after_topology(&self) -> &PreparedCoverageHoldPhase {
        &self.after_topology
    }

    pub(super) const fn work_counters(&self) -> CoverageHoldLedgerWorkCounters {
        self.work_counters
    }

    /// Consumes the two-phase result while preserving distinct ownership of
    /// the intermediate and final ledgers for publication and retirement.
    pub(super) fn into_parts(
        self,
    ) -> (
        PreparedCoverageHoldPhase,
        PreparedCoverageHoldPhase,
        CoverageHoldLedgerWorkCounters,
    ) {
        (
            self.before_topology,
            self.after_topology,
            self.work_counters,
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct RefcountDelta {
    additions: u64,
    removals: u64,
}

#[derive(Debug, Default)]
struct ResourceDeltas {
    mesh: BTreeMap<CoverageMeshResource, RefcountDelta>,
    data: BTreeMap<CoverageDataResource, RefcountDelta>,
    required_data: BTreeSet<CoverageDataResource>,
}

struct SortedUnion<L, R>
where
    L: Iterator,
    R: Iterator<Item = L::Item>,
{
    left: std::iter::Peekable<L>,
    right: std::iter::Peekable<R>,
}

fn sorted_union<T, L, R>(left: L, right: R) -> SortedUnion<L, R>
where
    T: Copy + Ord,
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    SortedUnion {
        left: left.peekable(),
        right: right.peekable(),
    }
}

impl<T, L, R> Iterator for SortedUnion<L, R>
where
    T: Copy + Ord,
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match (self.left.peek(), self.right.peek()) {
            (Some(left), Some(right)) => match left.cmp(right) {
                Ordering::Less => self.left.next(),
                Ordering::Greater => self.right.next(),
                Ordering::Equal => {
                    self.right.next();
                    self.left.next()
                }
            },
            (Some(_), None) => self.left.next(),
            (None, Some(_)) => self.right.next(),
            (None, None) => None,
        }
    }
}

fn collect_resource_deltas(
    changes: &[OwnerRecordChange],
    work: &mut CoverageHoldLedgerWorkCounters,
) -> Result<ResourceDeltas, CoverageHoldLedgerError> {
    let mut deltas = ResourceDeltas::default();
    for change in changes {
        if let Some(next) = &change.next {
            deltas
                .required_data
                .extend(next.data_halo().iter().copied());
        }
        collect_one_resource_delta(
            change.old.as_ref().map(|record| record.mesh_resources()),
            change.next.as_ref().map(|record| record.mesh_resources()),
            &mut deltas.mesh,
            work,
            |resource| CoverageHoldLedgerError::MeshRefcountOverflow { resource },
        )?;
        collect_one_resource_delta(
            change.old.as_ref().map(|record| record.data_halo()),
            change.next.as_ref().map(|record| record.data_halo()),
            &mut deltas.data,
            work,
            |resource| CoverageHoldLedgerError::DataRefcountOverflow { resource },
        )?;
    }
    Ok(deltas)
}

fn collect_one_resource_delta<R, F>(
    old: Option<&[R]>,
    next: Option<&[R]>,
    output: &mut BTreeMap<R, RefcountDelta>,
    work: &mut CoverageHoldLedgerWorkCounters,
    overflow: F,
) -> Result<(), CoverageHoldLedgerError>
where
    R: Copy + Ord,
    F: Fn(R) -> CoverageHoldLedgerError,
{
    let old = old
        .unwrap_or_default()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let next = next
        .unwrap_or_default()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    for resource in old.difference(&next).copied() {
        work.resource_differences_visited += 1;
        let delta = output.entry(resource).or_default();
        delta.removals = delta
            .removals
            .checked_add(1)
            .ok_or_else(|| overflow(resource))?;
    }
    for resource in next.difference(&old).copied() {
        work.resource_differences_visited += 1;
        let delta = output.entry(resource).or_default();
        delta.additions = delta
            .additions
            .checked_add(1)
            .ok_or_else(|| overflow(resource))?;
    }
    Ok(())
}

#[derive(Default)]
struct ResourceCountShadow {
    mesh: BTreeMap<CoverageMeshResource, u32>,
    data: BTreeMap<CoverageDataResource, CoverageDataResourceState>,
}

fn prepare_phase(
    ledger: CoverageHoldLedger,
    deltas: ResourceDeltas,
    resources: &dyn CoverageHoldResourceView,
    shadow: &mut ResourceCountShadow,
) -> Result<PreparedCoverageHoldPhase, CoverageHoldLedgerError> {
    let ResourceDeltas {
        mesh,
        data,
        required_data,
    } = deltas;
    let mut mesh_refcount_updates = Vec::with_capacity(mesh.len());
    for (resource, delta) in mesh {
        let expected = shadow
            .mesh
            .get(&resource)
            .copied()
            .unwrap_or_else(|| resources.mesh_refcount(resource));
        let next = checked_mesh_next(expected, delta, resource)?;
        shadow.mesh.insert(resource, next);
        if expected != next {
            mesh_refcount_updates.push(CoverageMeshRefcountUpdate {
                resource,
                expected,
                next,
            });
        }
    }

    let mut queried_data = required_data.clone();
    queried_data.extend(data.keys().copied());
    let mut data_refcount_updates = Vec::with_capacity(data.len());
    let mut all_required_data_ready = true;
    for resource in queried_data {
        let expected_state = shadow
            .data
            .get(&resource)
            .copied()
            .unwrap_or_else(|| resources.data_state(resource));
        let delta = data.get(&resource).copied().unwrap_or_default();
        let next = checked_data_next(expected_state.refcount(), delta, resource)?;
        let next_state = CoverageDataResourceState::new(next, expected_state.ready());
        shadow.data.insert(resource, next_state);
        if required_data.contains(&resource) && !expected_state.ready() {
            all_required_data_ready = false;
        }
        if expected_state.refcount() != next {
            data_refcount_updates.push(CoverageDataRefcountUpdate {
                resource,
                expected: expected_state.refcount(),
                next,
                ready: expected_state.ready(),
            });
        }
    }

    Ok(PreparedCoverageHoldPhase {
        ledger,
        mesh_refcount_updates,
        data_refcount_updates,
        all_required_data_ready,
    })
}

fn checked_mesh_next(
    expected: u32,
    delta: RefcountDelta,
    resource: CoverageMeshResource,
) -> Result<u32, CoverageHoldLedgerError> {
    checked_next(expected, delta).map_err(|direction| match direction {
        RefcountFailure::Overflow => CoverageHoldLedgerError::MeshRefcountOverflow { resource },
        RefcountFailure::Underflow => CoverageHoldLedgerError::MeshRefcountUnderflow { resource },
    })
}

fn checked_data_next(
    expected: u32,
    delta: RefcountDelta,
    resource: CoverageDataResource,
) -> Result<u32, CoverageHoldLedgerError> {
    checked_next(expected, delta).map_err(|direction| match direction {
        RefcountFailure::Overflow => CoverageHoldLedgerError::DataRefcountOverflow { resource },
        RefcountFailure::Underflow => CoverageHoldLedgerError::DataRefcountUnderflow { resource },
    })
}

#[derive(Debug, Clone, Copy)]
enum RefcountFailure {
    Overflow,
    Underflow,
}

fn checked_next(expected: u32, delta: RefcountDelta) -> Result<u32, RefcountFailure> {
    if delta.additions >= delta.removals {
        let increase = delta.additions - delta.removals;
        let increase = u32::try_from(increase).map_err(|_| RefcountFailure::Overflow)?;
        expected
            .checked_add(increase)
            .ok_or(RefcountFailure::Overflow)
    } else {
        let decrease = delta.removals - delta.additions;
        let decrease = u32::try_from(decrease).map_err(|_| RefcountFailure::Underflow)?;
        expected
            .checked_sub(decrease)
            .ok_or(RefcountFailure::Underflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Box3i, Vector3i};
    use crate::meshers::MeshBlockLocation;
    use crate::storage::BlockLocation;
    use crate::terrain::lod_clipbox::clipped_meshing_data_box;
    use crate::terrain::{
        CoverageFeature, CoverageHoldId, CoverageHoldIntentManifest, CoverageHoldOwnerDelta,
    };
    use std::cell::RefCell;

    #[derive(Default)]
    struct InstrumentedResourceView {
        mesh: BTreeMap<CoverageMeshResource, u32>,
        data: BTreeMap<CoverageDataResource, CoverageDataResourceState>,
        mesh_queries: RefCell<Vec<CoverageMeshResource>>,
        data_queries: RefCell<Vec<CoverageDataResource>>,
    }

    impl CoverageHoldResourceView for InstrumentedResourceView {
        fn mesh_refcount(&self, resource: CoverageMeshResource) -> u32 {
            self.mesh_queries.borrow_mut().push(resource);
            self.mesh.get(&resource).copied().unwrap_or(0)
        }

        fn data_state(&self, resource: CoverageDataResource) -> CoverageDataResourceState {
            self.data_queries.borrow_mut().push(resource);
            self.data.get(&resource).copied().unwrap_or_default()
        }
    }

    fn settings() -> LodClipboxSettings {
        LodClipboxSettings {
            data_block_size: 16,
            mesh_block_size: 16,
            lod_count: 4,
            lod0_distance_voxels: 96,
            secondary_distance_voxels: 64,
            unload_hysteresis_blocks: 2,
        }
    }

    fn bounds() -> Box3i {
        Box3i::new(Vector3i::splat(-512), Vector3i::splat(1024))
    }

    fn location(x: i32, y: i32, z: i32, lod: u8) -> MeshBlockLocation {
        MeshBlockLocation::new(Vector3i::new(x, y, z), lod)
    }

    fn manifest(
        parent: MeshBlockLocation,
        feature: CoverageFeature,
        generation: u64,
        fallbacks: Vec<MeshBlockLocation>,
    ) -> CoverageHoldIntentManifest {
        CoverageHoldIntentManifest {
            id: CoverageHoldId {
                join_parent: parent,
                feature,
                transition_generation: generation,
            },
            target: parent,
            fallbacks,
        }
    }

    fn delta(
        old: Option<CoverageHoldIntentManifest>,
        before_topology: Option<CoverageHoldIntentManifest>,
        after_topology: Option<CoverageHoldIntentManifest>,
    ) -> CoverageHoldOwnerDelta {
        CoverageHoldOwnerDelta {
            old,
            before_topology,
            after_topology,
        }
    }

    fn acquire(
        ledger: &CoverageHoldLedger,
        intent: CoverageHoldIntentManifest,
        snapshot: &CoverageHoldResourceSnapshot,
    ) -> PreparedCoverageHoldPhases {
        ledger
            .prepare_phases(
                &[delta(None, Some(intent.clone()), Some(intent))],
                settings(),
                bounds(),
                snapshot,
            )
            .unwrap()
    }

    fn apply_phase(snapshot: &mut CoverageHoldResourceSnapshot, phase: &PreparedCoverageHoldPhase) {
        for update in phase.mesh_refcount_updates() {
            assert_eq!(snapshot.mesh_refcount(update.resource()), update.expected());
            snapshot.set_mesh_refcount(update.resource(), update.next());
        }
        for update in phase.data_refcount_updates() {
            assert_eq!(snapshot.data_refcount(update.resource()), update.expected());
            let ready = snapshot.data_is_ready(update.resource());
            snapshot.set_data_resource(update.resource(), update.next(), ready);
        }
    }

    fn snapshot_after(
        mut snapshot: CoverageHoldResourceSnapshot,
        phase: &PreparedCoverageHoldPhase,
    ) -> CoverageHoldResourceSnapshot {
        apply_phase(&mut snapshot, phase);
        snapshot
    }

    #[test]
    fn install_stores_fallback_mesh_holds_but_only_the_target_clipped_zxy_halo() {
        let parent = location(0, 0, 0, 1);
        let fallbacks = vec![location(0, 0, 0, 0), location(1, 0, 0, 0)];
        let intent = manifest(parent, CoverageFeature::Visual, 7, fallbacks.clone());
        let phases = acquire(
            &CoverageHoldLedger::default(),
            intent.clone(),
            &CoverageHoldResourceSnapshot::default(),
        );

        let record = phases.before_topology().ledger().record(intent.id).unwrap();
        assert_eq!(record.intent(), &intent);
        assert_eq!(
            record.mesh_resources(),
            &[
                CoverageMeshResource::new(parent, CoverageFeature::Visual),
                CoverageMeshResource::new(fallbacks[0], CoverageFeature::Visual),
                CoverageMeshResource::new(fallbacks[1], CoverageFeature::Visual),
            ]
        );

        let expected_halo = clipped_meshing_data_box(parent, 16, 1, bounds())
            .unwrap()
            .iter_cells_zxy()
            .map(|position| CoverageDataResource::new(position, parent.lod_index))
            .collect::<Vec<_>>();
        assert_eq!(record.data_halo(), expected_halo);
        assert!(record
            .data_halo()
            .iter()
            .all(|resource| resource.lod_index() == parent.lod_index));
        assert!(fallbacks.into_iter().all(|fallback| {
            clipped_meshing_data_box(fallback, 16, 1, bounds())
                .unwrap()
                .iter_cells_zxy()
                .map(|position| CoverageDataResource::new(position, fallback.lod_index))
                .all(|resource| !record.data_halo().contains(&resource))
        }));
        assert_eq!(phases.before_topology().mesh_refcount_updates().len(), 3);
        assert_eq!(
            phases.before_topology().data_refcount_updates().len(),
            expected_halo.len()
        );
        assert_eq!(phases.work_counters().mesh_resources_resolved, 1);
        assert_eq!(
            phases.work_counters().data_cells_resolved,
            expected_halo.len()
        );
    }

    #[test]
    fn on_demand_view_queries_each_newly_resolved_resource_once() {
        let parent = location(-2, 3, 1, 1);
        let intent = manifest(
            parent,
            CoverageFeature::Visual,
            701,
            vec![location(-4, 6, 2, 0), location(-3, 6, 2, 0)],
        );
        let small = InstrumentedResourceView::default();
        let small_phases = CoverageHoldLedger::default()
            .prepare_phases(
                &[delta(None, Some(intent.clone()), Some(intent.clone()))],
                settings(),
                bounds(),
                &small,
            )
            .unwrap();
        let record = small_phases
            .after_topology()
            .ledger()
            .record(intent.id)
            .unwrap();
        let mut expected_mesh = record.mesh_resources().to_vec();
        expected_mesh.sort();
        let mut expected_data = record.data_halo().to_vec();
        expected_data.sort();

        assert_eq!(*small.mesh_queries.borrow(), expected_mesh);
        assert_eq!(*small.data_queries.borrow(), expected_data);
        assert_eq!(small_phases.work_counters().mesh_resources_resolved, 1);
        assert_eq!(
            small_phases.work_counters().data_cells_resolved,
            expected_data.len()
        );

        let mut large = InstrumentedResourceView::default();
        for index in 0..2_000_i32 {
            large.mesh.insert(
                CoverageMeshResource::new(
                    location(index + 10_000, 8_000, -9_000, 0),
                    CoverageFeature::Collision,
                ),
                u32::try_from(index).unwrap(),
            );
            large.data.insert(
                CoverageDataResource::new(Vector3i::new(index + 20_000, 7_000, -8_000), 0),
                CoverageDataResourceState::new(u32::try_from(index).unwrap(), index % 2 == 0),
            );
        }
        let large_phases = CoverageHoldLedger::default()
            .prepare_phases(
                &[delta(None, Some(intent.clone()), Some(intent))],
                settings(),
                bounds(),
                &large,
            )
            .unwrap();
        assert_eq!(*large.mesh_queries.borrow(), expected_mesh);
        assert_eq!(*large.data_queries.borrow(), expected_data);
        assert_eq!(large_phases.work_counters(), small_phases.work_counters());
    }

    #[test]
    fn on_demand_view_rechecks_saved_halo_without_resolving_or_querying_mesh() {
        let intent = manifest(
            location(1, -1, 2, 1),
            CoverageFeature::Collision,
            702,
            Vec::new(),
        );
        let acquired = acquire(
            &CoverageHoldLedger::default(),
            intent.clone(),
            &CoverageHoldResourceSnapshot::default(),
        );
        let ledger = acquired.after_topology().ledger();
        let record = ledger.record(intent.id).unwrap();
        let mut expected_data = record.data_halo().to_vec();
        expected_data.sort();
        let mut view = InstrumentedResourceView::default();
        for (index, resource) in expected_data.iter().copied().enumerate() {
            view.data
                .insert(resource, CoverageDataResourceState::new(1, index != 0));
        }

        let retried = ledger
            .prepare_phases(
                &[delta(
                    Some(intent.clone()),
                    Some(intent.clone()),
                    Some(intent),
                )],
                LodClipboxSettings {
                    mesh_block_size: 0,
                    ..settings()
                },
                Box3i::new(Vector3i::splat(i32::MAX), Vector3i::splat(1)),
                &view,
            )
            .unwrap();

        assert!(view.mesh_queries.borrow().is_empty());
        assert_eq!(*view.data_queries.borrow(), expected_data);
        assert!(!retried.before_topology().all_required_data_ready());
        assert!(retried.before_topology().data_refcount_updates().is_empty());
        assert_eq!(retried.work_counters().mesh_resources_resolved, 0);
        assert_eq!(retried.work_counters().data_cells_resolved, 0);
    }

    #[test]
    fn immediate_acquire_release_returns_two_exact_phases() {
        let intent = manifest(
            location(2, -1, 3, 1),
            CoverageFeature::Visual,
            4,
            Vec::new(),
        );
        let phases = CoverageHoldLedger::default()
            .prepare_phases(
                &[delta(None, Some(intent.clone()), None)],
                settings(),
                bounds(),
                &CoverageHoldResourceSnapshot::default(),
            )
            .unwrap();

        assert_eq!(phases.before_topology().ledger().len(), 1);
        assert!(phases.after_topology().ledger().is_empty());
        assert!(phases
            .before_topology()
            .mesh_refcount_updates()
            .iter()
            .all(|update| update.expected() == 0 && update.next() == 1));
        assert!(phases
            .after_topology()
            .mesh_refcount_updates()
            .iter()
            .all(|update| update.expected() == 1 && update.next() == 0));
        assert!(phases
            .before_topology()
            .data_refcount_updates()
            .iter()
            .all(|update| update.expected() == 0 && update.next() == 1));
        assert!(phases
            .after_topology()
            .data_refcount_updates()
            .iter()
            .all(|update| update.expected() == 1 && update.next() == 0));
    }

    #[test]
    fn invalid_manifests_and_old_mismatch_leave_inputs_unchanged() {
        let parent = location(0, 0, 0, 1);
        let valid = manifest(
            parent,
            CoverageFeature::Visual,
            1,
            vec![location(0, 0, 0, 0)],
        );
        let initial_snapshot = CoverageHoldResourceSnapshot::default();
        let installed = acquire(
            &CoverageHoldLedger::default(),
            valid.clone(),
            &initial_snapshot,
        );
        let ledger = installed.before_topology().ledger().clone();
        let snapshot = snapshot_after(initial_snapshot, installed.before_topology());

        let mut wrong_target = valid.clone();
        wrong_target.target = location(1, 0, 0, 1);
        let mut unsorted = manifest(
            location(4, 0, 0, 1),
            CoverageFeature::Visual,
            2,
            vec![location(1, 0, 0, 0), location(0, 0, 0, 0)],
        );
        let duplicate = manifest(
            location(5, 0, 0, 1),
            CoverageFeature::Visual,
            3,
            vec![location(0, 0, 0, 0), location(0, 0, 0, 0)],
        );
        let target_in_fallbacks = manifest(
            location(6, 0, 0, 1),
            CoverageFeature::Visual,
            4,
            vec![location(6, 0, 0, 1)],
        );
        let mut wrong_old = valid.clone();
        wrong_old.fallbacks.clear();

        let cases = [
            (
                delta(None, Some(wrong_target), None),
                CoverageHoldLedgerErrorKind::TargetMismatch,
            ),
            (
                delta(None, Some(unsorted.clone()), None),
                CoverageHoldLedgerErrorKind::NonCanonicalManifest,
            ),
            (
                delta(None, Some(duplicate), None),
                CoverageHoldLedgerErrorKind::NonCanonicalManifest,
            ),
            (
                delta(None, Some(target_in_fallbacks), None),
                CoverageHoldLedgerErrorKind::NonCanonicalManifest,
            ),
            (
                delta(Some(wrong_old), Some(valid.clone()), None),
                CoverageHoldLedgerErrorKind::OldManifestMismatch,
            ),
        ];
        // Keep the value live to prove validation does not canonicalize in place.
        unsorted.fallbacks.reverse();
        assert_eq!(unsorted.fallbacks[0], location(0, 0, 0, 0));

        for (owner_delta, expected_kind) in cases {
            let ledger_before = ledger.clone();
            let snapshot_before = snapshot.clone();
            let error = ledger
                .prepare_phases(&[owner_delta], settings(), bounds(), &snapshot)
                .unwrap_err();
            assert_eq!(error.kind(), expected_kind);
            assert_eq!(ledger, ledger_before);
            assert_eq!(snapshot, snapshot_before);
        }
    }

    #[test]
    fn phase_relation_requires_old_and_after_to_be_subsets_of_before() {
        let parent = location(0, 0, 0, 1);
        let child_a = location(0, 0, 0, 0);
        let child_b = location(1, 0, 0, 0);
        let old = manifest(parent, CoverageFeature::Visual, 8, vec![child_a]);
        let installed = acquire(
            &CoverageHoldLedger::default(),
            old.clone(),
            &CoverageHoldResourceSnapshot::default(),
        );
        let ledger = installed.before_topology().ledger();

        let before_missing_old = manifest(parent, CoverageFeature::Visual, 8, Vec::new());
        let error = ledger
            .prepare_phases(
                &[delta(Some(old.clone()), Some(before_missing_old), None)],
                settings(),
                bounds(),
                &snapshot_after(
                    CoverageHoldResourceSnapshot::default(),
                    installed.before_topology(),
                ),
            )
            .unwrap_err();
        assert_eq!(
            error.kind(),
            CoverageHoldLedgerErrorKind::InvalidPhaseRelation
        );

        let after_not_in_before = manifest(parent, CoverageFeature::Visual, 8, vec![child_b]);
        let error = ledger
            .prepare_phases(
                &[delta(
                    Some(old.clone()),
                    Some(old),
                    Some(after_not_in_before),
                )],
                settings(),
                bounds(),
                &snapshot_after(
                    CoverageHoldResourceSnapshot::default(),
                    installed.before_topology(),
                ),
            )
            .unwrap_err();
        assert_eq!(
            error.kind(),
            CoverageHoldLedgerErrorKind::InvalidPhaseRelation
        );
    }

    #[test]
    fn visual_and_collision_holds_are_independent_but_share_data_counts() {
        let parent = location(1, 2, 3, 1);
        let visual = manifest(parent, CoverageFeature::Visual, 9, Vec::new());
        let collision = manifest(parent, CoverageFeature::Collision, 9, Vec::new());
        let phases = CoverageHoldLedger::default()
            .prepare_phases(
                &[
                    delta(None, Some(visual), None),
                    delta(None, Some(collision), None),
                ],
                settings(),
                bounds(),
                &CoverageHoldResourceSnapshot::default(),
            )
            .unwrap();

        assert_eq!(phases.before_topology().ledger().len(), 2);
        assert_eq!(phases.before_topology().mesh_refcount_updates().len(), 2);
        assert!(phases
            .before_topology()
            .mesh_refcount_updates()
            .iter()
            .all(|update| update.expected() == 0 && update.next() == 1));
        assert!(phases
            .before_topology()
            .data_refcount_updates()
            .iter()
            .all(|update| update.expected() == 0 && update.next() == 2));
    }

    #[test]
    fn overlapping_owners_refcount_zero_two_one_zero() {
        let parent = location(-2, 1, 3, 1);
        let first = manifest(parent, CoverageFeature::Visual, 10, Vec::new());
        let second = manifest(parent, CoverageFeature::Visual, 11, Vec::new());
        let mut snapshot = CoverageHoldResourceSnapshot::default();
        let acquired = CoverageHoldLedger::default()
            .prepare_phases(
                &[
                    delta(None, Some(first.clone()), Some(first.clone())),
                    delta(None, Some(second.clone()), Some(second.clone())),
                ],
                settings(),
                bounds(),
                &snapshot,
            )
            .unwrap();
        assert!(acquired
            .before_topology()
            .mesh_refcount_updates()
            .iter()
            .all(|update| update.expected() == 0 && update.next() == 2));
        assert!(acquired
            .before_topology()
            .data_refcount_updates()
            .iter()
            .all(|update| update.expected() == 0 && update.next() == 2));
        apply_phase(&mut snapshot, acquired.before_topology());

        let release_first = acquired
            .after_topology()
            .ledger()
            .prepare_phases(
                &[delta(Some(first.clone()), Some(first), None)],
                settings(),
                bounds(),
                &snapshot,
            )
            .unwrap();
        assert!(release_first
            .after_topology()
            .mesh_refcount_updates()
            .iter()
            .all(|update| update.expected() == 2 && update.next() == 1));
        apply_phase(&mut snapshot, release_first.after_topology());

        let release_second = release_first
            .after_topology()
            .ledger()
            .prepare_phases(
                &[delta(Some(second.clone()), Some(second), None)],
                settings(),
                bounds(),
                &snapshot,
            )
            .unwrap();
        assert!(release_second
            .after_topology()
            .mesh_refcount_updates()
            .iter()
            .all(|update| update.expected() == 1 && update.next() == 0));
        assert!(release_second
            .after_topology()
            .data_refcount_updates()
            .iter()
            .all(|update| update.expected() == 1 && update.next() == 0));
        assert!(release_second.after_topology().ledger().is_empty());
    }

    #[test]
    fn refcount_overflow_and_underflow_are_atomic_for_mesh_and_data() {
        let intent = manifest(
            location(0, 0, 0, 1),
            CoverageFeature::Visual,
            12,
            Vec::new(),
        );
        let empty = CoverageHoldLedger::default();
        let resolved = acquire(
            &empty,
            intent.clone(),
            &CoverageHoldResourceSnapshot::default(),
        );
        let record = resolved
            .before_topology()
            .ledger()
            .record(intent.id)
            .unwrap();
        let mesh = record.mesh_resources()[0];
        let data = record.data_halo()[0];

        let mut mesh_overflow = CoverageHoldResourceSnapshot::default();
        mesh_overflow.set_mesh_refcount(mesh, u32::MAX);
        let error = empty
            .prepare_phases(
                &[delta(None, Some(intent.clone()), None)],
                settings(),
                bounds(),
                &mesh_overflow,
            )
            .unwrap_err();
        assert_eq!(
            error.kind(),
            CoverageHoldLedgerErrorKind::MeshRefcountOverflow
        );
        assert_eq!(mesh_overflow.mesh_refcount(mesh), u32::MAX);
        assert!(empty.is_empty());

        let mut data_overflow = CoverageHoldResourceSnapshot::default();
        data_overflow.set_data_resource(data, u32::MAX, true);
        let error = empty
            .prepare_phases(
                &[delta(None, Some(intent.clone()), None)],
                settings(),
                bounds(),
                &data_overflow,
            )
            .unwrap_err();
        assert_eq!(
            error.kind(),
            CoverageHoldLedgerErrorKind::DataRefcountOverflow
        );
        assert_eq!(data_overflow.data_refcount(data), u32::MAX);
        assert!(empty.is_empty());

        let ledger = resolved.before_topology().ledger();
        let release = delta(Some(intent.clone()), Some(intent.clone()), None);
        let data_underflow = CoverageHoldResourceSnapshot::default();
        let error = ledger
            .prepare_phases(
                std::slice::from_ref(&release),
                settings(),
                bounds(),
                &data_underflow,
            )
            .unwrap_err();
        assert_eq!(
            error.kind(),
            CoverageHoldLedgerErrorKind::MeshRefcountUnderflow
        );
        assert_eq!(ledger.len(), 1);

        let mut mesh_only = CoverageHoldResourceSnapshot::default();
        mesh_only.set_mesh_refcount(mesh, 1);
        let error = ledger
            .prepare_phases(&[release], settings(), bounds(), &mesh_only)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            CoverageHoldLedgerErrorKind::DataRefcountUnderflow
        );
        assert_eq!(mesh_only.mesh_refcount(mesh), 1);
        assert_eq!(ledger.len(), 1);
    }

    #[test]
    fn existing_record_replays_saved_halo_when_bounds_and_settings_change() {
        let intent = manifest(
            location(0, 0, 0, 1),
            CoverageFeature::Visual,
            13,
            Vec::new(),
        );
        let initial_snapshot = CoverageHoldResourceSnapshot::default();
        let acquired = acquire(
            &CoverageHoldLedger::default(),
            intent.clone(),
            &initial_snapshot,
        );
        let ledger = acquired.before_topology().ledger();
        let saved_halo = ledger.record(intent.id).unwrap().data_halo().to_vec();
        let snapshot = snapshot_after(initial_snapshot, acquired.before_topology());
        let unusable_now = LodClipboxSettings {
            mesh_block_size: 0,
            ..settings()
        };

        let released = ledger
            .prepare_phases(
                &[delta(Some(intent.clone()), Some(intent), None)],
                unusable_now,
                Box3i::new(Vector3i::splat(i32::MAX), Vector3i::splat(1)),
                &snapshot,
            )
            .unwrap();
        assert_eq!(
            released
                .after_topology()
                .data_refcount_updates()
                .iter()
                .map(|update| update.resource())
                .collect::<Vec<_>>(),
            {
                let mut resources = saved_halo;
                resources.sort();
                resources
            }
        );
    }

    #[test]
    fn newly_acquired_data_readiness_is_reported_without_mutating_snapshot() {
        let intent = manifest(
            location(0, 0, 0, 1),
            CoverageFeature::Collision,
            14,
            Vec::new(),
        );
        let halo = clipped_meshing_data_box(intent.target, 16, 1, bounds()).unwrap();
        let mut snapshot = CoverageHoldResourceSnapshot::default();
        for position in halo.iter_cells_zxy() {
            snapshot.set_data_resource(
                CoverageDataResource::new(position, intent.target.lod_index),
                0,
                true,
            );
        }
        let missing = CoverageDataResource::new(halo.iter_cells_zxy().next().unwrap(), 1);
        snapshot.set_data_resource(missing, 0, false);
        let before = snapshot.clone();
        let phases = acquire(&CoverageHoldLedger::default(), intent.clone(), &snapshot);
        assert!(!phases.before_topology().all_required_data_ready());
        assert_eq!(snapshot, before);

        snapshot.set_data_resource(missing, 0, true);
        let phases = acquire(&CoverageHoldLedger::default(), intent, &snapshot);
        assert!(phases.before_topology().all_required_data_ready());
        assert!(phases
            .before_topology()
            .data_refcount_updates()
            .iter()
            .all(|update| update.ready()));
    }

    #[test]
    fn unchanged_owner_rechecks_saved_halo_readiness_without_resolving_it() {
        let intent = manifest(
            location(0, 0, 0, 1),
            CoverageFeature::Visual,
            140,
            Vec::new(),
        );
        let initial_snapshot = CoverageHoldResourceSnapshot::default();
        let acquired = acquire(
            &CoverageHoldLedger::default(),
            intent.clone(),
            &initial_snapshot,
        );
        let ledger = acquired.after_topology().ledger();
        let snapshot = snapshot_after(initial_snapshot, acquired.before_topology());

        let retried = ledger
            .prepare_phases(
                &[delta(
                    Some(intent.clone()),
                    Some(intent.clone()),
                    Some(intent),
                )],
                LodClipboxSettings {
                    mesh_block_size: 0,
                    ..settings()
                },
                Box3i::new(Vector3i::splat(i32::MAX), Vector3i::splat(1)),
                &snapshot,
            )
            .unwrap();
        assert!(retried.before_topology().data_refcount_updates().is_empty());
        assert!(!retried.before_topology().all_required_data_ready());
        assert_eq!(retried.work_counters().mesh_resources_resolved, 0);
        assert_eq!(retried.work_counters().data_cells_resolved, 0);
    }

    #[test]
    fn work_is_bounded_by_changed_owners_not_unrelated_ledger_size() {
        let relevant = manifest(
            location(0, 0, 0, 1),
            CoverageFeature::Visual,
            15,
            Vec::new(),
        );
        let initial = acquire(
            &CoverageHoldLedger::default(),
            relevant.clone(),
            &CoverageHoldResourceSnapshot::default(),
        );
        let small = initial.before_topology().ledger().clone();
        let mut large = small.clone();
        for index in 0..2_000_i32 {
            let unrelated = manifest(
                location(index * 8 + 100, 100, 100, 0),
                CoverageFeature::Visual,
                u64::try_from(index).unwrap() + 100,
                Vec::new(),
            );
            large = acquire(&large, unrelated, &CoverageHoldResourceSnapshot::default())
                .after_topology()
                .ledger()
                .clone();
        }
        assert_eq!(large.len(), 2_001);

        let no_change = delta(
            Some(relevant.clone()),
            Some(relevant.clone()),
            Some(relevant),
        );
        let small_prepared = small
            .prepare_phases(
                std::slice::from_ref(&no_change),
                settings(),
                bounds(),
                &CoverageHoldResourceSnapshot::default(),
            )
            .unwrap();
        let large_prepared = large
            .prepare_phases(
                &[no_change],
                settings(),
                bounds(),
                &CoverageHoldResourceSnapshot::default(),
            )
            .unwrap();
        assert_eq!(
            small_prepared.work_counters(),
            large_prepared.work_counters()
        );
        assert_eq!(small_prepared.work_counters().owners_visited, 1);
        assert_eq!(small_prepared.work_counters().full_ledger_iterations, 0);
    }

    #[test]
    fn two_touched_owners_have_bounded_work_and_preserve_two_thousand_untouched_owners() {
        let first = manifest(
            location(0, 0, 0, 1),
            CoverageFeature::Visual,
            20_001,
            Vec::new(),
        );
        let second = manifest(
            location(2, 0, 0, 1),
            CoverageFeature::Collision,
            20_002,
            Vec::new(),
        );
        let first_acquire = acquire(
            &CoverageHoldLedger::default(),
            first.clone(),
            &CoverageHoldResourceSnapshot::default(),
        );
        let second_acquire = acquire(
            first_acquire.after_topology().ledger(),
            second.clone(),
            &CoverageHoldResourceSnapshot::default(),
        );
        let small = second_acquire.after_topology().ledger().clone();
        let mut large = small.clone();
        let mut untouched = Vec::with_capacity(2_000);
        for index in 0..2_000_i32 {
            let intent = manifest(
                location(index * 8 + 10_000, 200, -200, 0),
                CoverageFeature::Visual,
                u64::try_from(index).unwrap() + 30_000,
                Vec::new(),
            );
            large = acquire(
                &large,
                intent.clone(),
                &CoverageHoldResourceSnapshot::default(),
            )
            .after_topology()
            .ledger()
            .clone();
            untouched.push(intent);
        }
        assert_eq!(large.len(), 2_002);

        let touched = [
            delta(Some(first.clone()), Some(first.clone()), Some(first)),
            delta(Some(second.clone()), Some(second.clone()), Some(second)),
        ];
        let small_prepared = small
            .prepare_phases(
                &touched,
                settings(),
                bounds(),
                &CoverageHoldResourceSnapshot::default(),
            )
            .unwrap();
        let large_prepared = large
            .prepare_phases(
                &touched,
                settings(),
                bounds(),
                &CoverageHoldResourceSnapshot::default(),
            )
            .unwrap();

        assert_eq!(
            small_prepared.work_counters(),
            large_prepared.work_counters()
        );
        assert_eq!(small_prepared.work_counters().owners_visited, 2);
        assert_eq!(small_prepared.work_counters().owner_records_read, 6);
        assert_eq!(small_prepared.work_counters().full_ledger_iterations, 0);
        assert_eq!(large_prepared.after_topology().ledger().len(), 2_002);
        for intent in untouched {
            assert_eq!(
                large_prepared
                    .after_topology()
                    .ledger()
                    .record(intent.id)
                    .map(CoverageHoldRecord::intent),
                Some(&intent),
            );
        }
    }

    #[test]
    fn coordinate_overflow_is_reported_before_any_result_is_published() {
        let intent = manifest(
            location(i32::MAX, 0, 0, 1),
            CoverageFeature::Visual,
            16,
            Vec::new(),
        );
        let ledger = CoverageHoldLedger::default();
        let snapshot = CoverageHoldResourceSnapshot::default();
        let error = ledger
            .prepare_phases(
                &[delta(None, Some(intent), None)],
                settings(),
                Box3i::new(Vector3i::splat(i32::MAX), Vector3i::splat(1)),
                &snapshot,
            )
            .unwrap_err();
        assert_eq!(
            error.kind(),
            CoverageHoldLedgerErrorKind::CoordinateOverflow
        );
        assert!(ledger.is_empty());
        assert_eq!(snapshot, CoverageHoldResourceSnapshot::default());
    }

    #[test]
    fn reverse_indices_share_data_owners_but_index_only_exact_targets() {
        let parent = location(3, -2, 1, 1);
        let fallback = location(6, -4, 2, 0);
        let visual = manifest(parent, CoverageFeature::Visual, 21_001, vec![fallback]);
        let collision = manifest(parent, CoverageFeature::Collision, 21_002, Vec::new());
        let prepared = CoverageHoldLedger::default()
            .prepare_phases(
                &[
                    delta(None, Some(visual.clone()), Some(visual.clone())),
                    delta(None, Some(collision.clone()), Some(collision.clone())),
                ],
                settings(),
                bounds(),
                &CoverageHoldResourceSnapshot::default(),
            )
            .unwrap();
        let ledger = prepared.after_topology().ledger();
        let data = ledger.record(visual.id).unwrap().data_halo()[0];

        assert_eq!(
            ledger
                .data_owner_ids(BlockLocation {
                    position: data.position_in_blocks(),
                    lod_index: data.lod_index(),
                })
                .collect::<Vec<_>>(),
            vec![visual.id, collision.id],
        );
        assert_eq!(
            ledger.target_owner_ids(parent).collect::<Vec<_>>(),
            vec![visual.id, collision.id],
        );
        assert!(ledger.target_owner_ids(fallback).next().is_none());
    }

    #[test]
    fn reverse_indices_replace_without_duplicates_and_release_empty_entries() {
        let parent = location(-3, 1, 4, 1);
        let first_fallback = location(-6, 2, 8, 0);
        let second_fallback = location(-5, 2, 8, 0);
        let old = manifest(
            parent,
            CoverageFeature::Visual,
            21_010,
            vec![first_fallback],
        );
        let installed = acquire(
            &CoverageHoldLedger::default(),
            old.clone(),
            &CoverageHoldResourceSnapshot::default(),
        );
        let ledger = installed.after_topology().ledger();
        let data = ledger.record(old.id).unwrap().data_halo()[0];
        let expanded = manifest(
            parent,
            CoverageFeature::Visual,
            21_010,
            vec![first_fallback, second_fallback],
        );
        let replaced = ledger
            .prepare_phases(
                &[delta(
                    Some(old.clone()),
                    Some(expanded.clone()),
                    Some(expanded.clone()),
                )],
                settings(),
                bounds(),
                &snapshot_after(
                    CoverageHoldResourceSnapshot::default(),
                    installed.before_topology(),
                ),
            )
            .unwrap();
        let replaced_ledger = replaced.after_topology().ledger();
        let data_location = BlockLocation {
            position: data.position_in_blocks(),
            lod_index: data.lod_index(),
        };
        assert_eq!(
            replaced_ledger
                .data_owner_ids(data_location)
                .collect::<Vec<_>>(),
            vec![expanded.id],
        );
        assert_eq!(
            replaced_ledger.target_owner_ids(parent).collect::<Vec<_>>(),
            vec![expanded.id],
        );

        let released = replaced_ledger
            .prepare_resolution(
                &[delta(Some(expanded.clone()), Some(expanded), None)],
                settings(),
                bounds(),
            )
            .unwrap();
        assert!(released.after_topology_ledger().is_empty());
        assert!(released
            .after_topology_ledger()
            .data_owner_ids(data_location)
            .next()
            .is_none());
        assert!(released
            .after_topology_ledger()
            .target_owner_ids(parent)
            .next()
            .is_none());
    }

    #[test]
    fn cloned_before_and_after_ledgers_keep_reverse_indices_consistent() {
        let parent = location(8, 0, -2, 1);
        let intent = manifest(parent, CoverageFeature::Collision, 21_020, Vec::new());
        let resolution = CoverageHoldLedger::default()
            .prepare_resolution(
                &[delta(None, Some(intent.clone()), None)],
                settings(),
                bounds(),
            )
            .unwrap();
        let before = resolution.before_topology_ledger().clone();
        let after = resolution.after_topology_ledger().clone();
        let data = before.record(intent.id).unwrap().data_halo()[0];
        let data_location = BlockLocation {
            position: data.position_in_blocks(),
            lod_index: data.lod_index(),
        };

        assert_eq!(
            before.data_owner_ids(data_location).collect::<Vec<_>>(),
            vec![intent.id]
        );
        assert_eq!(
            before.target_owner_ids(parent).collect::<Vec<_>>(),
            vec![intent.id]
        );
        assert!(after.data_owner_ids(data_location).next().is_none());
        assert!(after.target_owner_ids(parent).next().is_none());
    }

    #[test]
    fn reverse_owner_lookup_work_is_independent_of_unrelated_owner_count() {
        let parent = location(0, 0, 0, 1);
        let visual = manifest(parent, CoverageFeature::Visual, 22_001, Vec::new());
        let collision = manifest(parent, CoverageFeature::Collision, 22_002, Vec::new());
        let prepared = CoverageHoldLedger::default()
            .prepare_resolution(
                &[
                    delta(None, Some(visual.clone()), Some(visual.clone())),
                    delta(None, Some(collision.clone()), Some(collision.clone())),
                ],
                settings(),
                bounds(),
            )
            .unwrap();
        let small = prepared.after_topology_ledger().clone();
        let mut large = small.clone();
        for index in 0..2_000_i32 {
            let unrelated = manifest(
                location(index + 10_000, 100, -100, 0),
                CoverageFeature::Visual,
                u64::try_from(index).unwrap() + 30_000,
                Vec::new(),
            );
            let mut work = CoverageHoldLedgerWorkCounters::default();
            let record = resolve_record(&unrelated, None, settings(), bounds(), &mut work).unwrap();
            replace_record(
                &mut large,
                CoverageHoldKey::from_id(unrelated.id),
                Some(record),
            );
        }
        let data = small.record(visual.id).unwrap().data_halo()[0];
        let data_location = BlockLocation {
            position: data.position_in_blocks(),
            lod_index: data.lod_index(),
        };
        let mut small_work = CoverageHoldReverseIndexWorkCounters::default();
        let small_ids = small
            .data_owner_ids_with_work(data_location, &mut small_work)
            .collect::<Vec<_>>();
        let mut large_work = CoverageHoldReverseIndexWorkCounters::default();
        let large_ids = large
            .data_owner_ids_with_work(data_location, &mut large_work)
            .collect::<Vec<_>>();

        assert_eq!(small_ids, vec![visual.id, collision.id]);
        assert_eq!(large_ids, small_ids);
        assert_eq!(small_work, large_work);
        assert_eq!(
            large_work,
            CoverageHoldReverseIndexWorkCounters {
                index_lookups: 1,
                owner_keys_visited: 2,
            }
        );
    }

    #[test]
    fn split_resolution_exposes_exact_union_and_binding_matches_wrapper() {
        let parent = location(1, 2, -1, 1);
        let fallback_a = location(2, 4, -2, 0);
        let fallback_b = location(3, 4, -2, 0);
        let intent = manifest(
            parent,
            CoverageFeature::Visual,
            23_001,
            vec![fallback_a, fallback_b],
        );
        let owner_delta = delta(None, Some(intent.clone()), Some(intent));
        let resolution = CoverageHoldLedger::default()
            .prepare_resolution(std::slice::from_ref(&owner_delta), settings(), bounds())
            .unwrap();
        let record = resolution
            .after_topology_ledger()
            .record(owner_delta.after_topology.as_ref().unwrap().id)
            .unwrap();
        let mut expected_mesh = record.mesh_resources().to_vec();
        expected_mesh.sort();
        let mut expected_data = record.data_halo().to_vec();
        expected_data.sort();
        assert_eq!(
            resolution.mesh_resources().collect::<Vec<_>>(),
            expected_mesh
        );
        assert_eq!(
            resolution.data_resources().collect::<Vec<_>>(),
            expected_data
        );

        let mut snapshot = CoverageHoldResourceSnapshot::default();
        for resource in resolution.data_resources() {
            snapshot.set_data_resource(resource, 0, true);
        }
        let wrapper = CoverageHoldLedger::default()
            .prepare_phases(
                std::slice::from_ref(&owner_delta),
                settings(),
                bounds(),
                &snapshot,
            )
            .unwrap();
        let split = resolution.bind(&snapshot).unwrap();
        assert_eq!(split, wrapper);
        assert!(split.before_topology().all_required_data_ready());
    }

    #[test]
    fn split_binding_preserves_errors_and_resolution_preserves_relation_errors() {
        let parent = location(0, 0, 0, 1);
        let intent = manifest(parent, CoverageFeature::Visual, 23_010, Vec::new());
        let owner_delta = delta(None, Some(intent.clone()), None);
        let resolution = CoverageHoldLedger::default()
            .prepare_resolution(std::slice::from_ref(&owner_delta), settings(), bounds())
            .unwrap();
        let mesh = resolution.mesh_resources().next().unwrap();
        let mut overflow = CoverageHoldResourceSnapshot::default();
        overflow.set_mesh_refcount(mesh, u32::MAX);
        let wrapper_error = CoverageHoldLedger::default()
            .prepare_phases(
                std::slice::from_ref(&owner_delta),
                settings(),
                bounds(),
                &overflow,
            )
            .unwrap_err();
        let split_error = resolution.bind(&overflow).unwrap_err();
        assert_eq!(split_error, wrapper_error);

        let wrong_old = delta(Some(intent.clone()), Some(intent), None);
        let split_relation = CoverageHoldLedger::default()
            .prepare_resolution(std::slice::from_ref(&wrong_old), settings(), bounds())
            .unwrap_err();
        let wrapper_relation = CoverageHoldLedger::default()
            .prepare_phases(
                &[wrong_old],
                settings(),
                bounds(),
                &CoverageHoldResourceSnapshot::default(),
            )
            .unwrap_err();
        assert_eq!(split_relation, wrapper_relation);
        assert_eq!(
            split_relation.kind(),
            CoverageHoldLedgerErrorKind::OldManifestMismatch
        );
    }

    #[test]
    fn prepared_phases_into_parts_moves_ledgers_updates_and_readiness() {
        let parent = location(2, -1, 3, 1);
        let intent = manifest(parent, CoverageFeature::Visual, 24_001, Vec::new());
        let phases = CoverageHoldLedger::default()
            .prepare_phases(
                &[delta(None, Some(intent.clone()), Some(intent.clone()))],
                settings(),
                bounds(),
                &CoverageHoldResourceSnapshot::default(),
            )
            .unwrap();
        let expected_work = phases.work_counters();

        let (before, after, work) = phases.into_parts();
        let (before_ledger, before_mesh, before_data, before_ready) = before.into_parts();
        let (after_ledger, after_mesh, after_data, after_ready) = after.into_parts();

        assert_eq!(work, expected_work);
        assert_eq!(before_ledger.record(intent.id).unwrap().intent(), &intent);
        assert_eq!(after_ledger.record(intent.id).unwrap().intent(), &intent);
        assert_eq!(before_mesh.len(), 1);
        assert!(!before_data.is_empty());
        assert!(!before_ready);
        assert!(after_mesh.is_empty());
        assert!(after_data.is_empty());
        assert!(!after_ready);
    }
}
