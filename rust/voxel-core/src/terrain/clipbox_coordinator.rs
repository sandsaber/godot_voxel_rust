//! Transactional multi-viewer demand coordination for variable-LOD clipboxes.

use super::lod_clipbox::{compute_lod_clipboxes, LodClipboxSettings, LodClipboxes, LodMathError};
use super::voxel_terrain_core::ViewerId;
use crate::math::{Box3i, Vector3i};
use crate::meshers::MeshBlockLocation;
use imbl::{OrdMap, OrdSet};
use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeshDemand {
    pub visuals: bool,
    pub collisions: bool,
}

impl MeshDemand {
    pub const fn any(self) -> bool {
        self.visuals || self.collisions
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoverageFeature {
    Visual,
    Collision,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DemandCounts {
    pub resident: u32,
    pub visuals: u32,
    pub collisions: u32,
    pub visual_splits: u32,
    pub collision_splits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipboxViewerUpdate {
    pub id: ViewerId,
    pub position_voxels: Vector3i,
    pub view_distance_voxels: Vector3i,
    pub demand: MeshDemand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentBlockKind {
    Data,
    Mesh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResidentBlockKey {
    pub kind: ResidentBlockKind,
    pub location: MeshBlockLocation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentDemandChange {
    pub key: ResidentBlockKey,
    pub old_counts: DemandCounts,
    pub new_counts: DemandCounts,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ResidentDemandDelta {
    pub revision: u64,
    pub changes: Vec<ResidentDemandChange>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemandCountField {
    Resident,
    Visuals,
    Collisions,
    VisualSplits,
    CollisionSplits,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorError {
    LodMath(LodMathError),
    DuplicateViewerId(ViewerId),
    InvalidSplitLod,
    CoordinateOverflow,
    RefcountOverflow {
        key: ResidentBlockKey,
        field: DemandCountField,
    },
    RefcountUnderflow {
        key: ResidentBlockKey,
        field: DemandCountField,
    },
    RevisionOverflow,
    StalePreparedIdentity,
}

impl From<LodMathError> for CoordinatorError {
    fn from(value: LodMathError) -> Self {
        Self::LodMath(value)
    }
}

type PositionKey = (i32, i32, i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DemandKey {
    kind: u8,
    lod: u8,
    x: i32,
    y: i32,
    z: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ViewerClipboxState {
    update: ClipboxViewerUpdate,
    clipboxes: LodClipboxes,
    data_resident: Vec<OrdSet<PositionKey>>,
    mesh_resident: Vec<OrdSet<PositionKey>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CoordinatorState {
    revision: u64,
    viewers: OrdMap<ViewerId, Arc<ViewerClipboxState>>,
    aggregate: OrdMap<DemandKey, DemandCounts>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CoordinatorWorkCounters {
    /// Logical release-path work only. Debug-only invariant proof scans are
    /// deliberately excluded because they are not part of shipped hot paths.
    shell_candidates: usize,
    resident_mutations: usize,
    aggregate_mutations: usize,
    changed_aggregate_keys: usize,
    full_resident_iterations: usize,
}

/// Identity-bound viewer reconciliation prepared against one exact immutable
/// coordinator snapshot. The value is intentionally non-`Clone`: applying it
/// transfers its owned next snapshot and delta exactly once.
#[derive(Debug)]
pub(super) struct PreparedCoordinatorUpdate {
    base: Arc<CoordinatorState>,
    next: Arc<CoordinatorState>,
    delta: ResidentDemandDelta,
    #[cfg(test)]
    work: CoordinatorWorkCounters,
}

impl PreparedCoordinatorUpdate {
    pub(super) fn delta(&self) -> &ResidentDemandDelta {
        &self.delta
    }

    fn validate_base_identity(
        &self,
        coordinator: &ClipboxCoordinator,
    ) -> Result<(), CoordinatorError> {
        if !Arc::ptr_eq(&coordinator.state, &self.base) {
            return Err(CoordinatorError::StalePreparedIdentity);
        }
        Ok(())
    }

    /// Consumes this update after proving that it still targets the exact live
    /// coordinator snapshot. Callers can perform this initial fallible check
    /// before a wider storage transaction. A retained token must pass
    /// [`ValidatedCoordinatorUpdate::revalidate_for`] immediately before the
    /// transaction enters its publication fence.
    pub(super) fn validate_for(
        self,
        coordinator: &ClipboxCoordinator,
    ) -> Result<ValidatedCoordinatorUpdate, CoordinatorError> {
        self.validate_base_identity(coordinator)?;
        Ok(ValidatedCoordinatorUpdate(self))
    }

    #[cfg(test)]
    fn work_counters_for_test(&self) -> CoordinatorWorkCounters {
        self.work
    }
}

const _: for<'a> fn(&'a PreparedCoordinatorUpdate) -> &'a ResidentDemandDelta =
    PreparedCoordinatorUpdate::delta;

/// Identity-validated coordinator update ready for an infallible publication.
/// The token is non-clone and consumes every prepared owner exactly once.
#[derive(Debug)]
pub(super) struct ValidatedCoordinatorUpdate(PreparedCoordinatorUpdate);

impl ValidatedCoordinatorUpdate {
    pub(super) fn delta(&self) -> &ResidentDemandDelta {
        self.0.delta()
    }

    /// Rechecks exact base identity without consuming the validated token.
    /// This is the release-safe final gate immediately before a wider storage
    /// transaction enters its publication fence.
    pub(super) fn revalidate_for(
        &self,
        coordinator: &ClipboxCoordinator,
    ) -> Result<(), CoordinatorError> {
        self.0.validate_base_identity(coordinator)
    }

    /// Publishes the prepared next state and returns every displaced/base owner
    /// in an opaque retirement bag. The caller must retain that bag until all
    /// enclosing publication fences have been released. The caller must have
    /// successfully called [`Self::revalidate_for`] as its immediately
    /// preceding fallible identity gate and must not mutate the coordinator
    /// before this infallible publication step.
    pub(super) fn publish(
        self,
        coordinator: &mut ClipboxCoordinator,
    ) -> PublishedCoordinatorUpdate {
        let PreparedCoordinatorUpdate {
            base,
            next,
            delta,
            #[cfg(test)]
                work: _,
        } = self.0;
        debug_assert!(Arc::ptr_eq(&coordinator.state, &base));
        let live = std::mem::replace(&mut coordinator.state, next);
        PublishedCoordinatorUpdate {
            delta,
            retirement: CoordinatorStateRetirement {
                _live: live,
                _prepared_base: base,
            },
        }
    }
}

const _: for<'a> fn(&'a ValidatedCoordinatorUpdate) -> &'a ResidentDemandDelta =
    ValidatedCoordinatorUpdate::delta;
const _: fn(&ValidatedCoordinatorUpdate, &ClipboxCoordinator) -> Result<(), CoordinatorError> =
    ValidatedCoordinatorUpdate::revalidate_for;

#[derive(Debug)]
pub(super) struct PublishedCoordinatorUpdate {
    delta: ResidentDemandDelta,
    retirement: CoordinatorStateRetirement,
}

impl PublishedCoordinatorUpdate {
    pub(super) fn into_parts(self) -> (ResidentDemandDelta, CoordinatorStateRetirement) {
        (self.delta, self.retirement)
    }
}

/// Opaque ownership that must outlive any enclosing publication fence.
#[derive(Debug)]
pub(super) struct CoordinatorStateRetirement {
    _live: Arc<CoordinatorState>,
    _prepared_base: Arc<CoordinatorState>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClipboxCoordinator {
    settings: LodClipboxSettings,
    bounds_voxels: Box3i,
    state: Arc<CoordinatorState>,
}

impl Eq for ClipboxCoordinator {}

impl ClipboxCoordinator {
    pub fn new(
        settings: LodClipboxSettings,
        bounds_voxels: Box3i,
    ) -> Result<Self, CoordinatorError> {
        compute_lod_clipboxes(Vector3i::zero(), Vector3i::zero(), bounds_voxels, settings)?;
        Ok(Self {
            settings,
            bounds_voxels,
            state: Arc::new(CoordinatorState {
                revision: 0,
                viewers: OrdMap::new(),
                aggregate: OrdMap::new(),
            }),
        })
    }

    pub fn revision(&self) -> u64 {
        self.state.revision
    }

    pub fn data_demand(&self, location: MeshBlockLocation) -> DemandCounts {
        self.demand(ResidentBlockKind::Data, location)
    }

    pub fn mesh_demand(&self, location: MeshBlockLocation) -> DemandCounts {
        self.demand(ResidentBlockKind::Mesh, location)
    }

    pub fn split_demand(&self, location: MeshBlockLocation, feature: CoverageFeature) -> u32 {
        let counts = self.mesh_demand(location);
        match feature {
            CoverageFeature::Visual => counts.visual_splits,
            CoverageFeature::Collision => counts.collision_splits,
        }
    }

    pub fn update_viewers(
        &mut self,
        viewers: &[ClipboxViewerUpdate],
    ) -> Result<ResidentDemandDelta, CoordinatorError> {
        let prepared = self.prepare_update(viewers)?;
        self.apply_prepared(prepared)
    }

    pub(super) fn prepare_update(
        &self,
        viewers: &[ClipboxViewerUpdate],
    ) -> Result<PreparedCoordinatorUpdate, CoordinatorError> {
        // Persistent state makes cloning the base O(1). Reconciliation costs
        // O(V^2 + S log N + D log D) work and O(S log N) allocation, where S
        // is the enumerated retain/load/feature shell and D is the number of
        // aggregate keys whose contributions may change. It is independent of
        // the total resident volume; the V^2 term is the deliberately
        // allocation-free duplicate/equality path for the small viewer set.
        validate_unique_viewer_ids(viewers)?;

        // This path is deliberately allocation-free: Arc clones and an empty
        // Vec only adjust counters/store dangling capacity. It is also
        // replayable because base and next have the same identity.
        if self.is_exact_viewer_snapshot(viewers) {
            return Ok(PreparedCoordinatorUpdate {
                base: self.state.clone(),
                next: self.state.clone(),
                delta: ResidentDemandDelta {
                    revision: self.state.revision,
                    changes: Vec::new(),
                },
                #[cfg(test)]
                work: CoordinatorWorkCounters::default(),
            });
        }

        let mut accumulator = TransitionAccumulator::default();
        let mut next_viewers = self.state.viewers.clone();

        for (viewer_id, previous) in self.state.viewers.iter() {
            if viewers.iter().all(|viewer| viewer.id != *viewer_id) {
                reconcile_viewer_transition(
                    Some(previous.as_ref()),
                    None,
                    self.settings,
                    self.bounds_voxels,
                    &mut accumulator,
                )?;
                next_viewers.remove(viewer_id);
            }
        }

        for viewer in viewers {
            let previous = self.state.viewers.get(&viewer.id).map(Arc::as_ref);
            if previous.is_some_and(|state| state.update == *viewer) {
                continue;
            }
            let next = reconcile_viewer_transition(
                previous,
                Some(viewer),
                self.settings,
                self.bounds_voxels,
                &mut accumulator,
            )?
            .expect("a supplied viewer update always produces viewer state");
            next_viewers.insert(viewer.id, Arc::new(next));
        }

        let (next_aggregate, changes) = accumulator.apply_to_aggregate(&self.state.aggregate)?;

        let next_revision = if changes.is_empty() {
            self.state.revision
        } else {
            self.state
                .revision
                .checked_add(1)
                .ok_or(CoordinatorError::RevisionOverflow)?
        };

        Ok(PreparedCoordinatorUpdate {
            base: self.state.clone(),
            next: Arc::new(CoordinatorState {
                revision: next_revision,
                viewers: next_viewers,
                aggregate: next_aggregate,
            }),
            delta: ResidentDemandDelta {
                revision: next_revision,
                changes,
            },
            #[cfg(test)]
            work: accumulator.work,
        })
    }

    pub(super) fn apply_prepared(
        &mut self,
        prepared: PreparedCoordinatorUpdate,
    ) -> Result<ResidentDemandDelta, CoordinatorError> {
        let validated = prepared.validate_for(self)?;
        let (delta, retirement) = validated.publish(self).into_parts();
        drop(retirement);
        Ok(delta)
    }

    fn is_exact_viewer_snapshot(&self, viewers: &[ClipboxViewerUpdate]) -> bool {
        viewers.len() == self.state.viewers.len()
            && viewers.iter().all(|viewer| {
                self.state
                    .viewers
                    .get(&viewer.id)
                    .is_some_and(|state| state.update == *viewer)
            })
    }

    #[cfg(test)]
    pub(super) fn state_identity_for_test(&self) -> usize {
        Arc::as_ptr(&self.state) as usize
    }

    fn demand(&self, kind: ResidentBlockKind, location: MeshBlockLocation) -> DemandCounts {
        self.state
            .aggregate
            .get(&demand_key(ResidentBlockKey { kind, location }))
            .copied()
            .unwrap_or_default()
    }
}

fn validate_unique_viewer_ids(viewers: &[ClipboxViewerUpdate]) -> Result<(), CoordinatorError> {
    for (index, viewer) in viewers.iter().enumerate() {
        if viewers[..index]
            .iter()
            .any(|previous| previous.id == viewer.id)
        {
            return Err(CoordinatorError::DuplicateViewerId(viewer.id));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default)]
struct DemandCountTransition {
    exits: DemandCounts,
    enters: DemandCounts,
}

#[derive(Default)]
struct TransitionAccumulator {
    counts: BTreeMap<DemandKey, DemandCountTransition>,
    #[cfg(test)]
    work: CoordinatorWorkCounters,
}

impl TransitionAccumulator {
    fn record(
        &mut self,
        key: DemandKey,
        field: DemandCountField,
        entering: bool,
    ) -> Result<(), CoordinatorError> {
        let transition = self.counts.entry(key).or_default();
        let counts = if entering {
            &mut transition.enters
        } else {
            &mut transition.exits
        };
        let value = count_mut(counts, field);
        *value = value.checked_add(1).ok_or_else(|| {
            let key = resident_key(key);
            CoordinatorError::RefcountOverflow { key, field }
        })?;
        Ok(())
    }

    fn apply_to_aggregate(
        &mut self,
        aggregate: &OrdMap<DemandKey, DemandCounts>,
    ) -> Result<(OrdMap<DemandKey, DemandCounts>, Vec<ResidentDemandChange>), CoordinatorError>
    {
        let mut next = aggregate.clone();
        let mut changes = Vec::with_capacity(self.counts.len());
        for (key, transition) in &self.counts {
            let old_counts = aggregate.get(key).copied().unwrap_or_default();
            let resident_key = resident_key(*key);
            let mut new_counts = old_counts;
            for field in ALL_DEMAND_COUNT_FIELDS {
                *count_mut(&mut new_counts, field) = checked_net_count(
                    count(old_counts, field),
                    count(transition.exits, field),
                    count(transition.enters, field),
                    resident_key,
                    field,
                )?;
            }
            if old_counts == new_counts {
                continue;
            }
            if new_counts == DemandCounts::default() {
                next.remove(key);
            } else {
                next.insert(*key, new_counts);
            }
            changes.push(ResidentDemandChange {
                key: resident_key,
                old_counts,
                new_counts,
            });
            #[cfg(test)]
            {
                self.work.aggregate_mutations += 1;
            }
        }
        #[cfg(test)]
        {
            self.work.changed_aggregate_keys = changes.len();
        }
        Ok((next, changes))
    }

    fn record_shell_candidate(&mut self) {
        #[cfg(test)]
        {
            self.work.shell_candidates += 1;
        }
    }

    fn record_resident_mutation(&mut self) {
        #[cfg(test)]
        {
            self.work.resident_mutations += 1;
        }
    }
}

const ALL_DEMAND_COUNT_FIELDS: [DemandCountField; 5] = [
    DemandCountField::Resident,
    DemandCountField::Visuals,
    DemandCountField::Collisions,
    DemandCountField::VisualSplits,
    DemandCountField::CollisionSplits,
];

fn reconcile_viewer_transition(
    previous: Option<&ViewerClipboxState>,
    update: Option<&ClipboxViewerUpdate>,
    settings: LodClipboxSettings,
    bounds_voxels: Box3i,
    accumulator: &mut TransitionAccumulator,
) -> Result<Option<ViewerClipboxState>, CoordinatorError> {
    let lod_count = usize::from(settings.lod_count);
    let mut next_clipboxes = if let Some(update) = update {
        compute_lod_clipboxes(
            update.position_voxels,
            update.view_distance_voxels,
            bounds_voxels,
            settings,
        )?
    } else {
        empty_clipboxes(lod_count)
    };
    if update.is_none_or(|update| !update.demand.any()) {
        next_clipboxes.mesh_load.fill(Box3i::default());
        next_clipboxes.mesh_retain.fill(Box3i::default());
    }

    let mut data_resident = Vec::with_capacity(lod_count);
    let mut mesh_resident = Vec::with_capacity(lod_count);
    for lod in 0..lod_count {
        let lod_index = u8::try_from(lod).map_err(|_| CoordinatorError::CoordinateOverflow)?;
        data_resident.push(reconcile_resident_set(
            previous.and_then(|state| state.data_resident.get(lod)),
            previous
                .and_then(|state| state.clipboxes.data_load.get(lod))
                .copied()
                .unwrap_or_default(),
            previous
                .and_then(|state| state.clipboxes.data_retain.get(lod))
                .copied()
                .unwrap_or_default(),
            next_clipboxes.data_load[lod],
            next_clipboxes.data_retain[lod],
            ResidentBlockKind::Data,
            lod_index,
            accumulator,
        )?);
        mesh_resident.push(reconcile_resident_set(
            previous.and_then(|state| state.mesh_resident.get(lod)),
            previous
                .and_then(|state| state.clipboxes.mesh_load.get(lod))
                .copied()
                .unwrap_or_default(),
            previous
                .and_then(|state| state.clipboxes.mesh_retain.get(lod))
                .copied()
                .unwrap_or_default(),
            next_clipboxes.mesh_load[lod],
            next_clipboxes.mesh_retain[lod],
            ResidentBlockKind::Mesh,
            lod_index,
            accumulator,
        )?);

        record_feature_box_transition(
            feature_box(previous, lod, CoverageFeature::Visual, false),
            feature_box_from_update(update, &next_clipboxes, lod, CoverageFeature::Visual, false),
            lod_index,
            DemandCountField::Visuals,
            accumulator,
        )?;
        record_feature_box_transition(
            feature_box(previous, lod, CoverageFeature::Collision, false),
            feature_box_from_update(
                update,
                &next_clipboxes,
                lod,
                CoverageFeature::Collision,
                false,
            ),
            lod_index,
            DemandCountField::Collisions,
            accumulator,
        )?;
    }

    for parent_lod in 1..lod_count {
        let lod_index =
            u8::try_from(parent_lod).map_err(|_| CoordinatorError::CoordinateOverflow)?;
        record_feature_box_transition(
            feature_box(previous, parent_lod, CoverageFeature::Visual, true),
            feature_box_from_update(
                update,
                &next_clipboxes,
                parent_lod,
                CoverageFeature::Visual,
                true,
            ),
            lod_index,
            DemandCountField::VisualSplits,
            accumulator,
        )?;
        record_feature_box_transition(
            feature_box(previous, parent_lod, CoverageFeature::Collision, true),
            feature_box_from_update(
                update,
                &next_clipboxes,
                parent_lod,
                CoverageFeature::Collision,
                true,
            ),
            lod_index,
            DemandCountField::CollisionSplits,
            accumulator,
        )?;
    }

    Ok(update.map(|update| ViewerClipboxState {
        update: update.clone(),
        clipboxes: next_clipboxes,
        data_resident,
        mesh_resident,
    }))
}

#[allow(clippy::too_many_arguments)]
fn reconcile_resident_set(
    previous: Option<&OrdSet<PositionKey>>,
    old_load: Box3i,
    old_retain: Box3i,
    new_load: Box3i,
    new_retain: Box3i,
    kind: ResidentBlockKind,
    lod_index: u8,
    accumulator: &mut TransitionAccumulator,
) -> Result<OrdSet<PositionKey>, CoordinatorError> {
    // With L_old subset R_old subset T_old and L_new subset T_new, removing
    // candidates in T_old \ T_new and inserting candidates in L_new \ L_old
    // yields exactly (R_old intersect T_new) union L_new. Membership checks
    // keep overlap idempotent without traversing R_old.
    debug_assert!(old_load.is_empty() || old_retain.contains_box(old_load));
    debug_assert!(new_load.is_empty() || new_retain.contains_box(new_load));
    let mut next = previous.cloned().unwrap_or_default();
    for_each_box_difference_zxy(old_retain, new_retain, |position| {
        accumulator.record_shell_candidate();
        let position_key = position_key(position);
        if next.remove(&position_key).is_some() {
            accumulator.record_resident_mutation();
            accumulator.record(
                demand_key_at(kind, lod_index, position),
                DemandCountField::Resident,
                false,
            )?;
        }
        Ok(())
    })?;
    for_each_box_difference_zxy(new_load, old_load, |position| {
        accumulator.record_shell_candidate();
        if next.insert(position_key(position)).is_none() {
            accumulator.record_resident_mutation();
            accumulator.record(
                demand_key_at(kind, lod_index, position),
                DemandCountField::Resident,
                true,
            )?;
        }
        Ok(())
    })?;
    debug_assert_resident_reconcile(previous, old_load, old_retain, &next, new_load, new_retain);
    Ok(next)
}

#[cfg(debug_assertions)]
fn debug_assert_resident_reconcile(
    previous: Option<&OrdSet<PositionKey>>,
    old_load: Box3i,
    old_retain: Box3i,
    next: &OrdSet<PositionKey>,
    new_load: Box3i,
    new_retain: Box3i,
) {
    let previous_contains =
        |position: Vector3i| previous.is_some_and(|set| set.contains(&position_key(position)));
    debug_assert!(old_load.iter_cells_zxy().all(previous_contains));
    debug_assert!(previous.is_none_or(|set| set.iter().all(|position| {
        old_retain.contains_point(Vector3i::new(position.0, position.1, position.2))
    })));
    debug_assert!(new_load
        .iter_cells_zxy()
        .all(|position| next.contains(&position_key(position))));
    debug_assert!(next.iter().all(|position| {
        new_retain.contains_point(Vector3i::new(position.0, position.1, position.2))
    }));
    debug_assert!(previous.is_none_or(|set| set.iter().all(|position| {
        let position = Vector3i::new(position.0, position.1, position.2);
        next.contains(&position_key(position)) == new_retain.contains_point(position)
    })));
    debug_assert!(next.iter().all(|position| {
        let position = Vector3i::new(position.0, position.1, position.2);
        previous_contains(position) || new_load.contains_point(position)
    }));
}

#[cfg(not(debug_assertions))]
fn debug_assert_resident_reconcile(
    _previous: Option<&OrdSet<PositionKey>>,
    _old_load: Box3i,
    _old_retain: Box3i,
    _next: &OrdSet<PositionKey>,
    _new_load: Box3i,
    _new_retain: Box3i,
) {
}

fn record_feature_box_transition(
    old_box: Box3i,
    new_box: Box3i,
    lod_index: u8,
    field: DemandCountField,
    accumulator: &mut TransitionAccumulator,
) -> Result<(), CoordinatorError> {
    for_each_box_difference_zxy(old_box, new_box, |position| {
        accumulator.record_shell_candidate();
        accumulator.record(
            demand_key_at(ResidentBlockKind::Mesh, lod_index, position),
            field,
            false,
        )
    })?;
    for_each_box_difference_zxy(new_box, old_box, |position| {
        accumulator.record_shell_candidate();
        accumulator.record(
            demand_key_at(ResidentBlockKind::Mesh, lod_index, position),
            field,
            true,
        )
    })
}

fn feature_box(
    state: Option<&ViewerClipboxState>,
    lod: usize,
    feature: CoverageFeature,
    split: bool,
) -> Box3i {
    let Some(state) = state else {
        return Box3i::default();
    };
    let enabled = match feature {
        CoverageFeature::Visual => state.update.demand.visuals,
        CoverageFeature::Collision => state.update.demand.collisions,
    };
    if !enabled {
        return Box3i::default();
    }
    if split {
        split_box(&state.clipboxes, lod)
    } else {
        state.clipboxes.mesh_load[lod]
    }
}

fn feature_box_from_update(
    update: Option<&ClipboxViewerUpdate>,
    clipboxes: &LodClipboxes,
    lod: usize,
    feature: CoverageFeature,
    split: bool,
) -> Box3i {
    let Some(update) = update else {
        return Box3i::default();
    };
    let enabled = match feature {
        CoverageFeature::Visual => update.demand.visuals,
        CoverageFeature::Collision => update.demand.collisions,
    };
    if !enabled {
        return Box3i::default();
    }
    if split {
        split_box(clipboxes, lod)
    } else {
        clipboxes.mesh_load[lod]
    }
}

fn split_box(clipboxes: &LodClipboxes, parent_lod: usize) -> Box3i {
    debug_assert!(parent_lod > 0);
    clipboxes.mesh_load[parent_lod].clipped(clipboxes.mesh_load[parent_lod - 1].downscaled_inner(2))
}

fn empty_clipboxes(lod_count: usize) -> LodClipboxes {
    LodClipboxes {
        mesh_load: vec![Box3i::default(); lod_count],
        mesh_retain: vec![Box3i::default(); lod_count],
        data_load: vec![Box3i::default(); lod_count],
        data_retain: vec![Box3i::default(); lod_count],
    }
}

fn for_each_box_difference_zxy(
    a: Box3i,
    b: Box3i,
    mut visit: impl FnMut(Vector3i) -> Result<(), CoordinatorError>,
) -> Result<(), CoordinatorError> {
    if a.is_empty() {
        return Ok(());
    }
    let intersection = a.clipped(b);
    if intersection.is_empty() {
        for position in a.iter_cells_zxy() {
            visit(position)?;
        }
        return Ok(());
    }

    let a_min = a.position;
    let a_max = a.position + a.size;
    let i_min = intersection.position;
    let i_max = intersection.position + intersection.size;
    let slabs = [
        Box3i::from_min_max(a_min, Vector3i::new(i_min.x, a_max.y, a_max.z)),
        Box3i::from_min_max(Vector3i::new(i_max.x, a_min.y, a_min.z), a_max),
        Box3i::from_min_max(
            Vector3i::new(i_min.x, a_min.y, a_min.z),
            Vector3i::new(i_max.x, i_min.y, a_max.z),
        ),
        Box3i::from_min_max(
            Vector3i::new(i_min.x, i_max.y, a_min.z),
            Vector3i::new(i_max.x, a_max.y, a_max.z),
        ),
        Box3i::from_min_max(
            Vector3i::new(i_min.x, i_min.y, a_min.z),
            Vector3i::new(i_max.x, i_max.y, i_min.z),
        ),
        Box3i::from_min_max(
            Vector3i::new(i_min.x, i_min.y, i_max.z),
            Vector3i::new(i_max.x, i_max.y, a_max.z),
        ),
    ];
    for slab in slabs {
        if slab.is_empty() {
            continue;
        }
        for position in slab.iter_cells_zxy() {
            visit(position)?;
        }
    }
    Ok(())
}

fn position_key(position: Vector3i) -> PositionKey {
    (position.x, position.y, position.z)
}

fn demand_key_at(kind: ResidentBlockKind, lod: u8, position: Vector3i) -> DemandKey {
    DemandKey {
        kind: match kind {
            ResidentBlockKind::Data => 0,
            ResidentBlockKind::Mesh => 1,
        },
        lod,
        x: position.x,
        y: position.y,
        z: position.z,
    }
}

fn demand_key(key: ResidentBlockKey) -> DemandKey {
    demand_key_at(
        key.kind,
        key.location.lod_index,
        key.location.position_in_blocks,
    )
}

fn resident_key(key: DemandKey) -> ResidentBlockKey {
    ResidentBlockKey {
        kind: if key.kind == 0 {
            ResidentBlockKind::Data
        } else {
            ResidentBlockKind::Mesh
        },
        location: MeshBlockLocation::new(Vector3i::new(key.x, key.y, key.z), key.lod),
    }
}

#[cfg(test)]
fn checked_increment(
    aggregate: &mut HashMap<ResidentBlockKey, DemandCounts>,
    key: ResidentBlockKey,
    field: DemandCountField,
) -> Result<(), CoordinatorError> {
    let counts = aggregate.entry(key).or_default();
    let value = count_mut(counts, field);
    *value = value
        .checked_add(1)
        .ok_or(CoordinatorError::RefcountOverflow { key, field })?;
    Ok(())
}

fn count_mut(counts: &mut DemandCounts, field: DemandCountField) -> &mut u32 {
    match field {
        DemandCountField::Resident => &mut counts.resident,
        DemandCountField::Visuals => &mut counts.visuals,
        DemandCountField::Collisions => &mut counts.collisions,
        DemandCountField::VisualSplits => &mut counts.visual_splits,
        DemandCountField::CollisionSplits => &mut counts.collision_splits,
    }
}

fn count(counts: DemandCounts, field: DemandCountField) -> u32 {
    match field {
        DemandCountField::Resident => counts.resident,
        DemandCountField::Visuals => counts.visuals,
        DemandCountField::Collisions => counts.collisions,
        DemandCountField::VisualSplits => counts.visual_splits,
        DemandCountField::CollisionSplits => counts.collision_splits,
    }
}

#[cfg(test)]
fn checked_children(parent: MeshBlockLocation) -> Result<[MeshBlockLocation; 8], CoordinatorError> {
    let child_lod = parent
        .lod_index
        .checked_sub(1)
        .ok_or(CoordinatorError::InvalidSplitLod)?;
    let parent_position = parent.position_in_blocks;
    let mut children = [MeshBlockLocation::new(Vector3i::zero(), child_lod); 8];
    for (index, child) in children.iter_mut().enumerate() {
        let x = i64::from(parent_position.x)
            .checked_mul(2)
            .and_then(|value| value.checked_add(i64::from((index & 1) as u8)))
            .ok_or(CoordinatorError::CoordinateOverflow)?;
        let y = i64::from(parent_position.y)
            .checked_mul(2)
            .and_then(|value| value.checked_add(i64::from(((index >> 1) & 1) as u8)))
            .ok_or(CoordinatorError::CoordinateOverflow)?;
        let z = i64::from(parent_position.z)
            .checked_mul(2)
            .and_then(|value| value.checked_add(i64::from(((index >> 2) & 1) as u8)))
            .ok_or(CoordinatorError::CoordinateOverflow)?;
        *child = MeshBlockLocation::new(
            Vector3i::new(
                i32::try_from(x).map_err(|_| CoordinatorError::CoordinateOverflow)?,
                i32::try_from(y).map_err(|_| CoordinatorError::CoordinateOverflow)?,
                i32::try_from(z).map_err(|_| CoordinatorError::CoordinateOverflow)?,
            ),
            child_lod,
        );
    }
    Ok(children)
}

fn checked_net_count(
    old: u32,
    exits: u32,
    enters: u32,
    key: ResidentBlockKey,
    field: DemandCountField,
) -> Result<u32, CoordinatorError> {
    old.checked_sub(exits)
        .ok_or(CoordinatorError::RefcountUnderflow { key, field })?
        .checked_add(enters)
        .ok_or(CoordinatorError::RefcountOverflow { key, field })
}

#[cfg(test)]
mod tests {
    use super::{
        checked_children, checked_increment, checked_net_count, compute_lod_clipboxes, demand_key,
        resident_key, split_box, validate_unique_viewer_ids, ClipboxCoordinator,
        ClipboxViewerUpdate, CoordinatorError, CoordinatorWorkCounters, CoverageFeature,
        DemandCountField, DemandCounts, LodClipboxes, MeshDemand, ResidentBlockKey,
        ResidentBlockKind, ResidentDemandChange, ResidentDemandDelta, ViewerId,
    };
    use crate::math::{Box3i, Vector3i};
    use crate::meshers::MeshBlockLocation;
    use crate::terrain::lod_clipbox::{LodClipboxSettings, LodMathError};
    use imbl::OrdSet;
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::sync::Arc;

    fn settings() -> LodClipboxSettings {
        LodClipboxSettings {
            data_block_size: 16,
            mesh_block_size: 16,
            lod_count: 3,
            lod0_distance_voxels: 16,
            secondary_distance_voxels: 16,
            unload_hysteresis_blocks: 2,
        }
    }

    fn bounds() -> Box3i {
        Box3i::new(Vector3i::splat(-512), Vector3i::splat(1024))
    }

    fn test_coordinator() -> ClipboxCoordinator {
        ClipboxCoordinator::new(settings(), bounds()).unwrap()
    }

    fn viewer(
        id: u32,
        position_voxels: Vector3i,
        visuals: bool,
        collisions: bool,
    ) -> ClipboxViewerUpdate {
        ClipboxViewerUpdate {
            id,
            position_voxels,
            view_distance_voxels: Vector3i::splat(64),
            demand: MeshDemand {
                visuals,
                collisions,
            },
        }
    }

    fn loc(position: Vector3i, lod_index: u8) -> MeshBlockLocation {
        MeshBlockLocation::new(position, lod_index)
    }

    #[derive(Clone)]
    struct MaterializedViewerState {
        update: ClipboxViewerUpdate,
        clipboxes: LodClipboxes,
        data_resident: Vec<HashSet<Vector3i>>,
        mesh_resident: Vec<HashSet<Vector3i>>,
    }

    #[derive(Clone, Copy, Default)]
    struct MaterializedWork {
        shell_candidates: usize,
        resident_mutations: usize,
        aggregate_mutations: usize,
        changed_aggregate_keys: usize,
    }

    struct MaterializedCoordinator {
        settings: LodClipboxSettings,
        bounds: Box3i,
        revision: u64,
        viewers: BTreeMap<ViewerId, MaterializedViewerState>,
        aggregate: HashMap<ResidentBlockKey, DemandCounts>,
        last_work: MaterializedWork,
    }

    impl MaterializedCoordinator {
        fn new(settings: LodClipboxSettings, bounds: Box3i) -> Self {
            Self {
                settings,
                bounds,
                revision: 0,
                viewers: BTreeMap::new(),
                aggregate: HashMap::new(),
                last_work: MaterializedWork::default(),
            }
        }

        fn update(
            &mut self,
            viewers: &[ClipboxViewerUpdate],
        ) -> Result<ResidentDemandDelta, CoordinatorError> {
            validate_unique_viewer_ids(viewers)?;
            let mut next_viewers = BTreeMap::new();
            for viewer in viewers {
                let mut clipboxes = compute_lod_clipboxes(
                    viewer.position_voxels,
                    viewer.view_distance_voxels,
                    self.bounds,
                    self.settings,
                )?;
                let previous = self.viewers.get(&viewer.id);
                let data_resident = materialized_reconcile_resident_sets(
                    previous.map(|state| state.data_resident.as_slice()),
                    &clipboxes.data_load,
                    &clipboxes.data_retain,
                );
                let mesh_resident = if viewer.demand.any() {
                    materialized_reconcile_resident_sets(
                        previous.map(|state| state.mesh_resident.as_slice()),
                        &clipboxes.mesh_load,
                        &clipboxes.mesh_retain,
                    )
                } else {
                    (0..usize::from(self.settings.lod_count))
                        .map(|_| HashSet::new())
                        .collect()
                };
                if !viewer.demand.any() {
                    clipboxes.mesh_load.fill(Box3i::default());
                    clipboxes.mesh_retain.fill(Box3i::default());
                }
                next_viewers.insert(
                    viewer.id,
                    MaterializedViewerState {
                        update: viewer.clone(),
                        clipboxes,
                        data_resident,
                        mesh_resident,
                    },
                );
            }

            let next_aggregate = materialized_build_aggregate(&next_viewers)?;
            let mut keys: HashSet<_> = self.aggregate.keys().copied().collect();
            keys.extend(next_aggregate.keys().copied());
            let mut changes = keys
                .into_iter()
                .filter_map(|key| {
                    let old_counts = self.aggregate.get(&key).copied().unwrap_or_default();
                    let new_counts = next_aggregate.get(&key).copied().unwrap_or_default();
                    (old_counts != new_counts).then_some(ResidentDemandChange {
                        key,
                        old_counts,
                        new_counts,
                    })
                })
                .collect::<Vec<_>>();
            changes.sort_by_key(|change| demand_key(change.key));
            if !changes.is_empty() {
                self.revision = self
                    .revision
                    .checked_add(1)
                    .ok_or(CoordinatorError::RevisionOverflow)?;
            }
            self.last_work = materialized_work(&self.viewers, &next_viewers, changes.len());
            self.viewers = next_viewers;
            self.aggregate = next_aggregate;
            Ok(ResidentDemandDelta {
                revision: self.revision,
                changes,
            })
        }
    }

    fn materialized_work(
        old_viewers: &BTreeMap<ViewerId, MaterializedViewerState>,
        new_viewers: &BTreeMap<ViewerId, MaterializedViewerState>,
        changed_aggregate_keys: usize,
    ) -> MaterializedWork {
        let mut work = MaterializedWork {
            aggregate_mutations: changed_aggregate_keys,
            changed_aggregate_keys,
            ..MaterializedWork::default()
        };
        let ids = old_viewers
            .keys()
            .chain(new_viewers.keys())
            .copied()
            .collect::<HashSet<_>>();
        for id in ids {
            let old = old_viewers.get(&id);
            let new = new_viewers.get(&id);
            let lod_count = old
                .map_or(0, |state| state.data_resident.len())
                .max(new.map_or(0, |state| state.data_resident.len()));
            for lod in 0..lod_count {
                for (old_sets, new_sets, old_load, old_retain, new_load, new_retain) in [
                    (
                        old.map(|state| &state.data_resident),
                        new.map(|state| &state.data_resident),
                        old.and_then(|state| state.clipboxes.data_load.get(lod))
                            .copied(),
                        old.and_then(|state| state.clipboxes.data_retain.get(lod))
                            .copied(),
                        new.and_then(|state| state.clipboxes.data_load.get(lod))
                            .copied(),
                        new.and_then(|state| state.clipboxes.data_retain.get(lod))
                            .copied(),
                    ),
                    (
                        old.map(|state| &state.mesh_resident),
                        new.map(|state| &state.mesh_resident),
                        old.and_then(|state| state.clipboxes.mesh_load.get(lod))
                            .copied(),
                        old.and_then(|state| state.clipboxes.mesh_retain.get(lod))
                            .copied(),
                        new.and_then(|state| state.clipboxes.mesh_load.get(lod))
                            .copied(),
                        new.and_then(|state| state.clipboxes.mesh_retain.get(lod))
                            .copied(),
                    ),
                ] {
                    let old_set = old_sets.and_then(|sets| sets.get(lod));
                    let new_set = new_sets.and_then(|sets| sets.get(lod));
                    work.resident_mutations += old_set.map_or(0, |old_set| {
                        old_set
                            .iter()
                            .filter(|position| new_set.is_none_or(|set| !set.contains(position)))
                            .count()
                    });
                    work.resident_mutations += new_set.map_or(0, |new_set| {
                        new_set
                            .iter()
                            .filter(|position| old_set.is_none_or(|set| !set.contains(position)))
                            .count()
                    });
                    work.shell_candidates += materialized_difference_len(
                        old_retain.unwrap_or_default(),
                        new_retain.unwrap_or_default(),
                    );
                    work.shell_candidates += materialized_difference_len(
                        new_load.unwrap_or_default(),
                        old_load.unwrap_or_default(),
                    );
                }
                for feature in [CoverageFeature::Visual, CoverageFeature::Collision] {
                    work.shell_candidates += materialized_symmetric_box_difference_len(
                        materialized_feature_box(old, lod, feature, false),
                        materialized_feature_box(new, lod, feature, false),
                    );
                    if lod > 0 {
                        work.shell_candidates += materialized_symmetric_box_difference_len(
                            materialized_feature_box(old, lod, feature, true),
                            materialized_feature_box(new, lod, feature, true),
                        );
                    }
                }
            }
        }
        work
    }

    fn materialized_feature_box(
        state: Option<&MaterializedViewerState>,
        lod: usize,
        feature: CoverageFeature,
        split: bool,
    ) -> Box3i {
        let Some(state) = state else {
            return Box3i::default();
        };
        let enabled = match feature {
            CoverageFeature::Visual => state.update.demand.visuals,
            CoverageFeature::Collision => state.update.demand.collisions,
        };
        if !enabled {
            return Box3i::default();
        }
        if split {
            split_box(&state.clipboxes, lod)
        } else {
            state.clipboxes.mesh_load[lod]
        }
    }

    fn materialized_symmetric_box_difference_len(a: Box3i, b: Box3i) -> usize {
        materialized_difference_len(a, b) + materialized_difference_len(b, a)
    }

    fn materialized_difference_len(a: Box3i, b: Box3i) -> usize {
        a.iter_cells_zxy()
            .filter(|position| !b.contains_point(*position))
            .count()
    }

    fn materialized_reconcile_resident_sets(
        previous: Option<&[HashSet<Vector3i>]>,
        load_boxes: &[Box3i],
        retain_boxes: &[Box3i],
    ) -> Vec<HashSet<Vector3i>> {
        load_boxes
            .iter()
            .zip(retain_boxes)
            .enumerate()
            .map(|(lod, (load, retain))| {
                let mut next = HashSet::new();
                if let Some(previous_lod) = previous.and_then(|sets| sets.get(lod)) {
                    next.extend(
                        previous_lod
                            .iter()
                            .copied()
                            .filter(|position| retain.contains_point(*position)),
                    );
                }
                next.extend(load.iter_cells_zxy());
                next
            })
            .collect()
    }

    fn materialized_build_aggregate(
        viewers: &BTreeMap<ViewerId, MaterializedViewerState>,
    ) -> Result<HashMap<ResidentBlockKey, DemandCounts>, CoordinatorError> {
        let mut aggregate = HashMap::new();
        for state in viewers.values() {
            for (lod, positions) in state.data_resident.iter().enumerate() {
                let lod_index =
                    u8::try_from(lod).map_err(|_| CoordinatorError::CoordinateOverflow)?;
                for &position in positions {
                    checked_increment(
                        &mut aggregate,
                        ResidentBlockKey {
                            kind: ResidentBlockKind::Data,
                            location: MeshBlockLocation::new(position, lod_index),
                        },
                        DemandCountField::Resident,
                    )?;
                }
            }
            for (lod, positions) in state.mesh_resident.iter().enumerate() {
                let lod_index =
                    u8::try_from(lod).map_err(|_| CoordinatorError::CoordinateOverflow)?;
                for &position in positions {
                    checked_increment(
                        &mut aggregate,
                        ResidentBlockKey {
                            kind: ResidentBlockKind::Mesh,
                            location: MeshBlockLocation::new(position, lod_index),
                        },
                        DemandCountField::Resident,
                    )?;
                }
            }
            for (lod, load_box) in state.clipboxes.mesh_load.iter().enumerate() {
                let lod_index =
                    u8::try_from(lod).map_err(|_| CoordinatorError::CoordinateOverflow)?;
                for position in load_box.iter_cells_zxy() {
                    let key = ResidentBlockKey {
                        kind: ResidentBlockKind::Mesh,
                        location: MeshBlockLocation::new(position, lod_index),
                    };
                    if state.update.demand.visuals {
                        checked_increment(&mut aggregate, key, DemandCountField::Visuals)?;
                    }
                    if state.update.demand.collisions {
                        checked_increment(&mut aggregate, key, DemandCountField::Collisions)?;
                    }
                }
            }
            for parent_lod in 1..state.clipboxes.mesh_load.len() {
                let lod_index =
                    u8::try_from(parent_lod).map_err(|_| CoordinatorError::CoordinateOverflow)?;
                let child_load = state.clipboxes.mesh_load[parent_lod - 1];
                for position in state.clipboxes.mesh_load[parent_lod].iter_cells_zxy() {
                    let parent = MeshBlockLocation::new(position, lod_index);
                    if !checked_children(parent)?
                        .into_iter()
                        .all(|child| child_load.contains_point(child.position_in_blocks))
                    {
                        continue;
                    }
                    let key = ResidentBlockKey {
                        kind: ResidentBlockKind::Mesh,
                        location: parent,
                    };
                    if state.update.demand.visuals {
                        checked_increment(&mut aggregate, key, DemandCountField::VisualSplits)?;
                    }
                    if state.update.demand.collisions {
                        checked_increment(&mut aggregate, key, DemandCountField::CollisionSplits)?;
                    }
                }
            }
        }
        Ok(aggregate)
    }

    fn assert_matches_materialized(
        persistent: &ClipboxCoordinator,
        materialized: &MaterializedCoordinator,
    ) {
        assert_eq!(persistent.revision(), materialized.revision);
        let persistent_aggregate = persistent
            .state
            .aggregate
            .iter()
            .map(|(key, counts)| (resident_key(*key), *counts))
            .collect::<HashMap<_, _>>();
        assert_eq!(persistent_aggregate, materialized.aggregate);
        assert_eq!(persistent.state.viewers.len(), materialized.viewers.len());
        for (viewer_id, expected) in &materialized.viewers {
            let actual = persistent.state.viewers.get(viewer_id).unwrap();
            assert_eq!(actual.update, expected.update);
            assert_eq!(actual.clipboxes, expected.clipboxes);
            for (actual_sets, expected_sets) in [
                (&actual.data_resident, &expected.data_resident),
                (&actual.mesh_resident, &expected.mesh_resident),
            ] {
                for (actual_set, expected_set) in actual_sets.iter().zip(expected_sets) {
                    let actual_set = actual_set
                        .iter()
                        .map(|position| Vector3i::new(position.0, position.1, position.2))
                        .collect::<HashSet<_>>();
                    assert_eq!(&actual_set, expected_set);
                }
            }
        }
    }

    fn assert_delta_matches(actual: &ResidentDemandDelta, expected: &ResidentDemandDelta) {
        if actual == expected {
            return;
        }
        let mismatch = actual
            .changes
            .iter()
            .zip(&expected.changes)
            .position(|(actual, expected)| actual != expected);
        let actual_map = actual
            .changes
            .iter()
            .map(|change| (change.key, (change.old_counts, change.new_counts)))
            .collect::<HashMap<_, _>>();
        let expected_map = expected
            .changes
            .iter()
            .map(|change| (change.key, (change.old_counts, change.new_counts)))
            .collect::<HashMap<_, _>>();
        let missing = expected_map
            .iter()
            .find(|(key, value)| actual_map.get(key) != Some(value));
        let extra = actual_map
            .iter()
            .find(|(key, value)| expected_map.get(key) != Some(value));
        panic!(
            "delta mismatch: revisions {} != {}, lengths {} != {}, first mismatch {:?}: actual={:?}, expected={:?}, missing={missing:?}, extra={extra:?}",
            actual.revision,
            expected.revision,
            actual.changes.len(),
            expected.changes.len(),
            mismatch,
            mismatch.and_then(|index| actual.changes.get(index)),
            mismatch.and_then(|index| expected.changes.get(index)),
        );
    }

    #[test]
    fn crossing_one_load_cell_inside_retain_box_does_not_emit_exit() {
        let mut coordinator = test_coordinator();
        coordinator
            .update_viewers(&[viewer(1, Vector3i::new(15, 0, 0), true, false)])
            .unwrap();
        let revision = coordinator.revision();
        let delta = coordinator
            .update_viewers(&[viewer(1, Vector3i::new(16, 0, 0), true, false)])
            .unwrap();
        assert!(delta
            .changes
            .iter()
            .all(|change| change.new_counts.resident > 0));
        assert!(!delta.changes.is_empty());
        assert!(coordinator.revision() > revision);
        assert_eq!(delta.revision, coordinator.revision());
    }

    #[test]
    fn two_viewers_keep_separate_visual_and_collision_counts() {
        let mut coordinator = test_coordinator();
        coordinator
            .update_viewers(&[
                viewer(1, Vector3i::zero(), true, false),
                viewer(2, Vector3i::zero(), false, true),
            ])
            .unwrap();
        let parent = loc(Vector3i::zero(), 1);
        assert_eq!(
            coordinator.mesh_demand(parent),
            DemandCounts {
                resident: 2,
                visuals: 1,
                collisions: 1,
                visual_splits: 1,
                collision_splits: 1,
            }
        );
        let lod0 = loc(Vector3i::zero(), 0);
        assert_eq!(coordinator.split_demand(lod0, CoverageFeature::Visual), 0);
        assert_eq!(
            coordinator.split_demand(lod0, CoverageFeature::Collision),
            0
        );

        coordinator
            .update_viewers(&[viewer(2, Vector3i::zero(), false, true)])
            .unwrap();
        assert_eq!(
            coordinator.mesh_demand(parent),
            DemandCounts {
                resident: 1,
                visuals: 0,
                collisions: 1,
                visual_splits: 0,
                collision_splits: 1,
            }
        );
    }

    #[test]
    fn retain_only_mesh_has_residency_without_feature_demand() {
        let mut coordinator = test_coordinator();
        coordinator
            .update_viewers(&[viewer(1, Vector3i::zero(), true, true)])
            .unwrap();
        coordinator
            .update_viewers(&[viewer(1, Vector3i::new(32, 0, 0), true, true)])
            .unwrap();
        assert_eq!(
            coordinator.mesh_demand(loc(Vector3i::new(-2, 0, 0), 0)),
            DemandCounts {
                resident: 1,
                ..DemandCounts::default()
            }
        );
    }

    #[test]
    fn data_and_mesh_counts_survive_both_viewer_removal_orders() {
        for remaining in [1, 2] {
            let mut coordinator = test_coordinator();
            coordinator
                .update_viewers(&[
                    viewer(1, Vector3i::zero(), true, false),
                    viewer(2, Vector3i::zero(), false, true),
                ])
                .unwrap();
            coordinator
                .update_viewers(&[viewer(
                    remaining,
                    Vector3i::zero(),
                    remaining == 1,
                    remaining == 2,
                )])
                .unwrap();
            assert_eq!(
                coordinator.data_demand(loc(Vector3i::zero(), 0)).resident,
                1
            );
            assert_eq!(
                coordinator.mesh_demand(loc(Vector3i::zero(), 0)).resident,
                1
            );
            coordinator.update_viewers(&[]).unwrap();
            assert_eq!(
                coordinator.data_demand(loc(Vector3i::zero(), 0)),
                DemandCounts::default()
            );
            assert_eq!(
                coordinator.mesh_demand(loc(Vector3i::zero(), 0)),
                DemandCounts::default()
            );
        }
    }

    #[test]
    fn finer_demand_wins_without_erasing_coarse_residency() {
        let mut coordinator = test_coordinator();
        coordinator
            .update_viewers(&[viewer(1, Vector3i::zero(), true, true)])
            .unwrap();
        let parent = loc(Vector3i::zero(), 1);
        assert!(coordinator.mesh_demand(parent).resident > 0);
        assert!(coordinator.split_demand(parent, CoverageFeature::Visual) > 0);
        assert!(coordinator.split_demand(parent, CoverageFeature::Collision) > 0);
        for child in checked_children(parent).unwrap() {
            assert!(coordinator.mesh_demand(child).resident > 0);
        }
    }

    #[test]
    fn duplicate_viewer_ids_reject_the_whole_snapshot_without_mutation() {
        let mut coordinator = test_coordinator();
        coordinator
            .update_viewers(&[viewer(1, Vector3i::zero(), true, false)])
            .unwrap();
        let before = coordinator.clone();
        assert_eq!(
            coordinator.update_viewers(&[
                viewer(1, Vector3i::zero(), true, false),
                viewer(1, Vector3i::new(16, 0, 0), false, true),
            ]),
            Err(CoordinatorError::DuplicateViewerId(1))
        );
        assert_eq!(coordinator, before);
    }

    #[test]
    fn failed_clipbox_math_is_transactional() {
        let mut coordinator = test_coordinator();
        coordinator
            .update_viewers(&[viewer(1, Vector3i::zero(), true, true)])
            .unwrap();
        let before = coordinator.clone();
        let mut invalid = viewer(1, Vector3i::zero(), true, true);
        invalid.view_distance_voxels.x = -1;
        assert_eq!(
            coordinator.update_viewers(&[invalid]),
            Err(CoordinatorError::LodMath(LodMathError::NegativeDistance))
        );
        assert_eq!(coordinator, before);
    }

    #[test]
    fn viewer_input_order_does_not_change_delta_order_or_state() {
        let updates = [
            viewer(2, Vector3i::new(32, 0, 0), false, true),
            viewer(1, Vector3i::zero(), true, false),
        ];
        let mut a = test_coordinator();
        let mut b = test_coordinator();
        let delta_a = a.update_viewers(&updates).unwrap();
        let delta_b = b
            .update_viewers(&[updates[1].clone(), updates[0].clone()])
            .unwrap();
        assert_eq!(delta_a, delta_b);
        assert_eq!(a, b);
    }

    #[test]
    fn viewer_without_mesh_features_streams_data_but_not_meshes() {
        let mut coordinator = test_coordinator();
        coordinator
            .update_viewers(&[viewer(1, Vector3i::zero(), false, false)])
            .unwrap();
        assert!(coordinator.data_demand(loc(Vector3i::zero(), 0)).resident > 0);
        assert_eq!(
            coordinator.mesh_demand(loc(Vector3i::zero(), 0)),
            DemandCounts::default()
        );
    }

    #[test]
    fn disabling_last_mesh_feature_clears_all_per_viewer_mesh_state() {
        let mut coordinator = test_coordinator();
        coordinator
            .update_viewers(&[viewer(1, Vector3i::zero(), true, true)])
            .unwrap();
        coordinator
            .update_viewers(&[viewer(1, Vector3i::new(32, 0, 0), true, true)])
            .unwrap();
        assert_eq!(
            coordinator.mesh_demand(loc(Vector3i::new(-2, 0, 0), 0)),
            DemandCounts {
                resident: 1,
                ..DemandCounts::default()
            }
        );

        coordinator
            .update_viewers(&[viewer(1, Vector3i::new(32, 0, 0), false, false)])
            .unwrap();

        let state = coordinator.state.viewers.get(&1).unwrap();
        assert!(state.clipboxes.mesh_load.iter().all(Box3i::is_empty));
        assert!(state.clipboxes.mesh_retain.iter().all(Box3i::is_empty));
        assert!(state
            .mesh_resident
            .iter()
            .all(|positions| positions.is_empty()));
        assert!(state
            .data_resident
            .iter()
            .any(|positions| !positions.is_empty()));
        assert!(coordinator.state.aggregate.keys().all(|key| key.kind == 0));
    }

    #[test]
    fn aggregate_empty_swap_commits_latest_per_viewer_snapshot() {
        let left = Vector3i::new(-384, 0, 0);
        let right = Vector3i::new(384, 0, 0);
        let initial = [viewer(1, left, true, false), viewer(2, right, true, false)];
        let swapped = [viewer(1, right, true, false), viewer(2, left, true, false)];
        let mut coordinator = test_coordinator();
        coordinator.update_viewers(&initial).unwrap();
        let revision = coordinator.revision();

        let delta = coordinator.update_viewers(&swapped).unwrap();
        assert!(delta.changes.is_empty());
        assert_eq!(delta.revision, revision);

        let mut expected = test_coordinator();
        expected.update_viewers(&swapped).unwrap();
        assert_eq!(coordinator, expected);

        let follow_up = [
            viewer(1, Vector3i::new(368, 0, 0), true, false),
            swapped[1].clone(),
        ];
        assert_eq!(
            coordinator.update_viewers(&follow_up).unwrap(),
            expected.update_viewers(&follow_up).unwrap()
        );
        assert_eq!(coordinator, expected);
    }

    #[test]
    fn empty_delta_carries_the_preserved_post_call_revision() {
        let mut coordinator = test_coordinator();
        let update = viewer(1, Vector3i::zero(), true, true);
        coordinator
            .update_viewers(std::slice::from_ref(&update))
            .unwrap();
        let delta = coordinator.update_viewers(&[update]).unwrap();
        assert!(delta.changes.is_empty());
        assert_eq!(delta.revision, coordinator.revision());
    }

    #[test]
    fn constructor_rejects_invalid_settings_even_without_viewers() {
        let mut invalid = settings();
        invalid.mesh_block_size = 0;
        assert_eq!(
            ClipboxCoordinator::new(invalid, bounds()),
            Err(CoordinatorError::LodMath(LodMathError::InvalidBlockSize))
        );
    }

    #[test]
    fn checked_count_exchange_is_order_independent_at_u32_max() {
        let key = ResidentBlockKey {
            kind: ResidentBlockKind::Mesh,
            location: loc(Vector3i::zero(), 1),
        };
        assert_eq!(
            checked_net_count(u32::MAX, 1, 1, key, DemandCountField::Resident).unwrap(),
            u32::MAX
        );
        assert_eq!(
            checked_net_count(0, 1, 0, key, DemandCountField::Resident),
            Err(CoordinatorError::RefcountUnderflow {
                key,
                field: DemandCountField::Resident,
            })
        );
        assert_eq!(
            checked_net_count(u32::MAX, 0, 1, key, DemandCountField::Resident),
            Err(CoordinatorError::RefcountOverflow {
                key,
                field: DemandCountField::Resident,
            })
        );
    }

    #[test]
    fn every_aggregate_count_field_reports_typed_overflow_without_mutation() {
        let key = ResidentBlockKey {
            kind: ResidentBlockKind::Mesh,
            location: loc(Vector3i::zero(), 1),
        };
        for field in [
            DemandCountField::Resident,
            DemandCountField::Visuals,
            DemandCountField::Collisions,
            DemandCountField::VisualSplits,
            DemandCountField::CollisionSplits,
        ] {
            let mut aggregate = std::collections::HashMap::from([(
                key,
                DemandCounts {
                    resident: u32::MAX,
                    visuals: u32::MAX,
                    collisions: u32::MAX,
                    visual_splits: u32::MAX,
                    collision_splits: u32::MAX,
                },
            )]);
            let before = aggregate.clone();
            assert_eq!(
                checked_increment(&mut aggregate, key, field),
                Err(CoordinatorError::RefcountOverflow { key, field })
            );
            assert_eq!(aggregate, before);
        }
    }

    #[test]
    fn checked_children_reject_coordinate_overflow() {
        assert_eq!(
            checked_children(loc(Vector3i::splat(i32::MAX), 1)),
            Err(CoordinatorError::CoordinateOverflow)
        );
    }

    #[test]
    fn revision_overflow_is_transactional() {
        let mut coordinator = test_coordinator();
        let mut state = (*coordinator.state).clone();
        state.revision = u64::MAX;
        coordinator.state = Arc::new(state);
        let before = coordinator.clone();
        assert_eq!(
            coordinator.update_viewers(&[viewer(1, Vector3i::zero(), true, true)]),
            Err(CoordinatorError::RevisionOverflow)
        );
        assert_eq!(coordinator, before);
    }

    #[test]
    fn exact_identical_viewer_snapshot_reuses_coordinator_arc() {
        let mut coordinator = test_coordinator();
        let viewers = [viewer(1, Vector3i::zero(), true, true)];
        coordinator.update_viewers(&viewers).unwrap();
        let before = coordinator.state_identity_for_test();

        let prepared = coordinator.prepare_update(&viewers).unwrap();
        assert!(prepared.delta().changes.is_empty());
        coordinator.apply_prepared(prepared).unwrap();

        assert_eq!(coordinator.state_identity_for_test(), before);
    }

    #[test]
    fn replaying_exact_noop_prepared_updates_is_allowed() {
        let mut coordinator = test_coordinator();
        let viewers = [viewer(1, Vector3i::zero(), true, true)];
        coordinator.update_viewers(&viewers).unwrap();
        let before = coordinator.state_identity_for_test();
        let first = coordinator.prepare_update(&viewers).unwrap();
        let second = coordinator.prepare_update(&viewers).unwrap();

        coordinator.apply_prepared(first).unwrap();
        coordinator.apply_prepared(second).unwrap();

        assert_eq!(coordinator.state_identity_for_test(), before);
    }

    #[test]
    fn aggregate_empty_viewer_cache_change_gets_a_new_coordinator_arc() {
        let left = Vector3i::new(-384, 0, 0);
        let right = Vector3i::new(384, 0, 0);
        let initial = [viewer(1, left, true, false), viewer(2, right, true, false)];
        let swapped = [viewer(1, right, true, false), viewer(2, left, true, false)];
        let mut coordinator = test_coordinator();
        coordinator.update_viewers(&initial).unwrap();
        let before = coordinator.state_identity_for_test();

        let prepared = coordinator.prepare_update(&swapped).unwrap();
        assert!(prepared.delta().changes.is_empty());
        coordinator.apply_prepared(prepared).unwrap();

        assert_ne!(coordinator.state_identity_for_test(), before);
        let mut expected = test_coordinator();
        expected.update_viewers(&swapped).unwrap();
        assert_eq!(coordinator, expected);
    }

    #[test]
    fn prepared_coordinator_update_accepts_undiverged_clone() {
        let source = test_coordinator();
        let prepared = source
            .prepare_update(&[viewer(1, Vector3i::zero(), true, false)])
            .unwrap();
        let mut draft = source.clone();

        draft.apply_prepared(prepared).unwrap();

        assert_ne!(draft, source);
    }

    #[test]
    fn validated_coordinator_publication_retires_old_and_base_arcs_after_publish() {
        let mut coordinator = test_coordinator();
        let old_state = Arc::downgrade(&coordinator.state);
        let prepared = coordinator
            .prepare_update(&[viewer(1, Vector3i::zero(), true, false)])
            .unwrap();
        let expected_delta = prepared.delta().clone();
        let validated = prepared.validate_for(&coordinator).unwrap();
        assert!(!validated.delta().changes.is_empty());

        let published = validated.publish(&mut coordinator);
        assert!(old_state.upgrade().is_some());
        let (delta, retirement) = published.into_parts();
        assert_eq!(delta, expected_delta);
        assert!(old_state.upgrade().is_some());

        drop(retirement);
        assert!(old_state.upgrade().is_none());
    }

    #[test]
    fn coordinator_validation_rejects_stale_base_without_mutation() {
        let mut coordinator = test_coordinator();
        let stale = coordinator
            .prepare_update(&[viewer(1, Vector3i::zero(), true, false)])
            .unwrap();
        coordinator
            .update_viewers(&[viewer(2, Vector3i::new(32, 0, 0), false, true)])
            .unwrap();
        let before = coordinator.clone();

        assert!(matches!(
            stale.validate_for(&coordinator),
            Err(CoordinatorError::StalePreparedIdentity)
        ));
        assert_eq!(coordinator, before);
    }

    #[test]
    fn coordinator_revalidation_rejects_state_changed_after_validation_without_mutation() {
        let mut coordinator = test_coordinator();
        let validated = coordinator
            .prepare_update(&[viewer(1, Vector3i::zero(), true, false)])
            .unwrap()
            .validate_for(&coordinator)
            .unwrap();
        coordinator
            .update_viewers(&[viewer(2, Vector3i::new(32, 0, 0), false, true)])
            .unwrap();
        let before = coordinator.clone();
        let before_identity = coordinator.state_identity_for_test();

        assert_eq!(
            validated.revalidate_for(&coordinator),
            Err(CoordinatorError::StalePreparedIdentity)
        );
        assert_eq!(coordinator, before);
        assert_eq!(coordinator.state_identity_for_test(), before_identity);
    }

    #[test]
    fn competing_mutating_coordinator_previews_make_second_stale() {
        let mut coordinator = test_coordinator();
        let first = coordinator
            .prepare_update(&[viewer(1, Vector3i::zero(), true, false)])
            .unwrap();
        let second = coordinator
            .prepare_update(&[viewer(2, Vector3i::new(32, 0, 0), false, true)])
            .unwrap();
        coordinator.apply_prepared(first).unwrap();
        let before = coordinator.clone();

        assert_eq!(
            coordinator.apply_prepared(second),
            Err(CoordinatorError::StalePreparedIdentity)
        );
        assert_eq!(coordinator, before);
    }

    #[test]
    fn independent_equal_coordinator_rejects_prepared_update() {
        let source = test_coordinator();
        let prepared = source
            .prepare_update(&[viewer(1, Vector3i::zero(), true, false)])
            .unwrap();
        let mut independent = test_coordinator();
        assert_eq!(source, independent);

        assert_eq!(
            independent.apply_prepared(prepared),
            Err(CoordinatorError::StalePreparedIdentity)
        );
    }

    #[test]
    fn diverged_coordinator_clone_rejects_old_prepared_update() {
        let source = test_coordinator();
        let mut branch = source.clone();
        let stale = branch
            .prepare_update(&[viewer(1, Vector3i::zero(), true, false)])
            .unwrap();
        let diverging = branch
            .prepare_update(&[viewer(2, Vector3i::new(32, 0, 0), false, true)])
            .unwrap();
        branch.apply_prepared(diverging).unwrap();
        let before = branch.clone();

        assert_eq!(
            branch.apply_prepared(stale),
            Err(CoordinatorError::StalePreparedIdentity)
        );
        assert_eq!(branch, before);
    }

    #[test]
    fn persistent_shell_reconcile_avoids_full_resident_iteration() {
        let mut coordinator = test_coordinator();
        coordinator
            .update_viewers(&[viewer(1, Vector3i::zero(), true, true)])
            .unwrap();
        let prepared = coordinator
            .prepare_update(&[viewer(1, Vector3i::new(16, 0, 0), true, true)])
            .unwrap();
        let work = prepared.work_counters_for_test();

        assert!(work.shell_candidates > 0);
        assert!(work.resident_mutations > 0);
        assert_eq!(work.full_resident_iterations, 0);
    }

    #[test]
    fn persistent_shell_reconcile_matches_materialized_reference_sequences() {
        let sequences = [
            vec![
                vec![viewer(1, Vector3i::zero(), true, true)],
                vec![viewer(1, Vector3i::new(320, -192, 64), true, true)],
                vec![],
                vec![viewer(1, Vector3i::new(-320, 192, -64), false, true)],
            ],
            vec![
                vec![viewer(1, Vector3i::zero(), true, false)],
                vec![viewer(1, Vector3i::new(16, 0, 0), true, false)],
                vec![viewer(1, Vector3i::new(32, 0, 0), true, false)],
                vec![viewer(1, Vector3i::new(16, 0, 0), true, false)],
                vec![viewer(1, Vector3i::zero(), true, false)],
            ],
            (0..48)
                .map(|step| {
                    let x = (step % 8) * 16 - 64;
                    let y = ((step / 8) % 2) * 32 - 16;
                    vec![viewer(
                        1,
                        Vector3i::new(x, y, 0),
                        step % 3 != 0,
                        step % 4 != 0,
                    )]
                })
                .collect::<Vec<_>>(),
        ];

        for sequence in sequences {
            let mut persistent = test_coordinator();
            let mut materialized = MaterializedCoordinator::new(settings(), bounds());
            for viewers in sequence {
                let prepared = persistent.prepare_update(&viewers).unwrap();
                let actual_delta = prepared.delta().clone();
                let work = prepared.work_counters_for_test();
                assert_eq!(work.full_resident_iterations, 0);
                let expected_delta = materialized.update(&viewers).unwrap();
                assert_delta_matches(&actual_delta, &expected_delta);
                assert_eq!(
                    work.shell_candidates,
                    materialized.last_work.shell_candidates
                );
                assert_eq!(
                    work.resident_mutations,
                    materialized.last_work.resident_mutations
                );
                assert_eq!(
                    work.aggregate_mutations,
                    materialized.last_work.aggregate_mutations
                );
                assert_eq!(
                    work.changed_aggregate_keys,
                    materialized.last_work.changed_aggregate_keys
                );
                persistent.apply_prepared(prepared).unwrap();
                assert_matches_materialized(&persistent, &materialized);
            }
        }
    }

    #[test]
    fn persistent_shell_reconcile_matches_randomized_materialized_trajectory() {
        let mut persistent = test_coordinator();
        let mut materialized = MaterializedCoordinator::new(settings(), bounds());
        let mut seed = 0x5eed_cafe_u64;

        for _ in 0..1_000 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let viewer_count = (seed % 4) as u32;
            let mut viewers = Vec::with_capacity(viewer_count as usize);
            for id in 1..=viewer_count {
                seed = seed
                    .wrapping_mul(2_862_933_555_777_941_757)
                    .wrapping_add(3_037_000_493);
                let x = ((seed & 0x1f) as i32 - 16) * 16;
                let y = (((seed >> 5) & 0x1f) as i32 - 16) * 16;
                let z = (((seed >> 10) & 0x1f) as i32 - 16) * 16;
                viewers.push(viewer(
                    id,
                    Vector3i::new(x, y, z),
                    seed & (1 << 20) != 0,
                    seed & (1 << 21) != 0,
                ));
            }
            if seed & (1 << 22) != 0 {
                viewers.reverse();
            }

            let prepared = persistent.prepare_update(&viewers).unwrap();
            let actual_delta = prepared.delta().clone();
            let work = prepared.work_counters_for_test();
            assert_eq!(work.full_resident_iterations, 0);
            let expected_delta = materialized.update(&viewers).unwrap();
            assert_delta_matches(&actual_delta, &expected_delta);
            assert_eq!(
                work.shell_candidates,
                materialized.last_work.shell_candidates
            );
            assert_eq!(
                work.resident_mutations,
                materialized.last_work.resident_mutations
            );
            assert_eq!(
                work.aggregate_mutations,
                materialized.last_work.aggregate_mutations
            );
            assert_eq!(
                work.changed_aggregate_keys,
                materialized.last_work.changed_aggregate_keys
            );
            persistent.apply_prepared(prepared).unwrap();
            assert_matches_materialized(&persistent, &materialized);
        }
    }

    #[test]
    fn one_block_crossing_work_scales_with_shell_not_resident_volume() {
        fn measure(distance: i32) -> (CoordinatorWorkCounters, usize) {
            let measured_settings = LodClipboxSettings {
                lod0_distance_voxels: distance,
                secondary_distance_voxels: distance,
                ..settings()
            };
            let measured_bounds = Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048));
            let update = |position_voxels| ClipboxViewerUpdate {
                id: 1,
                position_voxels,
                view_distance_voxels: Vector3i::splat(distance),
                demand: MeshDemand {
                    visuals: true,
                    collisions: true,
                },
            };
            let mut persistent =
                ClipboxCoordinator::new(measured_settings, measured_bounds).unwrap();
            let mut materialized = MaterializedCoordinator::new(measured_settings, measured_bounds);
            let crossing_x = (-256..256)
                .find(|x| {
                    let before = compute_lod_clipboxes(
                        Vector3i::new(*x, 0, 0),
                        Vector3i::splat(distance),
                        measured_bounds,
                        measured_settings,
                    )
                    .unwrap();
                    let after = compute_lod_clipboxes(
                        Vector3i::new(*x + 1, 0, 0),
                        Vector3i::splat(distance),
                        measured_bounds,
                        measured_settings,
                    )
                    .unwrap();
                    before.data_load[0] != after.data_load[0]
                        || before.mesh_load[0] != after.mesh_load[0]
                })
                .expect("fixture must contain a one-voxel LOD-0 block crossing");
            persistent
                .update_viewers(&[update(Vector3i::new(crossing_x, 0, 0))])
                .unwrap();
            materialized
                .update(&[update(Vector3i::new(crossing_x, 0, 0))])
                .unwrap();
            let prepared = persistent
                .prepare_update(&[update(Vector3i::new(crossing_x + 1, 0, 0))])
                .unwrap();
            let work = prepared.work_counters_for_test();
            materialized
                .update(&[update(Vector3i::new(crossing_x + 1, 0, 0))])
                .unwrap();
            assert_eq!(
                work.shell_candidates,
                materialized.last_work.shell_candidates
            );
            assert_eq!(
                work.resident_mutations,
                materialized.last_work.resident_mutations
            );
            assert_eq!(
                work.aggregate_mutations,
                materialized.last_work.aggregate_mutations
            );
            assert_eq!(work.full_resident_iterations, 0);
            let resident = persistent
                .state
                .viewers
                .get(&1)
                .unwrap()
                .data_resident
                .iter()
                .chain(&persistent.state.viewers.get(&1).unwrap().mesh_resident)
                .map(OrdSet::len)
                .sum();
            (work, resident)
        }

        let (small_work, small_resident) = measure(32);
        let (large_work, large_resident) = measure(128);
        assert!(
            large_resident > small_resident * 4,
            "small resident={small_resident}, shell={}; large resident={large_resident}, shell={}",
            small_work.shell_candidates,
            large_work.shell_candidates,
        );
        assert!(
            large_work.shell_candidates > small_work.shell_candidates,
            "small resident={small_resident}, shell={}; large resident={large_resident}, shell={}",
            small_work.shell_candidates,
            large_work.shell_candidates,
        );
        assert!(
            large_work.shell_candidates * small_resident
                < small_work.shell_candidates * large_resident
        );
    }

    #[test]
    fn production_aggregate_overflow_and_underflow_preserve_identity_and_state() {
        let updates = [viewer(1, Vector3i::zero(), true, true)];

        let mut overflow = test_coordinator();
        let overflow_key = overflow
            .prepare_update(&updates)
            .unwrap()
            .delta()
            .changes
            .iter()
            .find(|change| {
                change.key.kind == ResidentBlockKind::Data
                    && change.old_counts.resident == 0
                    && change.new_counts.resident == 1
            })
            .unwrap()
            .key;
        let mut overflow_state = (*overflow.state).clone();
        overflow_state.aggregate.insert(
            demand_key(overflow_key),
            DemandCounts {
                resident: u32::MAX,
                ..DemandCounts::default()
            },
        );
        overflow.state = Arc::new(overflow_state);
        let overflow_before = overflow.clone();
        let overflow_identity = overflow.state_identity_for_test();
        assert_eq!(
            overflow.prepare_update(&updates).unwrap_err(),
            CoordinatorError::RefcountOverflow {
                key: overflow_key,
                field: DemandCountField::Resident,
            }
        );
        assert_eq!(overflow, overflow_before);
        assert_eq!(overflow.state_identity_for_test(), overflow_identity);

        let mut underflow = test_coordinator();
        underflow.update_viewers(&updates).unwrap();
        let underflow_key = underflow
            .prepare_update(&[])
            .unwrap()
            .delta()
            .changes
            .iter()
            .find(|change| {
                change.key.kind == ResidentBlockKind::Data
                    && change.old_counts.resident == 1
                    && change.new_counts.resident == 0
            })
            .unwrap()
            .key;
        let mut underflow_state = (*underflow.state).clone();
        underflow_state.aggregate.remove(&demand_key(underflow_key));
        underflow.state = Arc::new(underflow_state);
        let underflow_before = underflow.clone();
        let underflow_identity = underflow.state_identity_for_test();
        assert_eq!(
            underflow.prepare_update(&[]).unwrap_err(),
            CoordinatorError::RefcountUnderflow {
                key: underflow_key,
                field: DemandCountField::Resident,
            }
        );
        assert_eq!(underflow, underflow_before);
        assert_eq!(underflow.state_identity_for_test(), underflow_identity);
    }
}
