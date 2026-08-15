use super::*;
use crate::storage::voxel_data::SharedVoxelDataTransactionReservation;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, Weak};
use std::thread;
use std::time::Duration;

const COMMITTED_POSITION: Vector3i = Vector3i::new(40, 0, 0);
const LOAD_GENERATION: u64 = 41;
const MESH_REVISION: u64 = 51;
const VOXEL_MARKER: u64 = 0xC2_C2;

#[derive(Debug, PartialEq, Eq)]
struct CommittedFixedObservation {
    commit_marker: bool,
    voxel_marker: u64,
    data_viewers: u32,
    loading_generation: Option<u64>,
    next_load_generation: u64,
    resident_viewers: u32,
    visual_viewers: u32,
    visual_active: bool,
    requested_mesh_revision: Option<u64>,
    applied_mesh_revision: Option<u64>,
    next_mesh_revision: u64,
    loaded_events: usize,
    mesh_events: usize,
    topology_events: usize,
}

struct FixedStateObserverTask {
    core: Weak<Mutex<VoxelTerrainCore>>,
    commit_marker: Arc<AtomicBool>,
    runs: Arc<AtomicUsize>,
    observed: Option<mpsc::Sender<CommittedFixedObservation>>,
}

impl ThreadedTask for FixedStateObserverTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        self.runs.fetch_add(1, Ordering::SeqCst);
        let core = self
            .core
            .upgrade()
            .expect("terrain owner remains alive until the observer reports");
        let core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let block = core
            .data
            .block_snapshot(COMMITTED_POSITION, 0)
            .expect("the committed load is visible after publication");
        let mesh = core.mesh_maps[0]
            .get(&COMMITTED_POSITION)
            .expect("the committed mesh entry is visible after publication");
        let observation = CommittedFixedObservation {
            commit_marker: self.commit_marker.load(Ordering::SeqCst),
            voxel_marker: block
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            data_viewers: block.viewers.get(),
            loading_generation: core.loading_blocks[0]
                .get(&COMMITTED_POSITION)
                .map(|entry| entry.request_generation),
            next_load_generation: core.next_request_generation,
            resident_viewers: mesh.resident_viewers,
            visual_viewers: mesh.visual_viewers,
            visual_active: mesh.visual_active,
            requested_mesh_revision: mesh.requested_revision,
            applied_mesh_revision: mesh.applied_revision,
            next_mesh_revision: core.next_mesh_revision,
            loaded_events: core
                .event_outbox
                .iter()
                .filter(|event| {
                    matches!(event, VoxelTerrainEvent::DataBlockLoaded(location)
                        if *location == BlockLocation { position: COMMITTED_POSITION, lod_index: 0 })
                })
                .count(),
            mesh_events: core
                .event_outbox
                .iter()
                .filter(|event| {
                    event
                        .mesh_descriptor()
                        .is_some_and(|descriptor| descriptor.key.revision == MESH_REVISION)
                })
                .count(),
            topology_events: core
                .event_outbox
                .iter()
                .filter(|event| matches!(event, VoxelTerrainEvent::RenderTopologyChanged(_)))
                .count(),
        };
        self.observed
            .take()
            .expect("the observer task runs exactly once")
            .send(observation)
            .expect("the observation receiver remains live");
        TaskRunStatus::Complete {
            follow_up_tasks: Vec::new(),
        }
    }

    fn debug_name(&self) -> &'static str {
        "C2cC2FixedStateObserver"
    }
}

struct BlockingPermitTask {
    entered: Option<mpsc::Sender<()>>,
    release: mpsc::Receiver<()>,
}

impl ThreadedTask for BlockingPermitTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        self.entered
            .take()
            .expect("the blocking task runs exactly once")
            .send(())
            .expect("the blocker entry receiver remains live");
        self.release
            .recv()
            .expect("the blocker release sender remains live");
        TaskRunStatus::Complete {
            follow_up_tasks: Vec::new(),
        }
    }

    fn debug_name(&self) -> &'static str {
        "C2cC2StalePermitBlocker"
    }
}

struct PermitWitnessTask {
    finished: Option<mpsc::Sender<()>>,
}

impl ThreadedTask for PermitWitnessTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        self.finished
            .take()
            .expect("the permit witness runs exactly once")
            .send(())
            .expect("the permit witness receiver remains live");
        TaskRunStatus::Complete {
            follow_up_tasks: Vec::new(),
        }
    }

    fn debug_name(&self) -> &'static str {
        "C2cC2StalePermitWitness"
    }
}

struct ExactBatchTask {
    name: &'static str,
    runs: Arc<AtomicUsize>,
}

struct IdentitySaveStream {
    calls: AtomicUsize,
    observed_payload: AtomicUsize,
    started: Mutex<Option<mpsc::Sender<()>>>,
    released: Mutex<bool>,
    release: Condvar,
}

#[derive(Clone, Copy, Debug)]
enum DirtyExitFailure {
    TerrainCapacity(FixedCapacityDestination),
    StorageCapacity(SharedVoxelDataTransactionReservation),
    LiveSpatialRegistry,
}

impl IdentitySaveStream {
    fn unblocked() -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            observed_payload: AtomicUsize::new(0),
            started: Mutex::new(None),
            released: Mutex::new(true),
            release: Condvar::new(),
        })
    }

    fn blocked() -> (Arc<Self>, mpsc::Receiver<()>) {
        let (started, received) = mpsc::channel();
        (
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                observed_payload: AtomicUsize::new(0),
                started: Mutex::new(Some(started)),
                released: Mutex::new(false),
                release: Condvar::new(),
            }),
            received,
        )
    }

    fn release(&self) {
        *self
            .released
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
        self.release.notify_all();
    }
}

impl VoxelStream for IdentitySaveStream {
    fn save_voxel_block(
        &self,
        query: crate::streams::VoxelSaveQuery<'_>,
    ) -> crate::streams::StreamResult<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.observed_payload.store(
            query
                .voxel_buffer
                .channel_bytes(ChannelId::Type.index())
                .as_ptr() as usize,
            Ordering::SeqCst,
        );
        if let Some(started) = self
            .started
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            started
                .send(())
                .expect("the blocked-save observer remains live");
        }
        let mut released = self
            .released
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while !*released {
            released = self
                .release
                .wait(released)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        Ok(())
    }
}

impl ThreadedTask for ExactBatchTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        self.runs.fetch_add(1, Ordering::SeqCst);
        TaskRunStatus::Complete {
            follow_up_tasks: Vec::new(),
        }
    }

    fn debug_name(&self) -> &'static str {
        self.name
    }
}

fn task_identity(task: &dyn ThreadedTask) -> usize {
    task as *const dyn ThreadedTask as *const () as usize
}

fn stage_loaded_completion(core: &mut VoxelTerrainCore, follow_up_tasks: Vec<ScheduledTask>) {
    core.loading_blocks[0].insert(
        COMMITTED_POSITION,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: LOAD_GENERATION,
            request_state: LoadRequestState::InFlight,
            physical_request: None,
        },
    );
    core.next_request_generation = LOAD_GENERATION + 1;
    let mut task = LoadBlockForTerrainTask::new(
        COMMITTED_POSITION,
        0,
        LOAD_GENERATION,
        core.data.clone(),
        core.stream.clone(),
    );
    task.output = Some(loaded_output(
        core,
        COMMITTED_POSITION,
        LOAD_GENERATION,
        VOXEL_MARKER,
    ));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        follow_up_tasks,
    ));
    core.try_normalize_raw_completions()
        .expect("the exact raw completion normalizes before the commit probe");
}

fn stage_direct_mesh_acceptance(core: &mut VoxelTerrainCore) {
    let features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };
    core.mesh_maps[0].insert(
        COMMITTED_POSITION,
        MeshBlockEntry {
            position: COMMITTED_POSITION,
            resident_viewers: 1,
            visual_viewers: 1,
            requested_revision: Some(MESH_REVISION),
            requested_features: features,
            ..MeshBlockEntry::default()
        },
    );
    core.next_mesh_revision = MESH_REVISION + 1;
    let output = pooled_mesh_output(
        Arc::new(MeshArraysPool::new()),
        MeshBlockKey {
            location: MeshBlockLocation::new(COMMITTED_POSITION, 0),
            revision: MESH_REVISION,
        },
        features,
        1,
        0,
        false,
    )
    .into_upload();
    let (upload, dropped) = output.into_parts();
    assert!(!dropped);
    core.direct_mesh_retry_inbox
        .push_back(DurableCompletion::DirectMesh { upload, dropped });
}

fn resident_buffer_ptr(core: &VoxelTerrainCore, position: Vector3i) -> usize {
    core.data.with_lod_map(0, |map| {
        map.get_block(position)
            .expect("the resident owner remains present")
            .voxels()
            .channel_bytes(ChannelId::Type.index())
            .as_ptr() as usize
    })
}

fn build_dirty_exit_core(stream: Arc<dyn VoxelStream>) -> (VoxelTerrainCore, usize) {
    let mut core = build_core_with_stream(stream);
    core.task_runner.set_thread_count(1);
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
    voxels.set_voxel(VOXEL_MARKER, 0, 0, 0, ChannelId::Type.index());
    let payload_ptr = voxels.channel_bytes(ChannelId::Type.index()).as_ptr() as usize;
    let mut block = VoxelDataBlock::with_voxels(voxels, 0);
    block.set_modified(true);
    assert!(core.data.try_set_block(COMMITTED_POSITION, block).unwrap());
    core.data
        .view_area(single_block_box(COMMITTED_POSITION), 0, None, None, None)
        .unwrap();
    core.loaded_data_residency[0].insert(
        COMMITTED_POSITION,
        DataResidencyRefs::with_resident_viewers(1),
    );
    let state = ViewerState {
        data_box: single_block_box(COMMITTED_POSITION),
        ..ViewerState::default()
    };
    core.paired_viewers.push(PairedViewer {
        id: 1,
        state: state.clone(),
        prev_state: state,
    });
    assert_eq!(resident_buffer_ptr(&core, COMMITTED_POSITION), payload_ptr);
    (core, payload_ptr)
}

#[derive(Clone, Copy, Debug)]
enum SnapshotRetirementCase {
    Replacement,
    StaleOutput,
    Exit,
}

fn visual_mesh_features() -> MeshBuildFeatures {
    MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    }
}

fn install_clean_data_witness(core: &mut VoxelTerrainCore, mesh_viewed: bool) {
    let voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
    let block = VoxelDataBlock::with_voxels(voxels, 0);
    assert!(core.data.try_set_block(COMMITTED_POSITION, block).unwrap());
    core.data
        .view_area(single_block_box(COMMITTED_POSITION), 0, None, None, None)
        .unwrap();
    core.loaded_data_residency[0].insert(
        COMMITTED_POSITION,
        DataResidencyRefs::with_resident_viewers(1),
    );
    let state = ViewerState {
        data_box: single_block_box(COMMITTED_POSITION),
        mesh_box: if mesh_viewed {
            single_block_box(COMMITTED_POSITION)
        } else {
            Box3i::default()
        },
        demand: MeshDemand {
            visuals: mesh_viewed,
            collisions: false,
        },
        ..ViewerState::default()
    };
    core.paired_viewers.push(PairedViewer {
        id: 18,
        state: state.clone(),
        prev_state: state,
    });
}

fn install_live_mesh_snapshot(
    core: &mut VoxelTerrainCore,
    pool: Arc<MeshArraysPool>,
    revision: u64,
) -> Weak<MeshUploadSnapshot> {
    let features = visual_mesh_features();
    let (upload, dropped) = pooled_mesh_output(
        pool,
        MeshBlockKey {
            location: MeshBlockLocation::new(COMMITTED_POSITION, 0),
            revision,
        },
        features,
        1,
        0,
        false,
    )
    .into_upload()
    .into_parts();
    assert!(!dropped);
    let weak = Arc::downgrade(&upload);
    core.mesh_maps[0].insert(
        COMMITTED_POSITION,
        MeshBlockEntry {
            position: COMMITTED_POSITION,
            resident_viewers: 1,
            visual_viewers: 1,
            visual_active: true,
            is_loaded: true,
            requested_revision: Some(revision),
            requested_features: features,
            applied_revision: Some(revision),
            has_geometry: true,
            accepted_upload: Some(upload),
            ..MeshBlockEntry::default()
        },
    );
    weak
}

fn admitted_direct_upload_weak(core: &VoxelTerrainCore) -> Weak<MeshUploadSnapshot> {
    let DurableCompletion::DirectMesh { upload, .. } = core
        .direct_mesh_retry_inbox
        .front()
        .expect("the admitted output remains in the exact retry owner")
    else {
        unreachable!("the direct retry inbox contains the admitted mesh upload")
    };
    Arc::downgrade(upload)
}

fn assert_live_upload_is(core: &VoxelTerrainCore, expected: &Weak<MeshUploadSnapshot>) {
    let live = core.mesh_maps[0][&COMMITTED_POSITION]
        .accepted_upload()
        .expect("the exact last accepted snapshot remains live");
    assert_eq!(Arc::as_ptr(live), expected.as_ptr());
}

struct RetirementBoundary {
    core: Arc<Mutex<VoxelTerrainCore>>,
    pause: FixedCommitPauseHandle,
    marker: Arc<AtomicBool>,
    data: Arc<SharedVoxelData>,
    transaction: thread::JoinHandle<Result<(), VoxelTerrainRuntimeError>>,
}

fn wait_at_retirement_boundary(core: VoxelTerrainCore) -> RetirementBoundary {
    let data = core.data.clone();
    let core = Arc::new(Mutex::new(core));
    let (pause, marker) = {
        let mut core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let pause = core.install_fixed_commit_pause_for_test(
            FixedCommitPausePhase::AfterWakeBeforeRetirementDrop,
        );
        (pause.clone(), pause.commit_marker())
    };
    let transaction_core = core.clone();
    let transaction = thread::spawn(move || {
        transaction_core
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .try_fixed_viewer_transaction_for_test(&[])
    });
    pause.wait_until_reached();
    RetirementBoundary {
        core,
        pause,
        marker,
        data,
        transaction,
    }
}

#[test]
fn worker_observes_only_committed_fixed_state() {
    let core = Arc::new(Mutex::new(build_core()));
    let weak_core = Arc::downgrade(&core);
    let observer_runs = Arc::new(AtomicUsize::new(0));
    let (observed_tx, observed_rx) = mpsc::channel();
    let (pause, marker, data) = {
        let mut core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        core.task_runner.set_thread_count(1);
        let pause = core.install_fixed_commit_pause_for_test(
            FixedCommitPausePhase::StorageFencedBeforeCorePublish,
        );
        let marker = pause.commit_marker();
        stage_loaded_completion(
            &mut core,
            vec![ScheduledTask::new(
                Box::new(FixedStateObserverTask {
                    core: weak_core,
                    commit_marker: marker.clone(),
                    runs: observer_runs.clone(),
                    observed: Some(observed_tx),
                }),
                TaskLane::Parallel,
            )],
        );
        stage_direct_mesh_acceptance(&mut core);
        (pause, marker, core.data.clone())
    };

    let transaction_core = core.clone();
    let transaction = thread::spawn(move || {
        transaction_core
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .try_fixed_viewer_transaction_for_test(&[])
    });

    pause.wait_until_reached();
    let affected_region = Box3i::new(COMMITTED_POSITION * 16, Vector3i::splat(16));
    assert!(
        data.try_read_region(0, affected_region).is_none(),
        "the affected spatial region remains excluded while storage is fenced"
    );
    assert!(
        data.try_lod_map_read(0).is_none(),
        "the affected map remains excluded while storage is fenced"
    );
    assert!(!marker.load(Ordering::SeqCst));
    assert_eq!(observer_runs.load(Ordering::SeqCst), 0);

    pause.release();
    transaction
        .join()
        .expect("the terrain transaction thread does not panic")
        .expect("the fixed transaction commits");
    let observed = observed_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("the committed follow-up observer runs after wake");
    assert_eq!(observer_runs.load(Ordering::SeqCst), 1);
    assert_eq!(
        observed,
        CommittedFixedObservation {
            commit_marker: true,
            voxel_marker: VOXEL_MARKER,
            data_viewers: 1,
            loading_generation: None,
            next_load_generation: LOAD_GENERATION + 1,
            resident_viewers: 1,
            visual_viewers: 1,
            visual_active: true,
            requested_mesh_revision: Some(MESH_REVISION),
            applied_mesh_revision: Some(MESH_REVISION),
            next_mesh_revision: MESH_REVISION + 1,
            loaded_events: 1,
            mesh_events: 1,
            topology_events: 1,
        }
    );
}

#[test]
fn link_without_wake_and_stale_permit_cannot_cross_commit_boundary() {
    let core = Arc::new(Mutex::new(build_core()));
    let serial_runs = Arc::new(AtomicUsize::new(0));
    let parallel_runs = Arc::new(AtomicUsize::new(0));
    let (blocker_entered_tx, blocker_entered_rx) = mpsc::channel();
    let (blocker_release_tx, blocker_release_rx) = mpsc::channel();
    let (permit_witness_tx, permit_witness_rx) = mpsc::channel();
    let (pause, marker, data, serial_identity, parallel_identity) = {
        let mut core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        core.task_runner.set_thread_count(1);
        core.task_runner.enqueue(ScheduledTask::new(
            Box::new(BlockingPermitTask {
                entered: Some(blocker_entered_tx),
                release: blocker_release_rx,
            }),
            TaskLane::Parallel,
        ));
        blocker_entered_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("the sole worker is held by the blocker");
        core.task_runner.enqueue(ScheduledTask::new(
            Box::new(PermitWitnessTask {
                finished: Some(permit_witness_tx),
            }),
            TaskLane::Parallel,
        ));

        let serial: Box<dyn ThreadedTask> = Box::new(ExactBatchTask {
            name: "C2cC2ExactSerialBatchItem",
            runs: serial_runs.clone(),
        });
        let parallel: Box<dyn ThreadedTask> = Box::new(ExactBatchTask {
            name: "C2cC2ExactParallelBatchItem",
            runs: parallel_runs.clone(),
        });
        let serial_identity = task_identity(serial.as_ref());
        let parallel_identity = task_identity(parallel.as_ref());
        stage_loaded_completion(
            &mut core,
            vec![
                ScheduledTask::new(serial, TaskLane::Serial),
                ScheduledTask::new(parallel, TaskLane::Parallel),
            ],
        );
        let pause =
            core.install_fixed_commit_pause_for_test(FixedCommitPausePhase::BatchLinkedBeforeWake);
        (
            pause.clone(),
            pause.commit_marker(),
            core.data.clone(),
            serial_identity,
            parallel_identity,
        )
    };

    let transaction_core = core.clone();
    let transaction = thread::spawn(move || {
        transaction_core
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .try_fixed_viewer_transaction_for_test(&[])
    });

    pause.wait_until_reached();
    assert!(marker.load(Ordering::SeqCst));
    let affected_region = Box3i::new(COMMITTED_POSITION * 16, Vector3i::splat(16));
    assert!(
        data.try_read_region(0, affected_region).is_some(),
        "storage is readable only after the matching terrain/event commit"
    );
    blocker_release_tx
        .send(())
        .expect("the blocker remains live until the link pause");
    permit_witness_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("the stale permit reaches the already-linked unready batch");
    assert_eq!(serial_runs.load(Ordering::SeqCst), 0);
    assert_eq!(parallel_runs.load(Ordering::SeqCst), 0);

    pause.release();
    transaction
        .join()
        .expect("the terrain transaction thread does not panic")
        .expect("the fixed transaction commits");
    let mut completed = VecDeque::new();
    {
        let mut core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        core.task_runner.wait_for_all_tasks();
        core.task_runner
            .try_drain_completed_into(&mut completed)
            .expect("the completed identity sink reserves");
    }
    assert_eq!(serial_runs.load(Ordering::SeqCst), 1);
    assert_eq!(parallel_runs.load(Ordering::SeqCst), 1);

    let mut exact_items = completed
        .iter()
        .filter_map(|completed| match completed.task().debug_name() {
            "C2cC2ExactSerialBatchItem" | "C2cC2ExactParallelBatchItem" => Some((
                completed.task().debug_name(),
                task_identity(completed.task()),
                completed.lane(),
                completed.status(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    exact_items.sort_unstable_by_key(|item| item.0);
    assert_eq!(
        exact_items,
        vec![
            (
                "C2cC2ExactParallelBatchItem",
                parallel_identity,
                TaskLane::Parallel,
                TaskCompletionStatus::Finished,
            ),
            (
                "C2cC2ExactSerialBatchItem",
                serial_identity,
                TaskLane::Serial,
                TaskCompletionStatus::Finished,
            ),
        ]
    );
}

#[test]
fn fixed_dirty_exit_has_exact_journal_owner_before_storage_becomes_visible() {
    let failures = [
        DirtyExitFailure::TerrainCapacity(FixedCapacityDestination::SaveJournal),
        DirtyExitFailure::TerrainCapacity(FixedCapacityDestination::EventOutbox),
        DirtyExitFailure::TerrainCapacity(FixedCapacityDestination::Retirement),
        DirtyExitFailure::TerrainCapacity(FixedCapacityDestination::PreparedTaskBatch),
        DirtyExitFailure::StorageCapacity(SharedVoxelDataTransactionReservation::LiveKeyRevisions),
        DirtyExitFailure::LiveSpatialRegistry,
    ];
    for failure in failures {
        let stream = IdentitySaveStream::unblocked();
        let (mut core, payload_ptr) = build_dirty_exit_core(stream.clone());
        match failure {
            DirtyExitFailure::TerrainCapacity(destination) => {
                core.fail_fixed_capacity_for_test(destination, 1)
            }
            DirtyExitFailure::StorageCapacity(reservation) => core
                .data
                .set_test_transaction_reservation_failpoint(Some(reservation)),
            DirtyExitFailure::LiveSpatialRegistry => core
                .data
                .set_test_transaction_live_spatial_registry_fail_lod(Some(0)),
        }

        let result = core.try_fixed_viewer_transaction_for_test(&[]);
        match failure {
            DirtyExitFailure::TerrainCapacity(destination) => {
                assert!(matches!(
                    result,
                    Err(VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)
                ));
                assert_eq!(
                    core.last_fixed_capacity_failure_for_test(),
                    Some(destination)
                );
            }
            DirtyExitFailure::StorageCapacity(reservation) => {
                assert!(matches!(
                    result,
                    Err(VoxelTerrainRuntimeError::DataMutation(
                        SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                            reservation: actual,
                        }
                    )) if actual == reservation
                ));
                core.data.set_test_transaction_reservation_failpoint(None);
            }
            DirtyExitFailure::LiveSpatialRegistry => {
                assert!(matches!(
                    result,
                    Err(VoxelTerrainRuntimeError::DataMutation(
                        SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                            reservation: SharedVoxelDataTransactionReservation::LiveSpatialRegistry,
                        }
                    ))
                ));
                core.data
                    .set_test_transaction_live_spatial_registry_fail_lod(None);
            }
        }
        assert_eq!(resident_buffer_ptr(&core, COMMITTED_POSITION), payload_ptr);
        let resident = core
            .data
            .block_snapshot(COMMITTED_POSITION, 0)
            .expect("a failed draft keeps the dirty block resident");
        assert_eq!(resident.viewers.get(), 1);
        assert!(resident.is_modified());
        assert!(core.save_journal.is_empty());
        assert!(core.retained_save_admission_failures.is_empty());
        assert_eq!(core.paired_viewers.len(), 1);
        assert!(core.event_outbox.is_empty());
        assert_eq!(core.pending_task_count(), 0);
        assert_eq!(stream.calls.load(Ordering::SeqCst), 0);
    }

    let (stream, save_started) = IdentitySaveStream::blocked();
    let (core, payload_ptr) = build_dirty_exit_core(stream.clone());
    let data = core.data.clone();
    let core = Arc::new(Mutex::new(core));
    let (pause, marker, journal_owner) = {
        let mut core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let journal_owner = core.install_fixed_dirty_owner_probe_for_test(BlockLocation {
            position: COMMITTED_POSITION,
            lod_index: 0,
        });
        let pause = core.install_fixed_commit_pause_for_test(
            FixedCommitPausePhase::AfterTerrainPublishBeforeFenceFinish,
        );
        (pause.clone(), pause.commit_marker(), journal_owner)
    };
    assert_eq!(journal_owner.load(Ordering::SeqCst), 0);
    let transaction_core = core.clone();
    let transaction = thread::spawn(move || {
        transaction_core
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .try_fixed_viewer_transaction_for_test(&[])
    });

    pause.wait_until_reached();
    assert!(marker.load(Ordering::SeqCst));
    assert_eq!(
        journal_owner.load(Ordering::SeqCst),
        payload_ptr,
        "the exact dirty allocation already has its canonical journal/task owner"
    );
    let affected_region = Box3i::new(COMMITTED_POSITION * 16, Vector3i::splat(16));
    assert!(data.try_read_region(0, affected_region).is_none());
    assert!(data.try_lod_map_read(0).is_none());
    assert_eq!(stream.calls.load(Ordering::SeqCst), 0);

    pause.release();
    transaction
        .join()
        .expect("the dirty-exit transaction thread does not panic")
        .expect("the dirty-exit transaction commits");
    save_started
        .recv_timeout(Duration::from_secs(2))
        .expect("one serial save starts after fence finish and wake");
    assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
    assert_eq!(stream.observed_payload.load(Ordering::SeqCst), payload_ptr);
    {
        let core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(core.data.block_snapshot(COMMITTED_POSITION, 0).is_none());
        assert_eq!(core.save_journal.len(), 1);
        assert!(matches!(
            core.save_journal[&SaveKey::new(COMMITTED_POSITION, 0)].active,
            Some(ActiveSaveAttempt::WriteInFlight { .. })
        ));
        assert_eq!(core.stats.blocks_unloaded, 1);
        assert_eq!(
            core.event_outbox
                .iter()
                .filter(
                    |event| matches!(event, VoxelTerrainEvent::DataBlockUnloaded(location)
                    if *location == BlockLocation { position: COMMITTED_POSITION, lod_index: 0 })
                )
                .count(),
            1
        );
    }

    stream.release();
    let mut completed = VecDeque::new();
    {
        let mut core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        core.task_runner.wait_for_all_tasks();
        core.task_runner
            .try_drain_completed_into(&mut completed)
            .expect("the exact save completion destination reserves");
    }
    assert_eq!(completed.len(), 1);
    let completed = completed.front().unwrap();
    assert_eq!(completed.lane(), TaskLane::Serial);
    assert_eq!(completed.status(), TaskCompletionStatus::Finished);
    let save = completed
        .task_any()
        .downcast_ref::<SaveBlockDataTask>()
        .expect("the only published task is the prepared serial save");
    let terminal = save
        .terminal_ref()
        .expect("the acknowledged task retains its exact terminal owner");
    assert_eq!(
        terminal.location,
        BlockLocation {
            position: COMMITTED_POSITION,
            lod_index: 0,
        }
    );
    assert_eq!(terminal.save_generation, 1);
    assert_eq!(
        terminal
            .payload
            .channel_bytes(ChannelId::Type.index())
            .as_ptr() as usize,
        payload_ptr
    );
}

#[test]
fn late_failure_never_drops_last_snapshot_under_commit_guards() {
    for case in [
        SnapshotRetirementCase::Replacement,
        SnapshotRetirementCase::StaleOutput,
        SnapshotRetirementCase::Exit,
    ] {
        let mut core = build_core();
        core.task_runner.set_thread_count(1);
        install_clean_data_witness(&mut core, matches!(case, SnapshotRetirementCase::Exit));

        let live_pool = Arc::new(MeshArraysPool::new());
        let live_revision = match case {
            SnapshotRetirementCase::Replacement => 180,
            SnapshotRetirementCase::StaleOutput => 280,
            SnapshotRetirementCase::Exit => 380,
        };
        let live = install_live_mesh_snapshot(&mut core, live_pool.clone(), live_revision);
        assert_eq!(live_pool.idle_count(), 0);

        let mut incoming = None;
        let mut incoming_pool = None;
        match case {
            SnapshotRetirementCase::Replacement | SnapshotRetirementCase::StaleOutput => {
                let requested_revision = live_revision + 1;
                core.mesh_maps[0]
                    .get_mut(&COMMITTED_POSITION)
                    .expect("the live mesh entry is installed")
                    .requested_revision = Some(requested_revision);
                let output_revision = match case {
                    SnapshotRetirementCase::Replacement => requested_revision,
                    SnapshotRetirementCase::StaleOutput => live_revision,
                    SnapshotRetirementCase::Exit => unreachable!("handled by the outer match"),
                };
                let pool = Arc::new(MeshArraysPool::new());
                core.fail_fixed_capacity_for_test(FixedCapacityDestination::Retirement, 1);
                let result = core.try_apply_mesh_output(pooled_mesh_output(
                    pool.clone(),
                    MeshBlockKey {
                        location: MeshBlockLocation::new(COMMITTED_POSITION, 0),
                        revision: output_revision,
                    },
                    visual_mesh_features(),
                    2,
                    0,
                    false,
                ));
                assert!(matches!(
                    result,
                    Err(MeshOutputApplyError::Admitted {
                        error: VoxelTerrainRuntimeError::CompletionDrainCapacityFailed,
                    })
                ));
                assert_eq!(
                    core.last_fixed_capacity_failure_for_test(),
                    Some(FixedCapacityDestination::Retirement)
                );
                incoming = Some(admitted_direct_upload_weak(&core));
                incoming_pool = Some(pool);
                assert_eq!(core.direct_mesh_retry_inbox.len(), 1);
            }
            SnapshotRetirementCase::Exit => {
                core.fail_fixed_capacity_for_test(FixedCapacityDestination::Retirement, 1);
                assert!(matches!(
                    core.try_fixed_viewer_transaction_for_test(&[]),
                    Err(VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)
                ));
                assert_eq!(
                    core.last_fixed_capacity_failure_for_test(),
                    Some(FixedCapacityDestination::Retirement)
                );
                assert!(core.direct_mesh_retry_inbox.is_empty());
            }
        }

        assert_live_upload_is(&core, &live);
        assert!(live.upgrade().is_some());
        assert_eq!(live_pool.idle_count(), 0);
        if let (Some(incoming), Some(pool)) = (&incoming, &incoming_pool) {
            assert!(incoming.upgrade().is_some());
            assert_eq!(pool.idle_count(), 0);
        }
        assert!(core.data.block_snapshot(COMMITTED_POSITION, 0).is_some());
        assert_eq!(core.paired_viewers.len(), 1);
        assert!(core.event_outbox.is_empty());

        let RetirementBoundary {
            core,
            pause,
            marker,
            data,
            transaction,
        } = wait_at_retirement_boundary(core);
        assert!(
            marker.load(Ordering::SeqCst),
            "terrain and event publication precede snapshot retirement"
        );
        let affected_region = Box3i::new(COMMITTED_POSITION * 16, Vector3i::splat(16));
        assert!(
            data.try_read_region(0, affected_region).is_some(),
            "the affected spatial fence is released before snapshot retirement"
        );
        assert!(
            data.try_lod_map_read(0).is_some(),
            "the live map guard is released before snapshot retirement"
        );

        let retiring = match case {
            SnapshotRetirementCase::Replacement | SnapshotRetirementCase::Exit => &live,
            SnapshotRetirementCase::StaleOutput => incoming
                .as_ref()
                .expect("the stale output has one retirement owner"),
        };
        assert!(
            retiring.upgrade().is_some(),
            "the relevant snapshot remains owned at the post-wake boundary"
        );
        assert_eq!(live_pool.idle_count(), 0);
        if let Some(pool) = &incoming_pool {
            assert_eq!(pool.idle_count(), 0);
        }

        pause.release();
        transaction
            .join()
            .expect("the snapshot-retirement transaction thread does not panic")
            .expect("the retry transaction commits");

        match case {
            SnapshotRetirementCase::Replacement => {
                assert!(live.upgrade().is_none());
                assert_eq!(live_pool.idle_count(), 1);
                let incoming = incoming.as_ref().unwrap();
                let pool = incoming_pool.as_ref().unwrap();
                assert!(incoming.upgrade().is_some());
                assert_eq!(pool.idle_count(), 0);
                let core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                assert_live_upload_is(&core, incoming);
                assert!(core.direct_mesh_retry_inbox.is_empty());
            }
            SnapshotRetirementCase::StaleOutput => {
                assert!(live.upgrade().is_some());
                assert_eq!(live_pool.idle_count(), 0);
                let incoming = incoming.as_ref().unwrap();
                let pool = incoming_pool.as_ref().unwrap();
                assert!(incoming.upgrade().is_none());
                assert_eq!(pool.idle_count(), 1);
                let core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                assert_live_upload_is(&core, &live);
                assert!(core.direct_mesh_retry_inbox.is_empty());
            }
            SnapshotRetirementCase::Exit => {
                assert!(live.upgrade().is_none());
                assert_eq!(live_pool.idle_count(), 1);
                let core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                assert!(!core.mesh_maps[0].contains_key(&COMMITTED_POSITION));
            }
        }
        let core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(core.data.block_snapshot(COMMITTED_POSITION, 0).is_none());
    }
}
