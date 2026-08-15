use super::*;
use crate::storage::voxel_data::SharedVoxelDataTransactionReservation;
use crate::storage::VoxelDataKeyRevision;
use crate::tasks::threaded_task_runner::RunnerTaskObservable;
use crate::tasks::TaskPanicPhase;
use std::sync::atomic::AtomicUsize;

#[derive(Debug, Clone, PartialEq, Eq)]
struct StorageObservation {
    location: BlockLocation,
    present: bool,
    key_revision: VoxelDataKeyRevision,
    viewers: u32,
    modified: bool,
    edited: bool,
    voxel_buffer: usize,
    voxel_allocation: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LoadingObservation {
    position: Vector3i,
    viewers: u32,
    coverage_holds: u32,
    retry_count: u32,
    generation: u64,
    state: LoadRequestState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MeshObservation {
    position: Vector3i,
    resident_viewers: u32,
    visual_viewers: u32,
    collision_viewers: u32,
    visual_coverage_holds: u32,
    collision_coverage_holds: u32,
    visual_active: bool,
    collision_active: bool,
    loaded: bool,
    requested_revision: Option<u64>,
    requested_features: MeshBuildFeatures,
    applied_features: MeshBuildFeatures,
    applied_revision: Option<u64>,
    has_geometry: bool,
    in_update_list: bool,
    terminal_retry_count: u32,
    accepted_upload: usize,
    accepted_key: Option<MeshBlockKey>,
    renderer_payload: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TaskOwnerObservation {
    task: usize,
    lane: TaskLane,
    status: TaskCompletionStatus,
    followups: Vec<(usize, TaskLane)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DurableOwnerObservation {
    kind: CompletionTaskKind,
    owner: TaskOwnerObservation,
    payload: usize,
    upload: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QuarantineTerminalObservation {
    Save {
        location: BlockLocation,
        generation: u64,
        payload: usize,
        phase: PersistenceIoPhase,
        acknowledgement: Option<String>,
    },
    Flush {
        generation: u64,
        phase: PersistenceIoPhase,
        acknowledgement: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QuarantineObservation {
    Persistence {
        malformed: bool,
        kind: PersistenceTaskKind,
        attempt: u64,
        terminal: QuarantineTerminalObservation,
        owner: TaskOwnerObservation,
    },
    Other {
        kind: CompletionTaskKind,
        owner: TaskOwnerObservation,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JournalActiveObservation {
    Pending {
        generation: u64,
        retry_count: u32,
        error: Option<String>,
        payload: usize,
    },
    WriteInFlight {
        generation: u64,
        retry_count: u32,
        error: Option<String>,
        attempt: u64,
    },
    Indeterminate {
        generation: u64,
        retry_count: u32,
        error: Option<String>,
        attempt: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct JournalObservation {
    key: SaveKey,
    written: Option<(u64, usize)>,
    active: Option<JournalActiveObservation>,
    queued: Vec<(u64, u32, Option<String>, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CheckpointObservation {
    generation: u64,
    acknowledged: Vec<(SaveKey, u64)>,
    state: CheckpointAttemptState,
    retry_count: u32,
    max_attempts: u32,
    origin: CheckpointOrigin,
    record_per_block_failure: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RetainedSaveObservation {
    error: String,
    location: BlockLocation,
    payload: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EventObservation {
    DataLoaded(BlockLocation),
    DataUnloaded(BlockLocation),
    Mesh {
        descriptor: MeshLifecycleEventDescriptor,
        upload: usize,
        key: MeshBlockKey,
    },
    MeshExited(MeshBlockLocation),
    Topology(RenderTopologyBatch),
}

#[derive(Debug, Clone, PartialEq)]
struct FixedObservableSnapshot {
    paired_viewers: Vec<PairedViewer>,
    settings_revision: u64,
    settings_bounds: Box3i,
    settings_streaming: bool,
    settings_loaded: bool,
    generator: usize,
    stream: usize,
    storage: Vec<StorageObservation>,
    data_residency: Vec<(BlockLocation, DataResidencyRefs)>,
    loading: Vec<LoadingObservation>,
    meshes: Vec<MeshObservation>,
    pools: Vec<(usize, usize, usize)>,
    pending_load: Vec<Vector3i>,
    pending_mesh: Vec<Vector3i>,
    view_retries: Vec<PendingDataMutation>,
    unview_retries: Vec<PendingDataMutation>,
    last_data_mutation_error: Option<SharedVoxelDataMutationError>,
    retained_save_admission_failures: Vec<RetainedSaveObservation>,
    deferred_save_keys: Vec<SaveKey>,
    deferred_checkpoint: bool,
    next_load_generation: u64,
    next_mesh_revision: u64,
    next_topology_revision: u64,
    next_save_generation: u64,
    next_checkpoint_generation: u64,
    next_attempt: u64,
    journal: Vec<JournalObservation>,
    checkpoint: Option<CheckpointObservation>,
    save_dispatch_error: Option<String>,
    checkpoint_error: Option<String>,
    last_checkpoint_outcome: Option<String>,
    automatic_checkpoint_blocked: bool,
    force_checkpoint_requested: bool,
    automatic_empty_checkpoint_satisfied: bool,
    stats: VoxelTerrainStats,
    raw: Vec<TaskOwnerObservation>,
    durable: Vec<DurableOwnerObservation>,
    direct: Vec<(usize, MeshBlockKey, bool)>,
    quarantine: Vec<QuarantineObservation>,
    events: Vec<EventObservation>,
    runner_tasks: Vec<RunnerTaskObservable>,
    runner_remaining: usize,
    terrain_pending: usize,
    started: Vec<usize>,
    commit_marker: bool,
    shutdown: bool,
    shutdown_in_progress: bool,
}

impl FixedObservableSnapshot {
    fn capture(
        core: &VoxelTerrainCore,
        tracked_storage: &[BlockLocation],
        tracked_pools: &[Arc<MeshArraysPool>],
        started: &[Arc<AtomicUsize>],
        commit_marker: Option<&AtomicBool>,
    ) -> Self {
        let settings = core.data.settings_snapshot();
        let mut storage = tracked_storage
            .iter()
            .copied()
            .map(|location| {
                core.data
                    .with_lod_map(usize::from(location.lod_index), |map| {
                        let revision = map.key_revision(location.position);
                        let block = map.get_block(location.position);
                        StorageObservation {
                            location,
                            present: block.is_some(),
                            key_revision: if block.is_some() {
                                VoxelDataKeyRevision::Present(revision)
                            } else {
                                VoxelDataKeyRevision::Tombstone(revision)
                            },
                            viewers: block.map_or(0, |block| block.viewers.get()),
                            modified: block.is_some_and(VoxelDataBlock::is_modified),
                            edited: block.is_some_and(VoxelDataBlock::is_edited),
                            voxel_buffer: block
                                .filter(|block| block.has_voxels())
                                .map_or(0, |block| block.voxels() as *const VoxelBuffer as usize),
                            voxel_allocation: block
                                .filter(|block| block.has_voxels())
                                .map_or(0, |block| voxel_allocation_identity(block.voxels())),
                        }
                    })
            })
            .collect::<Vec<_>>();
        storage.sort_unstable_by_key(|item| {
            (
                item.location.lod_index,
                item.location.position.x,
                item.location.position.y,
                item.location.position.z,
            )
        });
        storage.dedup_by_key(|item| item.location);

        let mut loading = core.loading_blocks[0]
            .iter()
            .map(|(position, entry)| LoadingObservation {
                position: *position,
                viewers: entry.residency.resident_viewers,
                coverage_holds: entry.residency.coverage_holds,
                retry_count: entry.retry_count,
                generation: entry.request_generation,
                state: entry.request_state,
            })
            .collect::<Vec<_>>();
        loading.sort_unstable_by_key(|item| (item.position.x, item.position.y, item.position.z));

        let mut meshes = core.mesh_maps[0]
            .iter()
            .map(|(position, entry)| MeshObservation {
                position: *position,
                resident_viewers: entry.resident_viewers,
                visual_viewers: entry.visual_viewers,
                collision_viewers: entry.collision_viewers,
                visual_coverage_holds: entry.visual_coverage_holds,
                collision_coverage_holds: entry.collision_coverage_holds,
                visual_active: entry.visual_active,
                collision_active: entry.collision_active,
                loaded: entry.is_loaded,
                requested_revision: entry.requested_revision,
                requested_features: entry.requested_features,
                applied_features: entry.applied_features,
                applied_revision: entry.applied_revision,
                has_geometry: entry.has_geometry,
                in_update_list: entry.is_in_update_list,
                terminal_retry_count: entry.terminal_retry_count,
                accepted_upload: entry
                    .accepted_upload()
                    .map_or(0, |upload| Arc::as_ptr(upload) as usize),
                accepted_key: entry.accepted_upload().map(|upload| upload.key()),
                renderer_payload: entry
                    .accepted_upload()
                    .map_or(0, |upload| upload.output().surfaces.as_ptr() as usize),
            })
            .collect::<Vec<_>>();
        meshes.sort_unstable_by_key(|item| (item.position.x, item.position.y, item.position.z));

        let mut data_residency = core
            .loaded_data_residency
            .iter()
            .enumerate()
            .flat_map(|(lod_index, entries)| {
                entries.iter().map(move |(position, residency)| {
                    (
                        BlockLocation {
                            position: *position,
                            lod_index: lod_index as u8,
                        },
                        *residency,
                    )
                })
            })
            .collect::<Vec<_>>();
        data_residency.sort_unstable_by_key(|(location, _)| {
            (
                location.lod_index,
                location.position.z,
                location.position.x,
                location.position.y,
            )
        });

        let mut journal = core
            .save_journal
            .iter()
            .map(|(key, entry)| JournalObservation {
                key: *key,
                written: entry.written_unflushed.as_ref().map(|written| {
                    (
                        written.generation,
                        voxel_allocation_identity(&written.payload),
                    )
                }),
                active: entry.active.as_ref().map(|active| match active {
                    ActiveSaveAttempt::Pending(pending) => JournalActiveObservation::Pending {
                        generation: pending.meta.generation,
                        retry_count: pending.meta.retry_count,
                        error: pending
                            .meta
                            .last_error
                            .as_ref()
                            .map(|error| format!("{error:?}")),
                        payload: voxel_allocation_identity(&pending.payload),
                    },
                    ActiveSaveAttempt::WriteInFlight {
                        meta,
                        attempt_ordinal,
                    } => JournalActiveObservation::WriteInFlight {
                        generation: meta.generation,
                        retry_count: meta.retry_count,
                        error: meta.last_error.as_ref().map(|error| format!("{error:?}")),
                        attempt: *attempt_ordinal,
                    },
                    ActiveSaveAttempt::Indeterminate {
                        meta,
                        attempt_ordinal,
                    } => JournalActiveObservation::Indeterminate {
                        generation: meta.generation,
                        retry_count: meta.retry_count,
                        error: meta.last_error.as_ref().map(|error| format!("{error:?}")),
                        attempt: *attempt_ordinal,
                    },
                }),
                queued: entry
                    .queued_newer
                    .iter()
                    .map(|pending| {
                        (
                            pending.meta.generation,
                            pending.meta.retry_count,
                            pending
                                .meta
                                .last_error
                                .as_ref()
                                .map(|error| format!("{error:?}")),
                            voxel_allocation_identity(&pending.payload),
                        )
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        journal.sort_unstable_by_key(|item| {
            (
                item.key.lod_index,
                item.key.position.x,
                item.key.position.y,
                item.key.position.z,
            )
        });

        Self {
            paired_viewers: core.paired_viewers.clone(),
            settings_revision: settings.revision,
            settings_bounds: settings.bounds_in_voxels,
            settings_streaming: settings.streaming_enabled,
            settings_loaded: settings.full_load_completed,
            generator: settings
                .generator
                .as_ref()
                .map_or(0, |value| Arc::as_ptr(value) as *const () as usize),
            stream: settings
                .stream
                .as_ref()
                .map_or(0, |value| Arc::as_ptr(value) as *const () as usize),
            storage,
            data_residency,
            loading,
            meshes,
            pools: tracked_pools
                .iter()
                .map(|pool| {
                    (
                        Arc::as_ptr(pool) as usize,
                        pool.idle_count(),
                        Arc::strong_count(pool),
                    )
                })
                .collect(),
            pending_load: core.blocks_pending_load[0].clone(),
            pending_mesh: core.blocks_pending_update[0].clone(),
            view_retries: core.data_view_retries[0].clone(),
            unview_retries: core.data_unview_retries[0].clone(),
            last_data_mutation_error: core.last_data_mutation_error.clone(),
            retained_save_admission_failures: core
                .retained_save_admission_failures
                .iter()
                .map(|failure| RetainedSaveObservation {
                    error: format!("{:?}", failure.error),
                    location: BlockLocation {
                        position: failure.save.position,
                        lod_index: failure.save.lod_index,
                    },
                    payload: failure
                        .save
                        .voxels
                        .as_ref()
                        .map_or(0, voxel_allocation_identity),
                })
                .collect(),
            deferred_save_keys: core.deferred_save_dispatch_keys.clone(),
            deferred_checkpoint: core.deferred_checkpoint_dispatch,
            next_load_generation: core.next_request_generation,
            next_mesh_revision: core.next_mesh_revision,
            next_topology_revision: core.next_render_topology_revision,
            next_save_generation: core.next_save_generation,
            next_checkpoint_generation: core.next_save_checkpoint_generation,
            next_attempt: core.next_persistence_attempt_ordinal,
            journal,
            checkpoint: core.save_checkpoint_in_flight.as_ref().map(|checkpoint| {
                CheckpointObservation {
                    generation: checkpoint.checkpoint_generation,
                    acknowledged: checkpoint
                        .acknowledged
                        .iter()
                        .map(|snapshot| (snapshot.key, snapshot.generation))
                        .collect(),
                    state: checkpoint.state,
                    retry_count: checkpoint.retry_count,
                    max_attempts: checkpoint.max_attempts,
                    origin: checkpoint.origin,
                    record_per_block_failure: checkpoint.record_per_block_failure,
                }
            }),
            save_dispatch_error: core
                .save_dispatch_error
                .as_ref()
                .map(|error| format!("{error:?}")),
            checkpoint_error: core
                .last_save_checkpoint_error
                .as_ref()
                .map(|error| format!("{error:?}")),
            last_checkpoint_outcome: core
                .last_checkpoint_outcome
                .as_ref()
                .map(|outcome| format!("{outcome:?}")),
            automatic_checkpoint_blocked: core.automatic_save_checkpoint_blocked,
            force_checkpoint_requested: core.force_checkpoint_requested,
            automatic_empty_checkpoint_satisfied: core.automatic_checkpoint_satisfied_empty_flush,
            stats: core.stats,
            raw: core
                .raw_completion_inbox
                .iter()
                .map(task_owner_observation)
                .collect(),
            durable: core
                .durable_completion_inbox
                .iter()
                .map(durable_owner_observation)
                .collect(),
            direct: core
                .direct_mesh_retry_inbox
                .iter()
                .map(|completion| {
                    let DurableCompletion::DirectMesh { upload, dropped } = completion else {
                        panic!("direct FIFO changed variant")
                    };
                    (Arc::as_ptr(upload) as usize, upload.key(), *dropped)
                })
                .collect(),
            quarantine: core
                .completion_quarantine
                .iter()
                .map(quarantine_observation)
                .collect(),
            events: core.event_outbox.iter().map(event_observation).collect(),
            runner_tasks: core.task_runner.observable_tasks_for_test(),
            runner_remaining: core.task_runner.remaining_task_count(),
            terrain_pending: core.pending_task_count(),
            started: started
                .iter()
                .map(|counter| counter.load(Ordering::SeqCst))
                .collect(),
            commit_marker: commit_marker.is_some_and(|marker| marker.load(Ordering::SeqCst)),
            shutdown: core.shut_down,
            shutdown_in_progress: core.shutdown_in_progress,
        }
    }
}

fn voxel_allocation_identity(voxels: &VoxelBuffer) -> usize {
    (0..MAX_CHANNELS)
        .find_map(|channel| {
            let bytes = voxels.channel_bytes(channel);
            (!bytes.is_empty()).then_some(bytes.as_ptr() as usize)
        })
        .unwrap_or(voxels as *const VoxelBuffer as usize)
}

fn task_owner_observation(completed: &CompletedTask) -> TaskOwnerObservation {
    TaskOwnerObservation {
        task: completed.task() as *const dyn ThreadedTask as *const () as usize,
        lane: completed.lane(),
        status: completed.status(),
        followups: (0..completed.follow_up_count())
            .map(|index| {
                let task = completed.follow_up_task(index).unwrap();
                (
                    task.task() as *const dyn ThreadedTask as *const () as usize,
                    task.lane(),
                )
            })
            .collect(),
    }
}

fn completed_for_durable(completion: &DurableCompletion) -> Option<&CompletedTask> {
    match completion {
        DurableCompletion::LoadFinished { completed, .. }
        | DurableCompletion::LoadTerminal { completed, .. }
        | DurableCompletion::MeshFinished { completed, .. }
        | DurableCompletion::MeshTerminal { completed, .. }
        | DurableCompletion::SaveAcknowledged { completed, .. }
        | DurableCompletion::FlushAcknowledged { completed, .. }
        | DurableCompletion::PersistenceTerminal { completed, .. }
        | DurableCompletion::MalformedPersistence { completed, .. }
        | DurableCompletion::MalformedFinished { completed, .. }
        | DurableCompletion::UnknownTerminal { completed } => Some(completed),
        DurableCompletion::DirectMesh { .. } => None,
    }
}

fn durable_owner_observation(completion: &DurableCompletion) -> DurableOwnerObservation {
    let completed = completed_for_durable(completion).expect("durable prefix owns a task box");
    let (payload, upload) = match completion {
        DurableCompletion::LoadFinished { output, .. } => (
            output
                .block_data
                .voxels
                .as_ref()
                .map_or(0, voxel_allocation_identity),
            0,
        ),
        DurableCompletion::MeshFinished { output, .. } => {
            (0, Arc::as_ptr(output.upload()) as usize)
        }
        DurableCompletion::SaveAcknowledged { terminal, .. }
        | DurableCompletion::PersistenceTerminal {
            terminal: PersistenceTaskTerminal::Save(terminal),
            ..
        }
        | DurableCompletion::MalformedPersistence {
            terminal: PersistenceTaskTerminal::Save(terminal),
            ..
        } => (voxel_allocation_identity(&terminal.payload), 0),
        DurableCompletion::LoadTerminal { .. }
        | DurableCompletion::MeshTerminal { .. }
        | DurableCompletion::FlushAcknowledged { .. }
        | DurableCompletion::PersistenceTerminal {
            terminal: PersistenceTaskTerminal::Flush(_),
            ..
        }
        | DurableCompletion::MalformedPersistence {
            terminal: PersistenceTaskTerminal::Flush(_),
            ..
        }
        | DurableCompletion::MalformedFinished { .. }
        | DurableCompletion::UnknownTerminal { .. } => (0, 0),
        DurableCompletion::DirectMesh { .. } => unreachable!(),
    };
    DurableOwnerObservation {
        kind: completion.descriptor().kind,
        owner: task_owner_observation(completed),
        payload,
        upload,
    }
}

fn quarantine_terminal_observation(
    terminal: &PersistenceTaskTerminal,
) -> QuarantineTerminalObservation {
    match terminal {
        PersistenceTaskTerminal::Save(terminal) => QuarantineTerminalObservation::Save {
            location: terminal.location,
            generation: terminal.save_generation,
            payload: voxel_allocation_identity(&terminal.payload),
            phase: terminal.phase,
            acknowledgement: terminal
                .acknowledgement
                .as_ref()
                .map(|acknowledgement| format!("{acknowledgement:?}")),
        },
        PersistenceTaskTerminal::Flush(terminal) => QuarantineTerminalObservation::Flush {
            generation: terminal.checkpoint_generation,
            phase: terminal.phase,
            acknowledgement: terminal
                .acknowledgement
                .as_ref()
                .map(|acknowledgement| format!("{acknowledgement:?}")),
        },
    }
}

fn quarantine_observation(completion: &QuarantinedCompletion) -> QuarantineObservation {
    match completion {
        QuarantinedCompletion::Persistence {
            kind,
            terminal,
            attempt_ordinal,
            completed,
        } => QuarantineObservation::Persistence {
            malformed: false,
            kind: *kind,
            attempt: *attempt_ordinal,
            terminal: quarantine_terminal_observation(terminal),
            owner: task_owner_observation(completed),
        },
        QuarantinedCompletion::MalformedPersistence {
            kind,
            terminal,
            attempt_ordinal,
            completed,
        } => QuarantineObservation::Persistence {
            malformed: true,
            kind: *kind,
            attempt: *attempt_ordinal,
            terminal: quarantine_terminal_observation(terminal),
            owner: task_owner_observation(completed),
        },
        QuarantinedCompletion::Other { kind, completed } => QuarantineObservation::Other {
            kind: *kind,
            owner: task_owner_observation(completed),
        },
    }
}

fn event_observation(event: &VoxelTerrainEvent) -> EventObservation {
    match event {
        VoxelTerrainEvent::MeshBlockEntered(upload)
        | VoxelTerrainEvent::MeshBlockUpdated(upload)
        | VoxelTerrainEvent::MeshBlockBecameEmpty(upload) => EventObservation::Mesh {
            descriptor: event.mesh_descriptor().unwrap(),
            upload: Arc::as_ptr(upload) as usize,
            key: upload.key(),
        },
        VoxelTerrainEvent::DataBlockLoaded(position) => EventObservation::DataLoaded(*position),
        VoxelTerrainEvent::DataBlockUnloaded(position) => EventObservation::DataUnloaded(*position),
        VoxelTerrainEvent::MeshBlockExited(location) => EventObservation::MeshExited(*location),
        VoxelTerrainEvent::RenderTopologyChanged(batch) => {
            EventObservation::Topology(batch.clone())
        }
    }
}

fn viewer_state(core: &VoxelTerrainCore, update: ViewerUpdate) -> ViewerState {
    let mut state = ViewerState {
        local_position_voxels: update.world_position_voxels,
        horizontal_view_distance_voxels: update.horizontal_view_distance_voxels,
        vertical_view_distance_voxels: update.vertical_view_distance_voxels,
        demand: update.demand,
        ..ViewerState::default()
    };
    compute_viewer_boxes(&mut state, core.data_block_size(), core.data_block_size());
    state
}

fn locations_in(box_in_blocks: Box3i) -> Vec<BlockLocation> {
    box_in_blocks
        .iter_cells_zxy()
        .map(|position| BlockLocation {
            position,
            lod_index: 0,
        })
        .collect()
}

fn unique_positions(values: &[Vector3i]) -> bool {
    let mut values = values.to_vec();
    values.sort_unstable_by_key(|position| (position.x, position.y, position.z));
    let len = values.len();
    values.dedup();
    values.len() == len
}

#[test]
fn fixed_mixed_enter_exit_load_generation_overflow_rolls_back_then_retries_once() {
    let mut core = build_core();
    let old = fixed_zero_distance_viewer(101, Vector3i::zero(), MeshDemand::default());
    let new = fixed_zero_distance_viewer(102, Vector3i::new(16, 0, 0), MeshDemand::default());
    let old_state = viewer_state(&core, old);
    let new_state = viewer_state(&core, new);
    assert!(!old_state.data_box.intersects(&new_state.data_box));
    core.prepare_fixed_viewer_transaction(&[old], true, false, false)
        .unwrap();
    let mut tracked = locations_in(old_state.data_box);
    tracked.extend(locations_in(new_state.data_box));
    core.next_request_generation = u64::MAX;
    let before = FixedObservableSnapshot::capture(&core, &tracked, &[], &[], None);

    assert!(matches!(
        core.prepare_fixed_viewer_transaction(&[new], true, false, false),
        Err(VoxelTerrainRuntimeError::RequestGenerationOverflow)
    ));
    assert_eq!(
        FixedObservableSnapshot::capture(&core, &tracked, &[], &[], None),
        before
    );

    core.next_request_generation = 100;
    core.prepare_fixed_viewer_transaction(&[new], true, false, false)
        .unwrap();
    assert_eq!(core.paired_viewers.len(), 1);
    assert_eq!(core.paired_viewers[0].id, new.id);
    assert_eq!(core.paired_viewers[0].state, new_state);
    assert_eq!(core.next_request_generation, 164);
    let mut new_positions = new_state.data_box.iter_cells_zxy().collect::<Vec<_>>();
    new_positions.sort_unstable_by_key(|position| (position.x, position.y, position.z));
    assert_eq!(new_positions.len(), 64);
    assert!(unique_positions(&new_positions));
    for (index, position) in new_positions.into_iter().enumerate() {
        let entry = &core.loading_blocks[0][&position];
        assert_eq!(entry.residency.resident_viewers, 1);
        assert_eq!(entry.retry_count, 0);
        assert_eq!(entry.request_generation, 100 + index as u64);
        assert_eq!(entry.request_state, LoadRequestState::Queued);
    }
    for position in old_state.data_box.iter_cells_zxy() {
        assert!(!core.loading_blocks[0].contains_key(&position));
    }
    assert_eq!(core.blocks_pending_load[0].len(), 64);
    assert!(unique_positions(&core.blocks_pending_load[0]));
    assert_eq!(core.task_runner.remaining_task_count(), 0);
    assert!(core.event_outbox.is_empty());
}

fn tagged_block(marker: u64, viewers: u32) -> VoxelDataBlock {
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(16));
    voxels.set_voxel(marker, 0, 0, 0, ChannelId::Type.index());
    let mut block = VoxelDataBlock::with_voxels(voxels, 0);
    block.viewers.set_exact(viewers);
    block
}

fn record_test_storage_residency(
    core: &mut VoxelTerrainCore,
    position: Vector3i,
    resident_viewers: u32,
) {
    core.loaded_data_residency[0].insert(
        position,
        DataResidencyRefs::with_resident_viewers(resident_viewers),
    );
}

#[test]
fn fixed_mixed_enter_exit_mesh_revision_overflow_rolls_back_then_retries_once() {
    let mut core = build_core();
    let demand = MeshDemand {
        visuals: true,
        collisions: false,
    };
    let old = fixed_zero_distance_viewer(111, Vector3i::zero(), demand);
    let new = fixed_zero_distance_viewer(112, Vector3i::new(16, 0, 0), demand);
    let old_state = viewer_state(&core, old);
    let new_state = viewer_state(&core, new);
    assert!(!old_state.data_box.intersects(&new_state.data_box));
    let mut tracked = locations_in(old_state.data_box);
    tracked.extend(locations_in(new_state.data_box));
    for (index, location) in tracked.iter().copied().enumerate() {
        let viewers = u32::from(old_state.data_box.contains_point(location.position));
        assert!(core
            .data
            .try_set_block(location.position, tagged_block(index as u64 + 1, viewers))
            .unwrap());
        record_test_storage_residency(&mut core, location.position, viewers);
    }
    core.paired_viewers.push(PairedViewer {
        id: old.id,
        state: old_state.clone(),
        prev_state: old_state.clone(),
    });
    for position in old_state.mesh_box.iter_cells_zxy() {
        core.mesh_maps[0].insert(
            position,
            MeshBlockEntry {
                position,
                resident_viewers: 1,
                visual_viewers: 1,
                requested_features: MeshBuildFeatures {
                    visuals: demand.visuals,
                    collisions: demand.collisions,
                    variable_lod: false,
                },
                ..MeshBlockEntry::default()
            },
        );
    }
    core.next_mesh_revision = u64::MAX;
    let before = FixedObservableSnapshot::capture(&core, &tracked, &[], &[], None);

    assert!(matches!(
        core.prepare_fixed_viewer_transaction(&[new], true, false, false),
        Err(VoxelTerrainRuntimeError::MeshRevisionOverflow)
    ));
    assert_eq!(
        FixedObservableSnapshot::capture(&core, &tracked, &[], &[], None),
        before
    );

    core.next_mesh_revision = 100;
    core.prepare_fixed_viewer_transaction(&[new], true, false, false)
        .unwrap();
    assert_eq!(core.next_mesh_revision, 108);
    let mut new_positions = new_state.mesh_box.iter_cells_zxy().collect::<Vec<_>>();
    new_positions.sort_unstable_by_key(|position| (position.x, position.y, position.z));
    assert_eq!(new_positions.len(), 8);
    for (index, position) in new_positions.into_iter().enumerate() {
        let entry = &core.mesh_maps[0][&position];
        assert_eq!(entry.resident_viewers, 1);
        assert_eq!(entry.visual_viewers, 1);
        assert_eq!(entry.collision_viewers, 0);
        assert_eq!(entry.requested_revision, Some(100 + index as u64));
        assert!(entry.is_in_update_list);
    }
    for position in old_state.mesh_box.iter_cells_zxy() {
        assert!(!core.mesh_maps[0].contains_key(&position));
    }
    assert_eq!(core.blocks_pending_update[0].len(), 8);
    assert!(unique_positions(&core.blocks_pending_update[0]));
    assert_eq!(core.task_runner.remaining_task_count(), 0);
}

#[derive(Debug, Clone, Copy)]
enum StoragePreflightFailure {
    ViewerOverflow,
    ViewerUnderflow,
    KeyRevisionOverflow,
    SettingsConflict,
    KeyConflict,
    Reservation(SharedVoxelDataTransactionReservation),
    LiveSpatialRegistry,
}

fn install_finished_load(
    core: &mut VoxelTerrainCore,
    position: Vector3i,
    generation: u64,
    marker: u64,
) {
    core.loading_blocks[0].insert(
        position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(0),
            retry_count: 0,
            request_generation: generation,
            request_state: LoadRequestState::InFlight,
            physical_request: None,
        },
    );
    let mut task = LoadBlockForTerrainTask::new(
        position,
        0,
        generation,
        core.data.clone(),
        core.stream.clone(),
    );
    task.output = Some(loaded_output(core, position, generation, marker));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));
    core.try_normalize_raw_completions().unwrap();
}

#[test]
fn fixed_mixed_enter_exit_storage_preflight_failure_rolls_back() {
    let failures = [
        StoragePreflightFailure::ViewerOverflow,
        StoragePreflightFailure::ViewerUnderflow,
        StoragePreflightFailure::KeyRevisionOverflow,
        StoragePreflightFailure::SettingsConflict,
        StoragePreflightFailure::KeyConflict,
        StoragePreflightFailure::Reservation(SharedVoxelDataTransactionReservation::LiveMap),
        StoragePreflightFailure::Reservation(
            SharedVoxelDataTransactionReservation::LiveKeyRevisions,
        ),
        StoragePreflightFailure::LiveSpatialRegistry,
    ];

    for failure in failures {
        let mut core = build_core();
        let old = fixed_zero_distance_viewer(121, Vector3i::zero(), MeshDemand::default());
        let new = fixed_zero_distance_viewer(122, Vector3i::new(16, 0, 0), MeshDemand::default());
        let old_state = viewer_state(&core, old);
        let new_state = viewer_state(&core, new);
        assert!(!old_state.data_box.intersects(&new_state.data_box));
        let old_failure_position = old_state.data_box.position;
        let accepted_position = new_state.data_box.position;
        let mut tracked = locations_in(old_state.data_box);
        tracked.extend(locations_in(new_state.data_box));
        for (index, position) in old_state.data_box.iter_cells_zxy().enumerate() {
            assert!(core
                .data
                .try_set_block(position, tagged_block(index as u64 + 201, 1))
                .unwrap());
            record_test_storage_residency(&mut core, position, 1);
        }
        core.paired_viewers.push(PairedViewer {
            id: old.id,
            state: old_state.clone(),
            prev_state: old_state.clone(),
        });
        for position in old_state.mesh_box.iter_cells_zxy() {
            core.mesh_maps[0].insert(
                position,
                MeshBlockEntry {
                    position,
                    resident_viewers: 1,
                    ..MeshBlockEntry::default()
                },
            );
        }

        let uses_finished_load = !matches!(
            failure,
            StoragePreflightFailure::ViewerOverflow | StoragePreflightFailure::ViewerUnderflow
        );
        if uses_finished_load {
            install_finished_load(&mut core, accepted_position, 73, 0xC203);
        }
        match failure {
            StoragePreflightFailure::ViewerOverflow => {
                assert!(core
                    .data
                    .try_set_block(accepted_position, tagged_block(0xC204, u32::MAX))
                    .unwrap());
                record_test_storage_residency(&mut core, accepted_position, u32::MAX);
            }
            StoragePreflightFailure::ViewerUnderflow => {
                core.data.with_lod_map_mut(0, |map| {
                    map.get_block_mut(old_failure_position)
                        .unwrap()
                        .viewers
                        .set_exact(0);
                });
                record_test_storage_residency(&mut core, old_failure_position, 0);
            }
            StoragePreflightFailure::KeyRevisionOverflow => {
                core.data.with_lod_map_mut(0, |map| {
                    map.set_key_revision_for_test(accepted_position, u64::MAX);
                });
            }
            StoragePreflightFailure::SettingsConflict => {
                core.fixed_after_prepare_settings_conflict_for_test = true;
            }
            StoragePreflightFailure::KeyConflict => {
                core.fixed_after_prepare_data_conflict_for_test = Some(accepted_position);
            }
            StoragePreflightFailure::Reservation(reservation) => {
                core.data
                    .set_test_transaction_reservation_failpoint(Some(reservation));
            }
            StoragePreflightFailure::LiveSpatialRegistry => {
                core.data
                    .set_test_transaction_live_spatial_registry_fail_lod(Some(0));
            }
        }
        let before = FixedObservableSnapshot::capture(&core, &tracked, &[], &[], None);
        let result = core.prepare_fixed_viewer_transaction(&[new], true, true, false);
        let expected = match failure {
            StoragePreflightFailure::ViewerOverflow => {
                VoxelTerrainRuntimeError::DataRefcountOverflow {
                    location: BlockLocation {
                        position: accepted_position,
                        lod_index: 0,
                    },
                    field: DataRefField::ResidentViewers,
                }
            }
            StoragePreflightFailure::ViewerUnderflow => {
                VoxelTerrainRuntimeError::DataRefcountUnderflow {
                    location: BlockLocation {
                        position: old_failure_position,
                        lod_index: 0,
                    },
                    field: DataRefField::ResidentViewers,
                }
            }
            StoragePreflightFailure::KeyRevisionOverflow => VoxelTerrainRuntimeError::DataMutation(
                SharedVoxelDataMutationError::KeyRevisionOverflow {
                    position: accepted_position,
                    lod_index: 0,
                },
            ),
            StoragePreflightFailure::SettingsConflict => VoxelTerrainRuntimeError::DataMutation(
                SharedVoxelDataMutationError::ConcurrentSettingsMutation {
                    expected_revision: 0,
                    actual_revision: 1,
                },
            ),
            StoragePreflightFailure::KeyConflict => VoxelTerrainRuntimeError::DataMutation(
                SharedVoxelDataMutationError::ConcurrentDataMutation {
                    position: accepted_position,
                    lod_index: 0,
                    expected_revision: VoxelDataKeyRevision::Tombstone(0),
                    actual_revision: VoxelDataKeyRevision::Present(1),
                },
            ),
            StoragePreflightFailure::Reservation(reservation) => {
                VoxelTerrainRuntimeError::DataMutation(
                    SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                        reservation,
                    },
                )
            }
            StoragePreflightFailure::LiveSpatialRegistry => VoxelTerrainRuntimeError::DataMutation(
                SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::LiveSpatialRegistry,
                },
            ),
        };
        assert_eq!(
            result.unwrap_err(),
            expected,
            "wrong typed failure for {failure:?}"
        );

        match failure {
            StoragePreflightFailure::SettingsConflict => {
                assert_eq!(core.data.settings_snapshot().revision, 1);
                core.data.set_test_settings_revision(0);
            }
            StoragePreflightFailure::KeyConflict => {
                assert_eq!(
                    core.data
                        .with_lod_map(0, |map| VoxelDataKeyRevision::Present(
                            map.key_revision(accepted_position)
                        )),
                    VoxelDataKeyRevision::Present(1)
                );
                core.data.with_lod_map_mut(0, |map| {
                    assert!(map.remove_block(accepted_position).is_some());
                    map.set_key_revision_for_test(accepted_position, 0);
                });
            }
            StoragePreflightFailure::Reservation(_) => {
                core.data.set_test_transaction_reservation_failpoint(None)
            }
            StoragePreflightFailure::LiveSpatialRegistry => core
                .data
                .set_test_transaction_live_spatial_registry_fail_lod(None),
            StoragePreflightFailure::ViewerOverflow
            | StoragePreflightFailure::ViewerUnderflow
            | StoragePreflightFailure::KeyRevisionOverflow => {}
        }
        assert_eq!(
            FixedObservableSnapshot::capture(&core, &tracked, &[], &[], None),
            before,
            "partial terrain/storage publication for {failure:?}"
        );
        if uses_finished_load {
            let DurableCompletion::LoadFinished { output, .. } =
                core.durable_completion_inbox.front().unwrap()
            else {
                panic!("load owner changed variant for {failure:?}")
            };
            assert_eq!(
                output.block_data.voxels.as_ref().unwrap().get_voxel(
                    0,
                    0,
                    0,
                    ChannelId::Type.index()
                ),
                0xC203
            );
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MeshAtomicFailure {
    Overflow(MeshRefField),
    Underflow(MeshRefField),
    TopologyOverflow,
}

fn set_mesh_feature_count(entry: &mut MeshBlockEntry, feature: MeshRefField, value: u32) {
    match feature {
        MeshRefField::ResidentViewers => entry.resident_viewers = value,
        MeshRefField::VisualViewers => entry.visual_viewers = value,
        MeshRefField::CollisionViewers => entry.collision_viewers = value,
        MeshRefField::VisualCoverageHolds => entry.visual_coverage_holds = value,
        MeshRefField::CollisionCoverageHolds => entry.collision_coverage_holds = value,
    }
}

fn mesh_demand_for(feature: MeshRefField) -> MeshDemand {
    MeshDemand {
        visuals: feature == MeshRefField::VisualViewers,
        collisions: feature == MeshRefField::CollisionViewers,
    }
}

#[test]
fn fixed_mesh_refcount_and_topology_counter_failures_are_atomic() {
    let failures = [
        MeshAtomicFailure::Overflow(MeshRefField::ResidentViewers),
        MeshAtomicFailure::Overflow(MeshRefField::VisualViewers),
        MeshAtomicFailure::Overflow(MeshRefField::CollisionViewers),
        MeshAtomicFailure::Underflow(MeshRefField::ResidentViewers),
        MeshAtomicFailure::Underflow(MeshRefField::VisualViewers),
        MeshAtomicFailure::Underflow(MeshRefField::CollisionViewers),
        MeshAtomicFailure::TopologyOverflow,
    ];

    for failure in failures {
        let mut core = build_core();
        let feature = match failure {
            MeshAtomicFailure::Overflow(feature) | MeshAtomicFailure::Underflow(feature) => feature,
            MeshAtomicFailure::TopologyOverflow => MeshRefField::VisualViewers,
        };
        let demand = mesh_demand_for(feature);
        let viewer = fixed_zero_distance_viewer(131, Vector3i::zero(), demand);
        let state = viewer_state(&core, viewer);
        let position = state.mesh_box.position;
        let pool = Arc::new(MeshArraysPool::new());
        let features = MeshBuildFeatures {
            visuals: true,
            collisions: true,
            variable_lod: false,
        };
        let key = MeshBlockKey {
            location: MeshBlockLocation::new(position, 0),
            revision: 0xC204,
        };
        let (upload, dropped) = pooled_mesh_output(Arc::clone(&pool), key, features, 1, 1, false)
            .into_upload()
            .into_parts();
        assert!(!dropped);
        let upload_identity = Arc::as_ptr(&upload) as usize;

        match failure {
            MeshAtomicFailure::Overflow(_) | MeshAtomicFailure::TopologyOverflow => {
                core.mesh_maps[0].insert(
                    position,
                    MeshBlockEntry {
                        position,
                        requested_features: features,
                        applied_revision: Some(key.revision),
                        has_geometry: true,
                        accepted_upload: Some(upload),
                        ..MeshBlockEntry::default()
                    },
                );
            }
            MeshAtomicFailure::Underflow(_) => {
                core.paired_viewers.push(PairedViewer {
                    id: viewer.id,
                    state: state.clone(),
                    prev_state: state.clone(),
                });
                for data_position in state.data_box.iter_cells_zxy() {
                    core.loading_blocks[0].insert(
                        data_position,
                        LoadingBlockEntry {
                            residency: DataResidencyRefs::with_resident_viewers(1),
                            retry_count: 0,
                            request_generation: 31,
                            request_state: LoadRequestState::Queued,
                            physical_request: None,
                        },
                    );
                }
                for mesh_position in state.mesh_box.iter_cells_zxy() {
                    core.mesh_maps[0].insert(
                        mesh_position,
                        MeshBlockEntry {
                            position: mesh_position,
                            resident_viewers: 1,
                            visual_viewers: u32::from(demand.visuals),
                            collision_viewers: u32::from(demand.collisions),
                            ..MeshBlockEntry::default()
                        },
                    );
                }
                let entry = core.mesh_maps[0].get_mut(&position).unwrap();
                entry.requested_features = features;
                entry.applied_revision = Some(key.revision);
                entry.has_geometry = true;
                entry.accepted_upload = Some(upload);
            }
        }
        let entry = core.mesh_maps[0].get_mut(&position).unwrap();
        match failure {
            MeshAtomicFailure::Overflow(failed_feature) => {
                set_mesh_feature_count(entry, failed_feature, u32::MAX);
            }
            MeshAtomicFailure::Underflow(failed_feature) => {
                set_mesh_feature_count(entry, failed_feature, 0);
            }
            MeshAtomicFailure::TopologyOverflow => {
                entry.visual_active = false;
                core.next_render_topology_revision = u64::MAX;
            }
        }

        let tracked = locations_in(state.data_box);
        let before = FixedObservableSnapshot::capture(
            &core,
            &tracked,
            std::slice::from_ref(&pool),
            &[],
            None,
        );
        let desired = if matches!(failure, MeshAtomicFailure::Underflow(_)) {
            &[][..]
        } else {
            std::slice::from_ref(&viewer)
        };
        let expected = match failure {
            MeshAtomicFailure::Overflow(failed_feature) => {
                VoxelTerrainRuntimeError::MeshRefcountOverflow {
                    location: MeshBlockLocation::new(position, 0),
                    field: failed_feature,
                }
            }
            MeshAtomicFailure::Underflow(failed_feature) => {
                VoxelTerrainRuntimeError::MeshRefcountUnderflow {
                    location: MeshBlockLocation::new(position, 0),
                    field: failed_feature,
                }
            }
            MeshAtomicFailure::TopologyOverflow => {
                VoxelTerrainRuntimeError::RenderTopologyRevisionOverflow
            }
        };
        assert_eq!(
            core.prepare_fixed_viewer_transaction(desired, true, false, false)
                .unwrap_err(),
            expected,
            "wrong typed mesh failure for {failure:?}"
        );
        assert_eq!(
            FixedObservableSnapshot::capture(
                &core,
                &tracked,
                std::slice::from_ref(&pool),
                &[],
                None,
            ),
            before,
            "partial mesh publication for {failure:?}"
        );
        assert_eq!(
            Arc::as_ptr(core.mesh_maps[0][&position].accepted_upload().unwrap()) as usize,
            upload_identity
        );
        assert_eq!(pool.idle_count(), 0);
        assert!(core.event_outbox.is_empty());
    }
}

fn completed_owner_with_followup(
    owner_runs: Arc<AtomicUsize>,
    followup_runs: Arc<AtomicUsize>,
    owner_lane: TaskLane,
    followup_lane: TaskLane,
) -> CompletedTask {
    CompletedTask::new(
        Box::new(CountingCompletionFollowUpTask { runs: owner_runs }),
        owner_lane,
        TaskCompletionStatus::Finished,
        vec![ScheduledTask::new(
            Box::new(CountingCompletionFollowUpTask {
                runs: followup_runs,
            }),
            followup_lane,
        )],
    )
}

fn assert_followup_counts(counters: &[Arc<AtomicUsize>], expected: &[usize]) {
    assert_eq!(counters.len(), expected.len());
    for (counter, expected) in counters.iter().zip(expected) {
        assert_eq!(counter.load(Ordering::SeqCst), *expected);
    }
}

fn run_middle_load_checked_failure_case() {
    let mut core = build_core();
    let positions = [
        Vector3i::new(201, 0, 0),
        Vector3i::new(202, 0, 0),
        Vector3i::new(203, 0, 0),
    ];
    let live_generations = [10, 20, 30];
    let output_generations = [9, 20, 30];
    let owner_runs = (0..3)
        .map(|_| Arc::new(AtomicUsize::new(0)))
        .collect::<Vec<_>>();
    let followup_runs = (0..3)
        .map(|_| Arc::new(AtomicUsize::new(0)))
        .collect::<Vec<_>>();
    for index in 0..3 {
        core.loading_blocks[0].insert(
            positions[index],
            LoadingBlockEntry {
                residency: DataResidencyRefs::with_resident_viewers(1),
                retry_count: 0,
                request_generation: live_generations[index],
                request_state: LoadRequestState::InFlight,
                physical_request: None,
            },
        );
        core.durable_completion_inbox
            .push_back(DurableCompletion::LoadFinished {
                completed: completed_owner_with_followup(
                    Arc::clone(&owner_runs[index]),
                    Arc::clone(&followup_runs[index]),
                    if index == 1 {
                        TaskLane::Serial
                    } else {
                        TaskLane::Parallel
                    },
                    if index == 1 {
                        TaskLane::Parallel
                    } else {
                        TaskLane::Serial
                    },
                ),
                output: loaded_output(
                    &core,
                    positions[index],
                    output_generations[index],
                    0xC250 + index as u64,
                ),
            });
    }
    core.stats.blocks_loaded = u64::MAX;
    let tracked = positions
        .into_iter()
        .map(|position| BlockLocation {
            position,
            lod_index: 0,
        })
        .collect::<Vec<_>>();
    let before = FixedObservableSnapshot::capture(&core, &tracked, &[], &followup_runs, None);
    assert_eq!(
        core.prepare_fixed_viewer_transaction(&[], false, false, false)
            .unwrap_err(),
        VoxelTerrainRuntimeError::StatsOverflow
    );
    assert_eq!(
        FixedObservableSnapshot::capture(&core, &tracked, &[], &followup_runs, None),
        before
    );
    assert_followup_counts(&followup_runs, &[0, 0, 0]);

    core.stats.blocks_loaded = 0;
    core.prepare_fixed_viewer_transaction(&[], false, false, false)
        .unwrap();
    core.wait_for_pending_tasks();
    assert!(core.durable_completion_inbox.is_empty());
    assert!(core.data.block_snapshot(positions[0], 0).is_none());
    for (index, position) in positions.iter().enumerate().skip(1).take(2) {
        assert_eq!(
            core.data
                .block_snapshot(*position, 0)
                .unwrap()
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            0xC250 + index as u64
        );
    }
    assert_eq!(core.stats.blocks_loaded, 2);
    assert_followup_counts(&followup_runs, &[0, 1, 1]);
    assert_followup_counts(&owner_runs, &[0, 0, 0]);
}

fn run_middle_mesh_checked_failure_case() {
    let mut core = build_core();
    let positions = [
        Vector3i::new(211, 0, 0),
        Vector3i::new(212, 0, 0),
        Vector3i::new(213, 0, 0),
    ];
    let live_revisions = [10, 20, 30];
    let output_revisions = [9, 20, 30];
    let features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };
    let pools = (0..3)
        .map(|_| Arc::new(MeshArraysPool::new()))
        .collect::<Vec<_>>();
    let owner_runs = (0..3)
        .map(|_| Arc::new(AtomicUsize::new(0)))
        .collect::<Vec<_>>();
    let followup_runs = (0..3)
        .map(|_| Arc::new(AtomicUsize::new(0)))
        .collect::<Vec<_>>();
    let mut upload_identities = Vec::new();
    for index in 0..3 {
        core.mesh_maps[0].insert(
            positions[index],
            MeshBlockEntry {
                position: positions[index],
                resident_viewers: 1,
                visual_viewers: 1,
                requested_revision: Some(live_revisions[index]),
                requested_features: features,
                is_in_update_list: true,
                ..MeshBlockEntry::default()
            },
        );
        let output = pooled_mesh_output(
            Arc::clone(&pools[index]),
            MeshBlockKey {
                location: MeshBlockLocation::new(positions[index], 0),
                revision: output_revisions[index],
            },
            features,
            1,
            0,
            false,
        )
        .into_upload();
        upload_identities.push(Arc::as_ptr(output.upload()) as usize);
        core.durable_completion_inbox
            .push_back(DurableCompletion::MeshFinished {
                completed: completed_owner_with_followup(
                    Arc::clone(&owner_runs[index]),
                    Arc::clone(&followup_runs[index]),
                    if index == 1 {
                        TaskLane::Serial
                    } else {
                        TaskLane::Parallel
                    },
                    if index == 1 {
                        TaskLane::Parallel
                    } else {
                        TaskLane::Serial
                    },
                ),
                output,
            });
    }
    core.stats.meshes_built = u64::MAX;
    let before = FixedObservableSnapshot::capture(&core, &[], &pools, &followup_runs, None);
    assert_eq!(
        core.prepare_fixed_viewer_transaction(&[], false, false, false)
            .unwrap_err(),
        VoxelTerrainRuntimeError::StatsOverflow
    );
    assert_eq!(
        FixedObservableSnapshot::capture(&core, &[], &pools, &followup_runs, None),
        before
    );
    assert_followup_counts(&followup_runs, &[0, 0, 0]);

    core.stats.meshes_built = 0;
    core.prepare_fixed_viewer_transaction(&[], false, false, false)
        .unwrap();
    core.wait_for_pending_tasks();
    assert!(core.durable_completion_inbox.is_empty());
    assert!(core.mesh_maps[0][&positions[0]].accepted_upload().is_none());
    for index in 1..3 {
        assert_eq!(
            Arc::as_ptr(
                core.mesh_maps[0][&positions[index]]
                    .accepted_upload()
                    .unwrap()
            ) as usize,
            upload_identities[index]
        );
    }
    assert_eq!(core.stats.meshes_built, 2);
    assert_followup_counts(&followup_runs, &[0, 1, 1]);
    assert_followup_counts(&owner_runs, &[0, 0, 0]);
}

fn terminal_payload(marker: u64) -> VoxelBuffer {
    let mut payload = VoxelBuffer::with_size(Vector3i::splat(16));
    payload.set_voxel(marker, 0, 0, 0, ChannelId::Type.index());
    payload
}

fn run_middle_save_checked_failure_case() {
    let mut core = build_core();
    let positions = [
        Vector3i::new(221, 0, 0),
        Vector3i::new(222, 0, 0),
        Vector3i::new(223, 0, 0),
    ];
    let block_revisions = [111, 222, 333];
    let generations = [11, 22, 33];
    let live_attempts = [10, 20, 30];
    let terminal_attempts = [9, 20, 30];
    let owner_runs = (0..3)
        .map(|_| Arc::new(AtomicUsize::new(0)))
        .collect::<Vec<_>>();
    let followup_runs = (0..3)
        .map(|_| Arc::new(AtomicUsize::new(0)))
        .collect::<Vec<_>>();
    let mut payload_identities = Vec::new();
    for index in 0..3 {
        let key = SaveKey::new(positions[index], 0);
        core.save_journal.insert(
            key,
            SaveJournalEntry {
                written_unflushed: None,
                active: Some(ActiveSaveAttempt::WriteInFlight {
                    meta: SaveAttemptMeta {
                        block_revision: block_revisions[index],
                        generation: generations[index],
                        retry_count: if index == 1 { u32::MAX } else { 0 },
                        last_error: None,
                    },
                    attempt_ordinal: live_attempts[index],
                }),
                queued_newer: VecDeque::new(),
            },
        );
        let payload = terminal_payload(0xC260 + index as u64);
        payload_identities.push(voxel_allocation_identity(&payload));
        core.durable_completion_inbox
            .push_back(DurableCompletion::SaveAcknowledged {
                completed: completed_owner_with_followup(
                    Arc::clone(&owner_runs[index]),
                    Arc::clone(&followup_runs[index]),
                    if index == 1 {
                        TaskLane::Serial
                    } else {
                        TaskLane::Parallel
                    },
                    if index == 1 {
                        TaskLane::Parallel
                    } else {
                        TaskLane::Serial
                    },
                ),
                terminal: SaveTaskTerminal {
                    location: BlockLocation {
                        position: positions[index],
                        lod_index: 0,
                    },
                    block_revision: block_revisions[index],
                    save_generation: generations[index],
                    payload,
                    task_panic_phase: None,
                    phase: PersistenceIoPhase::Acknowledged,
                    acknowledgement: Some(PersistenceAcknowledgement::Save(if index == 1 {
                        Err(VoxelStreamError::Io(
                            "matrix middle save failure".to_owned(),
                        ))
                    } else {
                        Ok(())
                    })),
                },
                attempt_ordinal: terminal_attempts[index],
            });
    }
    let before = FixedObservableSnapshot::capture(&core, &[], &[], &followup_runs, None);
    assert_eq!(
        core.prepare_fixed_viewer_transaction(&[], false, false, false)
            .unwrap_err(),
        VoxelTerrainRuntimeError::PersistenceRetryCountOverflow {
            operation: PersistenceOperation::Save {
                location: BlockLocation {
                    position: positions[1],
                    lod_index: 0,
                },
                block_revision: block_revisions[1],
                save_generation: generations[1],
            },
        }
    );
    assert_eq!(
        FixedObservableSnapshot::capture(&core, &[], &[], &followup_runs, None),
        before
    );
    assert_followup_counts(&followup_runs, &[0, 0, 0]);

    let Some(ActiveSaveAttempt::WriteInFlight { meta, .. }) = core
        .save_journal
        .get_mut(&SaveKey::new(positions[1], 0))
        .unwrap()
        .active
        .as_mut()
    else {
        panic!("middle save owner changed state before retry")
    };
    meta.retry_count = 0;
    core.prepare_fixed_viewer_transaction(&[], false, false, false)
        .unwrap();
    core.wait_for_pending_tasks();
    assert!(core.durable_completion_inbox.is_empty());
    let first = &core.save_journal[&SaveKey::new(positions[0], 0)];
    assert!(matches!(
        first.active,
        Some(ActiveSaveAttempt::WriteInFlight {
            attempt_ordinal: 10,
            ..
        })
    ));
    let middle = &core.save_journal[&SaveKey::new(positions[1], 0)];
    let Some(ActiveSaveAttempt::Pending(pending)) = middle.active.as_ref() else {
        panic!("failed save did not restore its exact payload")
    };
    assert_eq!(pending.meta.block_revision, block_revisions[1]);
    assert_eq!(pending.meta.retry_count, 1);
    assert_eq!(
        voxel_allocation_identity(&pending.payload),
        payload_identities[1]
    );
    let third = &core.save_journal[&SaveKey::new(positions[2], 0)];
    assert_eq!(
        third.written_unflushed.as_ref().unwrap().block_revision,
        block_revisions[2]
    );
    assert_eq!(
        voxel_allocation_identity(&third.written_unflushed.as_ref().unwrap().payload),
        payload_identities[2]
    );
    assert_followup_counts(&followup_runs, &[0, 1, 1]);
    assert_followup_counts(&owner_runs, &[0, 0, 0]);
}

fn run_middle_flush_checked_failure_case() {
    let mut core = build_core();
    let key = SaveKey::new(Vector3i::new(231, 0, 0), 0);
    let block_revision = 707;
    let written = terminal_payload(0xC270);
    let written_identity = voxel_allocation_identity(&written);
    core.save_journal.insert(
        key,
        SaveJournalEntry {
            written_unflushed: Some(WrittenSave {
                block_revision,
                generation: 77,
                payload: written,
            }),
            active: None,
            queued_newer: VecDeque::new(),
        },
    );
    core.save_checkpoint_in_flight = Some(SaveCheckpointInFlight {
        checkpoint_generation: 88,
        acknowledged: vec![SaveCheckpointSnapshot {
            key,
            block_revision,
            generation: 77,
        }],
        state: CheckpointAttemptState::WriteInFlight {
            attempt_ordinal: 20,
        },
        retry_count: u32::MAX,
        max_attempts: u32::MAX,
        origin: CheckpointOrigin::Explicit,
        record_per_block_failure: false,
    });
    let owner_runs = (0..3)
        .map(|_| Arc::new(AtomicUsize::new(0)))
        .collect::<Vec<_>>();
    let followup_runs = (0..3)
        .map(|_| Arc::new(AtomicUsize::new(0)))
        .collect::<Vec<_>>();
    for index in 0..3 {
        core.durable_completion_inbox
            .push_back(DurableCompletion::FlushAcknowledged {
                completed: completed_owner_with_followup(
                    Arc::clone(&owner_runs[index]),
                    Arc::clone(&followup_runs[index]),
                    if index == 1 {
                        TaskLane::Serial
                    } else {
                        TaskLane::Parallel
                    },
                    if index == 1 {
                        TaskLane::Parallel
                    } else {
                        TaskLane::Serial
                    },
                ),
                terminal: FlushTaskTerminal {
                    checkpoint_generation: [87, 88, 89][index],
                    task_panic_phase: None,
                    phase: PersistenceIoPhase::BeforeIo,
                    acknowledgement: None,
                },
                attempt_ordinal: [19, 20, 21][index],
            });
    }
    let before = FixedObservableSnapshot::capture(&core, &[], &[], &followup_runs, None);
    assert_eq!(
        core.prepare_fixed_viewer_transaction(&[], false, false, false)
            .unwrap_err(),
        VoxelTerrainRuntimeError::PersistenceRetryCountOverflow {
            operation: PersistenceOperation::Flush {
                checkpoint_generation: 88,
            },
        }
    );
    assert_eq!(
        FixedObservableSnapshot::capture(&core, &[], &[], &followup_runs, None),
        before
    );
    assert_followup_counts(&followup_runs, &[0, 0, 0]);

    core.save_checkpoint_in_flight.as_mut().unwrap().retry_count = 0;
    core.prepare_fixed_viewer_transaction(&[], false, false, false)
        .unwrap();
    core.wait_for_pending_tasks();
    assert!(core.durable_completion_inbox.is_empty());
    let checkpoint = core.save_checkpoint_in_flight.as_ref().unwrap();
    assert_eq!(checkpoint.retry_count, 1);
    assert_eq!(checkpoint.state, CheckpointAttemptState::Pending);
    assert_eq!(checkpoint.acknowledged.len(), 1);
    assert_eq!(checkpoint.acknowledged[0].block_revision, block_revision);
    assert_eq!(
        voxel_allocation_identity(
            &core.save_journal[&key]
                .written_unflushed
                .as_ref()
                .unwrap()
                .payload
        ),
        written_identity
    );
    assert_followup_counts(&followup_runs, &[0, 1, 0]);
    assert_followup_counts(&owner_runs, &[0, 0, 0]);
}

#[test]
fn three_durable_completions_middle_checked_failure_consumes_zero() {
    run_middle_load_checked_failure_case();
    run_middle_mesh_checked_failure_case();
    run_middle_save_checked_failure_case();
    run_middle_flush_checked_failure_case();
}

#[derive(Debug, Clone, Copy)]
enum PrefixCommitFailure {
    EventOutbox,
    StorageLiveMap,
    PreparedTaskBatch,
}

#[test]
fn accepted_completion_followups_wait_for_prefix_commit() {
    for failure in [
        PrefixCommitFailure::EventOutbox,
        PrefixCommitFailure::StorageLiveMap,
        PrefixCommitFailure::PreparedTaskBatch,
    ] {
        let mut core = build_core();
        let accepted_position = Vector3i::new(241, 0, 0);
        let stale_position = Vector3i::new(242, 0, 0);
        core.loading_blocks[0].insert(
            accepted_position,
            LoadingBlockEntry {
                residency: DataResidencyRefs::with_resident_viewers(1),
                retry_count: 0,
                request_generation: 7,
                request_state: LoadRequestState::InFlight,
                physical_request: None,
            },
        );
        core.loading_blocks[0].insert(
            stale_position,
            LoadingBlockEntry {
                residency: DataResidencyRefs::with_resident_viewers(1),
                retry_count: 0,
                request_generation: 11,
                request_state: LoadRequestState::InFlight,
                physical_request: None,
            },
        );
        let owner_runs = (0..2)
            .map(|_| Arc::new(AtomicUsize::new(0)))
            .collect::<Vec<_>>();
        let accepted_serial_runs = Arc::new(AtomicUsize::new(0));
        let accepted_parallel_runs = Arc::new(AtomicUsize::new(0));
        let stale_runs = Arc::new(AtomicUsize::new(0));
        core.durable_completion_inbox
            .push_back(DurableCompletion::LoadFinished {
                completed: CompletedTask::new(
                    Box::new(CountingCompletionFollowUpTask {
                        runs: Arc::clone(&owner_runs[0]),
                    }),
                    TaskLane::Parallel,
                    TaskCompletionStatus::Finished,
                    vec![
                        ScheduledTask::new(
                            Box::new(CountingCompletionFollowUpTask {
                                runs: Arc::clone(&accepted_serial_runs),
                            }),
                            TaskLane::Serial,
                        ),
                        ScheduledTask::new(
                            Box::new(CountingCompletionFollowUpTask {
                                runs: Arc::clone(&accepted_parallel_runs),
                            }),
                            TaskLane::Parallel,
                        ),
                    ],
                ),
                output: loaded_output(&core, accepted_position, 7, 0xC280),
            });
        core.durable_completion_inbox
            .push_back(DurableCompletion::LoadFinished {
                completed: CompletedTask::new(
                    Box::new(CountingCompletionFollowUpTask {
                        runs: Arc::clone(&owner_runs[1]),
                    }),
                    TaskLane::Serial,
                    TaskCompletionStatus::Finished,
                    vec![ScheduledTask::new(
                        Box::new(CountingCompletionFollowUpTask {
                            runs: Arc::clone(&stale_runs),
                        }),
                        TaskLane::Serial,
                    )],
                ),
                output: loaded_output(&core, stale_position, 10, 0xC281),
            });
        match failure {
            PrefixCommitFailure::EventOutbox => {
                core.fail_fixed_capacity_for_test(FixedCapacityDestination::EventOutbox, 1);
            }
            PrefixCommitFailure::StorageLiveMap => {
                core.data.set_test_transaction_reservation_failpoint(Some(
                    SharedVoxelDataTransactionReservation::LiveMap,
                ))
            }
            PrefixCommitFailure::PreparedTaskBatch => core
                .task_runner
                .fail_next_prepared_batch_reservation_for_test(),
        }
        let tracked = [accepted_position, stale_position]
            .into_iter()
            .map(|position| BlockLocation {
                position,
                lod_index: 0,
            })
            .collect::<Vec<_>>();
        let started = [
            Arc::clone(&accepted_serial_runs),
            Arc::clone(&accepted_parallel_runs),
            Arc::clone(&stale_runs),
        ];
        let before = FixedObservableSnapshot::capture(&core, &tracked, &[], &started, None);
        let expected = match failure {
            PrefixCommitFailure::EventOutbox | PrefixCommitFailure::PreparedTaskBatch => {
                VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
            }
            PrefixCommitFailure::StorageLiveMap => VoxelTerrainRuntimeError::DataMutation(
                SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::LiveMap,
                },
            ),
        };
        assert_eq!(
            core.prepare_fixed_viewer_transaction(&[], false, false, false)
                .unwrap_err(),
            expected,
            "wrong late prefix failure for {failure:?}"
        );
        if matches!(failure, PrefixCommitFailure::EventOutbox) {
            assert_eq!(
                core.last_fixed_capacity_failure_for_test(),
                Some(FixedCapacityDestination::EventOutbox)
            );
        }
        assert_eq!(
            FixedObservableSnapshot::capture(&core, &tracked, &[], &started, None),
            before,
            "completion/follow-up ownership changed for {failure:?}"
        );
        assert_followup_counts(&started, &[0, 0, 0]);
        assert_eq!(core.task_runner.remaining_task_count(), 0);
        if matches!(failure, PrefixCommitFailure::StorageLiveMap) {
            core.data.set_test_transaction_reservation_failpoint(None);
        }

        core.prepare_fixed_viewer_transaction(&[], false, false, false)
            .unwrap();
        core.wait_for_pending_tasks();
        assert!(core.durable_completion_inbox.is_empty());
        assert_eq!(core.task_runner.remaining_task_count(), 0);
        assert_eq!(
            core.data
                .block_snapshot(accepted_position, 0)
                .unwrap()
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            0xC280
        );
        assert!(core.data.block_snapshot(stale_position, 0).is_none());
        assert_followup_counts(&started, &[1, 1, 0]);
        assert_followup_counts(&owner_runs, &[0, 0]);
        assert_eq!(core.event_outbox.len(), 1);
        assert!(matches!(
            core.event_outbox.front(),
            Some(VoxelTerrainEvent::DataBlockLoaded(location))
                if *location == BlockLocation { position: accepted_position, lod_index: 0 }
        ));
    }
}

struct CombinedCapacityFixture {
    core: VoxelTerrainCore,
    desired: ViewerUpdate,
    tracked: Vec<BlockLocation>,
    pools: Vec<Arc<MeshArraysPool>>,
}

fn mark_block_modified(block: &mut VoxelDataBlock) {
    block.set_modified(true);
    block.set_edited(true);
}

fn build_combined_capacity_fixture() -> CombinedCapacityFixture {
    let mut core = build_core();
    let demand = MeshDemand {
        visuals: true,
        collisions: false,
    };
    let old = fixed_zero_distance_viewer(301, Vector3i::zero(), demand);
    let desired = fixed_zero_distance_viewer(302, Vector3i::new(16, 0, 0), demand);
    let old_state = viewer_state(&core, old);
    let desired_state = viewer_state(&core, desired);
    let old_positions = old_state.data_box.iter_cells_zxy().collect::<Vec<_>>();
    for (index, position) in old_positions.iter().copied().enumerate() {
        let mut block = tagged_block(0xC300 + index as u64, 1);
        if index < 2 {
            mark_block_modified(&mut block);
        }
        assert!(core.data.try_set_block(position, block).unwrap());
        record_test_storage_residency(&mut core, position, 1);
    }
    let retained_position = old_positions[2];
    let mut retained = VoxelDataBlock::empty(0);
    retained.viewers.set_exact(1);
    mark_block_modified(&mut retained);
    core.data.with_lod_map_mut(0, |map| {
        assert!(map.remove_block(retained_position).is_some());
        map.set_block(retained_position, retained, true);
        map.set_key_revision_for_test(retained_position, 1);
    });
    let queued_dirty_key = SaveKey::new(old_positions[0], 0);
    core.save_journal.insert(
        queued_dirty_key,
        SaveJournalEntry {
            written_unflushed: None,
            active: Some(ActiveSaveAttempt::WriteInFlight {
                meta: SaveAttemptMeta {
                    block_revision: 1,
                    generation: 700,
                    retry_count: 0,
                    last_error: None,
                },
                attempt_ordinal: 701,
            }),
            queued_newer: VecDeque::new(),
        },
    );
    core.paired_viewers.push(PairedViewer {
        id: old.id,
        state: old_state.clone(),
        prev_state: old_state.clone(),
    });
    for position in old_state.mesh_box.iter_cells_zxy() {
        core.mesh_maps[0].insert(
            position,
            MeshBlockEntry {
                position,
                resident_viewers: 1,
                visual_viewers: 1,
                ..MeshBlockEntry::default()
            },
        );
    }

    let ready_mesh_position = desired_state.mesh_box.position;
    let ready_halo = Box3i::new(ready_mesh_position - Vector3i::splat(1), Vector3i::splat(3));
    for (index, position) in ready_halo.iter_cells_zxy().enumerate() {
        assert!(core
            .data
            .try_set_block(position, tagged_block(0xC340 + index as u64, 0))
            .unwrap());
        record_test_storage_residency(&mut core, position, 0);
    }

    let load_position = Vector3i::new(700, 0, 0);
    install_finished_load(&mut core, load_position, 801, 0xC380);
    core.loading_blocks[0]
        .get_mut(&load_position)
        .unwrap()
        .residency
        .resident_viewers = 1;

    let failed_save_key = SaveKey::new(Vector3i::new(710, 0, 0), 0);
    core.save_journal.insert(
        failed_save_key,
        SaveJournalEntry {
            written_unflushed: None,
            active: Some(ActiveSaveAttempt::WriteInFlight {
                meta: SaveAttemptMeta {
                    block_revision: 810,
                    generation: 811,
                    retry_count: 0,
                    last_error: None,
                },
                attempt_ordinal: 812,
            }),
            queued_newer: VecDeque::new(),
        },
    );
    core.durable_completion_inbox
        .push_back(DurableCompletion::SaveAcknowledged {
            completed: CompletedTask::new(
                Box::new(CountingCompletionFollowUpTask {
                    runs: Arc::new(AtomicUsize::new(0)),
                }),
                TaskLane::Serial,
                TaskCompletionStatus::Finished,
                Vec::new(),
            ),
            terminal: SaveTaskTerminal {
                location: BlockLocation {
                    position: failed_save_key.position,
                    lod_index: 0,
                },
                block_revision: 810,
                save_generation: 811,
                payload: terminal_payload(0xC381),
                task_panic_phase: None,
                phase: PersistenceIoPhase::Acknowledged,
                acknowledgement: Some(PersistenceAcknowledgement::Save(Err(VoxelStreamError::Io(
                    "capacity fixture save failure".to_owned(),
                )))),
            },
            attempt_ordinal: 812,
        });
    core.durable_completion_inbox
        .push_back(DurableCompletion::UnknownTerminal {
            completed: CompletedTask::new(
                Box::new(CountingCompletionFollowUpTask {
                    runs: Arc::new(AtomicUsize::new(0)),
                }),
                TaskLane::Parallel,
                TaskCompletionStatus::Cancelled,
                Vec::new(),
            ),
        });

    let direct_position = Vector3i::new(720, 0, 0);
    let direct_features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };
    let direct_key = MeshBlockKey {
        location: MeshBlockLocation::new(direct_position, 0),
        revision: 821,
    };
    core.mesh_maps[0].insert(
        direct_position,
        MeshBlockEntry {
            position: direct_position,
            resident_viewers: 1,
            visual_viewers: 1,
            requested_revision: Some(direct_key.revision),
            requested_features: direct_features,
            is_in_update_list: true,
            ..MeshBlockEntry::default()
        },
    );
    let direct_pool = Arc::new(MeshArraysPool::new());
    let (direct_upload, direct_dropped) = pooled_mesh_output(
        Arc::clone(&direct_pool),
        direct_key,
        direct_features,
        1,
        0,
        false,
    )
    .into_upload()
    .into_parts();
    core.direct_mesh_retry_inbox
        .push_back(DurableCompletion::DirectMesh {
            upload: direct_upload,
            dropped: direct_dropped,
        });

    let unview_retry_position = Vector3i::new(730, 0, 0);
    assert!(core
        .data
        .try_set_block(unview_retry_position, tagged_block(0xC382, 1))
        .unwrap());
    record_test_storage_residency(&mut core, unview_retry_position, 1);
    let view_retry_position = Vector3i::new(731, 0, 0);
    core.data_unview_retries[0].push(PendingDataMutation {
        box_in_blocks: Box3i::new(unview_retry_position, Vector3i::splat(1)),
        retry_count: 2,
    });
    core.data_view_retries[0].push(PendingDataMutation {
        box_in_blocks: Box3i::new(view_retry_position, Vector3i::splat(1)),
        retry_count: 3,
    });

    let mut tracked = locations_in(old_state.data_box);
    tracked.extend(locations_in(desired_state.data_box));
    tracked.extend(ready_halo.iter_cells_zxy().map(|position| BlockLocation {
        position,
        lod_index: 0,
    }));
    tracked.extend(
        [load_position, unview_retry_position, view_retry_position]
            .into_iter()
            .map(|position| BlockLocation {
                position,
                lod_index: 0,
            }),
    );
    CombinedCapacityFixture {
        core,
        desired,
        tracked,
        pools: vec![direct_pool],
    }
}

#[derive(Debug, Clone, Copy)]
enum LiveCapacityFailure {
    Terrain(FixedCapacityDestination),
    Storage(SharedVoxelDataTransactionReservation),
    LiveSpatialRegistry,
    RunnerBatch,
}

#[test]
fn every_fixed_live_capacity_reservation_failure_is_precommit() {
    let terrain_destinations = [
        FixedCapacityDestination::PairedViewers,
        FixedCapacityDestination::MeshMap,
        FixedCapacityDestination::LoadingMap,
        FixedCapacityDestination::PendingLoadQueue,
        FixedCapacityDestination::PendingMeshQueue,
        FixedCapacityDestination::DataViewRetries,
        FixedCapacityDestination::DataUnviewRetries,
        FixedCapacityDestination::DurableEffects,
        FixedCapacityDestination::DirectEffects,
        FixedCapacityDestination::EventOutbox,
        FixedCapacityDestination::Quarantine,
        FixedCapacityDestination::SaveJournal,
        FixedCapacityDestination::SaveJournalQueue,
        FixedCapacityDestination::DeferredSaveQueue,
        FixedCapacityDestination::DirtyRetention,
        FixedCapacityDestination::Retirement,
        FixedCapacityDestination::PreparedTaskBatch,
    ];
    let mut failures = terrain_destinations
        .into_iter()
        .map(LiveCapacityFailure::Terrain)
        .collect::<Vec<_>>();
    failures.extend([
        LiveCapacityFailure::Storage(SharedVoxelDataTransactionReservation::LiveMap),
        LiveCapacityFailure::Storage(SharedVoxelDataTransactionReservation::LiveKeyRevisions),
        LiveCapacityFailure::LiveSpatialRegistry,
        LiveCapacityFailure::RunnerBatch,
    ]);

    for failure in failures {
        let CombinedCapacityFixture {
            mut core,
            desired,
            tracked,
            pools,
        } = build_combined_capacity_fixture();
        match failure {
            LiveCapacityFailure::Terrain(destination) => {
                core.fail_fixed_capacity_for_test(destination, 1);
            }
            LiveCapacityFailure::Storage(reservation) => core
                .data
                .set_test_transaction_reservation_failpoint(Some(reservation)),
            LiveCapacityFailure::LiveSpatialRegistry => core
                .data
                .set_test_transaction_live_spatial_registry_fail_lod(Some(0)),
            LiveCapacityFailure::RunnerBatch => core
                .task_runner
                .fail_next_prepared_batch_reservation_for_test(),
        }
        let before = FixedObservableSnapshot::capture(&core, &tracked, &pools, &[], None);
        let expected = match failure {
            LiveCapacityFailure::Storage(reservation) => VoxelTerrainRuntimeError::DataMutation(
                SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation,
                },
            ),
            LiveCapacityFailure::LiveSpatialRegistry => VoxelTerrainRuntimeError::DataMutation(
                SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                    reservation: SharedVoxelDataTransactionReservation::LiveSpatialRegistry,
                },
            ),
            LiveCapacityFailure::Terrain(_) | LiveCapacityFailure::RunnerBatch => {
                VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
            }
        };
        let result = core.prepare_fixed_viewer_transaction(&[desired], true, true, true);
        assert_eq!(
            result,
            Err(expected),
            "wrong combined capacity failure for {failure:?}"
        );
        if let LiveCapacityFailure::Terrain(destination) = failure {
            assert_eq!(
                core.last_fixed_capacity_failure_for_test(),
                Some(destination),
                "combined fixture never reached {destination:?}"
            );
        }
        assert_eq!(
            FixedObservableSnapshot::capture(&core, &tracked, &pools, &[], None),
            before,
            "capacity failure partially published {failure:?}"
        );
        assert!(core.task_runner.observable_tasks_for_test().is_empty());
    }
}

#[derive(Debug, Clone, Copy)]
enum CheckedCounterFailure {
    LoadGeneration,
    MeshRevision,
    TopologyRevision,
    DataOverflow,
    DataUnderflow,
    MeshOverflow(MeshRefField),
    MeshUnderflow(MeshRefField),
    LoadTerminalRetry,
    MeshTerminalRetry,
    LoadedStats,
    MeshedStats,
    SaveGeneration,
    CheckpointGeneration,
    AttemptGeneration,
    TaskCountSum,
}

fn completion_owner_without_followups(lane: TaskLane) -> CompletedTask {
    CompletedTask::new(
        Box::new(CountingCompletionFollowUpTask {
            runs: Arc::new(AtomicUsize::new(0)),
        }),
        lane,
        TaskCompletionStatus::Finished,
        Vec::new(),
    )
}

#[test]
fn every_fixed_checked_counter_failure_preserves_payload_identity() {
    let failures = [
        CheckedCounterFailure::LoadGeneration,
        CheckedCounterFailure::MeshRevision,
        CheckedCounterFailure::TopologyRevision,
        CheckedCounterFailure::DataOverflow,
        CheckedCounterFailure::DataUnderflow,
        CheckedCounterFailure::MeshOverflow(MeshRefField::ResidentViewers),
        CheckedCounterFailure::MeshOverflow(MeshRefField::VisualViewers),
        CheckedCounterFailure::MeshOverflow(MeshRefField::CollisionViewers),
        CheckedCounterFailure::MeshUnderflow(MeshRefField::ResidentViewers),
        CheckedCounterFailure::MeshUnderflow(MeshRefField::VisualViewers),
        CheckedCounterFailure::MeshUnderflow(MeshRefField::CollisionViewers),
        CheckedCounterFailure::LoadTerminalRetry,
        CheckedCounterFailure::MeshTerminalRetry,
        CheckedCounterFailure::LoadedStats,
        CheckedCounterFailure::MeshedStats,
        CheckedCounterFailure::SaveGeneration,
        CheckedCounterFailure::CheckpointGeneration,
        CheckedCounterFailure::AttemptGeneration,
        CheckedCounterFailure::TaskCountSum,
    ];

    for failure in failures {
        let CombinedCapacityFixture {
            mut core,
            mut desired,
            mut tracked,
            pools,
        } = build_combined_capacity_fixture();
        let old_state = core.paired_viewers[0].state.clone();
        let desired_state = viewer_state(&core, desired);
        let old_position = old_state.mesh_box.position;
        let desired_position = desired_state.mesh_box.position;
        let terminal_load_position = Vector3i::new(740, 0, 0);
        let terminal_mesh_position = Vector3i::new(741, 0, 0);
        let initial_save_generation = core.next_save_generation;

        match failure {
            CheckedCounterFailure::LoadGeneration => core.next_request_generation = u64::MAX,
            CheckedCounterFailure::MeshRevision => core.next_mesh_revision = u64::MAX,
            CheckedCounterFailure::TopologyRevision => {
                core.next_render_topology_revision = u64::MAX;
            }
            CheckedCounterFailure::DataOverflow => {
                core.data.with_lod_map_mut(0, |map| {
                    map.get_block_mut(desired_state.data_box.position)
                        .unwrap()
                        .viewers
                        .set_exact(u32::MAX);
                });
                record_test_storage_residency(&mut core, desired_state.data_box.position, u32::MAX);
            }
            CheckedCounterFailure::DataUnderflow => {
                core.data.with_lod_map_mut(0, |map| {
                    map.get_block_mut(old_state.data_box.position)
                        .unwrap()
                        .viewers
                        .set_exact(0);
                });
                record_test_storage_residency(&mut core, old_state.data_box.position, 0);
            }
            CheckedCounterFailure::MeshOverflow(feature) => {
                if feature == MeshRefField::CollisionViewers {
                    desired.demand.collisions = true;
                }
                let entry = core.mesh_maps[0]
                    .entry(desired_position)
                    .or_insert_with(|| MeshBlockEntry {
                        position: desired_position,
                        ..MeshBlockEntry::default()
                    });
                set_mesh_feature_count(entry, feature, u32::MAX);
            }
            CheckedCounterFailure::MeshUnderflow(feature) => {
                if feature == MeshRefField::CollisionViewers {
                    core.paired_viewers[0].state.demand.collisions = true;
                    core.paired_viewers[0].prev_state.demand.collisions = true;
                    for entry in core.mesh_maps[0].values_mut() {
                        if old_state.mesh_box.contains_point(entry.position) {
                            entry.collision_viewers = 1;
                        }
                    }
                }
                set_mesh_feature_count(
                    core.mesh_maps[0].get_mut(&old_position).unwrap(),
                    feature,
                    0,
                );
            }
            CheckedCounterFailure::LoadTerminalRetry => {
                core.loading_blocks[0].insert(
                    terminal_load_position,
                    LoadingBlockEntry {
                        residency: DataResidencyRefs::with_resident_viewers(1),
                        retry_count: u32::MAX,
                        request_generation: 901,
                        request_state: LoadRequestState::InFlight,
                        physical_request: None,
                    },
                );
                core.durable_completion_inbox
                    .push_back(DurableCompletion::LoadTerminal {
                        completed: completion_owner_without_followups(TaskLane::Parallel),
                        position: terminal_load_position,
                        lod_index: 0,
                        request_generation: 901,
                        request_tag: None,
                    });
                tracked.push(BlockLocation {
                    position: terminal_load_position,
                    lod_index: 0,
                });
            }
            CheckedCounterFailure::MeshTerminalRetry => {
                let key = MeshBlockKey {
                    location: MeshBlockLocation::new(terminal_mesh_position, 0),
                    revision: 902,
                };
                core.mesh_maps[0].insert(
                    terminal_mesh_position,
                    MeshBlockEntry {
                        position: terminal_mesh_position,
                        resident_viewers: 1,
                        requested_revision: Some(key.revision),
                        terminal_retry_count: u32::MAX,
                        ..MeshBlockEntry::default()
                    },
                );
                core.durable_completion_inbox
                    .push_back(DurableCompletion::MeshTerminal {
                        completed: completion_owner_without_followups(TaskLane::Parallel),
                        key,
                        request_tag: None,
                    });
            }
            CheckedCounterFailure::LoadedStats => core.stats.blocks_loaded = u64::MAX,
            CheckedCounterFailure::MeshedStats => core.stats.meshes_built = u64::MAX,
            CheckedCounterFailure::SaveGeneration => core.next_save_generation = u64::MAX,
            CheckedCounterFailure::CheckpointGeneration => {
                core.next_save_checkpoint_generation = u64::MAX;
            }
            CheckedCounterFailure::AttemptGeneration => {
                core.next_persistence_attempt_ordinal = u64::MAX;
                core.data.with_lod_map_mut(0, |map| {
                    map.get_block_mut(old_state.data_box.position)
                        .unwrap()
                        .set_modified(false);
                });
            }
            CheckedCounterFailure::TaskCountSum => {
                core.set_fixed_task_count_bias_for_test(usize::MAX);
            }
        }

        let before = FixedObservableSnapshot::capture(&core, &tracked, &pools, &[], None);
        let result = if matches!(failure, CheckedCounterFailure::CheckpointGeneration) {
            core.prepare_fixed_viewer_transaction_with_checkpoint(
                &[desired],
                true,
                true,
                true,
                Some(FixedCheckpointRequest {
                    origin: CheckpointOrigin::Explicit,
                    max_attempts: 3,
                    record_per_block_failure: false,
                    reset_pending_retry_count: false,
                }),
                None,
                None,
                false,
            )
        } else {
            core.prepare_fixed_viewer_transaction(&[desired], true, true, true)
        };
        let correct_error = match (failure, result.as_ref().err()) {
            (
                CheckedCounterFailure::LoadGeneration,
                Some(VoxelTerrainRuntimeError::RequestGenerationOverflow),
            )
            | (
                CheckedCounterFailure::MeshRevision,
                Some(VoxelTerrainRuntimeError::MeshRevisionOverflow),
            )
            | (
                CheckedCounterFailure::TopologyRevision,
                Some(VoxelTerrainRuntimeError::RenderTopologyRevisionOverflow),
            )
            | (
                CheckedCounterFailure::SaveGeneration,
                Some(VoxelTerrainRuntimeError::SaveGenerationOverflow),
            )
            | (
                CheckedCounterFailure::CheckpointGeneration,
                Some(VoxelTerrainRuntimeError::SaveGenerationOverflow),
            )
            | (
                CheckedCounterFailure::LoadedStats | CheckedCounterFailure::MeshedStats,
                Some(VoxelTerrainRuntimeError::StatsOverflow),
            )
            | (
                CheckedCounterFailure::TaskCountSum,
                Some(VoxelTerrainRuntimeError::TaskCountOverflow),
            ) => true,
            (
                CheckedCounterFailure::DataOverflow,
                Some(VoxelTerrainRuntimeError::DataRefcountOverflow {
                    location:
                        BlockLocation {
                            position,
                            lod_index: 0,
                        },
                    field: DataRefField::ResidentViewers,
                }),
            ) => *position == desired_state.data_box.position,
            (
                CheckedCounterFailure::DataUnderflow,
                Some(VoxelTerrainRuntimeError::DataRefcountUnderflow {
                    location:
                        BlockLocation {
                            position,
                            lod_index: 0,
                        },
                    field: DataRefField::ResidentViewers,
                }),
            ) => *position == old_state.data_box.position,
            (
                CheckedCounterFailure::MeshOverflow(feature),
                Some(VoxelTerrainRuntimeError::MeshRefcountOverflow {
                    location,
                    field: actual_feature,
                }),
            ) => {
                *actual_feature == feature
                    && *location == MeshBlockLocation::new(desired_position, 0)
            }
            (
                CheckedCounterFailure::MeshUnderflow(feature),
                Some(VoxelTerrainRuntimeError::MeshRefcountUnderflow {
                    location,
                    field: actual_feature,
                }),
            ) => *actual_feature == feature && *location == MeshBlockLocation::new(old_position, 0),
            (
                CheckedCounterFailure::LoadTerminalRetry,
                Some(VoxelTerrainRuntimeError::LoadRetryCountOverflow { location }),
            ) => {
                *location
                    == BlockLocation {
                        position: terminal_load_position,
                        lod_index: 0,
                    }
            }
            (
                CheckedCounterFailure::MeshTerminalRetry,
                Some(VoxelTerrainRuntimeError::MeshTerminalRetryCountOverflow { key }),
            ) => {
                *key == MeshBlockKey {
                    location: MeshBlockLocation::new(terminal_mesh_position, 0),
                    revision: 902,
                }
            }
            (
                CheckedCounterFailure::AttemptGeneration,
                Some(VoxelTerrainRuntimeError::PersistenceAttemptOverflow {
                    operation:
                        PersistenceOperation::Save {
                            location,
                            block_revision,
                            save_generation,
                        },
                }),
            ) => {
                location.position == old_state.data_box.iter_cells_zxy().nth(1).unwrap()
                    && location.lod_index == 0
                    && *block_revision == 1
                    && *save_generation == initial_save_generation
            }
            _ => false,
        };
        assert!(
            correct_error,
            "wrong checked counter failure for {failure:?}: {result:?}"
        );
        assert_eq!(
            FixedObservableSnapshot::capture(&core, &tracked, &pools, &[], None),
            before,
            "checked counter failure changed payload/state for {failure:?}"
        );
    }
}

#[test]
fn load_mesh_save_payload_identity_survives_late_failure_and_retry() {
    // Pre-admission failure returns the exact caller-owned output. A failure
    // after admission instead leaves the exact immutable upload in the common
    // direct prefix until the next successful transaction.
    let mut direct_core = build_core();
    let direct_position = Vector3i::new(760, 0, 0);
    let direct_features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };
    let direct_key = prepare_direct_mesh(&mut direct_core, direct_position);
    let not_admitted_pool = Arc::new(MeshArraysPool::new());
    let not_admitted = pooled_mesh_output(
        Arc::clone(&not_admitted_pool),
        direct_key,
        direct_features,
        1,
        0,
        false,
    );
    let not_admitted_pool_ptr = Arc::as_ptr(not_admitted.pool()) as usize;
    let not_admitted_payload_ptr = not_admitted.output().surfaces.as_ptr() as usize;
    direct_core.fail_next_direct_mesh_reservation_for_test = true;
    let returned = match direct_core.try_apply_mesh_output(not_admitted) {
        Err(MeshOutputApplyError::NotAdmitted { output, .. }) => output,
        _ => panic!("pre-admission failure did not return the exact output"),
    };
    assert_eq!(Arc::as_ptr(returned.pool()) as usize, not_admitted_pool_ptr);
    assert_eq!(
        returned.output().surfaces.as_ptr() as usize,
        not_admitted_payload_ptr
    );
    assert!(direct_core.direct_mesh_retry_inbox.is_empty());
    direct_core.fail_next_mesh_event_reservation_for_test = true;
    assert!(matches!(
        direct_core.try_apply_mesh_output(returned),
        Err(MeshOutputApplyError::Admitted {
            error: VoxelTerrainRuntimeError::MeshOutputApplyFailed,
        })
    ));
    let DurableCompletion::DirectMesh {
        upload: admitted_upload,
        dropped: false,
    } = direct_core.direct_mesh_retry_inbox.front().unwrap()
    else {
        panic!("admitted direct owner changed variant")
    };
    let admitted_upload = Arc::clone(admitted_upload);
    assert_eq!(not_admitted_pool.idle_count(), 0);
    let direct_events = direct_core.try_drain_completed_tasks().unwrap();
    let direct_event_upload = direct_events
        .iter()
        .find_map(|event| match event {
            VoxelTerrainEvent::MeshBlockEntered(upload)
            | VoxelTerrainEvent::MeshBlockUpdated(upload)
            | VoxelTerrainEvent::MeshBlockBecameEmpty(upload)
                if upload.key() == direct_key =>
            {
                Some(upload)
            }
            _ => None,
        })
        .expect("retry published the admitted direct event");
    assert!(Arc::ptr_eq(direct_event_upload, &admitted_upload));
    assert!(Arc::ptr_eq(
        direct_core.mesh_maps[0][&direct_position]
            .accepted_upload()
            .unwrap(),
        &admitted_upload
    ));

    let CombinedCapacityFixture {
        mut core,
        desired,
        tracked,
        pools,
    } = build_combined_capacity_fixture();
    let load_position = Vector3i::new(700, 0, 0);
    let failed_save_key = SaveKey::new(Vector3i::new(710, 0, 0), 0);
    let old_state = core.paired_viewers[0].state.clone();
    let queued_dirty_position = old_state.data_box.iter_cells_zxy().next().unwrap();
    let queued_dirty_key = SaveKey::new(queued_dirty_position, 0);
    let dirty_generation = core.next_save_generation;
    let dirty_payload_ptr = core.data.with_lod_map(0, |map| {
        voxel_allocation_identity(map.get_block(queued_dirty_position).unwrap().voxels())
    });

    let DurableCompletion::LoadFinished {
        completed: load_owner,
        output: load_output,
    } = &core.durable_completion_inbox[0]
    else {
        panic!("combined load owner changed variant")
    };
    let load_task_ptr = task_owner_observation(load_owner).task;
    let load_payload_ptr =
        voxel_allocation_identity(load_output.block_data.voxels.as_ref().unwrap());
    let DurableCompletion::SaveAcknowledged {
        completed: save_owner,
        terminal: save_terminal,
        ..
    } = &core.durable_completion_inbox[1]
    else {
        panic!("combined save owner changed variant")
    };
    let save_task_ptr = task_owner_observation(save_owner).task;
    let save_payload_ptr = voxel_allocation_identity(&save_terminal.payload);
    let save_generation = save_terminal.save_generation;
    let DurableCompletion::UnknownTerminal {
        completed: unknown_owner,
    } = &core.durable_completion_inbox[2]
    else {
        panic!("combined quarantine owner changed variant")
    };
    let unknown_task_ptr = task_owner_observation(unknown_owner).task;
    let DurableCompletion::DirectMesh {
        upload: combined_upload,
        dropped: false,
    } = core.direct_mesh_retry_inbox.front().unwrap()
    else {
        panic!("combined direct owner changed variant")
    };
    let combined_upload = Arc::clone(combined_upload);
    let combined_key = combined_upload.key();
    let combined_pool_ptr = Arc::as_ptr(&pools[0]) as usize;

    let before = FixedObservableSnapshot::capture(&core, &tracked, &pools, &[], None);
    core.fixed_after_prepare_data_conflict_for_test = Some(load_position);
    assert_eq!(
        core.prepare_fixed_viewer_transaction(&[desired], true, false, false)
            .unwrap_err(),
        VoxelTerrainRuntimeError::DataMutation(
            SharedVoxelDataMutationError::ConcurrentDataMutation {
                position: load_position,
                lod_index: 0,
                expected_revision: VoxelDataKeyRevision::Tombstone(0),
                actual_revision: VoxelDataKeyRevision::Present(1),
            }
        )
    );
    assert_eq!(
        task_owner_observation(completed_for_durable(&core.durable_completion_inbox[0]).unwrap())
            .task,
        load_task_ptr
    );
    assert_eq!(
        task_owner_observation(completed_for_durable(&core.durable_completion_inbox[1]).unwrap())
            .task,
        save_task_ptr
    );
    core.data.with_lod_map_mut(0, |map| {
        assert!(map.remove_block(load_position).is_some());
        map.set_key_revision_for_test(load_position, 0);
    });
    assert_eq!(
        FixedObservableSnapshot::capture(&core, &tracked, &pools, &[], None),
        before,
        "late C1 conflict changed an exact owner"
    );

    core.prepare_fixed_viewer_transaction(&[desired], true, false, false)
        .unwrap();
    assert!(core.durable_completion_inbox.is_empty());
    assert!(core.direct_mesh_retry_inbox.is_empty());
    assert_eq!(
        core.data.with_lod_map(0, |map| voxel_allocation_identity(
            map.get_block(load_position).unwrap().voxels()
        )),
        load_payload_ptr
    );
    let failed_save = &core.save_journal[&failed_save_key];
    let Some(ActiveSaveAttempt::Pending(failed_pending)) = failed_save.active.as_ref() else {
        panic!("failed save did not retain its terminal payload")
    };
    assert_eq!(failed_pending.meta.generation, save_generation);
    assert_eq!(
        voxel_allocation_identity(&failed_pending.payload),
        save_payload_ptr
    );
    let dirty_journal = &core.save_journal[&queued_dirty_key];
    let dirty_pending = dirty_journal
        .queued_newer
        .iter()
        .find(|pending| pending.meta.generation == dirty_generation)
        .expect("dirty exit moved into the busy journal queue");
    assert_eq!(
        voxel_allocation_identity(&dirty_pending.payload),
        dirty_payload_ptr
    );
    assert!(core.data.block_snapshot(queued_dirty_position, 0).is_none());
    let combined_entry = &core.mesh_maps[0][&combined_key.location.position_in_blocks];
    assert!(Arc::ptr_eq(
        combined_entry.accepted_upload().unwrap(),
        &combined_upload
    ));
    let combined_event = core
        .event_outbox
        .iter()
        .find_map(|event| match event {
            VoxelTerrainEvent::MeshBlockEntered(upload)
            | VoxelTerrainEvent::MeshBlockUpdated(upload)
            | VoxelTerrainEvent::MeshBlockBecameEmpty(upload)
                if upload.key() == combined_key =>
            {
                Some(upload)
            }
            _ => None,
        })
        .expect("combined retry published the admitted mesh event");
    assert!(Arc::ptr_eq(combined_event, &combined_upload));
    assert_eq!(Arc::as_ptr(&pools[0]) as usize, combined_pool_ptr);
    assert_eq!(pools[0].idle_count(), 0);
    assert_eq!(core.completion_quarantine.len(), 1);
    assert_eq!(
        task_owner_observation(core.completion_quarantine[0].completed()).task,
        unknown_task_ptr
    );
}

#[derive(Default)]
struct ResolverCountingStream {
    save_calls: AtomicUsize,
    flush_calls: AtomicUsize,
}

impl VoxelStream for ResolverCountingStream {
    fn save_voxel_block(&self, _query: crate::streams::VoxelSaveQuery<'_>) -> StreamResult<()> {
        self.save_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn flush(&self) -> StreamResult<()> {
        self.flush_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum ResolverKind {
    Save,
    Flush,
}

#[derive(Debug, Clone, Copy)]
enum ResolverLateFailure {
    KeyConflict,
    LiveMapCapacity,
}

struct ResolverFixture {
    core: VoxelTerrainCore,
    operation: PersistenceOperation,
    load_position: Vector3i,
    load_payload: usize,
    stream: Arc<ResolverCountingStream>,
}

fn build_resolver_fixture(kind: ResolverKind) -> ResolverFixture {
    let mut core = build_core();
    let stream = Arc::new(ResolverCountingStream::default());
    core.stream = stream.clone();
    let load_position = Vector3i::new(780, 0, 0);
    install_finished_load(&mut core, load_position, 991, 0xC390);
    core.loading_blocks[0]
        .get_mut(&load_position)
        .unwrap()
        .residency
        .resident_viewers = 1;
    let DurableCompletion::LoadFinished { output, .. } =
        core.durable_completion_inbox.front().unwrap()
    else {
        panic!("resolver sentinel load changed variant")
    };
    let load_payload = voxel_allocation_identity(output.block_data.voxels.as_ref().unwrap());

    let operation = match kind {
        ResolverKind::Save => {
            let location = BlockLocation {
                position: Vector3i::new(781, 0, 0),
                lod_index: 0,
            };
            let block_revision = 1000;
            let generation = 1001;
            let attempt = 1002;
            core.save_journal.insert(
                SaveKey::new(location.position, location.lod_index),
                SaveJournalEntry {
                    written_unflushed: None,
                    active: Some(ActiveSaveAttempt::Indeterminate {
                        meta: SaveAttemptMeta {
                            block_revision,
                            generation,
                            retry_count: 0,
                            last_error: None,
                        },
                        attempt_ordinal: attempt,
                    }),
                    queued_newer: VecDeque::new(),
                },
            );
            core.completion_quarantine
                .push_back(QuarantinedCompletion::Persistence {
                    kind: PersistenceTaskKind::Save,
                    terminal: PersistenceTaskTerminal::Save(SaveTaskTerminal {
                        location,
                        block_revision,
                        save_generation: generation,
                        payload: terminal_payload(0xC391),
                        task_panic_phase: Some(TaskPanicPhase::Run),
                        phase: PersistenceIoPhase::CallEntered,
                        acknowledgement: None,
                    }),
                    attempt_ordinal: attempt,
                    completed: completion_owner_without_followups(TaskLane::Serial),
                });
            PersistenceOperation::Save {
                location,
                block_revision,
                save_generation: generation,
            }
        }
        ResolverKind::Flush => {
            let key = SaveKey::new(Vector3i::new(782, 0, 0), 0);
            let block_revision = 1010;
            core.save_journal.insert(
                key,
                SaveJournalEntry {
                    written_unflushed: Some(WrittenSave {
                        block_revision,
                        generation: 1011,
                        payload: terminal_payload(0xC392),
                    }),
                    active: None,
                    queued_newer: VecDeque::new(),
                },
            );
            let checkpoint_generation = 1012;
            let attempt = 1013;
            core.save_checkpoint_in_flight = Some(SaveCheckpointInFlight {
                checkpoint_generation,
                acknowledged: vec![SaveCheckpointSnapshot {
                    key,
                    block_revision,
                    generation: 1011,
                }],
                state: CheckpointAttemptState::Indeterminate {
                    attempt_ordinal: attempt,
                },
                retry_count: 0,
                max_attempts: 4,
                origin: CheckpointOrigin::Explicit,
                record_per_block_failure: false,
            });
            core.completion_quarantine
                .push_back(QuarantinedCompletion::Persistence {
                    kind: PersistenceTaskKind::Flush,
                    terminal: PersistenceTaskTerminal::Flush(FlushTaskTerminal {
                        checkpoint_generation,
                        task_panic_phase: Some(TaskPanicPhase::Run),
                        phase: PersistenceIoPhase::CallEntered,
                        acknowledgement: None,
                    }),
                    attempt_ordinal: attempt,
                    completed: completion_owner_without_followups(TaskLane::Serial),
                });
            PersistenceOperation::Flush {
                checkpoint_generation,
            }
        }
    };
    ResolverFixture {
        core,
        operation,
        load_position,
        load_payload,
        stream,
    }
}

#[test]
fn fixed_indeterminate_resolver_is_atomic_with_late_c1_failure() {
    for kind in [ResolverKind::Save, ResolverKind::Flush] {
        for resolution in [
            IndeterminateIoResolution::AssumeNotWrittenAndRetry,
            IndeterminateIoResolution::AssumeWrittenAndFlush,
        ] {
            for late_failure in [
                ResolverLateFailure::KeyConflict,
                ResolverLateFailure::LiveMapCapacity,
            ] {
                let ResolverFixture {
                    mut core,
                    operation,
                    load_position,
                    load_payload,
                    stream,
                } = build_resolver_fixture(kind);
                match late_failure {
                    ResolverLateFailure::KeyConflict => {
                        core.fixed_after_prepare_data_conflict_for_test = Some(load_position);
                    }
                    ResolverLateFailure::LiveMapCapacity => {
                        core.data.set_test_transaction_reservation_failpoint(Some(
                            SharedVoxelDataTransactionReservation::LiveMap,
                        ))
                    }
                }
                let tracked = [BlockLocation {
                    position: load_position,
                    lod_index: 0,
                }];
                let before = FixedObservableSnapshot::capture(&core, &tracked, &[], &[], None);
                let expected = match late_failure {
                    ResolverLateFailure::KeyConflict => VoxelTerrainRuntimeError::DataMutation(
                        SharedVoxelDataMutationError::ConcurrentDataMutation {
                            position: load_position,
                            lod_index: 0,
                            expected_revision: VoxelDataKeyRevision::Tombstone(0),
                            actual_revision: VoxelDataKeyRevision::Present(1),
                        },
                    ),
                    ResolverLateFailure::LiveMapCapacity => {
                        VoxelTerrainRuntimeError::DataMutation(
                            SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                                reservation: SharedVoxelDataTransactionReservation::LiveMap,
                            },
                        )
                    }
                };
                assert_eq!(
                    core.try_resolve_indeterminate_persistence(operation, resolution)
                        .unwrap_err(),
                    expected,
                    "wrong resolver failure for {kind:?}/{resolution:?}/{late_failure:?}"
                );
                assert_eq!(stream.save_calls.load(Ordering::SeqCst), 0);
                assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 0);
                match late_failure {
                    ResolverLateFailure::KeyConflict => core.data.with_lod_map_mut(0, |map| {
                        assert!(map.remove_block(load_position).is_some());
                        map.set_key_revision_for_test(load_position, 0);
                    }),
                    ResolverLateFailure::LiveMapCapacity => {
                        core.data.set_test_transaction_reservation_failpoint(None)
                    }
                }
                assert_eq!(
                    FixedObservableSnapshot::capture(&core, &tracked, &[], &[], None),
                    before,
                    "resolver failure changed journal/checkpoint/quarantine owner"
                );

                core.try_resolve_indeterminate_persistence(operation, resolution)
                    .unwrap();
                core.wait_for_pending_tasks();
                assert!(core.completion_quarantine.is_empty());
                assert!(core.durable_completion_inbox.is_empty());
                assert_eq!(
                    core.data.with_lod_map(0, |map| voxel_allocation_identity(
                        map.get_block(load_position).unwrap().voxels()
                    )),
                    load_payload
                );
                let (expected_save_calls, expected_flush_calls) = match (kind, resolution) {
                    (ResolverKind::Save, IndeterminateIoResolution::AssumeNotWrittenAndRetry) => {
                        (1, 0)
                    }
                    (ResolverKind::Save, IndeterminateIoResolution::AssumeWrittenAndFlush)
                    | (ResolverKind::Flush, IndeterminateIoResolution::AssumeNotWrittenAndRetry) => {
                        (0, 1)
                    }
                    (ResolverKind::Flush, IndeterminateIoResolution::AssumeWrittenAndFlush) => {
                        (0, 0)
                    }
                };
                assert_eq!(
                    stream.save_calls.load(Ordering::SeqCst),
                    expected_save_calls,
                    "unexpected save reissue for {kind:?}/{resolution:?}"
                );
                assert_eq!(
                    stream.flush_calls.load(Ordering::SeqCst),
                    expected_flush_calls,
                    "unexpected flush reissue for {kind:?}/{resolution:?}"
                );
            }
        }
    }
}

#[derive(Debug)]
struct NoLodSupportMesher;

impl VoxelMesher for NoLodSupportMesher {
    fn build(&self, _output: &mut MesherOutput, _input: &crate::meshers::MesherInput<'_>) {}

    fn supports_lod(&self) -> bool {
        false
    }
}

fn dormant_variable_lod_settings(lod_count: u8) -> crate::terrain::lod_clipbox::LodClipboxSettings {
    crate::terrain::lod_clipbox::LodClipboxSettings {
        data_block_size: 16,
        mesh_block_size: 16,
        lod_count,
        lod0_distance_voxels: 48,
        secondary_distance_voxels: 48,
        unload_hysteresis_blocks: 2,
    }
}

#[test]
fn data_residency_refs_report_each_checked_failure_without_mutation() {
    let location = BlockLocation {
        position: Vector3i::new(-3, 2, 7),
        lod_index: 1,
    };
    let max_viewers = DataResidencyRefs {
        resident_viewers: u32::MAX,
        coverage_holds: 0,
    };
    assert_eq!(
        max_viewers.checked_apply_delta(location, DataRefField::ResidentViewers, 1),
        Err(VoxelTerrainRuntimeError::DataRefcountOverflow {
            location,
            field: DataRefField::ResidentViewers,
        })
    );
    assert_eq!(max_viewers.resident_viewers, u32::MAX);

    let empty = DataResidencyRefs::default();
    assert_eq!(
        empty.checked_apply_delta(location, DataRefField::CoverageHolds, -1),
        Err(VoxelTerrainRuntimeError::DataRefcountUnderflow {
            location,
            field: DataRefField::CoverageHolds,
        })
    );
    assert_eq!(empty, DataResidencyRefs::default());

    let overflowing_total = DataResidencyRefs {
        resident_viewers: u32::MAX,
        coverage_holds: 1,
    };
    assert_eq!(
        overflowing_total.checked_total(location),
        Err(VoxelTerrainRuntimeError::DataRefcountOverflow {
            location,
            field: DataRefField::Total,
        })
    );
}

#[test]
fn mesh_residency_and_feature_holds_are_independent() {
    let mut entry = MeshBlockEntry {
        resident_viewers: 0,
        visual_viewers: 0,
        collision_viewers: 0,
        visual_coverage_holds: 1,
        collision_coverage_holds: 0,
        ..MeshBlockEntry::default()
    };
    assert_eq!(entry.resident_refcount(), 1);
    assert!(entry.needs_visual());
    assert!(!entry.needs_collision());

    entry.visual_coverage_holds = 0;
    entry.collision_coverage_holds = 2;
    assert_eq!(entry.resident_refcount(), 2);
    assert!(!entry.needs_visual());
    assert!(entry.needs_collision());

    entry.applied_features = MeshBuildFeatures {
        visuals: false,
        collisions: true,
        variable_lod: true,
    };
    assert!(entry.applied_features.collisions);
    assert!(!entry.applied_features.visuals);
}

#[test]
fn every_mesh_ref_field_is_checked_and_failure_is_non_mutating() {
    let location = MeshBlockLocation::new(Vector3i::new(4, -2, 9), 1);
    for field in [
        MeshRefField::ResidentViewers,
        MeshRefField::VisualViewers,
        MeshRefField::CollisionViewers,
        MeshRefField::VisualCoverageHolds,
        MeshRefField::CollisionCoverageHolds,
    ] {
        let mut overflowing = MeshBlockEntry::default();
        *overflowing.ref_field_mut(field) = u32::MAX;
        let before = (
            overflowing.resident_viewers,
            overflowing.visual_viewers,
            overflowing.collision_viewers,
            overflowing.visual_coverage_holds,
            overflowing.collision_coverage_holds,
        );
        assert_eq!(
            overflowing.checked_apply_ref_delta(location, field, 1),
            Err(VoxelTerrainRuntimeError::MeshRefcountOverflow { location, field })
        );
        assert_eq!(
            (
                overflowing.resident_viewers,
                overflowing.visual_viewers,
                overflowing.collision_viewers,
                overflowing.visual_coverage_holds,
                overflowing.collision_coverage_holds,
            ),
            before
        );

        let mut underflowing = MeshBlockEntry::default();
        assert_eq!(
            underflowing.checked_apply_ref_delta(location, field, -1),
            Err(VoxelTerrainRuntimeError::MeshRefcountUnderflow { location, field })
        );
        assert_eq!(underflowing.resident_refcount(), 0);
        assert!(!underflowing.needs_visual());
        assert!(!underflowing.needs_collision());
    }
}

#[test]
fn fixed_constructor_preserves_preloaded_higher_lods_and_builds_exact_sidecars() {
    let mut data = VoxelData::new();
    data.try_resize_lods_preserving(2).unwrap();
    let position = Vector3i::new(-5, 3, 9);
    let mut block = VoxelDataBlock::empty(1);
    block.viewers.set_exact(4);
    assert!(data.try_set_block(position, block));

    let mesher: Arc<dyn VoxelMesher> = Arc::new(AlwaysOneTriangleMesher);
    let core = VoxelTerrainCore::new(
        data,
        Arc::new(MemoryStream::new()),
        MeshingDependency::new(mesher, None),
    );

    assert_eq!(core.lod_count, 1);
    assert_eq!(core.data.lod_count(), 2);
    assert!(core.data.with_lod_map(1, |map| map.has_block(position)));
    assert_eq!(core.loaded_data_residency.len(), 2);
    assert_eq!(
        core.loaded_data_residency[1][&position],
        DataResidencyRefs {
            resident_viewers: 4,
            coverage_holds: 0,
        }
    );
}

#[test]
fn checked_variable_lod_constructor_preserves_maps_and_builds_runtime_state() {
    let mut data = VoxelData::new();
    data.try_resize_lods_preserving(2).unwrap();
    let negative = Vector3i::new(-3, 1, 2);
    let nonorigin = Vector3i::new(7, -4, 5);
    assert!(data.try_set_block(negative, VoxelDataBlock::empty(0)));
    let mut upper = VoxelDataBlock::empty(1);
    upper.viewers.set_exact(3);
    assert!(data.try_set_block(nonorigin, upper));

    let mesher: Arc<dyn VoxelMesher> = Arc::new(AlwaysOneTriangleMesher);
    let core = VoxelTerrainCore::new_variable_lod(
        data,
        Arc::new(MemoryStream::new()),
        MeshingDependency::new(mesher, None),
        dormant_variable_lod_settings(3),
    )
    .unwrap();

    assert_eq!(core.lod_count, 3);
    assert_eq!(core.data.lod_count(), 3);
    assert!(core.data.with_lod_map(0, |map| map.has_block(negative)));
    assert!(core.data.with_lod_map(1, |map| map.has_block(nonorigin)));
    assert_eq!(core.loaded_data_residency.len(), 3);
    assert_eq!(
        core.loaded_data_residency[1][&nonorigin],
        DataResidencyRefs {
            resident_viewers: 3,
            coverage_holds: 0,
        }
    );
    let runtime = core.variable_lod.as_ref().unwrap();
    assert_eq!(runtime.settings, dormant_variable_lod_settings(3));
    assert_eq!(runtime.coordinator.revision(), 0);
    assert_eq!(runtime.coverage.lod_count(), 3);
    assert!(runtime.coverage_holds.is_empty());
}

#[test]
fn checked_variable_lod_constructor_reports_validation_and_mesher_errors() {
    let mismatched = crate::terrain::lod_clipbox::LodClipboxSettings {
        mesh_block_size: 32,
        ..dormant_variable_lod_settings(3)
    };
    let mesher: Arc<dyn VoxelMesher> = Arc::new(AlwaysOneTriangleMesher);
    assert!(matches!(
        VoxelTerrainCore::new_variable_lod(
            VoxelData::new(),
            Arc::new(MemoryStream::new()),
            MeshingDependency::new(mesher, None),
            mismatched,
        ),
        Err(VariableLodConstructionError::LodMath(
            LodMathError::UnsupportedMeshToDataFactor { .. }
        ))
    ));

    let no_lod: Arc<dyn VoxelMesher> = Arc::new(NoLodSupportMesher);
    assert!(matches!(
        VoxelTerrainCore::new_variable_lod(
            VoxelData::new(),
            Arc::new(MemoryStream::new()),
            MeshingDependency::new(no_lod, None),
            dormant_variable_lod_settings(3),
        ),
        Err(VariableLodConstructionError::UnsupportedLodMesher)
    ));
}

#[test]
fn fixed_loaded_viewer_delta_and_unload_publish_exact_sidecar_changes() {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048)));
    let position = Vector3i::zero();
    assert!(data.try_set_block(position, VoxelDataBlock::empty(0)));
    let mesher: Arc<dyn VoxelMesher> = Arc::new(AlwaysOneTriangleMesher);
    let mut core = VoxelTerrainCore::new(
        data,
        Arc::new(MemoryStream::new()),
        MeshingDependency::new(mesher, None),
    );
    let viewer = fixed_zero_distance_viewer(1, position, MeshDemand::default());
    let viewers = normalize_and_validate_viewer_updates(&[viewer]).unwrap();

    core.prepare_fixed_viewer_transaction(&viewers, true, false, false)
        .unwrap();
    assert_eq!(
        core.loaded_data_residency[0][&position],
        DataResidencyRefs {
            resident_viewers: 1,
            coverage_holds: 0,
        }
    );
    assert_eq!(
        core.data.block_snapshot(position, 0).unwrap().viewers.get(),
        1
    );

    core.prepare_fixed_viewer_transaction(&[], true, false, false)
        .unwrap();
    assert!(core.data.block_snapshot(position, 0).is_none());
    assert!(!core.loaded_data_residency[0].contains_key(&position));
}

#[test]
fn fixed_loading_to_loaded_transfer_preserves_viewers_and_holds_independently() {
    let mut core = build_core();
    let position = Vector3i::new(19, -2, 7);
    let location = BlockLocation {
        position,
        lod_index: 0,
    };
    let generation = 41;
    core.loading_blocks[0].insert(
        position,
        LoadingBlockEntry {
            residency: DataResidencyRefs {
                resident_viewers: 1,
                coverage_holds: 1,
            },
            retry_count: 0,
            request_generation: generation,
            request_state: LoadRequestState::InFlight,
            physical_request: None,
        },
    );
    let mut task = LoadBlockForTerrainTask::new(
        position,
        0,
        generation,
        core.data.clone(),
        core.stream.clone(),
    );
    task.output = Some(loaded_output(&core, position, generation, 77));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));
    core.try_normalize_raw_completions().unwrap();
    core.prepare_fixed_viewer_transaction(&[], false, false, false)
        .unwrap();

    assert!(!core.loading_blocks[0].contains_key(&position));
    assert_eq!(
        core.loaded_data_residency[0][&position],
        DataResidencyRefs {
            resident_viewers: 1,
            coverage_holds: 1,
        }
    );
    assert_eq!(
        core.data.block_snapshot(position, 0).unwrap().viewers.get(),
        2
    );

    let state = ViewerState {
        data_box: Box3i::new(position, Vector3i::splat(1)),
        ..ViewerState::default()
    };
    core.paired_viewers.push(PairedViewer {
        id: 9,
        state: state.clone(),
        prev_state: state,
    });
    core.prepare_fixed_viewer_transaction(&[], true, false, false)
        .unwrap();
    assert_eq!(
        core.loaded_data_residency[0][&position],
        DataResidencyRefs {
            resident_viewers: 0,
            coverage_holds: 1,
        }
    );
    assert_eq!(
        core.data.block_snapshot(position, 0).unwrap().viewers.get(),
        1
    );
    assert_eq!(
        core.loaded_data_residency[0][&position]
            .checked_total(location)
            .unwrap(),
        1
    );
}
