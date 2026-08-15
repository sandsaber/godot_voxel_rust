use super::*;

#[derive(Default)]
struct MatrixCountingPersistenceStream {
    save_calls: AtomicUsize,
    flush_calls: AtomicUsize,
    fail_save: bool,
    fail_flush: bool,
}

impl MatrixCountingPersistenceStream {
    fn new(fail_save: bool, fail_flush: bool) -> Self {
        Self {
            save_calls: AtomicUsize::new(0),
            flush_calls: AtomicUsize::new(0),
            fail_save,
            fail_flush,
        }
    }
}

impl VoxelStream for MatrixCountingPersistenceStream {
    fn save_voxel_block(&self, _query: crate::streams::VoxelSaveQuery<'_>) -> StreamResult<()> {
        self.save_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_save {
            Err(VoxelStreamError::Io(
                "matrix external save failure".to_owned(),
            ))
        } else {
            Ok(())
        }
    }

    fn flush(&self) -> StreamResult<()> {
        self.flush_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_flush {
            Err(VoxelStreamError::Io(
                "matrix external flush failure".to_owned(),
            ))
        } else {
            Ok(())
        }
    }
}

fn make_upload(
    pool: Arc<MeshArraysPool>,
    key: MeshBlockKey,
    features: MeshBuildFeatures,
    visual_triangles: usize,
    collision_triangles: usize,
) -> Arc<MeshUploadSnapshot> {
    let (upload, dropped) = pooled_mesh_output(
        pool,
        key,
        features,
        visual_triangles,
        collision_triangles,
        false,
    )
    .into_upload()
    .into_parts();
    assert!(!dropped);
    upload
}

fn insert_mesh_snapshot(
    core: &mut VoxelTerrainCore,
    position: Vector3i,
    upload: Arc<MeshUploadSnapshot>,
    demand: MeshDemand,
) {
    let key = upload.key();
    core.mesh_maps[0].insert(
        position,
        MeshBlockEntry {
            position,
            resident_viewers: 1,
            visual_viewers: u32::from(demand.visuals),
            collision_viewers: u32::from(demand.collisions),
            visual_coverage_holds: 0,
            collision_coverage_holds: 0,
            visual_active: demand.visuals && upload.visual_state() != PayloadState::NotBuilt,
            collision_active: demand.collisions
                && upload.collision_state() != PayloadState::NotBuilt,
            is_loaded: true,
            requested_revision: Some(key.revision),
            request_generation: 0,
            requested_features: upload.features(),
            applied_features: upload.features(),
            applied_revision: Some(key.revision),
            has_geometry: upload.visual_state() == PayloadState::NonEmpty,
            is_in_update_list: false,
            terminal_retry_count: 0,
            physical_request: None,
            accepted_upload: Some(upload),
        },
    );
}

fn topology_batch(event: &VoxelTerrainEvent) -> &RenderTopologyBatch {
    let VoxelTerrainEvent::RenderTopologyChanged(batch) = event else {
        panic!("expected topology event, got {event:?}")
    };
    batch
}

fn sorted_locations(mut locations: Vec<MeshBlockLocation>) -> Vec<MeshBlockLocation> {
    locations.sort_unstable_by_key(|location| {
        (
            location.lod_index,
            location.position_in_blocks.x,
            location.position_in_blocks.y,
            location.position_in_blocks.z,
        )
    });
    locations
}

#[test]
fn fixed_event_fifo_is_committed_not_truncated_rollback() {
    let mut core = build_core();
    let old_event_position = Vector3i::new(-900, 0, 0);
    let exit_position = Vector3i::new(-40, 0, 0);
    let load_position = Vector3i::new(40, 0, 0);
    let enter_position = Vector3i::new(60, 0, 0);
    let empty_position = Vector3i::new(80, 0, 0);
    let visual_demand = MeshDemand {
        visuals: true,
        collisions: false,
    };
    let visual_features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };

    core.event_outbox
        .push_back(VoxelTerrainEvent::DataBlockUnloaded(BlockLocation {
            position: old_event_position,
            lod_index: 0,
        }));
    core.next_render_topology_revision = 700;

    assert!(core
        .data
        .try_set_block(exit_position, VoxelDataBlock::empty(0))
        .unwrap());
    core.data.with_lod_map_mut(0, |map| {
        map.get_block_mut(exit_position)
            .unwrap()
            .viewers
            .set_exact(1);
    });
    core.loaded_data_residency[0]
        .insert(exit_position, DataResidencyRefs::with_resident_viewers(1));
    let exiting_pool = Arc::new(MeshArraysPool::new());
    let exiting_key = MeshBlockKey {
        location: MeshBlockLocation::new(exit_position, 0),
        revision: 11,
    };
    let exiting_upload = make_upload(exiting_pool, exiting_key, visual_features, 1, 0);
    insert_mesh_snapshot(
        &mut core,
        exit_position,
        exiting_upload.clone(),
        visual_demand,
    );
    let exiting_state = ViewerState {
        data_box: single_block_box(exit_position),
        mesh_box: single_block_box(exit_position),
        demand: visual_demand,
        ..ViewerState::default()
    };
    core.paired_viewers.push(PairedViewer {
        id: 9,
        state: exiting_state.clone(),
        prev_state: exiting_state,
    });

    let load_generation = 17;
    core.loading_blocks[0].insert(
        load_position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: load_generation,
            request_state: LoadRequestState::InFlight,
            physical_request: None,
        },
    );
    let mut load_task = LoadBlockForTerrainTask::new(
        load_position,
        0,
        load_generation,
        core.data.clone(),
        core.stream.clone(),
    );
    load_task.output = Some(loaded_output(&core, load_position, load_generation, 0xC213));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(load_task),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));

    let enter_key = MeshBlockKey {
        location: MeshBlockLocation::new(enter_position, 0),
        revision: 21,
    };
    core.mesh_maps[0].insert(
        enter_position,
        MeshBlockEntry {
            position: enter_position,
            resident_viewers: 1,
            visual_viewers: 1,
            requested_revision: Some(enter_key.revision),
            requested_features: visual_features,
            ..MeshBlockEntry::default()
        },
    );
    let enter_pool = Arc::new(MeshArraysPool::new());
    let enter_upload = make_upload(enter_pool.clone(), enter_key, visual_features, 1, 0);
    core.direct_mesh_retry_inbox
        .push_back(DurableCompletion::DirectMesh {
            upload: enter_upload.clone(),
            dropped: false,
        });

    let prior_empty_key = MeshBlockKey {
        location: MeshBlockLocation::new(empty_position, 0),
        revision: 30,
    };
    let empty_key = MeshBlockKey {
        location: prior_empty_key.location,
        revision: 31,
    };
    let empty_pool = Arc::new(MeshArraysPool::new());
    let prior_empty_upload =
        make_upload(empty_pool.clone(), prior_empty_key, visual_features, 1, 0);
    insert_mesh_snapshot(&mut core, empty_position, prior_empty_upload, visual_demand);
    {
        let entry = core.mesh_maps[0].get_mut(&empty_position).unwrap();
        entry.requested_revision = Some(empty_key.revision);
        entry.requested_features = visual_features;
    }
    let empty_upload = make_upload(empty_pool.clone(), empty_key, visual_features, 0, 0);
    core.direct_mesh_retry_inbox
        .push_back(DurableCompletion::DirectMesh {
            upload: empty_upload.clone(),
            dropped: false,
        });

    let stats_before = core.stats;
    core.fixed_after_prepare_data_conflict_for_test = Some(load_position);
    assert!(matches!(
        core.try_process(&[]),
        Err(VoxelTerrainRuntimeError::DataMutation(
            SharedVoxelDataMutationError::ConcurrentDataMutation { .. }
                | SharedVoxelDataMutationError::PreparedTransactionExpectedTombstone { .. }
        ))
    ));

    assert_eq!(core.event_outbox.len(), 1);
    assert!(matches!(
        core.event_outbox.front(),
        Some(VoxelTerrainEvent::DataBlockUnloaded(location))
            if *location == BlockLocation { position: old_event_position, lod_index: 0 }
    ));
    assert_eq!(core.stats, stats_before);
    assert_eq!(core.next_render_topology_revision, 700);
    assert_eq!(core.durable_completion_inbox.len(), 1);
    assert_eq!(core.direct_mesh_retry_inbox.len(), 2);
    for (completion, expected) in core
        .direct_mesh_retry_inbox
        .iter()
        .zip([&enter_upload, &empty_upload])
    {
        let DurableCompletion::DirectMesh { upload, dropped } = completion else {
            panic!("direct retry FIFO changed owner variant")
        };
        assert!(!dropped);
        assert!(Arc::ptr_eq(upload, expected));
    }
    assert_eq!(
        core.data
            .block_snapshot(exit_position, 0)
            .unwrap()
            .viewers
            .get(),
        1
    );
    assert!(Arc::ptr_eq(
        core.mesh_maps[0][&exit_position].accepted_upload().unwrap(),
        &exiting_upload
    ));
    assert!(core.mesh_maps[0][&enter_position]
        .accepted_upload()
        .is_none());

    core.data.with_lod_map_mut(0, |map| {
        assert!(map.remove_block(load_position).is_some());
    });
    let events = core.try_process(&[]).unwrap();

    assert_eq!(events.len(), 7, "exact committed FIFO: {events:?}");
    assert!(matches!(
        events[0],
        VoxelTerrainEvent::DataBlockUnloaded(location)
            if location == BlockLocation { position: old_event_position, lod_index: 0 }
    ));
    assert!(matches!(
        events[1],
        VoxelTerrainEvent::DataBlockUnloaded(location)
            if location == BlockLocation { position: exit_position, lod_index: 0 }
    ));
    assert!(matches!(
        events[2],
        VoxelTerrainEvent::DataBlockLoaded(location)
            if location == BlockLocation { position: load_position, lod_index: 0 }
    ));
    assert!(matches!(events[3], VoxelTerrainEvent::MeshBlockEntered(_)));
    assert!(matches!(
        events[4],
        VoxelTerrainEvent::MeshBlockBecameEmpty(_)
    ));
    assert!(matches!(
        events[6],
        VoxelTerrainEvent::MeshBlockExited(location)
            if location == MeshBlockLocation::new(exit_position, 0)
    ));

    let enter_event_upload = mesh_event_upload(&events[3]);
    let empty_event_upload = mesh_event_upload(&events[4]);
    assert!(Arc::ptr_eq(enter_event_upload, &enter_upload));
    assert!(Arc::ptr_eq(empty_event_upload, &empty_upload));
    assert!(Arc::ptr_eq(
        core.mesh_maps[0][&enter_position]
            .accepted_upload()
            .unwrap(),
        enter_event_upload
    ));
    assert!(Arc::ptr_eq(
        core.mesh_maps[0][&empty_position]
            .accepted_upload()
            .unwrap(),
        empty_event_upload
    ));
    assert_eq!(enter_pool.idle_count(), 0);
    assert_eq!(
        empty_pool.idle_count(),
        1,
        "the replaced upload returns one array while the exact empty upload stays live"
    );

    let topology = topology_batch(&events[5]);
    assert_eq!(topology.revision, 700);
    assert_eq!(
        topology.visual_activations(),
        vec![MeshBlockLocation::new(enter_position, 0)]
    );
    assert_eq!(
        topology.visual_deactivations(),
        vec![MeshBlockLocation::new(exit_position, 0)]
    );
    assert!(topology.collision_activations().is_empty());
    assert!(topology.collision_deactivations().is_empty());
    assert_eq!(core.next_render_topology_revision, 701);
    assert_eq!(core.stats.blocks_loaded, stats_before.blocks_loaded + 1);
    assert_eq!(core.stats.blocks_unloaded, stats_before.blocks_unloaded + 1);
    assert_eq!(core.stats.meshes_built, stats_before.meshes_built + 2);
    assert!(core.event_outbox.is_empty());
    assert!(core.durable_completion_inbox.is_empty());
    assert!(core.direct_mesh_retry_inbox.is_empty());
}

#[test]
fn direct_apply_events_survive_failed_process_and_drain_in_fifo_order() {
    let mut core = build_core();
    let positions = [Vector3i::new(120, 0, 0), Vector3i::new(121, 0, 0)];
    let old_event_position = Vector3i::new(-901, 0, 0);
    let features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };
    let mut pools = Vec::new();
    let mut keys = Vec::new();
    let mut expected_uploads = Vec::new();

    for (index, position) in positions.into_iter().enumerate() {
        let key = MeshBlockKey {
            location: MeshBlockLocation::new(position, 0),
            revision: 50 + index as u64,
        };
        core.mesh_maps[0].insert(
            position,
            MeshBlockEntry {
                position,
                resident_viewers: 1,
                visual_viewers: 1,
                requested_revision: Some(key.revision),
                requested_features: features,
                ..MeshBlockEntry::default()
            },
        );
        let pool = Arc::new(MeshArraysPool::new());
        core.fail_next_mesh_event_reservation_for_test = true;
        assert!(matches!(
            core.try_apply_mesh_output(pooled_mesh_output(
                pool.clone(),
                key,
                features,
                index + 1,
                0,
                false,
            )),
            Err(MeshOutputApplyError::Admitted {
                error: VoxelTerrainRuntimeError::MeshOutputApplyFailed,
            })
        ));
        let DurableCompletion::DirectMesh { upload, dropped } =
            core.direct_mesh_retry_inbox.back().unwrap()
        else {
            panic!("admitted output changed owner variant")
        };
        assert!(!dropped);
        assert_eq!(upload.key(), key);
        expected_uploads.push(upload.clone());
        pools.push(pool);
        keys.push(key);
    }
    assert_eq!(core.direct_mesh_retry_inbox.len(), 2);
    assert!(core.event_outbox.is_empty());
    assert_eq!(core.stats.meshes_built, 0);
    core.next_render_topology_revision = 900;
    core.event_outbox
        .push_back(VoxelTerrainEvent::DataBlockUnloaded(BlockLocation {
            position: old_event_position,
            lod_index: 0,
        }));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(DebugNameCollisionTask),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));
    core.fail_next_completion_normalization_for_test = true;

    assert!(matches!(
        core.try_process(&[]),
        Err(VoxelTerrainRuntimeError::CompletionNormalizationFailed)
    ));
    assert_eq!(core.event_outbox.len(), 1);
    assert!(matches!(
        core.event_outbox.front(),
        Some(VoxelTerrainEvent::DataBlockUnloaded(location))
            if *location == BlockLocation { position: old_event_position, lod_index: 0 }
    ));
    assert_eq!(core.stats.meshes_built, 0);
    assert_eq!(core.next_render_topology_revision, 900);
    assert_eq!(core.direct_mesh_retry_inbox.len(), 2);
    for ((completion, expected), pool) in core
        .direct_mesh_retry_inbox
        .iter()
        .zip(&expected_uploads)
        .zip(&pools)
    {
        let DurableCompletion::DirectMesh { upload, dropped } = completion else {
            panic!("failed process changed admitted owner variant")
        };
        assert!(!dropped);
        assert!(Arc::ptr_eq(upload, expected));
        assert_eq!(pool.idle_count(), 0);
    }
    for position in positions {
        assert!(core.mesh_maps[0][&position].accepted_upload().is_none());
    }

    let events = core.try_drain_completed_tasks().unwrap();
    assert_eq!(
        events.len(),
        4,
        "old/direct/direct/topology FIFO: {events:?}"
    );
    assert!(matches!(
        events[0],
        VoxelTerrainEvent::DataBlockUnloaded(location)
            if location == BlockLocation { position: old_event_position, lod_index: 0 }
    ));
    for index in 0..2 {
        assert_eq!(
            events[index + 1].mesh_descriptor(),
            Some(MeshLifecycleEventDescriptor {
                kind: MeshLifecycleEventKind::Entered,
                key: keys[index],
                features,
                visual_state: PayloadState::NonEmpty,
                collision_state: PayloadState::NotBuilt,
            })
        );
        let event_upload = mesh_event_upload(&events[index + 1]);
        assert!(Arc::ptr_eq(event_upload, &expected_uploads[index]));
        assert!(Arc::ptr_eq(
            core.mesh_maps[0][&positions[index]]
                .accepted_upload()
                .unwrap(),
            event_upload
        ));
        assert_eq!(pools[index].idle_count(), 0);
    }
    let topology = topology_batch(&events[3]);
    assert_eq!(topology.revision, 900);
    assert_eq!(
        sorted_locations(topology.visual_activations()),
        sorted_locations(
            positions
                .into_iter()
                .map(|position| MeshBlockLocation::new(position, 0))
                .collect()
        )
    );
    assert!(topology.visual_deactivations().is_empty());
    assert!(topology.collision_activations().is_empty());
    assert!(topology.collision_deactivations().is_empty());
    assert_eq!(core.next_render_topology_revision, 901);
    assert_eq!(core.stats.meshes_built, 2);
    assert!(core.event_outbox.is_empty());
    assert!(core.direct_mesh_retry_inbox.is_empty());
    drop(events);
    for pool in pools {
        assert_eq!(
            pool.idle_count(),
            0,
            "each live entry retains its exact accepted upload after event drain"
        );
    }
}

#[test]
fn fixed_lod_feature_activity_changes_on_process_without_remesh() {
    let mut core = build_core();
    let both = MeshDemand {
        visuals: true,
        collisions: true,
    };
    let update = fixed_zero_distance_viewer(42, Vector3i::zero(), both);
    let mut state = ViewerState {
        local_position_voxels: update.world_position_voxels,
        horizontal_view_distance_voxels: update.horizontal_view_distance_voxels,
        vertical_view_distance_voxels: update.vertical_view_distance_voxels,
        demand: both,
        ..ViewerState::default()
    };
    compute_viewer_boxes(&mut state, core.data_block_size(), core.data_block_size());
    core.paired_viewers.push(PairedViewer {
        id: update.id,
        state: state.clone(),
        prev_state: state.clone(),
    });
    for position in state.data_box.iter_cells_zxy() {
        assert!(core
            .data
            .try_set_block(position, VoxelDataBlock::empty(0))
            .unwrap());
        core.data.with_lod_map_mut(0, |map| {
            map.get_block_mut(position).unwrap().viewers.set_exact(1);
        });
    }

    let features = MeshBuildFeatures {
        visuals: true,
        collisions: true,
        variable_lod: false,
    };
    let mut expected_uploads = Vec::new();
    let mut target_pool = None;
    let mut target_location = None;
    for (index, position) in state.mesh_box.iter_cells_zxy().enumerate() {
        let pool = Arc::new(MeshArraysPool::new());
        let key = MeshBlockKey {
            location: MeshBlockLocation::new(position, 0),
            revision: 1_000 + index as u64,
        };
        let upload = make_upload(pool.clone(), key, features, 1, 1);
        insert_mesh_snapshot(&mut core, position, upload.clone(), both);
        if target_location.is_none() {
            target_location = Some(key.location);
            target_pool = Some(pool.clone());
        }
        expected_uploads.push((position, upload));
    }
    let target_location = target_location.expect("non-empty fixed mesh box");
    let target_pool = target_pool.unwrap();
    let target_upload = expected_uploads
        .iter()
        .find(|(position, _)| *position == target_location.position_in_blocks)
        .unwrap()
        .1
        .clone();
    let payload_surfaces_token = target_upload.output().surfaces.as_ptr() as usize;
    let payload_indices_token = match &target_upload.output().surfaces[0].arrays {
        SurfaceArrays::Transvoxel(arrays) => arrays.indices.as_ptr() as usize,
        other => panic!("expected transvoxel renderer payload, got {other:?}"),
    };
    let pool_strong_count = Arc::strong_count(&target_pool);
    let upload_strong_count = Arc::strong_count(&target_upload);
    let mesh_revision_before = core.next_mesh_revision;
    let pending_mesh_before = core.blocks_pending_update[0].clone();
    let locations = sorted_locations(
        state
            .mesh_box
            .iter_cells_zxy()
            .map(|position| MeshBlockLocation::new(position, 0))
            .collect(),
    );

    let transitions = [
        (
            MeshDemand {
                visuals: true,
                collisions: false,
            },
            false,
            true,
        ),
        (
            MeshDemand {
                visuals: false,
                collisions: true,
            },
            true,
            true,
        ),
        (both, true, false),
        (MeshDemand::default(), true, true),
    ];
    for (expected_topology_revision, (demand, visual_changes, collision_changes)) in
        (core.next_render_topology_revision..).zip(transitions)
    {
        let mut next = update;
        next.demand = demand;
        let events = core.try_process(&[next]).unwrap();
        assert_eq!(events.len(), 1, "demand-only events: {events:?}");
        let batch = topology_batch(&events[0]);
        assert_eq!(batch.revision, expected_topology_revision);

        let expected_visual = if visual_changes {
            locations.clone()
        } else {
            Vec::new()
        };
        let expected_collision = if collision_changes {
            locations.clone()
        } else {
            Vec::new()
        };
        if demand.visuals {
            assert_eq!(
                sorted_locations(batch.visual_activations()),
                expected_visual
            );
            assert!(batch.visual_deactivations().is_empty());
        } else {
            assert_eq!(
                sorted_locations(batch.visual_deactivations()),
                expected_visual
            );
            assert!(batch.visual_activations().is_empty());
        }
        if demand.collisions {
            assert_eq!(
                sorted_locations(batch.collision_activations()),
                expected_collision
            );
            assert!(batch.collision_deactivations().is_empty());
        } else {
            assert_eq!(
                sorted_locations(batch.collision_deactivations()),
                expected_collision
            );
            assert!(batch.collision_activations().is_empty());
        }

        assert_eq!(core.next_mesh_revision, mesh_revision_before);
        assert_eq!(core.blocks_pending_update[0], pending_mesh_before);
        assert_eq!(core.pending_task_count(), 0);
        for (position, expected) in &expected_uploads {
            let entry = &core.mesh_maps[0][position];
            let live = entry.accepted_upload().unwrap();
            assert_eq!(live.key(), expected.key());
            assert!(Arc::ptr_eq(live, expected));
            assert_eq!(entry.resident_viewers, 1);
            assert_eq!(entry.visual_viewers, u32::from(demand.visuals));
            assert_eq!(entry.collision_viewers, u32::from(demand.collisions));
            assert_eq!(entry.visual_active, demand.visuals);
            assert_eq!(entry.collision_active, demand.collisions);
            assert_eq!(entry.requested_revision, Some(expected.key().revision));
            assert_eq!(entry.applied_revision, Some(expected.key().revision));
        }
        let live_target = core.mesh_maps[0][&target_location.position_in_blocks]
            .accepted_upload()
            .unwrap();
        assert!(Arc::ptr_eq(live_target, &target_upload));
        assert_eq!(
            live_target.output().surfaces.as_ptr() as usize,
            payload_surfaces_token
        );
        let live_indices_token = match &live_target.output().surfaces[0].arrays {
            SurfaceArrays::Transvoxel(arrays) => arrays.indices.as_ptr() as usize,
            other => panic!("expected transvoxel renderer payload, got {other:?}"),
        };
        assert_eq!(live_indices_token, payload_indices_token);
        assert_eq!(target_pool.idle_count(), 0);
        assert_eq!(Arc::strong_count(&target_pool), pool_strong_count);
        assert_eq!(Arc::strong_count(&target_upload), upload_strong_count);
    }
}

fn stage_conflicting_load_completion(
    core: &mut VoxelTerrainCore,
    position: Vector3i,
    generation: u64,
) {
    core.loading_blocks[0].insert(
        position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
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
    task.output = Some(loaded_output(core, position, generation, 0xC216));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));
}

fn remove_injected_conflict(core: &VoxelTerrainCore, position: Vector3i) {
    core.data.with_lod_map_mut(0, |map| {
        assert!(map.remove_block(position).is_some());
    });
}

fn run_external_save_completion_case(fails: bool, case_index: i32) {
    let stream = Arc::new(MatrixCountingPersistenceStream::new(fails, false));
    let mut core = build_core_with_stream(stream.clone());
    let save_location = BlockLocation {
        position: Vector3i::new(200 + case_index, 0, 0),
        lod_index: 0,
    };
    let block_revision = 300 + case_index as u64;
    let generation = 400 + case_index as u64;
    let attempt = 800 + case_index as u64;
    let operation = PersistenceOperation::Save {
        location: save_location,
        block_revision,
        save_generation: generation,
    };
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let mut task = stage_c2b_save_attempt_at_revision(
        &mut core,
        stream.clone(),
        save_location,
        block_revision,
        generation,
        attempt,
        payload,
    );
    assert!(matches!(
        task.run(ThreadedTaskContext::new(0, TaskPriority::max())),
        TaskRunStatus::Complete { .. }
    ));
    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 1);
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Serial,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));
    let conflict_position = Vector3i::new(300 + case_index, 0, 0);
    stage_conflicting_load_completion(&mut core, conflict_position, 900 + case_index as u64);

    core.fixed_after_prepare_data_conflict_for_test = Some(conflict_position);
    assert!(matches!(
        core.try_drain_completed_tasks(),
        Err(VoxelTerrainRuntimeError::DataMutation(
            SharedVoxelDataMutationError::ConcurrentDataMutation { .. }
                | SharedVoxelDataMutationError::PreparedTransactionExpectedTombstone { .. }
        ))
    ));
    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        core.journal_persistence_state_for_test(operation),
        Some(JournalPersistenceState::WriteInFlight)
    );
    assert_eq!(core.journal_payload_ptr_for_test(operation), None);
    assert_eq!(core.durable_completion_inbox.len(), 2);
    assert!(core.event_outbox.is_empty());
    assert!(core.save_error_for_test(operation).is_none());
    remove_injected_conflict(&core, conflict_position);
    let events = core.try_drain_completed_tasks().unwrap();
    assert_eq!(events.len(), 1);
    assert!(matches!(
        events[0],
        VoxelTerrainEvent::DataBlockLoaded(location)
            if location == BlockLocation { position: conflict_position, lod_index: 0 }
    ));
    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 1);
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 0);
    assert!(core.durable_completion_inbox.is_empty());
    assert_eq!(core.pending_task_count(), 0);
    if fails {
        assert_eq!(
            core.journal_persistence_state_for_test(operation),
            Some(JournalPersistenceState::PendingWrite)
        );
        assert_eq!(
            core.journal_payload_ptr_for_test(operation),
            Some(payload_ptr)
        );
        assert!(matches!(
            core.save_error_for_test(operation),
            Some(VoxelStreamError::Io(message)) if message == "matrix external save failure"
        ));
    } else {
        assert_eq!(
            core.journal_persistence_state_for_test(operation),
            Some(JournalPersistenceState::WrittenUnflushed)
        );
        assert_eq!(
            core.journal_payload_ptr_for_test(operation),
            Some(payload_ptr)
        );
        assert!(core.save_error_for_test(operation).is_none());
    }
}

fn run_external_flush_completion_case(fails: bool, case_index: i32) {
    let stream = Arc::new(MatrixCountingPersistenceStream::new(false, fails));
    let mut core = build_core_with_stream(stream.clone());
    let save_location = BlockLocation {
        position: Vector3i::new(400 + case_index, 0, 0),
        lod_index: 0,
    };
    let block_revision = 450 + case_index as u64;
    let save_generation = 500 + case_index as u64;
    let checkpoint_generation = 600 + case_index as u64;
    let attempt = 1_000 + case_index as u64;
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let operation = stage_single_written_checkpoint_member_at_revision(
        &mut core,
        save_location,
        block_revision,
        save_generation,
        payload,
    );
    let mut task = stage_c2b_flush_attempt(
        &mut core,
        stream.clone(),
        checkpoint_generation,
        attempt,
        SaveCheckpointSnapshot {
            key: SaveKey::new(save_location.position, 0),
            block_revision,
            generation: save_generation,
        },
    );
    assert!(matches!(
        task.run(ThreadedTaskContext::new(0, TaskPriority::max())),
        TaskRunStatus::Complete { .. }
    ));
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Serial,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));
    let conflict_position = Vector3i::new(500 + case_index, 0, 0);
    stage_conflicting_load_completion(&mut core, conflict_position, 1_100 + case_index as u64);

    core.fixed_after_prepare_data_conflict_for_test = Some(conflict_position);
    assert!(matches!(
        core.try_drain_completed_tasks(),
        Err(VoxelTerrainRuntimeError::DataMutation(
            SharedVoxelDataMutationError::ConcurrentDataMutation { .. }
                | SharedVoxelDataMutationError::PreparedTransactionExpectedTombstone { .. }
        ))
    ));
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        core.checkpoint_persistence_state_for_test(PersistenceOperation::Flush {
            checkpoint_generation,
        }),
        Some(JournalPersistenceState::WriteInFlight)
    );
    assert_eq!(
        core.journal_persistence_state_for_test(operation),
        Some(JournalPersistenceState::WrittenUnflushed)
    );
    assert_eq!(core.durable_completion_inbox.len(), 2);
    assert!(core.event_outbox.is_empty());
    assert!(core.last_save_checkpoint_error().is_none());
    remove_injected_conflict(&core, conflict_position);
    let events = core.try_drain_completed_tasks().unwrap();
    assert_eq!(events.len(), 1);
    assert!(matches!(
        events[0],
        VoxelTerrainEvent::DataBlockLoaded(location)
            if location == BlockLocation { position: conflict_position, lod_index: 0 }
    ));
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 0);
    assert!(core.durable_completion_inbox.is_empty());
    assert_eq!(core.pending_task_count(), 0);
    assert_eq!(
        core.checkpoint_persistence_state_for_test(PersistenceOperation::Flush {
            checkpoint_generation,
        }),
        None
    );
    if fails {
        assert_eq!(
            core.journal_persistence_state_for_test(operation),
            Some(JournalPersistenceState::PendingWrite)
        );
        assert_eq!(
            core.journal_payload_ptr_for_test(operation),
            Some(payload_ptr)
        );
        assert!(matches!(
            core.last_save_checkpoint_error(),
            Some(VoxelStreamError::Io(message)) if message == "matrix external flush failure"
        ));
    } else {
        assert_eq!(core.journal_persistence_state_for_test(operation), None);
        assert!(core.last_save_checkpoint_error().is_none());
    }
}

#[test]
fn external_save_and_flush_completion_is_applied_without_reissuing_io() {
    run_external_save_completion_case(false, 0);
    run_external_save_completion_case(true, 1);
    run_external_flush_completion_case(false, 2);
    run_external_flush_completion_case(true, 3);
}
