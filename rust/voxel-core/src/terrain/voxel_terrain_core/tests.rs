mod c2c_c2_matrix_atomic_prefix;
mod c2c_c2_matrix_concurrency;

use super::*;
use crate::engine::MeshingDependency;
use crate::generators::base::{GenResult, VoxelGenerator, VoxelQueryData};
use crate::generators::simple::Flat;
use crate::meshers::{
    MeshBlockKey, MeshBlockLocation, MesherOutput, Surface, SurfaceArrays, TransvoxelMesher,
    VoxelMesher,
};
use crate::storage::voxel_data::SharedVoxelDataEditPhase;
use crate::storage::{ChannelId, VoxelData, VoxelDataBlock};
use crate::streams::{LoadResult, PersistenceIoPhase};
use crate::tasks::{TaskPriority, TaskRunStatus, ThreadedTask, ThreadedTaskContext};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

mod c2c_c2_matrix_events_persistence_tests;

/// A mesher that always emits one triangle, so we can tell from
/// `mesh_blocks()[pos].is_loaded` whether the paging loop ran end-to-end.
struct AlwaysOneTriangleMesher;
impl VoxelMesher for AlwaysOneTriangleMesher {
    fn build(&self, output: &mut MesherOutput, _input: &crate::meshers::MesherInput<'_>) {
        let mut arrays = crate::meshers::transvoxel::structures::MeshArrays::default();
        let a = arrays.add_vertex(
            crate::math::Vector3f::zero(),
            crate::math::Vector3f::new(0.0, 1.0, 0.0),
            0,
            0,
            0,
            crate::math::Vector3f::zero(),
        );
        arrays.indices.extend_from_slice(&[a, a, a]);
        output
            .surfaces
            .push(Surface::new(SurfaceArrays::Transvoxel(arrays), 0));
    }
}

struct DebugNameCollisionTask;

impl ThreadedTask for DebugNameCollisionTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        TaskRunStatus::Complete {
            follow_up_tasks: Vec::new(),
        }
    }

    fn debug_name(&self) -> &'static str {
        "MeshBlockTask"
    }
}

struct CompletionFollowUpTask {
    ran: Arc<AtomicBool>,
}

impl ThreadedTask for CompletionFollowUpTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        self.ran.store(true, Ordering::SeqCst);
        TaskRunStatus::Complete {
            follow_up_tasks: Vec::new(),
        }
    }
}

struct CountingCompletionFollowUpTask {
    runs: Arc<AtomicUsize>,
}

impl ThreadedTask for CountingCompletionFollowUpTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        self.runs.fetch_add(1, Ordering::SeqCst);
        TaskRunStatus::Complete {
            follow_up_tasks: Vec::new(),
        }
    }
}

struct VoxelDataLockProbeGenerator {
    data: std::sync::Weak<SharedVoxelData>,
}

impl VoxelGenerator for VoxelDataLockProbeGenerator {
    fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
        let data = self.data.upgrade().expect("voxel data still alive");
        let guard = data
            .try_lock()
            .expect("VoxelData lock must be released before generator calls");
        drop(guard);

        input.buffer.clear_channel_f(ChannelId::Sdf.index(), -0.5);
        GenResult::default()
    }
}

struct VoxelDataLockProbeStream {
    data: std::sync::Weak<SharedVoxelData>,
}

impl VoxelStream for VoxelDataLockProbeStream {
    fn load_voxel_block(
        &self,
        _query: crate::streams::VoxelLoadQuery<'_>,
    ) -> crate::streams::StreamResult<LoadResult> {
        let data = self.data.upgrade().expect("voxel data still alive");
        let guard = data
            .try_lock()
            .expect("VoxelData lock must be released before stream calls");
        drop(guard);
        Ok(LoadResult::NotFound)
    }
}

struct SlowNotFoundStream {
    delay: Duration,
}

impl VoxelStream for SlowNotFoundStream {
    fn load_voxel_block(
        &self,
        _query: crate::streams::VoxelLoadQuery<'_>,
    ) -> crate::streams::StreamResult<LoadResult> {
        thread::sleep(self.delay);
        Ok(LoadResult::NotFound)
    }
}

#[derive(Default)]
struct BlockingWorkerGate {
    state: Mutex<(bool, bool)>,
    changed: Condvar,
}

impl BlockingWorkerGate {
    fn enter_and_wait(&self) {
        let mut state = self.state.lock().unwrap();
        state.0 = true;
        self.changed.notify_all();
        while !state.1 {
            state = self.changed.wait(state).unwrap();
        }
    }

    fn wait_until_entered(&self) {
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut state = self.state.lock().unwrap();
        while !state.0 {
            let now = Instant::now();
            assert!(now < deadline, "blocking worker did not enter in time");
            let (next, timeout) = self
                .changed
                .wait_timeout(state, deadline.saturating_duration_since(now))
                .unwrap();
            state = next;
            assert!(
                !timeout.timed_out() || state.0,
                "blocking worker did not enter"
            );
        }
    }

    fn release(&self) {
        let mut state = self.state.lock().unwrap();
        state.1 = true;
        self.changed.notify_all();
    }
}

struct BlockingNotFoundStream {
    gate: Arc<BlockingWorkerGate>,
}

impl VoxelStream for BlockingNotFoundStream {
    fn load_voxel_block(
        &self,
        _query: crate::streams::VoxelLoadQuery<'_>,
    ) -> crate::streams::StreamResult<LoadResult> {
        self.gate.enter_and_wait();
        Ok(LoadResult::NotFound)
    }
}

struct BlockingMesher {
    gate: Arc<BlockingWorkerGate>,
}

impl VoxelMesher for BlockingMesher {
    fn build(&self, _output: &mut MesherOutput, _input: &crate::meshers::MesherInput<'_>) {
        self.gate.enter_and_wait();
    }
}

struct FailOnceLoadStream {
    load_attempts: AtomicUsize,
    inner: MemoryStream,
}

impl FailOnceLoadStream {
    fn new() -> Self {
        Self {
            load_attempts: AtomicUsize::new(0),
            inner: MemoryStream::new(),
        }
    }

    fn load_block(&self, position: Vector3i, lod: u8, out: &mut VoxelBuffer) -> LoadResult {
        self.inner.load_block(position, lod, out)
    }
}

impl VoxelStream for FailOnceLoadStream {
    fn save_voxel_block(
        &self,
        query: crate::streams::VoxelSaveQuery<'_>,
    ) -> crate::streams::StreamResult<()> {
        self.inner
            .save_voxel_block(crate::streams::VoxelSaveQuery::new(
                query.voxel_buffer,
                query.position_in_blocks,
                query.lod_index,
            ))
    }

    fn load_voxel_block(
        &self,
        query: crate::streams::VoxelLoadQuery<'_>,
    ) -> crate::streams::StreamResult<LoadResult> {
        if self.load_attempts.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(crate::streams::VoxelStreamError::Io(
                "injected one-shot load failure".into(),
            ));
        }
        self.inner.load_voxel_block(query)
    }

    fn flush(&self) -> crate::streams::StreamResult<()> {
        self.inner.flush()
    }
}

#[derive(Default)]
struct FlushCountingStream {
    flush_count: AtomicUsize,
}

struct PersistencePhaseStream {
    panic_save: bool,
    panic_flush: bool,
    save_calls: AtomicUsize,
    flush_calls: AtomicUsize,
}

impl PersistencePhaseStream {
    fn healthy() -> Self {
        Self {
            panic_save: false,
            panic_flush: false,
            save_calls: AtomicUsize::new(0),
            flush_calls: AtomicUsize::new(0),
        }
    }

    fn panic_save() -> Self {
        Self {
            panic_save: true,
            ..Self::healthy()
        }
    }

    fn panic_flush() -> Self {
        Self {
            panic_flush: true,
            ..Self::healthy()
        }
    }
}

impl VoxelStream for PersistencePhaseStream {
    fn save_voxel_block(
        &self,
        _query: crate::streams::VoxelSaveQuery<'_>,
    ) -> crate::streams::StreamResult<()> {
        self.save_calls.fetch_add(1, Ordering::SeqCst);
        if self.panic_save {
            panic!("injected save I/O panic");
        }
        Ok(())
    }

    fn flush(&self) -> crate::streams::StreamResult<()> {
        self.flush_calls.fetch_add(1, Ordering::SeqCst);
        if self.panic_flush {
            panic!("injected flush I/O panic");
        }
        Ok(())
    }
}

impl VoxelStream for FlushCountingStream {
    fn flush(&self) -> crate::streams::StreamResult<()> {
        self.flush_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct FailThenMemoryStream {
    fails_remaining: AtomicUsize,
    inner: MemoryStream,
}

impl FailThenMemoryStream {
    fn new(fails: usize) -> Self {
        Self {
            fails_remaining: AtomicUsize::new(fails),
            inner: MemoryStream::new(),
        }
    }

    fn load_block(&self, position: Vector3i, lod: u8, out: &mut VoxelBuffer) -> LoadResult {
        self.inner.load_block(position, lod, out)
    }
}

impl VoxelStream for FailThenMemoryStream {
    fn save_voxel_block(
        &self,
        query: crate::streams::VoxelSaveQuery<'_>,
    ) -> crate::streams::StreamResult<()> {
        if self
            .fails_remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| v.checked_sub(1))
            .is_ok()
        {
            return Err(crate::streams::VoxelStreamError::Io(
                "injected save failure".into(),
            ));
        }
        self.inner
            .save_voxel_block(crate::streams::VoxelSaveQuery::new(
                query.voxel_buffer,
                query.position_in_blocks,
                query.lod_index,
            ))
    }

    fn load_voxel_block(
        &self,
        query: crate::streams::VoxelLoadQuery<'_>,
    ) -> crate::streams::StreamResult<LoadResult> {
        self.inner.load_voxel_block(query)
    }

    fn flush(&self) -> crate::streams::StreamResult<()> {
        self.inner.flush()
    }
}

struct ControlledFailureStream {
    fail_saves: AtomicBool,
    save_attempts: AtomicUsize,
    inner: MemoryStream,
}

impl ControlledFailureStream {
    fn new(fail_saves: bool) -> Self {
        Self {
            fail_saves: AtomicBool::new(fail_saves),
            save_attempts: AtomicUsize::new(0),
            inner: MemoryStream::new(),
        }
    }

    fn set_fail_saves(&self, fail_saves: bool) {
        self.fail_saves.store(fail_saves, Ordering::SeqCst);
    }

    fn save_attempts(&self) -> usize {
        self.save_attempts.load(Ordering::SeqCst)
    }

    fn load_block(&self, position: Vector3i, lod: u8, out: &mut VoxelBuffer) -> LoadResult {
        self.inner.load_block(position, lod, out)
    }
}

impl VoxelStream for ControlledFailureStream {
    fn save_voxel_block(
        &self,
        query: crate::streams::VoxelSaveQuery<'_>,
    ) -> crate::streams::StreamResult<()> {
        self.save_attempts.fetch_add(1, Ordering::SeqCst);
        if self.fail_saves.load(Ordering::SeqCst) {
            return if query.position_in_blocks.x == 1 {
                Err(crate::streams::VoxelStreamError::Io(
                    "/tmp/terrain/region_1_0_0.vxr: permission denied".into(),
                ))
            } else {
                Err(crate::streams::VoxelStreamError::CorruptData(
                    "invalid region header".into(),
                ))
            };
        }
        self.inner
            .save_voxel_block(crate::streams::VoxelSaveQuery::new(
                query.voxel_buffer,
                query.position_in_blocks,
                query.lod_index,
            ))
    }

    fn load_voxel_block(
        &self,
        query: crate::streams::VoxelLoadQuery<'_>,
    ) -> crate::streams::StreamResult<LoadResult> {
        self.inner.load_voxel_block(query)
    }

    fn flush(&self) -> crate::streams::StreamResult<()> {
        self.inner.flush()
    }
}

struct MixedOutcomeStream {
    failed_position: Vector3i,
    flush_attempts: AtomicUsize,
    inner: MemoryStream,
}

impl MixedOutcomeStream {
    fn new(failed_position: Vector3i) -> Self {
        Self {
            failed_position,
            flush_attempts: AtomicUsize::new(0),
            inner: MemoryStream::new(),
        }
    }

    fn flush_attempts(&self) -> usize {
        self.flush_attempts.load(Ordering::SeqCst)
    }

    fn load_block(&self, position: Vector3i, lod: u8, out: &mut VoxelBuffer) -> LoadResult {
        self.inner.load_block(position, lod, out)
    }
}

impl VoxelStream for MixedOutcomeStream {
    fn save_voxel_block(
        &self,
        query: crate::streams::VoxelSaveQuery<'_>,
    ) -> crate::streams::StreamResult<()> {
        if query.position_in_blocks == self.failed_position {
            return Err(crate::streams::VoxelStreamError::Io(
                "permanent block save failure".into(),
            ));
        }
        self.inner
            .save_voxel_block(crate::streams::VoxelSaveQuery::new(
                query.voxel_buffer,
                query.position_in_blocks,
                query.lod_index,
            ))
    }

    fn load_voxel_block(
        &self,
        query: crate::streams::VoxelLoadQuery<'_>,
    ) -> crate::streams::StreamResult<LoadResult> {
        self.inner.load_voxel_block(query)
    }

    fn flush(&self) -> crate::streams::StreamResult<()> {
        self.flush_attempts.fetch_add(1, Ordering::SeqCst);
        self.inner.flush()
    }
}

struct DiscardingBufferedStream {
    fail_next_flush: AtomicBool,
    flush_attempts: AtomicUsize,
    save_attempts: AtomicUsize,
    staging: Mutex<HashMap<(Vector3i, u8), VoxelBuffer>>,
    persisted: MemoryStream,
}

impl DiscardingBufferedStream {
    fn new() -> Self {
        Self::with_flush_failure(true)
    }

    fn healthy() -> Self {
        Self::with_flush_failure(false)
    }

    fn with_flush_failure(fail_next_flush: bool) -> Self {
        Self {
            fail_next_flush: AtomicBool::new(fail_next_flush),
            flush_attempts: AtomicUsize::new(0),
            save_attempts: AtomicUsize::new(0),
            staging: Mutex::new(HashMap::new()),
            persisted: MemoryStream::new(),
        }
    }

    fn flush_attempts(&self) -> usize {
        self.flush_attempts.load(Ordering::SeqCst)
    }

    fn save_attempts(&self) -> usize {
        self.save_attempts.load(Ordering::SeqCst)
    }

    fn load_block(&self, position: Vector3i, lod: u8, out: &mut VoxelBuffer) -> LoadResult {
        self.persisted.load_block(position, lod, out)
    }
}

impl VoxelStream for DiscardingBufferedStream {
    fn save_voxel_block(
        &self,
        query: crate::streams::VoxelSaveQuery<'_>,
    ) -> crate::streams::StreamResult<()> {
        self.save_attempts.fetch_add(1, Ordering::SeqCst);
        self.staging
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(
                (query.position_in_blocks, query.lod_index),
                query.voxel_buffer.copy_to_owned(),
            );
        Ok(())
    }

    fn load_voxel_block(
        &self,
        query: crate::streams::VoxelLoadQuery<'_>,
    ) -> crate::streams::StreamResult<LoadResult> {
        self.persisted.load_voxel_block(query)
    }

    fn flush(&self) -> crate::streams::StreamResult<()> {
        self.flush_attempts.fetch_add(1, Ordering::SeqCst);
        let mut staging = self
            .staging
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if self.fail_next_flush.swap(false, Ordering::SeqCst) {
            staging.clear();
            return Err(crate::streams::VoxelStreamError::Io(
                "buffered flush discarded staging".into(),
            ));
        }
        for ((position, lod), voxels) in staging.drain() {
            self.persisted
                .save_voxel_block(crate::streams::VoxelSaveQuery::new(&voxels, position, lod))?;
        }
        self.persisted.flush()
    }
}

struct CoordinatedBufferedStream {
    blocked_save_position: Option<Vector3i>,
    save_started: AtomicBool,
    save_released: Mutex<bool>,
    save_release: Condvar,
    block_flush: bool,
    flush_started: AtomicBool,
    flush_released: Mutex<bool>,
    flush_release: Condvar,
    fail_next_flush: AtomicBool,
    flush_attempts: AtomicUsize,
    save_attempts: AtomicUsize,
    staging: Mutex<HashMap<(Vector3i, u8), VoxelBuffer>>,
    persisted: MemoryStream,
}

impl CoordinatedBufferedStream {
    fn with_blocked_save(position: Vector3i) -> Self {
        Self {
            blocked_save_position: Some(position),
            save_started: AtomicBool::new(false),
            save_released: Mutex::new(false),
            save_release: Condvar::new(),
            block_flush: false,
            flush_started: AtomicBool::new(false),
            flush_released: Mutex::new(true),
            flush_release: Condvar::new(),
            fail_next_flush: AtomicBool::new(false),
            flush_attempts: AtomicUsize::new(0),
            save_attempts: AtomicUsize::new(0),
            staging: Mutex::new(HashMap::new()),
            persisted: MemoryStream::new(),
        }
    }

    fn with_blocked_flush(fail_next_flush: bool) -> Self {
        Self {
            blocked_save_position: None,
            save_started: AtomicBool::new(false),
            save_released: Mutex::new(true),
            save_release: Condvar::new(),
            block_flush: true,
            flush_started: AtomicBool::new(false),
            flush_released: Mutex::new(false),
            flush_release: Condvar::new(),
            fail_next_flush: AtomicBool::new(fail_next_flush),
            flush_attempts: AtomicUsize::new(0),
            save_attempts: AtomicUsize::new(0),
            staging: Mutex::new(HashMap::new()),
            persisted: MemoryStream::new(),
        }
    }

    fn wait_for_save_started(&self) {
        wait_for_atomic_flag(&self.save_started, "blocked save did not start");
    }

    fn release_save(&self) {
        *self
            .save_released
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = true;
        self.save_release.notify_all();
    }

    fn wait_for_flush_started(&self) {
        wait_for_atomic_flag(&self.flush_started, "blocked flush did not start");
    }

    fn release_flush(&self) {
        *self
            .flush_released
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = true;
        self.flush_release.notify_all();
    }

    fn save_attempts(&self) -> usize {
        self.save_attempts.load(Ordering::SeqCst)
    }

    fn flush_attempts(&self) -> usize {
        self.flush_attempts.load(Ordering::SeqCst)
    }

    fn wait_for_save_attempts_at_least(&self, expected: usize) -> bool {
        let deadline = Instant::now() + Duration::from_millis(100);
        while Instant::now() < deadline {
            if self.save_attempts() >= expected {
                return true;
            }
            thread::sleep(Duration::from_millis(1));
        }
        self.save_attempts() >= expected
    }

    fn load_block(&self, position: Vector3i, lod: u8, out: &mut VoxelBuffer) -> LoadResult {
        self.persisted.load_block(position, lod, out)
    }
}

impl VoxelStream for CoordinatedBufferedStream {
    fn save_voxel_block(
        &self,
        query: crate::streams::VoxelSaveQuery<'_>,
    ) -> crate::streams::StreamResult<()> {
        self.save_attempts.fetch_add(1, Ordering::SeqCst);
        if self.blocked_save_position == Some(query.position_in_blocks) {
            self.save_started.store(true, Ordering::SeqCst);
            let mut released = self
                .save_released
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            while !*released {
                released = self
                    .save_release
                    .wait(released)
                    .unwrap_or_else(|error| error.into_inner());
            }
        }
        self.staging
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(
                (query.position_in_blocks, query.lod_index),
                query.voxel_buffer.copy_to_owned(),
            );
        Ok(())
    }

    fn load_voxel_block(
        &self,
        query: crate::streams::VoxelLoadQuery<'_>,
    ) -> crate::streams::StreamResult<LoadResult> {
        self.persisted.load_voxel_block(query)
    }

    fn flush(&self) -> crate::streams::StreamResult<()> {
        self.flush_attempts.fetch_add(1, Ordering::SeqCst);
        self.flush_started.store(true, Ordering::SeqCst);
        if self.block_flush {
            let mut released = self
                .flush_released
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            while !*released {
                released = self
                    .flush_release
                    .wait(released)
                    .unwrap_or_else(|error| error.into_inner());
            }
        }

        let mut staging = self
            .staging
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if self.fail_next_flush.swap(false, Ordering::SeqCst) {
            staging.clear();
            return Err(crate::streams::VoxelStreamError::Io(
                "coordinated checkpoint failure".into(),
            ));
        }
        for ((position, lod), voxels) in staging.drain() {
            self.persisted
                .save_voxel_block(crate::streams::VoxelSaveQuery::new(&voxels, position, lod))?;
        }
        self.persisted.flush()
    }
}

fn wait_for_atomic_flag(flag: &AtomicBool, message: &str) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if flag.load(Ordering::SeqCst) {
            return;
        }
        thread::sleep(Duration::from_millis(1));
    }
    panic!("{message}");
}

fn build_core() -> VoxelTerrainCore {
    build_core_with_stream(Arc::new(MemoryStream::new()))
}

fn build_core_with_stream(stream: Arc<dyn VoxelStream>) -> VoxelTerrainCore {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048)));
    let flat = Flat {
        channel: ChannelId::Sdf,
        ..Flat::default()
    };
    let generator: Arc<dyn crate::generators::base::VoxelGenerator> = Arc::new(flat);
    data.set_generator(Some(generator));
    let mesher: Arc<dyn VoxelMesher> = Arc::new(AlwaysOneTriangleMesher);
    let meshing_dependency = MeshingDependency::new(mesher, None);
    VoxelTerrainCore::new(data, stream, meshing_dependency)
}

fn build_boundary_readiness_core(bounds: Box3i) -> VoxelTerrainCore {
    let mut data = VoxelData::new();
    data.set_bounds(bounds);
    let mesher: Arc<dyn VoxelMesher> = Arc::new(AlwaysOneTriangleMesher);
    VoxelTerrainCore::legacy_variable_lod_for_parity(
        data,
        Arc::new(MemoryStream::new()),
        MeshingDependency::new(mesher, None),
        3,
    )
}

fn insert_readiness_block(core: &VoxelTerrainCore, position: Vector3i, lod_index: u8) {
    let block = VoxelDataBlock::with_voxels(
        VoxelBuffer::with_size(Vector3i::splat(core.data_block_size())),
        lod_index,
    );
    assert!(core.data.try_set_block(position, block).unwrap());
}

#[test]
fn boundary_mesh_readiness_ignores_only_out_of_bounds_neighbors() {
    use crate::terrain::lod_clipbox::{bounds_in_lod_blocks, clipped_meshing_data_box};

    let bounds = Box3i::new(
        Vector3i::new(-256, -192, -128),
        Vector3i::new(512, 384, 256),
    );
    for lod_index in [0_u8, 1, 2] {
        let lod_bounds = bounds_in_lod_blocks(bounds, 16, lod_index).unwrap();
        let min = lod_bounds.position;
        let max = lod_bounds.position + lod_bounds.size - Vector3i::splat(1);
        let face_positions = [
            Vector3i::new(min.x, 0, 0),
            Vector3i::new(max.x, 0, 0),
            Vector3i::new(0, min.y, 0),
            Vector3i::new(0, max.y, 0),
            Vector3i::new(0, 0, min.z),
            Vector3i::new(0, 0, max.z),
        ];

        for (face_index, position) in face_positions.into_iter().enumerate() {
            let location = MeshBlockLocation::new(position, lod_index);
            let clipped_halo = clipped_meshing_data_box(location, 16, 1, bounds).unwrap();
            let missing_in_bounds = clipped_halo
                .iter_cells_zxy()
                .find(|candidate| *candidate != position)
                .expect("a boundary halo retains at least one in-bound neighbour");
            let core = build_boundary_readiness_core(bounds);
            for block_position in clipped_halo.iter_cells_zxy() {
                if block_position != missing_in_bounds {
                    insert_readiness_block(&core, block_position, lod_index);
                }
            }

            assert!(
                !core.meshing_data_is_ready(location).unwrap(),
                "lod={lod_index} face={face_index} missing={missing_in_bounds:?}"
            );
            insert_readiness_block(&core, missing_in_bounds, lod_index);
            assert!(
                core.meshing_data_is_ready(location).unwrap(),
                "lod={lod_index} face={face_index}"
            );
        }
    }
}

#[test]
fn boundary_mesh_readiness_rejects_invalid_lod_and_unrepresentable_bounds() {
    use crate::terrain::lod_clipbox::LodMathError;

    let core =
        build_boundary_readiness_core(Box3i::new(Vector3i::splat(-128), Vector3i::splat(256)));
    assert_eq!(
        core.meshing_data_is_ready(MeshBlockLocation::new(Vector3i::zero(), 3)),
        Err(LodMathError::InvalidLodCount)
    );

    let overflow_core =
        build_boundary_readiness_core(Box3i::new(Vector3i::splat(i32::MAX), Vector3i::splat(1)));
    assert_eq!(
        overflow_core.meshing_data_is_ready(MeshBlockLocation::new(Vector3i::splat(i32::MAX), 0)),
        Err(LodMathError::CoordinateOverflow)
    );
}

#[test]
fn mesh_data_is_ready_returns_false_outside_finite_bounds() {
    let bounds = Box3i::new(
        Vector3i::new(-256, -192, -128),
        Vector3i::new(512, 384, 256),
    );
    let core = build_boundary_readiness_core(bounds);
    assert!(!core
        .meshing_data_is_ready(MeshBlockLocation::new(Vector3i::new(-17, 0, 0), 0))
        .unwrap());
}

#[test]
fn boundary_mesh_readiness_fixed_load_completion_launches_mesh() {
    let bounds = Box3i::new(Vector3i::zero(), Vector3i::splat(16));
    let stream = Arc::new(MemoryStream::new());
    let voxels = VoxelBuffer::with_size(Vector3i::splat(16));
    stream
        .save_voxel_block(crate::streams::VoxelSaveQuery::new(
            &voxels,
            Vector3i::zero(),
            0,
        ))
        .unwrap();
    let mut data = VoxelData::new();
    data.set_bounds(bounds);
    let mesher: Arc<dyn VoxelMesher> = Arc::new(AlwaysOneTriangleMesher);
    let mut core = VoxelTerrainCore::new(data, stream, MeshingDependency::new(mesher, None));
    let viewer = fixed_zero_distance_viewer(
        301,
        Vector3i::zero(),
        MeshDemand {
            visuals: true,
            collisions: false,
        },
    );

    for _ in 0..12 {
        core.try_process(&[viewer]).unwrap();
        core.wait_for_pending_tasks();
        if core
            .mesh_blocks()
            .get(&Vector3i::zero())
            .is_some_and(|entry| entry.applied_revision.is_some())
        {
            break;
        }
    }

    assert!(core.data.block_snapshot(Vector3i::zero(), 0).is_some());
    assert!(
        core.mesh_blocks()
            .get(&Vector3i::zero())
            .is_some_and(|entry| entry.applied_revision.is_some()),
        "the in-bound block must mesh without waiting for out-of-bound neighbours"
    );
}

#[test]
fn boundary_mesh_readiness_variable_launches_at_lod1_and_lod2() {
    let bounds = Box3i::new(
        Vector3i::new(-256, -192, -128),
        Vector3i::new(512, 384, 256),
    );
    for lod_index in [1_u8, 2] {
        let mut core = build_boundary_readiness_core(bounds);
        let lod_bounds = bounds_in_lod_blocks(bounds, 16, lod_index).unwrap();
        let location =
            MeshBlockLocation::new(Vector3i::new(lod_bounds.position.x, 0, 0), lod_index);
        let halo = core.meshing_data_box(location).unwrap();
        for position in halo.iter_cells_zxy() {
            insert_readiness_block(&core, position, lod_index);
        }

        core.legacy_view_mesh_block(location.position_in_blocks, usize::from(lod_index));
        let requested_revision = core.mesh_maps[usize::from(lod_index)]
            [&location.position_in_blocks]
            .requested_revision
            .expect("the ready boundary block is queued by the schedule gate");
        core.process_meshing().unwrap();
        assert!(core.blocks_pending_update[usize::from(lod_index)].is_empty());
        assert!(
            !core.mesh_maps[usize::from(lod_index)][&location.position_in_blocks].is_in_update_list
        );

        core.wait_for_pending_tasks();
        core.try_drain_completed_tasks().unwrap();
        assert_eq!(
            core.mesh_maps[usize::from(lod_index)][&location.position_in_blocks].applied_revision,
            Some(requested_revision),
            "LOD {lod_index} boundary task must be accepted"
        );
    }
}

fn save_result(
    block_data: BlockDataOutput,
    stream_error: Option<crate::streams::VoxelStreamError>,
    voxels: Option<VoxelBuffer>,
) -> SaveTaskTerminal {
    SaveTaskTerminal {
        location: crate::storage::BlockLocation {
            position: block_data.position_in_blocks,
            lod_index: block_data.lod_index,
        },
        block_revision: 0,
        save_generation: block_data.save_generation,
        payload: voxels.unwrap_or_else(|| VoxelBuffer::with_size(Vector3i::splat(2))),
        task_panic_phase: None,
        phase: PersistenceIoPhase::Acknowledged,
        acknowledgement: Some(PersistenceAcknowledgement::Save(match stream_error {
            Some(error) => Err(error),
            None => Ok(()),
        })),
    }
}

fn stage_acknowledged_checkpoint_batch(
    core: &mut VoxelTerrainCore,
    stream: &dyn VoxelStream,
    y: i32,
) -> Vec<(Vector3i, u64)> {
    let channel = ChannelId::Type.index();
    let mut expected = Vec::new();
    for index in 0..AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD {
        let position = Vector3i::new(index as i32, y, 0);
        let marker = (index + 1) as u64;
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(2));
        voxels.set_voxel(marker, 1, 1, 1, channel);
        stream
            .save_voxel_block(crate::streams::VoxelSaveQuery::new(&voxels, position, 0))
            .unwrap();
        core.save_journal.insert(
            SaveKey::new(position, 0),
            SaveJournalEntry {
                written_unflushed: Some(WrittenSave {
                    block_revision: 0,
                    generation: 1,
                    payload: voxels,
                }),
                active: None,
                queued_newer: VecDeque::new(),
            },
        );
        expected.push((position, marker));
    }
    expected
}

fn build_core_with_materializable_data(stream: Arc<dyn VoxelStream>) -> VoxelTerrainCore {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048)));
    data.set_streaming_enabled(false);
    data.set_full_load_completed(true);
    let flat = Flat {
        channel: ChannelId::Sdf,
        ..Flat::default()
    };
    let generator: Arc<dyn crate::generators::base::VoxelGenerator> = Arc::new(flat);
    data.set_generator(Some(generator));
    let mesher: Arc<dyn VoxelMesher> = Arc::new(AlwaysOneTriangleMesher);
    let meshing_dependency = MeshingDependency::new(mesher, None);
    VoxelTerrainCore::new(data, stream, meshing_dependency)
}

struct OneVoxelPaddingMesher;

impl VoxelMesher for OneVoxelPaddingMesher {
    fn build(&self, _output: &mut MesherOutput, _input: &crate::meshers::MesherInput<'_>) {}

    fn minimum_padding(&self) -> u32 {
        1
    }

    fn maximum_padding(&self) -> u32 {
        1
    }
}

fn make_resident_edit_core() -> VoxelTerrainCore {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048)));
    data.set_streaming_enabled(false);
    data.set_full_load_completed(true);
    let mesher: Arc<dyn VoxelMesher> = Arc::new(OneVoxelPaddingMesher);
    let mut core = VoxelTerrainCore::new(
        data,
        Arc::new(MemoryStream::new()),
        MeshingDependency::new(mesher, None),
    );

    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                assert!(core
                    .data
                    .try_set_block(
                        Vector3i::new(x, y, z),
                        VoxelDataBlock::with_voxels(
                            VoxelBuffer::with_size(Vector3i::splat(core.data_block_size())),
                            0,
                        ),
                    )
                    .unwrap());
            }
        }
    }
    for z in 0..=1 {
        for y in 0..=1 {
            for x in 0..=1 {
                core.legacy_view_mesh_block(Vector3i::new(x, y, z), 0);
            }
        }
    }
    for entry in core.mesh_maps[0].values_mut() {
        entry.requested_revision = None;
        entry.is_in_update_list = false;
    }
    core.blocks_pending_update[0].clear();
    core
}

fn make_edit_core_with_lods(lod_count: u8) -> VoxelTerrainCore {
    make_edit_core_with_mesher_and_lods(Arc::new(OneVoxelPaddingMesher), lod_count)
}

fn make_edit_core_with_mesher_and_lods(
    mesher: Arc<dyn VoxelMesher>,
    lod_count: u8,
) -> VoxelTerrainCore {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048)));
    data.set_streaming_enabled(false);
    data.set_full_load_completed(true);
    VoxelTerrainCore::legacy_variable_lod_for_parity(
        data,
        Arc::new(MemoryStream::new()),
        MeshingDependency::new(mesher, None),
        lod_count,
    )
}

#[test]
fn batch_sphere_edit_writes_one_block_instead_of_per_voxel() {
    let mut core = make_edit_core_with_lods(1);
    let edited = core
        .try_edit_sphere(
            crate::math::Vector3f::new(8.0, 8.0, 8.0),
            3.0,
            ChannelId::Type.index(),
            crate::edition::EditMode::Set,
            7,
        )
        .expect("sphere edit should publish");
    assert_eq!(
        edited, 1,
        "a 3-radius sphere at block center spans one block"
    );
    let snapshot = core
        .data()
        .block_snapshot(Vector3i::zero(), 0)
        .expect("edited block is resident");
    assert!(snapshot.has_voxels());
    assert_eq!(
        snapshot
            .voxels()
            .get_voxel(8, 8, 8, ChannelId::Type.index()),
        7
    );
    assert_eq!(
        snapshot
            .voxels()
            .get_voxel(0, 0, 0, ChannelId::Type.index()),
        0
    );
}

#[test]
fn batch_box_edit_fills_inclusive_bounds() {
    let mut core = make_edit_core_with_lods(1);
    let edited = core
        .try_edit_box(
            Vector3i::new(1, 1, 1),
            Vector3i::new(2, 2, 2),
            ChannelId::Type.index(),
            crate::edition::EditMode::Set,
            5,
        )
        .expect("box edit should publish");
    assert_eq!(edited, 1);
    let snapshot = core
        .data()
        .block_snapshot(Vector3i::zero(), 0)
        .expect("edited block is resident");
    assert_eq!(
        snapshot
            .voxels()
            .get_voxel(1, 1, 1, ChannelId::Type.index()),
        5
    );
    assert_eq!(
        snapshot
            .voxels()
            .get_voxel(2, 2, 2, ChannelId::Type.index()),
        5
    );
    assert_eq!(
        snapshot
            .voxels()
            .get_voxel(3, 3, 3, ChannelId::Type.index()),
        0
    );
}

#[test]
fn batch_paste_copies_source_buffer_into_lod0() {
    use crate::storage::VoxelBuffer;
    let mut core = make_edit_core_with_lods(1);
    let mut src = VoxelBuffer::with_size(Vector3i::splat(2));
    src.set_voxel(9, 0, 0, 0, ChannelId::Type.index());
    src.set_voxel(8, 1, 1, 1, ChannelId::Type.index());
    let edited = core
        .try_paste(Vector3i::new(2, 2, 2), &src, 1 << ChannelId::Type.index())
        .expect("paste should publish");
    assert_eq!(edited, 1);
    let snapshot = core
        .data()
        .block_snapshot(Vector3i::zero(), 0)
        .expect("pasted block is resident");
    assert_eq!(
        snapshot
            .voxels()
            .get_voxel(2, 2, 2, ChannelId::Type.index()),
        9
    );
    assert_eq!(
        snapshot
            .voxels()
            .get_voxel(3, 3, 3, ChannelId::Type.index()),
        8
    );
}

#[test]
fn voxel_metadata_survives_block_edit_and_paste() {
    use crate::storage::{MetadataValue, VoxelBuffer};
    let mut core = make_edit_core_with_lods(1);
    assert!(core
        .try_edit_voxel_metadata(Vector3i::new(3, 4, 5), Some(MetadataValue::Int(99)))
        .expect("metadata edit should publish")
        .is_some());
    assert_eq!(
        core.voxel_metadata(Vector3i::new(3, 4, 5)),
        Some(MetadataValue::Int(99))
    );

    let mut visited = Vec::new();
    core.for_each_voxel_metadata_in_area(
        Vector3i::new(3, 4, 5),
        Vector3i::new(4, 5, 6),
        |pos, value| {
            visited.push((pos, value.clone()));
        },
    );
    assert_eq!(
        visited,
        vec![(Vector3i::new(3, 4, 5), MetadataValue::Int(99))]
    );

    let mut src = VoxelBuffer::with_size(Vector3i::splat(1));
    src.set_voxel(4, 0, 0, 0, ChannelId::Type.index());
    src.set_voxel_metadata(Vector3i::zero(), MetadataValue::Text("pasted".into()));
    core.try_paste(Vector3i::new(1, 1, 1), &src, 1 << ChannelId::Type.index())
        .expect("paste should publish");
    assert_eq!(
        core.voxel_metadata(Vector3i::new(1, 1, 1)),
        Some(MetadataValue::Text("pasted".into()))
    );

    assert!(core
        .try_edit_voxel_metadata(Vector3i::new(3, 4, 5), None)
        .expect("clear should publish")
        .is_some());
    assert!(core.voxel_metadata(Vector3i::new(3, 4, 5)).is_none());
}

#[test]
fn debug_snapshot_reports_volume_and_edited_block() {
    let mut core = make_edit_core_with_lods(1);
    let snapshot = core.debug_snapshot();
    assert!(snapshot.volume_bounds.size.x > 0);
    assert_eq!(snapshot.lod_count, 1);
    assert!(core
        .try_edit_voxel(7, Vector3i::new(1, 1, 1), ChannelId::Type.index())
        .expect("edit")
        .is_some());
    let snapshot = core.debug_snapshot();
    assert!(snapshot.data_block_count > 0);
    assert!(snapshot
        .edited_blocks
        .iter()
        .any(|block| block.position == Vector3i::zero() && block.modified));
}

fn requested_mesh_locations(core: &VoxelTerrainCore) -> Vec<MeshBlockLocation> {
    let mut locations = core
        .mesh_maps
        .iter()
        .enumerate()
        .flat_map(|(lod, mesh_map)| {
            mesh_map.iter().filter_map(move |(&position, entry)| {
                entry
                    .requested_revision
                    .map(|_| MeshBlockLocation::new(position, lod as u8))
            })
        })
        .collect::<Vec<_>>();
    locations.sort_by_key(|location| {
        (
            location.lod_index,
            location.position_in_blocks.x,
            location.position_in_blocks.y,
            location.position_in_blocks.z,
        )
    });
    locations
}

fn reset_viewed_edit_mesh(core: &mut VoxelTerrainCore, position: Vector3i, lod_index: usize) {
    core.legacy_view_mesh_block(position, lod_index);
    let entry = core.mesh_maps[lod_index].get_mut(&position).unwrap();
    entry.requested_revision = None;
    entry.is_in_update_list = false;
    entry.terminal_retry_count = 7;
    core.blocks_pending_update[lod_index].clear();
}

fn terrain_data_revision(
    core: &VoxelTerrainCore,
    position: Vector3i,
    lod_index: usize,
) -> VoxelDataKeyRevision {
    core.data.with_lod_map(lod_index, |map| {
        let revision = map.key_revision(position);
        if map.has_block(position) {
            VoxelDataKeyRevision::Present(revision)
        } else {
            VoxelDataKeyRevision::Tombstone(revision)
        }
    })
}

fn voxel_allocation_identity(voxels: &VoxelBuffer) -> usize {
    (0..MAX_CHANNELS)
        .find_map(|channel| {
            let bytes = voxels.channel_bytes(channel);
            (!bytes.is_empty()).then_some(bytes.as_ptr() as usize)
        })
        .unwrap_or(voxels as *const VoxelBuffer as usize)
}

#[test]
fn transactional_edit_mesh_revision_overflow_rolls_back_every_lod() {
    let mut core = make_edit_core_with_lods(2);
    for lod_index in 0..2 {
        reset_viewed_edit_mesh(&mut core, Vector3i::zero(), lod_index);
    }
    core.next_mesh_revision = u64::MAX;

    assert_eq!(
        core.try_edit_voxel(9, Vector3i::zero(), ChannelId::Type.index(),)
            .unwrap_err(),
        VoxelTerrainRuntimeError::MeshRevisionOverflow
    );
    for lod_index in 0..2 {
        assert!(core
            .data
            .block_snapshot(Vector3i::zero(), lod_index)
            .is_none());
        assert_eq!(
            terrain_data_revision(&core, Vector3i::zero(), lod_index),
            VoxelDataKeyRevision::Tombstone(0)
        );
        let entry = &core.mesh_maps[lod_index][&Vector3i::zero()];
        assert_eq!(entry.requested_revision, None);
        assert!(!entry.is_in_update_list);
        assert_eq!(entry.terminal_retry_count, 7);
        assert!(core.blocks_pending_update[lod_index].is_empty());
    }
    assert_eq!(core.next_mesh_revision, u64::MAX);
}

#[test]
fn transactional_edit_every_terrain_capacity_failure_rolls_back_every_lod() {
    for destination in [
        FixedCapacityDestination::MeshMap,
        FixedCapacityDestination::PendingMeshQueue,
        FixedCapacityDestination::Retirement,
    ] {
        let mut core = make_edit_core_with_lods(2);
        for lod_index in 0..2 {
            reset_viewed_edit_mesh(&mut core, Vector3i::zero(), lod_index);
        }
        let next_revision = core.next_mesh_revision;
        core.fail_fixed_capacity_for_test(destination, 1);

        assert_eq!(
            core.try_edit_voxel(10, Vector3i::zero(), ChannelId::Type.index(),)
                .unwrap_err(),
            VoxelTerrainRuntimeError::CapacityReservationFailed,
            "destination {destination:?} must fail before publication"
        );
        assert_eq!(
            core.last_fixed_capacity_failure_for_test(),
            Some(destination)
        );
        for lod_index in 0..2 {
            assert!(core
                .data
                .block_snapshot(Vector3i::zero(), lod_index)
                .is_none());
            let entry = &core.mesh_maps[lod_index][&Vector3i::zero()];
            assert_eq!(entry.requested_revision, None);
            assert!(!entry.is_in_update_list);
            assert_eq!(entry.terminal_retry_count, 7);
            assert!(core.blocks_pending_update[lod_index].is_empty());
        }
        assert_eq!(core.next_mesh_revision, next_revision);
    }
}

#[test]
fn transactional_edit_returns_canonical_affected_mesh_locations() {
    let mut core = make_edit_core_with_lods(2);
    for x in 0..=1 {
        for y in 0..=1 {
            for z in 0..=1 {
                reset_viewed_edit_mesh(&mut core, Vector3i::new(x, y, z), 0);
            }
        }
    }
    reset_viewed_edit_mesh(&mut core, Vector3i::zero(), 1);
    let block_size = core.data_block_size();
    let outcome = core
        .try_edit_voxel(11, Vector3i::splat(block_size), ChannelId::Type.index())
        .unwrap()
        .unwrap();

    let expected = vec![
        MeshBlockLocation::new(Vector3i::new(0, 0, 0), 0),
        MeshBlockLocation::new(Vector3i::new(0, 0, 1), 0),
        MeshBlockLocation::new(Vector3i::new(0, 1, 0), 0),
        MeshBlockLocation::new(Vector3i::new(0, 1, 1), 0),
        MeshBlockLocation::new(Vector3i::new(1, 0, 0), 0),
        MeshBlockLocation::new(Vector3i::new(1, 0, 1), 0),
        MeshBlockLocation::new(Vector3i::new(1, 1, 0), 0),
        MeshBlockLocation::new(Vector3i::new(1, 1, 1), 0),
        MeshBlockLocation::new(Vector3i::zero(), 1),
    ];
    assert_eq!(outcome.affected_mesh_blocks, expected);
    assert_eq!(
        outcome.edited_block,
        BlockLocation {
            position: Vector3i::splat(1),
            lod_index: 0,
        }
    );
    assert_eq!(outcome.block_revision, 1);
    assert_eq!(requested_mesh_locations(&core), expected);
    for location in &outcome.affected_mesh_blocks {
        let entry = &core.mesh_maps[usize::from(location.lod_index)][&location.position_in_blocks];
        assert!(entry.is_in_update_list);
        assert_eq!(entry.terminal_retry_count, 0);
        assert!(core.blocks_pending_update[usize::from(location.lod_index)]
            .contains(&location.position_in_blocks));
    }
}

#[test]
fn transactional_edit_materialization_transfers_and_retires_loading_owner() {
    let mut core = make_edit_core_with_lods(2);
    let position = Vector3i::zero();
    core.apply_data_view(single_block_box(position), 0);
    core.apply_data_view(single_block_box(position), 0);
    let loading = &core.loading_blocks[0][&position];
    let generation = loading.request_generation;
    let request = loading
        .physical_request
        .as_ref()
        .expect("queued load owns an exact physical request")
        .clone();
    let mut late_output = loaded_output(&core, position, generation, 0xD1);

    core.try_edit_voxel(0xE2, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .unwrap();

    assert!(request.cancellation.is_cancelled());
    assert!(!core.loading_blocks[0].contains_key(&position));
    assert!(!core.blocks_pending_load[0].contains(&position));
    assert_eq!(
        core.loaded_data_residency[0][&position],
        DataResidencyRefs::with_resident_viewers(2)
    );
    assert_eq!(
        core.data.block_snapshot(position, 0).unwrap().viewers.get(),
        2
    );
    assert_eq!(
        core.try_legacy_variable_apply_load_response(&mut late_output),
        Ok(false)
    );
    assert!(late_output.block_data.voxels.is_some());
    assert_eq!(
        core.data
            .block_snapshot(position, 0)
            .unwrap()
            .voxels()
            .get_voxel(0, 0, 0, ChannelId::Type.index()),
        0xE2
    );
}

#[test]
fn failed_edit_materialization_preserves_exact_loading_owner() {
    let mut core = make_edit_core_with_lods(2);
    let position = Vector3i::zero();
    core.apply_data_view(single_block_box(position), 0);
    let loading_before = core.loading_blocks[0][&position].clone();
    let request = loading_before.physical_request.as_ref().unwrap().clone();
    let pending_before = core.blocks_pending_load[0].clone();
    core.edit_after_prepare_voxel_for_test =
        Some((0xA3, Vector3i::zero(), ChannelId::Type.index()));

    assert!(matches!(
        core.try_edit_voxel(0xB4, Vector3i::zero(), ChannelId::Type.index()),
        Err(VoxelTerrainRuntimeError::ConcurrentDataMutation { .. })
    ));

    let loading_after = &core.loading_blocks[0][&position];
    assert_eq!(loading_after.residency, loading_before.residency);
    assert_eq!(
        loading_after.request_generation,
        loading_before.request_generation
    );
    assert_eq!(loading_after.request_state, loading_before.request_state);
    assert_eq!(
        loading_after.physical_request.as_ref().unwrap().tag,
        request.tag
    );
    assert!(!request.cancellation.is_cancelled());
    assert_eq!(core.blocks_pending_load[0], pending_before);
    assert!(!core.loaded_data_residency[0].contains_key(&position));
}

#[test]
fn transactional_edit_same_key_conflict_loses_no_voxel_or_queue_state() {
    let mut core = make_edit_core_with_lods(2);
    for lod_index in 0..2 {
        assert!(core
            .data
            .try_set_block(
                Vector3i::zero(),
                VoxelDataBlock::with_voxels(
                    VoxelBuffer::with_size(Vector3i::splat(core.data_block_size())),
                    lod_index as u8,
                ),
            )
            .unwrap());
        reset_viewed_edit_mesh(&mut core, Vector3i::zero(), lod_index);
    }
    let next_revision = core.next_mesh_revision;
    core.edit_after_prepare_voxel_for_test = Some((55, Vector3i::zero(), ChannelId::Type.index()));

    assert_eq!(
        core.try_edit_voxel(99, Vector3i::zero(), ChannelId::Type.index(),)
            .unwrap_err(),
        VoxelTerrainRuntimeError::ConcurrentDataMutation {
            location: BlockLocation {
                position: Vector3i::zero(),
                lod_index: 0,
            },
            expected_revision: VoxelDataKeyRevision::Present(1),
            actual_revision: VoxelDataKeyRevision::Present(2),
        }
    );
    assert_eq!(
        core.data
            .block_snapshot(Vector3i::zero(), 0)
            .unwrap()
            .voxels()
            .get_voxel(0, 0, 0, ChannelId::Type.index()),
        55
    );
    assert_eq!(
        terrain_data_revision(&core, Vector3i::zero(), 1),
        VoxelDataKeyRevision::Present(1)
    );
    for lod_index in 0..2 {
        let entry = &core.mesh_maps[lod_index][&Vector3i::zero()];
        assert_eq!(entry.requested_revision, None);
        assert!(!entry.is_in_update_list);
        assert_eq!(entry.terminal_retry_count, 7);
        assert!(core.blocks_pending_update[lod_index].is_empty());
    }
    assert_eq!(core.next_mesh_revision, next_revision);
}

#[test]
fn transactional_edit_unload_race_is_all_or_nothing() {
    let mut core = make_edit_core_with_lods(1);
    let position = Vector3i::zero();
    let location = BlockLocation {
        position,
        lod_index: 0,
    };
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
    voxels.set_voxel(41, 0, 0, 0, ChannelId::Type.index());
    let original_identity = voxel_allocation_identity(&voxels);
    let mut block = VoxelDataBlock::with_voxels(voxels, 0);
    block.viewers.set_exact(1);
    assert!(core.data.try_set_block(position, block).unwrap());
    reset_viewed_edit_mesh(&mut core, position, 0);
    let next_mesh_revision = core.next_mesh_revision;

    let removed_owner = Arc::new(Mutex::new(None));
    let removed_owner_sink = Arc::clone(&removed_owner);
    let weak_data = Arc::downgrade(&core.data);
    core.data.set_test_edit_phase_hook(Arc::new(move |phase| {
        if phase != SharedVoxelDataEditPhase::PreparedVoxelEditDraftedBeforeTransactionPrepare {
            return;
        }
        let data = weak_data
            .upgrade()
            .expect("the terrain retains shared data");
        let mut unload = data
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::Remove {
                location,
            }])
            .unwrap();
        let mut removed = unload.commit().unwrap().into_removed_blocks();
        let (_, block) = removed
            .pop()
            .expect("the racing unload retains the exact resident owner")
            .into_parts();
        *removed_owner_sink
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(block);
    }));

    assert_eq!(
        core.try_edit_voxel(99, position, ChannelId::Type.index()),
        Err(VoxelTerrainRuntimeError::ConcurrentDataMutation {
            location,
            expected_revision: VoxelDataKeyRevision::Present(1),
            actual_revision: VoxelDataKeyRevision::Tombstone(2),
        })
    );
    assert_eq!(
        terrain_data_revision(&core, position, 0),
        VoxelDataKeyRevision::Tombstone(2)
    );
    assert!(core.data.block_snapshot(position, 0).is_none());
    let removed_owner = removed_owner
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let removed_owner = removed_owner
        .as_ref()
        .expect("the unload owner remains retained outside storage");
    assert_eq!(
        voxel_allocation_identity(removed_owner.voxels()),
        original_identity
    );
    assert_eq!(
        removed_owner
            .voxels()
            .get_voxel(0, 0, 0, ChannelId::Type.index()),
        41
    );
    assert_eq!(removed_owner.viewers.get(), 1);
    assert_eq!(core.next_mesh_revision, next_mesh_revision);
    let entry = &core.mesh_maps[0][&position];
    assert_eq!(entry.requested_revision, None);
    assert!(!entry.is_in_update_list);
    assert_eq!(entry.terminal_retry_count, 7);
    assert!(core.blocks_pending_update[0].is_empty());
}

#[test]
fn transactional_edit_load_adopt_race_preserves_exact_owner() {
    let mut core = make_edit_core_with_lods(1);
    let position = Vector3i::zero();
    let location = BlockLocation {
        position,
        lod_index: 0,
    };
    reset_viewed_edit_mesh(&mut core, position, 0);
    let next_mesh_revision = core.next_mesh_revision;

    let adopted_identity = Arc::new(AtomicUsize::new(0));
    let adopted_identity_sink = Arc::clone(&adopted_identity);
    let weak_data = Arc::downgrade(&core.data);
    core.data.set_test_edit_phase_hook(Arc::new(move |phase| {
        if phase != SharedVoxelDataEditPhase::PreparedVoxelEditDraftedBeforeTransactionPrepare {
            return;
        }
        let data = weak_data
            .upgrade()
            .expect("the terrain retains shared data");
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(data.block_size() as i32));
        voxels.set_voxel(73, 0, 0, 0, ChannelId::Type.index());
        adopted_identity_sink.store(voxel_allocation_identity(&voxels), Ordering::SeqCst);
        let mut adoption = data
            .prepare_transaction(vec![SharedVoxelDataTransactionOperation::Insert {
                location,
                block: VoxelDataBlock::with_voxels(voxels, 0),
                final_viewers: 1,
            }])
            .unwrap();
        assert!(adoption.commit().unwrap().removed_blocks().is_empty());
    }));

    assert_eq!(
        core.try_edit_voxel(99, position, ChannelId::Type.index()),
        Err(VoxelTerrainRuntimeError::ConcurrentDataMutation {
            location,
            expected_revision: VoxelDataKeyRevision::Tombstone(0),
            actual_revision: VoxelDataKeyRevision::Present(1),
        })
    );
    let live_observation = core.data.with_lod_map(0, |map| {
        let live = map
            .get_block(position)
            .expect("the racing load adoption remains resident");
        (
            live.viewers.get(),
            live.voxels().get_voxel(0, 0, 0, ChannelId::Type.index()),
            voxel_allocation_identity(live.voxels()),
        )
    });
    assert_eq!(
        live_observation,
        (1, 73, adopted_identity.load(Ordering::SeqCst))
    );
    assert_eq!(
        terrain_data_revision(&core, position, 0),
        VoxelDataKeyRevision::Present(1)
    );
    assert_eq!(core.next_mesh_revision, next_mesh_revision);
    let entry = &core.mesh_maps[0][&position];
    assert_eq!(entry.requested_revision, None);
    assert!(!entry.is_in_update_list);
    assert_eq!(entry.terminal_retry_count, 7);
    assert!(core.blocks_pending_update[0].is_empty());
}

#[test]
fn transactional_edit_superseded_load_generation_consumes_no_prefix() {
    let mut core = make_edit_core_with_lods(1);
    let position = Vector3i::zero();
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
    voxels.set_voxel(17, 0, 0, 0, ChannelId::Type.index());
    assert!(core
        .data
        .try_set_block(position, VoxelDataBlock::with_voxels(voxels, 0))
        .unwrap());
    reset_viewed_edit_mesh(&mut core, position, 0);
    let next_mesh_revision = core.next_mesh_revision;

    core.loading_blocks[0].insert(
        position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(0),
            retry_count: 0,
            request_generation: 7,
            request_state: LoadRequestState::InFlight,
            physical_request: None,
        },
    );
    let mut task = LoadBlockForTerrainTask::new(
        position,
        0,
        7,
        Arc::clone(&core.data),
        Arc::clone(&core.stream),
    );
    task.output = Some(loaded_output(&core, position, 7, 0xa7));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));
    core.try_normalize_raw_completions().unwrap();
    core.loading_blocks[0]
        .get_mut(&position)
        .unwrap()
        .request_generation = 8;

    let durable_identity = match &core.durable_completion_inbox[0] {
        DurableCompletion::LoadFinished { completed, output } => (
            completed.task() as *const dyn ThreadedTask as *const () as usize,
            output
                .block_data
                .voxels
                .as_ref()
                .map_or(0, voxel_allocation_identity),
            output.request_generation,
        ),
        _ => panic!("the staged durable prefix must own a finished load"),
    };
    let loading_before = {
        let loading = &core.loading_blocks[0][&position];
        (
            loading.residency.resident_viewers,
            loading.retry_count,
            loading.request_generation,
            loading.request_state,
        )
    };
    core.edit_after_prepare_voxel_for_test =
        Some((55, Vector3i::new(1, 1, 1), ChannelId::Type.index()));

    assert!(matches!(
        core.try_edit_voxel(91, Vector3i::new(1, 1, 1), ChannelId::Type.index()),
        Err(VoxelTerrainRuntimeError::ConcurrentDataMutation {
            location: BlockLocation {
                position: actual_position,
                lod_index: 0,
            },
            expected_revision: VoxelDataKeyRevision::Present(1),
            actual_revision: VoxelDataKeyRevision::Present(2),
        }) if actual_position == position
    ));
    assert!(core.raw_completion_inbox.is_empty());
    assert_eq!(core.durable_completion_inbox.len(), 1);
    let durable_after = match &core.durable_completion_inbox[0] {
        DurableCompletion::LoadFinished { completed, output } => (
            completed.task() as *const dyn ThreadedTask as *const () as usize,
            output
                .block_data
                .voxels
                .as_ref()
                .map_or(0, voxel_allocation_identity),
            output.request_generation,
        ),
        _ => panic!("the failed edit must retain the exact durable load owner"),
    };
    assert_eq!(durable_after, durable_identity);
    let loading_after = &core.loading_blocks[0][&position];
    assert_eq!(
        (
            loading_after.residency.resident_viewers,
            loading_after.retry_count,
            loading_after.request_generation,
            loading_after.request_state,
        ),
        loading_before
    );
    let live = core.data.block_snapshot(position, 0).unwrap();
    assert_eq!(
        live.voxels().get_voxel(1, 1, 1, ChannelId::Type.index()),
        55
    );
    assert_eq!(core.next_mesh_revision, next_mesh_revision);
    let entry = &core.mesh_maps[0][&position];
    assert_eq!(entry.requested_revision, None);
    assert!(!entry.is_in_update_list);
    assert!(core.blocks_pending_update[0].is_empty());
}

#[test]
fn transactional_edit_disjoint_key_mutation_does_not_false_conflict() {
    let mut core = make_edit_core_with_lods(2);
    for lod_index in 0..2 {
        assert!(core
            .data
            .try_set_block(
                Vector3i::zero(),
                VoxelDataBlock::with_voxels(
                    VoxelBuffer::with_size(Vector3i::splat(core.data_block_size())),
                    lod_index as u8,
                ),
            )
            .unwrap());
        reset_viewed_edit_mesh(&mut core, Vector3i::zero(), lod_index);
    }
    let disjoint_voxel = Vector3i::new(64, 0, 0);
    core.edit_after_prepare_voxel_for_test = Some((55, disjoint_voxel, ChannelId::Type.index()));

    let outcome = core
        .try_edit_voxel(99, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .unwrap();
    assert_eq!(outcome.block_revision, 2);
    for lod_index in 0..2 {
        let resident = core
            .data
            .block_snapshot(Vector3i::zero(), lod_index)
            .unwrap();
        assert_eq!(
            resident
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            99
        );
        assert_eq!(
            terrain_data_revision(&core, Vector3i::zero(), lod_index),
            VoxelDataKeyRevision::Present(2)
        );
    }
    let disjoint_block = floor_div_vec(disjoint_voxel, core.data_block_size());
    assert_eq!(
        core.data
            .block_snapshot(disjoint_block, 0)
            .unwrap()
            .voxels()
            .get_voxel(0, 0, 0, ChannelId::Type.index()),
        55
    );
    assert_eq!(
        terrain_data_revision(&core, disjoint_block, 0),
        VoxelDataKeyRevision::Present(1)
    );
}

#[test]
fn transactional_edit_supersedes_in_flight_mesh_revisions_once() {
    let mut core = make_edit_core_with_lods(2);
    let mut old_keys = Vec::new();
    for lod_index in 0..2 {
        core.legacy_view_mesh_block(Vector3i::zero(), lod_index);
        let old_key = core
            .request_mesh_update(Vector3i::zero(), lod_index)
            .expect("viewed mesh block receives an in-flight revision");
        core.blocks_pending_update[lod_index].clear();
        core.mesh_maps[lod_index]
            .get_mut(&Vector3i::zero())
            .unwrap()
            .is_in_update_list = false;
        old_keys.push(old_key);
    }
    let next_revision = core.next_mesh_revision;

    let outcome = core
        .try_edit_voxel(77, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .unwrap();
    assert_eq!(outcome.affected_mesh_blocks.len(), 2);
    assert_eq!(core.next_mesh_revision, next_revision + 2);
    for (lod_index, old_key) in old_keys.into_iter().enumerate() {
        let entry = &core.mesh_maps[lod_index][&Vector3i::zero()];
        let fresh_revision = entry.requested_revision.unwrap();
        assert_ne!(fresh_revision, old_key.revision);
        assert_eq!(fresh_revision, next_revision + lod_index as u64);
        assert!(entry.is_in_update_list);
        assert_eq!(
            core.blocks_pending_update[lod_index],
            vec![Vector3i::zero()]
        );

        core.apply_mesh_update_for_test(nonempty_mesh_output(old_key));
        assert_eq!(
            core.mesh_maps[lod_index][&Vector3i::zero()].applied_revision,
            None
        );
    }
}

#[test]
fn transactional_edit_every_storage_capacity_failure_is_precommit() {
    use crate::storage::voxel_data::SharedVoxelDataTransactionReservation;

    for reservation in [
        SharedVoxelDataTransactionReservation::OperationStorage,
        SharedVoxelDataTransactionReservation::SpatialPreparation,
        SharedVoxelDataTransactionReservation::LiveMap,
        SharedVoxelDataTransactionReservation::LiveKeyRevisions,
        SharedVoxelDataTransactionReservation::RemovedOutcome,
        SharedVoxelDataTransactionReservation::PreviewSnapshotStorage,
        SharedVoxelDataTransactionReservation::ObservationStorage,
    ] {
        let mut core = make_edit_core_with_lods(2);
        for lod_index in 0..2 {
            reset_viewed_edit_mesh(&mut core, Vector3i::zero(), lod_index);
        }
        let next_revision = core.next_mesh_revision;
        core.data
            .set_test_transaction_reservation_failpoint(Some(reservation));

        assert_eq!(
            core.try_edit_voxel(12, Vector3i::zero(), ChannelId::Type.index(),)
                .unwrap_err(),
            VoxelTerrainRuntimeError::CapacityReservationFailed,
            "reservation {reservation:?} must fail before publication"
        );
        assert_eq!(core.data.block_count(), 0);
        assert_eq!(core.next_mesh_revision, next_revision);
        for lod_index in 0..2 {
            let entry = &core.mesh_maps[lod_index][&Vector3i::zero()];
            assert_eq!(entry.requested_revision, None);
            assert!(!entry.is_in_update_list);
            assert_eq!(entry.terminal_retry_count, 7);
            assert!(core.blocks_pending_update[lod_index].is_empty());
        }
    }

    let mut core = make_edit_core_with_lods(2);
    for lod_index in 0..2 {
        reset_viewed_edit_mesh(&mut core, Vector3i::zero(), lod_index);
    }
    let next_revision = core.next_mesh_revision;
    core.data
        .set_test_transaction_live_spatial_registry_fail_lod(Some(1));
    assert_eq!(
        core.try_edit_voxel(13, Vector3i::zero(), ChannelId::Type.index(),)
            .unwrap_err(),
        VoxelTerrainRuntimeError::CapacityReservationFailed
    );
    core.data
        .set_test_transaction_live_spatial_registry_fail_lod(None);
    assert_eq!(core.data.block_count(), 0);
    assert_eq!(core.next_mesh_revision, next_revision);
    for lod_index in 0..2 {
        let entry = &core.mesh_maps[lod_index][&Vector3i::zero()];
        assert_eq!(entry.requested_revision, None);
        assert!(!entry.is_in_update_list);
        assert!(core.blocks_pending_update[lod_index].is_empty());
    }
}

#[test]
fn transactional_edit_every_lod_revision_overflow_is_precommit() {
    let mut core = make_edit_core_with_lods(3);
    for lod_index in 0..3 {
        let mut block = VoxelDataBlock::with_voxels(
            VoxelBuffer::with_size(Vector3i::splat(core.data_block_size())),
            lod_index as u8,
        );
        block
            .voxels_mut()
            .set_voxel(20 + lod_index as u64, 0, 0, 0, ChannelId::Type.index());
        assert!(core.data.try_set_block(Vector3i::zero(), block).unwrap());
    }
    core.data.with_lod_map_mut(2, |map| {
        map.set_key_revision_for_test(Vector3i::zero(), u64::MAX);
    });

    assert_eq!(
        core.try_edit_voxel(88, Vector3i::zero(), ChannelId::Type.index(),)
            .unwrap_err(),
        VoxelTerrainRuntimeError::BlockRevisionOverflow {
            location: BlockLocation {
                position: Vector3i::zero(),
                lod_index: 2,
            },
        }
    );
    for lod_index in 0..3 {
        let block = core
            .data
            .block_snapshot(Vector3i::zero(), lod_index)
            .unwrap();
        assert_eq!(
            block.voxels().get_voxel(0, 0, 0, ChannelId::Type.index()),
            20 + lod_index as u64
        );
        assert!(!block.is_modified());
        assert_eq!(
            terrain_data_revision(&core, Vector3i::zero(), lod_index),
            VoxelDataKeyRevision::Present(if lod_index == 2 { u64::MAX } else { 1 })
        );
    }
}

#[test]
fn transactional_edit_coordinate_overflow_rolls_back_every_effect() {
    let mut data = VoxelData::new();
    data.set_lod_count(2).unwrap();
    data.set_bounds(Box3i::new(
        Vector3i::splat(i32::MAX - 3),
        Vector3i::splat(10),
    ));
    data.set_streaming_enabled(false);
    data.set_full_load_completed(true);
    let mesher: Arc<dyn VoxelMesher> = Arc::new(OneVoxelPaddingMesher);
    let mut core = VoxelTerrainCore::legacy_variable_lod_for_parity(
        data,
        Arc::new(MemoryStream::new()),
        MeshingDependency::new(mesher, None),
        2,
    );

    assert_eq!(
        core.try_edit_voxel(1, Vector3i::splat(i32::MAX - 2), ChannelId::Type.index(),)
            .unwrap_err(),
        VoxelTerrainRuntimeError::CoordinateOverflow
    );
    assert_eq!(core.data.block_count(), 0);
    assert!(core.mesh_maps.iter().all(HashMap::is_empty));
    assert!(core.blocks_pending_update.iter().all(Vec::is_empty));
}

#[test]
fn transactional_edit_storage_is_fenced_until_mesh_publication() {
    for (phase, expected_marker) in [
        (FixedCommitPausePhase::StorageFencedBeforeCorePublish, false),
        (
            FixedCommitPausePhase::AfterTerrainPublishBeforeFenceFinish,
            true,
        ),
    ] {
        let mut terrain = make_edit_core_with_lods(2);
        for lod_index in 0..2 {
            reset_viewed_edit_mesh(&mut terrain, Vector3i::zero(), lod_index);
        }
        let data = Arc::clone(&terrain.data);
        let pause = terrain.install_fixed_commit_pause_for_test(phase);
        let marker = pause.commit_marker();
        let core = Arc::new(Mutex::new(terrain));
        let editing_core = Arc::clone(&core);
        let edit = thread::spawn(move || {
            editing_core
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .try_edit_voxel(91, Vector3i::zero(), ChannelId::Type.index())
        });

        pause.wait_until_reached();
        for lod_index in 0..2 {
            assert!(
                data.try_lod_map_read(lod_index).is_none(),
                "LOD {lod_index} became observable during {phase:?}"
            );
        }
        assert_eq!(marker.load(Ordering::SeqCst), expected_marker);

        pause.release();
        let outcome = edit
            .join()
            .expect("the edit publication thread does not panic")
            .expect("the edit publication succeeds")
            .expect("the edit materializes every configured LOD");
        assert_eq!(outcome.affected_mesh_blocks.len(), 2);

        let terrain = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        for lod_index in 0..2 {
            let block = terrain
                .data
                .block_snapshot(Vector3i::zero(), lod_index)
                .expect("the committed edit is resident at every configured LOD");
            assert_eq!(
                block.voxels().get_voxel(0, 0, 0, ChannelId::Type.index()),
                91
            );
            assert!(terrain.mesh_maps[lod_index][&Vector3i::zero()]
                .requested_revision
                .is_some());
            assert_eq!(
                terrain.blocks_pending_update[lod_index],
                vec![Vector3i::zero()]
            );
        }
    }
}

#[test]
fn interior_edit_requests_only_the_containing_mesh() {
    let mut core = make_resident_edit_core();

    assert!(core
        .try_edit_voxel(1, Vector3i::new(5, 5, 5), 0)
        .unwrap()
        .is_some());
    assert_eq!(
        requested_mesh_locations(&core),
        vec![MeshBlockLocation::new(Vector3i::zero(), 0)]
    );
}

#[test]
fn face_boundary_edit_requests_both_adjacent_meshes() {
    let mut core = make_resident_edit_core();

    assert!(core
        .try_edit_voxel(1, Vector3i::new(16, 5, 5), 0)
        .unwrap()
        .is_some());
    let locations: std::collections::HashSet<_> =
        requested_mesh_locations(&core).into_iter().collect();
    assert_eq!(
        locations,
        std::collections::HashSet::from([
            MeshBlockLocation::new(Vector3i::new(0, 0, 0), 0),
            MeshBlockLocation::new(Vector3i::new(1, 0, 0), 0),
        ])
    );
}

#[test]
fn corner_boundary_edit_requests_eight_meshes() {
    let mut core = make_resident_edit_core();

    assert!(core
        .try_edit_voxel(1, Vector3i::new(16, 16, 16), 0)
        .unwrap()
        .is_some());
    let locations: std::collections::HashSet<_> =
        requested_mesh_locations(&core).into_iter().collect();
    assert_eq!(
        locations,
        std::collections::HashSet::from([
            MeshBlockLocation::new(Vector3i::new(0, 0, 0), 0),
            MeshBlockLocation::new(Vector3i::new(1, 0, 0), 0),
            MeshBlockLocation::new(Vector3i::new(0, 1, 0), 0),
            MeshBlockLocation::new(Vector3i::new(1, 1, 0), 0),
            MeshBlockLocation::new(Vector3i::new(0, 0, 1), 0),
            MeshBlockLocation::new(Vector3i::new(1, 0, 1), 0),
            MeshBlockLocation::new(Vector3i::new(0, 1, 1), 0),
            MeshBlockLocation::new(Vector3i::new(1, 1, 1), 0),
        ])
    );
}

#[test]
fn block_box_conversion_scales_with_lod() {
    let block = Box3i::new(Vector3i::new(-1, 2, 3), Vector3i::splat(1));
    assert_eq!(
        block_box_to_voxel_box(block, 16, 2),
        Box3i::new(Vector3i::new(-64, 128, 192), Vector3i::splat(64))
    );
}

#[test]
fn edit_while_mesh_task_is_in_flight_applies_only_latest_revision() {
    let mut core = make_resident_edit_core();
    let position = Vector3i::zero();
    let older_key = core.request_mesh_update(position, 0).unwrap();

    assert!(core
        .try_edit_voxel(1, Vector3i::new(5, 5, 5), 0)
        .unwrap()
        .is_some());
    let newer_revision = core.mesh_maps[0][&position]
        .requested_revision
        .expect("edit requested a newer mesh revision");
    assert!(newer_revision > older_key.revision);

    core.apply_mesh_update_for_test(nonempty_mesh_output(older_key));
    assert_eq!(core.mesh_maps[0][&position].applied_revision, None);
}

#[test]
fn desired_mesh_revision_waits_for_neighbor_readiness_then_launches() {
    let mut core = build_core();
    let position = Vector3i::zero();
    core.legacy_view_mesh_block(position, 0);
    let requested = core.request_mesh_update(position, 0).unwrap();

    core.process_meshing().unwrap();
    assert_eq!(
        core.mesh_maps[0][&position].requested_revision,
        Some(requested.revision)
    );
    assert!(core.mesh_maps[0][&position].is_in_update_list);
    assert_eq!(core.blocks_pending_update[0], vec![position]);
    assert_eq!(core.pending_task_count(), 0);
    assert!(!core.data.is_mutation_admission_closed());

    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                assert!(core
                    .data
                    .try_set_block(
                        Vector3i::new(x, y, z),
                        VoxelDataBlock::with_voxels(
                            VoxelBuffer::with_size(Vector3i::splat(core.data_block_size())),
                            0,
                        ),
                    )
                    .unwrap());
            }
        }
    }
    core.try_schedule_mesh_update_from_data(
        block_box_to_voxel_box(
            Box3i::new(position, Vector3i::splat(1)),
            core.data_block_size(),
            0,
        ),
        0,
    )
    .unwrap();

    assert_eq!(
        core.mesh_maps[0][&position].requested_revision,
        Some(requested.revision)
    );
    core.process_meshing().unwrap();
    assert!(!core.mesh_maps[0][&position].is_in_update_list);
    assert!(core.blocks_pending_update[0].is_empty());
}

#[test]
fn boundary_mesh_readiness_lod_math_error_preserves_pending_mesh_owner() {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(i32::MAX), Vector3i::splat(1)));
    let mesher: Arc<dyn VoxelMesher> = Arc::new(AlwaysOneTriangleMesher);
    let mut core = VoxelTerrainCore::legacy_variable_lod_for_parity(
        data,
        Arc::new(MemoryStream::new()),
        MeshingDependency::new(mesher, None),
        2,
    );
    let position = Vector3i::zero();
    core.legacy_view_mesh_block(position, 0);
    core.request_mesh_update(position, 0)
        .expect("the resident mesh gets one exact pending owner");
    let requested_revision = core.mesh_maps[0][&position].requested_revision;

    assert!(matches!(
        core.try_process(&[]),
        Err(VoxelTerrainRuntimeError::LodMath(
            LodMathError::CoordinateOverflow
        ))
    ));
    assert_eq!(
        core.mesh_maps[0][&position].requested_revision,
        requested_revision
    );
    assert!(core.mesh_maps[0][&position].is_in_update_list);
    assert_eq!(core.blocks_pending_update[0], vec![position]);
    assert_eq!(core.pending_task_count(), 0);
}

#[test]
fn boundary_mesh_readiness_partial_final_block_fails_before_variable_mutation() {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(
        Vector3i::new(i32::MAX - 15, 0, 0),
        Vector3i::new(15, 16, 16),
    ));
    let mesher: Arc<dyn VoxelMesher> = Arc::new(AlwaysOneTriangleMesher);
    let mut core = VoxelTerrainCore::legacy_variable_lod_for_parity(
        data,
        Arc::new(MemoryStream::new()),
        MeshingDependency::new(mesher, None),
        2,
    );
    let position = Vector3i::new((i32::MAX - 15).div_euclid(16), 0, 0);
    core.legacy_view_mesh_block(position, 0);
    let requested = core
        .request_mesh_update(position, 0)
        .expect("the pre-existing mesh owner gets one pending revision");
    let preserved_event = MeshBlockLocation::new(Vector3i::new(7, 8, 9), 1);
    core.event_outbox
        .push_back(VoxelTerrainEvent::MeshBlockExited(preserved_event));
    let stats_before = *core.stats();

    assert!(matches!(
        core.try_process(&[]),
        Err(VoxelTerrainRuntimeError::LodMath(
            LodMathError::CoordinateOverflow
        ))
    ));

    let entry = &core.mesh_maps[0][&position];
    assert_eq!(entry.resident_viewers, 1);
    assert_eq!(entry.requested_revision, Some(requested.revision));
    assert!(entry.is_in_update_list);
    assert_eq!(core.blocks_pending_update[0], vec![position]);
    assert_eq!(core.pending_task_count(), 0);
    assert_eq!(*core.stats(), stats_before);
    assert_eq!(core.event_outbox.len(), 1);
    assert!(matches!(
        core.event_outbox.front(),
        Some(VoxelTerrainEvent::MeshBlockExited(location)) if *location == preserved_event
    ));

    assert!(matches!(
        core.try_drain_completed_tasks(),
        Err(VoxelTerrainRuntimeError::LodMath(
            LodMathError::CoordinateOverflow
        ))
    ));
    assert_eq!(core.event_outbox.len(), 1);
}

#[test]
fn lod1_edit_padding_scales_to_coarse_cells() {
    let mut core = make_edit_core_with_lods(2);
    core.legacy_view_mesh_block(Vector3i::new(0, 0, 0), 1);
    core.legacy_view_mesh_block(Vector3i::new(1, 0, 0), 1);

    assert!(core
        .try_edit_voxel(1, Vector3i::new(33, 5, 5), 0)
        .unwrap()
        .is_some());
    let locations: std::collections::HashSet<_> =
        requested_mesh_locations(&core).into_iter().collect();
    assert_eq!(
        locations,
        std::collections::HashSet::from([
            MeshBlockLocation::new(Vector3i::new(0, 0, 0), 1),
            MeshBlockLocation::new(Vector3i::new(1, 0, 0), 1),
        ])
    );
}

#[test]
fn lod1_negative_boundary_uses_euclidean_padding_conversion() {
    let mut core = make_edit_core_with_lods(2);
    core.legacy_view_mesh_block(Vector3i::new(-2, 0, 0), 1);
    core.legacy_view_mesh_block(Vector3i::new(-1, 0, 0), 1);

    assert!(core
        .try_edit_voxel(1, Vector3i::new(-31, 5, 5), 0)
        .unwrap()
        .is_some());
    let locations: std::collections::HashSet<_> =
        requested_mesh_locations(&core).into_iter().collect();
    assert_eq!(
        locations,
        std::collections::HashSet::from([
            MeshBlockLocation::new(Vector3i::new(-2, 0, 0), 1),
            MeshBlockLocation::new(Vector3i::new(-1, 0, 0), 1),
        ])
    );
}

#[test]
fn asymmetric_padding_reaches_previous_mesh_at_positive_boundary() {
    let mesher = TransvoxelMesher::new();
    let minimum_padding = mesher.minimum_padding() as i32;
    let maximum_padding = mesher.maximum_padding() as i32;
    assert!(maximum_padding > minimum_padding);
    let mut core = make_edit_core_with_mesher_and_lods(Arc::new(mesher), 1);
    let block_stride = core.data_block_size();
    core.legacy_view_mesh_block(Vector3i::zero(), 0);
    core.legacy_view_mesh_block(Vector3i::new(1, 0, 0), 0);

    let edit_position = Vector3i::new(
        block_stride + maximum_padding - 1,
        block_stride / 2,
        block_stride / 2,
    );
    assert!(core.try_edit_voxel(1, edit_position, 0).unwrap().is_some());

    assert_eq!(
        requested_mesh_locations(&core),
        vec![
            MeshBlockLocation::new(Vector3i::zero(), 0),
            MeshBlockLocation::new(Vector3i::new(1, 0, 0), 0),
        ]
    );
}

#[test]
fn asymmetric_padding_does_not_overinvalidate_negative_boundary_at_lod1() {
    const LOD: usize = 1;

    let mesher = TransvoxelMesher::new();
    let minimum_padding = mesher.minimum_padding() as i32;
    let maximum_padding = mesher.maximum_padding() as i32;
    assert!(maximum_padding > minimum_padding);
    let mut core = make_edit_core_with_mesher_and_lods(Arc::new(mesher), (LOD + 1) as u8);
    let lod_scale = 1i32 << LOD;
    let block_stride = core.data_block_size() * lod_scale;
    let scaled_maximum_padding = maximum_padding * lod_scale;
    core.legacy_view_mesh_block(Vector3i::new(-2, 0, 0), LOD);
    core.legacy_view_mesh_block(Vector3i::new(-1, 0, 0), LOD);

    let edit_position = Vector3i::new(
        -block_stride - scaled_maximum_padding,
        block_stride / 2,
        block_stride / 2,
    );
    assert!(core.try_edit_voxel(1, edit_position, 0).unwrap().is_some());

    assert_eq!(
        requested_mesh_locations(&core),
        vec![MeshBlockLocation::new(Vector3i::new(-2, 0, 0), LOD as u8)]
    );
}

#[test]
fn edit_propagates_lod0_voxels_into_coarse_mesh_source() {
    let mut core = make_edit_core_with_lods(2);
    let channel = ChannelId::Type.index();
    let edited_pos = Vector3i::new(20, 4, 6);

    assert!(core
        .try_edit_voxel(7, edited_pos, channel)
        .unwrap()
        .is_some());
    let lod1_block = core
        .data()
        .block_snapshot(Vector3i::zero(), 1)
        .expect("LOD1 source block must be updated after the edit");
    assert!(lod1_block.is_modified());
    assert_eq!(lod1_block.voxels().get_voxel(10, 2, 3, channel), 7);
}

fn nonempty_mesh_output(key: MeshBlockKey) -> BlockMeshOutput {
    let mut arrays = crate::meshers::transvoxel::structures::MeshArrays::default();
    arrays.indices.extend_from_slice(&[0, 0, 0]);
    BlockMeshOutput::new(
        key,
        MeshBuildFeatures {
            visuals: true,
            collisions: false,
            variable_lod: false,
        },
        MesherOutput {
            surfaces: vec![Surface::new(SurfaceArrays::Transvoxel(arrays), 0)],
            ..MesherOutput::default()
        },
        Arc::new(MeshArraysPool::new()),
        false,
    )
}

fn empty_mesh_output(key: MeshBlockKey) -> BlockMeshOutput {
    BlockMeshOutput::new(
        key,
        MeshBuildFeatures {
            visuals: true,
            collisions: false,
            variable_lod: false,
        },
        MesherOutput::default(),
        Arc::new(MeshArraysPool::new()),
        false,
    )
}

#[test]
fn mesh_block_location_distinguishes_lods() {
    let position = Vector3i::new(2, -3, 4);
    let a = MeshBlockLocation::new(position, 0);
    let b = MeshBlockLocation::new(position, 1);
    assert_ne!(a, b);
    let set = std::collections::HashSet::from([a, b]);
    assert_eq!(set.len(), 2);
}

#[test]
fn stale_mesh_output_cannot_replace_newer_revision() {
    let mut core = build_core();
    let position = Vector3i::zero();
    core.legacy_view_mesh_block(position, 0);
    let first = core.request_mesh_update(position, 0).unwrap();
    let second = core.request_mesh_update(position, 0).unwrap();
    assert!(second.revision > first.revision);

    core.apply_mesh_update_for_test(nonempty_mesh_output(first));
    assert_eq!(core.mesh_blocks_at_lod(0)[&position].applied_revision, None);
}

#[test]
fn mesh_events_distinguish_enter_update_empty_and_exit() {
    let mut core = build_core();
    let position = Vector3i::zero();
    core.legacy_view_mesh_block(position, 0);

    let enter = core.request_mesh_update(position, 0).unwrap();
    core.apply_mesh_update_for_test(nonempty_mesh_output(enter));
    assert!(matches!(
        core.event_outbox
            .iter()
            .rev()
            .find(|event| event.mesh_descriptor().is_some()),
        Some(VoxelTerrainEvent::MeshBlockEntered(upload)) if upload.key() == enter
    ));

    let update = core.request_mesh_update(position, 0).unwrap();
    core.apply_mesh_update_for_test(nonempty_mesh_output(update));
    assert!(matches!(
        core.event_outbox
            .iter()
            .rev()
            .find(|event| event.mesh_descriptor().is_some()),
        Some(VoxelTerrainEvent::MeshBlockUpdated(upload)) if upload.key() == update
    ));

    let empty = core.request_mesh_update(position, 0).unwrap();
    core.apply_mesh_update_for_test(empty_mesh_output(empty));
    assert!(matches!(
        core.event_outbox
            .iter()
            .rev()
            .find(|event| event.mesh_descriptor().is_some()),
        Some(VoxelTerrainEvent::MeshBlockBecameEmpty(upload)) if upload.key() == empty
    ));

    core.legacy_unview_mesh_block(position, 0);
    assert!(matches!(
        core.event_outbox.back(),
        Some(VoxelTerrainEvent::MeshBlockExited(location))
            if *location == MeshBlockLocation::new(position, 0)
    ));
}

fn process_until<F>(
    core: &mut VoxelTerrainCore,
    viewers: &[ViewerUpdate],
    mut done: F,
) -> Vec<VoxelTerrainEvent>
where
    F: FnMut(&VoxelTerrainCore, &[VoxelTerrainEvent]) -> bool,
{
    let mut last_events = Vec::new();
    for _ in 0..100 {
        let events = core.try_process(viewers).unwrap();
        if done(core, &events) {
            return events;
        }
        last_events = events;
        thread::sleep(Duration::from_millis(5));
    }
    panic!(
        "terrain process condition was not reached; last events {last_events:?}, stats {:?}",
        core.stats
    );
}

fn single_block_box(position: Vector3i) -> Box3i {
    Box3i::new(position, Vector3i::splat(1))
}

fn loaded_output(
    core: &VoxelTerrainCore,
    position: Vector3i,
    generation: u64,
    marker: u64,
) -> TerrainLoadOutput {
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
    voxels.set_voxel(marker, 0, 0, 0, ChannelId::Type.index());
    let request_tag = core.loading_blocks[0]
        .get(&position)
        .filter(|entry| entry.request_generation == generation)
        .and_then(|entry| entry.physical_request.as_ref())
        .map(|request| request.tag);
    TerrainLoadOutput::new_optional(
        BlockDataOutput::loaded(position, 0, voxels, false),
        generation,
        request_tag,
    )
}

fn tagged_current_load_task(
    core: &VoxelTerrainCore,
    position: Vector3i,
    lod_index: u8,
    generation: u64,
) -> LoadBlockForTerrainTask {
    let request = core.loading_blocks[usize::from(lod_index)][&position]
        .physical_request
        .as_ref()
        .filter(|request| request.tag.request_generation == generation)
        .expect("test fixture must use the current physical load request")
        .clone();
    LoadBlockForTerrainTask::new(
        position,
        lod_index,
        generation,
        core.data.clone(),
        core.stream.clone(),
    )
    .with_request_control(request.tag, request.cancellation)
}

fn complete_current_not_found(core: &mut VoxelTerrainCore, position: Vector3i) -> u64 {
    let generation = core.loading_blocks[0][&position].request_generation;
    core.loading_blocks[0]
        .get_mut(&position)
        .unwrap()
        .request_state = LoadRequestState::InFlight;
    core.blocks_pending_load[0].retain(|pending| *pending != position);
    core.legacy_variable_apply_load_response(TerrainLoadOutput::new(
        BlockDataOutput::not_found(position, 0),
        generation,
    ));
    generation
}

#[test]
fn load_lifecycle_fail_once_keeps_one_viewer_then_unloads_and_persists() {
    let stream = Arc::new(FailOnceLoadStream::new());
    let mut core = build_core_with_stream(stream.clone());
    let position = Vector3i::zero();
    let viewed_box = single_block_box(position);

    core.apply_data_view(viewed_box, 0);
    core.send_data_load_requests();
    core.wait_for_pending_tasks();
    core.drain_completed_tasks().unwrap();
    core.send_data_load_requests();
    core.wait_for_pending_tasks();
    core.drain_completed_tasks().unwrap();

    let block = core
        .data()
        .block_snapshot(position, 0)
        .expect("retry should install the requested block");
    assert_eq!(block.viewers.get(), 1);

    assert!(core
        .try_edit_voxel(77, Vector3i::new(1, 1, 1), ChannelId::Type.index())
        .unwrap()
        .is_some());
    core.apply_data_unview(viewed_box, 0);
    assert!(
        core.data().block_snapshot(position, 0).is_none(),
        "the only viewer must fully release the block"
    );

    core.shutdown_and_flush().unwrap();
    let mut persisted = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(position, 0, &mut persisted),
        LoadResult::Found
    );
    assert_eq!(persisted.get_voxel(1, 1, 1, ChannelId::Type.index()), 77);
}

#[test]
fn load_lifecycle_stale_loaded_output_cannot_consume_reviewed_demand() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let viewed_box = single_block_box(position);

    core.apply_data_view(viewed_box, 0);
    let request_a = core.loading_blocks[0][&position].request_generation;
    core.apply_data_unview(viewed_box, 0);
    core.apply_data_view(viewed_box, 0);
    let request_b = core.loading_blocks[0][&position].request_generation;
    assert!(request_b > request_a);

    core.legacy_variable_apply_load_response(loaded_output(&core, position, request_a, 11));
    assert!(
        core.data().block_snapshot(position, 0).is_none(),
        "request A must not install data for request B"
    );

    core.legacy_variable_apply_load_response(loaded_output(&core, position, request_b, 22));
    let block = core
        .data()
        .block_snapshot(position, 0)
        .expect("current request B should install");
    assert_eq!(block.viewers.get(), 1);
    assert_eq!(
        block.voxels().get_voxel(0, 0, 0, ChannelId::Type.index()),
        22
    );
}

#[test]
fn load_lifecycle_multiple_real_viewers_keep_exact_refcount() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let viewed_box = single_block_box(position);

    core.apply_data_view(viewed_box, 0);
    core.apply_data_view(viewed_box, 0);
    let generation = core.loading_blocks[0][&position].request_generation;
    core.legacy_variable_apply_load_response(loaded_output(&core, position, generation, 33));

    let block = core
        .data()
        .block_snapshot(position, 0)
        .expect("shared request should install");
    assert_eq!(block.viewers.get(), 2);

    core.apply_data_unview(viewed_box, 0);
    assert_eq!(
        core.data()
            .block_snapshot(position, 0)
            .expect("one viewer remains")
            .viewers
            .get(),
        1
    );
    core.apply_data_unview(viewed_box, 0);
    assert!(core.data().block_snapshot(position, 0).is_none());
}

#[test]
fn load_lifecycle_current_drop_retries_without_changing_viewer_count() {
    let mut core = build_core();
    let position = Vector3i::zero();

    core.apply_data_view(single_block_box(position), 0);
    let first_generation = core.loading_blocks[0][&position].request_generation;
    core.blocks_pending_load[0].clear();
    core.legacy_variable_apply_load_response(TerrainLoadOutput::new(
        BlockDataOutput::loaded_dropped(position, 0),
        first_generation,
    ));

    let entry = &core.loading_blocks[0][&position];
    assert_eq!(entry.residency.resident_viewers, 1);
    assert_eq!(entry.retry_count, 1);
    assert!(entry.request_generation > first_generation);
    assert_eq!(entry.request_state, LoadRequestState::Queued);
    assert_eq!(core.blocks_pending_load[0], vec![position]);
}

#[test]
fn load_adoption_storage_error_retains_exact_output_and_demand_for_retry() {
    let mut core = build_core();
    let position = Vector3i::zero();

    core.apply_data_view(single_block_box(position), 0);
    let first_generation = core.loading_blocks[0][&position].request_generation;
    core.loading_blocks[0]
        .get_mut(&position)
        .unwrap()
        .request_state = LoadRequestState::InFlight;
    core.blocks_pending_load[0].clear();
    core.data.with_lod_map_mut(0, |map| {
        map.set_key_revision_for_test(position, u64::MAX);
    });
    let mut output = loaded_output(&core, position, first_generation, 91);
    let payload_identity =
        voxel_allocation_identity(output.block_data.voxels.as_ref().expect("loaded payload"));

    assert!(matches!(
        core.try_legacy_variable_apply_load_response(&mut output),
        Err(VoxelTerrainRuntimeError::DataMutation(
            SharedVoxelDataMutationError::KeyRevisionOverflow {
                position: actual,
                lod_index: 0,
            }
        )) if actual == position
    ));

    assert!(core.data().block_snapshot(position, 0).is_none());
    let entry = &core.loading_blocks[0][&position];
    assert_eq!(entry.residency.resident_viewers, 1);
    assert_eq!(entry.retry_count, 0);
    assert_eq!(entry.request_generation, first_generation);
    assert_eq!(entry.request_state, LoadRequestState::InFlight);
    assert!(core.blocks_pending_load[0].is_empty());
    assert_eq!(
        voxel_allocation_identity(output.block_data.voxels.as_ref().unwrap()),
        payload_identity
    );

    core.data.with_lod_map_mut(0, |map| {
        map.set_key_revision_for_test(position, 0);
    });
    assert_eq!(
        core.try_legacy_variable_apply_load_response(&mut output),
        Ok(true)
    );
    assert!(output.block_data.voxels.is_none());
    core.data.with_lod_map(0, |map| {
        let resident = map.get_block(position).unwrap();
        assert_eq!(
            voxel_allocation_identity(resident.voxels()),
            payload_identity
        );
        assert_eq!(
            resident
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            91
        );
    });
}

#[test]
fn data_view_storage_error_is_queryable_and_retried_without_panicking() {
    let mut core = build_core();
    let position = Vector3i::zero();
    assert!(core
        .data
        .try_set_block(position, VoxelDataBlock::empty(0))
        .unwrap());
    core.data.with_lod_map_mut(0, |map| {
        map.get_block_mut(position)
            .unwrap()
            .viewers
            .set_exact(u32::MAX);
    });
    core.loaded_data_residency[0]
        .insert(position, DataResidencyRefs::with_resident_viewers(u32::MAX));
    let viewed_box = single_block_box(position);

    core.apply_data_view(viewed_box, 0);

    assert_eq!(
        core.last_data_mutation_error(),
        Some(&SharedVoxelDataMutationError::ViewerCountOverflow {
            position,
            lod_index: 0,
        })
    );
    assert_eq!(
        core.data_view_retries[0],
        vec![PendingDataMutation {
            box_in_blocks: viewed_box,
            retry_count: 1,
        }]
    );
    core.data.with_lod_map_mut(0, |map| {
        map.get_block_mut(position).unwrap().viewers.set_exact(0);
    });
    core.loaded_data_residency[0].insert(position, DataResidencyRefs::with_resident_viewers(0));

    core.try_process(&[]).unwrap();

    assert!(core.data_view_retries[0].is_empty());
    assert_eq!(
        core.data.block_snapshot(position, 0).unwrap().viewers.get(),
        1
    );
}

#[test]
fn fixed_data_view_storage_error_is_repeatable_and_non_mutating() {
    let mut core = build_core();
    let position = Vector3i::zero();
    assert!(core
        .data
        .try_set_block(position, VoxelDataBlock::empty(0))
        .unwrap());
    core.data.with_lod_map_mut(0, |map| {
        map.get_block_mut(position)
            .unwrap()
            .viewers
            .set_exact(u32::MAX);
    });
    core.loaded_data_residency[0]
        .insert(position, DataResidencyRefs::with_resident_viewers(u32::MAX));

    core.apply_data_view(single_block_box(position), 0);
    for _ in 0..MAX_LOAD_RETRIES {
        assert!(matches!(
            core.try_process(&[]),
            Err(VoxelTerrainRuntimeError::DataRefcountOverflow {
                location: BlockLocation {
                    position: actual,
                    lod_index: 0,
                },
                field: DataRefField::ResidentViewers,
            }) if actual == position
        ));
    }

    assert_eq!(
        core.data_view_retries[0],
        vec![PendingDataMutation {
            box_in_blocks: single_block_box(position),
            retry_count: 1,
        }]
    );
    assert_eq!(
        core.data.block_snapshot(position, 0).unwrap().viewers.get(),
        u32::MAX
    );
    assert!(matches!(
        core.last_data_mutation_error(),
        Some(SharedVoxelDataMutationError::ViewerCountOverflow { .. })
    ));
}

#[test]
fn terrain_edit_storage_error_is_queryable_and_non_mutating() {
    let stream = Arc::new(MemoryStream::new());
    let mut core = build_core_with_materializable_data(stream);
    let position = Vector3i::zero();
    core.data.with_lod_map_mut(0, |map| {
        map.set_key_revision_for_test(position, u64::MAX);
    });

    assert_eq!(
        core.try_edit_voxel(77, Vector3i::new(1, 1, 1), ChannelId::Type.index(),),
        Err(VoxelTerrainRuntimeError::BlockRevisionOverflow {
            location: BlockLocation {
                position,
                lod_index: 0,
            },
        })
    );
    assert_eq!(
        core.last_data_mutation_error(),
        None,
        "typed edit failures must not leak into the legacy diagnostic slot"
    );
    assert!(core.data.block_snapshot(position, 0).is_none());
}

#[test]
fn load_lifecycle_retry_limit_exhausts_without_leaking_viewer_demand() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let viewed_box = single_block_box(position);
    core.apply_data_view(viewed_box, 0);

    for expected_retry_count in 1..=MAX_LOAD_RETRIES {
        let generation = core.loading_blocks[0][&position].request_generation;
        core.blocks_pending_load[0].clear();
        core.legacy_variable_apply_load_response(TerrainLoadOutput::new(
            BlockDataOutput::loaded_dropped(position, 0),
            generation,
        ));

        let entry = &core.loading_blocks[0][&position];
        assert_eq!(entry.residency.resident_viewers, 1);
        assert_eq!(entry.retry_count, expected_retry_count);
        assert!(entry.request_generation > generation);
        assert_eq!(entry.request_state, LoadRequestState::Queued);
        assert_eq!(core.blocks_pending_load[0], vec![position]);
    }

    let final_generation = core.loading_blocks[0][&position].request_generation;
    core.blocks_pending_load[0].clear();
    core.legacy_variable_apply_load_response(TerrainLoadOutput::new(
        BlockDataOutput::loaded_dropped(position, 0),
        final_generation,
    ));

    let entry = &core.loading_blocks[0][&position];
    assert_eq!(entry.residency.resident_viewers, 1);
    assert_eq!(entry.retry_count, MAX_LOAD_RETRIES + 1);
    assert_eq!(entry.request_generation, final_generation);
    assert_eq!(entry.request_state, LoadRequestState::Exhausted);
    assert!(core.blocks_pending_load[0].is_empty());

    core.apply_data_unview(viewed_box, 0);
    assert!(!core.loading_blocks[0].contains_key(&position));
}

#[test]
fn load_lifecycle_generation_exhaustion_never_wraps_to_an_active_identity() {
    let mut core = build_core();
    core.next_request_generation = u64::MAX;
    let position = Vector3i::zero();

    assert_eq!(
        core.try_apply_variable_data_residency_delta(single_block_box(position), 0, 1),
        Err(VoxelTerrainRuntimeError::RequestGenerationOverflow)
    );
    assert_eq!(core.next_request_generation, u64::MAX);
    assert!(core.loading_blocks[0].is_empty());
    assert!(core.blocks_pending_load[0].is_empty());
    assert!(core.data.block_snapshot(position, 0).is_none());
}

#[test]
fn load_lifecycle_stale_missing_or_error_outputs_leave_newer_demand() {
    for stale_kind in [
        BlockDataOutput::not_found(Vector3i::zero(), 0),
        BlockDataOutput::loaded_dropped(Vector3i::zero(), 0),
    ] {
        let mut core = build_core();
        let position = Vector3i::zero();
        let viewed_box = single_block_box(position);

        core.apply_data_view(viewed_box, 0);
        let request_a = core.loading_blocks[0][&position].request_generation;
        core.apply_data_unview(viewed_box, 0);
        core.apply_data_view(viewed_box, 0);
        let request_b = core.loading_blocks[0][&position].request_generation;
        core.legacy_variable_apply_load_response(TerrainLoadOutput::new(stale_kind, request_a));
        core.legacy_variable_apply_load_response(loaded_output(&core, position, request_b, 44));

        let block = core
            .data()
            .block_snapshot(position, 0)
            .expect("stale output must not remove or mutate request B");
        assert_eq!(block.viewers.get(), 1);
    }
}

#[test]
fn load_lifecycle_not_found_retains_existing_demand_until_fresh_view_rearms() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let viewed_box = single_block_box(position);

    core.apply_data_view(viewed_box, 0);
    core.apply_data_view(viewed_box, 0);
    let not_found_generation = complete_current_not_found(&mut core, position);

    assert_eq!(
        core.loading_blocks[0][&position].residency.resident_viewers,
        2
    );
    assert_eq!(
        core.loading_blocks[0][&position].request_state,
        LoadRequestState::NotFound
    );
    assert!(core.blocks_pending_load[0].is_empty());

    core.apply_data_view(viewed_box, 0);
    let rearmed = &core.loading_blocks[0][&position];
    assert_eq!(rearmed.residency.resident_viewers, 3);
    assert_eq!(rearmed.retry_count, 0);
    assert!(rearmed.request_generation > not_found_generation);
    assert_eq!(rearmed.request_state, LoadRequestState::Queued);
    assert_eq!(core.blocks_pending_load[0], vec![position]);

    let found_generation = rearmed.request_generation;
    core.legacy_variable_apply_load_response(loaded_output(&core, position, found_generation, 55));
    assert_eq!(
        core.data()
            .block_snapshot(position, 0)
            .expect("rearmed load installs all retained demand")
            .viewers
            .get(),
        3
    );

    core.apply_data_unview(viewed_box, 0);
    assert_eq!(
        core.data()
            .block_snapshot(position, 0)
            .expect("two viewers remain")
            .viewers
            .get(),
        2
    );
    core.apply_data_unview(viewed_box, 0);
    assert_eq!(
        core.data()
            .block_snapshot(position, 0)
            .expect("one viewer remains")
            .viewers
            .get(),
        1
    );
    core.apply_data_unview(viewed_box, 0);
    assert!(core.data().block_snapshot(position, 0).is_none());
}

#[test]
fn load_lifecycle_terminal_not_found_unviews_to_zero_without_requeue() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let viewed_box = single_block_box(position);

    core.apply_data_view(viewed_box, 0);
    core.apply_data_view(viewed_box, 0);
    complete_current_not_found(&mut core, position);

    core.apply_data_unview(viewed_box, 0);
    assert_eq!(
        core.loading_blocks[0][&position].residency.resident_viewers,
        1
    );
    assert!(core.blocks_pending_load[0].is_empty());

    core.apply_data_unview(viewed_box, 0);
    assert!(!core.loading_blocks[0].contains_key(&position));
    assert!(core.blocks_pending_load[0].is_empty());
}

#[test]
fn load_lifecycle_fresh_demand_queues_terminal_block_only_once() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let viewed_box = single_block_box(position);

    core.apply_data_view(viewed_box, 0);
    complete_current_not_found(&mut core, position);

    core.apply_data_view(viewed_box, 0);
    core.apply_data_view(viewed_box, 0);

    let entry = &core.loading_blocks[0][&position];
    assert_eq!(entry.residency.resident_viewers, 3);
    assert_eq!(entry.retry_count, 0);
    assert_eq!(entry.request_state, LoadRequestState::Queued);
    assert_eq!(core.blocks_pending_load[0], vec![position]);
}

#[test]
fn load_lifecycle_bulk_terminal_rearm_queues_each_block_exactly_once() {
    let mut core = build_core();
    let region = Box3i::new(Vector3i::new(-8, -2, -2), Vector3i::new(16, 4, 4));
    let positions = region.iter_cells_zxy().collect::<Vec<_>>();

    core.apply_data_view(region, 0);
    for (index, position) in positions.iter().copied().enumerate() {
        let entry = core.loading_blocks[0].get_mut(&position).unwrap();
        if index % 2 == 0 {
            entry.request_state = LoadRequestState::NotFound;
        } else {
            entry.retry_count = MAX_LOAD_RETRIES + 1;
            entry.request_state = LoadRequestState::Exhausted;
        }
    }
    core.blocks_pending_load[0].clear();

    core.apply_data_view(region, 0);

    assert_eq!(core.blocks_pending_load[0].len(), positions.len());
    let queued = core.blocks_pending_load[0]
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(queued.len(), positions.len());
    assert!(positions.iter().all(|position| queued.contains(position)));
    assert!(positions.iter().all(|position| {
        let entry = &core.loading_blocks[0][position];
        entry.residency.resident_viewers == 2
            && entry.retry_count == 0
            && entry.request_state == LoadRequestState::Queued
    }));
}

#[test]
fn load_lifecycle_terminal_not_found_does_not_self_requeue() {
    let mut core = build_core();
    let position = Vector3i::zero();
    core.apply_data_view(single_block_box(position), 0);
    let generation = complete_current_not_found(&mut core, position);

    for _ in 0..3 {
        core.send_data_load_requests();
        assert_eq!(core.pending_task_count(), 0);
        assert!(core.blocks_pending_load[0].is_empty());
    }

    let entry = &core.loading_blocks[0][&position];
    assert_eq!(entry.residency.resident_viewers, 1);
    assert_eq!(entry.request_generation, generation);
    assert_eq!(entry.request_state, LoadRequestState::NotFound);
}

#[test]
fn load_lifecycle_fresh_demand_rearms_exhausted_transient_request() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let viewed_box = single_block_box(position);

    core.apply_data_view(viewed_box, 0);
    let exhausted_generation = core.loading_blocks[0][&position].request_generation;
    let entry = core.loading_blocks[0].get_mut(&position).unwrap();
    entry.retry_count = MAX_LOAD_RETRIES + 1;
    entry.request_state = LoadRequestState::Exhausted;
    core.blocks_pending_load[0].clear();

    core.apply_data_view(viewed_box, 0);

    let entry = &core.loading_blocks[0][&position];
    assert_eq!(entry.residency.resident_viewers, 2);
    assert_eq!(entry.retry_count, 0);
    assert!(entry.request_generation > exhausted_generation);
    assert_eq!(entry.request_state, LoadRequestState::Queued);
    assert_eq!(core.blocks_pending_load[0], vec![position]);
}

fn viewer_validation_core(lod_count: u8) -> VoxelTerrainCore {
    if lod_count == 1 {
        build_core()
    } else {
        make_edit_core_with_lods(lod_count)
    }
}

fn seed_nontrivial_viewer_validation_state(
    core: &mut VoxelTerrainCore,
) -> (PersistenceOperation, *const u8) {
    core.paired_viewers.push(PairedViewer {
        id: 77,
        state: ViewerState {
            local_position_voxels: Vector3i::new(4, 5, 6),
            demand: MeshDemand {
                visuals: true,
                collisions: false,
            },
            ..ViewerState::default()
        },
        prev_state: ViewerState::default(),
    });
    core.event_outbox
        .push_back(VoxelTerrainEvent::DataBlockUnloaded(BlockLocation {
            position: Vector3i::new(7, 8, 9),
            lod_index: 0,
        }));
    core.stats.blocks_loaded = 11;
    core.next_request_generation = 101;
    core.next_mesh_revision = 202;
    core.next_save_generation = 303;

    let location = BlockLocation {
        position: Vector3i::new(12, 13, 14),
        lod_index: 0,
    };
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let operation = stage_single_written_checkpoint_member(core, location, 404, payload);
    core.save_checkpoint_in_flight = Some(SaveCheckpointInFlight {
        checkpoint_generation: 505,
        acknowledged: vec![SaveCheckpointSnapshot {
            key: SaveKey::new(location.position, location.lod_index),
            block_revision: 0,
            generation: 404,
        }],
        state: CheckpointAttemptState::Pending,
        retry_count: 2,
        max_attempts: 8,
        origin: CheckpointOrigin::Automatic,
        record_per_block_failure: false,
    });
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(DebugNameCollisionTask),
        TaskLane::Parallel,
        TaskCompletionStatus::Cancelled,
        Vec::new(),
    ));
    core.durable_completion_inbox
        .push_back(DurableCompletion::UnknownTerminal {
            completed: CompletedTask::new(
                Box::new(DebugNameCollisionTask),
                TaskLane::Serial,
                TaskCompletionStatus::Finished,
                Vec::new(),
            ),
        });
    core.completion_quarantine
        .push_back(QuarantinedCompletion::Other {
            kind: CompletionTaskKind::Unknown,
            completed: CompletedTask::new(
                Box::new(DebugNameCollisionTask),
                TaskLane::Parallel,
                TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run),
                Vec::new(),
            ),
        });
    (operation, payload_ptr)
}

fn viewer_permutations(viewers: &[ViewerUpdate]) -> Vec<Vec<ViewerUpdate>> {
    match viewers {
        [a, b] => vec![vec![*a, *b], vec![*b, *a]],
        [a, b, c] => vec![
            vec![*a, *b, *c],
            vec![*a, *c, *b],
            vec![*b, *a, *c],
            vec![*b, *c, *a],
            vec![*c, *a, *b],
            vec![*c, *b, *a],
        ],
        _ => panic!("viewer permutation fixture supports only two or three values"),
    }
}

#[test]
fn valid_viewer_permutations_normalize_to_one_exact_snapshot() {
    let viewers = vec![
        ViewerUpdate {
            id: 9,
            world_position_voxels: Vector3i::new(9, 0, 0),
            horizontal_view_distance_voxels: 16,
            vertical_view_distance_voxels: 32,
            demand: MeshDemand {
                visuals: false,
                collisions: true,
            },
        },
        ViewerUpdate {
            id: 3,
            world_position_voxels: Vector3i::new(3, 0, 0),
            horizontal_view_distance_voxels: 8,
            vertical_view_distance_voxels: 24,
            demand: MeshDemand {
                visuals: true,
                collisions: false,
            },
        },
    ];
    let reversed = viewers.iter().copied().rev().collect::<Vec<_>>();
    let expected = normalize_and_validate_viewer_updates(&viewers).unwrap();
    assert_eq!(
        normalize_and_validate_viewer_updates(&reversed).unwrap(),
        expected
    );
    assert_eq!(
        expected.iter().map(|viewer| viewer.id).collect::<Vec<_>>(),
        vec![3, 9]
    );

    for lod_count in [1, 2] {
        let mut forward_core = viewer_validation_core(lod_count);
        let mut reverse_core = viewer_validation_core(lod_count);
        forward_core.try_process(&viewers).unwrap();
        reverse_core.try_process(&reversed).unwrap();
        for core in [&forward_core, &reverse_core] {
            assert_eq!(
                core.paired_viewers
                    .iter()
                    .map(|viewer| viewer.id)
                    .collect::<Vec<_>>(),
                vec![3, 9]
            );
            assert_eq!(
                core.paired_viewers
                    .iter()
                    .map(|viewer| ViewerUpdate {
                        id: viewer.id,
                        world_position_voxels: viewer.state.local_position_voxels,
                        horizontal_view_distance_voxels: viewer
                            .state
                            .horizontal_view_distance_voxels,
                        vertical_view_distance_voxels: viewer.state.vertical_view_distance_voxels,
                        demand: viewer.state.demand,
                    })
                    .collect::<Vec<_>>(),
                expected
            );
        }
        assert_eq!(forward_core.paired_viewers, reverse_core.paired_viewers);
    }
}

#[test]
fn paired_viewer_order_is_canonical_across_different_arrival_histories() {
    let viewer = |id| ViewerUpdate {
        id,
        world_position_voxels: Vector3i::new(2, 2, 2),
        horizontal_view_distance_voxels: 0,
        vertical_view_distance_voxels: 0,
        demand: MeshDemand::default(),
    };
    let both = [viewer(9), viewer(3)];

    for lod_count in [1, 2] {
        let mut first_nine = viewer_validation_core(lod_count);
        let mut first_three = viewer_validation_core(lod_count);
        first_nine.try_process(&[viewer(9)]).unwrap();
        first_three.try_process(&[viewer(3)]).unwrap();
        for _ in 0..2 {
            first_nine.try_process(&both).unwrap();
            first_three.try_process(&both).unwrap();
        }

        assert_eq!(
            first_nine
                .paired_viewers
                .iter()
                .map(|viewer| viewer.id)
                .collect::<Vec<_>>(),
            vec![3, 9]
        );
        assert_eq!(first_nine.paired_viewers, first_three.paired_viewers);
    }
}

#[test]
fn try_process_viewer_validation_is_deterministic_and_non_mutating() {
    let invalid_cases = [
        (
            vec![
                ViewerUpdate {
                    id: 9,
                    world_position_voxels: Vector3i::zero(),
                    horizontal_view_distance_voxels: -7,
                    vertical_view_distance_voxels: -8,
                    demand: MeshDemand::default(),
                },
                ViewerUpdate {
                    id: 3,
                    world_position_voxels: Vector3i::zero(),
                    horizontal_view_distance_voxels: -1,
                    vertical_view_distance_voxels: -2,
                    demand: MeshDemand::default(),
                },
                ViewerUpdate {
                    id: 3,
                    world_position_voxels: Vector3i::zero(),
                    horizontal_view_distance_voxels: 1,
                    vertical_view_distance_voxels: 2,
                    demand: MeshDemand::default(),
                },
            ],
            ViewerInputError::DuplicateId(3),
        ),
        (
            vec![
                ViewerUpdate {
                    id: 9,
                    world_position_voxels: Vector3i::zero(),
                    horizontal_view_distance_voxels: -7,
                    vertical_view_distance_voxels: -8,
                    demand: MeshDemand::default(),
                },
                ViewerUpdate {
                    id: 3,
                    world_position_voxels: Vector3i::zero(),
                    horizontal_view_distance_voxels: -1,
                    vertical_view_distance_voxels: -2,
                    demand: MeshDemand::default(),
                },
            ],
            ViewerInputError::NegativeHorizontalDistance { id: 3, value: -1 },
        ),
        (
            vec![
                ViewerUpdate {
                    id: 9,
                    world_position_voxels: Vector3i::zero(),
                    horizontal_view_distance_voxels: 7,
                    vertical_view_distance_voxels: -8,
                    demand: MeshDemand::default(),
                },
                ViewerUpdate {
                    id: 3,
                    world_position_voxels: Vector3i::zero(),
                    horizontal_view_distance_voxels: 1,
                    vertical_view_distance_voxels: -2,
                    demand: MeshDemand::default(),
                },
            ],
            ViewerInputError::NegativeVerticalDistance { id: 3, value: -2 },
        ),
    ];

    for (viewers, expected) in invalid_cases {
        for viewers in viewer_permutations(&viewers) {
            for lod_count in [1, 2] {
                let mut core = viewer_validation_core(lod_count);
                let (save, payload_ptr) = seed_nontrivial_viewer_validation_state(&mut core);
                let paired_before = core.paired_viewers.clone();
                let events_before_len = core.event_outbox.len();
                let stats_before = core.stats;
                let generations_before = (
                    core.next_request_generation,
                    core.next_mesh_revision,
                    core.next_save_generation,
                );

                assert!(matches!(
                    core.try_process(&viewers),
                    Err(VoxelTerrainRuntimeError::ViewerInput(error)) if error == expected
                ));
                assert_eq!(core.paired_viewers, paired_before);
                assert_eq!(core.event_outbox.len(), events_before_len);
                assert!(matches!(
                    core.event_outbox.front(),
                    Some(VoxelTerrainEvent::DataBlockUnloaded(position))
                        if *position == BlockLocation {
                            position: Vector3i::new(7, 8, 9),
                            lod_index: 0,
                        }
                ));
                assert_eq!(core.stats, stats_before);
                assert_eq!(
                    (
                        core.next_request_generation,
                        core.next_mesh_revision,
                        core.next_save_generation,
                    ),
                    generations_before
                );
                assert_eq!(
                    core.journal_persistence_state_for_test(save),
                    Some(JournalPersistenceState::WrittenUnflushed)
                );
                assert_eq!(core.journal_payload_ptr_for_test(save), Some(payload_ptr));
                let checkpoint = core.save_checkpoint_in_flight.as_ref().unwrap();
                assert_eq!(checkpoint.checkpoint_generation, 505);
                assert_eq!(checkpoint.state, CheckpointAttemptState::Pending);
                assert_eq!(checkpoint.retry_count, 2);
                assert_eq!(checkpoint.acknowledged.len(), 1);
                assert_eq!(checkpoint.acknowledged[0].generation, 404);
                assert_eq!(core.raw_completion_inbox.len(), 1);
                assert_eq!(core.durable_completion_inbox.len(), 1);
                assert_eq!(core.completion_quarantine.len(), 1);
                assert!(core.raw_completion_inbox[0]
                    .task_any()
                    .is::<DebugNameCollisionTask>());
                assert!(matches!(
                    core.durable_completion_inbox.front(),
                    Some(DurableCompletion::UnknownTerminal { .. })
                ));
                assert!(matches!(
                    core.completion_quarantine.front(),
                    Some(QuarantinedCompletion::Other {
                        kind: CompletionTaskKind::Unknown,
                        ..
                    })
                ));
            }
        }
    }
}

#[test]
fn visual_and_collision_demands_each_keep_fixed_mesh_residency() {
    for demand in [
        MeshDemand {
            visuals: true,
            collisions: false,
        },
        MeshDemand {
            visuals: false,
            collisions: true,
        },
    ] {
        let mut core = build_core();
        let distance = core.data_block_size();
        core.try_process(&[ViewerUpdate {
            id: 1,
            world_position_voxels: Vector3i::zero(),
            horizontal_view_distance_voxels: distance,
            vertical_view_distance_voxels: distance,
            demand,
        }])
        .unwrap();

        let state = &core.paired_viewers[0].state;
        assert_eq!(state.demand, demand);
        assert!(!state.mesh_box.is_empty());
        assert!(!core.mesh_blocks().is_empty());
    }
}

#[test]
fn try_process_preserves_viewer_demand_changes() {
    let mut core = build_core();
    let distance = core.data_block_size();
    let mut viewer = ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: distance,
        vertical_view_distance_voxels: distance,
        demand: MeshDemand {
            visuals: true,
            collisions: false,
        },
    };
    core.try_process(&[viewer]).unwrap();
    viewer.demand = MeshDemand {
        visuals: false,
        collisions: true,
    };
    core.try_process(&[viewer]).unwrap();

    assert_eq!(core.paired_viewers[0].state.demand, viewer.demand);
    assert!(!core.paired_viewers[0].state.mesh_box.is_empty());
}

#[test]
fn process_does_not_wait_for_slow_load_tasks() {
    let stream: Arc<dyn VoxelStream> = Arc::new(SlowNotFoundStream {
        delay: Duration::from_millis(250),
    });
    let mut core = build_core_with_stream(stream);
    let bs = core.data_block_size();
    let viewers = vec![ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: bs,
        vertical_view_distance_voxels: bs,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];

    let started = Instant::now();
    let _events = core.try_process(&viewers).unwrap();
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_millis(100),
        "process tick should only enqueue/drain tasks, elapsed {elapsed:?}"
    );
}

#[test]
fn load_task_releases_data_lock_before_generator_fallback() {
    let data = Arc::new(SharedVoxelData::new(VoxelData::new()));
    data.set_generator(Some(Arc::new(VoxelDataLockProbeGenerator {
        data: Arc::downgrade(&data),
    })))
    .unwrap();

    let mut task =
        LoadBlockForTerrainTask::new(Vector3i::zero(), 0, 1, data, Arc::new(MemoryStream::new()));
    let outcome = task.run(ThreadedTaskContext::new(0, TaskPriority::max()));
    assert!(matches!(outcome, TaskRunStatus::Complete { .. }));
    let output = task.output.take().expect("load output");

    assert_eq!(output.request_generation, 1);
    assert!(!output.block_data.dropped);
}

#[test]
fn load_task_request_cancellation_blocks_usable_output_after_stream_work() {
    struct CancellingStream(Arc<RequestCancellation>);

    impl VoxelStream for CancellingStream {
        fn load_voxel_block(
            &self,
            query: crate::streams::VoxelLoadQuery<'_>,
        ) -> StreamResult<LoadResult> {
            query
                .voxel_buffer
                .set_voxel(91, 0, 0, 0, ChannelId::Type.index());
            self.0.cancel();
            Ok(LoadResult::Found)
        }
    }

    let cancellation = Arc::new(RequestCancellation::new());
    let tag = TaskRequestTag::new(13, 17);
    let mut task = LoadBlockForTerrainTask::new(
        Vector3i::zero(),
        0,
        tag.request_generation,
        Arc::new(SharedVoxelData::new(VoxelData::new())),
        Arc::new(CancellingStream(cancellation.clone())),
    )
    .with_request_control(tag, cancellation.clone());

    assert_eq!(task.request_tag(), Some(tag));
    assert!(!task.is_cancelled());
    let outcome = task.run(ThreadedTaskContext::new(0, TaskPriority::max()));

    assert!(matches!(outcome, TaskRunStatus::Postponed));
    assert!(cancellation.is_cancelled());
    assert!(task.is_cancelled());
    assert!(task.output.is_none());
}

#[test]
fn runner_cancelled_tagged_load_rearms_live_demand_without_malformed_completion() {
    let mut core = build_core();
    core.automatic_loading_enabled = false;
    core.next_request_generation = 100;
    let position = Vector3i::new(76, 0, 0);
    let old_generation = 41;
    let old_request = PhysicalRequest::new(TaskRequestTag::new(core.request_epoch, old_generation));
    core.loading_blocks[0].insert(
        position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: old_generation,
            request_state: LoadRequestState::InFlight,
            physical_request: Some(old_request.clone()),
        },
    );
    old_request.cancel();
    let task = LoadBlockForTerrainTask::new(
        position,
        0,
        old_generation,
        core.data.clone(),
        core.stream.clone(),
    )
    .with_request_control(old_request.tag, old_request.cancellation.clone());
    core.task_runner
        .enqueue(ScheduledTask::new(Box::new(task), TaskLane::Parallel));
    core.task_runner.wait_for_all_tasks();

    core.try_drain_completed_tasks().unwrap();

    let entry = &core.loading_blocks[0][&position];
    assert_eq!(entry.request_state, LoadRequestState::Queued);
    assert_eq!(entry.request_generation, 100);
    assert_eq!(
        entry.physical_request.as_ref().unwrap().tag,
        TaskRequestTag::new(core.request_epoch, 100)
    );
    assert!(core.raw_completion_inbox.is_empty());
    assert!(core.durable_completion_inbox.is_empty());
    assert!(core.completion_quarantine.is_empty());
}

#[test]
fn fixed_old_epoch_finished_and_terminal_loads_are_stale_at_same_generation() {
    let mut core = build_core();
    core.request_epoch = 9;
    let position = Vector3i::new(73, 0, 0);
    let generation = 41;
    let current = PhysicalRequest::new(TaskRequestTag::new(9, generation));
    core.loading_blocks[0].insert(
        position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: generation,
            request_state: LoadRequestState::InFlight,
            physical_request: Some(current),
        },
    );

    let old_tag = TaskRequestTag::new(8, generation);
    let old_finished_cancellation = Arc::new(RequestCancellation::new());
    let mut finished = LoadBlockForTerrainTask::new(
        position,
        0,
        generation,
        core.data.clone(),
        core.stream.clone(),
    )
    .with_request_control(old_tag, old_finished_cancellation);
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
    voxels.set_voxel(0xE0, 0, 0, 0, ChannelId::Type.index());
    finished.output = Some(TerrainLoadOutput::new_tagged(
        BlockDataOutput::loaded(position, 0, voxels, false),
        old_tag,
    ));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(finished),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));

    let terminal = LoadBlockForTerrainTask::new(
        position,
        0,
        generation,
        core.data.clone(),
        core.stream.clone(),
    )
    .with_request_control(old_tag, Arc::new(RequestCancellation::new()));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(terminal),
        TaskLane::Parallel,
        TaskCompletionStatus::Cancelled,
        Vec::new(),
    ));

    core.try_drain_completed_tasks().unwrap();

    assert!(core.data.block_snapshot(position, 0).is_none());
    let entry = &core.loading_blocks[0][&position];
    assert_eq!(entry.request_generation, generation);
    assert_eq!(entry.retry_count, 0);
    assert_eq!(entry.request_state, LoadRequestState::InFlight);
    assert_eq!(
        entry.physical_request.as_ref().unwrap().tag,
        TaskRequestTag::new(9, generation)
    );
    assert!(core.blocks_pending_load[0].is_empty());
}

#[test]
fn fixed_old_epoch_finished_and_terminal_meshes_are_stale_at_same_revision() {
    let mut core = make_resident_edit_core();
    core.automatic_loading_enabled = false;
    core.request_epoch = 12;
    let position = Vector3i::zero();
    let revision = 55;
    let request_generation = 56;
    let current_tag = TaskRequestTag::new(12, request_generation);
    {
        let entry = core.mesh_maps[0].get_mut(&position).unwrap();
        entry.requested_revision = Some(revision);
        entry.request_generation = request_generation;
        entry.is_in_update_list = false;
        entry.terminal_retry_count = 0;
        entry.physical_request = Some(PhysicalRequest::new(current_tag));
    }
    core.blocks_pending_update[0].clear();

    let old_tag = TaskRequestTag::new(11, request_generation);
    let make_task = |core: &VoxelTerrainCore| {
        MeshBlockTask::new(MeshBlockTaskParams {
            key: MeshBlockKey {
                location: MeshBlockLocation::new(position, 0),
                revision,
            },
            data: core.data.clone(),
            meshing_dependency: core.meshing_dependency.clone(),
            collision_hint: false,
            lod_hint: false,
            mesh_arrays_pool: Some(core.mesh_arrays_pool.clone()),
        })
        .with_request_control(old_tag, Arc::new(RequestCancellation::new()))
    };
    let mut finished = make_task(&core);
    finished.run_meshing();
    assert_eq!(finished.take_output().unwrap().request_tag(), Some(old_tag));
    finished.run_meshing();
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(finished),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(make_task(&core)),
        TaskLane::Parallel,
        TaskCompletionStatus::Cancelled,
        Vec::new(),
    ));

    core.try_drain_completed_tasks().unwrap();

    let entry = &core.mesh_maps[0][&position];
    assert_eq!(entry.requested_revision, Some(revision));
    assert_eq!(entry.applied_revision, None);
    assert_eq!(entry.terminal_retry_count, 0);
    assert!(!entry.is_in_update_list);
    assert_eq!(entry.physical_request.as_ref().unwrap().tag, current_tag);
    assert!(core.blocks_pending_update[0].is_empty());
}

#[test]
fn fixed_cancelled_previous_epoch_terminals_rearm_live_demand_in_current_epoch() {
    let mut core = make_resident_edit_core();
    core.automatic_loading_enabled = false;
    core.request_epoch = 1;

    let load_position = Vector3i::new(74, 0, 0);
    let load_generation = 61;
    let old_load = PhysicalRequest::new(TaskRequestTag::new(0, load_generation));
    old_load.cancel();
    core.loading_blocks[0].insert(
        load_position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: load_generation,
            request_state: LoadRequestState::InFlight,
            physical_request: Some(old_load.clone()),
        },
    );
    let load_terminal = LoadBlockForTerrainTask::new(
        load_position,
        0,
        load_generation,
        core.data.clone(),
        core.stream.clone(),
    )
    .with_request_control(old_load.tag, old_load.cancellation.clone());
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(load_terminal),
        TaskLane::Parallel,
        TaskCompletionStatus::Cancelled,
        Vec::new(),
    ));

    let mesh_position = Vector3i::zero();
    let mesh_revision = 62;
    let old_mesh_generation = 63;
    let old_mesh = PhysicalRequest::new(TaskRequestTag::new(0, old_mesh_generation));
    old_mesh.cancel();
    {
        let entry = core.mesh_maps[0].get_mut(&mesh_position).unwrap();
        entry.requested_revision = Some(mesh_revision);
        entry.request_generation = old_mesh_generation;
        entry.is_in_update_list = false;
        entry.physical_request = Some(old_mesh.clone());
    }
    core.blocks_pending_update[0].clear();
    let mesh_terminal = MeshBlockTask::new(MeshBlockTaskParams {
        key: MeshBlockKey {
            location: MeshBlockLocation::new(mesh_position, 0),
            revision: mesh_revision,
        },
        data: core.data.clone(),
        meshing_dependency: core.meshing_dependency.clone(),
        collision_hint: false,
        lod_hint: false,
        mesh_arrays_pool: Some(core.mesh_arrays_pool.clone()),
    })
    .with_request_control(old_mesh.tag, old_mesh.cancellation.clone());
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(mesh_terminal),
        TaskLane::Parallel,
        TaskCompletionStatus::Cancelled,
        Vec::new(),
    ));

    core.try_drain_completed_tasks().unwrap();

    let load = &core.loading_blocks[0][&load_position];
    assert_ne!(load.request_generation, load_generation);
    assert_eq!(load.request_state, LoadRequestState::Queued);
    assert_eq!(
        load.physical_request.as_ref().unwrap().tag,
        TaskRequestTag::new(1, load.request_generation)
    );
    assert!(!load
        .physical_request
        .as_ref()
        .unwrap()
        .cancellation
        .is_cancelled());
    assert_eq!(core.blocks_pending_load[0], vec![load_position]);

    let mesh = &core.mesh_maps[0][&mesh_position];
    assert_eq!(mesh.requested_revision, Some(mesh_revision));
    assert_ne!(mesh.request_generation, old_mesh_generation);
    assert!(mesh.is_in_update_list);
    assert_eq!(
        mesh.physical_request.as_ref().unwrap().tag,
        TaskRequestTag::new(1, mesh.request_generation)
    );
    assert!(!mesh
        .physical_request
        .as_ref()
        .unwrap()
        .cancellation
        .is_cancelled());
    assert_eq!(core.blocks_pending_update[0], vec![mesh_position]);
}

#[test]
fn fixed_finished_previous_epoch_outputs_are_rejected_and_rearm_current_epoch() {
    let mut core = make_resident_edit_core();
    core.automatic_loading_enabled = false;
    core.request_epoch = 3;

    let load_position = Vector3i::new(75, 0, 0);
    let load_generation = 71;
    let old_load = PhysicalRequest::new(TaskRequestTag::new(2, load_generation));
    old_load.cancel();
    core.loading_blocks[0].insert(
        load_position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: load_generation,
            request_state: LoadRequestState::InFlight,
            physical_request: Some(old_load.clone()),
        },
    );
    let mut load_task = LoadBlockForTerrainTask::new(
        load_position,
        0,
        load_generation,
        core.data.clone(),
        core.stream.clone(),
    )
    .with_request_control(old_load.tag, old_load.cancellation.clone());
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
    voxels.set_voxel(0xE3, 0, 0, 0, ChannelId::Type.index());
    load_task.output = Some(TerrainLoadOutput::new_tagged(
        BlockDataOutput::loaded(load_position, 0, voxels, false),
        old_load.tag,
    ));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(load_task),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));

    let mesh_position = Vector3i::zero();
    let mesh_revision = 72;
    let old_mesh_generation = 73;
    let old_mesh = PhysicalRequest::new(TaskRequestTag::new(2, old_mesh_generation));
    {
        let entry = core.mesh_maps[0].get_mut(&mesh_position).unwrap();
        entry.requested_revision = Some(mesh_revision);
        entry.request_generation = old_mesh_generation;
        entry.is_in_update_list = false;
        entry.physical_request = Some(old_mesh.clone());
    }
    core.blocks_pending_update[0].clear();
    let mut mesh_task = MeshBlockTask::new(MeshBlockTaskParams {
        key: MeshBlockKey {
            location: MeshBlockLocation::new(mesh_position, 0),
            revision: mesh_revision,
        },
        data: core.data.clone(),
        meshing_dependency: core.meshing_dependency.clone(),
        collision_hint: false,
        lod_hint: false,
        mesh_arrays_pool: Some(core.mesh_arrays_pool.clone()),
    })
    .with_request_control(old_mesh.tag, old_mesh.cancellation.clone());
    mesh_task.run_meshing();
    old_mesh.cancel();
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(mesh_task),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));

    core.try_drain_completed_tasks().unwrap();

    assert!(core.data.block_snapshot(load_position, 0).is_none());
    let load = &core.loading_blocks[0][&load_position];
    assert_eq!(load.request_state, LoadRequestState::Queued);
    assert_ne!(load.request_generation, load_generation);
    assert_eq!(
        load.physical_request.as_ref().unwrap().tag,
        TaskRequestTag::new(3, load.request_generation)
    );
    assert_eq!(core.blocks_pending_load[0], vec![load_position]);

    let mesh = &core.mesh_maps[0][&mesh_position];
    assert_eq!(mesh.applied_revision, None);
    assert!(mesh.is_in_update_list);
    assert_ne!(mesh.request_generation, old_mesh_generation);
    assert_eq!(
        mesh.physical_request.as_ref().unwrap().tag,
        TaskRequestTag::new(3, mesh.request_generation)
    );
    assert_eq!(core.blocks_pending_update[0], vec![mesh_position]);
}

#[test]
fn fixed_request_tokens_cancel_only_after_committed_final_demand_removal() {
    let mut core = build_core();
    let demand = MeshDemand {
        visuals: true,
        collisions: true,
    };
    let first = fixed_zero_distance_viewer(801, Vector3i::zero(), demand);
    let second = fixed_zero_distance_viewer(802, Vector3i::zero(), demand);
    core.prepare_fixed_viewer_transaction(&[first, second], true, false, false)
        .unwrap();
    let position = *core.loading_blocks[0].keys().next().unwrap();
    let load_cancellation = core.loading_blocks[0][&position]
        .physical_request
        .as_ref()
        .unwrap()
        .cancellation
        .clone();
    let mesh_position = *core.mesh_maps[0].keys().next().unwrap();
    let mesh_cancellation = core.mesh_maps[0][&mesh_position]
        .physical_request
        .as_ref()
        .unwrap()
        .cancellation
        .clone();

    core.prepare_fixed_viewer_transaction(&[first], true, false, false)
        .unwrap();
    assert!(!load_cancellation.is_cancelled());
    assert!(!mesh_cancellation.is_cancelled());

    core.fail_fixed_capacity_for_test(FixedCapacityDestination::Retirement, 1);
    assert!(matches!(
        core.prepare_fixed_viewer_transaction(&[], true, false, false),
        Err(VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)
    ));
    assert!(!load_cancellation.is_cancelled());
    assert!(!mesh_cancellation.is_cancelled());

    core.prepare_fixed_viewer_transaction(&[], true, false, false)
        .unwrap();
    assert!(load_cancellation.is_cancelled());
    assert!(mesh_cancellation.is_cancelled());
}

#[test]
fn fixed_supersede_cancels_only_after_committed_replacement() {
    let mut core = build_core();
    let viewer = fixed_zero_distance_viewer(811, Vector3i::zero(), MeshDemand::default());
    core.prepare_fixed_viewer_transaction(&[viewer], true, false, false)
        .unwrap();
    let position = *core.loading_blocks[0].keys().next().unwrap();
    let old_generation = core.loading_blocks[0][&position].request_generation;
    let old_cancellation = {
        let entry = core.loading_blocks[0].get_mut(&position).unwrap();
        entry.request_state = LoadRequestState::NotFound;
        entry
            .physical_request
            .as_ref()
            .unwrap()
            .cancellation
            .clone()
    };
    let second = fixed_zero_distance_viewer(812, Vector3i::zero(), MeshDemand::default());

    core.fail_fixed_capacity_for_test(FixedCapacityDestination::Retirement, 1);
    assert!(matches!(
        core.prepare_fixed_viewer_transaction(&[viewer, second], true, false, false),
        Err(VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)
    ));
    assert!(!old_cancellation.is_cancelled());
    assert_eq!(
        core.loading_blocks[0][&position].request_generation,
        old_generation
    );

    core.prepare_fixed_viewer_transaction(&[viewer, second], true, false, false)
        .unwrap();
    assert!(old_cancellation.is_cancelled());
    let replacement = &core.loading_blocks[0][&position];
    assert!(replacement.request_generation > old_generation);
    assert_eq!(
        replacement.physical_request.as_ref().unwrap().tag,
        TaskRequestTag::new(core.request_epoch, replacement.request_generation)
    );
    assert!(!replacement
        .physical_request
        .as_ref()
        .unwrap()
        .cancellation
        .is_cancelled());
}

#[test]
fn shutdown_attempt_advances_epoch_once_and_cancels_before_waiting() {
    let mut core = build_core();
    let viewer = fixed_zero_distance_viewer(821, Vector3i::zero(), MeshDemand::default());
    core.prepare_fixed_viewer_transaction(&[viewer], true, false, false)
        .unwrap();
    let live = core.loading_blocks[0]
        .values()
        .next()
        .unwrap()
        .physical_request
        .as_ref()
        .unwrap()
        .cancellation
        .clone();
    assert_eq!(core.request_epoch, 0);

    core.data.set_test_transaction_reservation_failpoint(Some(
        crate::storage::voxel_data::SharedVoxelDataTransactionReservation::SpatialPreparation,
    ));
    assert!(matches!(
        core.shutdown_and_flush(),
        Err(SaveFlushError::DataMutation(
            SharedVoxelDataMutationError::PreparedTransactionCapacityReservationFailed {
                reservation:
                    crate::storage::voxel_data::SharedVoxelDataTransactionReservation::SpatialPreparation,
            }
        ))
    ));
    assert_eq!(core.request_epoch, 1);
    assert_eq!(core.next_shutdown_epoch, 1);
    assert_eq!(core.shutdown_epoch, Some(1));
    assert!(core.shutdown_mutation_permit.is_some());
    assert!(live.is_cancelled());
    assert!(!core.shutdown_in_progress);
    assert!(!core.shut_down);

    assert!(matches!(
        core.try_process(&[viewer]),
        Err(VoxelTerrainRuntimeError::ShutdownRetryPending)
    ));

    core.data.set_test_transaction_reservation_failpoint(None);
    core.shutdown_and_flush().unwrap();
    assert_eq!(core.request_epoch, 1);
    assert_eq!(core.next_shutdown_epoch, 1);
    assert_eq!(core.shutdown_epoch, Some(1));
    assert!(core.shutdown_mutation_permit.is_none());
    assert!(core.shut_down);
}

#[test]
fn begin_shutdown_rolls_back_before_signalling_on_either_epoch_overflow() {
    for overflow_request_epoch in [true, false] {
        let mut core = build_core();
        let viewer = fixed_zero_distance_viewer(831, Vector3i::zero(), MeshDemand::default());
        core.prepare_fixed_viewer_transaction(&[viewer], true, false, false)
            .unwrap();
        let cancellation = core.loading_blocks[0]
            .values()
            .next()
            .unwrap()
            .physical_request
            .as_ref()
            .unwrap()
            .cancellation
            .clone();
        core.request_epoch = if overflow_request_epoch { u64::MAX } else { 7 };
        core.next_shutdown_epoch = if overflow_request_epoch { 9 } else { u64::MAX };
        let before_request_epoch = core.request_epoch;
        let before_next_shutdown_epoch = core.next_shutdown_epoch;

        let expected = if overflow_request_epoch {
            VoxelTerrainRuntimeError::RequestEpochOverflow
        } else {
            VoxelTerrainRuntimeError::ShutdownEpochOverflow
        };
        assert_eq!(
            core.begin_shutdown_attempt(),
            Err(SaveFlushError::SaveAdmission { error: expected })
        );
        assert_eq!(core.request_epoch, before_request_epoch);
        assert_eq!(core.next_shutdown_epoch, before_next_shutdown_epoch);
        assert_eq!(core.shutdown_epoch, None);
        assert!(core.shutdown_mutation_permit.is_none());
        assert!(!core.shutdown_in_progress);
        assert!(!cancellation.is_cancelled());
        assert!(!core.data.is_mutation_admission_closed());
    }
}

fn finished_mesh_task_for_request(
    core: &VoxelTerrainCore,
    key: MeshBlockKey,
    request: &PhysicalRequest,
) -> CompletedTask {
    let mut task = MeshBlockTask::new(MeshBlockTaskParams {
        key,
        data: core.data.clone(),
        meshing_dependency: core.meshing_dependency.clone(),
        collision_hint: false,
        lod_hint: false,
        mesh_arrays_pool: Some(core.mesh_arrays_pool.clone()),
    })
    .with_request_control(request.tag, request.cancellation.clone());
    task.run_meshing();
    assert!(task.output_ref().is_some_and(|output| !output.dropped()));
    CompletedTask::new(
        Box::new(task),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    )
}

#[test]
fn runner_and_direct_same_revision_race_accepts_exactly_once_in_either_order() {
    for runner_first in [true, false] {
        let mut core = make_resident_edit_core();
        core.automatic_loading_enabled = false;
        let position = Vector3i::zero();
        let key = MeshBlockKey {
            location: MeshBlockLocation::new(position, 0),
            revision: 901,
        };
        let request = PhysicalRequest::new(TaskRequestTag::new(0, 902));
        {
            let entry = core.mesh_maps[0].get_mut(&position).unwrap();
            entry.requested_revision = Some(key.revision);
            entry.request_generation = request.tag.request_generation;
            entry.requested_features = MeshBuildFeatures {
                visuals: true,
                collisions: false,
                variable_lod: false,
            };
            entry.applied_revision = None;
            entry.physical_request = Some(request.clone());
            entry.is_in_update_list = false;
        }
        let runner_completion = finished_mesh_task_for_request(&core, key, &request);

        if runner_first {
            core.raw_completion_inbox.push_back(runner_completion);
            core.try_apply_mesh_output(nonempty_mesh_output(key))
                .unwrap();
        } else {
            core.try_apply_mesh_output(nonempty_mesh_output(key))
                .unwrap();
            core.raw_completion_inbox.push_back(runner_completion);
            core.try_drain_completed_tasks().unwrap();
        }

        let entry = &core.mesh_maps[0][&position];
        assert_eq!(entry.applied_revision, Some(key.revision));
        assert!(entry.physical_request.is_none());
        assert_eq!(core.stats.meshes_built, 1);
        assert_eq!(core.stats.meshes_dropped, 1);
    }
}

#[test]
fn dropped_mesh_retries_same_revision_with_fresh_physical_generation() {
    let mut core = make_resident_edit_core();
    core.automatic_loading_enabled = false;
    core.next_request_generation = 1000;
    let position = Vector3i::zero();
    let key = MeshBlockKey {
        location: MeshBlockLocation::new(position, 0),
        revision: 991,
    };
    let request = PhysicalRequest::new(TaskRequestTag::new(0, 992));
    {
        let entry = core.mesh_maps[0].get_mut(&position).unwrap();
        entry.requested_revision = Some(key.revision);
        entry.request_generation = request.tag.request_generation;
        entry.physical_request = Some(request.clone());
        entry.is_in_update_list = false;
    }
    request.cancel();
    let mut task = MeshBlockTask::new(MeshBlockTaskParams {
        key,
        data: core.data.clone(),
        meshing_dependency: core.meshing_dependency.clone(),
        collision_hint: false,
        lod_hint: false,
        mesh_arrays_pool: Some(core.mesh_arrays_pool.clone()),
    })
    .with_request_control(request.tag, request.cancellation.clone());
    task.run_meshing();
    assert!(task.output_ref().is_some_and(MeshBlockTaskOutput::dropped));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));

    core.try_drain_completed_tasks().unwrap();

    let entry = &core.mesh_maps[0][&position];
    assert_eq!(entry.requested_revision, Some(key.revision));
    assert_eq!(entry.request_generation, 1000);
    assert_eq!(
        entry.physical_request.as_ref().unwrap().tag,
        TaskRequestTag::new(0, 1000)
    );
    assert!(!entry
        .physical_request
        .as_ref()
        .unwrap()
        .cancellation
        .is_cancelled());
    assert!(entry.is_in_update_list);
}

#[test]
fn load_task_releases_data_lock_before_stream_load() {
    let data = Arc::new(SharedVoxelData::new(VoxelData::new()));
    let stream: Arc<dyn VoxelStream> = Arc::new(VoxelDataLockProbeStream {
        data: Arc::downgrade(&data),
    });

    let mut task = LoadBlockForTerrainTask::new(Vector3i::zero(), 0, 1, data, stream);
    let outcome = task.run(ThreadedTaskContext::new(0, TaskPriority::max()));
    assert!(matches!(outcome, TaskRunStatus::Complete { .. }));
    let output = task.output.take().expect("load output");

    assert_eq!(output.request_generation, 1);
    assert!(!output.block_data.dropped);
    assert!(output.block_data.voxels.is_none());
}

#[test]
fn paging_loads_and_meshes_blocks_around_a_viewer() {
    let mut core = build_core();
    let bs = core.data_block_size();

    // Place a viewer at the world origin with a small view distance. The
    // terrain should load the central data block and mesh the central
    // mesh block.
    let viewers = vec![ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: bs,
        vertical_view_distance_voxels: bs,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];

    // Async ticks: the first call schedules loads, later calls drain load
    // results, schedule meshing, and drain mesh outputs.
    let events = process_until(&mut core, &viewers, |core, events| {
        events
            .iter()
            .any(|e| matches!(e, VoxelTerrainEvent::MeshBlockEntered(_)))
            && core
                .mesh_blocks()
                .get(&Vector3i::zero())
                .is_some_and(|e| e.is_loaded)
    });
    assert!(
        events
            .iter()
            .any(|e| matches!(e, VoxelTerrainEvent::MeshBlockEntered(_))),
        "expected a mesh to be produced, events were {events:?}"
    );
    assert!(core
        .mesh_blocks()
        .get(&Vector3i::zero())
        .is_some_and(|e| e.is_loaded));
    assert!(core.stats.blocks_loaded > 0);
}

#[test]
fn paging_unloads_blocks_when_viewer_moves_away() {
    let mut core = build_core();
    let bs = core.data_block_size();

    let viewer_near = vec![ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: bs,
        vertical_view_distance_voxels: bs,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];
    process_until(&mut core, &viewer_near, |core, _events| {
        core.mesh_blocks()
            .get(&Vector3i::zero())
            .is_some_and(|entry| entry.is_loaded)
    });
    let loaded_after_first = core.mesh_blocks().len();
    assert!(loaded_after_first > 0);

    // Move the viewer very far away (out of view distance). The mesh
    // block should unload on the next tick.
    let viewer_far = vec![ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::splat(bs * 100),
        horizontal_view_distance_voxels: bs,
        vertical_view_distance_voxels: bs,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];
    let events = process_until(&mut core, &viewer_far, |_core, events| {
        events
            .iter()
            .any(|e| matches!(e, VoxelTerrainEvent::MeshBlockExited(_)))
    });
    assert!(
        events
            .iter()
            .any(|e| matches!(e, VoxelTerrainEvent::MeshBlockExited(_))),
        "expected mesh blocks to exit, events were {events:?}"
    );
    // The origin block should no longer be tracked (viewer is far away).
    assert!(
        core.mesh_blocks().get(&Vector3i::zero()).is_none(),
        "expected origin block unloaded, mesh_map still has {:?}",
        core.mesh_blocks().keys().collect::<Vec<_>>()
    );
}

#[test]
fn terrain_try_edit_voxel_materializes_marks_and_persists_on_unload() {
    let stream = Arc::new(MemoryStream::new());
    let mut core = build_core_with_materializable_data(stream.clone());
    let bs = core.data_block_size();
    let channel = ChannelId::Type.index();
    let edited_voxel = Vector3i::new(1, 1, 1);

    assert!(core
        .data
        .try_set_block(Vector3i::zero(), VoxelDataBlock::empty(0))
        .unwrap());
    core.loaded_data_residency[0].insert(
        Vector3i::zero(),
        DataResidencyRefs::with_resident_viewers(0),
    );

    let viewer = vec![ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: bs,
        vertical_view_distance_voxels: bs,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];
    core.try_process(&viewer).unwrap();

    assert!(core
        .try_edit_voxel(77, edited_voxel, channel)
        .unwrap()
        .is_some());
    let block = core
        .data()
        .block_snapshot(Vector3i::zero(), 0)
        .expect("terrain edit should materialize the viewed block");
    assert!(block.has_voxels());
    assert!(block.is_modified());
    assert!(block.is_edited());

    let empty_viewers = Vec::new();
    process_until(&mut core, &empty_viewers, |_core, _events| {
        let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
        stream.load_block(Vector3i::zero(), 0, &mut loaded) == LoadResult::Found
            && loaded.get_voxel(1, 1, 1, channel) == 77
    });

    let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(Vector3i::zero(), 0, &mut loaded),
        LoadResult::Found
    );
    assert_eq!(loaded.get_voxel(1, 1, 1, channel), 77);
}

#[test]
fn failed_unload_save_keeps_payload_and_retries() {
    let stream = Arc::new(FailThenMemoryStream::new(1));
    let mut core = build_core_with_stream(stream.clone());
    let bs = core.data_block_size();
    let channel = ChannelId::Type.index();
    let edited_voxel = Vector3i::new(1, 1, 1);

    let viewer = vec![ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: bs,
        vertical_view_distance_voxels: bs,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];
    process_until(&mut core, &viewer, |core, _events| {
        core.data().block_snapshot(Vector3i::zero(), 0).is_some()
    });

    assert!(core
        .try_edit_voxel(88, edited_voxel, channel)
        .unwrap()
        .is_some());

    let empty_viewers = Vec::new();
    process_until(&mut core, &empty_viewers, |_core, _events| {
        let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
        stream.load_block(Vector3i::zero(), 0, &mut loaded) == LoadResult::Found
            && loaded.get_voxel(1, 1, 1, channel) == 88
    });
}

#[test]
fn stale_save_completion_not_matching_in_flight_generation_is_ignored() {
    let mut core = build_core();
    let key = SaveKey::new(Vector3i::zero(), 0);
    core.save_journal
        .insert(key, SaveJournalEntry::write_in_flight_for_test(2, 0));

    core.apply_save_response(save_result(
        BlockDataOutput::saved(Vector3i::zero(), 0, true, 1),
        None,
        Some(VoxelBuffer::with_size(Vector3i::splat(2))),
    ));

    let entry = core.save_journal.get(&key).expect("newer save must remain");
    assert!(matches!(
        entry.active,
        Some(ActiveSaveAttempt::WriteInFlight {
            ref meta,
            attempt_ordinal: 0,
        }) if meta.generation == 2
    ));
}

#[test]
fn stale_save_completion_with_wrong_block_revision_is_ignored() {
    let mut core = build_core();
    let location = BlockLocation {
        position: Vector3i::new(4, 5, 6),
        lod_index: 0,
    };
    let key = SaveKey::new(location.position, location.lod_index);
    core.save_journal.insert(
        key,
        SaveJournalEntry::write_in_flight_for_test_at_revision(7, 13, 19),
    );
    let stale = SaveTaskTerminal {
        location,
        block_revision: 8,
        save_generation: 13,
        payload: VoxelBuffer::with_size(Vector3i::splat(2)),
        task_panic_phase: None,
        phase: PersistenceIoPhase::Acknowledged,
        acknowledgement: Some(PersistenceAcknowledgement::Save(Ok(()))),
    };

    assert!(!core.apply_save_response_for_attempt(stale, 19));

    assert!(matches!(
        core.save_journal[&key].active,
        Some(ActiveSaveAttempt::WriteInFlight {
            ref meta,
            attempt_ordinal: 19,
        }) if meta.block_revision == 7 && meta.generation == 13
    ));
}

#[test]
fn checkpoint_ack_with_wrong_block_revision_cannot_retire_written_owner() {
    let mut core = build_core();
    let key = SaveKey::new(Vector3i::new(7, 8, 9), 0);
    core.save_journal.insert(
        key,
        SaveJournalEntry {
            written_unflushed: Some(WrittenSave {
                block_revision: 17,
                generation: 23,
                payload: VoxelBuffer::with_size(Vector3i::splat(2)),
            }),
            active: None,
            queued_newer: VecDeque::new(),
        },
    );

    core.apply_acknowledged_checkpoint_result(
        vec![SaveCheckpointSnapshot {
            key,
            block_revision: 18,
            generation: 23,
        }],
        Ok(()),
        false,
    )
    .unwrap();

    let written = core.save_journal[&key].written_unflushed.as_ref().unwrap();
    assert_eq!((written.block_revision, written.generation), (17, 23));
    core.apply_acknowledged_checkpoint_result(
        vec![SaveCheckpointSnapshot {
            key,
            block_revision: 17,
            generation: 23,
        }],
        Ok(()),
        false,
    )
    .unwrap();
    assert!(!core.save_journal.contains_key(&key));
}

#[test]
fn failed_checkpoint_shadow_restores_written_owner_ahead_of_pending_successor() {
    let mut core = build_core();
    let key = SaveKey::new(Vector3i::new(8, 9, 10), 0);
    const WRITTEN_REVISION: u64 = 17;
    const WRITTEN_GENERATION: u64 = 23;
    const SUCCESSOR_REVISION: u64 = 18;
    const SUCCESSOR_GENERATION: u64 = 24;
    const ATTEMPT_ORDINAL: u64 = 29;
    core.save_journal.insert(
        key,
        SaveJournalEntry {
            written_unflushed: Some(WrittenSave {
                block_revision: WRITTEN_REVISION,
                generation: WRITTEN_GENERATION,
                payload: VoxelBuffer::with_size(Vector3i::splat(2)),
            }),
            active: Some(ActiveSaveAttempt::Pending(PendingSave {
                meta: SaveAttemptMeta {
                    block_revision: SUCCESSOR_REVISION,
                    generation: SUCCESSOR_GENERATION,
                    retry_count: 3,
                    last_error: None,
                },
                payload: VoxelBuffer::with_size(Vector3i::splat(2)),
            })),
            queued_newer: VecDeque::new(),
        },
    );
    core.save_checkpoint_in_flight = Some(SaveCheckpointInFlight {
        checkpoint_generation: 31,
        acknowledged: vec![SaveCheckpointSnapshot {
            key,
            block_revision: WRITTEN_REVISION,
            generation: WRITTEN_GENERATION,
        }],
        state: CheckpointAttemptState::WriteInFlight {
            attempt_ordinal: ATTEMPT_ORDINAL,
        },
        retry_count: 0,
        max_attempts: MAX_AUTOMATIC_CHECKPOINT_ATTEMPTS,
        origin: CheckpointOrigin::Automatic,
        record_per_block_failure: false,
    });
    let mut checkpoint_shadow = core
        .save_checkpoint_in_flight
        .as_ref()
        .map(PreparedCheckpointShadow::from_checkpoint);
    let mut journal_shadow = HashMap::from([(
        key,
        PreparedJournalShadow::from_entry(&core.save_journal[&key]),
    )]);
    let terminal = FlushTaskTerminal {
        checkpoint_generation: 31,
        task_panic_phase: None,
        phase: PersistenceIoPhase::Acknowledged,
        acknowledgement: Some(PersistenceAcknowledgement::Flush(Err(
            VoxelStreamError::Io("checkpoint failed".into()),
        ))),
    };

    let action = core
        .prepare_checkpoint_completion_action(
            &mut checkpoint_shadow,
            &mut journal_shadow,
            &terminal,
            ATTEMPT_ORDINAL,
        )
        .unwrap();

    assert!(matches!(
        action,
        Some(PreparedCheckpointAction::Acknowledge {
            succeeded: false,
            ..
        })
    ));
    let shadow = &journal_shadow[&key];
    assert_eq!(shadow.written_block_revision, None);
    assert_eq!(shadow.written_generation, None);
    assert_eq!(
        shadow.active,
        PreparedJournalActiveShadow::Pending {
            block_revision: WRITTEN_REVISION,
            generation: WRITTEN_GENERATION,
            retry_count: 0,
        }
    );
    assert_eq!(shadow.queued_len, 1);
    assert_eq!(
        shadow.queued_front,
        Some((SUCCESSOR_REVISION, SUCCESSOR_GENERATION, 3))
    );
}

#[test]
fn stale_save_failure_does_not_replace_newer_journal_cause() {
    let mut core = build_core();
    let key = SaveKey::new(Vector3i::zero(), 0);
    let newer_cause =
        crate::streams::VoxelStreamError::CorruptData("newer generation failure".into());
    core.shutdown_in_progress = true;
    core.save_journal.insert(
        key,
        SaveJournalEntry {
            written_unflushed: None,
            active: Some(ActiveSaveAttempt::WriteInFlight {
                meta: SaveAttemptMeta {
                    block_revision: 0,
                    generation: 1,
                    retry_count: 0,
                    last_error: None,
                },
                attempt_ordinal: 0,
            }),
            queued_newer: VecDeque::from([PendingSave {
                meta: SaveAttemptMeta {
                    block_revision: 0,
                    generation: 2,
                    retry_count: 3,
                    last_error: Some(newer_cause.clone()),
                },
                payload: VoxelBuffer::with_size(Vector3i::splat(2)),
            }]),
        },
    );
    let stale = BlockDataOutput::saved_dropped(Vector3i::zero(), 0, None, true, 1);
    core.apply_save_response(save_result(
        stale,
        Some(crate::streams::VoxelStreamError::Io(
            "stale generation failure".into(),
        )),
        None,
    ));

    let entry = core.save_journal.get(&key).expect("newer save must remain");
    let Some(ActiveSaveAttempt::Pending(old)) = &entry.active else {
        panic!("old acknowledged error must restore the old active payload");
    };
    assert_eq!(old.meta.generation, 1);
    assert!(matches!(old.meta.last_error, Some(VoxelStreamError::Io(_))));
    assert_eq!(entry.queued_newer[0].meta.generation, 2);
    assert_eq!(entry.queued_newer[0].meta.last_error, Some(newer_cause));
    assert_eq!(entry.queued_newer[0].meta.retry_count, 3);
}

#[test]
fn save_journal_keeps_same_position_separate_across_lods() {
    let mut core = build_core();
    let position = Vector3i::new(2, 3, 4);

    for lod_index in [0, 1] {
        core.enqueue_data_save(BlockToSave {
            voxels: Some(VoxelBuffer::with_size(Vector3i::splat(2))),
            position,
            lod_index,
            block_revision: 0,
        });
    }

    assert_eq!(core.save_journal.len(), 2);
    assert!(core.save_journal.contains_key(&SaveKey::new(position, 0)));
    assert!(core.save_journal.contains_key(&SaveKey::new(position, 1)));
}

#[test]
fn older_in_flight_completion_dispatches_queued_newer_save() {
    let mut core = build_core();
    let key = SaveKey::new(Vector3i::zero(), 0);
    core.save_journal.insert(
        key,
        SaveJournalEntry {
            written_unflushed: None,
            active: Some(ActiveSaveAttempt::WriteInFlight {
                meta: SaveAttemptMeta {
                    block_revision: 0,
                    generation: 1,
                    retry_count: 0,
                    last_error: None,
                },
                attempt_ordinal: 0,
            }),
            queued_newer: VecDeque::from([PendingSave {
                meta: SaveAttemptMeta {
                    block_revision: 0,
                    generation: 2,
                    retry_count: 0,
                    last_error: None,
                },
                payload: VoxelBuffer::with_size(Vector3i::splat(2)),
            }]),
        },
    );

    core.apply_save_response(save_result(
        BlockDataOutput::saved(Vector3i::zero(), 0, true, 1),
        None,
        Some(VoxelBuffer::with_size(Vector3i::splat(2))),
    ));

    let entry = core
        .save_journal
        .get(&key)
        .expect("newer save must remain behind the written predecessor");
    assert!(entry.active.is_none());
    assert_eq!(entry.written_unflushed.as_ref().unwrap().generation, 1);
    assert_eq!(entry.queued_newer[0].meta.generation, 2);
}

#[test]
fn flushed_old_acknowledgement_does_not_remove_newer_generation() {
    let mut core = build_core();
    let key = SaveKey::new(Vector3i::zero(), 0);
    core.save_journal.insert(
        key,
        SaveJournalEntry {
            written_unflushed: Some(WrittenSave {
                block_revision: 0,
                generation: 1,
                payload: VoxelBuffer::with_size(Vector3i::splat(2)),
            }),
            active: None,
            queued_newer: VecDeque::from([PendingSave {
                meta: SaveAttemptMeta {
                    block_revision: 0,
                    generation: 2,
                    retry_count: 0,
                    last_error: None,
                },
                payload: VoxelBuffer::with_size(Vector3i::splat(2)),
            }]),
        },
    );

    core.flush_acknowledged_saves().unwrap();

    let entry = core
        .save_journal
        .get(&key)
        .expect("newer generation must remain journaled");
    assert!(entry.written_unflushed.is_none());
    assert!(matches!(
        entry.active,
        Some(ActiveSaveAttempt::Pending(ref pending)) if pending.meta.generation == 2
    ));
}

#[test]
fn automatic_checkpoint_waits_for_delayed_save_result() {
    let delayed_position = Vector3i::new(99, 20, 0);
    let stream = Arc::new(CoordinatedBufferedStream::with_blocked_save(
        delayed_position,
    ));
    let mut core = build_core_with_stream(stream.clone());

    let mut delayed_voxels = VoxelBuffer::with_size(Vector3i::splat(2));
    delayed_voxels.set_voxel(91, 1, 1, 1, ChannelId::Type.index());
    core.enqueue_data_save(BlockToSave {
        voxels: Some(delayed_voxels),
        position: delayed_position,
        lod_index: 0,
        block_revision: 0,
    });
    core.try_process(&[]).unwrap();
    stream.wait_for_save_started();
    stage_acknowledged_checkpoint_batch(&mut core, stream.as_ref(), 20);

    core.try_process(&[]).unwrap();
    let flushes_while_save_was_in_flight = stream.flush_attempts();

    stream.release_save();
    core.wait_for_pending_tasks();
    core.try_process(&[]).unwrap();
    core.wait_for_pending_tasks();
    core.try_process(&[]).unwrap();

    assert_eq!(
        flushes_while_save_was_in_flight, 0,
        "checkpoint must wait until every save result is applied"
    );
    assert_eq!(stream.flush_attempts(), 1);
}

#[test]
fn checkpoint_gate_holds_new_save_through_failed_flush_and_explicit_recovery() {
    let stream = Arc::new(CoordinatedBufferedStream::with_blocked_flush(true));
    let mut core = build_core_with_stream(stream.clone());
    let mut expected = stage_acknowledged_checkpoint_batch(&mut core, stream.as_ref(), 21);
    let saves_before_checkpoint = stream.save_attempts();

    let (cancel_watchdog, watchdog_rx) = mpsc::channel();
    let watchdog_stream = stream.clone();
    let watchdog = thread::spawn(move || {
        if watchdog_rx
            .recv_timeout(Duration::from_millis(250))
            .is_err()
        {
            watchdog_stream.release_flush();
        }
    });

    core.try_process(&[]).unwrap();
    stream.wait_for_flush_started();

    let new_position = Vector3i::new(99, 21, 0);
    let mut new_voxels = VoxelBuffer::with_size(Vector3i::splat(2));
    new_voxels.set_voxel(77, 1, 1, 1, ChannelId::Type.index());
    core.enqueue_data_save(BlockToSave {
        voxels: Some(new_voxels),
        position: new_position,
        lod_index: 0,
        block_revision: 0,
    });
    expected.push((new_position, 77));
    let new_save_started_before_checkpoint_applied =
        stream.wait_for_save_attempts_at_least(saves_before_checkpoint + 1);

    stream.release_flush();
    let _ = cancel_watchdog.send(());
    watchdog.join().unwrap();
    core.wait_for_pending_tasks();
    core.try_process(&[]).unwrap();

    let attempts_after_checkpoint_failure = stream.save_attempts();
    for _ in 0..20 {
        core.try_process(&[]).unwrap();
    }
    assert_eq!(stream.flush_attempts(), 1);
    assert_eq!(stream.save_attempts(), attempts_after_checkpoint_failure);
    assert!(
        !new_save_started_before_checkpoint_applied,
        "new saves must remain queued behind the checkpoint gate"
    );

    core.flush_pending_saves().unwrap();

    assert!(
        stream.flush_attempts() >= 2,
        "explicit recovery must flush after the failed automatic checkpoint, got {}",
        stream.flush_attempts()
    );
    assert!(core.save_journal.is_empty());
    for (position, marker) in expected {
        let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
        assert_eq!(
            stream.load_block(position, 0, &mut loaded),
            LoadResult::Found
        );
        assert_eq!(loaded.get_voxel(1, 1, 1, ChannelId::Type.index()), marker);
    }
}

#[test]
fn automatic_checkpoint_does_not_block_process_thread() {
    let stream = Arc::new(CoordinatedBufferedStream::with_blocked_flush(false));
    let mut core = build_core_with_stream(stream.clone());
    stage_acknowledged_checkpoint_batch(&mut core, stream.as_ref(), 22);

    let (cancel_watchdog, watchdog_rx) = mpsc::channel();
    let watchdog_stream = stream.clone();
    let watchdog = thread::spawn(move || {
        if watchdog_rx
            .recv_timeout(Duration::from_millis(250))
            .is_err()
        {
            watchdog_stream.release_flush();
        }
    });

    let started = Instant::now();
    core.try_process(&[]).unwrap();
    let process_elapsed = started.elapsed();

    stream.wait_for_flush_started();
    stream.release_flush();
    let _ = cancel_watchdog.send(());
    watchdog.join().unwrap();
    core.wait_for_pending_tasks();
    core.try_process(&[]).unwrap();

    assert!(
        process_elapsed < Duration::from_millis(100),
        "process blocked on stream.flush for {process_elapsed:?}"
    );
    assert_eq!(stream.flush_attempts(), 1);
}

#[test]
fn automatic_checkpoint_coalesces_and_failure_rate_limits() {
    let stream = Arc::new(CoordinatedBufferedStream::with_blocked_flush(true));
    let mut core = build_core_with_stream(stream.clone());
    stage_acknowledged_checkpoint_batch(&mut core, stream.as_ref(), 23);

    let (cancel_watchdog, watchdog_rx) = mpsc::channel();
    let watchdog_stream = stream.clone();
    let watchdog = thread::spawn(move || {
        if watchdog_rx
            .recv_timeout(Duration::from_millis(250))
            .is_err()
        {
            watchdog_stream.release_flush();
        }
    });

    core.checkpoint_acknowledged_saves_if_needed();
    stream.wait_for_flush_started();
    for _ in 0..20 {
        core.checkpoint_acknowledged_saves_if_needed();
    }
    assert!(core.save_checkpoint_in_flight.is_some());
    assert_eq!(stream.flush_attempts(), 1);

    stream.release_flush();
    let _ = cancel_watchdog.send(());
    watchdog.join().unwrap();
    core.wait_for_pending_tasks();
    core.drain_completed_tasks().unwrap();

    assert!(core.save_checkpoint_in_flight.is_none());
    assert!(core.automatic_save_checkpoint_blocked);
    for _ in 0..20 {
        core.checkpoint_acknowledged_saves_if_needed();
    }
    assert_eq!(stream.flush_attempts(), 1);
}

#[test]
fn variable_planner_try_process_starts_automatic_checkpoint_without_shutdown() {
    let stream = Arc::new(DiscardingBufferedStream::healthy());
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-512), Vector3i::splat(1024)));
    data.set_generator(Some(Arc::new(Flat {
        channel: ChannelId::Sdf,
        ..Flat::default()
    })));
    let settings = LodClipboxSettings {
        data_block_size: 16,
        mesh_block_size: 16,
        lod_count: 3,
        lod0_distance_voxels: 16,
        secondary_distance_voxels: 16,
        unload_hysteresis_blocks: 2,
    };
    let mut core = VoxelTerrainCore::new_variable_lod(
        data,
        stream.clone(),
        MeshingDependency::new(Arc::new(crate::meshers::TransvoxelMesher::new()), None),
        settings,
    )
    .expect("variable LOD terrain constructs");
    assert!(
        core.variable_use_planner_path,
        "this regression pins the production planner path"
    );
    stage_acknowledged_checkpoint_batch(&mut core, stream.as_ref(), 7);
    assert!(core.save_checkpoint_in_flight.is_none());
    assert!(!core.shutdown_in_progress);

    core.try_process(&[ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: 48,
        vertical_view_distance_voxels: 48,
        demand: MeshDemand {
            visuals: true,
            collisions: false,
        },
    }])
    .expect("stationary planner tick must succeed");

    assert!(
        core.save_checkpoint_in_flight.is_some(),
        "production Variable-LOD try_process must start an automatic checkpoint without shutdown"
    );
}

#[test]
fn automatic_checkpoint_of_old_acknowledgement_keeps_newer_generation() {
    let stream = Arc::new(DiscardingBufferedStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    let channel = ChannelId::Type.index();
    let target = Vector3i::new(30, 0, 0);
    let target_key = SaveKey::new(target, 0);

    for index in 0..AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD {
        let position = if index == 0 {
            target
        } else {
            Vector3i::new(30 + index as i32, 0, 0)
        };
        let mut acknowledged = VoxelBuffer::with_size(Vector3i::splat(2));
        acknowledged.set_voxel((index + 1) as u64, 1, 1, 1, channel);
        stream
            .save_voxel_block(crate::streams::VoxelSaveQuery::new(
                &acknowledged,
                position,
                0,
            ))
            .unwrap();

        let queued_newer = if index == 0 {
            let mut newer = VoxelBuffer::with_size(Vector3i::splat(2));
            newer.set_voxel(99, 1, 1, 1, channel);
            VecDeque::from([PendingSave {
                meta: SaveAttemptMeta {
                    block_revision: 0,
                    generation: 2,
                    retry_count: 0,
                    last_error: None,
                },
                payload: newer,
            }])
        } else {
            VecDeque::new()
        };
        core.save_journal.insert(
            SaveKey::new(position, 0),
            SaveJournalEntry {
                written_unflushed: Some(WrittenSave {
                    block_revision: 0,
                    generation: 1,
                    payload: acknowledged,
                }),
                active: None,
                queued_newer,
            },
        );
    }

    core.checkpoint_acknowledged_saves_if_needed();
    core.wait_for_pending_tasks();
    core.drain_completed_tasks().unwrap();

    assert_eq!(stream.flush_attempts(), 1);
    assert_eq!(core.save_journal.len(), 1);
    let entry = core
        .save_journal
        .get(&target_key)
        .expect("newer generation must survive the old checkpoint");
    assert!(entry.written_unflushed.is_none());
    assert!(matches!(
        entry.state_for_generation(2),
        Some(
            JournalPersistenceState::PendingWrite
                | JournalPersistenceState::WriteInFlight
                | JournalPersistenceState::WrittenUnflushed
        )
    ));

    core.flush_pending_saves().unwrap();
    let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(stream.load_block(target, 0, &mut loaded), LoadResult::Found);
    assert_eq!(loaded.get_voxel(1, 1, 1, channel), 99);
}

#[test]
fn shutdown_and_flush_waits_for_pending_save() {
    let stream = Arc::new(MemoryStream::new());
    let mut core = build_core_with_stream(stream.clone());
    let bs = core.data_block_size();
    let channel = ChannelId::Type.index();
    let edited_voxel = Vector3i::new(1, 1, 1);

    let viewer = vec![ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: bs,
        vertical_view_distance_voxels: bs,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];
    process_until(&mut core, &viewer, |core, _events| {
        core.data().block_snapshot(Vector3i::zero(), 0).is_some()
    });
    assert!(core
        .try_edit_voxel(99, edited_voxel, channel)
        .unwrap()
        .is_some());
    core.try_process(&[]).unwrap();

    core.shutdown_and_flush().unwrap();

    let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(Vector3i::zero(), 0, &mut loaded),
        LoadResult::Found
    );
    assert_eq!(loaded.get_voxel(1, 1, 1, channel), 99);
}

#[test]
fn shutdown_saves_resident_edit_without_empty_viewer_tick() {
    let stream = Arc::new(MemoryStream::new());
    let mut core = build_core_with_stream(stream.clone());
    let bs = core.data_block_size();
    let channel = ChannelId::Type.index();
    let viewer = [ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: bs,
        vertical_view_distance_voxels: bs,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];

    process_until(&mut core, &viewer, |core, _events| {
        core.data().block_snapshot(Vector3i::zero(), 0).is_some()
    });
    assert!(core
        .try_edit_voxel(7, Vector3i::new(1, 1, 1), channel)
        .unwrap()
        .is_some());

    core.shutdown_and_flush().unwrap();

    let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(Vector3i::zero(), 0, &mut loaded),
        LoadResult::Found
    );
    assert_eq!(loaded.get_voxel(1, 1, 1, channel), 7);
}

#[test]
fn shutdown_saves_dirty_resident_lods_without_any_paired_viewer() {
    let stream = Arc::new(MemoryStream::new());
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::zero(), Vector3i::splat(32)));
    data.set_streaming_enabled(false);
    data.set_full_load_completed(true);
    let mut core = VoxelTerrainCore::legacy_variable_lod_for_parity(
        data,
        stream.clone(),
        MeshingDependency::new(Arc::new(OneVoxelPaddingMesher), None),
        2,
    );
    let channel = ChannelId::Type.index();

    assert!(core
        .try_edit_voxel(31, Vector3i::new(2, 2, 2), channel)
        .unwrap()
        .is_some());
    assert!(core.paired_viewers.is_empty());
    core.shutdown_and_flush().unwrap();

    let mut lod0 = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(Vector3i::zero(), 0, &mut lod0),
        LoadResult::Found
    );
    assert_eq!(lod0.get_voxel(2, 2, 2, channel), 31);
    let mut lod1 = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(Vector3i::zero(), 1, &mut lod1),
        LoadResult::Found
    );
    assert_eq!(lod1.get_voxel(1, 1, 1, channel), 31);
}

#[test]
fn successful_shutdown_closes_storage_mutation_admission_for_every_lod_mode() {
    for lod_count in [1, 2] {
        let mut core = if lod_count == 1 {
            build_core_with_materializable_data(Arc::new(MemoryStream::new()))
        } else {
            make_edit_core_with_lods(lod_count)
        };
        assert!(core
            .try_edit_voxel(41, Vector3i::zero(), ChannelId::Type.index())
            .unwrap()
            .is_some());
        let data = Arc::clone(&core.data);

        core.shutdown_and_flush().unwrap();

        assert!(data.is_mutation_admission_closed());
        let revisions = (0..lod_count as usize)
            .map(|lod| data.key_revision(Vector3i::zero(), lod))
            .collect::<Vec<_>>();
        assert_eq!(
            core.try_edit_voxel(42, Vector3i::zero(), ChannelId::Type.index()),
            Err(VoxelTerrainRuntimeError::ShutdownRetryPending)
        );
        assert_eq!(
            data.try_edit_voxel_checked(43, Vector3i::zero(), ChannelId::Type.index()),
            Err(SharedVoxelDataMutationError::MutationAdmissionClosed)
        );
        assert_eq!(
            (0..lod_count as usize)
                .map(|lod| data.key_revision(Vector3i::zero(), lod))
                .collect::<Vec<_>>(),
            revisions
        );
        assert!(core.flush_pending_saves().is_ok());
    }
}

#[test]
fn shutdown_blocks_external_edit_while_captured_save_is_in_flight() {
    let stream = Arc::new(CoordinatedBufferedStream::with_blocked_save(
        Vector3i::zero(),
    ));
    let mut core = build_core_with_materializable_data(stream.clone());
    assert!(core
        .try_edit_voxel(51, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .is_some());
    let data = Arc::clone(&core.data);
    let core = Arc::new(Mutex::new(core));
    let shutdown_core = Arc::clone(&core);
    let shutdown = thread::spawn(move || {
        shutdown_core
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .shutdown_and_flush()
    });

    stream.wait_for_save_started();
    assert!(data.is_mutation_admission_closed());
    assert_eq!(
        data.try_edit_voxel_checked(52, Vector3i::zero(), ChannelId::Type.index()),
        Err(SharedVoxelDataMutationError::MutationAdmissionClosed)
    );

    stream.release_save();
    shutdown.join().unwrap().unwrap();
    assert!(
        core.lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .shut_down
    );
}

#[test]
fn admitted_writer_crosses_begin_shutdown_before_boundary_and_capture_uses_permit() {
    let stream = Arc::new(MemoryStream::new());
    let mut core = build_core_with_materializable_data(stream.clone());
    assert!(core
        .try_edit_voxel(0, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .is_some());
    let data = Arc::clone(&core.data);
    let entered = Arc::new((Mutex::new(false), Condvar::new()));
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    data.set_test_edit_phase_hook(Arc::new({
        let entered = Arc::clone(&entered);
        let release = Arc::clone(&release);
        move |phase| {
            if phase != SharedVoxelDataEditPhase::MutationGateAcquiredBeforeSpatialWrite {
                return;
            }
            let (entered_lock, entered_cv) = &*entered;
            *entered_lock.lock().unwrap() = true;
            entered_cv.notify_all();
            let (release_lock, release_cv) = &*release;
            let mut released = release_lock.lock().unwrap();
            while !*released {
                released = release_cv.wait(released).unwrap();
            }
        }
    }));

    let writer_data = Arc::clone(&data);
    let writer = thread::spawn(move || {
        writer_data.try_edit_voxel_checked(0xCA, Vector3i::zero(), ChannelId::Type.index())
    });
    let (entered_lock, entered_cv) = &*entered;
    let mut writer_entered = entered_lock.lock().unwrap();
    while !*writer_entered {
        writer_entered = entered_cv.wait(writer_entered).unwrap();
    }
    drop(writer_entered);

    let core = Arc::new(Mutex::new(core));
    let boundary_core = Arc::clone(&core);
    let boundary = thread::spawn(move || {
        boundary_core
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .begin_shutdown_attempt()
    });

    let (release_lock, release_cv) = &*release;
    *release_lock.lock().unwrap() = true;
    release_cv.notify_all();
    assert!(writer.join().unwrap().is_ok());
    boundary.join().unwrap().unwrap();

    assert!(data.is_mutation_admission_closed());
    assert_eq!(
        data.try_edit_voxel_checked(0xCB, Vector3i::zero(), ChannelId::Type.index()),
        Err(SharedVoxelDataMutationError::MutationAdmissionClosed)
    );
    core.lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .shutdown_and_flush()
        .unwrap();

    let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(Vector3i::zero(), 0, &mut loaded),
        LoadResult::Found
    );
    assert_eq!(loaded.get_voxel(0, 0, 0, ChannelId::Type.index()), 0xCA);
}

#[test]
fn failed_variable_shutdown_is_retry_only_and_retry_saves_captured_edit() {
    let stream = Arc::new(ControlledFailureStream::new(true));
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-32), Vector3i::splat(64)));
    data.set_streaming_enabled(false);
    data.set_full_load_completed(true);
    let mut core = VoxelTerrainCore::legacy_variable_lod_for_parity(
        data,
        stream.clone(),
        MeshingDependency::new(Arc::new(OneVoxelPaddingMesher), None),
        2,
    );
    let channel = ChannelId::Type.index();
    assert!(core
        .try_edit_voxel(61, Vector3i::zero(), channel)
        .unwrap()
        .is_some());

    assert_eq!(
        core.shutdown_and_flush(),
        Err(SaveFlushError::UnsavedBlocks { count: 2 })
    );
    assert!(core.data.is_mutation_admission_closed());
    assert_eq!(
        core.try_edit_voxel(62, Vector3i::zero(), channel),
        Err(VoxelTerrainRuntimeError::ShutdownRetryPending)
    );
    assert!(matches!(
        core.try_process(&[]),
        Err(VoxelTerrainRuntimeError::ShutdownRetryPending)
    ));

    stream.set_fail_saves(false);
    core.shutdown_and_flush().unwrap();

    assert!(core.data.is_mutation_admission_closed());
    for lod_index in 0..2 {
        let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
        assert_eq!(
            stream.load_block(Vector3i::zero(), lod_index, &mut loaded),
            LoadResult::Found
        );
        assert_eq!(loaded.get_voxel(0, 0, 0, channel), 61);
    }
}

#[test]
fn shutdown_reports_data_unview_failure_and_remains_retryable() {
    let stream = Arc::new(MemoryStream::new());
    let mut core = build_core_with_stream(stream.clone());
    let bs = core.data_block_size();
    let channel = ChannelId::Type.index();
    let edited_voxel = Vector3i::new(1, 1, 1);
    let viewer = [ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: bs,
        vertical_view_distance_voxels: bs,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];

    process_until(&mut core, &viewer, |core, _events| {
        core.data().block_snapshot(Vector3i::zero(), 0).is_some()
    });
    assert!(core
        .try_edit_voxel(83, edited_voxel, channel)
        .unwrap()
        .is_some());
    let paired_before = core.paired_viewers.clone();
    core.data.with_lod_map_mut(0, |map| {
        map.set_key_revision_for_test(Vector3i::zero(), u64::MAX);
    });

    assert_eq!(
        core.shutdown_and_flush(),
        Err(SaveFlushError::DataMutation(
            SharedVoxelDataMutationError::KeyRevisionOverflow {
                position: Vector3i::zero(),
                lod_index: 0,
            }
        ))
    );
    assert!(!core.shut_down);
    assert!(!core.shutdown_in_progress);
    assert!(core.data_unview_retries[0].is_empty());
    assert_eq!(core.paired_viewers, paired_before);
    assert!(core
        .data
        .block_snapshot(Vector3i::zero(), 0)
        .is_some_and(|block| block.is_modified()));
    assert_eq!(
        core.data()
            .block_snapshot(Vector3i::zero(), 0)
            .unwrap()
            .voxels()
            .get_voxel(1, 1, 1, channel),
        83
    );

    core.data.with_lod_map_mut(0, |map| {
        map.set_key_revision_for_test(Vector3i::zero(), 1);
    });
    core.shutdown_and_flush().unwrap();

    let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(Vector3i::zero(), 0, &mut loaded),
        LoadResult::Found
    );
    assert_eq!(loaded.get_voxel(1, 1, 1, channel), 83);
}

#[test]
fn flush_pending_saves_persists_resident_edit_and_keeps_terrain_active() {
    let stream = Arc::new(MemoryStream::new());
    let mut core = build_core_with_stream(stream.clone());
    let bs = core.data_block_size();
    let channel = ChannelId::Type.index();
    let position = Vector3i::new(1, 1, 1);
    let viewer = [ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: bs,
        vertical_view_distance_voxels: bs,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];

    process_until(&mut core, &viewer, |core, _events| {
        core.data().block_snapshot(Vector3i::zero(), 0).is_some()
    });
    assert!(core
        .try_edit_voxel(61, position, channel)
        .unwrap()
        .is_some());

    core.flush_pending_saves().unwrap();

    assert!(!core.shut_down);
    assert!(!core.shutdown_in_progress);
    assert_eq!(core.paired_viewers.len(), 1);
    assert!(core.data().block_snapshot(Vector3i::zero(), 0).is_some());
    let events = core.try_process(&viewer).unwrap();
    assert!(!events
        .iter()
        .any(|event| matches!(event, VoxelTerrainEvent::DataBlockUnloaded(_))));
    let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(Vector3i::zero(), 0, &mut loaded),
        LoadResult::Found
    );
    assert_eq!(loaded.get_voxel(1, 1, 1, channel), 61);
}

#[test]
fn resident_dirty_snapshot_and_clear_commit_with_one_exact_block_revision() {
    let mut core = build_core_with_materializable_data(Arc::new(MemoryStream::new()));
    let location = BlockLocation {
        position: Vector3i::zero(),
        lod_index: 0,
    };
    assert!(core
        .try_edit_voxel(0xD1, Vector3i::new(1, 1, 1), ChannelId::Type.index())
        .unwrap()
        .is_some());
    let resident_ptr = core.data.with_lod_map(0, |map| {
        voxel_allocation_identity(map.get_block(location.position).unwrap().voxels())
    });
    assert_eq!(
        core.data.key_revision(location.position, 0),
        Some(VoxelDataKeyRevision::Present(1))
    );

    core.prepare_fixed_viewer_transaction_with_checkpoint(
        &[],
        false,
        false,
        false,
        None,
        None,
        None,
        true,
    )
    .unwrap();

    let entry = core
        .save_journal
        .get(&SaveKey::new(location.position, location.lod_index))
        .unwrap();
    let Some(ActiveSaveAttempt::Pending(pending)) = &entry.active else {
        panic!("resident save is admitted as one pending journal owner")
    };
    assert_eq!(pending.meta.block_revision, 1);
    assert_eq!(pending.meta.generation, 1);
    assert_ne!(voxel_allocation_identity(&pending.payload), resident_ptr);
    assert_eq!(
        core.data.key_revision(location.position, 0),
        Some(VoxelDataKeyRevision::Present(2))
    );
    core.data.with_lod_map(0, |map| {
        let resident = map.get_block(location.position).unwrap();
        assert!(!resident.is_modified());
        assert_eq!(voxel_allocation_identity(resident.voxels()), resident_ptr);
    });
}

#[test]
fn resident_capture_composes_dirty_clear_with_nonzero_final_viewer_count() {
    let mut core = build_core_with_materializable_data(Arc::new(MemoryStream::new()));
    let location = BlockLocation {
        position: Vector3i::zero(),
        lod_index: 0,
    };
    assert!(core
        .try_edit_voxel(0xDA, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .is_some());
    core.data.with_lod_map_mut(0, |map| {
        map.get_block_mut(location.position)
            .unwrap()
            .viewers
            .set_exact(2);
    });
    core.loaded_data_residency[0].insert(
        location.position,
        DataResidencyRefs::with_resident_viewers(2),
    );
    let state = ViewerState {
        data_box: Box3i::new(location.position, Vector3i::splat(1)),
        ..ViewerState::default()
    };
    core.paired_viewers.push(PairedViewer {
        id: 91,
        state,
        prev_state: ViewerState::default(),
    });

    core.prepare_fixed_viewer_transaction_with_checkpoint(
        &[],
        true,
        false,
        false,
        None,
        None,
        None,
        true,
    )
    .unwrap();

    core.data.with_lod_map(0, |map| {
        let resident = map.get_block(location.position).unwrap();
        assert_eq!(resident.viewers.get(), 1);
        assert!(!resident.is_modified());
    });
    assert_eq!(
        core.data.key_revision(location.position, 0),
        Some(VoxelDataKeyRevision::Present(2))
    );
    let entry = &core.save_journal[&SaveKey::new(location.position, 0)];
    let Some(ActiveSaveAttempt::Pending(pending)) = &entry.active else {
        panic!("the composed transaction publishes one save owner")
    };
    assert_eq!(
        (pending.meta.block_revision, pending.meta.generation),
        (1, 1)
    );
}

#[test]
fn resident_save_owner_is_published_before_storage_dirty_clear_becomes_visible() {
    let mut core = build_core_with_materializable_data(Arc::new(MemoryStream::new()));
    let location = BlockLocation {
        position: Vector3i::zero(),
        lod_index: 0,
    };
    assert!(core
        .try_edit_voxel(0xD7, Vector3i::new(1, 1, 1), ChannelId::Type.index())
        .unwrap()
        .is_some());
    let resident_ptr = core.data.with_lod_map(0, |map| {
        voxel_allocation_identity(map.get_block(location.position).unwrap().voxels())
    });
    let data = Arc::clone(&core.data);
    let owner = core.install_fixed_dirty_owner_probe_for_test(location);
    let pause = core.install_fixed_commit_pause_for_test(
        FixedCommitPausePhase::AfterTerrainPublishBeforeFenceFinish,
    );
    let marker = pause.commit_marker();
    let core = Arc::new(Mutex::new(core));
    let transaction_core = Arc::clone(&core);
    let transaction = thread::spawn(move || {
        transaction_core
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .prepare_fixed_viewer_transaction_with_checkpoint(
                &[],
                false,
                false,
                false,
                None,
                None,
                None,
                true,
            )
    });

    pause.wait_until_reached();
    assert!(marker.load(Ordering::SeqCst));
    assert!(
        data.try_lod_map_read(0).is_none(),
        "storage readers must remain fenced until the journal owner exists"
    );
    let owner_ptr = owner.load(Ordering::SeqCst);
    assert_ne!(
        owner_ptr, 0,
        "journal publication must expose one payload owner"
    );
    assert_ne!(
        owner_ptr, resident_ptr,
        "the persisted snapshot must not alias the live resident buffer"
    );

    pause.release();
    transaction
        .join()
        .expect("resident capture thread does not panic")
        .expect("resident capture commits");
    let core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let operation = PersistenceOperation::Save {
        location,
        block_revision: 1,
        save_generation: 1,
    };
    assert_eq!(
        core.journal_payload_ptr_for_test(operation),
        Some(owner_ptr as *const u8)
    );
    assert_eq!(
        core.journal_persistence_state_for_test(operation),
        Some(JournalPersistenceState::PendingWrite)
    );
    assert!(!core
        .data
        .block_snapshot(location.position, location.lod_index as usize)
        .unwrap()
        .is_modified());
}

#[test]
fn variable_lod_resident_capture_publishes_one_exact_owner_per_lod() {
    let mut core = make_edit_core_with_lods(2);
    assert!(core
        .try_edit_voxel(0xD8, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .is_some());
    for lod_index in 0..2 {
        let resident = core
            .data
            .block_snapshot(Vector3i::zero(), lod_index)
            .expect("the edit materializes every configured LOD");
        assert!(resident.is_modified());
        assert_eq!(
            core.data.key_revision(Vector3i::zero(), lod_index),
            Some(VoxelDataKeyRevision::Present(1))
        );
    }

    core.try_capture_variable_resident_saves(false).unwrap();

    assert_eq!(core.save_journal.len(), 2);
    let mut identities = Vec::new();
    for lod_index in 0..2u8 {
        let entry = &core.save_journal[&SaveKey::new(Vector3i::zero(), lod_index)];
        let Some(ActiveSaveAttempt::Pending(pending)) = &entry.active else {
            panic!("each variable-LOD snapshot has one pending journal owner")
        };
        identities.push((pending.meta.block_revision, pending.meta.generation));
        let resident = core
            .data
            .block_snapshot(Vector3i::zero(), lod_index as usize)
            .unwrap();
        assert!(!resident.is_modified());
        assert_eq!(
            core.data.key_revision(Vector3i::zero(), lod_index as usize),
            Some(VoxelDataKeyRevision::Present(2))
        );
    }
    identities.sort_unstable();
    assert_eq!(identities, vec![(1, 1), (1, 2)]);
    assert_eq!(core.next_save_generation, 3);
}

#[test]
fn variable_lod_save_owners_publish_before_dirty_clear_becomes_visible() {
    let mut core = make_edit_core_with_lods(2);
    assert!(core
        .try_edit_voxel(0xDB, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .is_some());
    let location = BlockLocation {
        position: Vector3i::zero(),
        lod_index: 1,
    };
    let resident_ptr = core.data.with_lod_map(1, |map| {
        voxel_allocation_identity(map.get_block(location.position).unwrap().voxels())
    });
    let data = Arc::clone(&core.data);
    let owner = core.install_fixed_dirty_owner_probe_for_test(location);
    let pause = core.install_fixed_commit_pause_for_test(
        FixedCommitPausePhase::AfterTerrainPublishBeforeFenceFinish,
    );
    let marker = pause.commit_marker();
    let core = Arc::new(Mutex::new(core));
    let capture_core = Arc::clone(&core);
    let capture = thread::spawn(move || {
        capture_core
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .try_capture_variable_resident_saves(false)
    });

    pause.wait_until_reached();
    assert!(marker.load(Ordering::SeqCst));
    for lod_index in 0..2 {
        assert!(
            data.try_lod_map_read(lod_index).is_none(),
            "LOD {lod_index} must remain fenced until every journal owner exists"
        );
    }
    let owner_ptr = owner.load(Ordering::SeqCst);
    assert_ne!(owner_ptr, 0);
    assert_ne!(owner_ptr, resident_ptr);

    pause.release();
    capture.join().unwrap().unwrap();
    let core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let operation = PersistenceOperation::Save {
        location,
        block_revision: 1,
        save_generation: 2,
    };
    assert_eq!(
        core.journal_payload_ptr_for_test(operation),
        Some(owner_ptr as *const u8)
    );
    assert_eq!(
        core.journal_persistence_state_for_test(operation),
        Some(JournalPersistenceState::PendingWrite)
    );
}

#[test]
fn variable_resident_capture_journal_capacity_failures_consume_nothing() {
    let mut vacant = make_edit_core_with_lods(2);
    assert!(vacant
        .try_edit_voxel(0xDD, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .is_some());
    vacant.fail_fixed_capacity_for_test(FixedCapacityDestination::SaveJournal, 1);
    assert_eq!(
        vacant.try_capture_variable_resident_saves(false),
        Err(VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)
    );
    assert!(vacant.save_journal.is_empty());
    assert_eq!(vacant.next_save_generation, 1);
    for lod_index in 0..2 {
        assert!(vacant
            .data
            .block_snapshot(Vector3i::zero(), lod_index)
            .unwrap()
            .is_modified());
    }

    let mut queued = make_edit_core_with_lods(2);
    assert!(queued
        .try_edit_voxel(0xDE, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .is_some());
    queued.try_capture_variable_resident_saves(false).unwrap();
    assert!(queued
        .try_edit_voxel(0xDF, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .is_some());
    queued.fail_fixed_capacity_for_test(FixedCapacityDestination::SaveJournalQueue, 1);
    assert_eq!(
        queued.try_capture_variable_resident_saves(false),
        Err(VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)
    );
    assert_eq!(queued.next_save_generation, 3);
    for lod_index in 0..2u8 {
        let entry = &queued.save_journal[&SaveKey::new(Vector3i::zero(), lod_index)];
        assert!(entry.queued_newer.is_empty());
        assert!(queued
            .data
            .block_snapshot(Vector3i::zero(), lod_index as usize)
            .unwrap()
            .is_modified());
    }
}

#[test]
fn variable_resident_capture_late_conflict_preserves_newer_edit_and_journal_head() {
    let mut core = make_edit_core_with_lods(2);
    assert!(core
        .try_edit_voxel(0xE1, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .is_some());
    core.variable_after_prepare_edit_conflict_for_test = Some(Vector3i::zero());

    assert!(matches!(
        core.try_capture_variable_resident_saves(false),
        Err(VoxelTerrainRuntimeError::DataMutation(
            SharedVoxelDataMutationError::ConcurrentDataMutation { .. }
        ))
    ));

    assert!(core.save_journal.is_empty());
    assert_eq!(core.next_save_generation, 1);
    let resident = core.data.block_snapshot(Vector3i::zero(), 0).unwrap();
    assert!(resident.is_modified());
    assert_eq!(
        resident
            .voxels()
            .get_voxel(0, 0, 0, ChannelId::Type.index()),
        0xDC
    );
    assert_eq!(
        core.data.key_revision(Vector3i::zero(), 0),
        Some(VoxelDataKeyRevision::Present(2))
    );
}

#[test]
fn variable_shutdown_generation_overflow_preserves_every_resident_dirty_owner() {
    let mut core = make_edit_core_with_lods(2);
    assert!(core
        .try_edit_voxel(0xD9, Vector3i::zero(), ChannelId::Type.index())
        .unwrap()
        .is_some());
    core.next_save_generation = u64::MAX;

    assert_eq!(
        core.shutdown_and_flush(),
        Err(SaveFlushError::SaveAdmission {
            error: VoxelTerrainRuntimeError::SaveGenerationOverflow,
        })
    );

    assert!(!core.shut_down);
    assert!(!core.shutdown_in_progress);
    assert!(core.data.is_mutation_admission_closed());
    assert_eq!(core.shutdown_epoch, Some(1));
    assert_eq!(
        core.try_edit_voxel(0xDA, Vector3i::zero(), ChannelId::Type.index()),
        Err(VoxelTerrainRuntimeError::ShutdownRetryPending)
    );
    assert_eq!(
        core.data
            .try_edit_voxel_checked(0xDA, Vector3i::zero(), ChannelId::Type.index(),),
        Err(SharedVoxelDataMutationError::MutationAdmissionClosed)
    );
    assert!(core.save_journal.is_empty());
    assert_eq!(core.next_save_generation, u64::MAX);
    for lod_index in 0..2 {
        let resident = core
            .data
            .block_snapshot(Vector3i::zero(), lod_index)
            .unwrap();
        assert!(resident.is_modified());
        assert_eq!(
            core.data.key_revision(Vector3i::zero(), lod_index),
            Some(VoxelDataKeyRevision::Present(1))
        );
    }
}

#[test]
fn variable_shutdown_missing_payload_preserves_resident_and_consumes_no_generation() {
    let mut core = make_edit_core_with_lods(2);
    let location = BlockLocation {
        position: Vector3i::new(3, 0, 0),
        lod_index: 1,
    };
    let mut dirty = VoxelDataBlock::empty(location.lod_index);
    dirty.set_modified(true);
    assert!(core.data.try_set_block(location.position, dirty).unwrap());

    assert_eq!(
        core.shutdown_and_flush(),
        Err(SaveFlushError::DataMutation(
            SharedVoxelDataMutationError::DirtyBlockMissingVoxels { location }
        ))
    );

    assert!(!core.shut_down);
    assert!(!core.shutdown_in_progress);
    assert!(core.data.is_mutation_admission_closed());
    assert_eq!(core.shutdown_epoch, Some(1));
    assert_eq!(
        core.try_edit_voxel(0xDB, Vector3i::zero(), ChannelId::Type.index()),
        Err(VoxelTerrainRuntimeError::ShutdownRetryPending)
    );
    assert_eq!(
        core.data
            .try_set_block(Vector3i::new(4, 0, 0), VoxelDataBlock::empty(1)),
        Err(SharedVoxelDataMutationError::MutationAdmissionClosed)
    );
    assert!(core.save_journal.is_empty());
    assert!(core.retained_save_admission_failures.is_empty());
    assert_eq!(core.next_save_generation, 1);
    let resident = core
        .data
        .block_snapshot(location.position, location.lod_index as usize)
        .unwrap();
    assert!(resident.is_modified());
    assert!(!resident.has_voxels());
    assert_eq!(
        core.data
            .key_revision(location.position, location.lod_index as usize),
        Some(VoxelDataKeyRevision::Present(1))
    );
}

#[test]
fn resident_edit_after_snapshot_commit_creates_a_superseding_generation() {
    let mut core = build_core_with_materializable_data(Arc::new(MemoryStream::new()));
    let location = BlockLocation {
        position: Vector3i::zero(),
        lod_index: 0,
    };
    let voxel = Vector3i::new(1, 1, 1);
    assert!(core
        .try_edit_voxel(0xD2, voxel, ChannelId::Type.index())
        .unwrap()
        .is_some());
    core.prepare_fixed_viewer_transaction_with_checkpoint(
        &[],
        false,
        false,
        false,
        None,
        None,
        None,
        true,
    )
    .unwrap();
    assert!(core
        .try_edit_voxel(0xD3, voxel, ChannelId::Type.index())
        .unwrap()
        .is_some());
    core.prepare_fixed_viewer_transaction_with_checkpoint(
        &[],
        false,
        false,
        false,
        None,
        None,
        None,
        true,
    )
    .unwrap();

    let entry = &core.save_journal[&SaveKey::new(location.position, location.lod_index)];
    let Some(ActiveSaveAttempt::Pending(first)) = &entry.active else {
        panic!("first resident snapshot remains the active owner")
    };
    assert_eq!((first.meta.block_revision, first.meta.generation), (1, 1));
    assert_eq!(entry.queued_newer.len(), 1);
    assert_eq!(
        (
            entry.queued_newer[0].meta.block_revision,
            entry.queued_newer[0].meta.generation,
        ),
        (3, 2)
    );
    assert_eq!(
        entry.queued_newer[0]
            .payload
            .get_voxel(1, 1, 1, ChannelId::Type.index()),
        0xD3
    );
    assert_eq!(
        core.data.key_revision(location.position, 0),
        Some(VoxelDataKeyRevision::Present(4))
    );
    assert!(!core
        .data
        .block_snapshot(location.position, 0)
        .unwrap()
        .is_modified());
}

#[test]
fn old_save_and_flush_ack_never_clear_or_remove_newer_resident_dirty() {
    let stream = Arc::new(MemoryStream::new());
    let mut core = build_core_with_materializable_data(stream);
    let location = BlockLocation {
        position: Vector3i::zero(),
        lod_index: 0,
    };
    let voxel = Vector3i::new(1, 1, 1);
    assert!(core
        .try_edit_voxel(0xD4, voxel, ChannelId::Type.index())
        .unwrap()
        .is_some());
    core.prepare_fixed_viewer_transaction_with_checkpoint(
        &[],
        false,
        false,
        false,
        None,
        None,
        None,
        true,
    )
    .unwrap();
    core.dispatch_queued_save(SaveKey::new(location.position, location.lod_index));
    core.wait_for_pending_tasks();

    assert!(core
        .try_edit_voxel(0xD5, voxel, ChannelId::Type.index())
        .unwrap()
        .is_some());
    assert_eq!(
        core.data.key_revision(location.position, 0),
        Some(VoxelDataKeyRevision::Present(3))
    );
    core.try_drain_completed_tasks().unwrap();
    core.flush_acknowledged_saves().unwrap();

    let resident = core.data.block_snapshot(location.position, 0).unwrap();
    assert!(resident.is_modified());
    assert_eq!(
        resident
            .voxels()
            .get_voxel(1, 1, 1, ChannelId::Type.index()),
        0xD5
    );
    assert_eq!(
        core.data.key_revision(location.position, 0),
        Some(VoxelDataKeyRevision::Present(3))
    );
}

#[test]
fn resident_snapshot_capacity_failure_preserves_dirty_and_journal_identity() {
    let mut core = build_core_with_materializable_data(Arc::new(MemoryStream::new()));
    let location = BlockLocation {
        position: Vector3i::zero(),
        lod_index: 0,
    };
    assert!(core
        .try_edit_voxel(0xD6, Vector3i::new(1, 1, 1), ChannelId::Type.index())
        .unwrap()
        .is_some());
    let resident_ptr = core.data.with_lod_map(0, |map| {
        voxel_allocation_identity(map.get_block(location.position).unwrap().voxels())
    });
    core.fail_fixed_capacity_for_test(FixedCapacityDestination::SaveJournal, 1);

    assert_eq!(
        core.prepare_fixed_viewer_transaction_with_checkpoint(
            &[],
            false,
            false,
            false,
            None,
            None,
            None,
            true,
        ),
        Err(VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)
    );
    assert!(core.save_journal.is_empty());
    assert_eq!(core.next_save_generation, 1);
    core.data.with_lod_map(0, |map| {
        let resident = map.get_block(location.position).unwrap();
        assert!(resident.is_modified());
        assert_eq!(voxel_allocation_identity(resident.voxels()), resident_ptr);
    });
    assert_eq!(
        core.data.key_revision(location.position, 0),
        Some(VoxelDataKeyRevision::Present(1))
    );
}

#[test]
fn resident_snapshot_middle_prefix_failure_consumes_zero() {
    let mut core = build_core_with_materializable_data(Arc::new(MemoryStream::new()));
    let positions = [
        Vector3i::new(1, 1, 1),
        Vector3i::new(17, 1, 1),
        Vector3i::new(33, 1, 1),
    ];
    for (index, position) in positions.into_iter().enumerate() {
        assert!(core
            .try_edit_voxel(0xE0 + index as u64, position, ChannelId::Type.index())
            .unwrap()
            .is_some());
    }
    core.fail_fixed_capacity_for_test(FixedCapacityDestination::Retirement, 1);

    assert_eq!(
        core.prepare_fixed_viewer_transaction_with_checkpoint(
            &[],
            false,
            false,
            false,
            None,
            None,
            None,
            true,
        ),
        Err(VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)
    );
    assert!(core.save_journal.is_empty());
    assert_eq!(core.next_save_generation, 1);
    for block_position in [
        Vector3i::new(0, 0, 0),
        Vector3i::new(1, 0, 0),
        Vector3i::new(2, 0, 0),
    ] {
        assert!(core
            .data
            .block_snapshot(block_position, 0)
            .unwrap()
            .is_modified());
        assert_eq!(
            core.data.key_revision(block_position, 0),
            Some(VoxelDataKeyRevision::Present(1))
        );
    }
}

#[test]
fn explicit_flush_and_shutdown_propagate_drain_failure_then_retry() {
    for shutdown in [false, true] {
        let mut core = build_core();
        core.raw_completion_inbox.push_back(CompletedTask::new(
            Box::new(DebugNameCollisionTask),
            TaskLane::Parallel,
            TaskCompletionStatus::Cancelled,
            Vec::new(),
        ));
        core.fail_next_completion_normalization_for_test = true;

        let first = if shutdown {
            core.shutdown_and_flush()
        } else {
            core.flush_pending_saves()
        };
        assert!(matches!(
            first,
            Err(SaveFlushError::CompletionDrain {
                error: VoxelTerrainRuntimeError::CompletionNormalizationFailed,
            })
        ));
        assert!(!core.shutdown_in_progress);
        assert!(!core.shut_down);
        assert_eq!(core.raw_completion_inbox.len(), 1);

        let retry = if shutdown {
            core.shutdown_and_flush()
        } else {
            core.flush_pending_saves()
        };
        assert_eq!(retry, Ok(()));
        assert!(core.raw_completion_inbox.is_empty());
        assert_eq!(core.shut_down, shutdown);
    }
}

#[test]
fn explicit_save_and_checkpoint_loops_propagate_drain_failure_then_retry() {
    let mut save_core = build_core();
    save_core.enqueue_data_save(BlockToSave {
        voxels: Some(VoxelBuffer::with_size(Vector3i::splat(2))),
        position: Vector3i::zero(),
        lod_index: 0,
        block_revision: 0,
    });
    save_core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(DebugNameCollisionTask),
        TaskLane::Parallel,
        TaskCompletionStatus::Cancelled,
        Vec::new(),
    ));
    save_core.fail_next_completion_normalization_for_test = true;

    assert!(matches!(
        save_core.flush_save_journal_with_attempts(1),
        Err(SaveFlushError::CompletionDrain {
            error: VoxelTerrainRuntimeError::CompletionNormalizationFailed,
        })
    ));
    assert!(!save_core.save_journal.is_empty());
    assert_eq!(save_core.flush_save_journal_with_attempts(8), Ok(()));
    assert!(save_core.save_journal.is_empty());

    let mut checkpoint_core = build_core();
    stage_single_written_checkpoint_member(
        &mut checkpoint_core,
        BlockLocation {
            position: Vector3i::zero(),
            lod_index: 0,
        },
        1,
        VoxelBuffer::with_size(Vector3i::splat(2)),
    );
    checkpoint_core
        .raw_completion_inbox
        .push_back(CompletedTask::new(
            Box::new(DebugNameCollisionTask),
            TaskLane::Parallel,
            TaskCompletionStatus::Cancelled,
            Vec::new(),
        ));
    checkpoint_core.fail_next_completion_normalization_for_test = true;

    assert!(matches!(
        checkpoint_core.flush_acknowledged_saves(),
        Err(SaveFlushError::CompletionDrain {
            error: VoxelTerrainRuntimeError::CompletionNormalizationFailed,
        })
    ));
    assert!(checkpoint_core.save_checkpoint_in_flight.is_none());
    assert_eq!(checkpoint_core.flush_acknowledged_saves(), Ok(()));
    assert!(checkpoint_core.save_journal.is_empty());
}

#[test]
fn flush_pending_saves_reports_dirty_extraction_failure_and_remains_retryable() {
    let stream = Arc::new(MemoryStream::new());
    let mut core = build_core_with_materializable_data(stream.clone());
    let position = Vector3i::new(1, 1, 1);
    let block_position = Vector3i::zero();
    let channel = ChannelId::Type.index();
    assert!(core
        .try_edit_voxel(71, position, channel)
        .unwrap()
        .is_some());
    core.data.with_lod_map_mut(0, |map| {
        map.set_key_revision_for_test(block_position, u64::MAX);
    });

    assert_eq!(
        core.flush_pending_saves(),
        Err(SaveFlushError::DataMutation(
            SharedVoxelDataMutationError::KeyRevisionOverflow {
                position: block_position,
                lod_index: 0,
            }
        ))
    );
    assert!(!core.shutdown_in_progress);
    assert!(core.save_journal.is_empty());
    assert!(core
        .data
        .block_snapshot(block_position, 0)
        .unwrap()
        .is_modified());

    core.data.with_lod_map_mut(0, |map| {
        map.set_key_revision_for_test(block_position, 1);
    });
    core.flush_pending_saves().unwrap();
    let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(block_position, 0, &mut loaded),
        LoadResult::Found
    );
    assert_eq!(loaded.get_voxel(1, 1, 1, channel), 71);
}

#[test]
fn healthy_runtime_saves_checkpoint_in_bounded_batches() {
    const BATCH_COUNT: usize = 3;

    let stream = Arc::new(DiscardingBufferedStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    let channel = ChannelId::Type.index();
    let mut expected = Vec::new();

    for batch in 0..BATCH_COUNT {
        for index in 0..AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD {
            let position = Vector3i::new(
                (batch * AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD + index) as i32,
                0,
                0,
            );
            let marker = (batch * AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD + index + 1) as u64;
            let mut voxels = VoxelBuffer::with_size(Vector3i::splat(2));
            voxels.set_voxel(marker, 1, 1, 1, channel);
            core.enqueue_data_save(BlockToSave {
                voxels: Some(voxels),
                position,
                lod_index: 0,
                block_revision: 0,
            });
            expected.push((position, marker));
        }

        core.try_process(&[]).unwrap();
        core.wait_for_pending_tasks();
        core.try_process(&[]).unwrap();
        core.wait_for_pending_tasks();
        core.try_process(&[]).unwrap();

        let acknowledged_count = core
            .save_journal
            .values()
            .filter(|entry| entry.written_unflushed.is_some())
            .count();
        assert!(
            acknowledged_count < AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD,
            "healthy acknowledged payloads must remain bounded"
        );
    }

    assert_eq!(stream.flush_attempts(), BATCH_COUNT);
    assert!(core.save_journal.is_empty());
    for (position, marker) in expected {
        let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
        assert_eq!(
            stream.load_block(position, 0, &mut loaded),
            LoadResult::Found
        );
        assert_eq!(loaded.get_voxel(1, 1, 1, channel), marker);
    }
}

#[test]
fn explicit_flush_after_applied_checkpoint_does_not_flush_twice() {
    let stream = Arc::new(DiscardingBufferedStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    for index in 0..AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD {
        core.enqueue_data_save(BlockToSave {
            voxels: Some(VoxelBuffer::with_size(Vector3i::splat(2))),
            position: Vector3i::new(index as i32, 24, 0),
            lod_index: 0,
            block_revision: 0,
        });
    }

    core.try_process(&[]).unwrap();
    core.wait_for_pending_tasks();
    core.try_process(&[]).unwrap();
    core.wait_for_pending_tasks();
    core.try_process(&[]).unwrap();
    assert_eq!(stream.flush_attempts(), 1);
    assert!(core.save_journal.is_empty());

    core.flush_pending_saves().unwrap();

    assert_eq!(
        stream.flush_attempts(),
        1,
        "an already-applied successful checkpoint satisfies an empty explicit flush"
    );
}

#[test]
fn failed_automatic_checkpoint_waits_for_explicit_recovery() {
    let stream = Arc::new(DiscardingBufferedStream::new());
    let mut core = build_core_with_stream(stream.clone());
    let channel = ChannelId::Type.index();
    let mut expected = Vec::new();
    for index in 0..AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD {
        let position = Vector3i::new(index as i32, 1, 0);
        let marker = (index + 41) as u64;
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(2));
        voxels.set_voxel(marker, 1, 1, 1, channel);
        core.enqueue_data_save(BlockToSave {
            voxels: Some(voxels),
            position,
            lod_index: 0,
            block_revision: 0,
        });
        expected.push((position, marker));
    }

    core.try_process(&[]).unwrap();
    core.wait_for_pending_tasks();
    core.try_process(&[]).unwrap();
    core.wait_for_pending_tasks();
    core.try_process(&[]).unwrap();

    assert_eq!(stream.flush_attempts(), 1);
    assert_eq!(
        stream.save_attempts(),
        AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD
    );
    assert!(matches!(
        core.last_save_checkpoint_error(),
        Some(crate::streams::VoxelStreamError::Io(message))
            if message == "buffered flush discarded staging"
    ));
    assert!(
        core.last_save_failures().is_empty(),
        "a stream-wide checkpoint error is not a per-block save failure"
    );
    assert!(core.save_journal.values().all(|entry| {
        entry.written_unflushed.is_none()
            && matches!(entry.active, Some(ActiveSaveAttempt::Pending(_)))
    }));

    for _ in 0..20 {
        core.try_process(&[]).unwrap();
    }
    assert_eq!(stream.flush_attempts(), 1);
    assert_eq!(
        stream.save_attempts(),
        AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD
    );

    core.flush_pending_saves().unwrap();

    // Recovery must flush at least once more. Extra flushes are allowed:
    // an explicit flush can also satisfy the automatic checkpoint on the
    // same tick, and worker completion order is not deterministic.
    assert!(
        stream.flush_attempts() >= 2,
        "explicit recovery must flush after the failed automatic checkpoint, got {}",
        stream.flush_attempts()
    );
    assert!(
        stream.save_attempts() >= AUTOMATIC_SAVE_CHECKPOINT_BLOCK_THRESHOLD * 2,
        "explicit recovery must rewrite the discarded staging payload, got {}",
        stream.save_attempts()
    );
    assert_eq!(core.last_save_checkpoint_error(), None);
    assert!(core.save_journal.is_empty());
    for (position, marker) in expected {
        let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
        assert_eq!(
            stream.load_block(position, 0, &mut loaded),
            LoadResult::Found
        );
        assert_eq!(loaded.get_voxel(1, 1, 1, channel), marker);
    }
}

#[test]
fn fixed_recovery_retires_heap_error_after_commit_guards() {
    let make_recovery_core = |y: i32| {
        let mut core = build_core();
        let key = SaveKey::new(Vector3i::new(63, y, 0), 0);
        let message = format!("unique fixed recovery error {y}");
        let error_heap_ptr = message.as_ptr() as usize;
        core.save_journal.insert(
            key,
            SaveJournalEntry {
                written_unflushed: None,
                active: Some(ActiveSaveAttempt::Pending(PendingSave {
                    meta: SaveAttemptMeta {
                        block_revision: 0,
                        generation: 1,
                        retry_count: MAX_AUTOMATIC_SAVE_ATTEMPTS,
                        last_error: Some(VoxelStreamError::Io(message)),
                    },
                    payload: VoxelBuffer::with_size(Vector3i::splat(2)),
                })),
                queued_newer: VecDeque::new(),
            },
        );
        core.automatic_save_checkpoint_blocked = true;
        (core, key, error_heap_ptr)
    };
    let recovery = FixedPersistenceRecoveryRequest {
        reset_pending_save_failures: true,
        authorize_automatic_checkpoint: true,
    };
    let error_ptr = |core: &VoxelTerrainCore, key: SaveKey| {
        let Some(ActiveSaveAttempt::Pending(pending)) = &core.save_journal[&key].active else {
            panic!("recovery fixture remains pending")
        };
        let Some(VoxelStreamError::Io(message)) = &pending.meta.last_error else {
            panic!("recovery fixture retains its exact heap error")
        };
        message.as_ptr() as usize
    };

    let (mut failed, failed_key, failed_error_ptr) = make_recovery_core(-1);
    let unchanged_ordinal = failed.next_persistence_attempt_ordinal;
    failed.fail_fixed_capacity_for_test(FixedCapacityDestination::Retirement, 1);
    assert!(matches!(
        failed.prepare_fixed_viewer_transaction_with_checkpoint(
            &[],
            false,
            false,
            false,
            None,
            Some(recovery),
            None,
            false,
        ),
        Err(VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)
    ));
    assert_eq!(error_ptr(&failed, failed_key), failed_error_ptr);
    let Some(ActiveSaveAttempt::Pending(pending)) = &failed.save_journal[&failed_key].active else {
        panic!("capacity failure retains the pending recovery owner")
    };
    assert_eq!(pending.meta.retry_count, MAX_AUTOMATIC_SAVE_ATTEMPTS);
    assert!(failed.automatic_save_checkpoint_blocked);
    assert_eq!(failed.next_persistence_attempt_ordinal, unchanged_ordinal);

    failed.fixed_after_prepare_settings_conflict_for_test = true;
    assert!(matches!(
        failed.prepare_fixed_viewer_transaction_with_checkpoint(
            &[],
            false,
            false,
            false,
            None,
            Some(recovery),
            None,
            false,
        ),
        Err(VoxelTerrainRuntimeError::DataMutation(
            SharedVoxelDataMutationError::ConcurrentSettingsMutation { .. }
        ))
    ));
    assert_eq!(error_ptr(&failed, failed_key), failed_error_ptr);
    let Some(ActiveSaveAttempt::Pending(pending)) = &failed.save_journal[&failed_key].active else {
        panic!("late C1 failure retains the pending recovery owner")
    };
    assert_eq!(pending.meta.retry_count, MAX_AUTOMATIC_SAVE_ATTEMPTS);
    assert!(failed.automatic_save_checkpoint_blocked);
    assert_eq!(failed.next_persistence_attempt_ordinal, unchanged_ordinal);

    for phase in [
        FixedCommitPausePhase::AfterTerrainPublishBeforeFenceFinish,
        FixedCommitPausePhase::AfterWakeBeforeRetirementDrop,
    ] {
        let (mut core, key, error_heap_ptr) = make_recovery_core(phase as i32);
        let dropped = core.install_fixed_stream_error_retirement_probe_for_test(error_heap_ptr);
        let pause = core.install_fixed_commit_pause_for_test(phase);
        let core = Arc::new(Mutex::new(core));
        let transaction_core = Arc::clone(&core);
        let transaction = thread::spawn(move || {
            transaction_core
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .prepare_fixed_viewer_transaction_with_checkpoint(
                    &[],
                    false,
                    false,
                    false,
                    None,
                    Some(recovery),
                    None,
                    false,
                )
        });

        pause.wait_until_reached();
        assert!(
            !dropped.load(Ordering::SeqCst),
            "the heap error owner must outlive {phase:?}"
        );
        pause.release();
        transaction.join().unwrap().unwrap();
        assert!(
            dropped.load(Ordering::SeqCst),
            "the retired heap error must drop after {phase:?} releases"
        );
        let core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(ActiveSaveAttempt::Pending(pending)) = &core.save_journal[&key].active else {
            panic!("recovery without dispatch keeps the save pending")
        };
        assert_eq!(pending.meta.retry_count, 0);
        assert!(pending.meta.last_error.is_none());
    }
}

#[test]
fn flush_failure_retains_acknowledged_payload_and_resubmits_it() {
    let stream = Arc::new(DiscardingBufferedStream::new());
    let mut core = build_core_with_stream(stream.clone());
    let position = Vector3i::new(3, 2, 1);
    let channel = ChannelId::Type.index();
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(2));
    voxels.set_voxel(91, 1, 1, 1, channel);
    core.enqueue_data_save(BlockToSave {
        voxels: Some(voxels),
        position,
        lod_index: 0,
        block_revision: 0,
    });

    assert_eq!(
        core.flush_pending_saves(),
        Err(SaveFlushError::Stream(
            crate::streams::VoxelStreamError::Io("buffered flush discarded staging".into())
        ))
    );
    assert_eq!(stream.save_attempts(), 1);
    assert!(core
        .save_journal
        .get(&SaveKey::new(position, 0))
        .is_some_and(|entry| matches!(entry.active, Some(ActiveSaveAttempt::Pending(_)))));
    assert!(core.last_save_failures().iter().any(|failure| {
        failure.position_in_blocks == position
            && matches!(
                &failure.error,
                Some(crate::streams::VoxelStreamError::Io(message))
                    if message == "buffered flush discarded staging"
            )
    }));

    core.flush_pending_saves().unwrap();

    assert_eq!(stream.save_attempts(), 2);
    assert!(core.save_journal.is_empty());
    let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(position, 0, &mut loaded),
        LoadResult::Found
    );
    assert_eq!(loaded.get_voxel(1, 1, 1, channel), 91);
}

#[test]
fn automatic_save_retries_stop_at_budget_and_explicit_flush_reauthorizes() {
    let stream = Arc::new(ControlledFailureStream::new(true));
    let mut core = build_core_with_stream(stream.clone());
    let position = Vector3i::new(2, 4, 6);
    let channel = ChannelId::Type.index();
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(2));
    voxels.set_voxel(37, 1, 1, 1, channel);
    core.enqueue_data_save(BlockToSave {
        voxels: Some(voxels),
        position,
        lod_index: 0,
        block_revision: 0,
    });

    for _ in 0..20 {
        core.wait_for_pending_tasks();
        core.try_process(&[]).unwrap();
    }

    assert_eq!(stream.save_attempts(), 3);
    let entry = core
        .save_journal
        .get(&SaveKey::new(position, 0))
        .expect("exhausted save remains journaled");
    let Some(ActiveSaveAttempt::Pending(pending)) = &entry.active else {
        panic!("exhausted save must retain its pending payload");
    };
    assert_eq!(pending.meta.retry_count, 3);

    stream.set_fail_saves(false);
    core.flush_pending_saves().unwrap();

    assert_eq!(stream.save_attempts(), 4);
    let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(position, 0, &mut loaded),
        LoadResult::Found
    );
    assert_eq!(loaded.get_voxel(1, 1, 1, channel), 37);
}

#[test]
fn mixed_save_failure_reports_only_the_failed_block() {
    let failed_position = Vector3i::new(7, 0, 0);
    let saved_position = Vector3i::new(8, 0, 0);
    let stream = Arc::new(MixedOutcomeStream::new(failed_position));
    let mut core = build_core_with_stream(stream.clone());
    let channel = ChannelId::Type.index();

    for (position, marker) in [(failed_position, 51), (saved_position, 52)] {
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(2));
        voxels.set_voxel(marker, 1, 1, 1, channel);
        core.enqueue_data_save(BlockToSave {
            voxels: Some(voxels),
            position,
            lod_index: 0,
            block_revision: 0,
        });
    }

    assert_eq!(
        core.flush_pending_saves(),
        Err(SaveFlushError::UnsavedBlocks { count: 1 })
    );
    assert_eq!(stream.flush_attempts(), 1);
    assert_eq!(
        core.last_save_failure_details(),
        vec![UnsavedBlockSaveDetails {
            position_in_blocks: failed_position,
            lod_index: 0,
            block_revision: 0,
            save_generation: 1,
            retry_count: 8,
            error: Some(crate::streams::VoxelStreamError::Io(
                "permanent block save failure".into()
            )),
        }]
    );
    assert_eq!(
        core.last_save_failures(),
        vec![UnsavedBlockSave {
            position_in_blocks: failed_position,
            lod_index: 0,
            retry_count: 8,
            error: Some(crate::streams::VoxelStreamError::Io(
                "permanent block save failure".into()
            )),
        }]
    );
    assert!(!core
        .save_journal
        .contains_key(&SaveKey::new(saved_position, 0)));

    let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(saved_position, 0, &mut loaded),
        LoadResult::Found
    );
    assert_eq!(loaded.get_voxel(1, 1, 1, channel), 52);
}

#[test]
fn shutdown_is_idempotent() {
    let stream = Arc::new(FlushCountingStream::default());
    let mut core = build_core_with_stream(stream.clone());

    core.shutdown_and_flush().unwrap();
    core.shutdown_and_flush().unwrap();

    assert_eq!(stream.flush_count.load(Ordering::SeqCst), 1);
}

#[test]
fn shutdown_and_flush_reports_unsaved_blocks_after_repeated_failures() {
    let stream = Arc::new(FailThenMemoryStream::new(usize::MAX));
    let mut core = build_core_with_stream(stream);
    let key = SaveKey::new(Vector3i::zero(), 0);
    core.save_journal.insert(
        key,
        SaveJournalEntry::new_pending(0, 1, VoxelBuffer::with_size(Vector3i::splat(2))),
    );
    core.dispatch_queued_save(key);

    assert!(matches!(
        core.shutdown_and_flush(),
        Err(SaveFlushError::UnsavedBlocks { count: 1 })
    ));
}

#[test]
fn legacy_unsaved_error_shape_remains_source_compatible() {
    let error = SaveFlushError::UnsavedBlocks { count: 2 };

    let SaveFlushError::UnsavedBlocks { count } = error else {
        panic!("expected legacy unsaved-block variant");
    };
    assert_eq!(count, 2);
}

#[test]
fn failed_shutdown_reports_payload_and_remains_retryable() {
    let stream = Arc::new(ControlledFailureStream::new(true));
    let mut core = build_core_with_stream(stream.clone());
    let position = Vector3i::new(1, 2, 3);
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(2));
    voxels.set_voxel(73, 1, 1, 1, ChannelId::Type.index());
    core.enqueue_data_save(BlockToSave {
        voxels: Some(voxels),
        position,
        lod_index: 0,
        block_revision: 0,
    });

    assert_eq!(
        core.shutdown_and_flush(),
        Err(SaveFlushError::UnsavedBlocks { count: 1 })
    );
    let blocks = core.last_save_failures();
    assert_eq!(
        blocks,
        vec![UnsavedBlockSave {
            position_in_blocks: position,
            lod_index: 0,
            retry_count: 8,
            error: Some(crate::streams::VoxelStreamError::Io(
                "/tmp/terrain/region_1_0_0.vxr: permission denied".into(),
            )),
        }]
    );
    assert_eq!(
        SaveFlushError::UnsavedBlocks {
            count: blocks.len()
        }
        .to_string(),
        "1 terrain block saves remain unsaved"
    );
    assert_eq!(core.pending_task_count(), 0);
    assert!(core
        .save_journal
        .get(&SaveKey::new(position, 0))
        .is_some_and(|entry| matches!(entry.active, Some(ActiveSaveAttempt::Pending(_)))));

    stream.set_fail_saves(false);
    core.shutdown_and_flush().unwrap();
    assert!(core.data.is_mutation_admission_closed());
    let mut loaded = VoxelBuffer::new(crate::storage::Allocator::Default);
    assert_eq!(
        stream.load_block(position, 0, &mut loaded),
        LoadResult::Found
    );
    assert_eq!(loaded.get_voxel(1, 1, 1, ChannelId::Type.index()), 73);
}

#[test]
fn shutdown_error_enumerates_each_block_lod_and_distinct_cause() {
    let stream = Arc::new(ControlledFailureStream::new(true));
    let mut core = build_core_with_stream(stream);
    for (position, lod_index) in [(Vector3i::new(1, 0, 0), 2), (Vector3i::new(2, 0, 0), 4)] {
        core.enqueue_data_save(BlockToSave {
            voxels: Some(VoxelBuffer::with_size(Vector3i::splat(2))),
            position,
            lod_index,
            block_revision: 0,
        });
    }

    assert_eq!(
        core.shutdown_and_flush(),
        Err(SaveFlushError::UnsavedBlocks { count: 2 })
    );
    let blocks = core.last_save_failures();

    assert_eq!(blocks.len(), 2);
    assert!(blocks.iter().any(|block| {
        block.position_in_blocks == Vector3i::new(1, 0, 0)
            && block.lod_index == 2
            && matches!(
                &block.error,
                Some(crate::streams::VoxelStreamError::Io(message))
                    if message.contains("permission denied")
            )
    }));
    assert!(blocks.iter().any(|block| {
        block.position_in_blocks == Vector3i::new(2, 0, 0)
            && block.lod_index == 4
            && matches!(
                &block.error,
                Some(crate::streams::VoxelStreamError::CorruptData(message))
                    if message == "invalid region header"
            )
    }));
}

#[test]
fn memory_stream_restores_saved_edit_in_a_new_terrain_core() {
    let stream = Arc::new(MemoryStream::new());
    let mut first = build_core_with_stream(stream.clone());
    let block_size = first.data_block_size();
    let edited_voxel = Vector3i::new(1, 1, 1);
    let channel = ChannelId::Type.index();
    let viewer = [ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: block_size,
        vertical_view_distance_voxels: block_size,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];

    process_until(&mut first, &viewer, |core, _| {
        core.data().block_snapshot(Vector3i::zero(), 0).is_some()
    });
    assert!(first
        .data
        .try_set_voxel_checked(91, edited_voxel, channel)
        .unwrap());
    first
        .data
        .mark_area_modified_checked(Box3i::new(edited_voxel, Vector3i::splat(1)), false)
        .unwrap();
    process_until(&mut first, &[], |_core, _| stream.len() == 1);
    drop(first);

    let mut second = build_core_with_stream(stream);
    process_until(&mut second, &viewer, |core, _| {
        core.data().block_snapshot(Vector3i::zero(), 0).is_some()
    });
    let restored = second
        .data()
        .block_snapshot(Vector3i::zero(), 0)
        .expect("saved block restored");
    assert_eq!(restored.voxels().get_voxel(1, 1, 1, channel), 91);
}

#[test]
fn loaded_blocks_keep_viewer_refs_from_coalesced_pending_loads() {
    let mut core = build_core();
    let bs = core.data_block_size();

    let two_viewers = vec![
        ViewerUpdate {
            id: 1,
            world_position_voxels: Vector3i::zero(),
            horizontal_view_distance_voxels: bs,
            vertical_view_distance_voxels: bs,
            demand: MeshDemand {
                visuals: true,
                collisions: true,
            },
        },
        ViewerUpdate {
            id: 2,
            world_position_voxels: Vector3i::zero(),
            horizontal_view_distance_voxels: bs,
            vertical_view_distance_voxels: bs,
            demand: MeshDemand {
                visuals: true,
                collisions: true,
            },
        },
    ];
    process_until(&mut core, &two_viewers, |core, _events| {
        core.data()
            .block_snapshot(Vector3i::zero(), 0)
            .is_some_and(|block| block.viewers.get() == 2)
    });

    let one_viewer = vec![ViewerUpdate {
        id: 1,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: bs,
        vertical_view_distance_voxels: bs,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];
    process_until(&mut core, &one_viewer, |core, _events| {
        core.data()
            .block_snapshot(Vector3i::zero(), 0)
            .is_some_and(|block| block.viewers.get() == 1)
    });

    let data = core.data();
    let origin_block = data
        .block_snapshot(Vector3i::zero(), 0)
        .expect("origin block should stay loaded while one viewer still references it");
    assert_eq!(origin_block.viewers.get(), 1);
}

#[test]
fn mesh_task_output_downcast_rejects_debug_name_collision() {
    let mut task: Box<dyn ThreadedTask> = Box::new(DebugNameCollisionTask);
    assert!(try_take_mesh_output(task.as_mut()).is_none());
}

#[test]
fn cancelled_and_panicked_load_completions_rearm_exact_in_flight_request() {
    for status in [
        TaskCompletionStatus::Cancelled,
        TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run),
    ] {
        let mut core = build_core();
        let position = Vector3i::zero();
        core.apply_data_view(single_block_box(position), 0);
        let old_generation = core.loading_blocks[0][&position].request_generation;
        core.loading_blocks[0]
            .get_mut(&position)
            .unwrap()
            .request_state = LoadRequestState::InFlight;
        core.blocks_pending_load[0].clear();

        let task = tagged_current_load_task(&core, position, 0, old_generation);
        core.raw_completion_inbox
            .push_back(crate::tasks::CompletedTask::new(
                Box::new(task),
                TaskLane::Parallel,
                status,
                Vec::new(),
            ));

        core.try_drain_completed_tasks().unwrap();

        let entry = &core.loading_blocks[0][&position];
        assert_eq!(entry.residency.resident_viewers, 1);
        assert_eq!(entry.request_state, LoadRequestState::InFlight);
        assert!(entry.request_generation > old_generation);
        assert!(core.blocks_pending_load[0].is_empty());
        assert!(core.raw_completion_inbox.is_empty());
        assert!(core.durable_completion_inbox.is_empty());
    }
}

#[test]
fn terminal_load_at_retry_limit_becomes_exhausted_without_new_generation() {
    let mut core = build_core();
    let position = Vector3i::zero();
    core.apply_data_view(single_block_box(position), 0);
    let old_generation = core.loading_blocks[0][&position].request_generation;
    {
        let entry = core.loading_blocks[0].get_mut(&position).unwrap();
        entry.retry_count = MAX_LOAD_RETRIES;
        entry.request_state = LoadRequestState::InFlight;
    }
    core.blocks_pending_load[0].clear();
    let next_generation_before = core.next_request_generation;
    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(tagged_current_load_task(&core, position, 0, old_generation)),
            TaskLane::Parallel,
            TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run),
            Vec::new(),
        ));

    core.try_drain_completed_tasks().unwrap();

    let exhausted = &core.loading_blocks[0][&position];
    assert_eq!(exhausted.retry_count, MAX_LOAD_RETRIES + 1);
    assert_eq!(exhausted.request_generation, old_generation);
    assert_eq!(exhausted.request_state, LoadRequestState::Exhausted);
    assert_eq!(core.next_request_generation, next_generation_before);
    assert!(core.blocks_pending_load[0].is_empty());
}

#[test]
fn terminal_load_at_generation_max_exhausts_and_later_valid_completion_advances() {
    let mut core = build_core();
    let failed_position = Vector3i::zero();
    let valid_position = Vector3i::new(1, 0, 0);
    core.apply_data_view(single_block_box(failed_position), 0);
    core.apply_data_view(single_block_box(valid_position), 0);
    let failed_generation = core.loading_blocks[0][&failed_position].request_generation;
    let valid_generation = core.loading_blocks[0][&valid_position].request_generation;
    for position in [failed_position, valid_position] {
        core.loading_blocks[0]
            .get_mut(&position)
            .unwrap()
            .request_state = LoadRequestState::InFlight;
    }
    core.blocks_pending_load[0].clear();
    core.next_request_generation = u64::MAX;

    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(tagged_current_load_task(
                &core,
                failed_position,
                0,
                failed_generation,
            )),
            TaskLane::Parallel,
            TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run),
            Vec::new(),
        ));
    let mut valid_task = tagged_current_load_task(&core, valid_position, 0, valid_generation);
    valid_task.output = Some(loaded_output(
        &core,
        valid_position,
        valid_generation,
        0xA11D,
    ));
    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(valid_task),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            Vec::new(),
        ));

    assert!(matches!(
        core.try_drain_completed_tasks(),
        Err(VoxelTerrainRuntimeError::RequestGenerationOverflow)
    ));
    assert_eq!(core.durable_completion_inbox.len(), 2);
    assert_eq!(core.loading_blocks[0][&failed_position].retry_count, 0);
    assert!(core.data().block_snapshot(valid_position, 0).is_none());

    core.next_request_generation = 100;
    core.try_drain_completed_tasks().unwrap();

    let failed = &core.loading_blocks[0][&failed_position];
    assert_eq!(failed.request_state, LoadRequestState::InFlight);
    assert_eq!(failed.request_generation, 100);
    assert_eq!(failed.retry_count, 1);
    assert_eq!(core.next_request_generation, 101);
    assert_eq!(
        core.data()
            .block_snapshot(valid_position, 0)
            .unwrap()
            .voxels()
            .get_voxel(0, 0, 0, ChannelId::Type.index()),
        0xA11D
    );
    assert!(core.raw_completion_inbox.is_empty());
    assert!(core.durable_completion_inbox.is_empty());
}

#[test]
fn cancelled_and_panicked_mesh_completions_requeue_current_mesh_demand() {
    for status in [
        TaskCompletionStatus::Cancelled,
        TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run),
    ] {
        let mut core = make_resident_edit_core();
        let position = Vector3i::zero();
        let key = core.request_mesh_update(position, 0).unwrap();
        core.mesh_maps[0]
            .get_mut(&position)
            .unwrap()
            .is_in_update_list = false;
        core.blocks_pending_update[0].clear();
        let request = core.mesh_maps[0][&position]
            .physical_request
            .as_ref()
            .unwrap()
            .clone();

        let task = MeshBlockTask::new(MeshBlockTaskParams {
            key,
            data: core.data.clone(),
            meshing_dependency: core.meshing_dependency.clone(),
            collision_hint: false,
            lod_hint: false,
            mesh_arrays_pool: Some(core.mesh_arrays_pool.clone()),
        })
        .with_request_control(request.tag, request.cancellation);
        core.raw_completion_inbox
            .push_back(crate::tasks::CompletedTask::new(
                Box::new(task),
                TaskLane::Parallel,
                status,
                Vec::new(),
            ));

        core.try_drain_completed_tasks().unwrap();

        let entry = &core.mesh_maps[0][&position];
        assert_eq!(entry.requested_revision, Some(key.revision));
        assert!(!entry.is_in_update_list);
        assert!(core.blocks_pending_update[0].is_empty());
        assert!(core.raw_completion_inbox.is_empty());
        assert!(core.durable_completion_inbox.is_empty());
    }
}

#[test]
fn repeated_mesh_terminals_quiesce_at_budget_and_fresh_revision_rearms() {
    let mut core = make_resident_edit_core();
    core.automatic_loading_enabled = false;
    let position = Vector3i::zero();
    let key = core.request_mesh_update(position, 0).unwrap();
    core.blocks_pending_update[0].clear();
    core.mesh_maps[0]
        .get_mut(&position)
        .unwrap()
        .is_in_update_list = false;
    for attempt in 0..=MAX_MESH_TERMINAL_RETRIES {
        let request = core.mesh_maps[0][&position]
            .physical_request
            .as_ref()
            .expect("every retry below the terminal budget owns a fresh request")
            .clone();
        let previous_generation = request.tag.request_generation;
        let task = MeshBlockTask::new(MeshBlockTaskParams {
            key,
            data: core.data.clone(),
            meshing_dependency: core.meshing_dependency.clone(),
            collision_hint: false,
            lod_hint: false,
            mesh_arrays_pool: Some(core.mesh_arrays_pool.clone()),
        })
        .with_request_control(request.tag, request.cancellation.clone());
        let status = if attempt % 2 == 0 {
            TaskCompletionStatus::Cancelled
        } else {
            TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run)
        };
        core.raw_completion_inbox
            .push_back(crate::tasks::CompletedTask::new(
                Box::new(task),
                TaskLane::Parallel,
                status,
                Vec::new(),
            ));
        core.try_drain_completed_tasks().unwrap();

        let entry = &core.mesh_maps[0][&position];
        assert_eq!(entry.terminal_retry_count, attempt + 1);
        if attempt < MAX_MESH_TERMINAL_RETRIES {
            assert!(entry.is_in_update_list);
            assert_eq!(core.blocks_pending_update[0], vec![position]);
            assert_ne!(entry.request_generation, previous_generation);
            assert_eq!(
                entry.physical_request.as_ref().unwrap().tag,
                TaskRequestTag::new(core.request_epoch, entry.request_generation)
            );
            core.blocks_pending_update[0].clear();
            core.mesh_maps[0]
                .get_mut(&position)
                .unwrap()
                .is_in_update_list = false;
        } else {
            assert!(!entry.is_in_update_list);
            assert!(entry.physical_request.is_none());
            assert!(core.blocks_pending_update[0].is_empty());
        }
    }

    let fresh = core.request_mesh_update(position, 0).unwrap();
    assert!(fresh.revision > key.revision);
    assert_eq!(core.mesh_maps[0][&position].terminal_retry_count, 0);
    assert!(core.mesh_maps[0][&position].is_in_update_list);
    assert_eq!(core.blocks_pending_update[0], vec![position]);
}

#[test]
fn checked_normalization_failure_retains_raw_load_box_and_payload() {
    let mut core = build_core();
    let position = Vector3i::zero();
    core.apply_data_view(single_block_box(position), 0);
    let generation = core.loading_blocks[0][&position].request_generation;
    core.loading_blocks[0]
        .get_mut(&position)
        .unwrap()
        .request_state = LoadRequestState::InFlight;
    core.blocks_pending_load[0].clear();

    let mut task = tagged_current_load_task(&core, position, 0, generation);
    assert!(matches!(
        task.run(ThreadedTaskContext::new(0, TaskPriority::max())),
        TaskRunStatus::Complete { .. }
    ));
    task.output
        .as_mut()
        .unwrap()
        .block_data
        .voxels
        .as_mut()
        .unwrap()
        .set_voxel(0xC1, 0, 0, 0, ChannelId::Type.index());
    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(task),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            Vec::new(),
        ));
    let raw_task_identity = core.raw_completion_inbox[0]
        .task_any()
        .downcast_ref::<LoadBlockForTerrainTask>()
        .unwrap() as *const LoadBlockForTerrainTask as usize;
    core.fail_next_completion_normalization_for_test = true;

    assert!(matches!(
        core.try_drain_completed_tasks(),
        Err(VoxelTerrainRuntimeError::CompletionNormalizationFailed)
    ));
    assert_eq!(core.raw_completion_inbox.len(), 1);
    assert!(core.durable_completion_inbox.is_empty());
    let retained_task = core.raw_completion_inbox[0]
        .task_any()
        .downcast_ref::<LoadBlockForTerrainTask>()
        .unwrap();
    assert_eq!(
        retained_task as *const LoadBlockForTerrainTask as usize,
        raw_task_identity
    );
    assert_eq!(
        retained_task
            .output
            .as_ref()
            .unwrap()
            .block_data
            .voxels
            .as_ref()
            .unwrap()
            .get_voxel(0, 0, 0, ChannelId::Type.index()),
        0xC1
    );

    core.try_drain_completed_tasks().unwrap();
    assert!(core.raw_completion_inbox.is_empty());
    assert!(core.durable_completion_inbox.is_empty());
    assert_eq!(
        core.data()
            .block_snapshot(position, 0)
            .unwrap()
            .voxels()
            .get_voxel(0, 0, 0, ChannelId::Type.index()),
        0xC1
    );
}

#[test]
fn completion_follow_up_is_published_only_after_matching_state_acceptance() {
    let mut core = build_core();
    let position = Vector3i::zero();
    core.apply_data_view(single_block_box(position), 0);
    let generation = core.loading_blocks[0][&position].request_generation;
    core.loading_blocks[0]
        .get_mut(&position)
        .unwrap()
        .request_state = LoadRequestState::InFlight;
    core.blocks_pending_load[0].clear();
    let mut task = tagged_current_load_task(&core, position, 0, generation);
    assert!(matches!(
        task.run(ThreadedTaskContext::new(0, TaskPriority::max())),
        TaskRunStatus::Complete { .. }
    ));
    let follow_up_ran = Arc::new(AtomicBool::new(false));
    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(task),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            vec![ScheduledTask::new(
                Box::new(CompletionFollowUpTask {
                    ran: follow_up_ran.clone(),
                }),
                TaskLane::Serial,
            )],
        ));
    core.fail_next_follow_up_reservation_for_test = true;

    assert!(core.try_drain_completed_tasks().is_err());
    thread::sleep(Duration::from_millis(10));
    assert!(!follow_up_ran.load(Ordering::SeqCst));
    assert!(core.raw_completion_inbox.is_empty());
    assert_eq!(core.durable_completion_inbox.len(), 1);
    assert!(matches!(
        &core.durable_completion_inbox[0],
        DurableCompletion::LoadFinished { completed, .. }
            if completed.follow_up_count() == 1
    ));

    core.try_drain_completed_tasks().unwrap();
    core.wait_for_pending_tasks();
    assert!(follow_up_ran.load(Ordering::SeqCst));
}

#[test]
fn stale_finished_load_mesh_save_and_flush_never_publish_followups() {
    let mut core = build_core();
    let load_position = Vector3i::zero();
    core.apply_data_view(single_block_box(load_position), 0);
    let current_generation = core.loading_blocks[0][&load_position].request_generation;
    core.loading_blocks[0]
        .get_mut(&load_position)
        .unwrap()
        .request_state = LoadRequestState::InFlight;
    core.blocks_pending_load[0].clear();
    let stale_generation = current_generation + 1;
    let mut stale_load = LoadBlockForTerrainTask::new(
        load_position,
        0,
        stale_generation,
        core.data.clone(),
        core.stream.clone(),
    );
    stale_load.output = Some(loaded_output(&core, load_position, stale_generation, 0xBAD));
    let load_follow_up_ran = Arc::new(AtomicBool::new(false));
    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(stale_load),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            vec![ScheduledTask::new(
                Box::new(CompletionFollowUpTask {
                    ran: load_follow_up_ran.clone(),
                }),
                TaskLane::Serial,
            )],
        ));
    core.try_drain_completed_tasks().unwrap();
    core.wait_for_pending_tasks();
    assert!(!load_follow_up_ran.load(Ordering::SeqCst));
    assert!(core.data().block_snapshot(load_position, 0).is_none());
    assert_eq!(
        core.loading_blocks[0][&load_position].request_generation,
        current_generation
    );

    let mut mesh_core = make_resident_edit_core();
    let mesh_position = Vector3i::zero();
    let stale_key = mesh_core.request_mesh_update(mesh_position, 0).unwrap();
    let current_key = mesh_core.request_mesh_update(mesh_position, 0).unwrap();
    mesh_core.blocks_pending_update[0].clear();
    mesh_core.mesh_maps[0]
        .get_mut(&mesh_position)
        .unwrap()
        .is_in_update_list = false;
    let mut stale_mesh = MeshBlockTask::new(MeshBlockTaskParams {
        key: stale_key,
        data: mesh_core.data.clone(),
        meshing_dependency: mesh_core.meshing_dependency.clone(),
        collision_hint: false,
        lod_hint: false,
        mesh_arrays_pool: Some(mesh_core.mesh_arrays_pool.clone()),
    });
    stale_mesh.run_meshing();
    let mesh_follow_up_ran = Arc::new(AtomicBool::new(false));
    mesh_core
        .raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(stale_mesh),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            vec![ScheduledTask::new(
                Box::new(CompletionFollowUpTask {
                    ran: mesh_follow_up_ran.clone(),
                }),
                TaskLane::Serial,
            )],
        ));
    mesh_core.try_drain_completed_tasks().unwrap();
    mesh_core.wait_for_pending_tasks();
    assert!(!mesh_follow_up_ran.load(Ordering::SeqCst));
    assert_eq!(
        mesh_core.mesh_maps[0][&mesh_position].requested_revision,
        Some(current_key.revision)
    );
    assert_eq!(
        mesh_core.mesh_maps[0][&mesh_position].applied_revision,
        None
    );

    let mut persistence_core = build_core();
    let mut stale_save = SaveBlockDataTask::new_voxels_with_generation_at_revision(
        Vector3i::new(7, 0, 0),
        0,
        VoxelBuffer::with_size(Vector3i::splat(2)),
        StreamingDependency::new(persistence_core.stream.clone()),
        None,
        0,
        88,
    );
    assert!(matches!(
        stale_save.run(ThreadedTaskContext::new(0, TaskPriority::max())),
        TaskRunStatus::Complete { .. }
    ));
    let save_follow_up_ran = Arc::new(AtomicBool::new(false));
    persistence_core
        .raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(stale_save),
            TaskLane::Serial,
            TaskCompletionStatus::Finished,
            vec![ScheduledTask::new(
                Box::new(CompletionFollowUpTask {
                    ran: save_follow_up_ran.clone(),
                }),
                TaskLane::Serial,
            )],
        ));

    let mut stale_flush = FlushVoxelStreamTask::new(persistence_core.stream.clone(), 51);
    assert!(matches!(
        stale_flush.run(ThreadedTaskContext::new(0, TaskPriority::max())),
        TaskRunStatus::Complete { .. }
    ));
    let flush_follow_up_ran = Arc::new(AtomicBool::new(false));
    persistence_core
        .raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(stale_flush),
            TaskLane::Serial,
            TaskCompletionStatus::Finished,
            vec![ScheduledTask::new(
                Box::new(CompletionFollowUpTask {
                    ran: flush_follow_up_ran.clone(),
                }),
                TaskLane::Serial,
            )],
        ));

    persistence_core.try_drain_completed_tasks().unwrap();
    persistence_core.wait_for_pending_tasks();
    assert!(!save_follow_up_ran.load(Ordering::SeqCst));
    assert!(!flush_follow_up_ran.load(Ordering::SeqCst));
}

#[test]
fn unknown_finished_completion_retains_followups_while_later_valid_completion_advances() {
    let mut core = build_core();
    let valid_position = Vector3i::new(2, 0, 0);
    core.apply_data_view(single_block_box(valid_position), 0);
    let valid_generation = core.loading_blocks[0][&valid_position].request_generation;
    core.loading_blocks[0]
        .get_mut(&valid_position)
        .unwrap()
        .request_state = LoadRequestState::InFlight;
    core.blocks_pending_load[0].clear();

    let follow_up_ran = Arc::new(AtomicBool::new(false));
    let follow_up: Box<dyn ThreadedTask> = Box::new(CompletionFollowUpTask {
        ran: follow_up_ran.clone(),
    });
    let follow_up_identity = follow_up.as_ref() as *const dyn ThreadedTask as *const () as usize;
    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(DebugNameCollisionTask),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            vec![ScheduledTask::new(follow_up, TaskLane::Serial)],
        ));
    let unknown_identity =
        core.raw_completion_inbox[0].task() as *const dyn ThreadedTask as *const () as usize;

    let mut valid_task = LoadBlockForTerrainTask::new(
        valid_position,
        0,
        valid_generation,
        core.data.clone(),
        core.stream.clone(),
    );
    valid_task.output = Some(loaded_output(
        &core,
        valid_position,
        valid_generation,
        0x0A11,
    ));
    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(valid_task),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            Vec::new(),
        ));

    core.try_drain_completed_tasks().unwrap();

    assert_eq!(core.completion_quarantine.len(), 1);
    let QuarantinedCompletion::Other {
        kind,
        completed: quarantined,
    } = &core.completion_quarantine[0]
    else {
        panic!("unknown finished task must use the ordered other quarantine variant");
    };
    assert_eq!(*kind, CompletionTaskKind::Unknown);
    assert_eq!(quarantined.lane(), TaskLane::Parallel);
    assert_eq!(quarantined.status(), TaskCompletionStatus::Finished);
    assert_eq!(
        quarantined.task() as *const dyn ThreadedTask as *const () as usize,
        unknown_identity
    );
    assert_eq!(quarantined.follow_up_count(), 1);
    assert_eq!(
        quarantined.follow_up_task(0).unwrap().task() as *const dyn ThreadedTask as *const ()
            as usize,
        follow_up_identity
    );
    assert_eq!(
        quarantined.follow_up_task(0).unwrap().lane(),
        TaskLane::Serial
    );
    thread::sleep(Duration::from_millis(10));
    assert!(!follow_up_ran.load(Ordering::SeqCst));
    assert_eq!(
        core.data()
            .block_snapshot(valid_position, 0)
            .unwrap()
            .voxels()
            .get_voxel(0, 0, 0, ChannelId::Type.index()),
        0x0A11
    );
    assert!(core.raw_completion_inbox.is_empty());
    assert!(core.durable_completion_inbox.is_empty());
}

#[test]
fn panicked_save_and_flush_completions_are_quarantined_without_losing_boxes() {
    let mut core = build_core();
    let save = SaveBlockDataTask::new_voxels_with_generation_at_revision(
        Vector3i::new(4, 5, 6),
        0,
        VoxelBuffer::with_size(Vector3i::splat(2)),
        StreamingDependency::new(core.stream.clone()),
        None,
        0,
        77,
    );
    let flush = FlushVoxelStreamTask::new(core.stream.clone(), 78);
    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(save),
            TaskLane::Serial,
            TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run),
            Vec::new(),
        ));
    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(DebugNameCollisionTask),
            TaskLane::Parallel,
            TaskCompletionStatus::Cancelled,
            Vec::new(),
        ));
    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(LoadBlockForTerrainTask::new(
                Vector3i::new(9, 9, 9),
                0,
                999,
                core.data.clone(),
                core.stream.clone(),
            )),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            Vec::new(),
        ));
    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(flush),
            TaskLane::Serial,
            TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run),
            Vec::new(),
        ));
    let persistence_raw_identities = [0, 3]
        .into_iter()
        .map(|index| {
            core.raw_completion_inbox[index].task() as *const dyn ThreadedTask as *const () as usize
        })
        .collect::<Vec<_>>();
    let save_payload_identity = core.raw_completion_inbox[0]
        .task_any()
        .downcast_ref::<SaveBlockDataTask>()
        .unwrap()
        .retained_payload()
        .unwrap()
        .channel_bytes(ChannelId::Type.index())
        .as_ptr();
    let raw_statuses = core
        .raw_completion_inbox
        .iter()
        .map(|completed| (completed.lane(), completed.status()))
        .collect::<Vec<_>>();

    core.try_drain_completed_tasks().unwrap();

    assert!(core.raw_completion_inbox.is_empty());
    assert!(core.durable_completion_inbox.is_empty());
    assert_eq!(core.completion_quarantine.len(), 4);
    let QuarantinedCompletion::Persistence {
        kind: first_kind,
        completed: first,
        terminal: PersistenceTaskTerminal::Save(first_terminal),
        ..
    } = &core.completion_quarantine[0]
    else {
        panic!("first mixed terminal must remain the save");
    };
    assert_eq!(*first_kind, PersistenceTaskKind::Save);
    let QuarantinedCompletion::Other {
        kind: second_kind, ..
    } = &core.completion_quarantine[1]
    else {
        panic!("second mixed terminal must remain the unknown task");
    };
    assert_eq!(*second_kind, CompletionTaskKind::Unknown);
    let QuarantinedCompletion::Other {
        kind: third_kind, ..
    } = &core.completion_quarantine[2]
    else {
        panic!("third mixed terminal must remain the malformed load");
    };
    assert_eq!(*third_kind, CompletionTaskKind::Load);
    let QuarantinedCompletion::Persistence {
        kind: fourth_kind,
        completed: fourth,
        terminal: PersistenceTaskTerminal::Flush(fourth_terminal),
        ..
    } = &core.completion_quarantine[3]
    else {
        panic!("fourth mixed terminal must remain the flush");
    };
    assert_eq!(*fourth_kind, PersistenceTaskKind::Flush);
    assert_eq!(
        vec![
            first.task() as *const dyn ThreadedTask as *const () as usize,
            fourth.task() as *const dyn ThreadedTask as *const () as usize,
        ],
        persistence_raw_identities
    );
    assert_eq!(
        vec![
            (first.lane(), first.status()),
            (fourth.lane(), fourth.status())
        ],
        vec![raw_statuses[0], raw_statuses[3]]
    );
    assert_eq!(first_terminal.location.position, Vector3i::new(4, 5, 6));
    assert_eq!(first_terminal.location.lod_index, 0);
    assert_eq!(first_terminal.save_generation, 77);
    assert_eq!(first_terminal.phase, PersistenceIoPhase::BeforeIo);
    assert_eq!(
        first_terminal.task_panic_phase,
        Some(crate::tasks::TaskPanicPhase::Run)
    );
    assert!(first_terminal.acknowledgement.is_none());
    assert_eq!(
        first_terminal
            .payload
            .channel_bytes(ChannelId::Type.index())
            .as_ptr(),
        save_payload_identity
    );
    assert_eq!(fourth_terminal.checkpoint_generation, 78);
    assert_eq!(fourth_terminal.phase, PersistenceIoPhase::BeforeIo);
    assert_eq!(
        fourth_terminal.task_panic_phase,
        Some(crate::tasks::TaskPanicPhase::Run)
    );
    assert!(fourth_terminal.acknowledgement.is_none());
    assert_eq!(
        core.completion_quarantine[1].completed().lane(),
        raw_statuses[1].0
    );
    assert_eq!(
        core.completion_quarantine[2].completed().status(),
        raw_statuses[2].1
    );
}

#[test]
fn persistence_panic_phases_are_typed_and_do_not_block_later_completion() {
    fn payload(value: u64) -> VoxelBuffer {
        let mut payload = VoxelBuffer::with_size(Vector3i::splat(2));
        payload.set_voxel(value, 0, 0, 0, ChannelId::Type.index());
        payload
    }

    fn push_panicked(core: &mut VoxelTerrainCore, task: Box<dyn ThreadedTask>) {
        core.raw_completion_inbox
            .push_back(crate::tasks::CompletedTask::new(
                task,
                TaskLane::Serial,
                TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run),
                Vec::new(),
            ));
    }

    fn save_terminal(completion: &QuarantinedCompletion) -> &SaveTaskTerminal {
        let QuarantinedCompletion::Persistence {
            terminal: PersistenceTaskTerminal::Save(terminal),
            completed,
            ..
        } = completion
        else {
            panic!("expected typed save terminal");
        };
        assert_eq!(
            completed.status(),
            TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run)
        );
        terminal
    }

    fn flush_terminal(completion: &QuarantinedCompletion) -> &FlushTaskTerminal {
        let QuarantinedCompletion::Persistence {
            terminal: PersistenceTaskTerminal::Flush(terminal),
            completed,
            ..
        } = completion
        else {
            panic!("expected typed flush terminal");
        };
        assert_eq!(
            completed.status(),
            TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run)
        );
        terminal
    }

    let mut core = build_core();
    let healthy = Arc::new(PersistencePhaseStream::healthy());
    let panic_save_stream = Arc::new(PersistencePhaseStream::panic_save());
    let panic_flush_stream = Arc::new(PersistencePhaseStream::panic_flush());

    let mut save_before = SaveBlockDataTask::new_voxels_with_generation_at_revision(
        Vector3i::new(1, 0, 0),
        0,
        payload(11),
        StreamingDependency::new(healthy.clone()),
        None,
        0,
        101,
    );
    save_before.set_panic_before_io_for_test(true);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        save_before.run(ThreadedTaskContext::new(0, TaskPriority::max()))
    }))
    .is_err());
    push_panicked(&mut core, Box::new(save_before));

    let mut flush_before = FlushVoxelStreamTask::new(healthy.clone(), 201);
    flush_before.set_panic_before_io_for_test(true);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        flush_before.run(ThreadedTaskContext::new(0, TaskPriority::max()))
    }))
    .is_err());
    push_panicked(&mut core, Box::new(flush_before));

    let mut save_during = SaveBlockDataTask::new_voxels_with_generation_at_revision(
        Vector3i::new(2, 0, 0),
        0,
        payload(12),
        StreamingDependency::new(panic_save_stream.clone()),
        None,
        0,
        102,
    );
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        save_during.run(ThreadedTaskContext::new(0, TaskPriority::max()))
    }))
    .is_err());
    push_panicked(&mut core, Box::new(save_during));

    let mut flush_during = FlushVoxelStreamTask::new(panic_flush_stream.clone(), 202);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        flush_during.run(ThreadedTaskContext::new(0, TaskPriority::max()))
    }))
    .is_err());
    push_panicked(&mut core, Box::new(flush_during));

    let mut save_after = SaveBlockDataTask::new_voxels_with_generation_at_revision(
        Vector3i::new(3, 0, 0),
        0,
        payload(13),
        StreamingDependency::new(healthy.clone()),
        None,
        0,
        103,
    );
    save_after.set_panic_after_ack_for_test(true);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        save_after.run(ThreadedTaskContext::new(0, TaskPriority::max()))
    }))
    .is_err());
    push_panicked(&mut core, Box::new(save_after));

    let mut flush_after = FlushVoxelStreamTask::new(healthy.clone(), 203);
    flush_after.set_panic_after_ack_for_test(true);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        flush_after.run(ThreadedTaskContext::new(0, TaskPriority::max()))
    }))
    .is_err());
    push_panicked(&mut core, Box::new(flush_after));

    assert_eq!(healthy.save_calls.load(Ordering::SeqCst), 1);
    assert_eq!(healthy.flush_calls.load(Ordering::SeqCst), 1);
    assert_eq!(panic_save_stream.save_calls.load(Ordering::SeqCst), 1);
    assert_eq!(panic_flush_stream.flush_calls.load(Ordering::SeqCst), 1);

    let valid_position = Vector3i::new(9, 0, 0);
    core.apply_data_view(single_block_box(valid_position), 0);
    let valid_generation = core.loading_blocks[0][&valid_position].request_generation;
    core.loading_blocks[0]
        .get_mut(&valid_position)
        .unwrap()
        .request_state = LoadRequestState::InFlight;
    core.blocks_pending_load[0].clear();
    let mut valid_load = LoadBlockForTerrainTask::new(
        valid_position,
        0,
        valid_generation,
        core.data.clone(),
        core.stream.clone(),
    );
    valid_load.output = Some(loaded_output(
        &core,
        valid_position,
        valid_generation,
        0xC2A,
    ));
    core.raw_completion_inbox
        .push_back(crate::tasks::CompletedTask::new(
            Box::new(valid_load),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            Vec::new(),
        ));

    core.try_drain_completed_tasks().unwrap();

    assert_eq!(core.completion_quarantine.len(), 6);
    let before_save = save_terminal(&core.completion_quarantine[0]);
    assert_eq!(before_save.save_generation, 101);
    assert_eq!(before_save.phase, PersistenceIoPhase::BeforeIo);
    assert!(before_save.acknowledgement.is_none());
    let before_flush = flush_terminal(&core.completion_quarantine[1]);
    assert_eq!(before_flush.checkpoint_generation, 201);
    assert_eq!(before_flush.phase, PersistenceIoPhase::BeforeIo);
    assert!(before_flush.acknowledgement.is_none());

    let during_save = save_terminal(&core.completion_quarantine[2]);
    assert_eq!(during_save.save_generation, 102);
    assert_eq!(during_save.phase, PersistenceIoPhase::CallEntered);
    assert!(during_save.acknowledgement.is_none());
    let during_flush = flush_terminal(&core.completion_quarantine[3]);
    assert_eq!(during_flush.checkpoint_generation, 202);
    assert_eq!(during_flush.phase, PersistenceIoPhase::CallEntered);
    assert!(during_flush.acknowledgement.is_none());

    let after_save = save_terminal(&core.completion_quarantine[4]);
    assert_eq!(after_save.save_generation, 103);
    assert_eq!(after_save.phase, PersistenceIoPhase::Acknowledged);
    assert!(matches!(
        after_save.acknowledgement,
        Some(crate::streams::PersistenceAcknowledgement::Save(Ok(())))
    ));
    let after_flush = flush_terminal(&core.completion_quarantine[5]);
    assert_eq!(after_flush.checkpoint_generation, 203);
    assert_eq!(after_flush.phase, PersistenceIoPhase::Acknowledged);
    assert!(matches!(
        after_flush.acknowledgement,
        Some(crate::streams::PersistenceAcknowledgement::Flush(Ok(())))
    ));
    for completion in &core.completion_quarantine {
        let phase = match completion {
            QuarantinedCompletion::Persistence {
                terminal: PersistenceTaskTerminal::Save(terminal),
                ..
            } => terminal.task_panic_phase,
            QuarantinedCompletion::Persistence {
                terminal: PersistenceTaskTerminal::Flush(terminal),
                ..
            } => terminal.task_panic_phase,
            QuarantinedCompletion::MalformedPersistence { .. }
            | QuarantinedCompletion::Other { .. } => unreachable!(),
        };
        assert_eq!(phase, Some(crate::tasks::TaskPanicPhase::Run));
    }
    assert_eq!(
        core.data()
            .block_snapshot(valid_position, 0)
            .unwrap()
            .voxels()
            .get_voxel(0, 0, 0, ChannelId::Type.index()),
        0xC2A
    );
}

fn stage_c2b_save_attempt(
    core: &mut VoxelTerrainCore,
    stream: Arc<dyn VoxelStream>,
    location: BlockLocation,
    generation: u64,
    attempt_ordinal: u64,
    payload: VoxelBuffer,
) -> SaveBlockDataTask {
    stage_c2b_save_attempt_at_revision(
        core,
        stream,
        location,
        0,
        generation,
        attempt_ordinal,
        payload,
    )
}

fn stage_c2b_save_attempt_at_revision(
    core: &mut VoxelTerrainCore,
    stream: Arc<dyn VoxelStream>,
    location: BlockLocation,
    block_revision: u64,
    generation: u64,
    attempt_ordinal: u64,
    payload: VoxelBuffer,
) -> SaveBlockDataTask {
    core.save_journal.insert(
        SaveKey::new(location.position, location.lod_index),
        SaveJournalEntry::write_in_flight_for_test_at_revision(
            block_revision,
            generation,
            attempt_ordinal,
        ),
    );
    core.next_persistence_attempt_ordinal = attempt_ordinal + 1;
    SaveBlockDataTask::new_voxels_with_generation_and_attempt_ordinal(
        location,
        payload,
        StreamingDependency::new(stream),
        None,
        block_revision,
        generation,
        attempt_ordinal,
    )
}

fn push_panicked_save_and_drain(core: &mut VoxelTerrainCore, mut task: SaveBlockDataTask) {
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        task.run(ThreadedTaskContext::new(0, TaskPriority::max()))
    }))
    .is_err());
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Serial,
        TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run),
        Vec::new(),
    ));
    core.try_drain_completed_tasks().unwrap();
}

fn stage_single_written_checkpoint_member(
    core: &mut VoxelTerrainCore,
    location: BlockLocation,
    save_generation: u64,
    payload: VoxelBuffer,
) -> PersistenceOperation {
    stage_single_written_checkpoint_member_at_revision(core, location, 0, save_generation, payload)
}

fn stage_single_written_checkpoint_member_at_revision(
    core: &mut VoxelTerrainCore,
    location: BlockLocation,
    block_revision: u64,
    save_generation: u64,
    payload: VoxelBuffer,
) -> PersistenceOperation {
    core.save_journal.insert(
        SaveKey::new(location.position, location.lod_index),
        SaveJournalEntry {
            written_unflushed: Some(WrittenSave {
                block_revision,
                generation: save_generation,
                payload,
            }),
            active: None,
            queued_newer: VecDeque::new(),
        },
    );
    PersistenceOperation::Save {
        location,
        block_revision,
        save_generation,
    }
}

fn stage_c2b_flush_attempt(
    core: &mut VoxelTerrainCore,
    stream: Arc<dyn VoxelStream>,
    checkpoint_generation: u64,
    attempt_ordinal: u64,
    member: SaveCheckpointSnapshot,
) -> FlushVoxelStreamTask {
    core.save_checkpoint_in_flight = Some(SaveCheckpointInFlight {
        checkpoint_generation,
        acknowledged: vec![member],
        state: CheckpointAttemptState::WriteInFlight { attempt_ordinal },
        retry_count: 0,
        max_attempts: MAX_AUTOMATIC_CHECKPOINT_ATTEMPTS,
        origin: CheckpointOrigin::Automatic,
        record_per_block_failure: false,
    });
    core.next_persistence_attempt_ordinal = attempt_ordinal + 1;
    FlushVoxelStreamTask::new_with_attempt_ordinal(stream, checkpoint_generation, attempt_ordinal)
}

fn push_panicked_flush_and_drain(core: &mut VoxelTerrainCore, mut task: FlushVoxelStreamTask) {
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        task.run(ThreadedTaskContext::new(0, TaskPriority::max()))
    }))
    .is_err());
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Serial,
        TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run),
        Vec::new(),
    ));
    core.try_drain_completed_tasks().unwrap();
}

#[test]
fn successor_blocked_by_one_written_save_forces_checkpoint_below_threshold() {
    let stream = Arc::new(PersistencePhaseStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    let key = SaveKey::new(Vector3i::new(59, 0, 0), 0);
    core.save_journal.insert(
        key,
        SaveJournalEntry {
            written_unflushed: Some(WrittenSave {
                block_revision: 0,
                generation: 490,
                payload: VoxelBuffer::with_size(Vector3i::splat(2)),
            }),
            active: None,
            queued_newer: VecDeque::from([PendingSave {
                meta: SaveAttemptMeta {
                    block_revision: 1,
                    generation: 491,
                    retry_count: 0,
                    last_error: None,
                },
                payload: VoxelBuffer::with_size(Vector3i::splat(2)),
            }]),
        },
    );

    core.checkpoint_acknowledged_saves_if_needed();
    core.wait_for_pending_tasks();
    core.try_drain_completed_tasks().unwrap();

    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    let entry = core.save_journal.get(&key).unwrap();
    assert!(entry.written_unflushed.is_none());
    assert!(matches!(
        entry.active,
        Some(ActiveSaveAttempt::Pending(ref pending)) if pending.meta.generation == 491
    ));
}

#[test]
fn same_key_dirty_exit_and_checkpoint_completion_promote_exact_new_owner_atomically() {
    let mut core = build_core();
    let position = Vector3i::new(61, 0, 0);
    let key = SaveKey::new(position, 0);
    let mut dirty_voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
    dirty_voxels.set_voxel(0xC2C2, 0, 0, 0, ChannelId::Type.index());
    let dirty_ptr = dirty_voxels.channel_bytes(ChannelId::Type.index()).as_ptr();
    let mut dirty_block = VoxelDataBlock::with_voxels(dirty_voxels, 0);
    dirty_block.set_modified(true);
    assert!(core.data.try_set_block(position, dirty_block).unwrap());
    core.data
        .view_area(single_block_box(position), 0, None, None, None)
        .unwrap();
    core.loaded_data_residency[0].insert(position, DataResidencyRefs::with_resident_viewers(1));
    let state = ViewerState {
        data_box: single_block_box(position),
        ..ViewerState::default()
    };
    core.paired_viewers.push(PairedViewer {
        id: 1,
        state: state.clone(),
        prev_state: state,
    });

    const SAVE_GENERATION: u64 = 700;
    const CHECKPOINT_GENERATION: u64 = 701;
    const ATTEMPT_ORDINAL: u64 = 702;
    core.save_journal.insert(
        key,
        SaveJournalEntry {
            written_unflushed: Some(WrittenSave {
                block_revision: 0,
                generation: SAVE_GENERATION,
                payload: VoxelBuffer::with_size(Vector3i::splat(2)),
            }),
            active: None,
            queued_newer: VecDeque::new(),
        },
    );
    core.save_checkpoint_in_flight = Some(SaveCheckpointInFlight {
        checkpoint_generation: CHECKPOINT_GENERATION,
        acknowledged: vec![SaveCheckpointSnapshot {
            key,
            block_revision: 0,
            generation: SAVE_GENERATION,
        }],
        state: CheckpointAttemptState::WriteInFlight {
            attempt_ordinal: ATTEMPT_ORDINAL,
        },
        retry_count: 0,
        max_attempts: MAX_AUTOMATIC_CHECKPOINT_ATTEMPTS,
        origin: CheckpointOrigin::Automatic,
        record_per_block_failure: false,
    });
    core.durable_completion_inbox
        .push_back(DurableCompletion::FlushAcknowledged {
            completed: CompletedTask::new(
                Box::new(DebugNameCollisionTask),
                TaskLane::Serial,
                TaskCompletionStatus::Finished,
                Vec::new(),
            ),
            terminal: FlushTaskTerminal {
                checkpoint_generation: CHECKPOINT_GENERATION,
                task_panic_phase: None,
                phase: PersistenceIoPhase::Acknowledged,
                acknowledgement: Some(PersistenceAcknowledgement::Flush(Ok(()))),
            },
            attempt_ordinal: ATTEMPT_ORDINAL,
        });

    core.fixed_after_prepare_settings_conflict_for_test = true;
    assert!(matches!(
        core.prepare_fixed_viewer_transaction(&[], true, true, false),
        Err(VoxelTerrainRuntimeError::DataMutation(
            SharedVoxelDataMutationError::ConcurrentSettingsMutation { .. }
        ))
    ));
    assert_eq!(
        core.data.with_lod_map(0, |map| {
            map.get_block(position)
                .unwrap()
                .voxels()
                .channel_bytes(ChannelId::Type.index())
                .as_ptr()
        }),
        dirty_ptr
    );
    assert!(core.save_journal[&key].active.is_none());
    assert!(core.save_journal[&key].queued_newer.is_empty());
    assert_eq!(core.durable_completion_inbox.len(), 1);

    core.prepare_fixed_viewer_transaction(&[], true, true, false)
        .unwrap();

    let entry = core.save_journal.get(&key).unwrap();
    assert!(entry.written_unflushed.is_none());
    let Some(ActiveSaveAttempt::Pending(pending)) = &entry.active else {
        panic!("the dirty successor must be promoted after the old checkpoint clears")
    };
    assert_eq!(pending.meta.generation, 1);
    assert_eq!(
        pending
            .payload
            .channel_bytes(ChannelId::Type.index())
            .as_ptr(),
        dirty_ptr
    );
    assert!(entry.queued_newer.is_empty());
    assert!(core.save_checkpoint_in_flight.is_none());
    assert!(core.durable_completion_inbox.is_empty());
}

#[test]
fn checkpoint_generation_and_attempt_overflow_are_typed_and_non_mutating() {
    for overflow_generation in [true, false] {
        let stream = Arc::new(PersistencePhaseStream::healthy());
        let mut core = build_core_with_stream(stream.clone());
        let location = BlockLocation {
            position: Vector3i::new(58 + i32::from(overflow_generation), 0, 0),
            lod_index: 0,
        };
        let payload = VoxelBuffer::with_size(Vector3i::splat(2));
        let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
        let save = stage_single_written_checkpoint_member(&mut core, location, 489, payload);
        if overflow_generation {
            core.next_save_checkpoint_generation = u64::MAX;
        } else {
            core.next_persistence_attempt_ordinal = u64::MAX;
        }

        let result = core.flush_pending_saves();
        let expected = if overflow_generation {
            VoxelTerrainRuntimeError::SaveGenerationOverflow
        } else {
            VoxelTerrainRuntimeError::PersistenceAttemptOverflow {
                operation: PersistenceOperation::Flush {
                    checkpoint_generation: 1,
                },
            }
        };
        assert_eq!(
            result,
            Err(SaveFlushError::SaveAdmission {
                error: expected.clone(),
            })
        );
        assert!(!core.shutdown_in_progress);
        assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 0);
        assert_eq!(core.journal_payload_ptr_for_test(save), Some(payload_ptr));
        assert_eq!(
            core.journal_persistence_state_for_test(save),
            Some(JournalPersistenceState::WrittenUnflushed)
        );
        assert_eq!(
            core.flush_pending_saves(),
            Err(SaveFlushError::SaveAdmission { error: expected })
        );
    }
}

#[test]
fn disabled_automatic_loading_still_drains_completed_checkpoint() {
    let stream = Arc::new(PersistencePhaseStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(57, 0, 0),
        lod_index: 0,
    };
    let save = stage_single_written_checkpoint_member(
        &mut core,
        location,
        488,
        VoxelBuffer::with_size(Vector3i::splat(2)),
    );
    core.force_checkpoint_requested = true;
    core.checkpoint_acknowledged_saves_if_needed();
    core.wait_for_pending_tasks();
    core.automatic_loading_enabled = false;

    core.try_process(&[]).unwrap();

    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert_eq!(core.journal_persistence_state_for_test(save), None);
    assert!(core.save_checkpoint_in_flight.is_none());
}

#[test]
fn explicit_flush_task_panics_are_typed_bounded_and_exactly_acknowledged() {
    let stream = Arc::new(PersistencePhaseStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(55, 0, 0),
        lod_index: 0,
    };
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let save = stage_single_written_checkpoint_member(&mut core, location, 486, payload);
    core.panic_next_flush_before_io_attempts_for_test = MAX_EXPLICIT_CHECKPOINT_ATTEMPTS as usize;
    let result = core.flush_pending_saves();
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 0);
    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        result,
        Err(SaveFlushError::SaveAdmission {
            error: VoxelTerrainRuntimeError::PersistenceRetryLimitExceeded {
                operation: PersistenceOperation::Flush {
                    checkpoint_generation: 1,
                },
            },
        })
    );
    assert!(!core.shutdown_in_progress);
    assert_eq!(
        core.checkpoint_retry_count_for_test(PersistenceOperation::Flush {
            checkpoint_generation: 1,
        }),
        Some(MAX_EXPLICIT_CHECKPOINT_ATTEMPTS)
    );
    let checkpoint = core.save_checkpoint_in_flight.as_ref().unwrap();
    assert_eq!(checkpoint.checkpoint_generation, 1);
    assert_eq!(checkpoint.acknowledged.len(), 1);
    assert_eq!(
        checkpoint.acknowledged[0].key,
        SaveKey::new(location.position, 0)
    );
    assert_eq!(checkpoint.acknowledged[0].generation, 486);
    assert_eq!(
        core.journal_persistence_state_for_test(save),
        Some(JournalPersistenceState::WrittenUnflushed)
    );
    assert_eq!(core.journal_payload_ptr_for_test(save), Some(payload_ptr));
    assert_eq!(core.flush_pending_saves(), Ok(()));
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        core.checkpoint_persistence_state_for_test(PersistenceOperation::Flush {
            checkpoint_generation: 1,
        }),
        None
    );
    assert_eq!(core.journal_persistence_state_for_test(save), None);
    assert!(!core.shutdown_in_progress);

    let stream = Arc::new(PersistencePhaseStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    core.panic_next_flush_after_ack_for_test = true;
    assert_eq!(core.flush_pending_saves(), Ok(()));
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert!(!core.shutdown_in_progress);
}

#[test]
fn automatic_before_io_checkpoint_retries_stop_at_budget_until_explicit_reauthorization() {
    let stream = Arc::new(PersistencePhaseStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(56, 0, 0),
        lod_index: 0,
    };
    let save = stage_single_written_checkpoint_member(
        &mut core,
        location,
        487,
        VoxelBuffer::with_size(Vector3i::splat(2)),
    );
    core.panic_next_flush_before_io_attempts_for_test = MAX_AUTOMATIC_CHECKPOINT_ATTEMPTS as usize;
    core.force_checkpoint_requested = true;

    let mut reached_budget = false;
    for _ in 0..(MAX_AUTOMATIC_CHECKPOINT_ATTEMPTS as usize * 2 + 4) {
        core.wait_for_pending_tasks();
        core.try_process(&[]).unwrap();
        let operation = PersistenceOperation::Flush {
            checkpoint_generation: 1,
        };
        if core.checkpoint_retry_count_for_test(operation)
            == Some(MAX_AUTOMATIC_CHECKPOINT_ATTEMPTS)
            && core.checkpoint_persistence_state_for_test(operation)
                == Some(JournalPersistenceState::PendingWrite)
        {
            reached_budget = true;
            break;
        }
    }
    assert!(reached_budget, "automatic retry budget must quiesce");
    core.try_process(&[]).unwrap();

    let checkpoint = PersistenceOperation::Flush {
        checkpoint_generation: 1,
    };
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        core.checkpoint_persistence_state_for_test(checkpoint),
        Some(JournalPersistenceState::PendingWrite)
    );
    assert_eq!(
        core.checkpoint_retry_count_for_test(checkpoint),
        Some(MAX_AUTOMATIC_CHECKPOINT_ATTEMPTS)
    );
    assert_eq!(
        core.journal_persistence_state_for_test(save),
        Some(JournalPersistenceState::WrittenUnflushed)
    );

    assert_eq!(core.flush_pending_saves(), Ok(()));
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert_eq!(core.journal_persistence_state_for_test(save), None);
}

#[test]
fn explicit_call_entered_flush_is_typed_and_resolver_keeps_core_retryable() {
    let stream = Arc::new(PersistencePhaseStream::panic_flush());
    let mut core = build_core_with_stream(stream.clone());
    let operation = PersistenceOperation::Flush {
        checkpoint_generation: 1,
    };

    assert_eq!(
        core.flush_pending_saves(),
        Err(SaveFlushError::IndeterminatePersistence { operation })
    );
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert!(!core.shutdown_in_progress);
    assert_eq!(
        core.checkpoint_persistence_state_for_test(operation),
        Some(JournalPersistenceState::Indeterminate)
    );
    core.try_resolve_indeterminate_persistence(
        operation,
        IndeterminateIoResolution::AssumeWrittenAndFlush,
    )
    .unwrap();
    assert_eq!(core.flush_pending_saves(), Ok(()));
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn multi_key_indeterminate_save_diagnostic_is_canonical() {
    let candidates = [
        (SaveKey::new(Vector3i::new(5, 0, 0), 0), 9),
        (SaveKey::new(Vector3i::new(-2, 9, 0), 0), 7),
        (SaveKey::new(Vector3i::new(-100, 0, 0), 1), 1),
    ];
    let expected = PersistenceOperation::Save {
        location: BlockLocation {
            position: Vector3i::new(-2, 9, 0),
            lod_index: 0,
        },
        block_revision: 0,
        save_generation: 7,
    };

    for reversed in [false, true] {
        let mut core = build_core();
        let indices: &[usize] = if reversed { &[2, 1, 0] } else { &[0, 1, 2] };
        for &index in indices {
            let (key, generation) = candidates[index];
            core.save_journal.insert(
                key,
                SaveJournalEntry {
                    written_unflushed: None,
                    active: Some(ActiveSaveAttempt::Indeterminate {
                        meta: SaveAttemptMeta {
                            block_revision: 0,
                            generation,
                            retry_count: 0,
                            last_error: None,
                        },
                        attempt_ordinal: generation + 100,
                    }),
                    queued_newer: VecDeque::new(),
                },
            );
        }

        assert_eq!(
            core.first_indeterminate_persistence_operation(),
            Some(expected)
        );
        assert_eq!(
            core.flush_pending_saves(),
            Err(SaveFlushError::IndeterminatePersistence {
                operation: expected,
            })
        );
        assert!(!core.shutdown_in_progress);
    }
}

#[test]
fn flush_panic_phase_matrix_preserves_exact_checkpoint_identity() {
    // BeforeIo: no external call, same member set, later fresh attempt.
    let stream = Arc::new(PersistencePhaseStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(60, 0, 0),
        lod_index: 0,
    };
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let save = stage_single_written_checkpoint_member(&mut core, location, 501, payload);
    let checkpoint = PersistenceOperation::Flush {
        checkpoint_generation: 601,
    };
    let mut task = stage_c2b_flush_attempt(
        &mut core,
        stream.clone(),
        601,
        1001,
        SaveCheckpointSnapshot {
            key: SaveKey::new(location.position, location.lod_index),
            block_revision: 0,
            generation: 501,
        },
    );
    task.set_panic_before_io_for_test(true);
    push_panicked_flush_and_drain(&mut core, task);

    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        core.checkpoint_persistence_state_for_test(checkpoint),
        Some(JournalPersistenceState::PendingWrite)
    );
    assert_eq!(core.checkpoint_retry_count_for_test(checkpoint), Some(1));
    assert_eq!(core.journal_payload_ptr_for_test(save), Some(payload_ptr));
    core.dispatch_pending_checkpoint();
    core.wait_for_pending_tasks();
    core.try_drain_completed_tasks().unwrap();
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert_eq!(core.journal_persistence_state_for_test(save), None);

    // CallEntered: one call, indeterminate quarantine, no blind retry.
    let stream = Arc::new(PersistencePhaseStream::panic_flush());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(61, 0, 0),
        lod_index: 0,
    };
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let save = stage_single_written_checkpoint_member(&mut core, location, 502, payload);
    let checkpoint = PersistenceOperation::Flush {
        checkpoint_generation: 602,
    };
    let task = stage_c2b_flush_attempt(
        &mut core,
        stream.clone(),
        602,
        1002,
        SaveCheckpointSnapshot {
            key: SaveKey::new(location.position, location.lod_index),
            block_revision: 0,
            generation: 502,
        },
    );
    push_panicked_flush_and_drain(&mut core, task);
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        core.checkpoint_persistence_state_for_test(checkpoint),
        Some(JournalPersistenceState::Indeterminate)
    );
    assert_eq!(core.journal_payload_ptr_for_test(save), Some(payload_ptr));
    for _ in 0..3 {
        core.try_process(&[]).unwrap();
    }
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        core.flush_pending_saves(),
        Err(SaveFlushError::IndeterminatePersistence {
            operation: checkpoint,
        })
    );
    assert!(!core.shutdown_in_progress);
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);

    // Acknowledged: the exact acknowledgement is applied once despite panic.
    let stream = Arc::new(PersistencePhaseStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(62, 0, 0),
        lod_index: 0,
    };
    let save = stage_single_written_checkpoint_member(
        &mut core,
        location,
        503,
        VoxelBuffer::with_size(Vector3i::splat(2)),
    );
    let mut task = stage_c2b_flush_attempt(
        &mut core,
        stream.clone(),
        603,
        1003,
        SaveCheckpointSnapshot {
            key: SaveKey::new(location.position, location.lod_index),
            block_revision: 0,
            generation: 503,
        },
    );
    task.set_panic_after_ack_for_test(true);
    push_panicked_flush_and_drain(&mut core, task);
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert_eq!(core.journal_persistence_state_for_test(save), None);
    core.try_drain_completed_tasks().unwrap();
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn indeterminate_flush_resolution_matrix_is_exact_and_one_shot() {
    for resolution in [
        IndeterminateIoResolution::AssumeNotWrittenAndRetry,
        IndeterminateIoResolution::AssumeWrittenAndFlush,
    ] {
        let panic_stream = Arc::new(PersistencePhaseStream::panic_flush());
        let healthy_stream = Arc::new(PersistencePhaseStream::healthy());
        let mut core = build_core_with_stream(panic_stream.clone());
        let checkpoint_generation = 610
            + u64::from(matches!(
                resolution,
                IndeterminateIoResolution::AssumeWrittenAndFlush
            ));
        let location = BlockLocation {
            position: Vector3i::new(checkpoint_generation as i32, 0, 0),
            lod_index: 0,
        };
        let save = stage_single_written_checkpoint_member(
            &mut core,
            location,
            510,
            VoxelBuffer::with_size(Vector3i::splat(2)),
        );
        let checkpoint = PersistenceOperation::Flush {
            checkpoint_generation,
        };
        let task = stage_c2b_flush_attempt(
            &mut core,
            panic_stream.clone(),
            checkpoint_generation,
            checkpoint_generation + 1000,
            SaveCheckpointSnapshot {
                key: SaveKey::new(location.position, location.lod_index),
                block_revision: 0,
                generation: 510,
            },
        );
        push_panicked_flush_and_drain(&mut core, task);
        let quarantine_len = core.completion_quarantine.len();
        core.stream = healthy_stream.clone();

        core.try_resolve_indeterminate_persistence(checkpoint, resolution)
            .unwrap();
        core.wait_for_pending_tasks();
        core.try_drain_completed_tasks().unwrap();

        assert_eq!(core.journal_persistence_state_for_test(save), None);
        assert_eq!(panic_stream.flush_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            healthy_stream.flush_calls.load(Ordering::SeqCst),
            usize::from(matches!(
                resolution,
                IndeterminateIoResolution::AssumeNotWrittenAndRetry
            ))
        );
        assert_eq!(healthy_stream.save_calls.load(Ordering::SeqCst), 0);
        assert_eq!(core.completion_quarantine.len(), quarantine_len - 1);
        assert_eq!(
            core.try_resolve_indeterminate_persistence(checkpoint, resolution),
            Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                requested: checkpoint,
            })
        );
    }
}

#[test]
fn indeterminate_flush_resolution_mismatch_and_attempt_overflow_are_atomic() {
    let stream = Arc::new(PersistencePhaseStream::panic_flush());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(63, 0, 0),
        lod_index: 0,
    };
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let save = stage_single_written_checkpoint_member(&mut core, location, 511, payload);
    let checkpoint = PersistenceOperation::Flush {
        checkpoint_generation: 612,
    };
    let task = stage_c2b_flush_attempt(
        &mut core,
        stream.clone(),
        612,
        1012,
        SaveCheckpointSnapshot {
            key: SaveKey::new(location.position, location.lod_index),
            block_revision: 0,
            generation: 511,
        },
    );
    push_panicked_flush_and_drain(&mut core, task);
    let quarantine_len = core.completion_quarantine.len();
    let wrong = PersistenceOperation::Flush {
        checkpoint_generation: 613,
    };

    assert_eq!(
        core.try_resolve_indeterminate_persistence(
            wrong,
            IndeterminateIoResolution::AssumeWrittenAndFlush,
        ),
        Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch { requested: wrong })
    );
    core.next_persistence_attempt_ordinal = u64::MAX;
    assert_eq!(
        core.try_resolve_indeterminate_persistence(
            checkpoint,
            IndeterminateIoResolution::AssumeNotWrittenAndRetry,
        ),
        Err(VoxelTerrainRuntimeError::PersistenceAttemptOverflow {
            operation: checkpoint,
        })
    );
    assert_eq!(core.completion_quarantine.len(), quarantine_len);
    assert_eq!(core.journal_payload_ptr_for_test(save), Some(payload_ptr));
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        core.checkpoint_persistence_state_for_test(checkpoint),
        Some(JournalPersistenceState::Indeterminate)
    );
}

#[test]
fn persistence_panic_before_io_retries_with_payload_and_fresh_attempt() {
    let stream = Arc::new(PersistencePhaseStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(40, 0, 0),
        lod_index: 0,
    };
    let operation = PersistenceOperation::Save {
        location,
        block_revision: 0,
        save_generation: 401,
    };
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let first_attempt = 900;
    let mut task = stage_c2b_save_attempt(
        &mut core,
        stream.clone(),
        location,
        401,
        first_attempt,
        payload,
    );
    task.set_panic_before_io_for_test(true);

    push_panicked_save_and_drain(&mut core, task);

    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        core.journal_persistence_state_for_test(operation),
        Some(JournalPersistenceState::PendingWrite)
    );
    assert_eq!(
        core.journal_payload_ptr_for_test(operation),
        Some(payload_ptr)
    );
    assert_eq!(core.next_persistence_attempt_ordinal, first_attempt + 1);

    core.dispatch_queued_saves_if_allowed();
    core.wait_for_pending_tasks();
    core.try_drain_completed_tasks().unwrap();

    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 1);
    assert!(core.next_persistence_attempt_ordinal > first_attempt + 1);
}

#[test]
fn persistence_panic_during_io_is_retained_and_never_blindly_reissued() {
    let stream = Arc::new(PersistencePhaseStream::panic_save());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(41, 0, 0),
        lod_index: 0,
    };
    let operation = PersistenceOperation::Save {
        location,
        block_revision: 0,
        save_generation: 402,
    };
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let task = stage_c2b_save_attempt(&mut core, stream.clone(), location, 402, 901, payload);

    push_panicked_save_and_drain(&mut core, task);

    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        core.journal_persistence_state_for_test(operation),
        Some(JournalPersistenceState::Indeterminate)
    );
    assert_eq!(
        core.quarantined_save_payload_ptr_for_test(operation),
        Some(payload_ptr)
    );
    for _ in 0..4 {
        core.try_process(&[]).unwrap();
        core.dispatch_queued_saves_if_allowed();
    }
    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 1);

    core.try_resolve_indeterminate_persistence(
        operation,
        IndeterminateIoResolution::AssumeWrittenAndFlush,
    )
    .unwrap();
    assert_eq!(
        core.journal_persistence_state_for_test(operation),
        Some(JournalPersistenceState::WrittenUnflushed)
    );
    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn persistence_panic_after_ack_applies_ack_without_duplicate_write_or_loss() {
    let stream = Arc::new(PersistencePhaseStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(42, 0, 0),
        lod_index: 0,
    };
    let operation = PersistenceOperation::Save {
        location,
        block_revision: 0,
        save_generation: 403,
    };
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let mut task = stage_c2b_save_attempt(&mut core, stream.clone(), location, 403, 902, payload);
    task.set_panic_after_ack_for_test(true);

    push_panicked_save_and_drain(&mut core, task);

    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        core.journal_persistence_state_for_test(operation),
        Some(JournalPersistenceState::WrittenUnflushed)
    );
    assert_eq!(
        core.journal_payload_ptr_for_test(operation),
        Some(payload_ptr)
    );
    core.try_drain_completed_tasks().unwrap();
    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 1);

    core.flush_pending_saves().unwrap();
    assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 1);
    assert_eq!(core.journal_persistence_state_for_test(operation), None);
}

#[test]
fn indeterminate_save_resolution_matrix_is_exact_and_one_shot() {
    for (generation, resolution, expected_save_calls, expected_state) in [
        (
            404,
            IndeterminateIoResolution::AssumeNotWrittenAndRetry,
            2,
            Some(JournalPersistenceState::WrittenUnflushed),
        ),
        (
            405,
            IndeterminateIoResolution::AssumeWrittenAndFlush,
            1,
            None,
        ),
    ] {
        let panic_stream = Arc::new(PersistencePhaseStream::panic_save());
        let healthy_stream = Arc::new(PersistencePhaseStream::healthy());
        let mut core = build_core_with_stream(panic_stream.clone());
        let location = BlockLocation {
            position: Vector3i::new(generation as i32, 0, 0),
            lod_index: 0,
        };
        let operation = PersistenceOperation::Save {
            location,
            block_revision: 0,
            save_generation: generation,
        };
        let task = stage_c2b_save_attempt(
            &mut core,
            panic_stream.clone(),
            location,
            generation,
            generation + 1000,
            VoxelBuffer::with_size(Vector3i::splat(2)),
        );
        push_panicked_save_and_drain(&mut core, task);
        core.stream = healthy_stream.clone();

        core.try_resolve_indeterminate_persistence(operation, resolution)
            .unwrap();
        core.wait_for_pending_tasks();
        core.try_drain_completed_tasks().unwrap();

        assert_eq!(
            panic_stream.save_calls.load(Ordering::SeqCst)
                + healthy_stream.save_calls.load(Ordering::SeqCst),
            expected_save_calls
        );
        assert_eq!(
            core.journal_persistence_state_for_test(operation),
            expected_state
        );
        assert_eq!(
            core.try_resolve_indeterminate_persistence(operation, resolution),
            Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
                requested: operation,
            })
        );
    }
}

#[test]
fn indeterminate_save_resolution_mismatch_is_atomic() {
    let stream = Arc::new(PersistencePhaseStream::panic_save());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(43, 0, 0),
        lod_index: 0,
    };
    let operation = PersistenceOperation::Save {
        location,
        block_revision: 0,
        save_generation: 406,
    };
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let task = stage_c2b_save_attempt(&mut core, stream.clone(), location, 406, 903, payload);
    push_panicked_save_and_drain(&mut core, task);
    let quarantine_len = core.completion_quarantine.len();
    let next_attempt = core.next_persistence_attempt_ordinal;
    let pending_tasks = core.pending_task_count();

    let wrong = PersistenceOperation::Save {
        location,
        block_revision: 0,
        save_generation: 407,
    };
    assert_eq!(
        core.try_resolve_indeterminate_persistence(
            wrong,
            IndeterminateIoResolution::AssumeNotWrittenAndRetry,
        ),
        Err(VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch { requested: wrong })
    );
    assert_eq!(core.completion_quarantine.len(), quarantine_len);
    assert_eq!(core.next_persistence_attempt_ordinal, next_attempt);
    assert_eq!(core.pending_task_count(), pending_tasks);
    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        core.quarantined_save_payload_ptr_for_test(operation),
        Some(payload_ptr)
    );
}

#[test]
fn acknowledged_save_error_restores_pending_exact_payload_and_error() {
    let stream = Arc::new(ControlledFailureStream::new(true));
    let mut core = build_core_with_stream(stream.clone());
    core.shutdown_in_progress = true;
    let location = BlockLocation {
        position: Vector3i::new(1, 0, 0),
        lod_index: 0,
    };
    let operation = PersistenceOperation::Save {
        location,
        block_revision: 0,
        save_generation: 408,
    };
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let mut task = stage_c2b_save_attempt(&mut core, stream, location, 408, 904, payload);
    task.set_panic_after_ack_for_test(true);

    push_panicked_save_and_drain(&mut core, task);

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
        Some(VoxelStreamError::Io(message)) if message.contains("permission denied")
    ));
}

#[test]
fn save_generation_overflow_is_typed_and_non_mutating() {
    let mut core = build_core();
    core.next_save_generation = u64::MAX;
    core.automatic_checkpoint_satisfied_empty_flush = true;
    let save = BlockToSave {
        voxels: Some(VoxelBuffer::with_size(Vector3i::splat(2))),
        position: Vector3i::new(44, 0, 0),
        lod_index: 0,
        block_revision: 0,
    };
    let payload_ptr = save
        .voxels
        .as_ref()
        .unwrap()
        .channel_bytes(ChannelId::Type.index())
        .as_ptr();

    let Err(error) = core.try_enqueue_data_save(save) else {
        panic!("generation exhaustion must return the exact unconsumed save");
    };
    let (error, returned) = *error;

    assert_eq!(error, VoxelTerrainRuntimeError::SaveGenerationOverflow);
    assert_eq!(core.next_save_generation, u64::MAX);
    assert!(core.automatic_checkpoint_satisfied_empty_flush);
    assert!(core.save_journal.is_empty());
    assert_eq!(
        returned
            .voxels
            .as_ref()
            .unwrap()
            .channel_bytes(ChannelId::Type.index())
            .as_ptr(),
        payload_ptr
    );
}

#[test]
fn c2b_public_state_and_structured_runtime_errors_are_exact() {
    let operation = PersistenceOperation::Save {
        location: BlockLocation {
            position: Vector3i::new(45, 0, 0),
            lod_index: 2,
        },
        block_revision: 0,
        save_generation: 409,
    };

    assert_ne!(
        JournalPersistenceState::Durable,
        JournalPersistenceState::WrittenUnflushed
    );
    assert!(matches!(
        VoxelTerrainRuntimeError::PersistenceIndeterminate { operation },
        VoxelTerrainRuntimeError::PersistenceIndeterminate {
            operation: requested
        } if requested == operation
    ));
    assert!(matches!(
        VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch {
            requested: operation
        },
        VoxelTerrainRuntimeError::IndeterminatePersistenceMismatch { requested }
            if requested == operation
    ));
}

#[test]
fn repeated_before_io_panics_stop_at_the_automatic_attempt_budget() {
    let stream = Arc::new(PersistencePhaseStream::healthy());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(46, 0, 0),
        lod_index: 0,
    };
    let key = SaveKey::new(location.position, location.lod_index);
    core.shutdown_in_progress = true;
    let operation = core
        .try_enqueue_data_save(BlockToSave {
            voxels: Some(VoxelBuffer::with_size(Vector3i::splat(2))),
            position: location.position,
            lod_index: location.lod_index,
            block_revision: 0,
        })
        .unwrap();
    core.shutdown_in_progress = false;

    for expected_retry_count in 1..=MAX_AUTOMATIC_SAVE_ATTEMPTS {
        core.panic_next_save_before_io_for_test = true;
        core.dispatch_queued_saves_if_allowed();
        core.wait_for_pending_tasks();
        core.try_drain_completed_tasks().unwrap();

        let entry = core.save_journal.get(&key).unwrap();
        let Some(ActiveSaveAttempt::Pending(pending)) = &entry.active else {
            panic!("before-I/O panic must restore the exact pending save");
        };
        assert_eq!(pending.meta.retry_count, expected_retry_count);
        assert!(pending.meta.last_error.is_none());
    }

    let ordinal_after_budget = core.next_persistence_attempt_ordinal;
    core.dispatch_queued_saves_if_allowed();
    core.wait_for_pending_tasks();
    core.try_drain_completed_tasks().unwrap();

    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 0);
    assert_eq!(core.next_persistence_attempt_ordinal, ordinal_after_budget);
    assert_eq!(
        core.journal_persistence_state_for_test(operation),
        Some(JournalPersistenceState::PendingWrite)
    );
    assert!(core.last_save_failures().iter().any(|failure| {
        failure.position_in_blocks == location.position
            && failure.retry_count == MAX_AUTOMATIC_SAVE_ATTEMPTS
            && failure.error.is_none()
    }));
}

#[test]
fn explicit_flush_and_shutdown_fail_fast_on_indeterminate_save() {
    for shutdown in [false, true] {
        let stream = Arc::new(PersistencePhaseStream::panic_save());
        let mut core = build_core_with_stream(stream.clone());
        let location = BlockLocation {
            position: Vector3i::new(49 + i32::from(shutdown), 0, 0),
            lod_index: 0,
        };
        let operation = PersistenceOperation::Save {
            location,
            block_revision: 0,
            save_generation: 410 + u64::from(shutdown),
        };
        let task = stage_c2b_save_attempt(
            &mut core,
            stream.clone(),
            location,
            410 + u64::from(shutdown),
            910 + u64::from(shutdown),
            VoxelBuffer::with_size(Vector3i::splat(2)),
        );
        push_panicked_save_and_drain(&mut core, task);

        let result = if shutdown {
            core.shutdown_and_flush()
        } else {
            core.flush_pending_saves()
        };

        assert_eq!(
            result,
            Err(SaveFlushError::IndeterminatePersistence { operation })
        );
        assert_eq!(stream.save_calls.load(Ordering::SeqCst), 1);
        assert!(!core.shutdown_in_progress);
        assert!(!core.shut_down);
        assert_eq!(
            core.journal_persistence_state_for_test(operation),
            Some(JournalPersistenceState::Indeterminate)
        );
    }
}

#[test]
fn retry_resolution_attempt_ordinal_exhaustion_is_atomic() {
    let stream = Arc::new(PersistencePhaseStream::panic_save());
    let mut core = build_core_with_stream(stream.clone());
    let location = BlockLocation {
        position: Vector3i::new(51, 0, 0),
        lod_index: 0,
    };
    let operation = PersistenceOperation::Save {
        location,
        block_revision: 0,
        save_generation: 412,
    };
    let payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let task = stage_c2b_save_attempt(&mut core, stream.clone(), location, 412, 912, payload);
    push_panicked_save_and_drain(&mut core, task);
    core.next_persistence_attempt_ordinal = u64::MAX;
    let quarantine_len = core.completion_quarantine.len();
    let pending_tasks = core.pending_task_count();

    assert_eq!(
        core.try_resolve_indeterminate_persistence(
            operation,
            IndeterminateIoResolution::AssumeNotWrittenAndRetry,
        ),
        Err(VoxelTerrainRuntimeError::PersistenceAttemptOverflow { operation })
    );
    assert_eq!(core.next_persistence_attempt_ordinal, u64::MAX);
    assert_eq!(core.completion_quarantine.len(), quarantine_len);
    assert_eq!(core.pending_task_count(), pending_tasks);
    assert_eq!(stream.save_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        core.journal_persistence_state_for_test(operation),
        Some(JournalPersistenceState::Indeterminate)
    );
    assert_eq!(
        core.quarantined_save_payload_ptr_for_test(operation),
        Some(payload_ptr)
    );
}

#[test]
fn ordinary_dispatch_attempt_ordinal_exhaustion_is_typed_lossless_and_repeatable() {
    for shutdown in [false, true] {
        let stream = Arc::new(PersistencePhaseStream::healthy());
        let mut core = build_core_with_stream(stream.clone());
        let location = BlockLocation {
            position: Vector3i::new(57 + i32::from(shutdown), 0, 0),
            lod_index: 0,
        };
        let operation = PersistenceOperation::Save {
            location,
            block_revision: 0,
            save_generation: 1,
        };
        let payload = VoxelBuffer::with_size(Vector3i::splat(2));
        let payload_ptr = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
        core.next_persistence_attempt_ordinal = u64::MAX;
        core.shutdown_in_progress = true;
        core.enqueue_data_save(BlockToSave {
            voxels: Some(payload),
            position: location.position,
            lod_index: location.lod_index,
            block_revision: 0,
        });
        core.shutdown_in_progress = false;

        for _ in 0..3 {
            assert!(matches!(
                core.try_process(&[]),
                Err(VoxelTerrainRuntimeError::PersistenceAttemptOverflow {
                    operation: actual,
                }) if actual == operation
            ));
        }
        assert_eq!(stream.save_calls.load(Ordering::SeqCst), 0);
        assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 0);
        assert_eq!(core.pending_task_count(), 0);
        assert_eq!(
            core.journal_persistence_state_for_test(operation),
            Some(JournalPersistenceState::PendingWrite)
        );
        assert_eq!(
            core.journal_payload_ptr_for_test(operation),
            Some(payload_ptr)
        );

        let expected = Err(SaveFlushError::SaveAdmission {
            error: VoxelTerrainRuntimeError::PersistenceAttemptOverflow { operation },
        });
        for _ in 0..2 {
            let result = if shutdown {
                core.shutdown_and_flush()
            } else {
                core.flush_pending_saves()
            };
            assert_eq!(result, expected);
            assert_eq!(stream.save_calls.load(Ordering::SeqCst), 0);
            assert_eq!(stream.flush_calls.load(Ordering::SeqCst), 0);
            assert_eq!(core.pending_task_count(), 0);
            assert_eq!(core.next_persistence_attempt_ordinal, u64::MAX);
            assert!(!core.shutdown_in_progress);
            assert!(!core.shut_down);
            assert_eq!(
                core.journal_persistence_state_for_test(operation),
                Some(JournalPersistenceState::PendingWrite)
            );
            assert_eq!(
                core.journal_payload_ptr_for_test(operation),
                Some(payload_ptr)
            );
            let entry = core
                .save_journal
                .get(&SaveKey::new(location.position, location.lod_index))
                .unwrap();
            let Some(ActiveSaveAttempt::Pending(pending)) = &entry.active else {
                panic!("attempt exhaustion must leave the save pending");
            };
            assert_eq!(pending.meta.generation, 1);
            assert_eq!(pending.meta.retry_count, 0);
            assert!(pending.meta.last_error.is_none());
            assert!(entry.queued_newer.is_empty());
        }
    }
}

#[test]
fn production_enqueue_retains_generation_overflow_without_panicking() {
    let mut core = build_core();
    core.next_save_generation = u64::MAX;
    let save = BlockToSave {
        voxels: Some(VoxelBuffer::with_size(Vector3i::splat(2))),
        position: Vector3i::new(52, 0, 0),
        lod_index: 0,
        block_revision: 0,
    };
    let payload_ptr = save
        .voxels
        .as_ref()
        .unwrap()
        .channel_bytes(ChannelId::Type.index())
        .as_ptr();

    core.enqueue_data_save(save);

    let retained = core.retained_save_admission_failures.front().unwrap();
    assert_eq!(
        retained.error,
        VoxelTerrainRuntimeError::SaveGenerationOverflow
    );
    assert_eq!(
        retained
            .save
            .voxels
            .as_ref()
            .unwrap()
            .channel_bytes(ChannelId::Type.index())
            .as_ptr(),
        payload_ptr
    );
    assert!(core.save_journal.is_empty());
    assert_eq!(core.next_save_generation, u64::MAX);
    assert_eq!(
        core.flush_pending_saves(),
        Err(SaveFlushError::SaveAdmission {
            error: VoxelTerrainRuntimeError::SaveGenerationOverflow,
        })
    );
    assert!(!core.shutdown_in_progress);
    assert_eq!(core.retained_save_admission_failures.len(), 1);
}

#[test]
fn production_enqueue_rejects_and_retains_absent_payload_without_panicking() {
    let mut core = build_core();
    let save = BlockToSave {
        voxels: None,
        position: Vector3i::new(53, 0, 0),
        lod_index: 0,
        block_revision: 0,
    };

    core.enqueue_data_save(save);

    let retained = core.retained_save_admission_failures.front().unwrap();
    assert_eq!(retained.error, VoxelTerrainRuntimeError::MissingSavePayload);
    assert!(retained.save.voxels.is_none());
    assert!(core.save_journal.is_empty());
    assert_eq!(core.next_save_generation, 1);
    assert_eq!(core.pending_task_count(), 0);
    assert_eq!(
        core.flush_pending_saves(),
        Err(SaveFlushError::SaveAdmission {
            error: VoxelTerrainRuntimeError::MissingSavePayload,
        })
    );
    assert!(!core.shutdown_in_progress);
    assert_eq!(core.retained_save_admission_failures.len(), 1);
}

#[test]
fn legacy_task_admission_failure_retains_exact_owner_until_retry() {
    let mut core = make_edit_core_with_lods(2);
    let runs = Arc::new(AtomicUsize::new(0));
    let task: Box<dyn ThreadedTask> = Box::new(CountingCompletionFollowUpTask {
        runs: Arc::clone(&runs),
    });
    let task_ptr = task.as_ref() as *const dyn ThreadedTask as *const () as usize;

    core.task_runner
        .fail_next_prepared_batch_reservation_for_test();
    core.legacy_link_or_retain_task_batch(vec![ScheduledTask::new(task, TaskLane::Serial)]);

    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert_eq!(core.legacy_task_admission_retry.len(), 1);
    assert_eq!(
        core.legacy_task_admission_retry[0].task() as *const dyn ThreadedTask as *const () as usize,
        task_ptr
    );
    assert!(core.task_runner.observable_tasks_for_test().is_empty());
    assert!(core.completion_quarantine.is_empty());

    core.legacy_link_or_retain_task_batch(Vec::new());
    assert!(core.legacy_task_admission_retry.is_empty());
    core.task_runner.wait_for_all_tasks();

    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert!(core
        .task_runner
        .observable_tasks_for_test()
        .iter()
        .any(|observable| observable.task_ptr == task_ptr));
    assert!(core.completion_quarantine.is_empty());
}

fn enqueue_two_same_key_saves_for_c2b(
    core: &mut VoxelTerrainCore,
    location: BlockLocation,
    old_payload: VoxelBuffer,
    newer_payload: VoxelBuffer,
) -> (PersistenceOperation, PersistenceOperation) {
    core.shutdown_in_progress = true;
    let old = core
        .try_enqueue_data_save(BlockToSave {
            voxels: Some(old_payload),
            position: location.position,
            lod_index: location.lod_index,
            block_revision: 0,
        })
        .unwrap();
    let newer = core
        .try_enqueue_data_save(BlockToSave {
            voxels: Some(newer_payload),
            position: location.position,
            lod_index: location.lod_index,
            block_revision: 1,
        })
        .unwrap();
    core.shutdown_in_progress = false;
    (old, newer)
}

#[test]
fn old_before_io_recovery_retains_old_and_newer_generations_and_pointers() {
    let stream = Arc::new(PersistencePhaseStream::healthy());
    let mut core = build_core_with_stream(stream);
    let location = BlockLocation {
        position: Vector3i::new(54, 0, 0),
        lod_index: 0,
    };
    let old_payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let old_ptr = old_payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    let newer_payload = VoxelBuffer::with_size(Vector3i::splat(3));
    let newer_ptr = newer_payload
        .channel_bytes(ChannelId::Type.index())
        .as_ptr();
    let (old, newer) =
        enqueue_two_same_key_saves_for_c2b(&mut core, location, old_payload, newer_payload);

    core.panic_next_save_before_io_for_test = true;
    core.dispatch_queued_save(SaveKey::new(location.position, location.lod_index));
    core.wait_for_pending_tasks();
    core.try_drain_completed_tasks().unwrap();

    assert_eq!(
        core.journal_persistence_state_for_test(old),
        Some(JournalPersistenceState::PendingWrite)
    );
    assert_eq!(
        core.journal_persistence_state_for_test(newer),
        Some(JournalPersistenceState::PendingWrite)
    );
    assert_eq!(core.journal_payload_ptr_for_test(old), Some(old_ptr));
    assert_eq!(core.journal_payload_ptr_for_test(newer), Some(newer_ptr));
}

#[test]
fn old_call_entered_blocks_newer_and_both_resolutions_preserve_order() {
    for resolution in [
        IndeterminateIoResolution::AssumeNotWrittenAndRetry,
        IndeterminateIoResolution::AssumeWrittenAndFlush,
    ] {
        let panic_stream = Arc::new(PersistencePhaseStream::panic_save());
        let healthy_stream = Arc::new(PersistencePhaseStream::healthy());
        let mut core = build_core_with_stream(panic_stream.clone());
        let location = BlockLocation {
            position: Vector3i::new(
                55 + i32::from(matches!(
                    resolution,
                    IndeterminateIoResolution::AssumeWrittenAndFlush
                )),
                0,
                0,
            ),
            lod_index: 0,
        };
        let old_payload = VoxelBuffer::with_size(Vector3i::splat(2));
        let old_ptr = old_payload.channel_bytes(ChannelId::Type.index()).as_ptr();
        let newer_payload = VoxelBuffer::with_size(Vector3i::splat(3));
        let newer_ptr = newer_payload
            .channel_bytes(ChannelId::Type.index())
            .as_ptr();
        let (old, newer) =
            enqueue_two_same_key_saves_for_c2b(&mut core, location, old_payload, newer_payload);

        core.dispatch_queued_save(SaveKey::new(location.position, location.lod_index));
        core.wait_for_pending_tasks();
        core.try_drain_completed_tasks().unwrap();
        core.dispatch_queued_saves_if_allowed();

        assert_eq!(panic_stream.save_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            core.journal_persistence_state_for_test(old),
            Some(JournalPersistenceState::Indeterminate)
        );
        assert_eq!(
            core.journal_persistence_state_for_test(newer),
            Some(JournalPersistenceState::PendingWrite)
        );
        assert_eq!(
            core.quarantined_save_payload_ptr_for_test(old),
            Some(old_ptr)
        );
        assert_eq!(core.journal_payload_ptr_for_test(newer), Some(newer_ptr));

        core.stream = healthy_stream.clone();
        core.try_resolve_indeterminate_persistence(old, resolution)
            .unwrap();
        core.wait_for_pending_tasks();
        core.try_drain_completed_tasks().unwrap();

        assert_eq!(
            core.journal_persistence_state_for_test(old),
            if resolution == IndeterminateIoResolution::AssumeNotWrittenAndRetry {
                Some(JournalPersistenceState::WrittenUnflushed)
            } else {
                None
            }
        );
        assert_eq!(
            core.journal_persistence_state_for_test(newer),
            Some(JournalPersistenceState::PendingWrite)
        );
        assert_eq!(
            core.journal_payload_ptr_for_test(old),
            (resolution == IndeterminateIoResolution::AssumeNotWrittenAndRetry).then_some(old_ptr)
        );
        assert_eq!(core.journal_payload_ptr_for_test(newer), Some(newer_ptr));
        assert_eq!(
            healthy_stream.save_calls.load(Ordering::SeqCst),
            usize::from(matches!(
                resolution,
                IndeterminateIoResolution::AssumeNotWrittenAndRetry
            ))
        );

        core.flush_pending_saves().unwrap();
        assert_eq!(core.journal_persistence_state_for_test(old), None);
        assert_eq!(core.journal_persistence_state_for_test(newer), None);
        assert_eq!(
            healthy_stream.save_calls.load(Ordering::SeqCst),
            1 + usize::from(matches!(
                resolution,
                IndeterminateIoResolution::AssumeNotWrittenAndRetry
            ))
        );
    }
}

#[test]
fn box_difference_returns_removing_slabs() {
    // Subtraction of a centred inner box from an outer box yields the
    // 6 surrounding slabs. Verify the helper used by paging diffs.
    let outer = Box3i::new(Vector3i::zero(), Vector3i::splat(4));
    let inner = Box3i::new(Vector3i::splat(1), Vector3i::splat(2));
    let slabs = outer.difference(inner);
    let cells_in_slabs: i64 = slabs
        .iter()
        .map(|b| (b.size.x as i64) * (b.size.y as i64) * (b.size.z as i64))
        .sum();
    // 4^3 - 2^3 = 56 cells outside the inner box.
    assert_eq!(cells_in_slabs, 64 - 8);

    // A non-overlapping subtraction returns the whole outer box.
    let disjoint = Box3i::new(Vector3i::splat(10), Vector3i::splat(1));
    assert_eq!(outer.difference(disjoint).len(), 1);
}

#[test]
fn compute_viewer_boxes_pads_data_for_meshing_neighbours() {
    let mut state = ViewerState {
        local_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: 16,
        vertical_view_distance_voxels: 16,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
        ..ViewerState::default()
    };
    compute_viewer_boxes(&mut state, 16, 16);
    // ceil(16/16) = 1 block "radius"; from_center_extents produces a box
    // of size 2*1 = 2 per axis (center +/- 1, exclusive max).
    assert_eq!(state.mesh_box.size, Vector3i::splat(2));
    // Data box adds 1 block of padding for meshing neighbours (factor 1).
    assert!(state.data_box.size.x >= state.mesh_box.size.x);
}

// ---- M2.1 step 3: multi-LOD VoxelTerrainCore ----

#[test]
fn multi_lod_terrain_creates_correct_lod_count() {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-512), Vector3i::splat(2048)));
    let mesher = Arc::new(crate::meshers::TransvoxelMesher::new());
    let dep = MeshingDependency::new(mesher, None);
    let core = VoxelTerrainCore::legacy_variable_lod_for_parity(
        data,
        Arc::new(MemoryStream::new()),
        dep,
        3,
    );
    assert_eq!(core.lod_count(), 3);
    // Each LOD has its own mesh map.
    assert_eq!(core.mesh_blocks_at_lod(0).len(), 0);
    assert_eq!(core.mesh_blocks_at_lod(1).len(), 0);
    assert_eq!(core.mesh_blocks_at_lod(2).len(), 0);
}

#[test]
fn single_lod_terrain_backward_compat() {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-512), Vector3i::splat(2048)));
    let mesher = Arc::new(crate::meshers::TransvoxelMesher::new());
    let dep = MeshingDependency::new(mesher, None);
    let core = VoxelTerrainCore::new_generator_only(data, dep);
    // Single-LOD: lod_count == 1, behaves identically to pre-M2.
    assert_eq!(core.lod_count(), 1);
    assert_eq!(core.mesh_blocks().len(), 0);
}

#[test]
fn multi_lod_terrain_loads_blocks_at_both_lod_levels() {
    // End-to-end: a 2-LOD terrain with a viewer should produce mesh blocks
    // at both LOD 0 (fine) and LOD 1 (coarse, larger blocks).
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-512), Vector3i::splat(2048)));
    let generator: Arc<Flat> = Arc::new(Flat::default());
    data.set_generator(Some(generator.clone()));
    let mesher = Arc::new(crate::meshers::TransvoxelMesher::new());
    let dep = MeshingDependency::new(mesher, Some(generator));
    let settings = crate::terrain::lod_clipbox::LodClipboxSettings {
        data_block_size: 16,
        mesh_block_size: 16,
        lod_count: 2,
        lod0_distance_voxels: 16,
        secondary_distance_voxels: 16,
        unload_hysteresis_blocks: 2,
    };
    let mut core =
        VoxelTerrainCore::new_variable_lod(data, Arc::new(MemoryStream::new()), dep, settings)
            .expect("variable LOD terrain constructs");
    assert_eq!(core.lod_count(), 2);

    // Viewer at origin with a small view distance.
    let viewers = vec![ViewerUpdate {
        id: 0,
        world_position_voxels: Vector3i::zero(),
        horizontal_view_distance_voxels: 48,
        vertical_view_distance_voxels: 48,
        demand: MeshDemand {
            visuals: true,
            collisions: true,
        },
    }];
    // Run several process ticks to let paging converge.
    for _ in 0..20 {
        core.try_process(&viewers).unwrap();
        core.wait_for_pending_tasks();
    }
    // Both LOD levels should have at least some mesh blocks.
    let lod0_count = core.mesh_blocks_at_lod(0).len();
    let lod1_count = core.mesh_blocks_at_lod(1).len();
    assert!(
        lod0_count > 0,
        "LOD 0 should have mesh blocks, got {lod0_count}"
    );
    assert!(
        lod1_count > 0,
        "LOD 1 should have mesh blocks, got {lod1_count}"
    );
}

fn pooled_mesh_output(
    pool: Arc<MeshArraysPool>,
    key: MeshBlockKey,
    features: MeshBuildFeatures,
    visual_triangles: usize,
    collision_triangles: usize,
    dropped: bool,
) -> BlockMeshOutput {
    let mut output = MesherOutput::default();
    if features.visuals {
        let mut arrays = pool.acquire();
        arrays
            .indices
            .extend(std::iter::repeat_n(0, visual_triangles * 3));
        output
            .surfaces
            .push(Surface::new(SurfaceArrays::Transvoxel(arrays), 0));
    }
    if features.collisions && collision_triangles != 0 {
        output
            .collision_surface
            .positions
            .push(crate::math::Vector3f::zero());
        output
            .collision_surface
            .indices
            .extend(std::iter::repeat_n(0, collision_triangles * 3));
    }
    BlockMeshOutput::new(key, features, output, pool, dropped)
}

fn mesh_event_upload(event: &VoxelTerrainEvent) -> &Arc<MeshUploadSnapshot> {
    match event {
        VoxelTerrainEvent::MeshBlockEntered(upload)
        | VoxelTerrainEvent::MeshBlockUpdated(upload)
        | VoxelTerrainEvent::MeshBlockBecameEmpty(upload) => upload,
        other => panic!("expected mesh upload event, got {other:?}"),
    }
}

fn prepare_direct_mesh(core: &mut VoxelTerrainCore, position: Vector3i) -> MeshBlockKey {
    if !core.mesh_maps[0].contains_key(&position) {
        core.legacy_view_mesh_block(position, 0);
    }
    core.request_mesh_update(position, 0).unwrap()
}

#[test]
fn accept_topology_and_exit_in_one_process_keeps_event_payload_alive() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let key = prepare_direct_mesh(&mut core, position);
    let pool = Arc::new(MeshArraysPool::new());
    let features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };

    core.try_apply_mesh_output(pooled_mesh_output(pool.clone(), key, features, 1, 0, false))
        .unwrap();
    let paired_state = ViewerState {
        mesh_box: single_block_box(position),
        demand: MeshDemand {
            visuals: true,
            collisions: false,
        },
        ..ViewerState::default()
    };
    core.paired_viewers.push(PairedViewer {
        id: 7,
        state: paired_state.clone(),
        prev_state: paired_state,
    });
    core.try_fixed_viewer_transaction_for_test(&[]).unwrap();
    let events = core.try_drain_completed_tasks().unwrap();

    assert!(core.mesh_blocks_at_lod(0).get(&position).is_none());
    assert_eq!(events.len(), 4);
    assert_eq!(
        events[0].mesh_descriptor().unwrap(),
        MeshLifecycleEventDescriptor {
            kind: MeshLifecycleEventKind::Entered,
            key,
            features,
            visual_state: PayloadState::NonEmpty,
            collision_state: PayloadState::NotBuilt,
        }
    );
    assert!(matches!(
        events[1],
        VoxelTerrainEvent::RenderTopologyChanged(ref batch)
            if batch.groups.iter().all(|group| group.operation == TopologyOperation::RootActivate)
    ));
    assert!(matches!(
        events[2],
        VoxelTerrainEvent::RenderTopologyChanged(ref batch)
            if batch.groups.iter().all(|group| group.operation == TopologyOperation::RootDeactivate)
    ));
    assert!(matches!(
        events[3],
        VoxelTerrainEvent::MeshBlockExited(location)
            if location == MeshBlockLocation::new(position, 0)
    ));
    assert_eq!(
        mesh_event_upload(&events[0])
            .output()
            .total_triangle_count(),
        1
    );
    assert_eq!(pool.idle_count(), 0);
    drop(events);
    assert_eq!(pool.idle_count(), 1);
}

#[test]
fn two_accepts_before_drain_keep_two_exact_payload_snapshots() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let pool = Arc::new(MeshArraysPool::new());
    let features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };
    let first_key = prepare_direct_mesh(&mut core, position);
    core.try_apply_mesh_output(pooled_mesh_output(
        pool.clone(),
        first_key,
        features,
        1,
        0,
        false,
    ))
    .unwrap();
    let second_key = prepare_direct_mesh(&mut core, position);
    core.try_apply_mesh_output(pooled_mesh_output(pool, second_key, features, 2, 0, false))
        .unwrap();

    let events = core.try_drain_completed_tasks().unwrap();
    let mesh_events = events
        .iter()
        .filter(|event| event.mesh_descriptor().is_some())
        .collect::<Vec<_>>();
    assert_eq!(mesh_events.len(), 2);
    let first = mesh_event_upload(mesh_events[0]);
    let second = mesh_event_upload(mesh_events[1]);
    let live = core.mesh_blocks_at_lod(0)[&position]
        .accepted_upload()
        .unwrap();

    for (event, kind, key) in [
        (mesh_events[0], MeshLifecycleEventKind::Entered, first_key),
        (mesh_events[1], MeshLifecycleEventKind::Updated, second_key),
    ] {
        assert_eq!(
            event.mesh_descriptor().unwrap(),
            MeshLifecycleEventDescriptor {
                kind,
                key,
                features,
                visual_state: PayloadState::NonEmpty,
                collision_state: PayloadState::NotBuilt,
            }
        );
    }
    assert_eq!(first.output().total_triangle_count(), 1);
    assert_eq!(second.output().total_triangle_count(), 2);
    assert!(!Arc::ptr_eq(first, second));
    assert!(Arc::ptr_eq(second, live));
}

#[test]
fn mesh_array_pool_returns_only_after_last_accepted_or_event_arc_drops() {
    let features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };
    let position = Vector3i::zero();
    let pool = Arc::new(MeshArraysPool::new());
    pool.release(Default::default());
    pool.release(Default::default());
    let mut core = build_core();

    let first_key = prepare_direct_mesh(&mut core, position);
    core.try_apply_mesh_output(pooled_mesh_output(
        pool.clone(),
        first_key,
        features,
        1,
        0,
        false,
    ))
    .unwrap();
    let second_key = prepare_direct_mesh(&mut core, position);
    core.try_apply_mesh_output(pooled_mesh_output(
        pool.clone(),
        second_key,
        features,
        1,
        0,
        false,
    ))
    .unwrap();
    assert_eq!(pool.idle_count(), 0);

    let mut events = core.try_drain_completed_tasks().unwrap();
    drop(events.remove(0));
    assert_eq!(pool.idle_count(), 1, "replacement waits for first event");
    core.legacy_unview_mesh_block(position, 0);
    assert_eq!(pool.idle_count(), 1, "unload waits for second event");
    drop(events);
    assert_eq!(pool.idle_count(), 2);

    let stale_pool = Arc::new(MeshArraysPool::new());
    let stale_key = prepare_direct_mesh(&mut core, position);
    let _newer_key = core.request_mesh_update(position, 0).unwrap();
    core.try_apply_mesh_output(pooled_mesh_output(
        stale_pool.clone(),
        stale_key,
        features,
        1,
        0,
        false,
    ))
    .unwrap();
    assert_eq!(stale_pool.idle_count(), 1, "stale upload releases by RAII");

    let dropped_pool = Arc::new(MeshArraysPool::new());
    let dropped_key = core.request_mesh_update(position, 0).unwrap();
    core.try_apply_mesh_output(pooled_mesh_output(
        dropped_pool.clone(),
        dropped_key,
        features,
        1,
        0,
        true,
    ))
    .unwrap();
    assert_eq!(
        dropped_pool.idle_count(),
        1,
        "dropped upload releases by RAII"
    );

    let unpublished_pool = Arc::new(MeshArraysPool::new());
    let unpublished_key = core.request_mesh_update(position, 0).unwrap();
    core.fail_next_direct_mesh_reservation_for_test = true;
    let returned = match core.try_apply_mesh_output(pooled_mesh_output(
        unpublished_pool.clone(),
        unpublished_key,
        features,
        1,
        0,
        false,
    )) {
        Err(MeshOutputApplyError::NotAdmitted { output, .. }) => output,
        _ => panic!("expected pre-admission failure"),
    };
    assert_eq!(unpublished_pool.idle_count(), 0);
    drop(returned);
    assert_eq!(
        unpublished_pool.idle_count(),
        1,
        "unpublished output returns its directly owned arrays"
    );
}

#[test]
fn legacy_partial_direct_apply_event_fifo() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let pool = Arc::new(MeshArraysPool::new());
    let features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };
    let first = prepare_direct_mesh(&mut core, position);
    core.try_apply_mesh_output(pooled_mesh_output(
        pool.clone(),
        first,
        features,
        1,
        0,
        false,
    ))
    .unwrap();
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(DebugNameCollisionTask),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));
    core.fail_next_completion_normalization_for_test = true;

    assert!(core.try_process(&[]).is_err());
    assert_eq!(core.event_outbox.len(), 2);
    let retry_events = core.try_drain_completed_tasks().unwrap();
    assert_eq!(retry_events[0].mesh_descriptor().unwrap().key, first);
    assert!(matches!(
        retry_events[1],
        VoxelTerrainEvent::RenderTopologyChanged(_)
    ));

    let second = core.request_mesh_update(position, 0).unwrap();
    core.try_apply_mesh_output(pooled_mesh_output(
        pool.clone(),
        second,
        features,
        2,
        0,
        false,
    ))
    .unwrap();
    let third = core.request_mesh_update(position, 0).unwrap();
    core.try_apply_mesh_output(pooled_mesh_output(pool, third, features, 3, 0, false))
        .unwrap();
    let ordered = core.try_drain_completed_tasks().unwrap();
    assert_eq!(ordered.len(), 2);
    assert_eq!(ordered[0].mesh_descriptor().unwrap().key, second);
    assert_eq!(ordered[1].mesh_descriptor().unwrap().key, third);
    assert_eq!(
        mesh_event_upload(&ordered[0])
            .output()
            .total_triangle_count(),
        2
    );
    assert_eq!(
        mesh_event_upload(&ordered[1])
            .output()
            .total_triangle_count(),
        3
    );
}

#[test]
fn direct_mesh_pre_and_post_admission_failures_preserve_exact_ownership() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };
    let pre_pool = Arc::new(MeshArraysPool::new());
    let pre_key = prepare_direct_mesh(&mut core, position);
    let output = pooled_mesh_output(pre_pool.clone(), pre_key, features, 1, 0, false);
    let pool_identity = Arc::as_ptr(output.pool());
    let payload_identity = output.output().surfaces.as_ptr();
    core.fail_next_direct_mesh_reservation_for_test = true;

    let output = match core.try_apply_mesh_output(output) {
        Err(MeshOutputApplyError::NotAdmitted { output, .. }) => output,
        _ => panic!("expected exact output to be returned before admission"),
    };
    assert_eq!(Arc::as_ptr(output.pool()), pool_identity);
    assert_eq!(output.output().surfaces.as_ptr(), payload_identity);
    assert!(core.direct_mesh_retry_inbox.is_empty());
    assert!(core.event_outbox.is_empty());
    assert_eq!(core.mesh_blocks_at_lod(0)[&position].applied_revision, None);
    core.try_apply_mesh_output(output).unwrap();

    let post_pool = Arc::new(MeshArraysPool::new());
    let post_key = core.request_mesh_update(position, 0).unwrap();
    core.fail_next_mesh_event_reservation_for_test = true;
    assert!(matches!(
        core.try_apply_mesh_output(pooled_mesh_output(
            post_pool.clone(),
            post_key,
            features,
            2,
            0,
            false,
        )),
        Err(MeshOutputApplyError::Admitted { .. })
    ));
    assert_eq!(
        post_pool.idle_count(),
        0,
        "admitted retry ownership keeps its arrays outstanding"
    );
    assert_eq!(core.direct_mesh_retry_inbox.len(), 1);
    assert_eq!(
        core.event_outbox.len(),
        2,
        "earlier accepted event is untouched"
    );
    assert_eq!(
        core.mesh_blocks_at_lod(0)[&position].applied_revision,
        Some(pre_key.revision)
    );

    let events = core.try_drain_completed_tasks().unwrap();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].mesh_descriptor().unwrap().key, pre_key);
    assert!(matches!(
        events[1],
        VoxelTerrainEvent::RenderTopologyChanged(_)
    ));
    assert_eq!(events[2].mesh_descriptor().unwrap().key, post_key);
    assert!(core.direct_mesh_retry_inbox.is_empty());
    let live = core.mesh_blocks_at_lod(0)[&position]
        .accepted_upload()
        .unwrap();
    assert!(Arc::ptr_eq(live, mesh_event_upload(&events[2])));
    drop(events);
    assert_eq!(post_pool.idle_count(), 0, "live entry retains the upload");
    core.legacy_unview_mesh_block(position, 0);
    assert_eq!(post_pool.idle_count(), 1);
}

#[test]
fn mesh_payload_descriptors_classify_features_independently() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let pool = Arc::new(MeshArraysPool::new());
    let cases = [
        (
            MeshBuildFeatures {
                visuals: true,
                collisions: false,
                variable_lod: false,
            },
            1,
            0,
            PayloadState::NonEmpty,
            PayloadState::NotBuilt,
            MeshLifecycleEventKind::Entered,
        ),
        (
            MeshBuildFeatures {
                visuals: false,
                collisions: true,
                variable_lod: false,
            },
            0,
            1,
            PayloadState::NotBuilt,
            PayloadState::NonEmpty,
            MeshLifecycleEventKind::Updated,
        ),
        (
            MeshBuildFeatures {
                visuals: true,
                collisions: true,
                variable_lod: true,
            },
            0,
            0,
            PayloadState::Empty,
            PayloadState::Empty,
            MeshLifecycleEventKind::BecameEmpty,
        ),
        (
            MeshBuildFeatures::default(),
            0,
            0,
            PayloadState::NotBuilt,
            PayloadState::NotBuilt,
            MeshLifecycleEventKind::BecameEmpty,
        ),
    ];

    for (features, visuals, collisions, visual_state, collision_state, kind) in cases {
        let key = prepare_direct_mesh(&mut core, position);
        core.try_apply_mesh_output(pooled_mesh_output(
            pool.clone(),
            key,
            features,
            visuals,
            collisions,
            false,
        ))
        .unwrap();
        let event = core.try_drain_completed_tasks().unwrap().remove(0);
        assert_eq!(
            event.mesh_descriptor().unwrap(),
            MeshLifecycleEventDescriptor {
                kind,
                key,
                features,
                visual_state,
                collision_state,
            }
        );
    }
}

fn fixed_zero_distance_viewer(
    id: ViewerId,
    block_position: Vector3i,
    demand: MeshDemand,
) -> ViewerUpdate {
    ViewerUpdate {
        id,
        world_position_voxels: block_position * 16,
        // `Box3i::from_center_extents` treats zero extents as an empty box.
        // One voxel therefore gives the smallest non-empty fixed-LOD
        // residency box (two cells per axis).
        horizontal_view_distance_voxels: 1,
        vertical_view_distance_voxels: 1,
        demand,
    }
}

type FixedLoadingRollbackObservation = (Vector3i, u32, u64, LoadRequestState);
type FixedMeshRollbackObservation = (Vector3i, u32, u32, u32, Option<u64>);

#[derive(Debug, PartialEq, Eq)]
struct FixedRollbackDescriptor {
    viewer_ids: Vec<ViewerId>,
    loading: Vec<FixedLoadingRollbackObservation>,
    meshes: Vec<FixedMeshRollbackObservation>,
    pending_load: Vec<Vector3i>,
    pending_mesh: Vec<Vector3i>,
    next_request_generation: u64,
    next_mesh_revision: u64,
    next_render_topology_revision: u64,
    stats: VoxelTerrainStats,
    event_count: usize,
}

fn fixed_rollback_descriptor(core: &VoxelTerrainCore) -> FixedRollbackDescriptor {
    let mut loading = core.loading_blocks[0]
        .iter()
        .map(|(position, entry)| {
            (
                *position,
                entry.residency.resident_viewers,
                entry.request_generation,
                entry.request_state,
            )
        })
        .collect::<Vec<_>>();
    loading.sort_unstable_by_key(|entry| (entry.0.z, entry.0.x, entry.0.y));
    let mut meshes = core.mesh_maps[0]
        .iter()
        .map(|(position, entry)| {
            (
                *position,
                entry.resident_viewers,
                entry.visual_viewers,
                entry.collision_viewers,
                entry.requested_revision,
            )
        })
        .collect::<Vec<_>>();
    meshes.sort_unstable_by_key(|entry| (entry.0.z, entry.0.x, entry.0.y));
    FixedRollbackDescriptor {
        viewer_ids: core.paired_viewers.iter().map(|viewer| viewer.id).collect(),
        loading,
        meshes,
        pending_load: core.blocks_pending_load[0].clone(),
        pending_mesh: core.blocks_pending_update[0].clone(),
        next_request_generation: core.next_request_generation,
        next_mesh_revision: core.next_mesh_revision,
        next_render_topology_revision: core.next_render_topology_revision,
        stats: core.stats,
        event_count: core.event_outbox.len(),
    }
}

#[test]
fn release_safe_data_residency_mismatch_is_typed_and_non_mutating() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let location = BlockLocation {
        position,
        lod_index: 0,
    };
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
    voxels.set_voxel(0xC2, 0, 0, 0, ChannelId::Type.index());
    let mut block = VoxelDataBlock::with_voxels(voxels, 0);
    block.viewers.set_exact(1);
    assert!(core.data.try_set_block(position, block).unwrap());
    core.loaded_data_residency[0].insert(position, DataResidencyRefs::default());
    let paired_state = ViewerState {
        data_box: single_block_box(position),
        ..ViewerState::default()
    };
    core.paired_viewers.push(PairedViewer {
        id: 7,
        state: paired_state.clone(),
        prev_state: paired_state,
    });
    let before = core.data.block_snapshot(position, 0).unwrap();
    let before_revision = core.data.key_revision(position, 0);
    let before_events = core.event_outbox.len();

    assert_eq!(
        core.try_fixed_viewer_transaction_for_test(&[]),
        Err(VoxelTerrainRuntimeError::DataResidencyMismatch {
            location,
            tracked_resident_viewers: Some(0),
            tracked_coverage_holds: Some(0),
            storage_viewers: 1,
        })
    );

    let after = core.data.block_snapshot(position, 0).unwrap();
    assert_eq!(after.viewers.get(), before.viewers.get());
    assert_eq!(
        after.voxels().get_voxel(0, 0, 0, ChannelId::Type.index()),
        before.voxels().get_voxel(0, 0, 0, ChannelId::Type.index())
    );
    assert_eq!(core.data.key_revision(position, 0), before_revision);
    assert_eq!(core.event_outbox.len(), before_events);
    assert_eq!(
        core.loaded_data_residency[0].get(&position),
        Some(&DataResidencyRefs::default())
    );
    assert_eq!(core.paired_viewers.len(), 1);
}

#[test]
fn fixed_zero_net_viewer_replacement_still_validates_touched_residency() {
    let mut core = build_core();
    let mut replacement = fixed_zero_distance_viewer(
        2,
        Vector3i::zero(),
        MeshDemand {
            visuals: false,
            collisions: false,
        },
    );
    let mut state = ViewerState {
        local_position_voxels: replacement.world_position_voxels,
        horizontal_view_distance_voxels: replacement.horizontal_view_distance_voxels,
        vertical_view_distance_voxels: replacement.vertical_view_distance_voxels,
        demand: replacement.demand,
        ..ViewerState::default()
    };
    compute_viewer_boxes(&mut state, core.data_block_size(), core.data_block_size());
    let position = state
        .data_box
        .iter_cells_zxy()
        .next()
        .expect("the minimal viewer owns a data block");
    let mut block = VoxelDataBlock::with_voxels(
        VoxelBuffer::with_size(Vector3i::splat(core.data_block_size())),
        0,
    );
    block.viewers.set_exact(1);
    assert!(core.data.try_set_block(position, block).unwrap());
    core.loaded_data_residency[0].insert(position, DataResidencyRefs::default());
    core.paired_viewers.push(PairedViewer {
        id: 1,
        state: state.clone(),
        prev_state: ViewerState::default(),
    });
    replacement.id = 2;

    assert!(matches!(
        core.try_fixed_viewer_transaction_for_test(&[replacement]),
        Err(VoxelTerrainRuntimeError::DataResidencyMismatch { location, .. })
            if location == BlockLocation { position, lod_index: 0 }
    ));
    assert_eq!(
        core.data.block_snapshot(position, 0).unwrap().viewers.get(),
        1
    );
    assert_eq!(
        core.loaded_data_residency[0].get(&position),
        Some(&DataResidencyRefs::default())
    );
    assert_eq!(core.paired_viewers[0].id, 1);
}

#[test]
fn fixed_zero_net_viewer_replacement_rejects_missing_storage_with_stale_sidecar() {
    let mut core = build_core();
    let mut replacement = fixed_zero_distance_viewer(
        2,
        Vector3i::zero(),
        MeshDemand {
            visuals: false,
            collisions: false,
        },
    );
    let mut state = ViewerState {
        local_position_voxels: replacement.world_position_voxels,
        horizontal_view_distance_voxels: replacement.horizontal_view_distance_voxels,
        vertical_view_distance_voxels: replacement.vertical_view_distance_voxels,
        demand: replacement.demand,
        ..ViewerState::default()
    };
    compute_viewer_boxes(&mut state, core.data_block_size(), core.data_block_size());
    let position = state.data_box.iter_cells_zxy().next().unwrap();
    core.loaded_data_residency[0].insert(position, DataResidencyRefs::with_resident_viewers(1));
    core.paired_viewers.push(PairedViewer {
        id: 1,
        state: state.clone(),
        prev_state: ViewerState::default(),
    });
    replacement.id = 2;

    assert!(matches!(
        core.try_fixed_viewer_transaction_for_test(&[replacement]),
        Err(VoxelTerrainRuntimeError::DataResidencyMismatch { location, .. })
            if location == BlockLocation { position, lod_index: 0 }
    ));
    assert!(core.data.block_snapshot(position, 0).is_none());
    assert_eq!(
        core.loaded_data_residency[0][&position],
        DataResidencyRefs::with_resident_viewers(1)
    );
    assert!(!core.loading_blocks[0].contains_key(&position));
    assert_eq!(core.paired_viewers[0].id, 1);
}

#[test]
fn fixed_zero_net_viewer_replacement_rejects_resident_loading_overlap() {
    let mut core = build_core();
    let mut replacement = fixed_zero_distance_viewer(
        2,
        Vector3i::zero(),
        MeshDemand {
            visuals: false,
            collisions: false,
        },
    );
    let mut state = ViewerState {
        local_position_voxels: replacement.world_position_voxels,
        horizontal_view_distance_voxels: replacement.horizontal_view_distance_voxels,
        vertical_view_distance_voxels: replacement.vertical_view_distance_voxels,
        demand: replacement.demand,
        ..ViewerState::default()
    };
    compute_viewer_boxes(&mut state, core.data_block_size(), core.data_block_size());
    let position = state.data_box.iter_cells_zxy().next().unwrap();
    let mut block = VoxelDataBlock::empty(0);
    block.viewers.set_exact(1);
    assert!(core.data.try_set_block(position, block).unwrap());
    core.loaded_data_residency[0].insert(position, DataResidencyRefs::with_resident_viewers(1));
    core.loading_blocks[0].insert(
        position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: 7,
            request_state: LoadRequestState::InFlight,
            physical_request: None,
        },
    );
    core.paired_viewers.push(PairedViewer {
        id: 1,
        state: state.clone(),
        prev_state: ViewerState::default(),
    });
    replacement.id = 2;

    assert!(matches!(
        core.try_fixed_viewer_transaction_for_test(&[replacement]),
        Err(VoxelTerrainRuntimeError::DataResidencyMismatch { location, .. })
            if location == BlockLocation { position, lod_index: 0 }
    ));
    assert_eq!(
        core.data.block_snapshot(position, 0).unwrap().viewers.get(),
        1
    );
    assert_eq!(core.loading_blocks[0][&position].request_generation, 7);
    assert_eq!(core.paired_viewers[0].id, 1);
}

#[test]
fn variable_requests_own_exact_physical_tags_before_dispatch() {
    let mut core = make_edit_core_with_lods(2);
    let load_position = Vector3i::new(8, 0, 0);
    core.apply_data_view(single_block_box(load_position), 1);
    let load = &core.loading_blocks[1][&load_position];
    assert_eq!(
        load.physical_request.as_ref().map(|request| request.tag),
        Some(TaskRequestTag::new(
            core.request_epoch,
            load.request_generation
        ))
    );

    let mesh_position = Vector3i::new(3, 0, 0);
    core.mesh_maps[1].insert(
        mesh_position,
        MeshBlockEntry {
            position: mesh_position,
            resident_viewers: 1,
            visual_viewers: 1,
            ..MeshBlockEntry::default()
        },
    );
    let key = core.request_mesh_update(mesh_position, 1).unwrap();
    let mesh = &core.mesh_maps[1][&mesh_position];
    assert_eq!(mesh.requested_revision, Some(key.revision));
    assert_eq!(
        mesh.physical_request.as_ref().map(|request| request.tag),
        Some(TaskRequestTag::new(
            core.request_epoch,
            mesh.request_generation
        ))
    );
}

#[test]
fn variable_mesh_view_schedule_overflow_is_atomic_and_typed() {
    for overflow_mesh_revision in [true, false] {
        let mut core = make_edit_core_with_lods(2);
        let position = Vector3i::new(21, 0, 0);
        let location = MeshBlockLocation::new(position, 1);
        for data_position in core.meshing_data_box(location).unwrap().iter_cells_zxy() {
            assert!(core
                .data
                .try_set_block(data_position, VoxelDataBlock::empty(1))
                .unwrap());
            core.loaded_data_residency[1].insert(data_position, DataResidencyRefs::default());
        }
        let next_mesh_revision = if overflow_mesh_revision { u64::MAX } else { 31 };
        let next_request_generation = if overflow_mesh_revision { 41 } else { u64::MAX };
        core.next_mesh_revision = next_mesh_revision;
        core.next_request_generation = next_request_generation;

        let result = core.try_legacy_view_mesh_block(position, 1);
        if overflow_mesh_revision {
            assert_eq!(result, Err(VoxelTerrainRuntimeError::MeshRevisionOverflow));
        } else {
            assert_eq!(
                result,
                Err(VoxelTerrainRuntimeError::RequestGenerationOverflow)
            );
        }
        assert!(!core.mesh_maps[1].contains_key(&position));
        assert!(core.blocks_pending_update[1].is_empty());
        assert_eq!(core.next_mesh_revision, next_mesh_revision);
        assert_eq!(core.next_request_generation, next_request_generation);
    }
}

#[test]
fn variable_load_mesh_schedule_overflow_retains_exact_completion_and_demand() {
    for overflow_mesh_revision in [true, false] {
        let mut core = make_edit_core_with_lods(2);
        core.automatic_loading_enabled = false;
        let position = Vector3i::new(22, 0, 0);
        core.apply_data_view(single_block_box(position), 1);
        let request = core.loading_blocks[1][&position]
            .physical_request
            .as_ref()
            .unwrap()
            .clone();
        core.loading_blocks[1]
            .get_mut(&position)
            .unwrap()
            .request_state = LoadRequestState::InFlight;
        core.blocks_pending_load[1].clear();
        core.mesh_maps[1].insert(
            position,
            MeshBlockEntry {
                position,
                resident_viewers: 1,
                visual_viewers: 1,
                ..MeshBlockEntry::default()
            },
        );
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
        voxels.set_voxel(0xF3, 0, 0, 0, ChannelId::Type.index());
        let mut task = LoadBlockForTerrainTask::new(
            position,
            1,
            request.tag.request_generation,
            core.data.clone(),
            core.stream.clone(),
        )
        .with_request_control(request.tag, request.cancellation.clone());
        task.output = Some(TerrainLoadOutput::new_tagged(
            BlockDataOutput::loaded(position, 1, voxels, false),
            request.tag,
        ));
        core.raw_completion_inbox.push_back(CompletedTask::new(
            Box::new(task),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            Vec::new(),
        ));
        core.try_normalize_raw_completions().unwrap();
        let descriptor = core.durable_completion_inbox[0].descriptor();
        let DurableCompletion::LoadFinished { output, .. } = &core.durable_completion_inbox[0]
        else {
            panic!("expected exact variable load completion")
        };
        let payload_identity =
            voxel_allocation_identity(output.block_data.voxels.as_ref().unwrap());
        if overflow_mesh_revision {
            core.next_mesh_revision = u64::MAX;
        } else {
            core.next_mesh_revision = 51;
            core.next_request_generation = u64::MAX;
        }

        let result = core.legacy_variable_apply_durable_fifo();
        if overflow_mesh_revision {
            assert_eq!(result, Err(VoxelTerrainRuntimeError::MeshRevisionOverflow));
        } else {
            assert_eq!(
                result,
                Err(VoxelTerrainRuntimeError::RequestGenerationOverflow)
            );
        }
        assert_eq!(core.durable_completion_inbox.len(), 1);
        assert_eq!(core.durable_completion_inbox[0].descriptor(), descriptor);
        let DurableCompletion::LoadFinished { output, .. } = &core.durable_completion_inbox[0]
        else {
            panic!("overflow changed the completion variant")
        };
        assert_eq!(
            voxel_allocation_identity(output.block_data.voxels.as_ref().unwrap()),
            payload_identity
        );
        assert!(core.data.block_snapshot(position, 1).is_none());
        assert_eq!(
            core.loading_blocks[1][&position].request_generation,
            request.tag.request_generation
        );
        let mesh = &core.mesh_maps[1][&position];
        assert_eq!(mesh.requested_revision, None);
        assert!(mesh.physical_request.is_none());
        assert!(!mesh.is_in_update_list);
        assert!(core.blocks_pending_update[1].is_empty());
        assert!(core.event_outbox.is_empty());

        core.next_mesh_revision = 300;
        core.next_request_generation = 400;
        core.legacy_variable_apply_durable_fifo().unwrap();
        core.data.with_lod_map(1, |map| {
            assert_eq!(
                voxel_allocation_identity(map.get_block(position).unwrap().voxels()),
                payload_identity
            );
        });
        let mesh = &core.mesh_maps[1][&position];
        assert_eq!(mesh.requested_revision, Some(300));
        assert_eq!(mesh.request_generation, 400);
        assert!(mesh.physical_request.is_some());
        assert!(mesh.is_in_update_list);
        assert_eq!(core.blocks_pending_update[1], vec![position]);
        assert!(core.durable_completion_inbox.is_empty());
    }
}

#[test]
fn variable_load_terminal_generation_overflow_retains_exact_completion() {
    let mut core = make_edit_core_with_lods(2);
    core.automatic_loading_enabled = false;
    let position = Vector3i::new(13, 0, 0);
    core.apply_data_view(single_block_box(position), 1);
    let generation = core.loading_blocks[1][&position].request_generation;
    core.loading_blocks[1]
        .get_mut(&position)
        .unwrap()
        .request_state = LoadRequestState::InFlight;
    core.blocks_pending_load[1].clear();
    let request_tag = core.loading_blocks[1][&position]
        .physical_request
        .as_ref()
        .unwrap()
        .tag;
    let task = tagged_current_load_task(&core, position, 1, generation);
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Parallel,
        TaskCompletionStatus::Panicked(crate::tasks::TaskPanicPhase::Run),
        Vec::new(),
    ));
    core.try_normalize_raw_completions().unwrap();
    let descriptor = core.durable_completion_inbox[0].descriptor();
    core.next_request_generation = u64::MAX;

    assert_eq!(
        core.legacy_variable_apply_durable_fifo(),
        Err(VoxelTerrainRuntimeError::RequestGenerationOverflow)
    );
    assert_eq!(core.durable_completion_inbox.len(), 1);
    assert_eq!(core.durable_completion_inbox[0].descriptor(), descriptor);
    let entry = &core.loading_blocks[1][&position];
    assert_eq!(entry.retry_count, 0);
    assert_eq!(entry.request_generation, generation);
    assert_eq!(entry.request_state, LoadRequestState::InFlight);
    assert_eq!(entry.physical_request.as_ref().unwrap().tag, request_tag);
    assert!(core.blocks_pending_load[1].is_empty());

    core.next_request_generation = 100;
    core.legacy_variable_apply_durable_fifo().unwrap();
    let entry = &core.loading_blocks[1][&position];
    assert_eq!(entry.retry_count, 1);
    assert_eq!(entry.request_generation, 100);
    assert_eq!(entry.request_state, LoadRequestState::Queued);
    assert_eq!(core.blocks_pending_load[1], vec![position]);
    assert!(core.durable_completion_inbox.is_empty());
}

#[test]
fn variable_mesh_terminal_generation_overflow_retains_exact_completion() {
    let mut core = make_edit_core_with_lods(2);
    core.automatic_loading_enabled = false;
    let position = Vector3i::new(4, 0, 0);
    core.mesh_maps[1].insert(
        position,
        MeshBlockEntry {
            position,
            resident_viewers: 1,
            visual_viewers: 1,
            ..MeshBlockEntry::default()
        },
    );
    let key = core.request_mesh_update(position, 1).unwrap();
    core.mesh_maps[1]
        .get_mut(&position)
        .unwrap()
        .is_in_update_list = false;
    core.blocks_pending_update[1].clear();
    let request = core.mesh_maps[1][&position]
        .physical_request
        .as_ref()
        .unwrap()
        .clone();
    let task = MeshBlockTask::new(MeshBlockTaskParams {
        key,
        data: core.data.clone(),
        meshing_dependency: core.meshing_dependency.clone(),
        collision_hint: false,
        lod_hint: true,
        mesh_arrays_pool: Some(core.mesh_arrays_pool.clone()),
    })
    .with_request_control(request.tag, request.cancellation.clone());
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Parallel,
        TaskCompletionStatus::Cancelled,
        Vec::new(),
    ));
    core.try_normalize_raw_completions().unwrap();
    let descriptor = core.durable_completion_inbox[0].descriptor();
    core.next_request_generation = u64::MAX;

    assert_eq!(
        core.legacy_variable_apply_durable_fifo(),
        Err(VoxelTerrainRuntimeError::RequestGenerationOverflow)
    );
    assert_eq!(core.durable_completion_inbox.len(), 1);
    assert_eq!(core.durable_completion_inbox[0].descriptor(), descriptor);
    let entry = &core.mesh_maps[1][&position];
    assert_eq!(entry.terminal_retry_count, 0);
    assert_eq!(entry.request_generation, request.tag.request_generation);
    assert_eq!(entry.physical_request.as_ref().unwrap().tag, request.tag);
    assert!(!entry.is_in_update_list);
    assert!(core.blocks_pending_update[1].is_empty());

    core.next_request_generation = 200;
    core.legacy_variable_apply_durable_fifo().unwrap();
    let entry = &core.mesh_maps[1][&position];
    assert_eq!(entry.terminal_retry_count, 1);
    assert_eq!(entry.request_generation, 200);
    assert!(entry.is_in_update_list);
    assert_eq!(core.blocks_pending_update[1], vec![position]);
    assert!(core.durable_completion_inbox.is_empty());
}

#[test]
fn variable_old_epoch_finished_load_retires_without_rearm_or_publication() {
    let mut core = make_edit_core_with_lods(2);
    core.automatic_loading_enabled = false;
    let position = Vector3i::new(9, 0, 0);
    core.apply_data_view(single_block_box(position), 1);
    let request = core.loading_blocks[1][&position]
        .physical_request
        .as_ref()
        .expect("variable load owns a physical request")
        .clone();
    core.loading_blocks[1]
        .get_mut(&position)
        .unwrap()
        .request_state = LoadRequestState::InFlight;
    core.blocks_pending_load[1].clear();
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
    voxels.set_voxel(0xE4, 0, 0, 0, ChannelId::Type.index());
    let mut task = LoadBlockForTerrainTask::new(
        position,
        1,
        request.tag.request_generation,
        core.data.clone(),
        core.stream.clone(),
    )
    .with_request_control(request.tag, request.cancellation.clone());
    task.output = Some(TerrainLoadOutput::new_tagged(
        BlockDataOutput::loaded(position, 1, voxels, false),
        request.tag,
    ));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        Vec::new(),
    ));
    core.begin_shutdown_attempt().unwrap();
    core.try_normalize_raw_completions().unwrap();

    core.legacy_variable_apply_durable_fifo().unwrap();

    assert!(core.data.block_snapshot(position, 1).is_none());
    let loading = &core.loading_blocks[1][&position];
    assert_eq!(loading.request_generation, request.tag.request_generation);
    assert_eq!(loading.request_state, LoadRequestState::InFlight);
    assert!(core.blocks_pending_load[1].is_empty());
    assert!(core.event_outbox.is_empty());
    assert!(core.durable_completion_inbox.is_empty());
}

#[test]
fn variable_residency_mismatch_is_typed_and_non_mutating() {
    let mut core = make_edit_core_with_lods(2);
    let position = Vector3i::new(4, 0, 0);
    let mut block = VoxelDataBlock::empty(1);
    block.viewers.set_exact(1);
    assert!(core.data.try_set_block(position, block).unwrap());
    core.loaded_data_residency[1].insert(position, DataResidencyRefs::default());

    assert!(matches!(
        core.try_apply_variable_data_residency_delta(single_block_box(position), 1, 1),
        Err(VoxelTerrainRuntimeError::DataResidencyMismatch { location, .. })
            if location == BlockLocation { position, lod_index: 1 }
    ));
    assert_eq!(
        core.data.block_snapshot(position, 1).unwrap().viewers.get(),
        1
    );
    assert_eq!(
        core.loaded_data_residency[1].get(&position),
        Some(&DataResidencyRefs::default())
    );
}

#[test]
fn variable_residency_overflow_and_underflow_are_non_mutating() {
    let mut core = make_edit_core_with_lods(2);
    let loaded = Vector3i::new(5, 0, 0);
    let missing = Vector3i::new(6, 0, 0);
    let mut block = VoxelDataBlock::empty(1);
    block.viewers.set_exact(u32::MAX);
    assert!(core.data.try_set_block(loaded, block).unwrap());
    core.loaded_data_residency[1]
        .insert(loaded, DataResidencyRefs::with_resident_viewers(u32::MAX));

    assert!(matches!(
        core.try_apply_variable_data_residency_delta(single_block_box(loaded), 1, 1),
        Err(VoxelTerrainRuntimeError::DataRefcountOverflow {
            location,
            field: DataRefField::ResidentViewers,
        }) if location == BlockLocation { position: loaded, lod_index: 1 }
    ));
    assert_eq!(
        core.data.block_snapshot(loaded, 1).unwrap().viewers.get(),
        u32::MAX
    );
    assert_eq!(
        core.loaded_data_residency[1][&loaded].resident_viewers,
        u32::MAX
    );

    assert!(matches!(
        core.try_apply_variable_data_residency_delta(single_block_box(missing), 1, -1),
        Err(VoxelTerrainRuntimeError::DataRefcountUnderflow {
            location,
            field: DataRefField::ResidentViewers,
        }) if location == BlockLocation { position: missing, lod_index: 1 }
    ));
    assert!(!core.loading_blocks[1].contains_key(&missing));
    assert!(core.blocks_pending_load[1].is_empty());
}

#[test]
fn variable_residency_storage_commit_failure_rolls_back_sidecars() {
    let mut core = make_edit_core_with_lods(2);
    let position = Vector3i::new(7, 0, 0);
    let mut block = VoxelDataBlock::empty(1);
    block.viewers.set_exact(1);
    assert!(core.data.try_set_block(position, block).unwrap());
    core.loaded_data_residency[1].insert(position, DataResidencyRefs::with_resident_viewers(1));
    core.data
        .set_test_transaction_live_spatial_registry_fail_lod(Some(1));

    assert!(matches!(
        core.try_apply_variable_data_residency_delta(single_block_box(position), 1, 1),
        Err(VoxelTerrainRuntimeError::DataMutation(_))
    ));
    core.data
        .set_test_transaction_live_spatial_registry_fail_lod(None);
    assert_eq!(
        core.data.block_snapshot(position, 1).unwrap().viewers.get(),
        1
    );
    assert_eq!(
        core.loaded_data_residency[1][&position],
        DataResidencyRefs::with_resident_viewers(1)
    );
    assert!(core.blocks_pending_load[1].is_empty());
}

#[test]
fn variable_dirty_final_unview_publishes_exact_save_owner_before_fence_release() {
    let mut core = make_edit_core_with_lods(2);
    core.shutdown_in_progress = true;
    let position = Vector3i::new(17, 0, 0);
    let location = BlockLocation {
        position,
        lod_index: 1,
    };
    let mut block = VoxelDataBlock::with_voxels(
        VoxelBuffer::with_size(Vector3i::splat(core.data_block_size())),
        1,
    );
    block
        .voxels_mut()
        .set_voxel(0xD4, 0, 0, 0, ChannelId::Type.index());
    block.viewers.set_exact(1);
    block.set_modified(true);
    assert!(core.data.try_set_block(position, block).unwrap());
    core.loaded_data_residency[1].insert(position, DataResidencyRefs::with_resident_viewers(1));
    let resident_identity = core.data.with_lod_map(1, |map| {
        voxel_allocation_identity(map.get_block(position).unwrap().voxels())
    });
    let owner = core.install_fixed_dirty_owner_probe_for_test(location);
    let pause = core.install_fixed_commit_pause_for_test(
        FixedCommitPausePhase::AfterTerrainPublishBeforeFenceFinish,
    );
    let data = Arc::clone(&core.data);
    let core = Arc::new(Mutex::new(core));
    let unview_core = Arc::clone(&core);
    let unview = thread::spawn(move || {
        unview_core
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .try_apply_variable_data_residency_delta(single_block_box(position), 1, -1)
    });

    pause.wait_until_reached();
    assert!(pause.commit_marker().load(Ordering::SeqCst));
    assert!(data.try_lod_map_read(1).is_none());
    assert_eq!(owner.load(Ordering::SeqCst), resident_identity);
    pause.release();
    unview.join().unwrap().unwrap();

    let core = core.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    assert!(core.data.block_snapshot(position, 1).is_none());
    assert!(!core.loaded_data_residency[1].contains_key(&position));
    let operation = PersistenceOperation::Save {
        location,
        block_revision: 1,
        save_generation: 1,
    };
    assert_eq!(
        core.journal_payload_ptr_for_test(operation),
        Some(resident_identity as *const u8)
    );
    assert_eq!(
        core.journal_persistence_state_for_test(operation),
        Some(JournalPersistenceState::PendingWrite)
    );
    assert!(core.event_outbox.iter().any(
        |event| matches!(event, VoxelTerrainEvent::DataBlockUnloaded(actual) if *actual == location)
    ));
}

#[test]
fn variable_dirty_final_unview_late_c1_failure_retains_resident_owner() {
    let mut core = make_edit_core_with_lods(2);
    let position = Vector3i::new(18, 0, 0);
    let mut block = VoxelDataBlock::with_voxels(
        VoxelBuffer::with_size(Vector3i::splat(core.data_block_size())),
        1,
    );
    block.viewers.set_exact(1);
    block.set_modified(true);
    assert!(core.data.try_set_block(position, block).unwrap());
    core.loaded_data_residency[1].insert(position, DataResidencyRefs::with_resident_viewers(1));
    let resident_identity = core.data.with_lod_map(1, |map| {
        voxel_allocation_identity(map.get_block(position).unwrap().voxels())
    });
    let revision = core.data.key_revision(position, 1);
    core.data
        .set_test_transaction_live_spatial_registry_fail_lod(Some(1));

    assert!(matches!(
        core.try_apply_variable_data_residency_delta(single_block_box(position), 1, -1),
        Err(VoxelTerrainRuntimeError::DataMutation(_))
    ));

    core.data
        .set_test_transaction_live_spatial_registry_fail_lod(None);
    core.data.with_lod_map(1, |map| {
        let resident = map.get_block(position).unwrap();
        assert_eq!(
            voxel_allocation_identity(resident.voxels()),
            resident_identity
        );
        assert_eq!(resident.viewers.get(), 1);
        assert!(resident.is_modified());
    });
    assert_eq!(core.data.key_revision(position, 1), revision);
    assert_eq!(
        core.loaded_data_residency[1][&position],
        DataResidencyRefs::with_resident_viewers(1)
    );
    assert_eq!(core.next_save_generation, 1);
    assert!(core.save_journal.is_empty());
    assert!(core.event_outbox.is_empty());
}

#[test]
fn variable_resident_and_loading_overlap_is_typed_and_non_mutating() {
    let mut core = make_edit_core_with_lods(2);
    let position = Vector3i::new(19, 0, 0);
    let mut block = VoxelDataBlock::empty(1);
    block.viewers.set_exact(1);
    assert!(core.data.try_set_block(position, block).unwrap());
    core.loaded_data_residency[1].insert(position, DataResidencyRefs::with_resident_viewers(1));
    core.loading_blocks[1].insert(
        position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: 41,
            request_state: LoadRequestState::InFlight,
            physical_request: Some(PhysicalRequest::new(TaskRequestTag::new(
                core.request_epoch,
                41,
            ))),
        },
    );

    assert!(matches!(
        core.try_apply_variable_data_residency_delta(single_block_box(position), 1, 1),
        Err(VoxelTerrainRuntimeError::DataResidencyMismatch { location, .. })
            if location == BlockLocation { position, lod_index: 1 }
    ));
    assert_eq!(
        core.data.block_snapshot(position, 1).unwrap().viewers.get(),
        1
    );
    assert_eq!(
        core.loaded_data_residency[1][&position],
        DataResidencyRefs::with_resident_viewers(1)
    );
    assert_eq!(core.loading_blocks[1][&position].request_generation, 41);
}

#[test]
fn variable_mesh_view_refcount_failures_are_checked_and_non_mutating() {
    let mut core = make_edit_core_with_lods(2);
    let overflow_position = Vector3i::new(8, 0, 0);
    core.mesh_maps[1].insert(
        overflow_position,
        MeshBlockEntry {
            position: overflow_position,
            resident_viewers: u32::MAX,
            visual_viewers: u32::MAX,
            ..MeshBlockEntry::default()
        },
    );
    assert!(matches!(
        core.try_legacy_view_mesh_block(overflow_position, 1),
        Err(VoxelTerrainRuntimeError::MeshRefcountOverflow {
            location,
            field: MeshRefField::ResidentViewers,
        }) if location == MeshBlockLocation::new(overflow_position, 1)
    ));
    assert_eq!(
        (
            core.mesh_maps[1][&overflow_position].resident_viewers,
            core.mesh_maps[1][&overflow_position].visual_viewers,
        ),
        (u32::MAX, u32::MAX)
    );

    let underflow_position = Vector3i::new(9, 0, 0);
    core.mesh_maps[1].insert(
        underflow_position,
        MeshBlockEntry {
            position: underflow_position,
            resident_viewers: 1,
            visual_viewers: 0,
            ..MeshBlockEntry::default()
        },
    );
    assert!(matches!(
        core.try_legacy_unview_mesh_block(underflow_position, 1),
        Err(VoxelTerrainRuntimeError::MeshRefcountUnderflow {
            location,
            field: MeshRefField::VisualViewers,
        }) if location == MeshBlockLocation::new(underflow_position, 1)
    ));
    assert_eq!(
        (
            core.mesh_maps[1][&underflow_position].resident_viewers,
            core.mesh_maps[1][&underflow_position].visual_viewers,
        ),
        (1, 0)
    );
}

#[test]
fn shutdown_permit_mismatch_preserves_completion_payload_and_followup() {
    let mut core = build_core();
    let position = Vector3i::new(10, 0, 0);
    core.apply_data_view(single_block_box(position), 0);
    let generation = core.loading_blocks[0][&position].request_generation;
    core.loading_blocks[0]
        .get_mut(&position)
        .unwrap()
        .request_state = LoadRequestState::InFlight;
    core.blocks_pending_load[0].clear();
    let mut task = tagged_current_load_task(&core, position, 0, generation);
    task.output = Some(loaded_output(&core, position, generation, 0xB7));
    let followup: Box<dyn ThreadedTask> = Box::new(CompletionFollowUpTask {
        ran: Arc::new(AtomicBool::new(false)),
    });
    let followup_identity = followup.as_ref() as *const dyn ThreadedTask as *const () as usize;
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(task),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        vec![ScheduledTask::new(followup, TaskLane::Serial)],
    ));
    core.try_normalize_raw_completions().unwrap();
    let descriptor = core.durable_completion_inbox[0].descriptor();
    let DurableCompletion::LoadFinished {
        output, completed, ..
    } = &core.durable_completion_inbox[0]
    else {
        panic!("expected exact load owner")
    };
    let payload_identity = voxel_allocation_identity(output.block_data.voxels.as_ref().unwrap());
    assert_eq!(completed.follow_up_count(), 1);

    core.begin_shutdown_attempt().unwrap();
    let wrong_core = build_core();
    core.shutdown_mutation_permit = Some(wrong_core.data.close_mutation_admission_for_shutdown());
    assert_eq!(
        core.prepare_fixed_viewer_transaction_with_checkpoint_and_admission(
            &[],
            false,
            false,
            false,
            None,
            None,
            None,
            true,
            true,
        ),
        Err(VoxelTerrainRuntimeError::DataMutation(
            SharedVoxelDataMutationError::ShutdownMutationPermitMismatch,
        ))
    );

    assert_eq!(core.durable_completion_inbox[0].descriptor(), descriptor);
    let DurableCompletion::LoadFinished {
        output, completed, ..
    } = &core.durable_completion_inbox[0]
    else {
        panic!("permit failure changed completion variant")
    };
    assert_eq!(
        voxel_allocation_identity(output.block_data.voxels.as_ref().unwrap()),
        payload_identity
    );
    assert_eq!(completed.follow_up_count(), 1);
    assert_eq!(
        completed.follow_up_task(0).unwrap().task() as *const dyn ThreadedTask as *const ()
            as usize,
        followup_identity
    );
    assert_eq!(
        core.loading_blocks[0][&position].request_generation,
        generation
    );
}

#[test]
fn variable_load_crossing_begin_shutdown_is_retired_without_rearm() {
    let gate = Arc::new(BlockingWorkerGate::default());
    let stream: Arc<dyn VoxelStream> = Arc::new(BlockingNotFoundStream {
        gate: Arc::clone(&gate),
    });
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048)));
    let mut core = VoxelTerrainCore::legacy_variable_lod_for_parity(
        data,
        stream,
        MeshingDependency::new(Arc::new(AlwaysOneTriangleMesher), None),
        2,
    );
    let position = Vector3i::new(11, 0, 0);
    core.apply_data_view(single_block_box(position), 1);
    core.send_data_load_requests();
    gate.wait_until_entered();
    let generation = core.loading_blocks[1][&position].request_generation;

    core.begin_shutdown_attempt().unwrap();
    gate.release();
    core.wait_for_pending_tasks();
    core.task_runner
        .try_drain_completed_into(&mut core.raw_completion_inbox)
        .unwrap();
    core.try_normalize_raw_completions().unwrap();
    core.legacy_variable_apply_durable_fifo().unwrap();

    let entry = &core.loading_blocks[1][&position];
    assert_eq!(entry.request_generation, generation);
    assert_eq!(entry.request_state, LoadRequestState::InFlight);
    assert!(core.blocks_pending_load[1].is_empty());
    assert!(core.data.block_snapshot(position, 1).is_none());
    assert!(core.event_outbox.is_empty());
}

#[test]
fn variable_mesh_crossing_begin_shutdown_is_retired_without_rearm() {
    let gate = Arc::new(BlockingWorkerGate::default());
    let mesher: Arc<dyn VoxelMesher> = Arc::new(BlockingMesher {
        gate: Arc::clone(&gate),
    });
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048)));
    let mut core = VoxelTerrainCore::legacy_variable_lod_for_parity(
        data,
        Arc::new(MemoryStream::new()),
        MeshingDependency::new(mesher, None),
        2,
    );
    let position = Vector3i::new(1, 0, 0);
    let mesh_location = MeshBlockLocation::new(position, 1);
    for data_position in core
        .meshing_data_box(mesh_location)
        .unwrap()
        .iter_cells_zxy()
    {
        assert!(core
            .data
            .try_set_block(data_position, VoxelDataBlock::empty(1))
            .unwrap());
        core.loaded_data_residency[1].insert(data_position, DataResidencyRefs::default());
    }
    core.mesh_maps[1].insert(
        position,
        MeshBlockEntry {
            position,
            resident_viewers: 1,
            visual_viewers: 1,
            ..MeshBlockEntry::default()
        },
    );
    let key = core.request_mesh_update(position, 1).unwrap();
    core.process_meshing().unwrap();
    gate.wait_until_entered();

    core.begin_shutdown_attempt().unwrap();
    gate.release();
    core.wait_for_pending_tasks();
    core.task_runner
        .try_drain_completed_into(&mut core.raw_completion_inbox)
        .unwrap();
    core.try_normalize_raw_completions().unwrap();
    core.legacy_variable_apply_durable_fifo().unwrap();

    let entry = &core.mesh_maps[1][&position];
    assert_eq!(entry.requested_revision, Some(key.revision));
    assert_eq!(entry.applied_revision, None);
    assert!(!entry.is_in_update_list);
    assert!(core.blocks_pending_update[1].is_empty());
    assert!(core.event_outbox.is_empty());
}

#[test]
fn successful_variable_shutdown_retires_all_runtime_request_owners() {
    let mut core = make_edit_core_with_lods(2);
    let load_position = Vector3i::new(12, 0, 0);
    core.apply_data_view(single_block_box(load_position), 1);
    let load_cancel = core.loading_blocks[1][&load_position]
        .physical_request
        .as_ref()
        .unwrap()
        .cancellation
        .clone();
    let mesh_position = Vector3i::new(2, 0, 0);
    core.mesh_maps[1].insert(
        mesh_position,
        MeshBlockEntry {
            position: mesh_position,
            resident_viewers: 1,
            visual_viewers: 1,
            ..MeshBlockEntry::default()
        },
    );
    core.request_mesh_update(mesh_position, 1).unwrap();
    let mesh_cancel = core.mesh_maps[1][&mesh_position]
        .physical_request
        .as_ref()
        .unwrap()
        .cancellation
        .clone();
    let resident_position = Vector3i::new(3, 0, 0);
    let mut resident = VoxelDataBlock::with_voxels(
        VoxelBuffer::with_size(Vector3i::splat(core.data_block_size())),
        1,
    );
    resident.viewers.set_exact(2);
    assert!(core
        .data
        .try_set_block(resident_position, resident)
        .unwrap());
    core.loaded_data_residency[1].insert(
        resident_position,
        DataResidencyRefs {
            resident_viewers: 1,
            coverage_holds: 1,
        },
    );
    let followup: Box<dyn ThreadedTask> = Box::new(CompletionFollowUpTask {
        ran: Arc::new(AtomicBool::new(false)),
    });
    core.completion_quarantine
        .push_back(QuarantinedCompletion::Other {
            kind: CompletionTaskKind::Unknown,
            completed: CompletedTask::new(
                Box::new(DebugNameCollisionTask),
                TaskLane::Parallel,
                TaskCompletionStatus::Cancelled,
                vec![ScheduledTask::new(followup, TaskLane::Serial)],
            ),
        });

    core.shutdown_and_flush().unwrap();

    assert!(load_cancel.is_cancelled());
    assert!(mesh_cancel.is_cancelled());
    assert!(core.loading_blocks.iter().all(HashMap::is_empty));
    assert!(core.mesh_maps.iter().all(HashMap::is_empty));
    assert!(core.blocks_pending_load.iter().all(Vec::is_empty));
    assert!(core.blocks_pending_update.iter().all(Vec::is_empty));
    assert!(core.raw_completion_inbox.is_empty());
    assert!(core.durable_completion_inbox.is_empty());
    assert!(core.direct_mesh_retry_inbox.is_empty());
    assert!(core.legacy_task_admission_retry.is_empty());
    assert!(core.completion_quarantine.is_empty());
    assert!(core.shutdown_mutation_permit.is_none());
    assert_eq!(
        core.data
            .block_snapshot(resident_position, 1)
            .unwrap()
            .viewers
            .get(),
        0
    );
    assert_eq!(
        core.loaded_data_residency[1][&resident_position],
        DataResidencyRefs::default()
    );
}

#[test]
fn common_prepared_map_publication_targets_exact_nonzero_lod() {
    let mut core = make_edit_core_with_lods(2);
    let position = Vector3i::new(13, 0, 0);
    core.mesh_maps[0].insert(
        position,
        MeshBlockEntry {
            position,
            requested_revision: Some(101),
            ..MeshBlockEntry::default()
        },
    );
    core.loading_blocks[0].insert(
        position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: 201,
            request_state: LoadRequestState::Queued,
            physical_request: None,
        },
    );
    core.loaded_data_residency[0].insert(position, DataResidencyRefs::with_resident_viewers(1));

    let mut mesh_diffs = vec![PreparedMeshEntryDiff {
        location: MeshBlockLocation::new(position, 1),
        expected_revision: None,
        action: PreparedMapAction::Insert(MeshBlockEntry {
            position,
            requested_revision: Some(102),
            ..MeshBlockEntry::default()
        }),
    }];
    let mut loading_diffs = vec![PreparedLoadingEntryDiff {
        location: BlockLocation {
            position,
            lod_index: 1,
        },
        expected_generation: None,
        action: PreparedMapAction::Insert(LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(2),
            retry_count: 0,
            request_generation: 202,
            request_state: LoadRequestState::Queued,
            physical_request: None,
        }),
    }];
    let mut data_residency_diffs = vec![PreparedDataResidencyDiff {
        location: BlockLocation {
            position,
            lod_index: 1,
        },
        expected: None,
        action: PreparedMapAction::Insert(DataResidencyRefs::with_resident_viewers(2)),
    }];
    let pending_load_queues = Vec::new();
    let pending_mesh_queues = Vec::new();
    core.try_reserve_prepared_runtime_publication(
        &mesh_diffs,
        &data_residency_diffs,
        &loading_diffs,
        &pending_load_queues,
        &pending_mesh_queues,
    )
    .unwrap();
    let mut retirement = RetirementBag::default();
    retirement.mesh_entries.reserve_exact(mesh_diffs.len());
    retirement
        .loading_entries
        .reserve_exact(loading_diffs.len());

    core.publish_prepared_runtime_diffs_no_fail(
        &mut mesh_diffs,
        &mut data_residency_diffs,
        &mut loading_diffs,
        &mut retirement,
    );

    assert_eq!(core.mesh_maps[0][&position].requested_revision, Some(101));
    assert_eq!(core.loading_blocks[0][&position].request_generation, 201);
    assert_eq!(
        core.loaded_data_residency[0][&position],
        DataResidencyRefs::with_resident_viewers(1)
    );
    assert_eq!(core.mesh_maps[1][&position].requested_revision, Some(102));
    assert_eq!(core.loading_blocks[1][&position].request_generation, 202);
    assert_eq!(
        core.loaded_data_residency[1][&position],
        DataResidencyRefs::with_resident_viewers(2)
    );
}

#[test]
fn common_prepared_queue_publication_replaces_only_declared_lods() {
    let mut core = make_edit_core_with_lods(3);
    let old_loads = [
        Vector3i::new(1, 0, 0),
        Vector3i::new(2, 0, 0),
        Vector3i::new(3, 0, 0),
    ];
    let old_meshes = [
        Vector3i::new(4, 0, 0),
        Vector3i::new(5, 0, 0),
        Vector3i::new(6, 0, 0),
    ];
    for lod in 0..3 {
        core.blocks_pending_load[lod].push(old_loads[lod]);
        core.blocks_pending_update[lod].push(old_meshes[lod]);
    }
    let old_load_tail = Vector3i::new(7, 0, 0);
    let old_mesh_tail = Vector3i::new(8, 0, 0);
    core.blocks_pending_load[1].push(old_load_tail);
    core.blocks_pending_update[2].push(old_mesh_tail);
    let new_load = Vector3i::new(20, 0, 0);
    let new_mesh = Vector3i::new(30, 0, 0);
    let mut pending_load_queues = vec![PreparedQueueDiff {
        lod_index: 1,
        final_values: vec![new_load],
    }];
    let mut pending_mesh_queues = vec![PreparedQueueDiff {
        lod_index: 2,
        final_values: vec![new_mesh],
    }];
    core.try_reserve_prepared_runtime_publication(
        &[],
        &[],
        &[],
        &pending_load_queues,
        &pending_mesh_queues,
    )
    .unwrap();

    core.publish_prepared_queue_diffs_no_fail(&mut pending_load_queues, &mut pending_mesh_queues);

    assert_eq!(core.blocks_pending_load[0], vec![old_loads[0]]);
    assert_eq!(core.blocks_pending_load[1], vec![new_load]);
    assert_eq!(core.blocks_pending_load[2], vec![old_loads[2]]);
    assert_eq!(core.blocks_pending_update[0], vec![old_meshes[0]]);
    assert_eq!(core.blocks_pending_update[1], vec![old_meshes[1]]);
    assert_eq!(core.blocks_pending_update[2], vec![new_mesh]);
    assert_eq!(
        pending_load_queues[0].final_values,
        vec![old_loads[1], old_load_tail]
    );
    assert_eq!(
        pending_mesh_queues[0].final_values,
        vec![old_meshes[2], old_mesh_tail]
    );
}

#[test]
fn common_prepared_publication_rejects_invalid_lod_before_mutation() {
    let mut core = make_edit_core_with_lods(2);
    let position = Vector3i::new(14, 0, 0);
    core.mesh_maps[0].insert(
        position,
        MeshBlockEntry {
            position,
            requested_revision: Some(301),
            ..MeshBlockEntry::default()
        },
    );
    core.blocks_pending_load[0].push(Vector3i::new(40, 0, 0));
    core.blocks_pending_update[1].push(Vector3i::new(41, 0, 0));
    core.next_request_generation = 401;
    core.next_mesh_revision = 402;
    core.event_outbox
        .push_back(VoxelTerrainEvent::MeshBlockExited(MeshBlockLocation::new(
            Vector3i::new(42, 0, 0),
            1,
        )));
    let invalid = vec![PreparedMeshEntryDiff {
        location: MeshBlockLocation::new(position, 2),
        expected_revision: None,
        action: PreparedMapAction::Insert(MeshBlockEntry {
            position,
            requested_revision: Some(302),
            ..MeshBlockEntry::default()
        }),
    }];
    let invalid_loading = vec![PreparedLoadingEntryDiff {
        location: BlockLocation {
            position,
            lod_index: 2,
        },
        expected_generation: None,
        action: PreparedMapAction::Insert(LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: 403,
            request_state: LoadRequestState::Queued,
            physical_request: None,
        }),
    }];
    let invalid_data_residency = vec![PreparedDataResidencyDiff {
        location: BlockLocation {
            position,
            lod_index: 2,
        },
        expected: None,
        action: PreparedMapAction::Insert(DataResidencyRefs::with_resident_viewers(1)),
    }];
    let invalid_load_queue = vec![prepared_queue_diff(2)];
    let invalid_mesh_queue = vec![prepared_queue_diff(2)];
    let before = common_prepared_publication_observation(&core);

    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&invalid, &[], &[], &[], &[]),
        Err(VoxelTerrainRuntimeError::LodMath(
            LodMathError::InvalidLodCount
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&[], &[], &invalid_loading, &[], &[],),
        Err(VoxelTerrainRuntimeError::LodMath(
            LodMathError::InvalidLodCount
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&[], &invalid_data_residency, &[], &[], &[],),
        Err(VoxelTerrainRuntimeError::LodMath(
            LodMathError::InvalidLodCount
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&[], &[], &[], &invalid_load_queue, &[],),
        Err(VoxelTerrainRuntimeError::LodMath(
            LodMathError::InvalidLodCount
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&[], &[], &[], &[], &invalid_mesh_queue,),
        Err(VoxelTerrainRuntimeError::LodMath(
            LodMathError::InvalidLodCount
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
}

#[derive(Debug, PartialEq, Eq)]
struct CommonPreparedPublicationObservation {
    mesh_entries: Vec<(u8, Vector3i, String)>,
    loading_entries: Vec<(u8, Vector3i, String)>,
    data_residency_entries: Vec<(u8, Vector3i, DataResidencyRefs)>,
    pending_load: Vec<Vec<Vector3i>>,
    pending_mesh: Vec<Vec<Vector3i>>,
    data_view_retry_lengths: Vec<usize>,
    data_unview_retry_lengths: Vec<usize>,
    paired_viewers: String,
    next_request_generation: u64,
    next_mesh_revision: u64,
    next_render_topology_revision: u64,
    stats: VoxelTerrainStats,
    events: String,
    raw_completion_count: usize,
    durable_completion_count: usize,
    direct_mesh_retry_count: usize,
    legacy_task_retry_count: usize,
    completion_quarantine_count: usize,
}

fn common_prepared_publication_observation(
    core: &VoxelTerrainCore,
) -> CommonPreparedPublicationObservation {
    let mut mesh_entries = core
        .mesh_maps
        .iter()
        .enumerate()
        .flat_map(|(lod_index, entries)| {
            entries.iter().map(move |(position, entry)| {
                (
                    u8::try_from(lod_index).unwrap(),
                    *position,
                    format!("{entry:?}"),
                )
            })
        })
        .collect::<Vec<_>>();
    mesh_entries
        .sort_unstable_by_key(|(lod, position, _)| (*lod, position.x, position.y, position.z));
    let mut loading_entries = core
        .loading_blocks
        .iter()
        .enumerate()
        .flat_map(|(lod_index, entries)| {
            entries.iter().map(move |(position, entry)| {
                (
                    u8::try_from(lod_index).unwrap(),
                    *position,
                    format!("{entry:?}"),
                )
            })
        })
        .collect::<Vec<_>>();
    loading_entries
        .sort_unstable_by_key(|(lod, position, _)| (*lod, position.x, position.y, position.z));
    let mut data_residency_entries = core
        .loaded_data_residency
        .iter()
        .enumerate()
        .flat_map(|(lod_index, entries)| {
            entries
                .iter()
                .map(move |(position, refs)| (u8::try_from(lod_index).unwrap(), *position, *refs))
        })
        .collect::<Vec<_>>();
    data_residency_entries
        .sort_unstable_by_key(|(lod, position, _)| (*lod, position.x, position.y, position.z));
    CommonPreparedPublicationObservation {
        mesh_entries,
        loading_entries,
        data_residency_entries,
        pending_load: core.blocks_pending_load.clone(),
        pending_mesh: core.blocks_pending_update.clone(),
        data_view_retry_lengths: core.data_view_retries.iter().map(Vec::len).collect(),
        data_unview_retry_lengths: core.data_unview_retries.iter().map(Vec::len).collect(),
        paired_viewers: format!("{:?}", core.paired_viewers),
        next_request_generation: core.next_request_generation,
        next_mesh_revision: core.next_mesh_revision,
        next_render_topology_revision: core.next_render_topology_revision,
        stats: core.stats,
        events: format!("{:?}", core.event_outbox),
        raw_completion_count: core.raw_completion_inbox.len(),
        durable_completion_count: core.durable_completion_inbox.len(),
        direct_mesh_retry_count: core.direct_mesh_retry_inbox.len(),
        legacy_task_retry_count: core.legacy_task_admission_retry.len(),
        completion_quarantine_count: core.completion_quarantine.len(),
    }
}

#[derive(Debug, PartialEq)]
struct VariablePublicationObservation {
    common: CommonPreparedPublicationObservation,
    settings: LodClipboxSettings,
    coordinator: ClipboxCoordinator,
    coordinator_state_identity: usize,
    coverage: VariableLodCoverage,
    coverage_state_identity: usize,
    coverage_holds: CoverageHoldLedger,
    data_view_retries: Vec<Vec<PendingDataMutation>>,
    data_unview_retries: Vec<Vec<PendingDataMutation>>,
    last_data_mutation_error: Option<SharedVoxelDataMutationError>,
    raw_completion_owners: Vec<CompletionOwnerIdentity>,
    durable_completion_owners: Vec<DurableCompletionDescriptor>,
    direct_completion_owners: Vec<DurableCompletionDescriptor>,
    runner_tasks: Vec<crate::tasks::threaded_task_runner::RunnerTaskObservable>,
}

fn variable_publication_observation(core: &VoxelTerrainCore) -> VariablePublicationObservation {
    let runtime = core
        .variable_lod
        .as_ref()
        .expect("variable publication observation requires variable LOD");
    VariablePublicationObservation {
        common: common_prepared_publication_observation(core),
        settings: runtime.settings,
        coordinator: runtime.coordinator.clone(),
        coordinator_state_identity: runtime.coordinator.state_identity_for_test(),
        coverage: runtime.coverage.clone(),
        coverage_state_identity: runtime.coverage.state_identity_for_test(),
        coverage_holds: runtime.coverage_holds.clone(),
        data_view_retries: core.data_view_retries.clone(),
        data_unview_retries: core.data_unview_retries.clone(),
        last_data_mutation_error: core.last_data_mutation_error.clone(),
        raw_completion_owners: core
            .raw_completion_inbox
            .iter()
            .map(completion_owner_identity)
            .collect(),
        durable_completion_owners: core
            .durable_completion_inbox
            .iter()
            .map(DurableCompletion::descriptor)
            .collect(),
        direct_completion_owners: core
            .direct_mesh_retry_inbox
            .iter()
            .map(DurableCompletion::descriptor)
            .collect(),
        runner_tasks: core.task_runner.observable_tasks_for_test(),
    }
}

fn assert_variable_publication_semantically_eq(
    actual: &VariablePublicationObservation,
    expected: &VariablePublicationObservation,
) {
    assert_eq!(actual.common, expected.common);
    assert_eq!(actual.settings, expected.settings);
    assert_eq!(actual.coordinator, expected.coordinator);
    assert_eq!(actual.coverage, expected.coverage);
    assert_eq!(actual.coverage_holds, expected.coverage_holds);
    assert_eq!(actual.data_view_retries, expected.data_view_retries);
    assert_eq!(actual.data_unview_retries, expected.data_unview_retries);
    assert_eq!(
        actual.last_data_mutation_error,
        expected.last_data_mutation_error
    );
    assert_eq!(actual.raw_completion_owners, expected.raw_completion_owners);
    assert_eq!(
        actual.durable_completion_owners,
        expected.durable_completion_owners
    );
    assert_eq!(
        actual.direct_completion_owners,
        expected.direct_completion_owners
    );
    // `runner_tasks` carries per-task pointer identities and a `phase` field
    // that depends on background worker timing. Comparing the raw observable
    // vectors is therefore inherently nondeterministic, in particular when
    // `actual` and `expected` come from independent cores whose task objects
    // live at different addresses. The observable contract we need here is
    // "the same work was scheduled": compare the deterministic per-lane task
    // counts, not the addresses or transient execution phases.
    assert_eq!(
        runner_task_lane_counts(&actual.runner_tasks),
        runner_task_lane_counts(&expected.runner_tasks),
    );
}

fn runner_task_lane_counts(
    tasks: &[crate::tasks::threaded_task_runner::RunnerTaskObservable],
) -> Vec<(crate::tasks::threaded_task::TaskLane, usize)> {
    let mut counts: Vec<(crate::tasks::threaded_task::TaskLane, usize)> = Vec::new();
    for observable in tasks {
        if let Some(entry) = counts.iter_mut().find(|(lane, _)| *lane == observable.lane) {
            entry.1 += 1;
        } else {
            counts.push((observable.lane, 1));
        }
    }
    // `TaskLane` is `Eq` but not `Ord`; order lanes by a stable variant key
    // (Parallel < Serial) so the projection is deterministic regardless of
    // the order tasks were observed in.
    counts.sort_unstable_by(|(lane_a, count_a), (lane_b, count_b)| {
        fn lane_rank(lane: crate::tasks::threaded_task::TaskLane) -> u8 {
            match lane {
                crate::tasks::threaded_task::TaskLane::Parallel => 0,
                crate::tasks::threaded_task::TaskLane::Serial => 1,
            }
        }
        lane_rank(*lane_a)
            .cmp(&lane_rank(*lane_b))
            .then(count_a.cmp(count_b))
    });
    counts
}

fn dormant_variable_core() -> VoxelTerrainCore {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-512), Vector3i::splat(1024)));
    let settings = LodClipboxSettings {
        data_block_size: 16,
        mesh_block_size: 16,
        lod_count: 3,
        lod0_distance_voxels: 16,
        secondary_distance_voxels: 16,
        unload_hysteresis_blocks: 2,
    };
    VoxelTerrainCore::new_variable_lod(
        data,
        Arc::new(MemoryStream::new()),
        MeshingDependency::new(Arc::new(crate::meshers::TransvoxelMesher::new()), None),
        settings,
    )
    .unwrap()
}

#[test]
fn reconfigure_variable_clipboxes_replaces_live_secondary_distance() {
    let mut core = dormant_variable_core();
    let mut settings = core
        .variable_lod_settings()
        .expect("variable core has clipbox settings");
    settings.secondary_distance_voxels = 48;
    core.try_reconfigure_variable_clipboxes(settings)
        .expect("aligned reconfigure succeeds");
    assert_eq!(
        core.variable_lod_settings()
            .expect("settings remain")
            .secondary_distance_voxels,
        48
    );
}

fn dormant_clipbox_viewer(id: ViewerId, position_voxels: Vector3i) -> ClipboxViewerUpdate {
    ClipboxViewerUpdate {
        id,
        position_voxels,
        view_distance_voxels: Vector3i::splat(64),
        demand: MeshDemand {
            visuals: true,
            collisions: false,
        },
    }
}

fn dormant_paired_viewer(update: &ClipboxViewerUpdate) -> PairedViewer {
    let state = ViewerState {
        local_position_voxels: update.position_voxels,
        horizontal_view_distance_voxels: update.view_distance_voxels.x,
        vertical_view_distance_voxels: update.view_distance_voxels.y,
        demand: update.demand,
        ..ViewerState::default()
    };
    PairedViewer {
        id: update.id,
        state,
        prev_state: ViewerState::default(),
    }
}

struct ExistingPendingJoinFixture {
    core: VoxelTerrainCore,
    parent: MeshBlockLocation,
    children: Vec<MeshBlockLocation>,
    halo: Vec<BlockLocation>,
    exit_inputs: Vec<CoverageInput>,
}

fn visual_counts(resident: u32, visuals: u32, visual_splits: u32) -> DemandCounts {
    DemandCounts {
        resident,
        visuals,
        visual_splits,
        ..DemandCounts::default()
    }
}

fn accepted_snapshot(
    revision: u64,
    visuals: bool,
    collisions: bool,
) -> crate::terrain::AcceptedFeatureSnapshot {
    crate::terrain::AcceptedFeatureSnapshot {
        revision,
        visuals,
        collisions,
    }
}

fn seed_existing_ready_data(core: &mut VoxelTerrainCore, location: BlockLocation) {
    let mut block = VoxelDataBlock::with_voxels(
        VoxelBuffer::with_size(Vector3i::splat(core.data_block_size())),
        location.lod_index,
    );
    block.viewers.set_exact(1);
    assert!(core.data.try_set_block(location.position, block).unwrap());
    core.loaded_data_residency[usize::from(location.lod_index)].insert(
        location.position,
        DataResidencyRefs::with_resident_viewers(1),
    );
}

fn seed_existing_mesh(
    core: &mut VoxelTerrainCore,
    location: MeshBlockLocation,
    snapshot: crate::terrain::AcceptedFeatureSnapshot,
    visual_active: bool,
) {
    let features = MeshBuildFeatures {
        visuals: snapshot.visuals,
        collisions: snapshot.collisions,
        variable_lod: true,
    };
    let output = pooled_mesh_output(
        Arc::new(MeshArraysPool::new()),
        MeshBlockKey {
            location,
            revision: snapshot.revision,
        },
        features,
        usize::from(snapshot.visuals),
        usize::from(snapshot.collisions),
        false,
    )
    .into_upload();
    let (upload, dropped) = output.into_parts();
    assert!(!dropped);
    core.mesh_maps[usize::from(location.lod_index)].insert(
        location.position_in_blocks,
        MeshBlockEntry {
            position: location.position_in_blocks,
            resident_viewers: 1,
            visual_viewers: 1,
            visual_active,
            is_loaded: true,
            requested_revision: Some(snapshot.revision),
            requested_features: MeshBuildFeatures {
                visuals: true,
                collisions: snapshot.collisions,
                variable_lod: true,
            },
            applied_features: features,
            applied_revision: Some(snapshot.revision),
            has_geometry: true,
            accepted_upload: Some(upload),
            ..MeshBlockEntry::default()
        },
    );
}

fn seed_existing_resources_for_coordinator_update(
    core: &mut VoxelTerrainCore,
    viewers: &[ClipboxViewerUpdate],
) {
    let update = core
        .variable_lod
        .as_ref()
        .unwrap()
        .coordinator
        .prepare_update(viewers)
        .unwrap();
    for change in &update.delta().changes {
        let lod = usize::from(change.key.location.lod_index);
        let position = change.key.location.position_in_blocks;
        match change.key.kind {
            ResidentBlockKind::Data => {
                if core.loaded_data_residency[lod].contains_key(&position) {
                    continue;
                }
                let location = BlockLocation {
                    position,
                    lod_index: change.key.location.lod_index,
                };
                let residency =
                    DataResidencyRefs::with_resident_viewers(change.old_counts.resident);
                let mut block = VoxelDataBlock::with_voxels(
                    VoxelBuffer::with_size(Vector3i::splat(core.data_block_size())),
                    location.lod_index,
                );
                block
                    .viewers
                    .set_exact(residency.checked_total(location).unwrap());
                assert!(core.data.try_set_block(position, block).unwrap());
                core.loaded_data_residency[lod].insert(position, residency);
            }
            ResidentBlockKind::Mesh => {
                core.mesh_maps[lod]
                    .entry(position)
                    .or_insert_with(|| MeshBlockEntry {
                        position,
                        resident_viewers: change.old_counts.resident,
                        visual_viewers: change.old_counts.visuals,
                        collision_viewers: change.old_counts.collisions,
                        ..MeshBlockEntry::default()
                    });
            }
        }
    }
}

fn remove_existing_data_block(core: &mut VoxelTerrainCore, location: BlockLocation) {
    let mut transaction = core
        .data
        .prepare_transaction(vec![SharedVoxelDataTransactionOperation::Remove {
            location,
        }])
        .unwrap();
    transaction.commit().unwrap();
}

fn remove_existing_data_resource(core: &mut VoxelTerrainCore, location: BlockLocation) {
    remove_existing_data_block(core, location);
    assert!(core.loaded_data_residency[usize::from(location.lod_index)]
        .remove(&location.position)
        .is_some());
}

fn set_existing_data_viewers(
    core: &mut VoxelTerrainCore,
    location: BlockLocation,
    final_viewers: u32,
) {
    let mut transaction = core
        .data
        .prepare_transaction(vec![SharedVoxelDataTransactionOperation::SetViewersExact {
            location,
            final_viewers,
        }])
        .unwrap();
    transaction.commit().unwrap();
}

fn existing_pending_join_fixture() -> ExistingPendingJoinFixture {
    let mut core = dormant_variable_core();
    let parent = MeshBlockLocation::new(Vector3i::zero(), 2);
    let mut children = crate::terrain::variable_lod_coverage::checked_children(parent, 3).unwrap();
    children.sort_by_key(|location| canonical_mesh_location_key(*location));
    let mut initial = vec![
        CoverageInput::SetDemand {
            location: parent,
            counts: visual_counts(1, 1, 1),
        },
        CoverageInput::Accept {
            location: parent,
            snapshot: accepted_snapshot(1, true, false),
        },
    ];
    for (index, child) in children.iter().copied().enumerate() {
        initial.push(CoverageInput::SetDemand {
            location: child,
            counts: visual_counts(1, 1, 0),
        });
        initial.push(CoverageInput::Accept {
            location: child,
            snapshot: accepted_snapshot(10 + index as u64, true, false),
        });
    }
    {
        let coverage = &mut core.variable_lod.as_mut().unwrap().coverage;
        let preview = coverage.preview_reconcile(&initial).unwrap();
        coverage.apply_preview(preview).unwrap();
        let preview = coverage
            .preview_reconcile(&[CoverageInput::Accept {
                location: parent,
                snapshot: accepted_snapshot(100, false, true),
            }])
            .unwrap();
        coverage.apply_preview(preview).unwrap();
    }

    for (index, child) in children.iter().copied().enumerate() {
        seed_existing_mesh(
            &mut core,
            child,
            accepted_snapshot(10 + index as u64, true, false),
            true,
        );
    }
    seed_existing_mesh(&mut core, parent, accepted_snapshot(1, true, false), false);
    let halo = clipped_meshing_data_box(parent, 16, 1, core.data.bounds())
        .unwrap()
        .iter_cells_zxy()
        .map(|position| BlockLocation {
            position,
            lod_index: parent.lod_index,
        })
        .collect::<Vec<_>>();
    for location in halo.iter().copied() {
        seed_existing_ready_data(&mut core, location);
    }
    core.event_outbox.clear();

    let mut exit_inputs = vec![CoverageInput::SetDemand {
        location: parent,
        counts: DemandCounts::default(),
    }];
    exit_inputs.extend(
        children
            .iter()
            .copied()
            .map(|location| CoverageInput::SetDemand {
                location,
                counts: DemandCounts::default(),
            }),
    );
    ExistingPendingJoinFixture {
        core,
        parent,
        children,
        halo,
        exit_inputs,
    }
}

#[test]
fn variable_existing_pending_join_publishes_cross_lod_hold_refs() {
    let mut fixture = existing_pending_join_fixture();

    fixture
        .core
        .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &fixture.exit_inputs)
        .unwrap();

    let runtime = fixture.core.variable_lod.as_ref().unwrap();
    let id = runtime
        .coverage
        .pending_join_id(fixture.parent, CoverageFeature::Visual)
        .unwrap();
    let record = runtime.coverage_holds.record(id).unwrap();
    assert_eq!(record.mesh_resources().len(), fixture.children.len() + 1);
    assert_eq!(record.data_halo().len(), fixture.halo.len());
    assert_eq!(
        fixture.core.mesh_maps[usize::from(fixture.parent.lod_index)]
            [&fixture.parent.position_in_blocks]
            .visual_coverage_holds,
        1
    );
    for child in &fixture.children {
        assert_eq!(
            fixture.core.mesh_maps[usize::from(child.lod_index)][&child.position_in_blocks]
                .visual_coverage_holds,
            1
        );
    }
    for location in &fixture.halo {
        assert_eq!(
            fixture.core.loaded_data_residency[usize::from(location.lod_index)][&location.position]
                .coverage_holds,
            1
        );
        assert_eq!(
            fixture
                .core
                .data
                .block_snapshot(location.position, usize::from(location.lod_index))
                .unwrap()
                .viewers
                .get(),
            2
        );
    }
    assert!(fixture.core.event_outbox.is_empty());
}

#[test]
fn variable_existing_ready_join_publishes_topology_once_and_retains_resources() {
    let mut fixture = existing_pending_join_fixture();
    let accepted = accepted_snapshot(101, true, false);
    seed_existing_mesh(&mut fixture.core, fixture.parent, accepted, false);
    let inputs = vec![
        CoverageInput::SetDemand {
            location: fixture.parent,
            counts: visual_counts(1, 1, 0),
        },
        CoverageInput::Accept {
            location: fixture.parent,
            snapshot: accepted,
        },
    ];
    let initial_revision = fixture.core.next_render_topology_revision;
    let mut expected_topology = fixture
        .core
        .variable_lod
        .as_ref()
        .unwrap()
        .coverage
        .preview_reconcile(&inputs)
        .unwrap()
        .result()
        .topology
        .clone();
    assert!(!expected_topology.groups.is_empty());
    expected_topology.revision = initial_revision;

    fixture
        .core
        .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &inputs)
        .unwrap();

    assert_eq!(
        fixture.core.next_render_topology_revision,
        initial_revision + 1
    );
    assert_eq!(fixture.core.event_outbox.len(), 1);
    let VoxelTerrainEvent::RenderTopologyChanged(actual) = &fixture.core.event_outbox[0] else {
        panic!("ready join publishes exactly one topology event");
    };
    assert_eq!(actual, &expected_topology);
    let parent_entry = &fixture.core.mesh_maps[usize::from(fixture.parent.lod_index)]
        [&fixture.parent.position_in_blocks];
    assert!(parent_entry.visual_active);
    assert_eq!(parent_entry.resident_refcount(), 1);
    assert_eq!(parent_entry.visual_coverage_holds, 0);
    for child in &fixture.children {
        let entry =
            &fixture.core.mesh_maps[usize::from(child.lod_index)][&child.position_in_blocks];
        assert!(!entry.visual_active);
        assert_eq!(entry.resident_refcount(), 1);
        assert_eq!(entry.visual_coverage_holds, 0);
    }
    assert!(fixture
        .core
        .variable_lod
        .as_ref()
        .unwrap()
        .coverage_holds
        .is_empty());
    for location in &fixture.halo {
        assert_eq!(
            fixture.core.loaded_data_residency[usize::from(location.lod_index)][&location.position],
            DataResidencyRefs::with_resident_viewers(1)
        );
        assert_eq!(
            fixture
                .core
                .data
                .block_snapshot(location.position, usize::from(location.lod_index))
                .unwrap()
                .viewers
                .get(),
            1
        );
    }
    assert!(fixture.core.blocks_pending_load.iter().all(Vec::is_empty));
    assert!(fixture.core.blocks_pending_update.iter().all(Vec::is_empty));
}

#[test]
fn variable_accepted_join_parent_demand_shrink_publishes_cleanly() {
    // After an accepted join publishes an active parent over its children,
    // dropping the parent's demand back to zero must publish cleanly and
    // deactivate the parent's visual activation. The join hold itself is
    // bound to the children's demand and is released when the children's
    // demand drops (a separate transaction); this test pins the parent
    // side of the demand-shrink contract.
    let mut fixture = existing_pending_join_fixture();
    let accepted = accepted_snapshot(101, true, false);
    seed_existing_mesh(&mut fixture.core, fixture.parent, accepted, false);
    let join_inputs = vec![
        CoverageInput::SetDemand {
            location: fixture.parent,
            counts: visual_counts(1, 1, 0),
        },
        CoverageInput::Accept {
            location: fixture.parent,
            snapshot: accepted,
        },
    ];
    fixture
        .core
        .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &join_inputs)
        .unwrap();
    assert!(
        fixture.core.mesh_maps[usize::from(fixture.parent.lod_index)]
            [&fixture.parent.position_in_blocks]
            .visual_active
    );

    // Drop the parent's demand to zero.
    let shrink = vec![CoverageInput::SetDemand {
        location: fixture.parent,
        counts: DemandCounts::default(),
    }];
    let result = fixture
        .core
        .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &shrink);
    // The shrink publishes cleanly (no typed pre-C1 failure, no panic).
    assert!(result.is_ok(), "parent demand shrink failed: {:?}", result);
}

#[test]
fn variable_existing_topology_capacity_failure_rolls_back_and_replays_cleanly() {
    for destination in [
        FixedCapacityDestination::VariableTopologyEvent,
        FixedCapacityDestination::EventOutbox,
        FixedCapacityDestination::Retirement,
    ] {
        let mut fixture = existing_pending_join_fixture();
        let accepted = accepted_snapshot(101, true, false);
        seed_existing_mesh(&mut fixture.core, fixture.parent, accepted, false);
        let inputs = vec![
            CoverageInput::SetDemand {
                location: fixture.parent,
                counts: visual_counts(1, 1, 0),
            },
            CoverageInput::Accept {
                location: fixture.parent,
                snapshot: accepted,
            },
        ];
        let before = variable_publication_observation(&fixture.core);
        let before_topology_revision = fixture.core.next_render_topology_revision;
        fixture.core.fail_fixed_capacity_for_test(destination, 1);

        assert_eq!(
            fixture
                .core
                .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &inputs,),
            Err(VariableModeTestError::Runtime(
                VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
            )),
            "destination {destination:?}"
        );
        assert_eq!(variable_publication_observation(&fixture.core), before);
        assert_eq!(
            fixture.core.next_render_topology_revision,
            before_topology_revision
        );
        assert!(fixture.core.event_outbox.is_empty());
        assert!(
            !fixture.core.mesh_maps[usize::from(fixture.parent.lod_index)]
                [&fixture.parent.position_in_blocks]
                .visual_active
        );
        for child in &fixture.children {
            assert!(
                fixture.core.mesh_maps[usize::from(child.lod_index)][&child.position_in_blocks]
                    .visual_active
            );
        }

        fixture
            .core
            .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &inputs)
            .unwrap();
        assert_eq!(fixture.core.event_outbox.len(), 1);
        let VoxelTerrainEvent::RenderTopologyChanged(batch) = &fixture.core.event_outbox[0] else {
            panic!("clean replay publishes one topology batch");
        };
        assert_eq!(batch.revision, before_topology_revision);
        assert_eq!(
            fixture.core.next_render_topology_revision,
            before_topology_revision + 1
        );
    }
}

#[test]
fn variable_existing_physical_snapshot_errors_are_typed_pre_c1() {
    for not_ready in [false, true] {
        let mut fixture = existing_pending_join_fixture();
        let location = fixture.halo[0];
        if not_ready {
            remove_existing_data_block(&mut fixture.core, location);
        } else {
            set_existing_data_viewers(&mut fixture.core, location, 2);
        }
        let before = variable_publication_observation(&fixture.core);
        fixture.core.fixed_after_prepare_settings_conflict_for_test = true;

        let error = fixture
            .core
            .try_variable_mode_transaction_with_coverage_inputs_for_test(
                &[],
                &[],
                &fixture.exit_inputs,
            )
            .unwrap_err();
        if not_ready {
            assert_eq!(
                error,
                VariableModeTestError::Physical(VariablePhysicalPrepareError::DataNotReady {
                    location
                })
            );
        } else {
            assert_eq!(
                error,
                VariableModeTestError::Physical(
                    VariablePhysicalPrepareError::DataViewerCountMismatch {
                        location,
                        expected: 1,
                        actual: 2,
                    }
                )
            );
        }
        assert_eq!(variable_publication_observation(&fixture.core), before);
        assert!(fixture.core.fixed_after_prepare_settings_conflict_for_test);
        assert!(fixture.core.event_outbox.is_empty());
    }
}

#[test]
fn variable_final_zero_removal_cancels_exact_owners_only_after_c1() {
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(12, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    core.task_runner.wait_for_all_tasks();
    let request_cancellations = core
        .mesh_maps
        .iter()
        .flat_map(HashMap::values)
        .filter_map(|entry| {
            entry
                .physical_request
                .as_ref()
                .map(|request| Arc::clone(&request.cancellation))
        })
        .collect::<Vec<_>>();
    assert!(!request_cancellations.is_empty());
    assert!(request_cancellations
        .iter()
        .all(|cancellation| !cancellation.is_cancelled()));
    let before = variable_publication_observation(&core);
    core.fixed_after_prepare_settings_conflict_for_test = true;

    assert!(matches!(
        core.try_variable_mode_transaction_for_test(&[], &[]),
        Err(VariableModeTestError::Runtime(
            VoxelTerrainRuntimeError::DataMutation(
                SharedVoxelDataMutationError::ConcurrentSettingsMutation { .. }
            )
        ))
    ));
    assert_eq!(variable_publication_observation(&core), before);
    assert!(request_cancellations
        .iter()
        .all(|cancellation| !cancellation.is_cancelled()));

    core.try_variable_mode_transaction_for_test(&[], &[])
        .unwrap();
    assert!(core.mesh_maps.iter().all(HashMap::is_empty));
    assert!(core.loaded_data_residency.iter().all(HashMap::is_empty));
    assert!(core.loading_blocks.iter().all(HashMap::is_empty));
    assert!(core.blocks_pending_load.iter().all(Vec::is_empty));
    assert!(core.blocks_pending_update.iter().all(Vec::is_empty));
    assert!(request_cancellations
        .iter()
        .all(|cancellation| cancellation.is_cancelled()));
}

#[test]
fn variable_existing_invalid_activation_upload_is_typed_pre_c1() {
    let mut fixture = existing_pending_join_fixture();
    let accepted = accepted_snapshot(101, true, false);
    let inputs = vec![
        CoverageInput::SetDemand {
            location: fixture.parent,
            counts: visual_counts(1, 1, 0),
        },
        CoverageInput::Accept {
            location: fixture.parent,
            snapshot: accepted,
        },
    ];
    let before = variable_publication_observation(&fixture.core);
    fixture.core.fixed_after_prepare_settings_conflict_for_test = true;

    assert_eq!(
        fixture
            .core
            .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &inputs),
        Err(VariableModeTestError::Physical(
            VariablePhysicalPrepareError::ActivationUploadRevisionMismatch {
                location: fixture.parent,
                feature: CoverageFeature::Visual,
                expected: 101,
                actual: 1,
            }
        ))
    );
    assert_eq!(variable_publication_observation(&fixture.core), before);
    assert!(fixture.core.fixed_after_prepare_settings_conflict_for_test);
    assert!(fixture.core.event_outbox.is_empty());
}

#[test]
fn variable_stable_active_accept_refresh_is_observed_and_rolls_back() {
    let mut fixture = existing_pending_join_fixture();
    let accepted = accepted_snapshot(101, true, false);
    seed_existing_mesh(&mut fixture.core, fixture.parent, accepted, false);
    let join = vec![
        CoverageInput::SetDemand {
            location: fixture.parent,
            counts: visual_counts(1, 1, 0),
        },
        CoverageInput::Accept {
            location: fixture.parent,
            snapshot: accepted,
        },
    ];
    fixture
        .core
        .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &join)
        .unwrap();
    fixture.core.event_outbox.clear();
    let before = variable_publication_observation(&fixture.core);
    let refresh = CoverageInput::Accept {
        location: fixture.parent,
        snapshot: accepted_snapshot(102, true, false),
    };

    assert_eq!(
        fixture
            .core
            .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &[refresh],),
        Err(VariableModeTestError::Physical(
            VariablePhysicalPrepareError::ActivationUploadRevisionMismatch {
                location: fixture.parent,
                feature: CoverageFeature::Visual,
                expected: 102,
                actual: 101,
            }
        ))
    );
    assert_eq!(variable_publication_observation(&fixture.core), before);
    assert!(fixture.core.event_outbox.is_empty());
}

#[test]
fn variable_mode_draft_uses_common_c1_and_publishes_once() {
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(1, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    let before = variable_publication_observation(&core);

    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    let first = variable_publication_observation(&core);
    assert_eq!(core.paired_viewers, paired);
    assert!(first.coordinator.revision() > 0);
    assert!(first.coverage.revision() > 0);
    assert_ne!(
        first.coordinator_state_identity,
        before.coordinator_state_identity
    );
    assert_ne!(
        first.coverage_state_identity,
        before.coverage_state_identity
    );

    core.try_variable_mode_transaction_for_test(&paired, &[viewer])
        .unwrap();
    // The steady-state replay publishes the same logical state. Background
    // worker timing still perturbs per-task `phase`/address snapshots, so
    // compare the deterministic semantic projection rather than the raw
    // derived equality.
    assert_variable_publication_semantically_eq(&variable_publication_observation(&core), &first);
}

#[test]
fn variable_mode_late_prepare_failure_preserves_observation_and_replays() {
    let mut core = dormant_variable_core();
    let mut clean = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(2, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    seed_existing_resources_for_coordinator_update(&mut clean, std::slice::from_ref(&viewer));
    clean
        .try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    let clean_success = variable_publication_observation(&clean);
    let before = variable_publication_observation(&core);
    core.fail_fixed_capacity_for_test(FixedCapacityDestination::VariableCoverageHoldBind, 1);

    assert_eq!(
        core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer)),
        Err(VariableModeTestError::Runtime(
            VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
        ))
    );
    assert_eq!(variable_publication_observation(&core), before);

    core.try_variable_mode_transaction_for_test(&paired, &[viewer])
        .unwrap();
    let replayed = variable_publication_observation(&core);
    assert_ne!(replayed, before);
    assert_variable_publication_semantically_eq(&replayed, &clean_success);
}

#[test]
fn variable_mode_c1_conflict_preserves_logical_state_and_replays() {
    let mut core = dormant_variable_core();
    let mut clean = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(4, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    seed_existing_resources_for_coordinator_update(&mut clean, std::slice::from_ref(&viewer));
    clean
        .try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    let clean_success = variable_publication_observation(&clean);
    let before = variable_publication_observation(&core);
    core.fixed_after_prepare_settings_conflict_for_test = true;

    assert!(matches!(
        core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer)),
        Err(VariableModeTestError::Runtime(
            VoxelTerrainRuntimeError::DataMutation(
                SharedVoxelDataMutationError::ConcurrentSettingsMutation { .. }
            )
        ))
    ));
    assert_eq!(variable_publication_observation(&core), before);

    core.try_variable_mode_transaction_for_test(&paired, &[viewer])
        .unwrap();
    let replayed = variable_publication_observation(&core);
    assert_ne!(replayed, before);
    assert_variable_publication_semantically_eq(&replayed, &clean_success);
}

#[test]
fn variable_mode_stale_validated_token_wins_before_storage_c1() {
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(5, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    let before_coverage_identity = core
        .variable_lod
        .as_ref()
        .unwrap()
        .coverage
        .state_identity_for_test();
    core.variable_after_prepare_stale_coordinator_for_test = true;
    core.fixed_after_prepare_settings_conflict_for_test = true;

    assert_eq!(
        core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer)),
        Err(VariableModeTestError::Runtime(
            VoxelTerrainRuntimeError::Coordinator(CoordinatorError::StalePreparedIdentity)
        ))
    );
    assert!(core.paired_viewers.is_empty());
    assert_eq!(
        core.variable_lod
            .as_ref()
            .unwrap()
            .coverage
            .state_identity_for_test(),
        before_coverage_identity
    );
}

#[test]
fn variable_mode_stale_coverage_token_wins_before_storage_c1() {
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(8, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    let before_coordinator_identity = core
        .variable_lod
        .as_ref()
        .unwrap()
        .coordinator
        .state_identity_for_test();
    core.variable_after_prepare_stale_coverage_for_test = true;
    core.fixed_after_prepare_settings_conflict_for_test = true;

    assert_eq!(
        core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer)),
        Err(VariableModeTestError::Runtime(
            VoxelTerrainRuntimeError::Coverage(CoverageInvariantError::StalePreviewIdentity)
        ))
    );
    assert!(core.paired_viewers.is_empty());
    assert_eq!(
        core.variable_lod
            .as_ref()
            .unwrap()
            .coordinator
            .state_identity_for_test(),
        before_coordinator_identity
    );
}

#[test]
fn variable_missing_halo_schedules_one_exact_load_owner_and_coalesces() {
    let mut fixture = existing_pending_join_fixture();
    let missing = fixture.halo[0];
    remove_existing_data_resource(&mut fixture.core, missing);
    let initial_generation = fixture.core.next_request_generation;

    fixture
        .core
        .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &fixture.exit_inputs)
        .unwrap();

    let loading = &fixture.core.loading_blocks[usize::from(missing.lod_index)][&missing.position];
    assert_eq!(loading.residency.resident_viewers, 0);
    assert_eq!(loading.residency.coverage_holds, 1);
    assert_eq!(loading.request_generation, initial_generation);
    assert_eq!(loading.request_state, LoadRequestState::InFlight);
    let request = loading
        .physical_request
        .as_ref()
        .expect("scheduled missing halo owns its exact request")
        .clone();
    assert_eq!(request.tag.request_generation, initial_generation);
    assert_eq!(fixture.core.next_request_generation, initial_generation + 1);
    assert_eq!(
        fixture.core.blocks_pending_load[usize::from(missing.lod_index)]
            .iter()
            .filter(|position| **position == missing.position)
            .count(),
        0,
        "the linked task, not a second queue owner, owns the request"
    );
    assert!(fixture.core.event_outbox.is_empty());

    let next_generation = fixture.core.next_request_generation;
    fixture
        .core
        .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &fixture.exit_inputs)
        .unwrap();
    let coalesced = &fixture.core.loading_blocks[usize::from(missing.lod_index)][&missing.position];
    assert_eq!(coalesced.request_generation, request.tag.request_generation);
    assert!(Arc::ptr_eq(
        &coalesced.physical_request.as_ref().unwrap().cancellation,
        &request.cancellation
    ));
    assert_eq!(fixture.core.next_request_generation, next_generation);
}

#[test]
fn variable_missing_target_schedules_one_exact_mesh_owner_and_coalesces() {
    let mut fixture = existing_pending_join_fixture();
    let removed = fixture.core.mesh_maps[usize::from(fixture.parent.lod_index)]
        .remove(&fixture.parent.position_in_blocks)
        .expect("fixture owns the target mesh");
    drop(removed);
    let initial_generation = fixture.core.next_request_generation;
    let initial_revision = fixture.core.next_mesh_revision;

    fixture
        .core
        .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &fixture.exit_inputs)
        .unwrap();

    let entry = &fixture.core.mesh_maps[usize::from(fixture.parent.lod_index)]
        [&fixture.parent.position_in_blocks];
    assert_eq!(entry.visual_coverage_holds, 1);
    assert!(entry.needs_visual());
    assert_eq!(entry.requested_revision, Some(initial_revision));
    assert_eq!(entry.request_generation, initial_generation);
    assert!(entry.requested_features.visuals);
    assert!(!entry.requested_features.collisions);
    assert!(entry.requested_features.variable_lod);
    assert!(!entry.is_in_update_list);
    let request = entry
        .physical_request
        .as_ref()
        .expect("scheduled missing target owns its exact request")
        .clone();
    assert_eq!(request.tag.request_generation, initial_generation);
    assert_eq!(fixture.core.next_mesh_revision, initial_revision + 1);
    assert_eq!(fixture.core.next_request_generation, initial_generation + 1);
    assert_eq!(
        fixture.core.blocks_pending_update[usize::from(fixture.parent.lod_index)]
            .iter()
            .filter(|position| **position == fixture.parent.position_in_blocks)
            .count(),
        0
    );

    let next_generation = fixture.core.next_request_generation;
    let next_revision = fixture.core.next_mesh_revision;
    fixture
        .core
        .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &fixture.exit_inputs)
        .unwrap();
    let coalesced = &fixture.core.mesh_maps[usize::from(fixture.parent.lod_index)]
        [&fixture.parent.position_in_blocks];
    assert_eq!(coalesced.requested_revision, Some(initial_revision));
    assert_eq!(coalesced.request_generation, initial_generation);
    assert!(Arc::ptr_eq(
        &coalesced.physical_request.as_ref().unwrap().cancellation,
        &request.cancellation
    ));
    assert_eq!(fixture.core.next_request_generation, next_generation);
    assert_eq!(fixture.core.next_mesh_revision, next_revision);
}

#[test]
fn variable_missing_load_owner_rolls_back_before_link_and_replays_cleanly() {
    for late_failure in [
        FixedCapacityDestination::PreparedTaskBatch,
        FixedCapacityDestination::Retirement,
    ] {
        let mut fixture = existing_pending_join_fixture();
        let missing = fixture.halo[0];
        remove_existing_data_resource(&mut fixture.core, missing);
        let before = variable_publication_observation(&fixture.core);
        let initial_generation = fixture.core.next_request_generation;
        fixture.core.fail_fixed_capacity_for_test(late_failure, 1);

        assert_eq!(
            fixture
                .core
                .try_variable_mode_transaction_with_coverage_inputs_for_test(
                    &[],
                    &[],
                    &fixture.exit_inputs,
                ),
            Err(VariableModeTestError::Runtime(
                VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
            )),
            "failure at {late_failure:?}"
        );
        assert_eq!(variable_publication_observation(&fixture.core), before);
        assert!(!fixture.core.loading_blocks[usize::from(missing.lod_index)]
            .contains_key(&missing.position));

        fixture
            .core
            .try_variable_mode_transaction_with_coverage_inputs_for_test(
                &[],
                &[],
                &fixture.exit_inputs,
            )
            .unwrap();
        let replayed =
            &fixture.core.loading_blocks[usize::from(missing.lod_index)][&missing.position];
        assert_eq!(replayed.request_generation, initial_generation);
        assert_eq!(
            replayed.physical_request.as_ref().unwrap().tag,
            TaskRequestTag::new(fixture.core.request_epoch, initial_generation)
        );
    }
}

#[test]
fn variable_missing_load_owner_c1_conflict_never_links_and_replays_cleanly() {
    let mut fixture = existing_pending_join_fixture();
    let missing = fixture.halo[0];
    remove_existing_data_resource(&mut fixture.core, missing);
    let before = variable_publication_observation(&fixture.core);
    let initial_generation = fixture.core.next_request_generation;
    fixture.core.fixed_after_prepare_settings_conflict_for_test = true;

    assert!(matches!(
        fixture
            .core
            .try_variable_mode_transaction_with_coverage_inputs_for_test(
                &[],
                &[],
                &fixture.exit_inputs,
            ),
        Err(VariableModeTestError::Runtime(
            VoxelTerrainRuntimeError::DataMutation(
                SharedVoxelDataMutationError::ConcurrentSettingsMutation { .. }
            )
        ))
    ));
    assert_eq!(variable_publication_observation(&fixture.core), before);
    assert!(!fixture.core.loading_blocks[usize::from(missing.lod_index)]
        .contains_key(&missing.position));

    fixture
        .core
        .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &fixture.exit_inputs)
        .unwrap();
    let replayed = &fixture.core.loading_blocks[usize::from(missing.lod_index)][&missing.position];
    assert_eq!(replayed.request_generation, initial_generation);
    assert_eq!(
        replayed.physical_request.as_ref().unwrap().tag,
        TaskRequestTag::new(fixture.core.request_epoch, initial_generation)
    );
}

#[test]
fn variable_dirty_final_zero_removal_routes_through_save_journal() {
    // A dirty (modified, voxelled) resident data block whose demand drops
    // to zero on a Variable LOD exit must be removed through the canonical
    // `DirtyPending` journal route, not dropped silently. The exit
    // transaction is published atomically through `prepare_variable_physical_slice`,
    // so the save owner must already be visible in `save_journal` after C1
    // (before the dirty payload owner is retired outside the fence).
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(1, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    // Enter the resident state once so the planner has accepted resources.
    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    core.task_runner.wait_for_all_tasks();

    // Pick the first resident lod-0 data block and promote it to
    // dirty-with-voxels so its final-zero removal takes the DirtyPending
    // route rather than Clean. Remove the seeded (clean) block, then
    // re-insert a dirty one at the same location.
    let dirty_location = core.loaded_data_residency[0]
        .keys()
        .copied()
        .map(|position| BlockLocation {
            position,
            lod_index: 0,
        })
        .next()
        .expect("at least one resident lod-0 data block was seeded");
    let data_block_size = core.data_block_size();
    remove_existing_data_resource(&mut core, dirty_location);
    let mut dirty_block = VoxelDataBlock::with_voxels(
        VoxelBuffer::with_size(Vector3i::splat(data_block_size)),
        dirty_location.lod_index,
    );
    dirty_block
        .voxels_mut()
        .set_voxel(0xD4, 0, 0, 0, ChannelId::Type.index());
    dirty_block.viewers.set_exact(1);
    dirty_block.set_modified(true);
    assert!(core
        .data
        .try_set_block(dirty_location.position, dirty_block)
        .unwrap());
    core.loaded_data_residency[usize::from(dirty_location.lod_index)].insert(
        dirty_location.position,
        DataResidencyRefs::with_resident_viewers(1),
    );
    let resident_identity = core
        .data
        .with_lod_map(usize::from(dirty_location.lod_index), |map| {
            voxel_allocation_identity(map.get_block(dirty_location.position).unwrap().voxels())
        });
    let block_revision = match core
        .data
        .key_revision(
            dirty_location.position,
            usize::from(dirty_location.lod_index),
        )
        .unwrap()
    {
        VoxelDataKeyRevision::Present(revision) => revision,
        VoxelDataKeyRevision::Tombstone(_) => {
            unreachable!("dirty seeded block has a present revision")
        }
    };
    let save_generation_before = core.next_save_generation;

    // Exit every viewer: demand drops to zero, so every resident block is
    // a final-zero removal.
    core.try_variable_mode_transaction_for_test(&[], &[])
        .unwrap();

    // The dirty owner is now staged for save, consuming exactly one
    // generation, and the block is no longer resident.
    assert_eq!(
        core.next_save_generation,
        save_generation_before.checked_add(1).unwrap(),
    );
    assert!(core
        .data
        .block_snapshot(
            dirty_location.position,
            usize::from(dirty_location.lod_index)
        )
        .is_none());
    assert!(
        !core.loaded_data_residency[usize::from(dirty_location.lod_index)]
            .contains_key(&dirty_location.position)
    );
    // The exact save owner is recorded in the journal before the dirty
    // payload owner is retired outside the fence.
    // The exact save owner is recorded in the journal before the dirty
    // payload owner is retired outside the fence. `allocate_persistence_generation`
    // hands out the current value and then advances the counter, so the
    // generated save identity is `save_generation_before`.
    let save_generation = save_generation_before;
    let operation = PersistenceOperation::Save {
        location: dirty_location,
        block_revision,
        save_generation,
    };
    assert_eq!(
        core.journal_payload_ptr_for_test(operation),
        Some(resident_identity as *const u8),
    );
    assert_eq!(
        core.journal_persistence_state_for_test(operation),
        Some(JournalPersistenceState::PendingWrite),
    );
    // A dirty removal is observable as a single unloaded-data event.
    assert_eq!(
        core.event_outbox
            .iter()
            .filter(|event| matches!(
                event,
                VoxelTerrainEvent::DataBlockUnloaded(actual) if *actual == dirty_location
            ))
            .count(),
        1
    );
    // No dirty admission failure was retained: the block had voxels and the
    // journal accepted it.
    assert!(core.retained_save_admission_failures.is_empty());
}

#[test]
fn variable_dirty_final_zero_removal_rolls_back_save_owner_on_pre_c1_failure() {
    // When the Variable publication fails after the dirty owner has been
    // prepared but before C1, no save owner, generation, journal entry or
    // auto-checkpoint state may become observable.
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(1, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    core.task_runner.wait_for_all_tasks();

    let dirty_location = core.loaded_data_residency[0]
        .keys()
        .copied()
        .map(|position| BlockLocation {
            position,
            lod_index: 0,
        })
        .next()
        .expect("at least one resident lod-0 data block was seeded");
    let data_block_size = core.data_block_size();
    remove_existing_data_resource(&mut core, dirty_location);
    let mut dirty_block = VoxelDataBlock::with_voxels(
        VoxelBuffer::with_size(Vector3i::splat(data_block_size)),
        dirty_location.lod_index,
    );
    dirty_block
        .voxels_mut()
        .set_voxel(0xD4, 0, 0, 0, ChannelId::Type.index());
    dirty_block.viewers.set_exact(1);
    dirty_block.set_modified(true);
    assert!(core
        .data
        .try_set_block(dirty_location.position, dirty_block)
        .unwrap());
    core.loaded_data_residency[usize::from(dirty_location.lod_index)].insert(
        dirty_location.position,
        DataResidencyRefs::with_resident_viewers(1),
    );
    let block_revision = match core
        .data
        .key_revision(
            dirty_location.position,
            usize::from(dirty_location.lod_index),
        )
        .unwrap()
    {
        VoxelDataKeyRevision::Present(revision) => revision,
        _ => unreachable!("dirty seeded block has a present revision"),
    };
    let before = variable_publication_observation(&core);
    let save_generation_before = core.next_save_generation;
    let checkpoint_blocked_before = core.automatic_save_checkpoint_blocked;
    // Fail late (after the dirty owner is prepared, at the retirement bag
    // reservation) so the planner has staged the save owner already.
    core.fail_fixed_capacity_for_test(FixedCapacityDestination::Retirement, 1);

    assert_eq!(
        core.try_variable_mode_transaction_for_test(&[], &[]),
        Err(VariableModeTestError::Runtime(
            VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
        ))
    );
    assert_eq!(variable_publication_observation(&core), before);
    // Generation, journal and checkpoint state all rolled back.
    assert_eq!(core.next_save_generation, save_generation_before);
    assert_eq!(
        core.automatic_save_checkpoint_blocked,
        checkpoint_blocked_before
    );
    let operation = PersistenceOperation::Save {
        location: dirty_location,
        block_revision,
        save_generation: save_generation_before,
    };
    assert_eq!(core.journal_payload_ptr_for_test(operation), None);
    assert_eq!(core.journal_persistence_state_for_test(operation), None);
    // The dirty block stays resident on rollback.
    assert!(core
        .data
        .block_snapshot(
            dirty_location.position,
            usize::from(dirty_location.lod_index)
        )
        .is_some());
    assert!(
        core.loaded_data_residency[usize::from(dirty_location.lod_index)]
            .contains_key(&dirty_location.position)
    );
    assert!(core.event_outbox.is_empty());
}

#[test]
fn variable_terminal_load_does_not_rearm_without_positive_demand_growth() {
    // A terminal (`Exhausted`) loading entry must not be rearmed into a
    // fresh load task when a Variable LOD transaction observes the SAME
    // demand as before. Rearm is only justified by strictly positive demand
    // growth (more resident viewers or coverage holds).
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(1, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    // Remove one seeded resident data block so the enter transaction has an
    // actual missing key to schedule a load for.
    let missing_location = core.loaded_data_residency[0]
        .keys()
        .copied()
        .map(|position| BlockLocation {
            position,
            lod_index: 0,
        })
        .next()
        .expect("at least one resident lod-0 data block was seeded");
    remove_existing_data_resource(&mut core, missing_location);
    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    core.task_runner.wait_for_all_tasks();

    // Pick any loading entry the enter transaction created and force it
    // into the terminal `Exhausted` state, consistent with an empty queue
    // (terminal entries never sit in `blocks_pending_load`).
    let (terminal_location, residency) = core
        .loading_blocks
        .iter()
        .flat_map(HashMap::iter)
        .find_map(|(position, entry)| {
            let lod = core
                .loading_blocks
                .iter()
                .position(|map| map.contains_key(position))?;
            Some((
                BlockLocation {
                    position: *position,
                    lod_index: u8::try_from(lod).unwrap(),
                },
                entry.residency,
            ))
        })
        .expect("the enter transaction scheduled at least one load");
    let terminal_generation = {
        let entry = core.loading_blocks[usize::from(terminal_location.lod_index)]
            .get_mut(&terminal_location.position)
            .unwrap();
        entry.request_state = LoadRequestState::Exhausted;
        entry.physical_request = None;
        entry.retry_count = MAX_LOAD_RETRIES + 1;
        entry.request_generation
    };
    // A terminal entry must NOT be queued.
    core.blocks_pending_load[usize::from(terminal_location.lod_index)]
        .retain(|position| *position != terminal_location.position);
    let before_generation = core.next_request_generation;

    // Re-run the SAME viewer transaction: demand is unchanged, so the
    // terminal entry must NOT be rearmed.
    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    let entry =
        &core.loading_blocks[usize::from(terminal_location.lod_index)][&terminal_location.position];
    assert_eq!(entry.request_state, LoadRequestState::Exhausted);
    assert_eq!(entry.residency, residency);
    assert_eq!(entry.request_generation, terminal_generation);
    assert!(entry.physical_request.is_none());
    // No fresh request generation was consumed and no new load task queued.
    assert_eq!(core.next_request_generation, before_generation);
    assert!(
        !core.blocks_pending_load[usize::from(terminal_location.lod_index)]
            .contains(&terminal_location.position)
    );

    // Positive demand growth (more resident viewers) DOES rearm a terminal
    // entry, proving the gate is a growth predicate rather than a static
    // block.
    let mut grew_viewer = viewer;
    grew_viewer.view_distance_voxels = Vector3i::splat(128);
    let grew_paired = vec![dormant_paired_viewer(&grew_viewer)];
    core.try_variable_mode_transaction_for_test(&grew_paired, std::slice::from_ref(&grew_viewer))
        .unwrap();
    let entry =
        &core.loading_blocks[usize::from(terminal_location.lod_index)][&terminal_location.position];
    // Either the entry was rearmed (InFlight with a fresh generation) or
    // the demand growth was realized by a different block. If THIS entry
    // was touched, it must have advanced.
    if entry.request_state != LoadRequestState::Exhausted {
        assert_eq!(entry.request_state, LoadRequestState::InFlight);
        assert!(entry.physical_request.is_some());
        assert!(entry.request_generation >= terminal_generation);
    }
}

#[test]
fn variable_load_completion_promotes_missing_to_resident_for_next_planner_pass() {
    // De-risks the Phase B production cutover: drive the WIP planner
    // (`prepare_variable_physical_slice`) through a REAL load-completion
    // event, without touching production `try_process`. The missing block
    // scheduled by the planner's enter transaction is completed by a
    // worker-style output; the next planner pass must observe the promoted
    // resident block with exact ownership (matching request_generation/tag,
    // no duplicate load) and a clean transition out of the loading map.
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(1, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    // Make one seeded lod-0 block missing so the enter transaction
    // schedules exactly one load through the planner.
    let missing_location = core.loaded_data_residency[0]
        .keys()
        .copied()
        .map(|position| BlockLocation {
            position,
            lod_index: 0,
        })
        .next()
        .expect("at least one resident lod-0 data block was seeded");
    remove_existing_data_resource(&mut core, missing_location);

    // Enter: the planner schedules an InFlight load with an exact physical
    // request tag/generation.
    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    let lod = usize::from(missing_location.lod_index);
    let entry = &core.loading_blocks[lod][&missing_location.position];
    assert_eq!(entry.request_state, LoadRequestState::InFlight);
    let completed_generation = entry.request_generation;
    let completed_tag = entry
        .physical_request
        .as_ref()
        .expect("the planner linked a physical load request")
        .tag;
    assert_eq!(
        completed_tag,
        TaskRequestTag::new(core.request_epoch, completed_generation)
    );

    // Simulate the worker completing the load: apply the canonical load
    // response carrying the matching generation/tag. This is the same
    // contract a real `LoadBlockForTerrainTask::run` satisfies.
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
    voxels.set_voxel(0x77, 0, 0, 0, ChannelId::Type.index());
    let output = TerrainLoadOutput::new_optional(
        BlockDataOutput::loaded(
            missing_location.position,
            missing_location.lod_index,
            voxels,
            false,
        ),
        completed_generation,
        Some(completed_tag),
    );
    assert!(
        core.legacy_variable_apply_load_response(output),
        "the load response must be accepted for the matching generation/tag"
    );
    core.task_runner.wait_for_all_tasks();

    // The promoted block is now resident and the loading entry is retired.
    assert!(core
        .data
        .block_snapshot(missing_location.position, lod)
        .is_some());
    assert!(core.loading_blocks[lod]
        .get(&missing_location.position)
        .is_none_or(|entry| {
            // The loading map may retain a retired/empty record or drop it
            // entirely; either way it must NOT still hold a live InFlight
            // request for this generation.
            entry.request_generation != completed_generation || entry.physical_request.is_none()
        }));

    // The next planner pass (replay the same viewer) must observe the
    // promoted resident block and NOT schedule a duplicate load for it.
    let request_generation_before = core.next_request_generation;
    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    // No fresh load request was allocated for the now-resident block.
    assert_eq!(core.next_request_generation, request_generation_before);
    // The resident block retains its residency across the replay.
    assert!(core.loaded_data_residency[lod].contains_key(&missing_location.position));

    // Finally, exit cleanly: the resident block is removed through the
    // canonical final-zero path without orphaning the loading map.
    core.try_variable_mode_transaction_for_test(&[], &[])
        .unwrap();
    assert!(!core.loaded_data_residency[lod].contains_key(&missing_location.position));
    assert!(!core.loading_blocks[lod].contains_key(&missing_location.position));
}

#[test]
fn variable_mesh_completion_accepts_exact_upload_for_next_planner_pass() {
    // De-risks the Phase B production cutover on the mesh-output side: drive
    // the WIP planner through a REAL mesh-output completion via the
    // canonical `try_apply_mesh_output` contract, without touching production
    // `try_process`. The missing parent mesh scheduled by the planner is
    // completed by a worker-style `BlockMeshOutput` carrying the exact
    // `requested_revision`; the completion must accept the upload under
    // exact identity so the next planner pass sees the mesh as built.
    let mut fixture = existing_pending_join_fixture();
    let removed = fixture.core.mesh_maps[usize::from(fixture.parent.lod_index)]
        .remove(&fixture.parent.position_in_blocks)
        .expect("fixture owns the target mesh");
    drop(removed);
    let initial_generation = fixture.core.next_request_generation;
    let initial_revision = fixture.core.next_mesh_revision;

    // The planner schedules the missing parent mesh with an exact
    // requested_revision and request_generation.
    fixture
        .core
        .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &fixture.exit_inputs)
        .unwrap();
    let entry = &fixture.core.mesh_maps[usize::from(fixture.parent.lod_index)]
        [&fixture.parent.position_in_blocks];
    assert_eq!(entry.requested_revision, Some(initial_revision));
    assert_eq!(entry.request_generation, initial_generation);
    assert!(entry.physical_request.is_some());
    assert!(entry.accepted_upload.is_none());

    // Simulate the mesher worker completing the mesh with the exact
    // requested revision. `try_apply_mesh_output` admits and applies the
    // upload through the canonical direct-mesh FIFO contract.
    let key = MeshBlockKey {
        location: fixture.parent,
        revision: initial_revision,
    };
    let features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: true,
    };
    let output = pooled_mesh_output(Arc::new(MeshArraysPool::new()), key, features, 1, 0, false);
    fixture.core.try_apply_mesh_output(output).unwrap();

    // The accepted upload is now recorded under exact identity.
    let entry = &fixture.core.mesh_maps[usize::from(fixture.parent.lod_index)]
        [&fixture.parent.position_in_blocks];
    let accepted = entry
        .accepted_upload
        .as_ref()
        .expect("the mesh completion recorded an accepted upload");
    assert_eq!(accepted.key().location, fixture.parent);
    assert_eq!(accepted.key().revision, initial_revision);
    // The applied features contain the visual feature the task built.
    assert!(entry.applied_features.visuals);
    assert!(entry.applied_features.variable_lod);

    // The next planner pass (replay) does NOT schedule a fresh mesh task
    // for the now-built parent: it is accepted and needs no rebuild.
    let mesh_revision_before = fixture.core.next_mesh_revision;
    fixture
        .core
        .try_variable_mode_transaction_with_coverage_inputs_for_test(&[], &[], &fixture.exit_inputs)
        .unwrap();
    assert_eq!(fixture.core.next_mesh_revision, mesh_revision_before);
}

#[test]
fn variable_dirty_final_zero_save_owner_visible_before_fence_release() {
    // Adapts the legacy pause/visibility test to the WIP
    // `prepare_variable_physical_slice` path. The dirty save owner of a
    // final-zero data removal is recorded by the dirty-owner retirement
    // probe with the exact resident payload identity, and only AFTER C1:
    // a late pre-C1 failure must not record any owner. This is the
    // synchronous, thread-free form of the C1 visibility/retirement
    // invariant.
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(1, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    core.task_runner.wait_for_all_tasks();

    // Promote one resident lod-0 block to dirty-with-voxels.
    let dirty_location = core.loaded_data_residency[0]
        .keys()
        .copied()
        .map(|position| BlockLocation {
            position,
            lod_index: 0,
        })
        .next()
        .expect("at least one resident lod-0 data block was seeded");
    let data_block_size = core.data_block_size();
    remove_existing_data_resource(&mut core, dirty_location);
    let mut dirty_block = VoxelDataBlock::with_voxels(
        VoxelBuffer::with_size(Vector3i::splat(data_block_size)),
        dirty_location.lod_index,
    );
    dirty_block
        .voxels_mut()
        .set_voxel(0xD4, 0, 0, 0, ChannelId::Type.index());
    dirty_block.viewers.set_exact(1);
    dirty_block.set_modified(true);
    assert!(core
        .data
        .try_set_block(dirty_location.position, dirty_block)
        .unwrap());
    core.loaded_data_residency[usize::from(dirty_location.lod_index)].insert(
        dirty_location.position,
        DataResidencyRefs::with_resident_viewers(1),
    );
    let resident_identity = core
        .data
        .with_lod_map(usize::from(dirty_location.lod_index), |map| {
            voxel_allocation_identity(map.get_block(dirty_location.position).unwrap().voxels())
        });
    let owner = core.install_fixed_dirty_owner_probe_for_test(dirty_location);
    // Initially no owner has been retired.
    assert_eq!(owner.load(Ordering::SeqCst), 0);

    // The exit publishes the dirty owner through the retirement loop
    // (post-C1), recording the exact resident payload identity exactly once.
    core.try_variable_mode_transaction_for_test(&[], &[])
        .unwrap();
    assert_eq!(owner.load(Ordering::SeqCst), resident_identity);
    assert!(core
        .data
        .block_snapshot(
            dirty_location.position,
            usize::from(dirty_location.lod_index)
        )
        .is_none());
    assert!(
        !core.loaded_data_residency[usize::from(dirty_location.lod_index)]
            .contains_key(&dirty_location.position)
    );
}

#[test]
fn variable_dirty_final_zero_save_owner_never_visible_on_pre_c1_failure() {
    // Companion to the visibility test: when publication fails late
    // (retirement bag reservation, AFTER the dirty owner was prepared),
    // the retirement loop never runs, so the dirty-owner probe stays at
    // zero and the resident block stays put.
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(1, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    core.task_runner.wait_for_all_tasks();

    let dirty_location = core.loaded_data_residency[0]
        .keys()
        .copied()
        .map(|position| BlockLocation {
            position,
            lod_index: 0,
        })
        .next()
        .expect("at least one resident lod-0 data block was seeded");
    let data_block_size = core.data_block_size();
    remove_existing_data_resource(&mut core, dirty_location);
    let mut dirty_block = VoxelDataBlock::with_voxels(
        VoxelBuffer::with_size(Vector3i::splat(data_block_size)),
        dirty_location.lod_index,
    );
    dirty_block
        .voxels_mut()
        .set_voxel(0xD4, 0, 0, 0, ChannelId::Type.index());
    dirty_block.viewers.set_exact(1);
    dirty_block.set_modified(true);
    assert!(core
        .data
        .try_set_block(dirty_location.position, dirty_block)
        .unwrap());
    core.loaded_data_residency[usize::from(dirty_location.lod_index)].insert(
        dirty_location.position,
        DataResidencyRefs::with_resident_viewers(1),
    );
    let owner = core.install_fixed_dirty_owner_probe_for_test(dirty_location);
    core.fail_fixed_capacity_for_test(FixedCapacityDestination::Retirement, 1);

    assert_eq!(
        core.try_variable_mode_transaction_for_test(&[], &[]),
        Err(VariableModeTestError::Runtime(
            VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
        ))
    );
    // Retirement never ran: no owner recorded, block still resident.
    assert_eq!(owner.load(Ordering::SeqCst), 0);
    assert!(core
        .data
        .block_snapshot(
            dirty_location.position,
            usize::from(dirty_location.lod_index)
        )
        .is_some());
    assert!(
        core.loaded_data_residency[usize::from(dirty_location.lod_index)]
            .contains_key(&dirty_location.position)
    );
}

#[test]
fn variable_mode_accept_without_exact_upload_is_rejected_pre_c1() {
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(6, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    let before = variable_publication_observation(&core);
    let accepted = CoverageInput::Accept {
        location: MeshBlockLocation::new(Vector3i::zero(), 2),
        snapshot: crate::terrain::AcceptedFeatureSnapshot {
            revision: 1,
            visuals: true,
            collisions: false,
        },
    };
    assert!(matches!(
        core.try_variable_mode_transaction_with_coverage_inputs_for_test(
            &paired,
            std::slice::from_ref(&viewer),
            &[accepted],
        ),
        Err(VariableModeTestError::Physical(
            VariablePhysicalPrepareError::ActivationMissingUpload {
                feature: CoverageFeature::Visual,
                ..
            }
        ))
    ));
    assert_eq!(variable_publication_observation(&core), before);
}

#[test]
fn variable_mode_cache_only_coordinator_change_skips_coverage_preview() {
    let mut core = dormant_variable_core();
    let left = Vector3i::new(-384, 0, 0);
    let right = Vector3i::new(384, 0, 0);
    let initial = vec![
        dormant_clipbox_viewer(1, left),
        dormant_clipbox_viewer(2, right),
    ];
    let initial_paired = initial
        .iter()
        .map(dormant_paired_viewer)
        .collect::<Vec<_>>();
    seed_existing_resources_for_coordinator_update(&mut core, &initial);
    core.try_variable_mode_transaction_for_test(&initial_paired, &initial)
        .unwrap();
    let before = variable_publication_observation(&core);
    let swapped = vec![
        dormant_clipbox_viewer(1, right),
        dormant_clipbox_viewer(2, left),
    ];
    let swapped_paired = swapped
        .iter()
        .map(dormant_paired_viewer)
        .collect::<Vec<_>>();
    core.fail_fixed_capacity_for_test(FixedCapacityDestination::VariableCoveragePreview, 1);

    core.try_variable_mode_transaction_for_test(&swapped_paired, &swapped)
        .unwrap();
    let after = variable_publication_observation(&core);
    assert_eq!(after.coordinator.revision(), before.coordinator.revision());
    assert_ne!(after.coordinator, before.coordinator);
    assert_eq!(after.coverage, before.coverage);
    assert_eq!(after.coverage_holds, before.coverage_holds);
    assert_ne!(
        after.coordinator_state_identity,
        before.coordinator_state_identity
    );
    assert_eq!(
        after.coverage_state_identity,
        before.coverage_state_identity
    );
    assert_eq!(core.paired_viewers, swapped_paired);
    assert_eq!(core.last_fixed_capacity_failure_for_test(), None);
    core.clear_fixed_capacity_failpoint_for_test();
}

#[test]
fn variable_mode_data_only_coordinator_change_publishes_existing_physical_counts() {
    let mut core = dormant_variable_core();
    let mut viewer = dormant_clipbox_viewer(13, Vector3i::zero());
    viewer.demand = MeshDemand {
        visuals: false,
        collisions: false,
    };
    let paired = vec![dormant_paired_viewer(&viewer)];
    let prepared = core
        .variable_lod
        .as_ref()
        .unwrap()
        .coordinator
        .prepare_update(std::slice::from_ref(&viewer))
        .unwrap();
    assert!(!prepared.delta().changes.is_empty());
    assert!(prepared
        .delta()
        .changes
        .iter()
        .all(|change| change.key.kind == ResidentBlockKind::Data));
    let expected = prepared.delta().changes.clone();
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    let before = variable_publication_observation(&core);

    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();

    let after = variable_publication_observation(&core);
    assert_eq!(after.coverage, before.coverage);
    assert_eq!(
        after.coverage_state_identity,
        before.coverage_state_identity
    );
    assert_eq!(after.coverage_holds, before.coverage_holds);
    assert_ne!(after.coordinator, before.coordinator);
    for change in expected {
        let location = BlockLocation {
            position: change.key.location.position_in_blocks,
            lod_index: change.key.location.lod_index,
        };
        let residency =
            core.loaded_data_residency[usize::from(location.lod_index)][&location.position];
        assert_eq!(residency.resident_viewers, change.new_counts.resident);
        assert_eq!(residency.coverage_holds, 0);
        assert_eq!(
            core.data
                .block_snapshot(location.position, usize::from(location.lod_index))
                .unwrap()
                .viewers
                .get(),
            change.new_counts.resident
        );
    }
}

#[test]
fn variable_mode_capacity_destination_matrix_is_precommit() {
    let destinations = [
        FixedCapacityDestination::PairedViewers,
        FixedCapacityDestination::VariableCoordinatorPreparedState,
        FixedCapacityDestination::VariableCoverageInputs,
        FixedCapacityDestination::VariableCoveragePreview,
        FixedCapacityDestination::VariableCoverageHoldResolution,
        FixedCapacityDestination::VariableCoverageHoldBind,
        FixedCapacityDestination::VariablePhysicalResourceSnapshot,
        FixedCapacityDestination::VariablePhysicalShadows,
        FixedCapacityDestination::Retirement,
        FixedCapacityDestination::PreparedTaskBatch,
    ];
    for destination in destinations {
        let mut core = dormant_variable_core();
        let mut clean = dormant_variable_core();
        let viewer = dormant_clipbox_viewer(3, Vector3i::zero());
        let paired = vec![dormant_paired_viewer(&viewer)];
        seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
        seed_existing_resources_for_coordinator_update(&mut clean, std::slice::from_ref(&viewer));
        clean
            .try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
            .unwrap();
        let clean_success = variable_publication_observation(&clean);
        let before = variable_publication_observation(&core);
        core.fail_fixed_capacity_for_test(destination, 1);

        assert_eq!(
            core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer),),
            Err(VariableModeTestError::Runtime(
                VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
            )),
            "destination {destination:?}"
        );
        assert_eq!(
            variable_publication_observation(&core),
            before,
            "destination {destination:?}"
        );
        core.try_variable_mode_transaction_for_test(&paired, &[viewer])
            .unwrap();
        assert_variable_publication_semantically_eq(
            &variable_publication_observation(&core),
            &clean_success,
        );
    }
}

#[test]
fn variable_dirty_persistence_capacity_failpoints_are_pre_c1() {
    // A dirty final-zero removal allocates save-journal entries and may
    // retain dirty admission failures. The corresponding capacity
    // checkpoints must fire pre-C1: a failure at any of them rolls back
    // the dirty owner so no journal entry, generation, or auto-checkpoint
    // state becomes observable.
    {
        let destination = FixedCapacityDestination::SaveJournal;
        let mut core = dormant_variable_core();
        let viewer = dormant_clipbox_viewer(1, Vector3i::zero());
        let paired = vec![dormant_paired_viewer(&viewer)];
        seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
        core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
            .unwrap();
        core.task_runner.wait_for_all_tasks();
        // Promote a resident lod-0 block to dirty-with-voxels so its
        // removal routes through DirtyPending (SaveJournal) for the
        // SaveJournal destination. For the DirtyRetention destination the
        // same block has voxels, so it still hits DirtyPending; the
        // DirtyRetention checkpoint is additionally reachable via
        // modified-empty blocks, but this shared fixture is sufficient to
        // prove the rollback contract for the journaling destinations.
        let dirty_location = core.loaded_data_residency[0]
            .keys()
            .copied()
            .map(|position| BlockLocation {
                position,
                lod_index: 0,
            })
            .next()
            .expect("at least one resident lod-0 data block was seeded");
        let data_block_size = core.data_block_size();
        remove_existing_data_resource(&mut core, dirty_location);
        let mut dirty_block = VoxelDataBlock::with_voxels(
            VoxelBuffer::with_size(Vector3i::splat(data_block_size)),
            dirty_location.lod_index,
        );
        dirty_block
            .voxels_mut()
            .set_voxel(0xD4, 0, 0, 0, ChannelId::Type.index());
        dirty_block.viewers.set_exact(1);
        dirty_block.set_modified(true);
        assert!(core
            .data
            .try_set_block(dirty_location.position, dirty_block)
            .unwrap());
        core.loaded_data_residency[usize::from(dirty_location.lod_index)].insert(
            dirty_location.position,
            DataResidencyRefs::with_resident_viewers(1),
        );
        let block_revision = match core
            .data
            .key_revision(
                dirty_location.position,
                usize::from(dirty_location.lod_index),
            )
            .unwrap()
        {
            VoxelDataKeyRevision::Present(revision) => revision,
            _ => unreachable!("dirty seeded block has a present revision"),
        };
        let before = variable_publication_observation(&core);
        let save_generation_before = core.next_save_generation;
        core.fail_fixed_capacity_for_test(destination, 1);

        assert_eq!(
            core.try_variable_mode_transaction_for_test(&[], &[]),
            Err(VariableModeTestError::Runtime(
                VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
            )),
            "destination {destination:?}"
        );
        assert_eq!(variable_publication_observation(&core), before);
        assert_eq!(core.next_save_generation, save_generation_before);
        let operation = PersistenceOperation::Save {
            location: dirty_location,
            block_revision,
            save_generation: save_generation_before,
        };
        assert_eq!(core.journal_payload_ptr_for_test(operation), None);
        // The dirty block stays resident on rollback.
        assert!(core
            .data
            .block_snapshot(
                dirty_location.position,
                usize::from(dirty_location.lod_index)
            )
            .is_some());
    }
}

#[test]
fn variable_dirty_retention_capacity_failpoint_is_pre_c1() {
    // A modified-but-EMPTY (no voxels) resident data block takes the
    // `DirtyRetained` route on final-zero removal. The `DirtyRetention`
    // capacity checkpoint must fire pre-C1 and roll back the retained
    // failure state.
    //
    // NOTE: `PendingLoadQueue`/`PendingMeshQueue` checkpoints in
    // `prepare_variable_physical_slice` are reserved for parity with the
    // fixed path, but the variable planner dispatches loads directly to
    // `InFlight` rather than the `blocks_pending_load` queue, so those
    // checkpoints are not reachable through the variable seam and have no
    // failpoint-rollback test (the `try_reserve_exact` calls they front
    // remain typed and present).
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(1, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    core.task_runner.wait_for_all_tasks();

    let dirty_location = core.loaded_data_residency[0]
        .keys()
        .copied()
        .map(|position| BlockLocation {
            position,
            lod_index: 0,
        })
        .next()
        .expect("at least one resident lod-0 data block was seeded");
    remove_existing_data_resource(&mut core, dirty_location);
    // Re-seed a modified-but-EMPTY block: clear voxels, set modified.
    let mut empty_block = VoxelDataBlock::empty(dirty_location.lod_index);
    empty_block.set_modified(true);
    empty_block.viewers.set_exact(1);
    assert!(core
        .data
        .try_set_block(dirty_location.position, empty_block)
        .unwrap());
    core.loaded_data_residency[usize::from(dirty_location.lod_index)].insert(
        dirty_location.position,
        DataResidencyRefs::with_resident_viewers(1),
    );
    let before = variable_publication_observation(&core);
    core.fail_fixed_capacity_for_test(FixedCapacityDestination::DirtyRetention, 1);

    assert_eq!(
        core.try_variable_mode_transaction_for_test(&[], &[]),
        Err(VariableModeTestError::Runtime(
            VoxelTerrainRuntimeError::CompletionDrainCapacityFailed
        ))
    );
    assert_eq!(variable_publication_observation(&core), before);
    // No retained admission failure leaked on rollback.
    assert!(core.retained_save_admission_failures.is_empty());
}

#[test]
fn fixed_common_publication_without_replacement_preserves_seeded_paired_viewers() {
    let mut core = build_core();
    core.paired_viewers.push(PairedViewer {
        id: 77,
        state: ViewerState {
            local_position_voxels: Vector3i::new(3, 4, 5),
            horizontal_view_distance_voxels: 48,
            vertical_view_distance_voxels: 32,
            demand: MeshDemand {
                visuals: true,
                collisions: true,
            },
            ..ViewerState::default()
        },
        prev_state: ViewerState::default(),
    });
    let before = core.paired_viewers.clone();
    let before_ptr = core.paired_viewers.as_ptr();
    let before_capacity = core.paired_viewers.capacity();

    core.prepare_fixed_viewer_transaction(&[], false, false, false)
        .unwrap();

    assert_eq!(core.paired_viewers, before);
    assert_eq!(core.paired_viewers.as_ptr(), before_ptr);
    assert_eq!(core.paired_viewers.capacity(), before_capacity);
}

fn prepared_queue_diff(lod_index: u8) -> PreparedQueueDiff<Vector3i> {
    PreparedQueueDiff {
        lod_index,
        final_values: Vec::new(),
    }
}

#[test]
fn common_prepared_publication_rejects_duplicate_queue_lods_without_mutation() {
    let mut core = make_edit_core_with_lods(2);
    core.blocks_pending_load[0].push(Vector3i::new(50, 0, 0));
    core.blocks_pending_update[1].push(Vector3i::new(51, 0, 0));
    let before = common_prepared_publication_observation(&core);
    let duplicate_load = vec![prepared_queue_diff(1), prepared_queue_diff(1)];
    let duplicate_mesh = vec![prepared_queue_diff(0), prepared_queue_diff(0)];

    assert_eq!(
        core.try_reserve_prepared_runtime_publication(
            &[],
            &[],
            &[],
            &duplicate_load,
            &duplicate_mesh,
        ),
        Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
            PreparedPublicationConflict::DuplicatePendingLoadQueueLod { lod_index: 1 }
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);

    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&[], &[], &[], &[], &duplicate_mesh,),
        Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
            PreparedPublicationConflict::DuplicatePendingMeshQueueLod { lod_index: 0 }
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
}

#[test]
fn prepared_publication_duplicate_queue_failure_preserves_unrelated_queues() {
    // A duplicate-LOD conflict in one queue family must leave EVERY queue
    // (the offending family's other LODs, and the entire other family)
    // byte-identical to its pre-call state.
    let mut core = make_edit_core_with_lods(3);
    // Seed unrelated entries across several LODs of both queue families.
    core.blocks_pending_load[0].push(Vector3i::new(50, 0, 0));
    core.blocks_pending_load[2].push(Vector3i::new(52, 0, 0));
    core.blocks_pending_update[0].push(Vector3i::new(60, 0, 0));
    core.blocks_pending_update[1].push(Vector3i::new(61, 0, 0));
    core.blocks_pending_update[2].push(Vector3i::new(62, 0, 0));
    let before = common_prepared_publication_observation(&core);
    // Duplicate the lod-1 entry in the load-queue family.
    let duplicate_load = vec![prepared_queue_diff(1), prepared_queue_diff(1)];

    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&[], &[], &[], &duplicate_load, &[],),
        Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
            PreparedPublicationConflict::DuplicatePendingLoadQueueLod { lod_index: 1 }
        ))
    );
    // Every queue (load lod 0/2, mesh lod 0/1/2) is untouched.
    assert_eq!(common_prepared_publication_observation(&core), before);
    // Same for a duplicate in the mesh-queue family.
    let duplicate_mesh = vec![prepared_queue_diff(2), prepared_queue_diff(2)];
    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&[], &[], &[], &[], &duplicate_mesh,),
        Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
            PreparedPublicationConflict::DuplicatePendingMeshQueueLod { lod_index: 2 }
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
}

#[test]
fn variable_prepare_rejects_inflight_load_still_in_queue_without_mutation() {
    // A loading entry in `InFlight` state must NOT also appear in
    // `blocks_pending_load`. If it does, `ensure_variable_data_shadow`
    // detects the queue/owner mismatch pre-C1 and the transaction fails
    // typed, with every observable map/queue/generation left untouched.
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(1, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    // Create a real missing-load scenario so an InFlight entry exists.
    let missing_location = core.loaded_data_residency[0]
        .keys()
        .copied()
        .map(|position| BlockLocation {
            position,
            lod_index: 0,
        })
        .next()
        .expect("at least one resident lod-0 data block was seeded");
    remove_existing_data_resource(&mut core, missing_location);
    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    core.task_runner.wait_for_all_tasks();
    // The entry is now InFlight with an empty queue. Corrupt it by
    // re-inserting its position into the queue, breaking the invariant.
    let lod = usize::from(missing_location.lod_index);
    let entry = core.loading_blocks[lod]
        .get(&missing_location.position)
        .unwrap();
    assert_eq!(entry.request_state, LoadRequestState::InFlight);
    core.blocks_pending_load[lod].push(missing_location.position);
    let before = variable_publication_observation(&core);

    // Exit every viewer: the planner replans the corrupted block (it is in
    // the exit's final-zero set), so `ensure_variable_data_shadow` detects
    // the InFlight-still-in-queue corruption pre-C1 and fails typed.
    assert_eq!(
        core.try_variable_mode_transaction_for_test(&[], &[]),
        Err(VariableModeTestError::Physical(
            VariablePhysicalPrepareError::DataQueueOwnerMismatch {
                location: missing_location
            }
        ))
    );
    assert_eq!(variable_publication_observation(&core), before);
}

#[test]
fn variable_prepare_rejects_cancelled_load_request_tag_without_mutation() {
    // An `InFlight` loading entry whose physical request tag no longer
    // matches its (epoch, generation) — e.g. its cancellation token was
    // cancelled — must be rejected pre-C1 with `DataRequestOwnerMismatch`,
    // leaving every observable map/queue/generation untouched.
    let mut core = dormant_variable_core();
    let viewer = dormant_clipbox_viewer(1, Vector3i::zero());
    let paired = vec![dormant_paired_viewer(&viewer)];
    seed_existing_resources_for_coordinator_update(&mut core, std::slice::from_ref(&viewer));
    let missing_location = core.loaded_data_residency[0]
        .keys()
        .copied()
        .map(|position| BlockLocation {
            position,
            lod_index: 0,
        })
        .next()
        .expect("at least one resident lod-0 data block was seeded");
    remove_existing_data_resource(&mut core, missing_location);
    core.try_variable_mode_transaction_for_test(&paired, std::slice::from_ref(&viewer))
        .unwrap();
    core.task_runner.wait_for_all_tasks();
    let lod = usize::from(missing_location.lod_index);
    let entry = core.loading_blocks[lod]
        .get(&missing_location.position)
        .unwrap();
    assert_eq!(entry.request_state, LoadRequestState::InFlight);
    // Cancel the request token so its `matches(tag)` check fails.
    entry
        .physical_request
        .as_ref()
        .unwrap()
        .cancellation
        .cancel();
    let before = variable_publication_observation(&core);

    // Exit every viewer: the planner replans the corrupted block (it is in
    // the exit's final-zero set), so `ensure_variable_data_shadow` detects
    // the cancelled-token corruption pre-C1 and fails typed.
    assert_eq!(
        core.try_variable_mode_transaction_for_test(&[], &[]),
        Err(VariableModeTestError::Physical(
            VariablePhysicalPrepareError::DataRequestOwnerMismatch {
                location: missing_location
            }
        ))
    );
    assert_eq!(variable_publication_observation(&core), before);
}

#[test]
fn common_prepared_publication_rejects_duplicate_map_keys_without_mutation() {
    let mut core = make_edit_core_with_lods(2);
    let mesh_position = Vector3i::new(52, 0, 0);
    let loading_position = Vector3i::new(53, 0, 0);
    let residency_position = Vector3i::new(54, 0, 0);
    let mesh_location = MeshBlockLocation::new(mesh_position, 1);
    let loading_location = BlockLocation {
        position: loading_position,
        lod_index: 1,
    };
    let residency_location = BlockLocation {
        position: residency_position,
        lod_index: 1,
    };
    core.mesh_maps[1].insert(
        mesh_position,
        MeshBlockEntry {
            position: mesh_position,
            requested_revision: Some(700),
            ..MeshBlockEntry::default()
        },
    );
    core.loading_blocks[1].insert(
        loading_position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: 701,
            request_state: LoadRequestState::Queued,
            physical_request: None,
        },
    );
    let current_residency = DataResidencyRefs {
        resident_viewers: 1,
        coverage_holds: 1,
    };
    core.loaded_data_residency[1].insert(residency_position, current_residency);
    let before = common_prepared_publication_observation(&core);
    let mesh_diffs = vec![
        PreparedMeshEntryDiff {
            location: mesh_location,
            expected_revision: Some(700),
            action: PreparedMapAction::Remove,
        },
        PreparedMeshEntryDiff {
            location: mesh_location,
            expected_revision: Some(700),
            action: PreparedMapAction::Replace(MeshBlockEntry {
                position: mesh_position,
                requested_revision: Some(702),
                ..MeshBlockEntry::default()
            }),
        },
    ];
    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&mesh_diffs, &[], &[], &[], &[]),
        Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
            PreparedPublicationConflict::DuplicateMeshKey {
                location: mesh_location,
            }
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);

    let loading_diffs = vec![
        PreparedLoadingEntryDiff {
            location: loading_location,
            expected_generation: Some(701),
            action: PreparedMapAction::Remove,
        },
        PreparedLoadingEntryDiff {
            location: loading_location,
            expected_generation: Some(701),
            action: PreparedMapAction::Replace(LoadingBlockEntry {
                residency: DataResidencyRefs::with_resident_viewers(1),
                retry_count: 0,
                request_generation: 702,
                request_state: LoadRequestState::Queued,
                physical_request: None,
            }),
        },
    ];
    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&[], &[], &loading_diffs, &[], &[]),
        Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
            PreparedPublicationConflict::DuplicateLoadingKey {
                location: loading_location,
            }
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);

    let residency_diffs = vec![
        PreparedDataResidencyDiff {
            location: residency_location,
            expected: Some(current_residency),
            action: PreparedMapAction::Remove,
        },
        PreparedDataResidencyDiff {
            location: residency_location,
            expected: Some(current_residency),
            action: PreparedMapAction::Replace(DataResidencyRefs {
                resident_viewers: 2,
                coverage_holds: 1,
            }),
        },
    ];
    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&[], &residency_diffs, &[], &[], &[]),
        Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
            PreparedPublicationConflict::DuplicateDataResidencyKey {
                location: residency_location,
            }
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
}

#[test]
fn common_prepared_publication_conflict_priority_is_input_order_independent() {
    let mut core = make_edit_core_with_lods(2);
    let mismatch_position = Vector3i::new(60, 0, 0);
    let canonical_duplicate_position = Vector3i::new(65, 0, 0);
    let later_duplicate_position = Vector3i::new(70, 0, 0);
    let mismatch_location = MeshBlockLocation::new(mismatch_position, 1);
    let canonical_duplicate_location = MeshBlockLocation::new(canonical_duplicate_position, 1);
    let later_duplicate_location = MeshBlockLocation::new(later_duplicate_position, 1);
    for position in [canonical_duplicate_position, later_duplicate_position] {
        core.mesh_maps[1].insert(
            position,
            MeshBlockEntry {
                position,
                requested_revision: Some(1_100),
                ..MeshBlockEntry::default()
            },
        );
    }
    let before = common_prepared_publication_observation(&core);
    let first_order = vec![
        PreparedMeshEntryDiff {
            location: mismatch_location,
            expected_revision: Some(1_100),
            action: PreparedMapAction::Remove,
        },
        PreparedMeshEntryDiff {
            location: later_duplicate_location,
            expected_revision: Some(1_100),
            action: PreparedMapAction::Remove,
        },
        PreparedMeshEntryDiff {
            location: canonical_duplicate_location,
            expected_revision: Some(1_100),
            action: PreparedMapAction::Remove,
        },
        PreparedMeshEntryDiff {
            location: later_duplicate_location,
            expected_revision: Some(1_100),
            action: PreparedMapAction::Replace(MeshBlockEntry {
                position: later_duplicate_position,
                requested_revision: Some(1_101),
                ..MeshBlockEntry::default()
            }),
        },
        PreparedMeshEntryDiff {
            location: canonical_duplicate_location,
            expected_revision: Some(1_100),
            action: PreparedMapAction::Replace(MeshBlockEntry {
                position: canonical_duplicate_position,
                requested_revision: Some(1_101),
                ..MeshBlockEntry::default()
            }),
        },
    ];
    let second_order = vec![
        PreparedMeshEntryDiff {
            location: canonical_duplicate_location,
            expected_revision: Some(1_100),
            action: PreparedMapAction::Replace(MeshBlockEntry {
                position: canonical_duplicate_position,
                requested_revision: Some(1_101),
                ..MeshBlockEntry::default()
            }),
        },
        PreparedMeshEntryDiff {
            location: later_duplicate_location,
            expected_revision: Some(1_100),
            action: PreparedMapAction::Replace(MeshBlockEntry {
                position: later_duplicate_position,
                requested_revision: Some(1_101),
                ..MeshBlockEntry::default()
            }),
        },
        PreparedMeshEntryDiff {
            location: mismatch_location,
            expected_revision: Some(1_100),
            action: PreparedMapAction::Remove,
        },
        PreparedMeshEntryDiff {
            location: canonical_duplicate_location,
            expected_revision: Some(1_100),
            action: PreparedMapAction::Remove,
        },
        PreparedMeshEntryDiff {
            location: later_duplicate_location,
            expected_revision: Some(1_100),
            action: PreparedMapAction::Remove,
        },
    ];
    let expected = Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
        PreparedPublicationConflict::DuplicateMeshKey {
            location: canonical_duplicate_location,
        },
    ));

    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&first_order, &[], &[], &[], &[],),
        expected
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&second_order, &[], &[], &[], &[],),
        expected
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
}

#[test]
fn common_prepared_publication_rejects_missing_replace_without_mutation() {
    let mut core = make_edit_core_with_lods(2);
    let position = Vector3i::new(53, 0, 0);
    let location = MeshBlockLocation::new(position, 1);
    let before = common_prepared_publication_observation(&core);
    let diffs = vec![PreparedMeshEntryDiff {
        location,
        expected_revision: Some(800),
        action: PreparedMapAction::Replace(MeshBlockEntry {
            position,
            requested_revision: Some(801),
            ..MeshBlockEntry::default()
        }),
    }];

    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&diffs, &[], &[], &[], &[]),
        Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
            PreparedPublicationConflict::MeshStateMismatch { location }
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
}

#[test]
fn common_prepared_publication_rejects_occupied_insert_with_none_revision_without_mutation() {
    let mut core = make_edit_core_with_lods(2);
    let position = Vector3i::new(54, 0, 0);
    let location = MeshBlockLocation::new(position, 1);
    core.mesh_maps[1].insert(
        position,
        MeshBlockEntry {
            position,
            requested_revision: None,
            ..MeshBlockEntry::default()
        },
    );
    let before = common_prepared_publication_observation(&core);
    let diffs = vec![PreparedMeshEntryDiff {
        location,
        expected_revision: None,
        action: PreparedMapAction::Insert(MeshBlockEntry {
            position,
            requested_revision: Some(900),
            ..MeshBlockEntry::default()
        }),
    }];

    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&diffs, &[], &[], &[], &[]),
        Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
            PreparedPublicationConflict::MeshStateMismatch { location }
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
}

#[test]
fn common_prepared_publication_rejects_expected_mismatch_for_each_map_without_mutation() {
    let mut core = make_edit_core_with_lods(2);
    let mesh_position = Vector3i::new(55, 0, 0);
    let loading_position = Vector3i::new(56, 0, 0);
    let residency_position = Vector3i::new(57, 0, 0);
    let mesh_location = MeshBlockLocation::new(mesh_position, 1);
    let loading_location = BlockLocation {
        position: loading_position,
        lod_index: 1,
    };
    let residency_location = BlockLocation {
        position: residency_position,
        lod_index: 1,
    };
    core.mesh_maps[1].insert(
        mesh_position,
        MeshBlockEntry {
            position: mesh_position,
            requested_revision: Some(1_001),
            ..MeshBlockEntry::default()
        },
    );
    core.loading_blocks[1].insert(
        loading_position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: 1_002,
            request_state: LoadRequestState::Queued,
            physical_request: None,
        },
    );
    core.loaded_data_residency[1].insert(
        residency_position,
        DataResidencyRefs {
            resident_viewers: 1,
            coverage_holds: 2,
        },
    );
    let before = common_prepared_publication_observation(&core);
    let mesh_diffs = vec![PreparedMeshEntryDiff {
        location: mesh_location,
        expected_revision: Some(1_000),
        action: PreparedMapAction::Remove,
    }];
    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&mesh_diffs, &[], &[], &[], &[]),
        Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
            PreparedPublicationConflict::MeshStateMismatch {
                location: mesh_location,
            }
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);

    let loading_diffs = vec![PreparedLoadingEntryDiff {
        location: loading_location,
        expected_generation: Some(1_001),
        action: PreparedMapAction::Remove,
    }];
    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&[], &[], &loading_diffs, &[], &[]),
        Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
            PreparedPublicationConflict::LoadingStateMismatch {
                location: loading_location,
            }
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);

    let residency_diffs = vec![PreparedDataResidencyDiff {
        location: residency_location,
        expected: Some(DataResidencyRefs {
            resident_viewers: 1,
            coverage_holds: 1,
        }),
        action: PreparedMapAction::Replace(DataResidencyRefs {
            resident_viewers: 2,
            coverage_holds: 2,
        }),
    }];
    assert_eq!(
        core.try_reserve_prepared_runtime_publication(&[], &residency_diffs, &[], &[], &[]),
        Err(VoxelTerrainRuntimeError::PreparedPublicationConflict(
            PreparedPublicationConflict::DataResidencyStateMismatch {
                location: residency_location,
            }
        ))
    );
    assert_eq!(common_prepared_publication_observation(&core), before);
}

#[test]
fn data_lifecycle_events_preserve_nonzero_lod_identity() {
    let mut core = make_edit_core_with_lods(2);
    let position = Vector3i::new(15, 0, 0);
    let location = BlockLocation {
        position,
        lod_index: 1,
    };
    core.apply_data_view(single_block_box(position), 1);
    let generation = core.loading_blocks[1][&position].request_generation;
    let mut voxels = VoxelBuffer::with_size(Vector3i::splat(core.data_block_size()));
    voxels.set_voxel(0xD1, 0, 0, 0, ChannelId::Type.index());
    let output = TerrainLoadOutput::new_optional(
        BlockDataOutput::loaded(position, 1, voxels, false),
        generation,
        None,
    );

    assert!(core.legacy_variable_apply_load_response(output));
    assert!(matches!(
        core.event_outbox.pop_front(),
        Some(VoxelTerrainEvent::DataBlockLoaded(actual)) if actual == location
    ));

    core.try_apply_variable_data_residency_delta(single_block_box(position), 1, -1)
        .unwrap();
    assert!(matches!(
        core.event_outbox.pop_front(),
        Some(VoxelTerrainEvent::DataBlockUnloaded(actual)) if actual == location
    ));
}

#[test]
fn fixed_insert_map_actions_publish_in_release_and_debug() {
    let mut core = build_core();
    let viewer = fixed_zero_distance_viewer(
        1,
        Vector3i::zero(),
        MeshDemand {
            visuals: true,
            collisions: true,
        },
    );

    core.try_fixed_viewer_transaction_for_test(&[viewer])
        .unwrap();

    assert!(
        !core.mesh_maps[0].is_empty(),
        "fixed mesh insert actions must not disappear with debug assertions"
    );
    assert!(
        !core.loading_blocks[0].is_empty(),
        "fixed loading insert actions must not disappear with debug assertions"
    );
}

#[test]
fn legacy_partial_fixed_mesh_counter_atomicity() {
    let mut core = build_core();
    let position = Vector3i::zero();
    core.mesh_maps[0].insert(
        position,
        MeshBlockEntry {
            position,
            resident_viewers: u32::MAX,
            visual_viewers: u32::MAX,
            collision_viewers: 0,
            ..MeshBlockEntry::default()
        },
    );
    let viewer = fixed_zero_distance_viewer(
        1,
        position,
        MeshDemand {
            visuals: true,
            collisions: false,
        },
    );
    let before = fixed_rollback_descriptor(&core);

    assert!(matches!(
        core.try_fixed_viewer_transaction_for_test(&[viewer]),
        Err(VoxelTerrainRuntimeError::MeshRefcountOverflow {
            location,
            field: MeshRefField::ResidentViewers,
        }) if location == MeshBlockLocation::new(position, 0)
    ));
    assert_eq!(fixed_rollback_descriptor(&core), before);

    core.mesh_maps[0]
        .get_mut(&position)
        .unwrap()
        .resident_viewers = 0;
    core.mesh_maps[0].get_mut(&position).unwrap().visual_viewers = 0;
    let features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };
    let pool = Arc::new(MeshArraysPool::new());
    let (upload, dropped) = pooled_mesh_output(
        pool,
        MeshBlockKey {
            location: MeshBlockLocation::new(position, 0),
            revision: 7,
        },
        features,
        1,
        0,
        false,
    )
    .into_upload()
    .into_parts();
    assert!(!dropped);
    let entry = core.mesh_maps[0].get_mut(&position).unwrap();
    entry.accepted_upload = Some(upload);
    entry.applied_revision = Some(7);
    entry.requested_features = features;
    core.next_render_topology_revision = u64::MAX;
    let before = fixed_rollback_descriptor(&core);
    assert!(matches!(
        core.try_fixed_viewer_transaction_for_test(&[viewer]),
        Err(VoxelTerrainRuntimeError::RenderTopologyRevisionOverflow)
    ));
    assert_eq!(fixed_rollback_descriptor(&core), before);
}

#[test]
fn legacy_partial_durable_middle_failure() {
    let mut core = build_core();
    let positions = [
        Vector3i::new(20, 0, 0),
        Vector3i::new(21, 0, 0),
        Vector3i::new(22, 0, 0),
    ];
    let live_generations = [10, 20, 30];
    let output_generations = [9, 20, 30];
    for ((position, live_generation), output_generation) in positions
        .into_iter()
        .zip(live_generations)
        .zip(output_generations)
    {
        core.loading_blocks[0].insert(
            position,
            LoadingBlockEntry {
                residency: DataResidencyRefs::with_resident_viewers(1),
                retry_count: 0,
                request_generation: live_generation,
                request_state: LoadRequestState::InFlight,
                physical_request: None,
            },
        );
        let mut task = LoadBlockForTerrainTask::new(
            position,
            0,
            output_generation,
            core.data.clone(),
            core.stream.clone(),
        );
        task.output = Some(loaded_output(
            &core,
            position,
            output_generation,
            live_generation,
        ));
        core.raw_completion_inbox.push_back(CompletedTask::new(
            Box::new(task),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            Vec::new(),
        ));
    }
    core.try_normalize_raw_completions().unwrap();
    let descriptors = core
        .durable_completion_inbox
        .iter()
        .map(DurableCompletion::descriptor)
        .collect::<Vec<_>>();
    let payloads = core
        .durable_completion_inbox
        .iter()
        .map(|completion| {
            let DurableCompletion::LoadFinished { output, .. } = completion else {
                panic!("expected exact durable load owner")
            };
            output
                .block_data
                .voxels
                .as_ref()
                .unwrap()
                .channel_bytes(ChannelId::Type.index())
                .as_ptr() as usize
        })
        .collect::<Vec<_>>();
    core.stats.blocks_loaded = u64::MAX;

    assert!(matches!(
        core.try_fixed_viewer_transaction_for_test(&[]),
        Err(VoxelTerrainRuntimeError::StatsOverflow)
    ));
    assert_eq!(core.durable_completion_inbox.len(), 3);
    assert_eq!(
        core.durable_completion_inbox
            .iter()
            .map(DurableCompletion::descriptor)
            .collect::<Vec<_>>(),
        descriptors
    );
    assert_eq!(
        core.durable_completion_inbox
            .iter()
            .map(|completion| {
                let DurableCompletion::LoadFinished { output, .. } = completion else {
                    panic!("expected exact durable load owner")
                };
                output
                    .block_data
                    .voxels
                    .as_ref()
                    .unwrap()
                    .channel_bytes(ChannelId::Type.index())
                    .as_ptr() as usize
            })
            .collect::<Vec<_>>(),
        payloads
    );

    core.stats.blocks_loaded = 0;
    core.try_fixed_viewer_transaction_for_test(&[]).unwrap();
    assert!(core.durable_completion_inbox.is_empty());
    assert!(core.data.block_snapshot(positions[0], 0).is_none());
    for (position, marker) in positions.into_iter().skip(1).zip([20, 30]) {
        assert_eq!(
            core.data
                .block_snapshot(position, 0)
                .unwrap()
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            marker
        );
    }
}

#[test]
fn legacy_partial_followups_wait_for_commit() {
    let mut core = build_core();
    let accepted_position = Vector3i::new(24, 0, 0);
    let stale_position = Vector3i::new(25, 0, 0);
    for (position, generation) in [(accepted_position, 7), (stale_position, 11)] {
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
    }
    let serial_runs = Arc::new(AtomicUsize::new(0));
    let parallel_runs = Arc::new(AtomicUsize::new(0));
    let stale_runs = Arc::new(AtomicUsize::new(0));
    let serial: Box<dyn ThreadedTask> = Box::new(CountingCompletionFollowUpTask {
        runs: serial_runs.clone(),
    });
    let parallel: Box<dyn ThreadedTask> = Box::new(CountingCompletionFollowUpTask {
        runs: parallel_runs.clone(),
    });
    let stale: Box<dyn ThreadedTask> = Box::new(CountingCompletionFollowUpTask {
        runs: stale_runs.clone(),
    });
    let serial_identity = serial.as_ref() as *const dyn ThreadedTask as *const () as usize;
    let parallel_identity = parallel.as_ref() as *const dyn ThreadedTask as *const () as usize;
    let stale_identity = stale.as_ref() as *const dyn ThreadedTask as *const () as usize;

    for (position, generation, output_generation, followups) in [
        (
            accepted_position,
            7,
            7,
            vec![
                ScheduledTask::new(serial, TaskLane::Serial),
                ScheduledTask::new(parallel, TaskLane::Parallel),
            ],
        ),
        (
            stale_position,
            11,
            10,
            vec![ScheduledTask::new(stale, TaskLane::Serial)],
        ),
    ] {
        let mut task = LoadBlockForTerrainTask::new(
            position,
            0,
            generation,
            core.data.clone(),
            core.stream.clone(),
        );
        task.output = Some(loaded_output(
            &core,
            position,
            output_generation,
            generation,
        ));
        core.raw_completion_inbox.push_back(CompletedTask::new(
            Box::new(task),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            followups,
        ));
    }
    core.try_normalize_raw_completions().unwrap();

    core.fixed_after_prepare_data_conflict_for_test = Some(accepted_position);
    assert!(matches!(
        core.prepare_fixed_viewer_transaction(&[], true, true, true),
        Err(VoxelTerrainRuntimeError::DataMutation(_))
    ));
    assert_eq!(serial_runs.load(Ordering::SeqCst), 0);
    assert_eq!(parallel_runs.load(Ordering::SeqCst), 0);
    assert_eq!(stale_runs.load(Ordering::SeqCst), 0);
    let DurableCompletion::LoadFinished {
        completed: accepted,
        ..
    } = &core.durable_completion_inbox[0]
    else {
        panic!("accepted load owner changed variant")
    };
    assert_eq!(accepted.follow_up_count(), 2);
    assert_eq!(accepted.follow_up_task(0).unwrap().lane(), TaskLane::Serial);
    assert_eq!(
        accepted.follow_up_task(1).unwrap().lane(),
        TaskLane::Parallel
    );
    assert_eq!(
        accepted.follow_up_task(0).unwrap().task() as *const dyn ThreadedTask as *const () as usize,
        serial_identity
    );
    assert_eq!(
        accepted.follow_up_task(1).unwrap().task() as *const dyn ThreadedTask as *const () as usize,
        parallel_identity
    );
    let DurableCompletion::LoadFinished {
        completed: stale_completion,
        ..
    } = &core.durable_completion_inbox[1]
    else {
        panic!("stale load owner changed variant")
    };
    assert_eq!(stale_completion.follow_up_count(), 1);
    assert_eq!(
        stale_completion.follow_up_task(0).unwrap().task() as *const dyn ThreadedTask as *const ()
            as usize,
        stale_identity
    );

    core.data.with_lod_map_mut(0, |map| {
        let removed = map.remove_block(accepted_position);
        assert!(removed.is_some());
    });
    core.try_fixed_viewer_transaction_for_test(&[]).unwrap();
    core.wait_for_pending_tasks();
    assert_eq!(serial_runs.load(Ordering::SeqCst), 1);
    assert_eq!(parallel_runs.load(Ordering::SeqCst), 1);
    assert_eq!(stale_runs.load(Ordering::SeqCst), 0);
    assert!(core.durable_completion_inbox.is_empty());
}

#[test]
fn combined_task_batch_reservation_failure_publishes_nothing() {
    let mut core = make_resident_edit_core();
    let mesh_position = Vector3i::zero();
    let mesh_key = core.request_mesh_update(mesh_position, 0).unwrap();
    let load_position = Vector3i::new(30, 0, 0);
    core.loading_blocks[0].insert(
        load_position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: 41,
            request_state: LoadRequestState::Queued,
            physical_request: None,
        },
    );
    core.blocks_pending_load[0].push(load_position);

    let save_key = SaveKey::new(Vector3i::new(31, 0, 0), 0);
    let save_payload = VoxelBuffer::with_size(Vector3i::splat(2));
    let save_payload_ptr = save_payload.channel_bytes(ChannelId::Type.index()).as_ptr();
    core.save_journal.insert(
        save_key,
        SaveJournalEntry {
            written_unflushed: None,
            active: Some(ActiveSaveAttempt::Pending(PendingSave {
                meta: SaveAttemptMeta {
                    block_revision: 0,
                    generation: 51,
                    retry_count: 0,
                    last_error: None,
                },
                payload: save_payload,
            })),
            queued_newer: VecDeque::new(),
        },
    );
    core.deferred_save_dispatch_keys.push(save_key);

    let completion_position = Vector3i::new(32, 0, 0);
    core.loading_blocks[0].insert(
        completion_position,
        LoadingBlockEntry {
            residency: DataResidencyRefs::with_resident_viewers(1),
            retry_count: 0,
            request_generation: 61,
            request_state: LoadRequestState::InFlight,
            physical_request: None,
        },
    );
    let serial_runs = Arc::new(AtomicUsize::new(0));
    let parallel_runs = Arc::new(AtomicUsize::new(0));
    let serial: Box<dyn ThreadedTask> = Box::new(CountingCompletionFollowUpTask {
        runs: serial_runs.clone(),
    });
    let parallel: Box<dyn ThreadedTask> = Box::new(CountingCompletionFollowUpTask {
        runs: parallel_runs.clone(),
    });
    let mut completed_load = LoadBlockForTerrainTask::new(
        completion_position,
        0,
        61,
        core.data.clone(),
        core.stream.clone(),
    );
    completed_load.output = Some(loaded_output(&core, completion_position, 61, 0xC2));
    core.raw_completion_inbox.push_back(CompletedTask::new(
        Box::new(completed_load),
        TaskLane::Parallel,
        TaskCompletionStatus::Finished,
        vec![
            ScheduledTask::new(serial, TaskLane::Serial),
            ScheduledTask::new(parallel, TaskLane::Parallel),
        ],
    ));
    core.try_normalize_raw_completions().unwrap();
    let completion_descriptor = core.durable_completion_inbox[0].descriptor();
    let before = fixed_rollback_descriptor(&core);
    let event_count = core.event_outbox.len();

    core.task_runner
        .fail_next_prepared_batch_reservation_for_test();
    assert!(matches!(
        core.prepare_fixed_viewer_transaction(&[], false, true, true),
        Err(VoxelTerrainRuntimeError::CompletionDrainCapacityFailed)
    ));
    assert_eq!(fixed_rollback_descriptor(&core), before);
    assert_eq!(core.event_outbox.len(), event_count);
    assert_eq!(core.durable_completion_inbox.len(), 1);
    assert_eq!(
        core.durable_completion_inbox[0].descriptor(),
        completion_descriptor
    );
    assert_eq!(serial_runs.load(Ordering::SeqCst), 0);
    assert_eq!(parallel_runs.load(Ordering::SeqCst), 0);
    assert_eq!(
        core.journal_payload_ptr_for_test(PersistenceOperation::Save {
            location: BlockLocation {
                position: save_key.position,
                lod_index: save_key.lod_index,
            },
            block_revision: 0,
            save_generation: 51,
        }),
        Some(save_payload_ptr)
    );
    assert_eq!(
        core.mesh_maps[0][&mesh_position].requested_revision,
        Some(mesh_key.revision)
    );

    core.prepare_fixed_viewer_transaction(&[], false, true, true)
        .unwrap();
    core.wait_for_pending_tasks();
    assert_eq!(serial_runs.load(Ordering::SeqCst), 1);
    assert_eq!(parallel_runs.load(Ordering::SeqCst), 1);
    assert!(core.durable_completion_inbox.is_empty());
    assert!(core.blocks_pending_load[0].is_empty());
    assert!(core.blocks_pending_update[0].is_empty());
    assert!(matches!(
        core.save_journal[&save_key].active,
        Some(ActiveSaveAttempt::WriteInFlight { .. })
    ));
}

#[test]
fn legacy_partial_snapshot_retirement_guards() {
    let mut core = build_core();
    let position = Vector3i::zero();
    let features = MeshBuildFeatures {
        visuals: true,
        collisions: false,
        variable_lod: false,
    };
    let first_pool = Arc::new(MeshArraysPool::new());
    let first_key = prepare_direct_mesh(&mut core, position);
    core.try_apply_mesh_output(pooled_mesh_output(
        first_pool.clone(),
        first_key,
        features,
        1,
        0,
        false,
    ))
    .unwrap();
    drop(core.try_drain_completed_tasks().unwrap());
    let first_ptr = Arc::as_ptr(core.mesh_maps[0][&position].accepted_upload().unwrap());

    let replacement_pool = Arc::new(MeshArraysPool::new());
    let replacement_key = core.request_mesh_update(position, 0).unwrap();
    core.fail_next_mesh_event_reservation_for_test = true;
    assert!(matches!(
        core.try_apply_mesh_output(pooled_mesh_output(
            replacement_pool.clone(),
            replacement_key,
            features,
            2,
            0,
            false,
        )),
        Err(MeshOutputApplyError::Admitted { .. })
    ));
    assert_eq!(
        Arc::as_ptr(core.mesh_maps[0][&position].accepted_upload().unwrap()),
        first_ptr
    );
    assert_eq!(first_pool.idle_count(), 0);
    assert_eq!(replacement_pool.idle_count(), 0);

    drop(core.try_drain_completed_tasks().unwrap());
    assert_eq!(first_pool.idle_count(), 1);
    assert_eq!(replacement_pool.idle_count(), 0);

    let stale_pool = Arc::new(MeshArraysPool::new());
    let stale_key = core.request_mesh_update(position, 0).unwrap();
    let _newer_key = core.request_mesh_update(position, 0).unwrap();
    core.try_apply_mesh_output(pooled_mesh_output(
        stale_pool.clone(),
        stale_key,
        features,
        1,
        0,
        false,
    ))
    .unwrap();
    assert_eq!(stale_pool.idle_count(), 1);

    let paired_state = ViewerState {
        mesh_box: single_block_box(position),
        demand: MeshDemand {
            visuals: true,
            collisions: false,
        },
        ..ViewerState::default()
    };
    core.paired_viewers.push(PairedViewer {
        id: 9,
        state: paired_state.clone(),
        prev_state: paired_state,
    });
    core.next_render_topology_revision = u64::MAX;
    assert!(matches!(
        core.try_fixed_viewer_transaction_for_test(&[]),
        Err(VoxelTerrainRuntimeError::RenderTopologyRevisionOverflow)
    ));
    assert_eq!(replacement_pool.idle_count(), 0);
    assert!(core.mesh_maps[0].contains_key(&position));

    core.next_render_topology_revision = 100;
    core.try_fixed_viewer_transaction_for_test(&[]).unwrap();
    assert!(!core.mesh_maps[0].contains_key(&position));
    assert_eq!(replacement_pool.idle_count(), 1);
}
