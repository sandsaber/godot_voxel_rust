//! Threaded stream save task ported from `streams/save_block_data_task.*`.

use crate::constants::voxel_constants::{TASK_PRIORITY_BAND3_DEFAULT, TASK_PRIORITY_SAVE_BAND2};
use crate::engine::StreamingDependency;
use crate::math::Vector3i;
use crate::storage::{BlockLocation, VoxelBuffer};
use crate::streams::{
    BlockDataOutput, PersistenceAcknowledgement, PersistenceIoPhase, SaveTaskTerminal,
    VoxelSaveQuery, VoxelStreamError,
};
use crate::tasks::{
    AsyncDependencyError, AsyncDependencyTracker, ScheduledTask, TaskLane, TaskPriority,
    TaskRunStatus, ThreadedTask, ThreadedTaskContext,
};
use std::sync::{Arc, Mutex};

pub struct SaveBlockDataTask {
    terminal: Option<SaveTaskTerminal>,
    prepared_payload: Option<Arc<PreparedSavePayloadSlot>>,
    physical_attempt_ordinal: u64,
    stream_dependency: Arc<StreamingDependency>,
    tracker: Option<Arc<AsyncDependencyTracker>>,
    tracker_error: Option<AsyncDependencyError>,
    follow_up_tasks: Vec<ScheduledTask>,
    #[cfg(test)]
    panic_before_io_for_test: bool,
    #[cfg(test)]
    panic_after_ack_for_test: bool,
}

/// Preallocated unique owner for a save task whose voxel payload is still
/// resident in a storage transaction. Splitting it yields the exact scheduled
/// task `Box` and one non-cloneable payload-installation capability.
pub(crate) struct PreparedSaveBlockDataTask {
    task: Box<SaveBlockDataTask>,
    installer: PreparedSaveBlockDataPayloadInstaller,
}

/// Unique capability that installs the payload into an already-boxed and
/// already-batched save task. It is intentionally not cloneable: the backing
/// slot starts empty and this capability is the only code allowed to fill it.
pub(crate) struct PreparedSaveBlockDataPayloadInstaller {
    slot: Arc<PreparedSavePayloadSlot>,
}

struct PreparedSavePayloadSlot {
    location: BlockLocation,
    block_revision: u64,
    save_generation: u64,
    payload: Mutex<Option<VoxelBuffer>>,
}

impl PreparedSaveBlockDataTask {
    pub(crate) fn new(
        location: BlockLocation,
        block_revision: u64,
        save_generation: u64,
        stream_dependency: Arc<StreamingDependency>,
        tracker: Option<Arc<AsyncDependencyTracker>>,
        physical_attempt_ordinal: u64,
    ) -> Self {
        let slot = Arc::new(PreparedSavePayloadSlot {
            location,
            block_revision,
            save_generation,
            payload: Mutex::new(None),
        });
        Self {
            task: Box::new(SaveBlockDataTask {
                terminal: None,
                prepared_payload: Some(slot.clone()),
                physical_attempt_ordinal,
                stream_dependency,
                tracker,
                tracker_error: None,
                follow_up_tasks: Vec::new(),
                #[cfg(test)]
                panic_before_io_for_test: false,
                #[cfg(test)]
                panic_after_ack_for_test: false,
            }),
            installer: PreparedSaveBlockDataPayloadInstaller { slot },
        }
    }

    /// Moves the exact task `Box` into a serial scheduled-task owner before a
    /// storage transaction begins. The returned unique installer can fill the
    /// payload slot later without allocating or replacing that `Box`.
    pub(crate) fn into_scheduled(self) -> (ScheduledTask, PreparedSaveBlockDataPayloadInstaller) {
        (
            ScheduledTask::new(self.task, TaskLane::Serial),
            self.installer,
        )
    }

    #[cfg(test)]
    pub(crate) fn set_panic_before_io_for_test(&mut self, enabled: bool) {
        self.task.panic_before_io_for_test = enabled;
    }
}

impl PreparedSaveBlockDataPayloadInstaller {
    /// Installs the only payload. The slot and mutex were allocated before the
    /// storage fence; locking is uncontended because the task is still in an
    /// unlinked prepared batch. Poison recovery keeps this move infallible and
    /// allocation-free even in test panic paths.
    pub(crate) fn install_payload(self, payload: VoxelBuffer) {
        let mut slot = self
            .slot
            .payload
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *slot = Some(payload);
    }
}

impl SaveBlockDataTask {
    pub fn new_voxels(
        position_in_blocks: Vector3i,
        lod_index: u8,
        voxels: VoxelBuffer,
        stream_dependency: Arc<StreamingDependency>,
        tracker: Option<Arc<AsyncDependencyTracker>>,
    ) -> Self {
        Self::new_voxels_at_revision(
            position_in_blocks,
            lod_index,
            voxels,
            stream_dependency,
            tracker,
            0,
        )
    }

    pub fn new_voxels_at_revision(
        position_in_blocks: Vector3i,
        lod_index: u8,
        voxels: VoxelBuffer,
        stream_dependency: Arc<StreamingDependency>,
        tracker: Option<Arc<AsyncDependencyTracker>>,
        block_revision: u64,
    ) -> Self {
        Self::new_voxels_with_generation_at_revision(
            position_in_blocks,
            lod_index,
            voxels,
            stream_dependency,
            tracker,
            block_revision,
            0,
        )
    }

    pub fn new_voxels_with_generation(
        position_in_blocks: Vector3i,
        lod_index: u8,
        voxels: VoxelBuffer,
        stream_dependency: Arc<StreamingDependency>,
        tracker: Option<Arc<AsyncDependencyTracker>>,
        save_generation: u64,
    ) -> Self {
        Self::new_voxels_with_generation_at_revision(
            position_in_blocks,
            lod_index,
            voxels,
            stream_dependency,
            tracker,
            0,
            save_generation,
        )
    }

    pub fn new_voxels_with_generation_at_revision(
        position_in_blocks: Vector3i,
        lod_index: u8,
        voxels: VoxelBuffer,
        stream_dependency: Arc<StreamingDependency>,
        tracker: Option<Arc<AsyncDependencyTracker>>,
        block_revision: u64,
        save_generation: u64,
    ) -> Self {
        Self::new_voxels_with_generation_and_attempt_ordinal(
            BlockLocation {
                position: position_in_blocks,
                lod_index,
            },
            voxels,
            stream_dependency,
            tracker,
            block_revision,
            save_generation,
            0,
        )
    }

    pub(crate) fn new_voxels_with_generation_and_attempt_ordinal(
        location: BlockLocation,
        voxels: VoxelBuffer,
        stream_dependency: Arc<StreamingDependency>,
        tracker: Option<Arc<AsyncDependencyTracker>>,
        block_revision: u64,
        save_generation: u64,
        physical_attempt_ordinal: u64,
    ) -> Self {
        let terminal = Some(SaveTaskTerminal {
            location,
            block_revision,
            save_generation,
            payload: voxels,
            task_panic_phase: None,
            phase: PersistenceIoPhase::BeforeIo,
            acknowledgement: None,
        });
        Self {
            terminal,
            prepared_payload: None,
            physical_attempt_ordinal,
            stream_dependency,
            tracker,
            tracker_error: None,
            follow_up_tasks: Vec::new(),
            #[cfg(test)]
            panic_before_io_for_test: false,
            #[cfg(test)]
            panic_after_ack_for_test: false,
        }
    }

    pub(crate) const fn physical_attempt_ordinal(&self) -> u64 {
        self.physical_attempt_ordinal
    }

    pub fn position_in_blocks(&self) -> Option<Vector3i> {
        self.terminal_ref()
            .map(|terminal| terminal.location.position)
    }

    pub fn lod_index(&self) -> Option<u8> {
        self.terminal_ref()
            .map(|terminal| terminal.location.lod_index)
    }

    pub fn has_run(&self) -> bool {
        self.terminal_ref().is_some_and(|terminal| {
            terminal.phase == PersistenceIoPhase::Acknowledged
                && matches!(
                    terminal.acknowledgement,
                    Some(PersistenceAcknowledgement::Save(_))
                )
        })
    }

    pub const fn had_voxels(&self) -> bool {
        true
    }

    pub fn stream_error(&self) -> Option<&VoxelStreamError> {
        match self.terminal_ref()?.acknowledgement.as_ref()? {
            PersistenceAcknowledgement::Save(Err(error)) => Some(error),
            PersistenceAcknowledgement::Save(Ok(())) | PersistenceAcknowledgement::Flush(_) => None,
        }
    }

    /// Borrows the exact task terminal without invoking stream or tracker code.
    pub fn terminal_ref(&self) -> Option<&SaveTaskTerminal> {
        self.terminal.as_ref()
    }

    /// Moves the exact task terminal without invoking stream or tracker code.
    ///
    /// This is a field move only: terrain completion normalization can call it
    /// after reserving destination capacity without allocating or running a
    /// user callback.
    pub fn take_terminal(&mut self) -> Option<SaveTaskTerminal> {
        self.materialize_prepared_terminal();
        self.terminal.take()
    }

    #[cfg(test)]
    pub(crate) fn set_panic_before_io_for_test(&mut self, enabled: bool) {
        self.panic_before_io_for_test = enabled;
    }

    #[cfg(test)]
    pub(crate) fn set_panic_after_ack_for_test(&mut self, enabled: bool) {
        self.panic_after_ack_for_test = enabled;
    }

    pub const fn tracker_error(&self) -> Option<AsyncDependencyError> {
        self.tracker_error
    }

    /// Legacy one-shot adapter for callers that still consume
    /// [`BlockDataOutput`]. It synthesizes the compatibility value from and
    /// consumes the sole typed terminal; terrain normalization uses
    /// [`Self::take_terminal`] directly.
    pub fn take_output(&mut self) -> Option<BlockDataOutput> {
        let terminal = self.terminal_ref()?;
        if !matches!(
            terminal.acknowledgement,
            Some(PersistenceAcknowledgement::Save(_))
        ) {
            return None;
        }
        let terminal = self
            .take_terminal()
            .expect("save terminal presence was checked before compatibility adaptation");
        match terminal
            .acknowledgement
            .expect("save acknowledgement presence was checked before moving the terminal")
        {
            PersistenceAcknowledgement::Save(Ok(())) => Some(BlockDataOutput::saved(
                terminal.location.position,
                terminal.location.lod_index,
                true,
                terminal.save_generation,
            )),
            PersistenceAcknowledgement::Save(Err(_)) => Some(BlockDataOutput::saved_dropped(
                terminal.location.position,
                terminal.location.lod_index,
                Some(terminal.payload),
                true,
                terminal.save_generation,
            )),
            PersistenceAcknowledgement::Flush(_) => {
                unreachable!("save acknowledgement variant was checked before moving the terminal")
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn retained_payload(&self) -> Option<&VoxelBuffer> {
        self.terminal_ref().map(|terminal| &terminal.payload)
    }

    fn run_save(&mut self) {
        self.materialize_prepared_terminal();
        #[cfg(test)]
        let panic_before_io = self.panic_before_io_for_test;
        #[cfg(test)]
        let panic_after_ack = self.panic_after_ack_for_test;
        let terminal = self
            .terminal
            .as_mut()
            .expect("save terminal remains owned until completion normalization");

        let stream = self.stream_dependency.stream();
        let query = VoxelSaveQuery::new(
            &terminal.payload,
            terminal.location.position,
            terminal.location.lod_index,
        );
        #[cfg(test)]
        if panic_before_io {
            panic!("injected panic before save I/O");
        }
        terminal.phase = PersistenceIoPhase::CallEntered;
        let acknowledgement = stream.save_voxel_block(query);
        terminal.acknowledgement = Some(PersistenceAcknowledgement::Save(acknowledgement));
        terminal.phase = PersistenceIoPhase::Acknowledged;
        #[cfg(test)]
        if panic_after_ack {
            panic!("injected panic after save acknowledgement");
        }

        let save_failed = matches!(
            terminal.acknowledgement,
            Some(PersistenceAcknowledgement::Save(Err(_)))
        );

        if save_failed {
            if let Some(tracker) = &self.tracker {
                tracker.abort();
            }
            self.follow_up_tasks.clear();
        } else if let Some(tracker) = &self.tracker {
            match tracker.post_complete() {
                Ok(completion) => {
                    self.follow_up_tasks = completion.next_tasks;
                }
                Err(error) => {
                    self.tracker_error = Some(error);
                }
            }
        }
    }

    fn materialize_prepared_terminal(&mut self) {
        if self.terminal.is_some() {
            return;
        }
        let Some(slot) = self.prepared_payload.take() else {
            return;
        };
        let payload = slot
            .payload
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("prepared save payload must be installed before task publication");
        self.terminal = Some(SaveTaskTerminal {
            location: slot.location,
            block_revision: slot.block_revision,
            save_generation: slot.save_generation,
            payload,
            task_panic_phase: None,
            phase: PersistenceIoPhase::BeforeIo,
            acknowledgement: None,
        });
    }
}

impl ThreadedTask for SaveBlockDataTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        self.run_save();
        TaskRunStatus::Complete {
            follow_up_tasks: std::mem::take(&mut self.follow_up_tasks),
        }
    }

    fn priority(&mut self) -> TaskPriority {
        TaskPriority::new(0, 0, TASK_PRIORITY_SAVE_BAND2, TASK_PRIORITY_BAND3_DEFAULT)
    }

    fn is_cancelled(&mut self) -> bool {
        false
    }

    fn debug_name(&self) -> &'static str {
        "SaveBlockData"
    }
}

#[cfg(test)]
mod tests {
    use super::{PreparedSaveBlockDataTask, SaveBlockDataTask};
    use crate::constants::voxel_constants::{
        TASK_PRIORITY_BAND3_DEFAULT, TASK_PRIORITY_SAVE_BAND2,
    };
    use crate::engine::StreamingDependency;
    use crate::math::Vector3i;
    use crate::storage::{ChannelId, VoxelBuffer};
    use crate::streams::flush_voxel_stream_task::FlushVoxelStreamTask;
    use crate::streams::{
        BlockDataOutputKind, LoadResult, MemoryStream, PersistenceAcknowledgement,
        PersistenceIoPhase, StreamResult, VoxelSaveQuery, VoxelStream, VoxelStreamError,
    };
    use crate::tasks::{
        AsyncDependencyTracker, CompletedTask, ScheduledTask, TaskCompletionStatus, TaskLane,
        TaskPanicPhase, TaskPriority, TaskRunStatus, ThreadedTask, ThreadedTaskContext,
        ThreadedTaskRunner,
    };
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Default)]
    struct CountingStream {
        saves: AtomicUsize,
        flushes: AtomicUsize,
    }

    impl CountingStream {
        fn saves(&self) -> usize {
            self.saves.load(Ordering::SeqCst)
        }

        fn flushes(&self) -> usize {
            self.flushes.load(Ordering::SeqCst)
        }
    }

    impl VoxelStream for CountingStream {
        fn save_voxel_block(&self, _query: VoxelSaveQuery<'_>) -> StreamResult<()> {
            self.saves.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn flush(&self) -> StreamResult<()> {
            self.flushes.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    struct ErrorSaveStream;

    impl VoxelStream for ErrorSaveStream {
        fn save_voxel_block(&self, _query: VoxelSaveQuery<'_>) -> StreamResult<()> {
            Err(VoxelStreamError::Io("save failed".to_string()))
        }
    }

    struct PanicSaveStream;

    impl VoxelStream for PanicSaveStream {
        fn save_voxel_block(&self, _query: VoxelSaveQuery<'_>) -> StreamResult<()> {
            panic!("injected save panic");
        }
    }

    fn filled_buffer(value: u64) -> VoxelBuffer {
        let mut voxels = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        voxels.set_voxel(value, 1, 0, 0, ChannelId::Type.index());
        voxels
    }

    fn run_owned(task: SaveBlockDataTask) -> CompletedTask {
        let mut runner = ThreadedTaskRunner::new(1);
        runner.enqueue(ScheduledTask::new(Box::new(task), TaskLane::Serial));
        runner.wait_for_all_tasks();
        let mut completed = VecDeque::new();
        runner.try_drain_completed_into(&mut completed).unwrap();
        completed.pop_front().unwrap()
    }

    #[test]
    fn prepared_save_is_batched_before_payload_and_preserves_all_owned_identity() {
        let stream = Arc::new(CountingStream::default());
        let dependency = StreamingDependency::new(stream.clone());
        let location = crate::storage::BlockLocation {
            position: Vector3i::new(19, -4, 7),
            lod_index: 3,
        };
        let mut prepared = PreparedSaveBlockDataTask::new(location, 71, 81, dependency, None, 93);
        prepared.set_panic_before_io_for_test(true);
        let (scheduled, installer) = prepared.into_scheduled();
        let task_identity = scheduled.task() as *const dyn ThreadedTask as *const ();

        let mut runner = ThreadedTaskRunner::new(1);
        let mut batch = runner.try_prepare_enqueue(1).unwrap();
        assert!(batch.push_reserved(scheduled).is_ok());
        let filled = batch.try_into_filled().ok().unwrap();

        let payload = filled_buffer(117);
        let payload_identity = payload.channel_bytes(ChannelId::Type.index()).as_ptr();
        installer.install_payload(payload);

        let wake = runner.link_prepared(filled);
        wake.wake();
        runner.wait_for_all_tasks();
        let mut completed = VecDeque::new();
        runner.try_drain_completed_into(&mut completed).unwrap();
        let mut completed = completed.pop_front().unwrap();

        assert_eq!(
            completed.task() as *const dyn ThreadedTask as *const (),
            task_identity
        );
        assert_eq!(
            completed.status(),
            TaskCompletionStatus::Panicked(TaskPanicPhase::Run)
        );
        assert_eq!(stream.saves(), 0);
        assert_eq!(stream.flushes(), 0);
        let task = completed
            .task_any_mut()
            .downcast_mut::<SaveBlockDataTask>()
            .unwrap();
        assert_eq!(task.physical_attempt_ordinal(), 93);
        let terminal = task.take_terminal().unwrap();
        assert_eq!(terminal.location, location);
        assert_eq!(terminal.block_revision, 71);
        assert_eq!(terminal.save_generation, 81);
        assert_eq!(terminal.phase, PersistenceIoPhase::BeforeIo);
        assert!(terminal.acknowledgement.is_none());
        assert_eq!(
            terminal
                .payload
                .channel_bytes(ChannelId::Type.index())
                .as_ptr(),
            payload_identity
        );
    }

    #[test]
    fn dropping_uninstalled_prepared_save_batch_performs_no_io() {
        let stream = Arc::new(CountingStream::default());
        let dependency = StreamingDependency::new(stream.clone());
        let prepared = PreparedSaveBlockDataTask::new(
            crate::storage::BlockLocation {
                position: Vector3i::new(1, 2, 3),
                lod_index: 0,
            },
            72,
            82,
            dependency,
            None,
            94,
        );
        let (scheduled, installer) = prepared.into_scheduled();
        let runner = ThreadedTaskRunner::new(0);
        let mut batch = runner.try_prepare_enqueue(1).unwrap();
        assert!(batch.push_reserved(scheduled).is_ok());
        let filled = batch.try_into_filled().ok().unwrap();

        drop(installer);
        drop(filled);

        assert_eq!(stream.saves(), 0);
        assert_eq!(stream.flushes(), 0);
        assert_eq!(runner.remaining_task_count(), 0);
    }

    #[test]
    fn run_saves_voxel_buffer_to_stream() {
        let stream = Arc::new(MemoryStream::new());
        let dependency = StreamingDependency::new(stream.clone());
        let position = Vector3i::new(3, 4, 5);
        let mut task = Box::new(SaveBlockDataTask::new_voxels_at_revision(
            position,
            2,
            filled_buffer(42),
            dependency,
            None,
            0,
        ));

        let outcome = task.run(ThreadedTaskContext::new(0, TaskPriority::min()));

        assert!(matches!(outcome, TaskRunStatus::Complete { .. }));
        let mut loaded = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        assert_eq!(
            stream.load_block(position, 2, &mut loaded),
            LoadResult::Found
        );
        assert_eq!(loaded.get_voxel(1, 0, 0, ChannelId::Type.index()), 42);
    }

    #[test]
    fn priority_matches_cpp_save_bands() {
        let stream = Arc::new(MemoryStream::new());
        let dependency = StreamingDependency::new(stream);
        let mut task = SaveBlockDataTask::new_voxels_at_revision(
            Vector3i::default(),
            0,
            filled_buffer(1),
            dependency,
            None,
            0,
        );

        assert_eq!(
            task.priority(),
            TaskPriority::new(0, 0, TASK_PRIORITY_SAVE_BAND2, TASK_PRIORITY_BAND3_DEFAULT,)
        );
        assert!(!task.is_cancelled());
        assert_eq!(task.debug_name(), "SaveBlockData");
    }

    #[test]
    fn run_exposes_saved_block_output() {
        let stream = Arc::new(MemoryStream::new());
        let dependency = StreamingDependency::new(stream);
        let position = Vector3i::new(5, 6, 7);
        let mut task = SaveBlockDataTask::new_voxels_at_revision(
            position,
            3,
            filled_buffer(9),
            dependency,
            None,
            0,
        );

        task.run_save();
        let output = task.take_output().unwrap();

        assert_eq!(output.kind, BlockDataOutputKind::Saved);
        assert_eq!(output.position_in_blocks, position);
        assert_eq!(output.lod_index, 3);
        assert!(output.had_voxels);
        assert!(output.voxels.is_none());
    }

    #[test]
    fn tracker_publishes_one_typed_serial_flush_only_after_final_success() {
        let stream = Arc::new(CountingStream::default());
        let dependency = StreamingDependency::new(stream.clone());
        let tracker = Arc::new(AsyncDependencyTracker::with_count(2));
        tracker
            .set_next_tasks(vec![ScheduledTask::new(
                Box::new(FlushVoxelStreamTask::new(stream.clone(), 404)),
                TaskLane::Serial,
            )])
            .unwrap();

        let mut first = SaveBlockDataTask::new_voxels_at_revision(
            Vector3i::new(0, 0, 0),
            0,
            filled_buffer(1),
            dependency.clone(),
            Some(tracker.clone()),
            0,
        );
        let TaskRunStatus::Complete {
            follow_up_tasks: first_followups,
        } = first.run(ThreadedTaskContext::new(0, TaskPriority::min()))
        else {
            panic!("first save must complete");
        };

        assert!(first_followups.is_empty());
        assert_eq!(stream.saves(), 1);
        assert_eq!(stream.flushes(), 0);
        assert_eq!(tracker.remaining_count(), 1);

        let mut second = SaveBlockDataTask::new_voxels_at_revision(
            Vector3i::new(1, 0, 0),
            0,
            filled_buffer(2),
            dependency,
            Some(tracker.clone()),
            0,
        );
        let TaskRunStatus::Complete {
            follow_up_tasks: final_followups,
        } = second.run(ThreadedTaskContext::new(0, TaskPriority::min()))
        else {
            panic!("final save must complete");
        };

        assert_eq!(stream.saves(), 2);
        assert_eq!(stream.flushes(), 0);
        assert!(tracker.is_complete());
        assert_eq!(final_followups.len(), 1);
        assert_eq!(final_followups[0].lane(), TaskLane::Serial);
        let flush = final_followups[0]
            .task_any()
            .downcast_ref::<FlushVoxelStreamTask>()
            .expect("tracker follow-up must be the typed flush task");
        let terminal = flush.terminal_ref().unwrap();
        assert_eq!(terminal.checkpoint_generation, 404);
        assert_eq!(terminal.phase, PersistenceIoPhase::BeforeIo);
        assert!(terminal.acknowledgement.is_none());
    }

    #[test]
    fn save_stream_error_aborts_tracker_and_publishes_no_followups() {
        let stream: Arc<dyn VoxelStream> = Arc::new(ErrorSaveStream);
        let dependency = StreamingDependency::new(stream.clone());
        let tracker = Arc::new(AsyncDependencyTracker::with_count(1));
        tracker
            .set_next_tasks(vec![ScheduledTask::new(
                Box::new(FlushVoxelStreamTask::new(stream, 405)),
                TaskLane::Serial,
            )])
            .unwrap();
        let mut task = SaveBlockDataTask::new_voxels_at_revision(
            Vector3i::default(),
            0,
            filled_buffer(1),
            dependency,
            Some(tracker.clone()),
            0,
        );

        let TaskRunStatus::Complete { follow_up_tasks } =
            task.run(ThreadedTaskContext::new(0, TaskPriority::min()))
        else {
            panic!("save task must terminate after returned stream error");
        };
        assert!(matches!(
            task.stream_error(),
            Some(VoxelStreamError::Io(message)) if message == "save failed"
        ));
        assert!(task.has_run());
        let output = task.take_output().unwrap();

        assert!(follow_up_tasks.is_empty());
        assert!(tracker.is_aborted());
        assert_eq!(tracker.remaining_count(), 1);
        assert_eq!(output.kind, BlockDataOutputKind::Saved);
        assert!(output.dropped);
        assert!(output.voxels.is_some());
        assert!(output.had_voxels);
        assert!(!task.has_run());
        assert!(task.stream_error().is_none());
    }

    #[test]
    fn failed_save_output_returns_generation_and_voxels() {
        let stream: Arc<dyn VoxelStream> = Arc::new(ErrorSaveStream);
        let dependency = StreamingDependency::new(stream);
        let mut task = SaveBlockDataTask::new_voxels_with_generation_at_revision(
            Vector3i::new(1, 2, 3),
            0,
            filled_buffer(55),
            dependency,
            None,
            32,
            42,
        );

        task.run_save();
        let output = task.take_output().unwrap();

        assert_eq!(output.kind, BlockDataOutputKind::Saved);
        assert_eq!(output.save_generation, 42);
        assert!(output.dropped);
        assert_eq!(
            output
                .voxels
                .unwrap()
                .get_voxel(1, 0, 0, ChannelId::Type.index()),
            55
        );
    }

    #[test]
    fn successful_save_output_keeps_generation_and_drops_local_payload() {
        let stream = Arc::new(MemoryStream::new());
        let dependency = StreamingDependency::new(stream);
        let mut task = SaveBlockDataTask::new_voxels_with_generation_at_revision(
            Vector3i::new(1, 2, 3),
            0,
            filled_buffer(55),
            dependency,
            None,
            33,
            43,
        );

        task.run_save();
        let output = task.take_output().unwrap();

        assert_eq!(output.save_generation, 43);
        assert!(!output.dropped);
        assert!(output.voxels.is_none());
    }

    #[test]
    fn acknowledged_terminal_retains_payload_until_terrain_normalizes() {
        let stream = Arc::new(MemoryStream::new());
        let dependency = StreamingDependency::new(stream);
        let mut task = SaveBlockDataTask::new_voxels_with_generation_at_revision(
            Vector3i::new(1, 2, 3),
            0,
            filled_buffer(55),
            dependency,
            None,
            34,
            44,
        );

        task.run_save();
        let terminal = task.take_terminal().unwrap();

        assert_eq!(terminal.block_revision, 34);
        assert_eq!(terminal.save_generation, 44);
        assert_eq!(terminal.phase, PersistenceIoPhase::Acknowledged);
        assert!(matches!(
            terminal.acknowledgement,
            Some(PersistenceAcknowledgement::Save(Ok(())))
        ));
        assert_eq!(
            terminal.payload.get_voxel(1, 0, 0, ChannelId::Type.index()),
            55
        );
    }

    #[test]
    fn taking_terminal_consumes_the_only_persistence_result() {
        let stream: Arc<dyn VoxelStream> = Arc::new(ErrorSaveStream);
        let mut task = SaveBlockDataTask::new_voxels_with_generation_at_revision(
            Vector3i::default(),
            0,
            filled_buffer(1),
            StreamingDependency::new(stream),
            None,
            306,
            406,
        );

        task.run(ThreadedTaskContext::new(0, TaskPriority::min()));
        let terminal = task.take_terminal().unwrap();

        assert_eq!(terminal.block_revision, 306);
        assert_eq!(terminal.save_generation, 406);
        assert!(matches!(
            terminal.acknowledgement,
            Some(PersistenceAcknowledgement::Save(Err(
                VoxelStreamError::Io(ref message)
            ))) if message == "save failed"
        ));
        assert!(task.stream_error().is_none());
        assert!(task.take_output().is_none());
    }

    #[test]
    fn compatibility_state_exists_only_while_the_authoritative_terminal_is_retained() {
        let stream = Arc::new(MemoryStream::new());
        let position = Vector3i::new(13, 14, 15);
        let mut task = SaveBlockDataTask::new_voxels_at_revision(
            position,
            3,
            filled_buffer(1),
            StreamingDependency::new(stream),
            None,
            203,
        );

        assert_eq!(task.position_in_blocks(), Some(position));
        assert_eq!(task.lod_index(), Some(3));
        assert!(!task.has_run());

        task.run_save();

        assert_eq!(task.position_in_blocks(), Some(position));
        assert_eq!(task.lod_index(), Some(3));
        assert!(task.has_run());

        let terminal = task.take_terminal().unwrap();
        assert_eq!(terminal.location.position, position);
        assert_eq!(terminal.location.lod_index, 3);
        assert_eq!(terminal.block_revision, 203);
        assert_eq!(task.position_in_blocks(), None);
        assert_eq!(task.lod_index(), None);
        assert!(!task.has_run());
    }

    #[test]
    fn save_panic_retains_owned_voxel_payload() {
        let dependency = StreamingDependency::new(Arc::new(PanicSaveStream));
        let mut task = SaveBlockDataTask::new_voxels_at_revision(
            Vector3i::default(),
            0,
            filled_buffer(7),
            dependency,
            None,
            0,
        );

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            task.run(ThreadedTaskContext::new(0, TaskPriority::min()))
        }));

        assert!(result.is_err());
        assert!(task.terminal_ref().is_some());
    }

    #[test]
    fn persistence_panic_before_save_io_retains_exact_payload_and_runner_phase() {
        let stream = Arc::new(CountingStream::default());
        let dependency = StreamingDependency::new(stream.clone());
        let mut task = SaveBlockDataTask::new_voxels_with_generation_at_revision(
            Vector3i::new(4, 5, 6),
            2,
            filled_buffer(71),
            dependency,
            None,
            801,
            901,
        );
        let payload_identity = task
            .terminal_ref()
            .unwrap()
            .payload
            .channel_bytes(ChannelId::Type.index())
            .as_ptr();
        task.set_panic_before_io_for_test(true);

        let mut completed = run_owned(task);
        assert_eq!(
            completed.status(),
            TaskCompletionStatus::Panicked(TaskPanicPhase::Run)
        );
        assert_eq!(stream.saves(), 0);
        assert_eq!(stream.flushes(), 0);

        let terminal = completed
            .task_any_mut()
            .downcast_mut::<SaveBlockDataTask>()
            .unwrap()
            .take_terminal()
            .unwrap();

        assert_eq!(terminal.location.position, Vector3i::new(4, 5, 6));
        assert_eq!(terminal.location.lod_index, 2);
        assert_eq!(terminal.block_revision, 801);
        assert_eq!(terminal.save_generation, 901);
        assert_eq!(terminal.phase, PersistenceIoPhase::BeforeIo);
        assert_eq!(terminal.task_panic_phase, None);
        assert!(terminal.acknowledgement.is_none());
        assert_eq!(
            terminal
                .payload
                .channel_bytes(ChannelId::Type.index())
                .as_ptr(),
            payload_identity
        );
    }

    #[test]
    fn persistence_panic_during_save_io_retains_call_entered_and_exact_payload() {
        let dependency = StreamingDependency::new(Arc::new(PanicSaveStream));
        let task = SaveBlockDataTask::new_voxels_with_generation_at_revision(
            Vector3i::new(-7, 8, 9),
            1,
            filled_buffer(72),
            dependency,
            None,
            802,
            902,
        );
        let payload_identity = task
            .terminal_ref()
            .unwrap()
            .payload
            .channel_bytes(ChannelId::Type.index())
            .as_ptr();

        let mut completed = run_owned(task);
        assert_eq!(
            completed.status(),
            TaskCompletionStatus::Panicked(TaskPanicPhase::Run)
        );
        let terminal = completed
            .task_any_mut()
            .downcast_mut::<SaveBlockDataTask>()
            .unwrap()
            .take_terminal()
            .unwrap();

        assert_eq!(terminal.block_revision, 802);
        assert_eq!(terminal.save_generation, 902);
        assert_eq!(terminal.phase, PersistenceIoPhase::CallEntered);
        assert_eq!(terminal.task_panic_phase, None);
        assert!(terminal.acknowledgement.is_none());
        assert_eq!(
            terminal
                .payload
                .channel_bytes(ChannelId::Type.index())
                .as_ptr(),
            payload_identity
        );
    }

    #[test]
    fn persistence_panic_after_save_ack_retains_ack_and_never_flushes_inline() {
        let stream = Arc::new(CountingStream::default());
        let dependency = StreamingDependency::new(stream.clone());
        let mut task = SaveBlockDataTask::new_voxels_with_generation_at_revision(
            Vector3i::new(10, 11, 12),
            3,
            filled_buffer(73),
            dependency,
            None,
            803,
            903,
        );
        let payload_identity = task
            .terminal_ref()
            .unwrap()
            .payload
            .channel_bytes(ChannelId::Type.index())
            .as_ptr();
        task.set_panic_after_ack_for_test(true);

        let mut completed = run_owned(task);
        assert_eq!(
            completed.status(),
            TaskCompletionStatus::Panicked(TaskPanicPhase::Run)
        );
        assert_eq!(stream.saves(), 1);
        assert_eq!(stream.flushes(), 0);
        let task = completed
            .task_any_mut()
            .downcast_mut::<SaveBlockDataTask>()
            .unwrap();
        assert_eq!(
            task.terminal_ref().unwrap().phase,
            PersistenceIoPhase::Acknowledged
        );
        assert!(matches!(
            task.terminal_ref().unwrap().acknowledgement,
            Some(PersistenceAcknowledgement::Save(Ok(())))
        ));
        assert!(task.has_run());

        let terminal = task.take_terminal().unwrap();
        assert_eq!(terminal.block_revision, 803);
        assert_eq!(terminal.save_generation, 903);
        assert_eq!(terminal.phase, PersistenceIoPhase::Acknowledged);
        assert_eq!(terminal.task_panic_phase, None);
        assert!(matches!(
            terminal.acknowledgement,
            Some(PersistenceAcknowledgement::Save(Ok(())))
        ));
        assert_eq!(
            terminal
                .payload
                .channel_bytes(ChannelId::Type.index())
                .as_ptr(),
            payload_identity
        );
        assert_eq!(stream.saves(), 1);
        assert_eq!(stream.flushes(), 0);
        assert!(!task.has_run());
    }
}
