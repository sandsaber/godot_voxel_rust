//! Internal threaded task for making prior voxel-stream writes durable.

use crate::constants::voxel_constants::{TASK_PRIORITY_BAND3_DEFAULT, TASK_PRIORITY_SAVE_BAND2};
use crate::streams::{
    FlushTaskTerminal, PersistenceAcknowledgement, PersistenceIoPhase, VoxelStream,
};
use crate::tasks::{TaskPriority, TaskRunStatus, ThreadedTask, ThreadedTaskContext};
use std::sync::Arc;

pub(crate) struct FlushVoxelStreamTask {
    stream: Arc<dyn VoxelStream>,
    terminal: Option<FlushTaskTerminal>,
    physical_attempt_ordinal: u64,
    #[cfg(test)]
    panic_before_io_for_test: bool,
    #[cfg(test)]
    panic_after_ack_for_test: bool,
}

impl FlushVoxelStreamTask {
    #[cfg(test)]
    pub(crate) fn new(stream: Arc<dyn VoxelStream>, checkpoint_generation: u64) -> Self {
        Self::new_with_attempt_ordinal(stream, checkpoint_generation, 0)
    }

    pub(crate) fn new_with_attempt_ordinal(
        stream: Arc<dyn VoxelStream>,
        checkpoint_generation: u64,
        physical_attempt_ordinal: u64,
    ) -> Self {
        Self {
            stream,
            terminal: Some(FlushTaskTerminal {
                checkpoint_generation,
                task_panic_phase: None,
                phase: PersistenceIoPhase::BeforeIo,
                acknowledgement: None,
            }),
            physical_attempt_ordinal,
            #[cfg(test)]
            panic_before_io_for_test: false,
            #[cfg(test)]
            panic_after_ack_for_test: false,
        }
    }

    pub(crate) const fn physical_attempt_ordinal(&self) -> u64 {
        self.physical_attempt_ordinal
    }

    pub(crate) fn terminal_ref(&self) -> Option<&FlushTaskTerminal> {
        self.terminal.as_ref()
    }

    pub(crate) fn take_terminal(&mut self) -> Option<FlushTaskTerminal> {
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
}

impl ThreadedTask for FlushVoxelStreamTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        #[cfg(test)]
        let panic_before_io = self.panic_before_io_for_test;
        #[cfg(test)]
        let panic_after_ack = self.panic_after_ack_for_test;
        let terminal = self
            .terminal
            .as_mut()
            .expect("flush terminal remains owned until completion normalization");
        #[cfg(test)]
        if panic_before_io {
            panic!("injected panic before flush I/O");
        }
        terminal.phase = PersistenceIoPhase::CallEntered;
        let acknowledgement = self.stream.flush();
        terminal.acknowledgement = Some(PersistenceAcknowledgement::Flush(acknowledgement));
        terminal.phase = PersistenceIoPhase::Acknowledged;
        #[cfg(test)]
        if panic_after_ack {
            panic!("injected panic after flush acknowledgement");
        }
        TaskRunStatus::Complete {
            follow_up_tasks: Vec::new(),
        }
    }

    fn priority(&mut self) -> TaskPriority {
        TaskPriority::new(0, 0, TASK_PRIORITY_SAVE_BAND2, TASK_PRIORITY_BAND3_DEFAULT)
    }

    fn is_cancelled(&mut self) -> bool {
        false
    }

    fn debug_name(&self) -> &'static str {
        "FlushVoxelStream"
    }
}

#[cfg(test)]
mod tests {
    use super::FlushVoxelStreamTask;
    use crate::streams::{
        PersistenceAcknowledgement, PersistenceIoPhase, StreamResult, VoxelStream,
    };
    use crate::tasks::{
        CompletedTask, ScheduledTask, TaskCompletionStatus, TaskLane, TaskPanicPhase,
        ThreadedTaskRunner,
    };
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Default)]
    struct CountingFlushStream {
        calls: AtomicUsize,
    }

    impl VoxelStream for CountingFlushStream {
        fn flush(&self) -> StreamResult<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[derive(Default)]
    struct PanicFlushStream {
        calls: AtomicUsize,
    }

    impl VoxelStream for PanicFlushStream {
        fn flush(&self) -> StreamResult<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            panic!("injected flush panic");
        }
    }

    fn run_owned(task: FlushVoxelStreamTask) -> CompletedTask {
        let mut runner = ThreadedTaskRunner::new(1);
        runner.enqueue(ScheduledTask::new(Box::new(task), TaskLane::Serial));
        runner.wait_for_all_tasks();
        let mut completed = VecDeque::new();
        runner.try_drain_completed_into(&mut completed).unwrap();
        completed.pop_front().unwrap()
    }

    #[test]
    fn persistence_panic_before_flush_io_retains_checkpoint_and_runner_phase() {
        let stream = Arc::new(CountingFlushStream::default());
        let mut task = FlushVoxelStreamTask::new(stream.clone(), 801);
        task.set_panic_before_io_for_test(true);

        let mut completed = run_owned(task);
        assert_eq!(
            completed.status(),
            TaskCompletionStatus::Panicked(TaskPanicPhase::Run)
        );
        assert_eq!(stream.calls.load(Ordering::SeqCst), 0);

        let terminal = completed
            .task_any_mut()
            .downcast_mut::<FlushVoxelStreamTask>()
            .unwrap()
            .take_terminal()
            .unwrap();

        assert_eq!(terminal.checkpoint_generation, 801);
        assert_eq!(terminal.phase, PersistenceIoPhase::BeforeIo);
        assert_eq!(terminal.task_panic_phase, None);
        assert!(terminal.acknowledgement.is_none());
    }

    #[test]
    fn persistence_panic_during_flush_io_retains_call_entered_and_checkpoint() {
        let stream = Arc::new(PanicFlushStream::default());
        let task = FlushVoxelStreamTask::new(stream.clone(), 802);

        let mut completed = run_owned(task);
        assert_eq!(
            completed.status(),
            TaskCompletionStatus::Panicked(TaskPanicPhase::Run)
        );
        assert_eq!(stream.calls.load(Ordering::SeqCst), 1);

        let terminal = completed
            .task_any_mut()
            .downcast_mut::<FlushVoxelStreamTask>()
            .unwrap()
            .take_terminal()
            .unwrap();
        assert_eq!(terminal.checkpoint_generation, 802);
        assert_eq!(terminal.phase, PersistenceIoPhase::CallEntered);
        assert_eq!(terminal.task_panic_phase, None);
        assert!(terminal.acknowledgement.is_none());
    }

    #[test]
    fn persistence_panic_after_flush_ack_retains_exact_ack_and_one_call() {
        let stream = Arc::new(CountingFlushStream::default());
        let mut task = FlushVoxelStreamTask::new(stream.clone(), 803);
        task.set_panic_after_ack_for_test(true);

        let mut completed = run_owned(task);
        assert_eq!(
            completed.status(),
            TaskCompletionStatus::Panicked(TaskPanicPhase::Run)
        );
        assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
        let task = completed
            .task_any_mut()
            .downcast_mut::<FlushVoxelStreamTask>()
            .unwrap();
        assert_eq!(
            task.terminal_ref().unwrap().phase,
            PersistenceIoPhase::Acknowledged
        );
        assert!(matches!(
            task.terminal_ref().unwrap().acknowledgement,
            Some(PersistenceAcknowledgement::Flush(Ok(())))
        ));

        let terminal = task.take_terminal().unwrap();
        assert_eq!(terminal.checkpoint_generation, 803);
        assert_eq!(terminal.phase, PersistenceIoPhase::Acknowledged);
        assert_eq!(terminal.task_panic_phase, None);
        assert!(matches!(
            terminal.acknowledgement,
            Some(PersistenceAcknowledgement::Flush(Ok(())))
        ));
        assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn physical_attempt_ordinal_is_private_metadata_not_terminal_identity() {
        let stream = Arc::new(CountingFlushStream::default());
        let task = FlushVoxelStreamTask::new_with_attempt_ordinal(stream, 804, 91);

        assert_eq!(task.physical_attempt_ordinal(), 91);
        assert_eq!(task.terminal_ref().unwrap().checkpoint_generation, 804);
    }
}
