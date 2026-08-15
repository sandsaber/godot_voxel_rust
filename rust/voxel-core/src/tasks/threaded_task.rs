//! Owned threaded-task contract.
//!
//! The runner always retains the task object, including when user callbacks
//! cancel or panic. Follow-up work carries its own lane and is published only
//! after a finished completion is accepted by its consumer.

use super::TaskPriority;
use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};

/// Stable identity copied into one physical load or mesh task. A request is
/// current only while both the owning runtime epoch and entry generation
/// match this tag exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskRequestTag {
    pub request_epoch: u64,
    pub request_generation: u64,
}

impl TaskRequestTag {
    pub const fn new(request_epoch: u64, request_generation: u64) -> Self {
        Self {
            request_epoch,
            request_generation,
        }
    }
}

/// Shared cooperative cancellation state owned by a physical request entry.
/// Logical viewers and coverage owners never cancel this token directly; the
/// entry signals it only after final aggregate demand reaches zero or a newer
/// physical generation supersedes the task.
#[derive(Debug, Default)]
pub struct RequestCancellation(AtomicBool);

impl RequestCancellation {
    pub const fn new() -> Self {
        Self(AtomicBool::new(false))
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Context passed to a task while it runs on a worker thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadedTaskContext {
    pub thread_index: u8,
    pub task_priority: TaskPriority,
}

impl ThreadedTaskContext {
    pub const fn new(thread_index: u8, task_priority: TaskPriority) -> Self {
        Self {
            thread_index,
            task_priority,
        }
    }
}

/// Execution lane assigned when a task is scheduled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskLane {
    Parallel,
    Serial,
}

/// Outcome of a successful call to [`ThreadedTask::run`].
pub enum TaskRunStatus {
    Complete { follow_up_tasks: Vec<ScheduledTask> },
    Postponed,
}

/// User callback in which a task panicked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskPanicPhase {
    Priority,
    Cancellation,
    Run,
}

/// Terminal state recorded by the runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskCompletionStatus {
    Finished,
    Cancelled,
    Panicked(TaskPanicPhase),
}

/// The only unit accepted by the task runner.
pub struct ScheduledTask {
    task: Box<dyn ThreadedTask>,
    lane: TaskLane,
}

impl ScheduledTask {
    pub fn new(task: Box<dyn ThreadedTask>, lane: TaskLane) -> Self {
        Self { task, lane }
    }

    pub const fn lane(&self) -> TaskLane {
        self.lane
    }

    pub fn task(&self) -> &dyn ThreadedTask {
        self.task.as_ref()
    }

    pub fn task_mut(&mut self) -> &mut dyn ThreadedTask {
        self.task.as_mut()
    }

    pub fn task_any(&self) -> &dyn Any {
        self.task.as_ref()
    }

    pub fn task_any_mut(&mut self) -> &mut dyn Any {
        self.task.as_mut()
    }

    pub fn into_parts(self) -> (Box<dyn ThreadedTask>, TaskLane) {
        (self.task, self.lane)
    }
}

/// Completed task ownership returned by the runner.
pub struct CompletedTask {
    task: Box<dyn ThreadedTask>,
    lane: TaskLane,
    status: TaskCompletionStatus,
    follow_up_tasks: Vec<ScheduledTask>,
}

/// Rollback owner for a completed task's follow-ups while they are staged in
/// an unlinked prepared runner batch.
///
/// `tasks` is the original `Vec` allocation taken from `CompletedTask`. After
/// [`Self::drain`] it remains empty with its capacity intact, so a failed outer
/// transaction can refill it from the recovered batch FIFO without allocating
/// and restore it to the same completed task.
pub(crate) struct PreparedCompletedTaskFollowUps {
    tasks: Vec<ScheduledTask>,
    expected_count: usize,
}

impl CompletedTask {
    pub(crate) fn new(
        task: Box<dyn ThreadedTask>,
        lane: TaskLane,
        status: TaskCompletionStatus,
        follow_up_tasks: Vec<ScheduledTask>,
    ) -> Self {
        Self {
            task,
            lane,
            status,
            follow_up_tasks,
        }
    }

    pub const fn lane(&self) -> TaskLane {
        self.lane
    }

    pub const fn status(&self) -> TaskCompletionStatus {
        self.status
    }

    pub fn follow_up_count(&self) -> usize {
        self.follow_up_tasks.len()
    }

    pub fn follow_up_task(&self, index: usize) -> Option<&ScheduledTask> {
        self.follow_up_tasks.get(index)
    }

    pub fn take_follow_up_tasks(&mut self) -> Vec<ScheduledTask> {
        std::mem::take(&mut self.follow_up_tasks)
    }

    /// Takes the exact follow-up owners while retaining their original `Vec`
    /// allocation in a typed rollback escrow. This operation is infallible and
    /// allocation-free (`CompletedTask` receives an empty zero-capacity Vec).
    pub(crate) fn prepare_follow_up_take(&mut self) -> PreparedCompletedTaskFollowUps {
        let tasks = std::mem::take(&mut self.follow_up_tasks);
        let expected_count = tasks.len();
        PreparedCompletedTaskFollowUps {
            tasks,
            expected_count,
        }
    }

    pub fn task(&self) -> &dyn ThreadedTask {
        self.task.as_ref()
    }

    pub fn task_mut(&mut self) -> &mut dyn ThreadedTask {
        self.task.as_mut()
    }

    pub fn task_any(&self) -> &dyn Any {
        self.task.as_ref()
    }

    pub fn task_any_mut(&mut self) -> &mut dyn Any {
        self.task.as_mut()
    }

    pub fn into_generic_parts(
        self,
    ) -> (
        Box<dyn ThreadedTask>,
        TaskCompletionStatus,
        Vec<ScheduledTask>,
    ) {
        (self.task, self.status, self.follow_up_tasks)
    }

    pub fn into_parts(
        self,
    ) -> (
        Box<dyn ThreadedTask>,
        TaskLane,
        TaskCompletionStatus,
        Vec<ScheduledTask>,
    ) {
        (self.task, self.lane, self.status, self.follow_up_tasks)
    }
}

impl PreparedCompletedTaskFollowUps {
    pub(crate) const fn len(&self) -> usize {
        self.expected_count
    }

    /// Moves the exact follow-up tasks in FIFO order while keeping their
    /// original Vec allocation in this escrow.
    pub(crate) fn drain(
        &mut self,
    ) -> impl ExactSizeIterator<Item = ScheduledTask> + DoubleEndedIterator + '_ {
        self.tasks.drain(..)
    }

    /// Refills the retained original allocation from the front of a recovered
    /// prepared-batch FIFO and restores it to `completed`.
    ///
    /// All preconditions are checked before consuming the iterator, so an
    /// error leaves every supplied owner untouched. On success exactly
    /// `self.len()` tasks are consumed, preserving lane and `Box` identity.
    pub(crate) fn restore<I>(
        mut self,
        completed: &mut CompletedTask,
        recovered: &mut I,
    ) -> Result<(), Self>
    where
        I: ExactSizeIterator<Item = ScheduledTask>,
    {
        if !completed.follow_up_tasks.is_empty()
            || !self.tasks.is_empty()
            || self.tasks.capacity() < self.expected_count
            || recovered.len() < self.expected_count
        {
            return Err(self);
        }

        for _ in 0..self.expected_count {
            let Some(task) = recovered.next() else {
                // `ExactSizeIterator` promises the prechecked length. Keep any
                // already-consumed owners in the retained allocation if a
                // faulty custom iterator violates that contract.
                return Err(self);
            };
            self.tasks.push(task);
        }
        completed.follow_up_tasks = self.tasks;
        Ok(())
    }
}

/// Task runnable by [`super::ThreadedTaskRunner`].
pub trait ThreadedTask: Send + Any + 'static {
    fn run(&mut self, ctx: ThreadedTaskContext) -> TaskRunStatus;

    /// Runs on the caller/main side after a finished completion is drained.
    fn apply_result(self: Box<Self>) {}

    /// Dynamic priority. Higher packed values run first.
    fn priority(&mut self) -> TaskPriority {
        TaskPriority::max()
    }

    /// Cooperative cancellation.
    fn is_cancelled(&mut self) -> bool {
        false
    }

    /// Static debug name for explicit third-party diagnostics.
    ///
    /// The runner itself deliberately never invokes this user callback.
    fn debug_name(&self) -> &'static str {
        "<unnamed>"
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompletedTask, RequestCancellation, ScheduledTask, TaskCompletionStatus, TaskLane,
        TaskRequestTag, TaskRunStatus, ThreadedTask, ThreadedTaskContext,
    };
    use crate::tasks::TaskPriority;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn request_cancellation_is_shared_monotonic_and_tags_keep_both_generations() {
        let cancellation = Arc::new(RequestCancellation::new());
        let worker = cancellation.clone();
        let tag = TaskRequestTag::new(7, 11);

        assert_eq!(tag.request_epoch, 7);
        assert_eq!(tag.request_generation, 11);
        assert!(!worker.is_cancelled());
        cancellation.cancel();
        assert!(worker.is_cancelled());
        cancellation.cancel();
        assert!(worker.is_cancelled());
    }

    struct CompleteTask {
        ran: Arc<AtomicBool>,
    }

    impl ThreadedTask for CompleteTask {
        fn run(&mut self, ctx: ThreadedTaskContext) -> TaskRunStatus {
            assert_eq!(ctx.thread_index, 7);
            assert_eq!(ctx.task_priority, TaskPriority::new(1, 2, 3, 4));
            self.ran.store(true, Ordering::SeqCst);
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }
    }

    #[test]
    fn task_context_carries_thread_index_and_priority() {
        let ran = Arc::new(AtomicBool::new(false));
        let mut task = CompleteTask { ran: ran.clone() };
        let outcome = task.run(ThreadedTaskContext::new(7, TaskPriority::new(1, 2, 3, 4)));

        assert!(matches!(outcome, TaskRunStatus::Complete { .. }));
        assert!(ran.load(Ordering::SeqCst));
    }

    #[test]
    fn default_task_metadata_matches_cpp_contract() {
        let mut task: Box<dyn ThreadedTask> = Box::new(CompleteTask {
            ran: Arc::new(AtomicBool::new(false)),
        });

        assert_eq!(task.priority(), TaskPriority::max());
        assert!(!task.is_cancelled());
        assert_eq!(task.debug_name(), "<unnamed>");
    }

    fn scheduled_identity(task: &ScheduledTask) -> *const () {
        task.task() as *const dyn ThreadedTask as *const ()
    }

    fn make_scheduled(lane: TaskLane) -> ScheduledTask {
        ScheduledTask::new(
            Box::new(CompleteTask {
                ran: Arc::new(AtomicBool::new(false)),
            }),
            lane,
        )
    }

    fn make_completed(follow_up_tasks: Vec<ScheduledTask>) -> CompletedTask {
        CompletedTask::new(
            Box::new(CompleteTask {
                ran: Arc::new(AtomicBool::new(false)),
            }),
            TaskLane::Parallel,
            TaskCompletionStatus::Finished,
            follow_up_tasks,
        )
    }

    #[test]
    fn follow_up_escrows_restore_interleaved_completions_with_original_allocations() {
        let mut first_tasks = Vec::with_capacity(7);
        first_tasks.push(make_scheduled(TaskLane::Serial));
        first_tasks.push(make_scheduled(TaskLane::Parallel));
        let first_allocation = first_tasks.as_ptr();
        let first_identities = first_tasks
            .iter()
            .map(scheduled_identity)
            .collect::<Vec<_>>();

        let mut second_tasks = Vec::with_capacity(5);
        second_tasks.push(make_scheduled(TaskLane::Parallel));
        second_tasks.push(make_scheduled(TaskLane::Serial));
        let second_allocation = second_tasks.as_ptr();
        let second_identities = second_tasks
            .iter()
            .map(scheduled_identity)
            .collect::<Vec<_>>();

        let mut first = make_completed(first_tasks);
        let mut second = make_completed(second_tasks);
        let first_completed_identity = first.task() as *const dyn ThreadedTask as *const ();
        let second_completed_identity = second.task() as *const dyn ThreadedTask as *const ();
        let mut first_escrow = first.prepare_follow_up_take();
        let mut second_escrow = second.prepare_follow_up_take();
        assert_eq!(first_escrow.len(), 2);
        assert_eq!(second_escrow.len(), 2);
        assert_eq!(first.follow_up_count(), 0);
        assert_eq!(second.follow_up_count(), 0);

        // Simulate the combined prepared-batch FIFO: completion A follow-ups,
        // one unrelated planned task, then completion B follow-ups.
        let mut recovered = Vec::with_capacity(5);
        recovered.extend(first_escrow.drain());
        let interleaved = make_scheduled(TaskLane::Serial);
        let interleaved_identity = scheduled_identity(&interleaved);
        recovered.push(interleaved);
        recovered.extend(second_escrow.drain());
        let mut recovered = recovered.into_iter();

        assert!(first_escrow.restore(&mut first, &mut recovered).is_ok());
        let interleaved = recovered.next().unwrap();
        assert_eq!(scheduled_identity(&interleaved), interleaved_identity);
        assert_eq!(interleaved.lane(), TaskLane::Serial);
        assert!(second_escrow.restore(&mut second, &mut recovered).is_ok());
        assert_eq!(recovered.len(), 0);

        assert_eq!(
            first.task() as *const dyn ThreadedTask as *const (),
            first_completed_identity
        );
        assert_eq!(
            second.task() as *const dyn ThreadedTask as *const (),
            second_completed_identity
        );
        assert_eq!(first.follow_up_tasks.as_ptr(), first_allocation);
        assert_eq!(second.follow_up_tasks.as_ptr(), second_allocation);
        assert_eq!(
            first
                .follow_up_tasks
                .iter()
                .map(scheduled_identity)
                .collect::<Vec<_>>(),
            first_identities
        );
        assert_eq!(
            second
                .follow_up_tasks
                .iter()
                .map(scheduled_identity)
                .collect::<Vec<_>>(),
            second_identities
        );
        assert_eq!(
            first
                .follow_up_tasks
                .iter()
                .map(ScheduledTask::lane)
                .collect::<Vec<_>>(),
            vec![TaskLane::Serial, TaskLane::Parallel]
        );
        assert_eq!(
            second
                .follow_up_tasks
                .iter()
                .map(ScheduledTask::lane)
                .collect::<Vec<_>>(),
            vec![TaskLane::Parallel, TaskLane::Serial]
        );
    }

    #[test]
    fn follow_up_escrow_failed_restore_consumes_zero_recovered_tasks() {
        let mut tasks = Vec::with_capacity(4);
        tasks.push(make_scheduled(TaskLane::Serial));
        tasks.push(make_scheduled(TaskLane::Parallel));
        let allocation = tasks.as_ptr();
        let identities = tasks.iter().map(scheduled_identity).collect::<Vec<_>>();
        let mut completed = make_completed(tasks);
        let mut escrow = completed.prepare_follow_up_take();
        let mut detached = escrow.drain();
        let first = detached.next().unwrap();
        let second = detached.next().unwrap();
        drop(detached);

        let mut insufficient = vec![first].into_iter();
        escrow = match escrow.restore(&mut completed, &mut insufficient) {
            Ok(()) => panic!("insufficient recovered FIFO must not restore"),
            Err(escrow) => escrow,
        };
        assert_eq!(insufficient.len(), 1);
        assert_eq!(completed.follow_up_count(), 0);

        let first = insufficient.next().unwrap();
        let mut exact = vec![first, second].into_iter();
        assert!(escrow.restore(&mut completed, &mut exact).is_ok());
        assert_eq!(exact.len(), 0);
        assert_eq!(completed.follow_up_tasks.as_ptr(), allocation);
        assert_eq!(
            completed
                .follow_up_tasks
                .iter()
                .map(scheduled_identity)
                .collect::<Vec<_>>(),
            identities
        );
    }
}
