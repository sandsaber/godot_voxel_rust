//! Minimal threaded task runner.
//!
//! Ported from `util/tasks/threaded_task_runner.{h,cpp}`. This implementation
//! keeps the core engine contract needed by stream tasks: priority picking,
//! serial-task gating, postponed requeueing, cooperative cancellation,
//! completed-task draining and idle waiting. Godot-specific debug/profiling
//! surfaces and hot resizing are intentionally deferred.

use super::{
    CompletedTask, ScheduledTask, TaskCompletionStatus, TaskLane, TaskPanicPhase, TaskPriority,
    TaskRunStatus, ThreadedTaskContext,
};
use crate::thread::Semaphore;
use std::cmp::Reverse;
use std::collections::TryReserveError;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Default priority-recompute period. Matches the C++ runner
/// (`_priority_update_period_ms = 32`): cached priorities and cancellation
/// drains run at most every 32 ms, so a worker wake does no per-task
/// `priority()`/`is_cancelled()` virtual dispatches in the common case.
const DEFAULT_PRIORITY_UPDATE_PERIOD: Duration = Duration::from_millis(32);

/// Generic thread pool for owned [`ThreadedTask`] objects.
pub struct ThreadedTaskRunner {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
}

pub(crate) struct PreparedTaskBatch {
    node: Box<StagedTaskBatchNode>,
    remaining_capacity: usize,
}

pub(crate) struct FilledPreparedTaskBatch {
    node: Box<StagedTaskBatchNode>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunnerTaskObservablePhase {
    StagedUnready,
    StagedReady,
    Pending,
    Postponed,
    Refreshing,
    Running,
    Completed,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RunnerTaskObservable {
    pub(crate) task_ptr: usize,
    pub(crate) lane: TaskLane,
    pub(crate) phase: RunnerTaskObservablePhase,
}

#[derive(Debug)]
pub(crate) enum PreparedTaskBatchError {
    Capacity(TryReserveError),
    #[cfg(test)]
    Injected,
}

#[must_use = "dropping the wake token publishes its already-linked task batch"]
pub(crate) struct PreparedTaskWake<'runner> {
    permit_count: usize,
    ready: Arc<AtomicBool>,
    shared: &'runner Shared,
    published: bool,
}

struct StagedTaskBatchNode {
    tasks: Vec<TaskItem>,
    ready: Arc<AtomicBool>,
    next: Option<Box<StagedTaskBatchNode>>,
}

impl PreparedTaskBatch {
    pub(crate) fn push_reserved(&mut self, task: ScheduledTask) -> Result<(), ScheduledTask> {
        if self.remaining_capacity == 0 {
            return Err(task);
        }
        self.node.tasks.push(TaskItem {
            task,
            cached_priority: TaskPriority::min(),
            sequence: 0,
        });
        self.remaining_capacity -= 1;
        Ok(())
    }

    pub(crate) const fn is_full(&self) -> bool {
        self.remaining_capacity == 0
    }

    pub(crate) fn try_into_filled(self) -> Result<FilledPreparedTaskBatch, Self> {
        if !self.is_full() {
            return Err(self);
        }
        Ok(FilledPreparedTaskBatch { node: self.node })
    }

    /// Fill the complete reserved suffix without another allocation.
    pub(crate) fn try_fill_exact(
        mut self,
        tasks: Vec<ScheduledTask>,
    ) -> Result<FilledPreparedTaskBatch, (Self, Vec<ScheduledTask>)> {
        if tasks.len() != self.remaining_capacity {
            return Err((self, tasks));
        }
        self.node
            .tasks
            .extend(tasks.into_iter().map(|task| TaskItem {
                task,
                cached_priority: TaskPriority::min(),
                sequence: 0,
            }));
        self.remaining_capacity = 0;
        Ok(FilledPreparedTaskBatch { node: self.node })
    }
}

impl FilledPreparedTaskBatch {
    /// Recovers every exact task from an unlinked batch in FIFO order.
    ///
    /// The iterator owns and reuses the batch's existing `Vec` allocation, so
    /// rollback can route the task owners into already-reserved destinations
    /// without allocating or changing their `Box` identities. Once a batch is
    /// linked this method is no longer available because linking consumes the
    /// typestate.
    pub(crate) fn into_scheduled_tasks(
        self,
    ) -> impl ExactSizeIterator<Item = ScheduledTask> + DoubleEndedIterator {
        let StagedTaskBatchNode { tasks, ready, next } = *self.node;
        drop(ready);
        drop(next);
        tasks.into_iter().map(|item| item.task)
    }
}

impl fmt::Display for PreparedTaskBatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity(error) => write!(formatter, "prepared task batch capacity: {error}"),
            #[cfg(test)]
            Self::Injected => formatter.write_str("injected prepared task batch failure"),
        }
    }
}

impl Error for PreparedTaskBatchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Capacity(error) => Some(error),
            #[cfg(test)]
            Self::Injected => None,
        }
    }
}

impl PreparedTaskWake<'_> {
    fn publish(&mut self) {
        if self.published {
            return;
        }
        self.ready.store(true, Ordering::Release);
        for _ in 0..self.permit_count {
            self.shared.work_semaphore.post();
        }
        self.published = true;
    }

    pub(crate) fn wake(mut self) {
        self.publish();
    }
}

impl Drop for PreparedTaskWake<'_> {
    fn drop(&mut self) {
        // Once linked, a batch must eventually become runnable. Making the
        // token source-bound and wake-on-drop prevents both cross-runner permit
        // injection and a forgotten token from stranding shutdown forever.
        self.publish();
    }
}

impl ThreadedTaskRunner {
    pub const MAX_THREADS: usize = 128;

    pub fn new(thread_count: usize) -> Self {
        let mut runner = Self {
            shared: Arc::new(Shared::default()),
            handles: Vec::new(),
        };
        // Default throttle matches C++ (`_priority_update_period_ms = 32`).
        runner.shared.lock_state().priority_update_period = DEFAULT_PRIORITY_UPDATE_PERIOD;
        runner.set_thread_count(thread_count);
        runner
    }

    /// Sets how often cached priorities are recomputed and cancelled tasks
    /// drained, mirroring `ThreadedTaskRunner::set_priority_update_period`.
    /// Smaller periods make the runner more responsive to priority changes
    /// (e.g. a moving viewer); larger periods reduce per-task virtual
    /// dispatches under heavy queue pressure. Setting to `Duration::ZERO`
    /// disables throttling and recomputes on every worker wake.
    pub fn set_priority_update_period(&self, period: Duration) {
        self.shared.lock_state().priority_update_period = period;
    }

    pub fn set_thread_count(&mut self, count: usize) {
        if !self.handles.is_empty() {
            self.wait_for_all_tasks();
            self.stop_threads();
        }

        let count = count.min(Self::MAX_THREADS);
        {
            let mut state = self.shared.lock_state();
            state.stopping = false;
        }

        self.handles.reserve(count);
        for index in 0..count {
            let shared = self.shared.clone();
            self.handles
                .push(thread::spawn(move || worker_loop(shared, index as u8)));
        }
    }

    pub fn thread_count(&self) -> usize {
        self.handles.len()
    }

    pub fn enqueue(&self, task: ScheduledTask) {
        let mut batch = self
            .try_prepare_enqueue(1)
            .expect("failed to allocate task ingress");
        if batch.push_reserved(task).is_err() {
            panic!("prepared single-task batch overfilled");
        }
        let filled = batch
            .try_into_filled()
            .unwrap_or_else(|_| panic!("prepared single-task batch is not full"));
        let wake = self.link_prepared_immediate(filled);
        wake.wake();
    }

    pub fn enqueue_many<I>(&self, tasks: I)
    where
        I: IntoIterator<Item = ScheduledTask>,
    {
        let tasks = tasks.into_iter().collect::<Vec<_>>();
        if tasks.is_empty() {
            return;
        }
        let mut batch = self
            .try_prepare_enqueue(tasks.len())
            .expect("failed to allocate task ingress");
        for task in tasks {
            if batch.push_reserved(task).is_err() {
                panic!("prepared multi-task batch overfilled");
            }
        }
        let filled = batch
            .try_into_filled()
            .unwrap_or_else(|_| panic!("prepared multi-task batch is not full"));
        let wake = self.link_prepared_immediate(filled);
        wake.wake();
    }

    pub(crate) fn try_prepare_enqueue(
        &self,
        count: usize,
    ) -> Result<PreparedTaskBatch, PreparedTaskBatchError> {
        #[cfg(test)]
        if self
            .shared
            .fail_next_prepared_batch_reservation
            .swap(false, Ordering::SeqCst)
        {
            return Err(PreparedTaskBatchError::Injected);
        }
        let mut tasks = Vec::new();
        tasks
            .try_reserve_exact(count)
            .map_err(PreparedTaskBatchError::Capacity)?;
        let ready = Arc::new(AtomicBool::new(false));
        Ok(PreparedTaskBatch {
            node: Box::new(StagedTaskBatchNode {
                tasks,
                ready,
                next: None,
            }),
            remaining_capacity: count,
        })
    }

    pub(crate) fn link_prepared(&mut self, batch: FilledPreparedTaskBatch) -> PreparedTaskWake<'_> {
        self.link_prepared_immediate(batch)
    }

    fn link_prepared_immediate(&self, mut batch: FilledPreparedTaskBatch) -> PreparedTaskWake<'_> {
        let permit_count = batch.node.tasks.len();
        let ready = batch.node.ready.clone();
        if permit_count != 0 {
            let mut incoming = self.shared.lock_incoming_batches();
            batch.node.next = incoming.take();
            *incoming = Some(batch.node);
        }
        PreparedTaskWake {
            permit_count,
            ready,
            shared: self.shared.as_ref(),
            published: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn fail_next_prepared_batch_reservation_for_test(&self) {
        self.shared
            .fail_next_prepared_batch_reservation
            .store(true, Ordering::SeqCst);
    }

    /// Stable task-owner identities across every runner-owned observable
    /// phase. The accessor takes runner locks in worker order and never calls
    /// a task virtual method.
    #[cfg(test)]
    pub(crate) fn observable_tasks_for_test(&self) -> Vec<RunnerTaskObservable> {
        let state = self.shared.lock_state();
        let incoming = self.shared.lock_incoming_batches();
        let mut observables = Vec::new();
        let mut node = incoming.as_deref();
        while let Some(batch) = node {
            let phase = if batch.ready.load(Ordering::Acquire) {
                RunnerTaskObservablePhase::StagedReady
            } else {
                RunnerTaskObservablePhase::StagedUnready
            };
            observables.extend(
                batch
                    .tasks
                    .iter()
                    .map(|item| runner_task_observable(&item.task, phase)),
            );
            node = batch.next.as_deref();
        }
        observables.extend(
            state
                .tasks
                .iter()
                .map(|item| runner_task_observable(&item.task, RunnerTaskObservablePhase::Pending)),
        );
        observables.extend(
            state.spinning_tasks.iter().map(|item| {
                runner_task_observable(&item.task, RunnerTaskObservablePhase::Postponed)
            }),
        );
        observables.extend(state.refreshing_tasks_for_test.iter().copied());
        observables.extend(state.running_tasks_for_test.iter().copied());
        observables.extend(
            state
                .completed_tasks
                .iter()
                .map(|completed| RunnerTaskObservable {
                    task_ptr: completed.task() as *const dyn super::ThreadedTask as *const ()
                        as usize,
                    lane: completed.lane(),
                    phase: RunnerTaskObservablePhase::Completed,
                }),
        );
        observables
            .sort_unstable_by_key(|observable| (observable.phase as u8, observable.task_ptr));
        observables
    }

    pub fn wait_for_all_tasks(&self) {
        let mut state = self.shared.lock_state();
        while state.has_pending_or_running_tasks() || self.shared.has_staged_tasks() {
            state = self.shared.wait(state);
        }
    }

    pub fn try_drain_completed_into(
        &mut self,
        destination: &mut VecDeque<CompletedTask>,
    ) -> Result<usize, TryReserveError> {
        let count = self.shared.lock_state().completed_tasks.len();
        destination.try_reserve(count)?;

        let mut state = self.shared.lock_state();
        let count = count.min(state.completed_tasks.len());
        destination.extend(state.completed_tasks.drain(..count));
        Ok(count)
    }

    /// Queued, postponed or running tasks. Completed-but-undrained tasks are
    /// not counted, matching the C++ debug remaining counter.
    pub fn remaining_task_count(&self) -> usize {
        let state = self.shared.lock_state();
        self.shared.incoming_task_count()
            + state.tasks.len()
            + state.spinning_tasks.len()
            + state.running_count
            + state.refreshing_count
    }

    pub fn shutdown(&mut self) {
        if self.handles.is_empty() {
            return;
        }
        self.wait_for_all_tasks();
        self.stop_threads();
    }

    fn stop_threads(&mut self) {
        let thread_count = self.handles.len();
        {
            let mut state = self.shared.lock_state();
            state.stopping = true;
            self.shared.cvar.notify_all();
        }

        for _ in 0..thread_count {
            self.shared.work_semaphore.post();
        }

        for handle in self.handles.drain(..) {
            handle.join().expect("threaded task worker panicked");
        }
    }
}

impl Default for ThreadedTaskRunner {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Drop for ThreadedTaskRunner {
    fn drop(&mut self) {
        self.stop_threads();
    }
}

struct Shared {
    state: Mutex<RunnerState>,
    incoming_batches: Mutex<Option<Box<StagedTaskBatchNode>>>,
    work_semaphore: Semaphore,
    cvar: Condvar,
    #[cfg(test)]
    fail_next_prepared_batch_reservation: AtomicBool,
    #[cfg(test)]
    unready_batch_observations: std::sync::atomic::AtomicUsize,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            state: Mutex::new(RunnerState::default()),
            incoming_batches: Mutex::new(None),
            work_semaphore: Semaphore::new(),
            cvar: Condvar::new(),
            #[cfg(test)]
            fail_next_prepared_batch_reservation: AtomicBool::new(false),
            #[cfg(test)]
            unready_batch_observations: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Shared {
    fn lock_state(&self) -> MutexGuard<'_, RunnerState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn has_staged_tasks(&self) -> bool {
        self.lock_incoming_batches().is_some()
    }

    fn lock_incoming_batches(&self) -> MutexGuard<'_, Option<Box<StagedTaskBatchNode>>> {
        self.incoming_batches
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn incoming_task_count(&self) -> usize {
        let incoming = self.lock_incoming_batches();
        let mut count = 0usize;
        let mut node = incoming.as_deref();
        while let Some(current) = node {
            count = count.saturating_add(current.tasks.len());
            node = current.next.as_deref();
        }
        count
    }

    fn wait<'a>(&self, guard: MutexGuard<'a, RunnerState>) -> MutexGuard<'a, RunnerState> {
        self.cvar
            .wait(guard)
            .unwrap_or_else(PoisonError::into_inner)
    }
}

#[derive(Default)]
struct RunnerState {
    tasks: Vec<TaskItem>,
    tasks_sorted: bool,
    spinning_tasks: VecDeque<TaskItem>,
    completed_tasks: VecDeque<CompletedTask>,
    stopping: bool,
    running_count: usize,
    refreshing_count: usize,
    serial_running: bool,
    prefer_postponed_next: bool,
    next_sequence: u64,
    #[cfg(test)]
    refreshing_tasks_for_test: Vec<RunnerTaskObservable>,
    #[cfg(test)]
    running_tasks_for_test: Vec<RunnerTaskObservable>,
    /// Last time priorities were recomputed and cancelled tasks were drained.
    /// Mirrors `_last_priority_update_time_ms` in the C++ runner.
    last_priority_update: Option<Instant>,
    /// Period of the priority/cancellation refresh. Defaults to 32 ms (matching
    /// C++); exposed via [`ThreadedTaskRunner::set_priority_update_period`].
    priority_update_period: Duration,
}

impl RunnerState {
    fn has_pending_or_running_tasks(&self) -> bool {
        !self.tasks.is_empty()
            || !self.spinning_tasks.is_empty()
            || self.running_count != 0
            || self.refreshing_count != 0
    }

    fn has_queued_tasks(&self) -> bool {
        !self.tasks.is_empty() || !self.spinning_tasks.is_empty()
    }

    /// Returns true when the throttle window has elapsed since the last
    /// priority refresh. Always true on the first refresh so initial picks
    /// don't run with stale `TaskPriority::min()` cache values.
    fn priority_refresh_due(&self, now: Instant) -> bool {
        match self.last_priority_update {
            None => true,
            Some(last) => now.duration_since(last) >= self.priority_update_period,
        }
    }
}

struct TaskItem {
    task: ScheduledTask,
    cached_priority: TaskPriority,
    sequence: u64,
}

#[cfg(test)]
fn runner_task_observable(
    task: &ScheduledTask,
    phase: RunnerTaskObservablePhase,
) -> RunnerTaskObservable {
    RunnerTaskObservable {
        task_ptr: task.task() as *const dyn super::ThreadedTask as *const () as usize,
        lane: task.lane(),
        phase,
    }
}

fn worker_loop(shared: Arc<Shared>, thread_index: u8) {
    loop {
        shared.work_semaphore.wait();

        let (item, refresh_items) = {
            let mut state = shared.lock_state();
            if state.stopping {
                return;
            }

            if drain_staged_tasks(&shared, &mut state) {
                state.tasks_sorted = false;
            }

            let now = Instant::now();
            let refresh_items = if (!state.tasks_sorted || state.priority_refresh_due(now))
                && !state.tasks.is_empty()
            {
                let count = state.tasks.len();
                if let Some(next) = state.refreshing_count.checked_add(count) {
                    state.refreshing_count = next;
                    state.last_priority_update = Some(now);
                    let items = std::mem::take(&mut state.tasks);
                    #[cfg(test)]
                    state
                        .refreshing_tasks_for_test
                        .extend(items.iter().map(|item| {
                            runner_task_observable(
                                &item.task,
                                RunnerTaskObservablePhase::Refreshing,
                            )
                        }));
                    Some(items)
                } else {
                    None
                }
            } else {
                None
            };

            let item = if refresh_items.is_none() {
                pick_next_task(&mut state)
            } else {
                None
            };
            if item.is_some() {
                state.running_count = state
                    .running_count
                    .checked_add(1)
                    .expect("runner running count overflow");
                #[cfg(test)]
                if let Some(item) = item.as_ref() {
                    state.running_tasks_for_test.push(runner_task_observable(
                        &item.task,
                        RunnerTaskObservablePhase::Running,
                    ));
                }
            }
            (item, refresh_items)
        };

        if let Some(items) = refresh_items {
            refresh_task_items(&shared, items);
            continue;
        }
        if let Some(item) = item {
            run_task_item(&shared, thread_index, item);
        }
    }
}

fn drain_staged_tasks(shared: &Shared, state: &mut RunnerState) -> bool {
    let mut incoming_guard = shared.lock_incoming_batches();
    let mut incoming = incoming_guard.take();
    let mut ordered = None;
    while let Some(mut node) = incoming {
        incoming = node.next.take();
        node.next = ordered;
        ordered = Some(node);
    }

    let mut changed = false;
    let mut remaining = None;
    while let Some(mut node) = ordered {
        let next = node.next.take();
        if !node.ready.load(Ordering::Acquire) {
            #[cfg(test)]
            shared
                .unready_batch_observations
                .fetch_add(1, Ordering::SeqCst);
            node.next = next;
            remaining = Some(node);
            break;
        }
        ordered = next;
        for item in &mut node.tasks {
            item.sequence = state.next_sequence;
            state.next_sequence = state
                .next_sequence
                .checked_add(1)
                .expect("runner task sequence overflow");
        }
        changed = true;
        state.tasks.append(&mut node.tasks);
    }

    // Restore the unready FIFO suffix to the newest-first ingress stack.
    let mut newest_first = None;
    while let Some(mut node) = remaining {
        remaining = node.next.take();
        node.next = newest_first;
        newest_first = Some(node);
    }
    *incoming_guard = newest_first;

    changed
}

fn refresh_task_items(shared: &Shared, items: Vec<TaskItem>) {
    let detached_count = items.len();
    #[cfg(test)]
    let detached_task_ptrs = items
        .iter()
        .map(|item| item.task.task() as *const dyn super::ThreadedTask as *const () as usize)
        .collect::<Vec<_>>();
    let mut runnable = Vec::with_capacity(detached_count);
    let mut completed = Vec::new();

    for mut item in items {
        let priority = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            item.task.task_mut().priority()
        }));
        let Ok(priority) = priority else {
            completed.push(complete_item(
                item,
                TaskCompletionStatus::Panicked(TaskPanicPhase::Priority),
                Vec::new(),
            ));
            continue;
        };
        item.cached_priority = priority;

        let cancelled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            item.task.task_mut().is_cancelled()
        }));
        match cancelled {
            Ok(true) => completed.push(complete_item(
                item,
                TaskCompletionStatus::Cancelled,
                Vec::new(),
            )),
            Ok(false) => runnable.push(item),
            Err(_) => completed.push(complete_item(
                item,
                TaskCompletionStatus::Panicked(TaskPanicPhase::Cancellation),
                Vec::new(),
            )),
        }
    }

    let runnable_count = runnable.len();
    let mut state = shared.lock_state();
    #[cfg(test)]
    state
        .refreshing_tasks_for_test
        .retain(|observable| !detached_task_ptrs.contains(&observable.task_ptr));
    state.refreshing_count = state
        .refreshing_count
        .checked_sub(detached_count)
        .expect("runner refreshing count underflow");
    state.completed_tasks.extend(completed);
    state.tasks.extend(runnable);
    state
        .tasks
        .sort_unstable_by_key(|item| (item.cached_priority, Reverse(item.sequence)));
    state.tasks_sorted = true;
    shared.cvar.notify_all();
    drop(state);

    for _ in 0..runnable_count {
        shared.work_semaphore.post();
    }
}

fn pick_next_task(state: &mut RunnerState) -> Option<TaskItem> {
    let prefer_postponed = state.prefer_postponed_next;
    let picked = if prefer_postponed {
        pick_postponed_task(state).or_else(|| pick_prioritized_task(state))
    } else {
        pick_prioritized_task(state).or_else(|| pick_postponed_task(state))
    };

    if let Some(item) = picked {
        state.prefer_postponed_next = !prefer_postponed;
        if item.task.lane() == TaskLane::Serial {
            debug_assert!(!state.serial_running);
            state.serial_running = true;
        }
        return Some(item);
    }

    None
}

fn pick_postponed_task(state: &mut RunnerState) -> Option<TaskItem> {
    for i in 0..state.spinning_tasks.len() {
        if state.spinning_tasks[i].task.lane() == TaskLane::Serial && state.serial_running {
            continue;
        }
        return state.spinning_tasks.remove(i);
    }
    None
}

fn pick_prioritized_task(state: &mut RunnerState) -> Option<TaskItem> {
    if !state.serial_running {
        return state.tasks.pop();
    }

    state
        .tasks
        .iter()
        .rposition(|item| item.task.lane() == TaskLane::Parallel)
        .map(|index| state.tasks.remove(index))
}

fn run_task_item(shared: &Shared, thread_index: u8, mut item: TaskItem) {
    #[cfg(test)]
    let running_task_ptr = item.task.task() as *const dyn super::ThreadedTask as *const () as usize;
    let is_serial = item.task.lane() == TaskLane::Serial;
    let cached_priority = item.cached_priority;

    let cancelled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        item.task.task_mut().is_cancelled()
    }));
    let outcome = match cancelled {
        Ok(true) => TaskExecutionOutcome::Completed(TaskCompletionStatus::Cancelled, Vec::new()),
        Err(_) => TaskExecutionOutcome::Completed(
            TaskCompletionStatus::Panicked(TaskPanicPhase::Cancellation),
            Vec::new(),
        ),
        Ok(false) => {
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                item.task
                    .task_mut()
                    .run(ThreadedTaskContext::new(thread_index, cached_priority))
            }));
            match run {
                Ok(TaskRunStatus::Complete { follow_up_tasks }) => {
                    TaskExecutionOutcome::Completed(TaskCompletionStatus::Finished, follow_up_tasks)
                }
                Ok(TaskRunStatus::Postponed) => TaskExecutionOutcome::Postponed,
                Err(_) => TaskExecutionOutcome::Completed(
                    TaskCompletionStatus::Panicked(TaskPanicPhase::Run),
                    Vec::new(),
                ),
            }
        }
    };

    let mut state = shared.lock_state();
    #[cfg(test)]
    state
        .running_tasks_for_test
        .retain(|observable| observable.task_ptr != running_task_ptr);
    state.running_count = state
        .running_count
        .checked_sub(1)
        .expect("runner running count underflow");
    if is_serial {
        debug_assert!(state.serial_running);
        state.serial_running = false;
    }

    let mut should_post_work = false;
    match outcome {
        TaskExecutionOutcome::Completed(status, follow_up_tasks) => {
            state
                .completed_tasks
                .push_back(complete_item(item, status, follow_up_tasks));
        }
        TaskExecutionOutcome::Postponed => {
            state.spinning_tasks.push_back(item);
            should_post_work = true;
        }
    }

    should_post_work |= is_serial && state.has_queued_tasks();
    shared.cvar.notify_all();
    drop(state);

    if should_post_work {
        shared.work_semaphore.post();
    }
}

enum TaskExecutionOutcome {
    Completed(TaskCompletionStatus, Vec<ScheduledTask>),
    Postponed,
}

fn complete_item(
    item: TaskItem,
    status: TaskCompletionStatus,
    follow_up_tasks: Vec<ScheduledTask>,
) -> CompletedTask {
    let (task, lane) = item.task.into_parts();
    CompletedTask::new(task, lane, status, follow_up_tasks)
}

#[cfg(test)]
mod tests {
    use super::{PreparedTaskBatchError, ThreadedTaskRunner};
    use crate::tasks::{
        CompletedTask, ScheduledTask, TaskCompletionStatus, TaskLane, TaskPanicPhase, TaskPriority,
        TaskRunStatus, ThreadedTask, ThreadedTaskContext,
    };
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    #[derive(Default)]
    struct Counter {
        current: AtomicUsize,
        max: AtomicUsize,
        completed: AtomicUsize,
        applied: AtomicUsize,
    }

    impl Counter {
        fn enter(&self) {
            let current = self.current.fetch_add(1, Ordering::SeqCst) + 1;
            let mut previous = self.max.load(Ordering::SeqCst);
            while previous < current {
                match self.max.compare_exchange_weak(
                    previous,
                    current,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(next) => previous = next,
                }
            }
        }

        fn leave(&self) {
            self.current.fetch_sub(1, Ordering::SeqCst);
            self.completed.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct CountingTask {
        counter: Arc<Counter>,
        sleep: Duration,
        completed: bool,
    }

    impl CountingTask {
        fn new(counter: Arc<Counter>, sleep: Duration) -> Self {
            Self {
                counter,
                sleep,
                completed: false,
            }
        }
    }

    impl ThreadedTask for CountingTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            self.counter.enter();
            thread::sleep(self.sleep);
            self.counter.leave();
            self.completed = true;
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }

        fn apply_result(self: Box<Self>) {
            assert!(self.completed);
            self.counter.applied.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct PriorityTask {
        priority: TaskPriority,
        id: usize,
        order: Arc<Mutex<Vec<usize>>>,
    }

    impl ThreadedTask for PriorityTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            self.order.lock().unwrap().push(self.id);
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }

        fn priority(&mut self) -> TaskPriority {
            self.priority
        }
    }

    struct CancelledTask {
        ran: Arc<AtomicBool>,
        applied: Arc<AtomicBool>,
    }

    impl ThreadedTask for CancelledTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            self.ran.store(true, Ordering::SeqCst);
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }

        fn apply_result(self: Box<Self>) {
            self.applied.store(true, Ordering::SeqCst);
        }

        fn is_cancelled(&mut self) -> bool {
            true
        }
    }

    struct PostponedTask {
        attempts: Arc<AtomicUsize>,
    }

    impl ThreadedTask for PostponedTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                TaskRunStatus::Postponed
            } else {
                TaskRunStatus::Complete {
                    follow_up_tasks: Vec::new(),
                }
            }
        }
    }

    struct EmptyTask;

    impl ThreadedTask for EmptyTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }
    }

    struct RunPanicTask {
        drop_count: Arc<AtomicUsize>,
    }

    struct FinishedMarkerTask {
        ran: Arc<AtomicBool>,
    }

    struct BlockingRunTask {
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }

    impl ThreadedTask for BlockingRunTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }
    }

    struct SignalTask {
        finished: Option<mpsc::Sender<()>>,
        ran: Option<Arc<AtomicBool>>,
    }

    impl ThreadedTask for SignalTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            if let Some(ran) = &self.ran {
                ran.store(true, Ordering::SeqCst);
            }
            if let Some(finished) = self.finished.take() {
                finished.send(()).unwrap();
            }
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }

        fn priority(&mut self) -> TaskPriority {
            TaskPriority::min()
        }
    }

    impl ThreadedTask for FinishedMarkerTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            self.ran.store(true, Ordering::SeqCst);
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }

        fn priority(&mut self) -> TaskPriority {
            TaskPriority::min()
        }
    }

    impl Drop for RunPanicTask {
        fn drop(&mut self) {
            self.drop_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl ThreadedTask for RunPanicTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            panic!("injected run panic");
        }
    }

    #[derive(Clone, Copy)]
    enum BlockingCallbackPhase {
        Priority,
        Cancellation,
    }

    struct BlockingCallbackTask {
        phase: BlockingCallbackPhase,
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
        blocked_once: bool,
    }

    impl BlockingCallbackTask {
        fn block(&self) {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
        }
    }

    impl ThreadedTask for BlockingCallbackTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }

        fn priority(&mut self) -> TaskPriority {
            if !self.blocked_once && matches!(self.phase, BlockingCallbackPhase::Priority) {
                self.block();
                self.blocked_once = true;
            }
            TaskPriority::max()
        }

        fn is_cancelled(&mut self) -> bool {
            if !self.blocked_once && matches!(self.phase, BlockingCallbackPhase::Cancellation) {
                self.block();
                self.blocked_once = true;
            }
            false
        }
    }

    struct PanicCallbackTask {
        phase: TaskPanicPhase,
    }

    impl ThreadedTask for PanicCallbackTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            panic!("run must not be reached after callback panic");
        }

        fn priority(&mut self) -> TaskPriority {
            if self.phase == TaskPanicPhase::Priority {
                panic!("injected priority panic");
            }
            TaskPriority::max()
        }

        fn is_cancelled(&mut self) -> bool {
            if self.phase == TaskPanicPhase::Cancellation {
                panic!("injected cancellation panic");
            }
            false
        }
    }

    struct FollowUpParentTask {
        order: Arc<Mutex<Vec<&'static str>>>,
        follow_up_tasks: Vec<ScheduledTask>,
    }

    impl ThreadedTask for FollowUpParentTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            self.order.lock().unwrap().push("parent");
            TaskRunStatus::Complete {
                follow_up_tasks: std::mem::take(&mut self.follow_up_tasks),
            }
        }
    }

    struct OrderedTask {
        name: &'static str,
        order: Arc<Mutex<Vec<&'static str>>>,
    }

    impl ThreadedTask for OrderedTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            self.order.lock().unwrap().push(self.name);
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }
    }

    fn scheduled(task: Box<dyn ThreadedTask>, lane: TaskLane) -> ScheduledTask {
        ScheduledTask::new(task, lane)
    }

    fn drain(runner: &mut ThreadedTaskRunner) -> VecDeque<CompletedTask> {
        let mut completed = VecDeque::new();
        runner.try_drain_completed_into(&mut completed).unwrap();
        completed
    }

    fn apply_all(runner: &mut ThreadedTaskRunner) {
        for completed in drain(runner) {
            let (task, status, follow_up_tasks) = completed.into_generic_parts();
            runner.enqueue_many(follow_up_tasks);
            if status == TaskCompletionStatus::Finished {
                task.apply_result();
            }
        }
    }

    #[test]
    fn parallel_tasks_run_and_completed_tasks_are_drained_explicitly() {
        let mut runner = ThreadedTaskRunner::new(4);
        let counter = Arc::new(Counter::default());

        for _ in 0..8 {
            runner.enqueue(scheduled(
                Box::new(CountingTask::new(
                    counter.clone(),
                    Duration::from_millis(10),
                )),
                TaskLane::Parallel,
            ));
        }

        runner.wait_for_all_tasks();
        assert_eq!(counter.completed.load(Ordering::SeqCst), 8);
        assert!(counter.max.load(Ordering::SeqCst) <= 4);
        assert_eq!(counter.applied.load(Ordering::SeqCst), 0);

        apply_all(&mut runner);
        assert_eq!(counter.applied.load(Ordering::SeqCst), 8);
        assert!(drain(&mut runner).is_empty());
    }

    #[test]
    fn serial_tasks_do_not_overlap_even_with_multiple_threads() {
        let mut runner = ThreadedTaskRunner::new(4);
        let counter = Arc::new(Counter::default());

        for _ in 0..8 {
            runner.enqueue(scheduled(
                Box::new(CountingTask::new(counter.clone(), Duration::from_millis(5))),
                TaskLane::Serial,
            ));
        }

        runner.wait_for_all_tasks();
        apply_all(&mut runner);

        assert_eq!(counter.completed.load(Ordering::SeqCst), 8);
        assert_eq!(counter.max.load(Ordering::SeqCst), 1);
        assert_eq!(counter.current.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn highest_priority_task_runs_first_when_worker_starts_after_enqueue() {
        let mut runner = ThreadedTaskRunner::new(0);
        let order = Arc::new(Mutex::new(Vec::new()));

        runner.enqueue(scheduled(
            Box::new(PriorityTask {
                priority: TaskPriority::new(1, 0, 0, 0),
                id: 1,
                order: order.clone(),
            }),
            TaskLane::Parallel,
        ));
        runner.enqueue(scheduled(
            Box::new(PriorityTask {
                priority: TaskPriority::new(0, 0, 1, 0),
                id: 2,
                order: order.clone(),
            }),
            TaskLane::Parallel,
        ));

        runner.set_thread_count(1);
        runner.wait_for_all_tasks();

        assert_eq!(*order.lock().unwrap(), vec![2, 1]);
    }

    #[test]
    fn cancelled_tasks_are_completed_without_running() {
        let mut runner = ThreadedTaskRunner::new(1);
        let ran = Arc::new(AtomicBool::new(false));
        let applied = Arc::new(AtomicBool::new(false));

        runner.enqueue(scheduled(
            Box::new(CancelledTask {
                ran: ran.clone(),
                applied: applied.clone(),
            }),
            TaskLane::Parallel,
        ));

        runner.wait_for_all_tasks();
        let completed = drain(&mut runner);
        assert_eq!(
            completed.front().unwrap().status(),
            TaskCompletionStatus::Cancelled
        );
        drop(completed);

        assert!(!ran.load(Ordering::SeqCst));
        assert!(!applied.load(Ordering::SeqCst));
    }

    #[test]
    fn postponed_tasks_are_requeued_until_complete() {
        let mut runner = ThreadedTaskRunner::new(1);
        let attempts = Arc::new(AtomicUsize::new(0));

        runner.enqueue(scheduled(
            Box::new(PostponedTask {
                attempts: attempts.clone(),
            }),
            TaskLane::Parallel,
        ));

        runner.wait_for_all_tasks();
        apply_all(&mut runner);

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn task_box_is_retained_when_run_panics() {
        let mut runner = ThreadedTaskRunner::new(1);
        let drop_count = Arc::new(AtomicUsize::new(0));

        runner.enqueue(scheduled(
            Box::new(RunPanicTask {
                drop_count: drop_count.clone(),
            }),
            TaskLane::Parallel,
        ));
        runner.wait_for_all_tasks();

        let completed = drain(&mut runner);
        assert_eq!(completed.len(), 1);
        assert_eq!(
            completed.front().unwrap().status(),
            TaskCompletionStatus::Panicked(TaskPanicPhase::Run)
        );
        assert_eq!(drop_count.load(Ordering::SeqCst), 0);
        drop(completed);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn panicked_front_completion_does_not_block_later_valid_completion() {
        let mut runner = ThreadedTaskRunner::new(0);
        let drop_count = Arc::new(AtomicUsize::new(0));
        let later_ran = Arc::new(AtomicBool::new(false));
        runner.enqueue(scheduled(
            Box::new(RunPanicTask {
                drop_count: drop_count.clone(),
            }),
            TaskLane::Parallel,
        ));
        runner.enqueue(scheduled(
            Box::new(FinishedMarkerTask {
                ran: later_ran.clone(),
            }),
            TaskLane::Parallel,
        ));

        runner.set_thread_count(1);
        runner.wait_for_all_tasks();
        let completed = drain(&mut runner);

        assert_eq!(completed.len(), 2);
        assert_eq!(
            completed[0].status(),
            TaskCompletionStatus::Panicked(TaskPanicPhase::Run)
        );
        assert_eq!(completed[1].status(), TaskCompletionStatus::Finished);
        assert!(later_ran.load(Ordering::SeqCst));
        assert_eq!(drop_count.load(Ordering::SeqCst), 0);
        drop(completed);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn blocked_callbacks_leave_runner_queries_responsive_and_remain_counted() {
        for phase in [
            BlockingCallbackPhase::Priority,
            BlockingCallbackPhase::Cancellation,
        ] {
            let mut runner = ThreadedTaskRunner::new(1);
            let (entered_tx, entered_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            runner.enqueue(scheduled(
                Box::new(BlockingCallbackTask {
                    phase,
                    entered: entered_tx,
                    release: release_rx,
                    blocked_once: false,
                }),
                TaskLane::Parallel,
            ));
            entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();

            let (remaining_tx, remaining_rx) = mpsc::channel();
            thread::scope(|scope| {
                scope.spawn(|| {
                    remaining_tx.send(runner.remaining_task_count()).unwrap();
                });
                runner.enqueue(scheduled(Box::new(EmptyTask), TaskLane::Parallel));
                let remaining = remaining_rx.recv_timeout(Duration::from_millis(100));
                release_tx.send(()).unwrap();
                assert!(
                    remaining.expect("runner query blocked behind task callback") >= 1,
                    "detached callback batch must remain counted"
                );
            });

            runner.wait_for_all_tasks();
            assert_eq!(drain(&mut runner).len(), 2);
            runner.shutdown();
        }
    }

    #[test]
    fn blocked_callback_keeps_wait_and_shutdown_blocked_until_release() {
        for shutdown in [false, true] {
            let mut runner = ThreadedTaskRunner::new(1);
            let (entered_tx, entered_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            runner.enqueue(scheduled(
                Box::new(BlockingCallbackTask {
                    phase: BlockingCallbackPhase::Priority,
                    entered: entered_tx,
                    release: release_rx,
                    blocked_once: false,
                }),
                TaskLane::Parallel,
            ));
            entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();

            let (finished_tx, finished_rx) = mpsc::channel();
            thread::scope(|scope| {
                let runner_ref = &mut runner;
                scope.spawn(move || {
                    if shutdown {
                        runner_ref.shutdown();
                    } else {
                        runner_ref.wait_for_all_tasks();
                    }
                    finished_tx.send(()).unwrap();
                });
                assert!(finished_rx.recv_timeout(Duration::from_millis(50)).is_err());
                release_tx.send(()).unwrap();
                finished_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            });

            if !shutdown {
                runner.shutdown();
            }
            assert_eq!(runner.thread_count(), 0);
        }
    }

    #[test]
    fn priority_and_cancellation_panics_keep_task_ownership_and_phase() {
        for phase in [TaskPanicPhase::Priority, TaskPanicPhase::Cancellation] {
            let mut runner = ThreadedTaskRunner::new(1);
            runner.enqueue(scheduled(
                Box::new(PanicCallbackTask { phase }),
                TaskLane::Parallel,
            ));
            runner.wait_for_all_tasks();

            let completed = drain(&mut runner);
            assert_eq!(completed.len(), 1);
            assert_eq!(
                completed.front().unwrap().status(),
                TaskCompletionStatus::Panicked(phase)
            );
        }
    }

    #[test]
    fn enqueue_does_not_block_on_worker_queue_lock() {
        let runner = ThreadedTaskRunner::new(0);
        let state_guard = runner.shared.lock_state();
        let enqueued = Arc::new(AtomicBool::new(false));

        thread::scope(|scope| {
            let thread_enqueued = enqueued.clone();
            let runner_ref = &runner;
            scope.spawn(move || {
                runner_ref.enqueue(scheduled(Box::new(EmptyTask), TaskLane::Parallel));
                thread_enqueued.store(true, Ordering::SeqCst);
            });

            thread::sleep(Duration::from_millis(50));
            let completed_while_state_was_locked = enqueued.load(Ordering::SeqCst);
            drop(state_guard);

            assert!(
                completed_while_state_was_locked,
                "enqueue should use a staging queue instead of blocking on the worker queue lock"
            );
        });
    }

    #[test]
    fn follow_up_tasks_are_enqueued_when_completed_tasks_are_drained() {
        let mut runner = ThreadedTaskRunner::new(1);
        let order = Arc::new(Mutex::new(Vec::new()));

        runner.enqueue(scheduled(
            Box::new(FollowUpParentTask {
                order: order.clone(),
                follow_up_tasks: vec![scheduled(
                    Box::new(OrderedTask {
                        name: "child",
                        order: order.clone(),
                    }),
                    TaskLane::Parallel,
                )],
            }),
            TaskLane::Parallel,
        ));
        runner.wait_for_all_tasks();

        apply_all(&mut runner);
        runner.wait_for_all_tasks();
        apply_all(&mut runner);

        assert_eq!(*order.lock().unwrap(), vec!["parent", "child"]);
    }

    #[test]
    fn completed_task_preserves_each_follow_up_lane_and_order() {
        let mut runner = ThreadedTaskRunner::new(1);
        let order = Arc::new(Mutex::new(Vec::new()));
        runner.enqueue(scheduled(
            Box::new(FollowUpParentTask {
                order: order.clone(),
                follow_up_tasks: vec![
                    scheduled(
                        Box::new(OrderedTask {
                            name: "serial",
                            order: order.clone(),
                        }),
                        TaskLane::Serial,
                    ),
                    scheduled(
                        Box::new(OrderedTask {
                            name: "parallel",
                            order,
                        }),
                        TaskLane::Parallel,
                    ),
                ],
            }),
            TaskLane::Parallel,
        ));
        runner.wait_for_all_tasks();

        let mut completed = drain(&mut runner).pop_front().unwrap();
        let follow_ups = completed.take_follow_up_tasks();
        assert_eq!(
            follow_ups
                .iter()
                .map(ScheduledTask::lane)
                .collect::<Vec<_>>(),
            vec![TaskLane::Serial, TaskLane::Parallel]
        );
    }

    #[test]
    fn prepared_batch_preserves_fifo_mixed_lanes_and_follow_up_lane() {
        let mut runner = ThreadedTaskRunner::new(1);
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut batch = runner.try_prepare_enqueue(3).unwrap();
        assert!(batch
            .push_reserved(scheduled(
                Box::new(OrderedTask {
                    name: "a",
                    order: order.clone(),
                }),
                TaskLane::Serial,
            ))
            .is_ok());
        assert!(batch
            .push_reserved(scheduled(
                Box::new(FollowUpParentTask {
                    order: order.clone(),
                    follow_up_tasks: vec![scheduled(
                        Box::new(OrderedTask {
                            name: "follow-up",
                            order: order.clone(),
                        }),
                        TaskLane::Serial,
                    )],
                }),
                TaskLane::Parallel,
            ))
            .is_ok());
        assert!(batch
            .push_reserved(scheduled(
                Box::new(OrderedTask {
                    name: "c",
                    order: order.clone(),
                }),
                TaskLane::Serial,
            ))
            .is_ok());

        let wake = runner.link_prepared(batch.try_into_filled().ok().unwrap());
        wake.wake();
        runner.wait_for_all_tasks();
        let mut completed = drain(&mut runner);

        assert_eq!(*order.lock().unwrap(), vec!["a", "parent", "c"]);
        assert_eq!(
            completed
                .iter()
                .map(CompletedTask::lane)
                .collect::<Vec<_>>(),
            vec![TaskLane::Serial, TaskLane::Parallel, TaskLane::Serial]
        );
        let follow_ups = completed[1].take_follow_up_tasks();
        assert_eq!(follow_ups.len(), 1);
        assert_eq!(follow_ups[0].lane(), TaskLane::Serial);
    }

    #[test]
    fn prepared_batch_stays_hidden_until_exact_wake() {
        let mut runner = ThreadedTaskRunner::new(1);
        let counter = Arc::new(Counter::default());
        let mut batch = runner.try_prepare_enqueue(1).unwrap();
        assert!(batch
            .push_reserved(scheduled(
                Box::new(CountingTask::new(counter.clone(), Duration::ZERO)),
                TaskLane::Parallel,
            ))
            .is_ok());
        assert!(batch.is_full());

        let wake = runner.link_prepared(batch.try_into_filled().ok().unwrap());
        thread::sleep(Duration::from_millis(20));
        assert_eq!(counter.completed.load(Ordering::SeqCst), 0);

        wake.wake();
        runner.wait_for_all_tasks();
        assert_eq!(counter.completed.load(Ordering::SeqCst), 1);
        assert_eq!(drain(&mut runner).len(), 1);
    }

    #[test]
    fn prepared_batch_zero_count_links_and_completes_without_permit() {
        let mut runner = ThreadedTaskRunner::new(0);
        let shared = runner.shared.clone();
        assert_eq!(shared.work_semaphore.count(), 0);
        let batch = runner.try_prepare_enqueue(0).unwrap();
        let wake = runner.link_prepared(batch.try_into_filled().ok().unwrap());
        assert!(!shared.has_staged_tasks());
        wake.wake();
        assert_eq!(shared.work_semaphore.count(), 0);
        runner.wait_for_all_tasks();
        assert_eq!(runner.remaining_task_count(), 0);
    }

    #[test]
    fn prepared_batch_underfill_returns_the_exact_batch() {
        let runner = ThreadedTaskRunner::new(0);
        let counter = Arc::new(Counter::default());
        let mut batch = runner.try_prepare_enqueue(2).unwrap();
        let first_counter = counter.clone();
        assert!(batch
            .push_reserved(scheduled(
                Box::new(CountingTask::new(first_counter, Duration::ZERO)),
                TaskLane::Parallel,
            ))
            .is_ok());
        let expected = batch.node.tasks[0].task.task() as *const dyn ThreadedTask as *const ();
        let mut batch = batch.try_into_filled().err().unwrap();
        assert!(!batch.is_full());
        assert_eq!(batch.node.tasks.len(), 1);
        assert_eq!(batch.node.tasks[0].task.lane(), TaskLane::Parallel);
        assert_eq!(
            batch.node.tasks[0].task.task() as *const dyn ThreadedTask as *const (),
            expected
        );
        assert!(batch
            .push_reserved(scheduled(
                Box::new(CountingTask::new(counter, Duration::ZERO)),
                TaskLane::Serial,
            ))
            .is_ok());
        assert!(batch.try_into_filled().is_ok());
    }

    #[test]
    fn prepared_batch_overfill_returns_the_exact_task() {
        let runner = ThreadedTaskRunner::new(0);
        let mut batch = runner.try_prepare_enqueue(0).unwrap();
        let task = scheduled(
            Box::new(CountingTask::new(
                Arc::new(Counter::default()),
                Duration::ZERO,
            )),
            TaskLane::Serial,
        );
        let expected = task.task() as *const dyn ThreadedTask as *const ();
        let returned = batch.push_reserved(task).err().unwrap();
        assert_eq!(
            returned.task() as *const dyn ThreadedTask as *const (),
            expected
        );
        assert_eq!(returned.lane(), TaskLane::Serial);
    }

    #[test]
    fn filled_prepared_batch_can_restore_exact_tasks_before_link_without_reordering() {
        let runner = ThreadedTaskRunner::new(0);
        let counter = Arc::new(Counter::default());
        let first = scheduled(
            Box::new(CountingTask::new(counter.clone(), Duration::ZERO)),
            TaskLane::Serial,
        );
        let second = scheduled(
            Box::new(CountingTask::new(counter, Duration::ZERO)),
            TaskLane::Parallel,
        );
        let first_identity = first.task() as *const dyn ThreadedTask as *const ();
        let second_identity = second.task() as *const dyn ThreadedTask as *const ();
        let mut batch = runner.try_prepare_enqueue(2).unwrap();
        assert!(batch.push_reserved(first).is_ok());
        assert!(batch.push_reserved(second).is_ok());
        let filled = batch.try_into_filled().ok().unwrap();

        let mut restored = filled.into_scheduled_tasks();
        assert_eq!(restored.len(), 2);
        let first = restored.next().unwrap();
        assert_eq!(first.lane(), TaskLane::Serial);
        assert_eq!(
            first.task() as *const dyn ThreadedTask as *const (),
            first_identity
        );
        assert_eq!(restored.len(), 1);
        let second = restored.next().unwrap();
        assert_eq!(second.lane(), TaskLane::Parallel);
        assert_eq!(
            second.task() as *const dyn ThreadedTask as *const (),
            second_identity
        );
        assert_eq!(restored.len(), 0);
        assert!(restored.next().is_none());
    }

    #[test]
    fn prepared_batch_injected_failure_is_one_shot_before_ownership() {
        let runner = ThreadedTaskRunner::new(0);
        let permits_before = runner.shared.work_semaphore.count();
        runner.fail_next_prepared_batch_reservation_for_test();
        assert!(matches!(
            runner.try_prepare_enqueue(1),
            Err(PreparedTaskBatchError::Injected)
        ));
        assert!(!runner.shared.has_staged_tasks());
        assert_eq!(runner.remaining_task_count(), 0);
        assert_eq!(runner.shared.work_semaphore.count(), permits_before);
        assert!(runner.try_prepare_enqueue(1).is_ok());
    }

    #[test]
    fn prepared_batch_dropped_wake_cannot_strand_shutdown() {
        let mut runner = ThreadedTaskRunner::new(1);
        let counter = Arc::new(Counter::default());
        let mut batch = runner.try_prepare_enqueue(1).unwrap();
        assert!(batch
            .push_reserved(scheduled(
                Box::new(CountingTask::new(counter.clone(), Duration::ZERO)),
                TaskLane::Parallel,
            ))
            .is_ok());
        let wake = runner.link_prepared(batch.try_into_filled().ok().unwrap());
        drop(wake);
        runner.shutdown();
        assert_eq!(counter.completed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn prepared_runner_stale_permit_respects_wake_gate() {
        let mut runner = ThreadedTaskRunner::new(1);
        let shared = runner.shared.clone();
        let (blocker_entered_tx, blocker_entered_rx) = mpsc::channel();
        let (blocker_release_tx, blocker_release_rx) = mpsc::channel();
        let (ordinary_finished_tx, ordinary_finished_rx) = mpsc::channel();
        runner.enqueue(scheduled(
            Box::new(BlockingRunTask {
                entered: blocker_entered_tx,
                release: blocker_release_rx,
            }),
            TaskLane::Parallel,
        ));
        runner.enqueue(scheduled(
            Box::new(SignalTask {
                finished: Some(ordinary_finished_tx),
                ran: None,
            }),
            TaskLane::Parallel,
        ));
        blocker_entered_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();

        let prepared_ran = Arc::new(AtomicBool::new(false));
        let mut batch = runner.try_prepare_enqueue(1).unwrap();
        assert!(batch
            .push_reserved(scheduled(
                Box::new(SignalTask {
                    finished: None,
                    ran: Some(prepared_ran.clone()),
                }),
                TaskLane::Parallel,
            ))
            .is_ok());
        let wake = runner.link_prepared(batch.try_into_filled().ok().unwrap());

        blocker_release_tx.send(()).unwrap();
        ordinary_finished_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while shared.unready_batch_observations.load(Ordering::SeqCst) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "worker never observed the linked unready batch"
            );
            thread::yield_now();
        }
        assert!(!prepared_ran.load(Ordering::SeqCst));

        wake.wake();
        runner.wait_for_all_tasks();
        assert!(prepared_ran.load(Ordering::SeqCst));
    }

    #[test]
    fn shutdown_waits_for_running_tasks_and_joins_workers() {
        let mut runner = ThreadedTaskRunner::new(1);
        let counter = Arc::new(Counter::default());

        runner.enqueue(scheduled(
            Box::new(CountingTask::new(
                counter.clone(),
                Duration::from_millis(10),
            )),
            TaskLane::Parallel,
        ));

        runner.shutdown();
        assert_eq!(runner.thread_count(), 0);
        assert_eq!(counter.completed.load(Ordering::SeqCst), 1);
        apply_all(&mut runner);
        assert_eq!(counter.applied.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn priority_recompute_is_throttle_within_the_update_period_single_worker() {
        // With a long priority-update period, the runner must NOT call
        // `priority()` on every worker wake: cached priorities are reused
        // across wakes until the period elapses. This mirrors the C++
        // `_priority_update_period_ms` throttle.
        //
        // Uses a single worker so the count is deterministic: the worker
        // wakes, runs the initial refresh (one priority() per task), picks a
        // task, runs it, returns to the loop. Without throttling it would
        // re-run the refresh on every wake (16+ priority() calls across
        // iterations); with throttling only the initial 16 happen.
        struct PriorityCountTask {
            priority_calls: Arc<AtomicUsize>,
        }
        impl ThreadedTask for PriorityCountTask {
            fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
                TaskRunStatus::Complete {
                    follow_up_tasks: Vec::new(),
                }
            }
            fn priority(&mut self) -> TaskPriority {
                self.priority_calls.fetch_add(1, Ordering::SeqCst);
                TaskPriority::max()
            }
        }

        let mut runner = ThreadedTaskRunner::new(1);
        runner.set_priority_update_period(Duration::from_secs(60));

        let priority_calls = Arc::new(AtomicUsize::new(0));
        let tasks = (0..16)
            .map(|_| {
                scheduled(
                    Box::new(PriorityCountTask {
                        priority_calls: priority_calls.clone(),
                    }),
                    TaskLane::Parallel,
                )
            })
            .collect::<Vec<_>>();
        // Publish one complete batch. Enqueuing one task at a time races the
        // live worker and legitimately creates additional sort epochs, making
        // an exact single-refresh assertion nondeterministic.
        runner.enqueue_many(tasks);

        runner.wait_for_all_tasks();
        apply_all(&mut runner);

        // Single worker + 60 s window ⇒ exactly one priority() per task.
        assert_eq!(
            priority_calls.load(Ordering::SeqCst),
            16,
            "throttled runner should call priority() exactly once per task"
        );
    }
}
