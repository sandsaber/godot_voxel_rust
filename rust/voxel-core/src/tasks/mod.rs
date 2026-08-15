//! Task utilities ported from `util/tasks/`.
//!
//! Phase 4 includes the engine-agnostic task runner used by streaming. Godot
//! scheduler bindings, profiling/debug UI and terrain-specific task types live
//! in later layers.

pub mod async_dependency_tracker;
pub mod cancellation_token;
pub mod task_priority;
pub mod threaded_task;
pub mod threaded_task_runner;

pub use async_dependency_tracker::{
    AsyncDependencyCompletion, AsyncDependencyError, AsyncDependencyTracker,
};
pub use cancellation_token::TaskCancellationToken;
pub use task_priority::TaskPriority;
pub use threaded_task::{
    CompletedTask, RequestCancellation, ScheduledTask, TaskCompletionStatus, TaskLane,
    TaskPanicPhase, TaskRequestTag, TaskRunStatus, ThreadedTask, ThreadedTaskContext,
};
pub use threaded_task_runner::ThreadedTaskRunner;

// Task 4 terrain integration seam. Keep the preallocated publication API in
// normal builds before the terrain transaction starts consuming it.
type PrepareTaskBatchResult =
    Result<threaded_task_runner::PreparedTaskBatch, threaded_task_runner::PreparedTaskBatchError>;
const _: fn(&ThreadedTaskRunner, usize) -> PrepareTaskBatchResult =
    ThreadedTaskRunner::try_prepare_enqueue;
const _: for<'runner> fn(
    &'runner mut ThreadedTaskRunner,
    threaded_task_runner::FilledPreparedTaskBatch,
) -> threaded_task_runner::PreparedTaskWake<'runner> = ThreadedTaskRunner::link_prepared;
const _: fn(
    &mut threaded_task_runner::PreparedTaskBatch,
    ScheduledTask,
) -> Result<(), ScheduledTask> = threaded_task_runner::PreparedTaskBatch::push_reserved;
const _: fn(
    threaded_task_runner::PreparedTaskBatch,
) -> Result<
    threaded_task_runner::FilledPreparedTaskBatch,
    threaded_task_runner::PreparedTaskBatch,
> = threaded_task_runner::PreparedTaskBatch::try_into_filled;
const _: fn(&threaded_task_runner::PreparedTaskBatch) -> bool =
    threaded_task_runner::PreparedTaskBatch::is_full;
