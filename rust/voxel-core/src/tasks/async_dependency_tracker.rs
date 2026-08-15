//! Thread-safe dependency countdown for batches of asynchronous streaming tasks.

use crate::tasks::threaded_task::ScheduledTask;
use std::sync::{Mutex, MutexGuard};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncDependencyError {
    NegativeCount,
    CountAlreadySet,
    TasksAlreadyStarted,
    NextTasksAlreadySet,
    PostCompleteAfterAbort,
    PostCompleteTooManyTimes,
}

impl std::fmt::Display for AsyncDependencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NegativeCount => write!(f, "dependency count cannot be negative"),
            Self::CountAlreadySet => write!(f, "dependency count was already set"),
            Self::TasksAlreadyStarted => write!(f, "dependency tasks have already started"),
            Self::NextTasksAlreadySet => write!(f, "dependency next tasks were already set"),
            Self::PostCompleteAfterAbort => {
                write!(f, "cannot complete an aborted dependency tracker")
            }
            Self::PostCompleteTooManyTimes => {
                write!(f, "dependency completion was posted too many times")
            }
        }
    }
}

impl std::error::Error for AsyncDependencyError {}

pub struct AsyncDependencyCompletion {
    pub remaining_count: i32,
    pub was_last: bool,
    pub next_tasks: Vec<ScheduledTask>,
}

/// Tracks completion of a fixed-size async task batch.
pub struct AsyncDependencyTracker {
    state: Mutex<AsyncDependencyState>,
}

struct AsyncDependencyState {
    count: i32,
    aborted: bool,
    tasks_have_started: bool,
    count_was_set: bool,
    next_tasks: Vec<ScheduledTask>,
}

impl Default for AsyncDependencyTracker {
    fn default() -> Self {
        Self {
            state: Mutex::new(AsyncDependencyState {
                count: 0,
                aborted: false,
                tasks_have_started: false,
                count_was_set: false,
                next_tasks: Vec::new(),
            }),
        }
    }
}

impl AsyncDependencyTracker {
    pub fn with_count(initial_count: i32) -> Self {
        assert!(initial_count >= 0, "dependency count cannot be negative");
        Self {
            state: Mutex::new(AsyncDependencyState {
                count: initial_count,
                aborted: false,
                tasks_have_started: false,
                count_was_set: true,
                next_tasks: Vec::new(),
            }),
        }
    }

    pub fn set_count(&self, count: i32) -> Result<(), AsyncDependencyError> {
        if count < 0 {
            return Err(AsyncDependencyError::NegativeCount);
        }
        let mut state = self.lock_state();
        if state.tasks_have_started || state.aborted {
            return Err(AsyncDependencyError::TasksAlreadyStarted);
        }
        if state.count_was_set {
            return Err(AsyncDependencyError::CountAlreadySet);
        }
        state.count_was_set = true;
        state.count = count;
        Ok(())
    }

    pub fn set_next_tasks(
        &self,
        next_tasks: Vec<ScheduledTask>,
    ) -> Result<(), AsyncDependencyError> {
        let mut state = self.lock_state();
        if state.tasks_have_started || state.aborted {
            return Err(AsyncDependencyError::TasksAlreadyStarted);
        }
        if !state.next_tasks.is_empty() {
            return Err(AsyncDependencyError::NextTasksAlreadySet);
        }
        state.next_tasks = next_tasks;
        Ok(())
    }

    pub fn has_next_tasks(&self) -> bool {
        !self.lock_state().next_tasks.is_empty()
    }

    pub fn post_complete(&self) -> Result<AsyncDependencyCompletion, AsyncDependencyError> {
        let mut state = self.lock_state();
        state.tasks_have_started = true;

        if state.aborted {
            return Err(AsyncDependencyError::PostCompleteAfterAbort);
        }
        if state.count <= 0 {
            return Err(AsyncDependencyError::PostCompleteTooManyTimes);
        }

        state.count -= 1;
        let remaining_count = state.count;
        let was_last = remaining_count == 0;
        let next_tasks = if was_last {
            std::mem::take(&mut state.next_tasks)
        } else {
            Vec::new()
        };

        Ok(AsyncDependencyCompletion {
            remaining_count,
            was_last,
            next_tasks,
        })
    }

    pub fn abort(&self) {
        let mut state = self.lock_state();
        state.aborted = true;
        state.tasks_have_started = true;
        state.next_tasks.clear();
    }

    pub fn is_aborted(&self) -> bool {
        self.lock_state().aborted
    }

    pub fn is_complete(&self) -> bool {
        self.remaining_count() == 0
    }

    pub fn remaining_count(&self) -> i32 {
        self.lock_state().count
    }

    fn lock_state(&self) -> MutexGuard<'_, AsyncDependencyState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::{AsyncDependencyError, AsyncDependencyTracker};
    use crate::tasks::{ScheduledTask, TaskLane, TaskRunStatus, ThreadedTask, ThreadedTaskContext};

    struct DependencyTestTask;

    impl ThreadedTask for DependencyTestTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }
    }

    fn scheduled_task() -> ScheduledTask {
        ScheduledTask::new(Box::new(DependencyTestTask), TaskLane::Parallel)
    }

    #[test]
    fn count_can_be_set_once_before_tasks_start() {
        let tracker = AsyncDependencyTracker::default();

        tracker.set_count(2).unwrap();

        assert_eq!(tracker.remaining_count(), 2);
        assert_eq!(
            tracker.set_count(3),
            Err(AsyncDependencyError::CountAlreadySet)
        );
    }

    #[test]
    fn post_complete_decrements_until_complete() {
        let tracker = AsyncDependencyTracker::with_count(2);

        let first = tracker.post_complete().unwrap();
        assert_eq!(first.remaining_count, 1);
        assert!(!first.was_last);
        assert_eq!(tracker.remaining_count(), 1);
        assert!(!tracker.is_complete());

        let second = tracker.post_complete().unwrap();
        assert_eq!(second.remaining_count, 0);
        assert!(second.was_last);
        assert_eq!(tracker.remaining_count(), 0);
        assert!(tracker.is_complete());
        assert!(matches!(
            tracker.post_complete(),
            Err(AsyncDependencyError::PostCompleteTooManyTimes)
        ));
    }

    #[test]
    fn post_complete_returns_next_tasks_only_when_count_reaches_zero() {
        let tracker = AsyncDependencyTracker::with_count(2);
        tracker
            .set_next_tasks(vec![scheduled_task(), scheduled_task()])
            .unwrap();

        let first = tracker.post_complete().unwrap();
        assert_eq!(first.remaining_count, 1);
        assert!(!first.was_last);
        assert!(first.next_tasks.is_empty());
        assert!(tracker.has_next_tasks());

        let second = tracker.post_complete().unwrap();
        assert_eq!(second.remaining_count, 0);
        assert!(second.was_last);
        assert_eq!(second.next_tasks.len(), 2);
        assert!(!tracker.has_next_tasks());
    }

    #[test]
    fn abort_prevents_completion_posts_and_late_count_changes() {
        let tracker = AsyncDependencyTracker::default();

        tracker.abort();

        assert!(tracker.is_aborted());
        assert_eq!(
            tracker.set_count(1),
            Err(AsyncDependencyError::TasksAlreadyStarted)
        );
        assert!(matches!(
            tracker.post_complete(),
            Err(AsyncDependencyError::PostCompleteAfterAbort)
        ));
    }

    #[test]
    fn abort_drops_next_tasks() {
        let tracker = AsyncDependencyTracker::with_count(1);
        tracker.set_next_tasks(vec![scheduled_task()]).unwrap();

        tracker.abort();

        assert!(!tracker.has_next_tasks());
    }
}
