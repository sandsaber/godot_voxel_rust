//! ThreadSanitizer target for `ThreadedTaskRunner`.
//!
//! Stresses the runner's enqueue path (staging queue + semaphore wakeup),
//! worker pop/sort, and the completed-task drain. Several producer threads
//! enqueue while workers execute tasks that touch a shared counter under the
//! runner's own scheduling — TSan verifies the runner's internal atomics and
//! Condvar handoffs keep every shared access properly synchronised.
//!
//! Run with:
//! ```text
//! RUSTFLAGS="-Zsanitizer=thread -Cunsafe-allow-abi-mismatch=sanitizer" \
//!   cargo +nightly test -p tsan --test task_runner_concurrency -Zbuild-std \
//!     --target x86_64-unknown-linux-gnu -- --test-threads=1
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use voxel_core::tasks::{
    ScheduledTask, TaskLane, TaskRunStatus, ThreadedTask, ThreadedTaskContext, ThreadedTaskRunner,
};

/// Trivial task that increments a shared counter from inside `run`. The
/// counter is only ever touched while the runner owns the task, so a race here
/// would point at the runner reusing or double-running tasks.
struct CounterTask {
    counter: Arc<AtomicUsize>,
}

impl ThreadedTask for CounterTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        self.counter.fetch_add(1, Ordering::SeqCst);
        TaskRunStatus::Complete {
            follow_up_tasks: Vec::new(),
        }
    }
}

#[test]
fn task_runner_concurrent_enqueue_and_completion_stays_race_free() {
    const WORKERS: usize = 6;
    const PRODUCERS: usize = 4;
    const TASKS_PER_PRODUCER: usize = 64;

    let mut runner = ThreadedTaskRunner::new(WORKERS);
    let counter = Arc::new(AtomicUsize::new(0));
    let expected = PRODUCERS * TASKS_PER_PRODUCER;
    let start = Arc::new(Barrier::new(PRODUCERS + 1));

    thread::scope(|scope| {
        for _ in 0..PRODUCERS {
            let runner = &runner;
            let counter = counter.clone();
            let start = start.clone();
            scope.spawn(move || {
                start.wait();
                let tasks = (0..TASKS_PER_PRODUCER)
                    .map(|_| {
                        ScheduledTask::new(
                            Box::new(CounterTask {
                                counter: counter.clone(),
                            }),
                            TaskLane::Parallel,
                        )
                    })
                    .collect::<Vec<_>>();
                runner.enqueue_many(tasks);
            });
        }

        start.wait();
    });

    runner.wait_for_all_tasks();
    let mut completed = std::collections::VecDeque::new();
    runner.try_drain_completed_into(&mut completed).unwrap();
    runner.shutdown();

    assert_eq!(completed.len(), expected);
    assert_eq!(counter.load(Ordering::SeqCst), expected);
}

/// Tasks that postpone themselves once exercise the requeue path (separate
/// enqueue lock + semaphore post). A postponed task re-enters the staging
/// queue from a worker thread while producers may still be enqueueing fresh
/// work. Each task carries its own once-flag so the test deterministically
/// expects every task to run exactly twice (once postponed, once complete).
struct PostponeOnceTask {
    counter: Arc<AtomicUsize>,
    ran: std::sync::atomic::AtomicBool,
}

impl ThreadedTask for PostponeOnceTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        // First run: flip the flag and postpone. Second run: complete.
        if !self.ran.swap(true, Ordering::SeqCst) {
            TaskRunStatus::Postponed
        } else {
            self.counter.fetch_add(1, Ordering::SeqCst);
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }
    }
}

#[test]
fn task_runner_postponed_requeue_stays_race_free() {
    let mut runner = ThreadedTaskRunner::new(4);
    let counter = Arc::new(AtomicUsize::new(0));
    const TASKS: usize = 32;

    let tasks = (0..TASKS)
        .map(|_| {
            ScheduledTask::new(
                Box::new(PostponeOnceTask {
                    counter: counter.clone(),
                    ran: std::sync::atomic::AtomicBool::new(false),
                }),
                TaskLane::Parallel,
            )
        })
        .collect::<Vec<_>>();
    runner.enqueue_many(tasks);
    runner.wait_for_all_tasks();
    let mut completed = std::collections::VecDeque::new();
    runner.try_drain_completed_into(&mut completed).unwrap();
    runner.shutdown();

    assert_eq!(completed.len(), TASKS);
    assert_eq!(counter.load(Ordering::SeqCst), TASKS);
}
