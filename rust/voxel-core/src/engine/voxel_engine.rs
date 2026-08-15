//! Engine-level registry and shared viewer priority data.
//!
//! This is the engine-agnostic subset of `engine/voxel_engine.*`: volume and
//! viewer registries plus `sync_viewers_task_priority_data`.

use super::PriorityViewersData;
use crate::math::Vector3f;
use crate::tasks::{ThreadedTask, ThreadedTaskRunner};
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SlotKey {
    index: u32,
    generation: u32,
}

trait SlotHandle: Copy {
    fn from_key(key: SlotKey) -> Self;
    fn key(self) -> SlotKey;
}

#[derive(Debug)]
struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

#[derive(Debug)]
struct GenerationalSlotMap<T, H> {
    slots: Vec<Slot<T>>,
    free_indices: Vec<usize>,
    _marker: PhantomData<fn() -> H>,
}

impl<T, H> Default for GenerationalSlotMap<T, H> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free_indices: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<T, H: SlotHandle> GenerationalSlotMap<T, H> {
    fn add(&mut self, value: T) -> H {
        if let Some(index) = self.free_indices.pop() {
            let slot = &mut self.slots[index];
            debug_assert!(slot.value.is_none());
            slot.value = Some(value);
            return H::from_key(SlotKey {
                index: index as u32,
                generation: slot.generation,
            });
        }

        let index = self.slots.len();
        assert!(
            u32::try_from(index).is_ok(),
            "slot index does not fit in u32"
        );
        self.slots.push(Slot {
            generation: 1,
            value: Some(value),
        });
        H::from_key(SlotKey {
            index: index as u32,
            generation: 1,
        })
    }

    fn remove(&mut self, handle: H) -> bool {
        let key = handle.key();
        let Some(slot) = self.slots.get_mut(key.index as usize) else {
            return false;
        };
        if slot.generation != key.generation || slot.value.is_none() {
            return false;
        }

        slot.value = None;
        slot.generation = slot.generation.wrapping_add(1);
        self.free_indices.push(key.index as usize);
        true
    }

    fn exists(&self, handle: H) -> bool {
        self.get(handle).is_some()
    }

    fn get(&self, handle: H) -> Option<&T> {
        let key = handle.key();
        self.slots.get(key.index as usize).and_then(|slot| {
            if slot.generation == key.generation {
                slot.value.as_ref()
            } else {
                None
            }
        })
    }

    fn get_mut(&mut self, handle: H) -> Option<&mut T> {
        let key = handle.key();
        self.slots.get_mut(key.index as usize).and_then(|slot| {
            if slot.generation == key.generation {
                slot.value.as_mut()
            } else {
                None
            }
        })
    }

    fn count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.value.is_some())
            .count()
    }

    fn values(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|slot| slot.value.as_ref())
    }
}

/// Generational volume handle, equivalent to C++ `VolumeID`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VolumeId {
    index: u32,
    generation: u32,
}

impl SlotHandle for VolumeId {
    fn from_key(key: SlotKey) -> Self {
        Self {
            index: key.index,
            generation: key.generation,
        }
    }

    fn key(self) -> SlotKey {
        SlotKey {
            index: self.index,
            generation: self.generation,
        }
    }
}

/// Generational viewer handle, equivalent to C++ `ViewerID`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ViewerId {
    index: u32,
    generation: u32,
}

impl SlotHandle for ViewerId {
    fn from_key(key: SlotKey) -> Self {
        Self {
            index: key.index,
            generation: key.generation,
        }
    }

    fn key(self) -> SlotKey {
        SlotKey {
            index: self.index,
            generation: self.generation,
        }
    }
}

/// Per-viewer horizontal/vertical view distances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewerDistances {
    pub horizontal: u32,
    pub vertical: u32,
}

impl ViewerDistances {
    pub const fn max(self) -> u32 {
        if self.horizontal > self.vertical {
            self.horizontal
        } else {
            self.vertical
        }
    }
}

impl Default for ViewerDistances {
    fn default() -> Self {
        Self {
            horizontal: 128,
            vertical: 128,
        }
    }
}

#[derive(Debug, Default)]
struct Volume;

/// A viewer tracked by [`VoxelEngine`].
#[derive(Debug, Clone, PartialEq)]
pub struct Viewer {
    pub world_position: Vector3f,
    pub view_distances: ViewerDistances,
    pub require_collisions: bool,
    pub require_visuals: bool,
    pub requires_data_block_notifications: bool,
    pub network_peer_id: i32,
}

impl Default for Viewer {
    fn default() -> Self {
        Self {
            world_position: Vector3f::zero(),
            view_distances: ViewerDistances::default(),
            require_collisions: true,
            require_visuals: true,
            requires_data_block_notifications: false,
            network_peer_id: -1,
        }
    }
}

/// Engine-agnostic subset of C++ `VoxelEngine`.
pub struct VoxelEngine {
    volumes: GenerationalSlotMap<Volume, VolumeId>,
    viewers: GenerationalSlotMap<Viewer, ViewerId>,
    shared_priority_dependency: Arc<PriorityViewersData>,
    task_runner: ThreadedTaskRunner,
}

impl Default for VoxelEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelEngine {
    pub fn new() -> Self {
        Self::with_thread_count(default_thread_count())
    }

    pub fn with_thread_count(thread_count: usize) -> Self {
        Self {
            volumes: GenerationalSlotMap::default(),
            viewers: GenerationalSlotMap::default(),
            shared_priority_dependency: Arc::new(PriorityViewersData::new(Vec::new())),
            task_runner: ThreadedTaskRunner::new(thread_count),
        }
    }

    pub fn add_volume(&mut self) -> VolumeId {
        self.volumes.add(Volume)
    }

    pub fn remove_volume(&mut self, volume_id: VolumeId) -> bool {
        self.volumes.remove(volume_id)
    }

    pub fn is_volume_valid(&self, volume_id: VolumeId) -> bool {
        self.volumes.exists(volume_id)
    }

    pub fn volume_count(&self) -> usize {
        self.volumes.count()
    }

    pub fn add_viewer(&mut self) -> ViewerId {
        self.viewers.add(Viewer::default())
    }

    pub fn remove_viewer(&mut self, viewer_id: ViewerId) -> bool {
        self.viewers.remove(viewer_id)
    }

    pub fn viewer_exists(&self, viewer_id: ViewerId) -> bool {
        self.viewers.exists(viewer_id)
    }

    pub fn viewer_count(&self) -> usize {
        self.viewers.count()
    }

    pub fn set_viewer_position(&mut self, viewer_id: ViewerId, position: Vector3f) -> bool {
        let Some(viewer) = self.viewers.get_mut(viewer_id) else {
            return false;
        };
        viewer.world_position = position;
        true
    }

    pub fn viewer_position(&self, viewer_id: ViewerId) -> Option<Vector3f> {
        self.viewers
            .get(viewer_id)
            .map(|viewer| viewer.world_position)
    }

    pub fn set_viewer_distances(
        &mut self,
        viewer_id: ViewerId,
        distances: ViewerDistances,
    ) -> bool {
        let Some(viewer) = self.viewers.get_mut(viewer_id) else {
            return false;
        };
        viewer.view_distances = distances;
        true
    }

    pub fn viewer_distances(&self, viewer_id: ViewerId) -> Option<ViewerDistances> {
        self.viewers
            .get(viewer_id)
            .map(|viewer| viewer.view_distances)
    }

    pub fn set_viewer_requires_visuals(&mut self, viewer_id: ViewerId, enabled: bool) -> bool {
        let Some(viewer) = self.viewers.get_mut(viewer_id) else {
            return false;
        };
        viewer.require_visuals = enabled;
        true
    }

    pub fn viewer_requires_visuals(&self, viewer_id: ViewerId) -> Option<bool> {
        self.viewers
            .get(viewer_id)
            .map(|viewer| viewer.require_visuals)
    }

    pub fn set_viewer_requires_collisions(&mut self, viewer_id: ViewerId, enabled: bool) -> bool {
        let Some(viewer) = self.viewers.get_mut(viewer_id) else {
            return false;
        };
        viewer.require_collisions = enabled;
        true
    }

    pub fn viewer_requires_collisions(&self, viewer_id: ViewerId) -> Option<bool> {
        self.viewers
            .get(viewer_id)
            .map(|viewer| viewer.require_collisions)
    }

    pub fn set_viewer_requires_data_block_notifications(
        &mut self,
        viewer_id: ViewerId,
        enabled: bool,
    ) -> bool {
        let Some(viewer) = self.viewers.get_mut(viewer_id) else {
            return false;
        };
        viewer.requires_data_block_notifications = enabled;
        true
    }

    pub fn viewer_requires_data_block_notifications(&self, viewer_id: ViewerId) -> Option<bool> {
        self.viewers
            .get(viewer_id)
            .map(|viewer| viewer.requires_data_block_notifications)
    }

    pub fn set_viewer_network_peer_id(&mut self, viewer_id: ViewerId, peer_id: i32) -> bool {
        let Some(viewer) = self.viewers.get_mut(viewer_id) else {
            return false;
        };
        viewer.network_peer_id = peer_id;
        true
    }

    pub fn viewer_network_peer_id(&self, viewer_id: ViewerId) -> Option<i32> {
        self.viewers
            .get(viewer_id)
            .map(|viewer| viewer.network_peer_id)
    }

    pub fn shared_viewers_data(&self) -> Arc<PriorityViewersData> {
        self.shared_priority_dependency.clone()
    }

    pub fn set_thread_count(&mut self, thread_count: usize) {
        self.task_runner.set_thread_count(thread_count);
    }

    pub fn thread_count(&self) -> usize {
        self.task_runner.thread_count()
    }

    pub fn push_async_task(&self, task: Box<dyn ThreadedTask>) {
        self.task_runner.enqueue(crate::tasks::ScheduledTask::new(
            task,
            crate::tasks::TaskLane::Parallel,
        ));
    }

    pub fn push_async_tasks<I>(&self, tasks: I)
    where
        I: IntoIterator<Item = Box<dyn ThreadedTask>>,
    {
        self.task_runner.enqueue_many(
            tasks.into_iter().map(|task| {
                crate::tasks::ScheduledTask::new(task, crate::tasks::TaskLane::Parallel)
            }),
        );
    }

    pub fn push_async_io_task(&self, task: Box<dyn ThreadedTask>) {
        self.task_runner.enqueue(crate::tasks::ScheduledTask::new(
            task,
            crate::tasks::TaskLane::Serial,
        ));
    }

    pub fn push_async_io_tasks<I>(&self, tasks: I)
    where
        I: IntoIterator<Item = Box<dyn ThreadedTask>>,
    {
        self.task_runner.enqueue_many(
            tasks
                .into_iter()
                .map(|task| crate::tasks::ScheduledTask::new(task, crate::tasks::TaskLane::Serial)),
        );
    }

    pub fn wait_for_all_tasks(&self) {
        self.task_runner.wait_for_all_tasks();
    }

    pub fn wait_and_clear_all_tasks(&mut self) {
        self.task_runner.wait_for_all_tasks();
        let mut completed = std::collections::VecDeque::new();
        let _ = self.task_runner.try_drain_completed_into(&mut completed);
    }

    pub fn pending_threaded_task_count(&self) -> usize {
        self.task_runner.remaining_task_count()
    }

    pub fn sync_viewers_task_priority_data(&self) {
        let mut max_distance = 0u32;
        let viewers: Vec<Vector3f> = self
            .viewers
            .values()
            .map(|viewer| {
                max_distance = max_distance.max(viewer.view_distances.max());
                viewer.world_position
            })
            .collect();

        self.shared_priority_dependency.set_viewers(viewers);
        self.shared_priority_dependency
            .set_highest_view_distance((max_distance as f32) * 2.0);
    }

    pub fn process(&mut self) {
        let mut completed = std::collections::VecDeque::new();
        if self
            .task_runner
            .try_drain_completed_into(&mut completed)
            .is_err()
        {
            return;
        }
        for completed_task in completed {
            let (task, status, follow_up_tasks) = completed_task.into_generic_parts();
            self.task_runner.enqueue_many(follow_up_tasks);
            if status == crate::tasks::TaskCompletionStatus::Finished {
                task.apply_result();
            }
        }
        self.sync_viewers_task_priority_data();
    }
}

fn default_thread_count() -> usize {
    let n = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    n.min(4)
}

#[cfg(test)]
mod tests {
    use super::{ViewerDistances, VoxelEngine};
    use crate::math::Vector3f;
    use crate::tasks::{ScheduledTask, TaskLane, TaskRunStatus, ThreadedTask, ThreadedTaskContext};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    struct FlagTask {
        ran: Arc<AtomicBool>,
        applied: Arc<AtomicBool>,
    }

    impl ThreadedTask for FlagTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            self.ran.store(true, Ordering::SeqCst);
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }

        fn apply_result(self: Box<Self>) {
            self.applied.store(true, Ordering::SeqCst);
        }
    }

    struct SerialCounterTask {
        current: Arc<AtomicUsize>,
        max: Arc<AtomicUsize>,
        completed: Arc<AtomicUsize>,
        applied: Arc<AtomicUsize>,
    }

    impl ThreadedTask for SerialCounterTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
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
            thread::sleep(Duration::from_millis(5));
            self.current.fetch_sub(1, Ordering::SeqCst);
            self.completed.fetch_add(1, Ordering::SeqCst);
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }

        fn apply_result(self: Box<Self>) {
            self.applied.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct FollowUpParentTask {
        parent_applied: Arc<AtomicBool>,
        follow_up_tasks: Vec<ScheduledTask>,
    }

    struct CancelledApplyTask {
        ran: Arc<AtomicBool>,
        applied: Arc<AtomicBool>,
    }

    impl ThreadedTask for CancelledApplyTask {
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

    impl ThreadedTask for FollowUpParentTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            TaskRunStatus::Complete {
                follow_up_tasks: std::mem::take(&mut self.follow_up_tasks),
            }
        }

        fn apply_result(self: Box<Self>) {
            self.parent_applied.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn viewer_ids_are_generational_after_remove_and_reuse() {
        let mut engine = VoxelEngine::new();
        let first = engine.add_viewer();
        assert!(engine.viewer_exists(first));

        assert!(engine.remove_viewer(first));
        assert!(!engine.viewer_exists(first));

        let second = engine.add_viewer();
        assert_ne!(first, second);
        assert!(!engine.set_viewer_position(first, Vector3f::new(1.0, 2.0, 3.0)));
        assert!(engine.set_viewer_position(second, Vector3f::new(4.0, 5.0, 6.0)));
    }

    #[test]
    fn volume_ids_are_generational_after_remove_and_reuse() {
        let mut engine = VoxelEngine::new();
        let first = engine.add_volume();
        assert!(engine.is_volume_valid(first));

        assert!(engine.remove_volume(first));
        assert!(!engine.is_volume_valid(first));

        let second = engine.add_volume();
        assert_ne!(first, second);
        assert!(!engine.remove_volume(first));
        assert!(engine.is_volume_valid(second));
    }

    #[test]
    fn viewer_properties_round_trip() {
        let mut engine = VoxelEngine::new();
        let viewer = engine.add_viewer();

        assert!(engine.set_viewer_position(viewer, Vector3f::new(1.0, 2.0, 3.0)));
        assert!(engine.set_viewer_distances(
            viewer,
            ViewerDistances {
                horizontal: 64,
                vertical: 96,
            },
        ));
        assert!(engine.set_viewer_requires_visuals(viewer, false));
        assert!(engine.set_viewer_requires_collisions(viewer, false));
        assert!(engine.set_viewer_requires_data_block_notifications(viewer, true));
        assert!(engine.set_viewer_network_peer_id(viewer, 42));

        assert_eq!(
            engine.viewer_position(viewer),
            Some(Vector3f::new(1.0, 2.0, 3.0))
        );
        assert_eq!(
            engine.viewer_distances(viewer),
            Some(ViewerDistances {
                horizontal: 64,
                vertical: 96,
            })
        );
        assert_eq!(engine.viewer_requires_visuals(viewer), Some(false));
        assert_eq!(engine.viewer_requires_collisions(viewer), Some(false));
        assert_eq!(
            engine.viewer_requires_data_block_notifications(viewer),
            Some(true)
        );
        assert_eq!(engine.viewer_network_peer_id(viewer), Some(42));
    }

    #[test]
    fn sync_viewers_task_priority_data_exports_positions_and_cancel_distance() {
        let mut engine = VoxelEngine::new();
        let first = engine.add_viewer();
        let second = engine.add_viewer();
        assert!(engine.set_viewer_position(first, Vector3f::new(10.0, 0.0, 0.0)));
        assert!(engine.set_viewer_position(second, Vector3f::new(20.0, 0.0, 0.0)));
        assert!(engine.set_viewer_distances(
            first,
            ViewerDistances {
                horizontal: 32,
                vertical: 48,
            },
        ));
        assert!(engine.set_viewer_distances(
            second,
            ViewerDistances {
                horizontal: 80,
                vertical: 16,
            },
        ));

        engine.sync_viewers_task_priority_data();

        let shared = engine.shared_viewers_data();
        assert_eq!(shared.viewers_count(), 2);
        assert_eq!(
            shared.viewers(),
            vec![Vector3f::new(10.0, 0.0, 0.0), Vector3f::new(20.0, 0.0, 0.0)]
        );
        assert_eq!(shared.highest_view_distance(), 160.0);
    }

    #[test]
    fn sync_viewers_handles_extreme_view_distance_without_u32_overflow() {
        let mut engine = VoxelEngine::new();
        let viewer = engine.add_viewer();
        assert!(engine.set_viewer_distances(
            viewer,
            ViewerDistances {
                horizontal: u32::MAX,
                vertical: 0,
            },
        ));

        engine.sync_viewers_task_priority_data();

        let shared = engine.shared_viewers_data();
        assert_eq!(shared.highest_view_distance(), (u32::MAX as f32) * 2.0);
    }

    #[test]
    fn process_syncs_shared_viewer_priority_data() {
        let mut engine = VoxelEngine::new();
        let viewer = engine.add_viewer();
        assert!(engine.set_viewer_position(viewer, Vector3f::new(7.0, 8.0, 9.0)));

        engine.process();

        let shared = engine.shared_viewers_data();
        assert_eq!(shared.viewers_count(), 1);
        assert_eq!(shared.viewers(), vec![Vector3f::new(7.0, 8.0, 9.0)]);
    }

    #[test]
    fn process_applies_completed_async_tasks_and_syncs_viewers() {
        let mut engine = VoxelEngine::with_thread_count(1);
        let viewer = engine.add_viewer();
        assert!(engine.set_viewer_position(viewer, Vector3f::new(1.0, 2.0, 3.0)));

        let ran = Arc::new(AtomicBool::new(false));
        let applied = Arc::new(AtomicBool::new(false));
        engine.push_async_task(Box::new(FlagTask {
            ran: ran.clone(),
            applied: applied.clone(),
        }));

        engine.wait_for_all_tasks();
        assert!(ran.load(Ordering::SeqCst));
        assert!(!applied.load(Ordering::SeqCst));
        assert_eq!(engine.shared_viewers_data().viewers_count(), 0);

        engine.process();

        assert!(applied.load(Ordering::SeqCst));
        let shared = engine.shared_viewers_data();
        assert_eq!(shared.viewers_count(), 1);
        assert_eq!(shared.viewers(), vec![Vector3f::new(1.0, 2.0, 3.0)]);
    }

    #[test]
    fn async_io_tasks_run_serially_through_engine() {
        let mut engine = VoxelEngine::with_thread_count(4);
        let current = Arc::new(AtomicUsize::new(0));
        let max = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        let applied = Arc::new(AtomicUsize::new(0));

        for _ in 0..8 {
            engine.push_async_io_task(Box::new(SerialCounterTask {
                current: current.clone(),
                max: max.clone(),
                completed: completed.clone(),
                applied: applied.clone(),
            }));
        }

        engine.wait_for_all_tasks();
        engine.process();

        assert_eq!(completed.load(Ordering::SeqCst), 8);
        assert_eq!(applied.load(Ordering::SeqCst), 8);
        assert_eq!(max.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn process_does_not_apply_cancelled_task_results() {
        let mut engine = VoxelEngine::with_thread_count(1);
        let ran = Arc::new(AtomicBool::new(false));
        let applied = Arc::new(AtomicBool::new(false));
        engine.push_async_task(Box::new(CancelledApplyTask {
            ran: ran.clone(),
            applied: applied.clone(),
        }));

        engine.wait_for_all_tasks();
        engine.process();

        assert!(!ran.load(Ordering::SeqCst));
        assert!(!applied.load(Ordering::SeqCst));
    }

    #[test]
    fn process_enqueues_and_applies_follow_up_tasks() {
        let mut engine = VoxelEngine::with_thread_count(1);
        let parent_applied = Arc::new(AtomicBool::new(false));
        let child_ran = Arc::new(AtomicBool::new(false));
        let child_applied = Arc::new(AtomicBool::new(false));

        engine.push_async_task(Box::new(FollowUpParentTask {
            parent_applied: parent_applied.clone(),
            follow_up_tasks: vec![ScheduledTask::new(
                Box::new(FlagTask {
                    ran: child_ran.clone(),
                    applied: child_applied.clone(),
                }),
                TaskLane::Parallel,
            )],
        }));

        engine.wait_for_all_tasks();
        engine.process();

        assert!(parent_applied.load(Ordering::SeqCst));
        assert!(!child_applied.load(Ordering::SeqCst));

        engine.wait_for_all_tasks();
        assert!(child_ran.load(Ordering::SeqCst));
        engine.process();

        assert!(child_applied.load(Ordering::SeqCst));
    }

    #[test]
    fn wait_and_clear_all_tasks_drops_completed_tasks_without_applying_results() {
        let mut engine = VoxelEngine::with_thread_count(1);
        let ran = Arc::new(AtomicBool::new(false));
        let applied = Arc::new(AtomicBool::new(false));
        engine.push_async_task(Box::new(FlagTask {
            ran: ran.clone(),
            applied: applied.clone(),
        }));

        engine.wait_and_clear_all_tasks();

        assert!(ran.load(Ordering::SeqCst));
        assert!(!applied.load(Ordering::SeqCst));

        engine.process();

        assert!(!applied.load(Ordering::SeqCst));
    }
}
