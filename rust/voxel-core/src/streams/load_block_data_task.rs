//! Threaded stream load task ported from the stream I/O half of
//! `streams/load_block_data_task.*`.

use crate::constants::voxel_constants::TASK_PRIORITY_LOAD_BAND2;
use crate::engine::{PriorityDependency, StreamingDependency};
use crate::math::Vector3i;
use crate::storage::{VoxelBuffer, VoxelFormat};
use crate::streams::{BlockDataOutput, LoadResult, VoxelLoadQuery, VoxelStreamError};
use crate::tasks::{
    ScheduledTask, TaskCancellationToken, TaskPriority, TaskRunStatus, ThreadedTask,
    ThreadedTaskContext,
};
use std::sync::Arc;

pub struct BlockGenerationRequest {
    pub position_in_blocks: Vector3i,
    pub lod_index: u8,
    pub block_size: u8,
    pub voxels: VoxelBuffer,
}

pub enum BlockGenerationTaskResult {
    Scheduled(ScheduledTask),
    // Boxed: `BlockGenerationRequest` carries a full `VoxelBuffer` (~348 B),
    // which would balloon the enum to ~360 B and starve the `Scheduled` arm.
    // Both arms are now pointer-sized; the extra indirection is negligible
    // next to the heap allocations `VoxelBuffer` already performs.
    NotScheduled(Box<BlockGenerationRequest>),
}

pub trait BlockGenerationTaskFactory: Send + Sync {
    // Takes the request by value; implementations that don't schedule a task
    // return it boxed inside `NotScheduled` so the caller can recover the voxels.
    fn create_task(&self, request: BlockGenerationRequest) -> BlockGenerationTaskResult;
}

pub struct LoadBlockDataParams {
    pub position_in_blocks: Vector3i,
    pub lod_index: u8,
    pub block_size: u8,
    pub format: VoxelFormat,
    pub generate_cache_data: bool,
    pub generation_task_factory: Option<Arc<dyn BlockGenerationTaskFactory>>,
    pub stream_dependency: Arc<StreamingDependency>,
    pub priority_dependency: PriorityDependency,
    pub cancellation_token: TaskCancellationToken,
}

pub struct LoadBlockDataTask {
    position_in_blocks: Vector3i,
    lod_index: u8,
    block_size: u8,
    format: VoxelFormat,
    generate_cache_data: bool,
    generation_task_factory: Option<Arc<dyn BlockGenerationTaskFactory>>,
    stream_dependency: Arc<StreamingDependency>,
    priority_dependency: PriorityDependency,
    cancellation_token: TaskCancellationToken,
    has_run: bool,
    too_far: bool,
    max_lod_hint: bool,
    output: Option<BlockDataOutput>,
    stream_error: Option<VoxelStreamError>,
    follow_up_tasks: Vec<ScheduledTask>,
}

impl LoadBlockDataTask {
    pub fn new(params: LoadBlockDataParams) -> Self {
        Self {
            position_in_blocks: params.position_in_blocks,
            lod_index: params.lod_index,
            block_size: params.block_size,
            format: params.format,
            generate_cache_data: params.generate_cache_data,
            generation_task_factory: params.generation_task_factory,
            stream_dependency: params.stream_dependency,
            priority_dependency: params.priority_dependency,
            cancellation_token: params.cancellation_token,
            has_run: false,
            too_far: false,
            max_lod_hint: false,
            output: None,
            stream_error: None,
            follow_up_tasks: Vec::new(),
        }
    }

    pub const fn position_in_blocks(&self) -> Vector3i {
        self.position_in_blocks
    }

    pub const fn lod_index(&self) -> u8 {
        self.lod_index
    }

    pub const fn has_run(&self) -> bool {
        self.has_run
    }

    pub fn take_output(&mut self) -> Option<BlockDataOutput> {
        self.output.take()
    }

    pub fn stream_error(&self) -> Option<&VoxelStreamError> {
        self.stream_error.as_ref()
    }

    fn run_load(&mut self) {
        self.output = None;
        self.stream_error = None;
        self.follow_up_tasks.clear();

        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(i32::from(self.block_size)));
        self.format.configure_buffer(&mut voxels);

        let stream = self.stream_dependency.stream();
        match stream.load_voxel_block(VoxelLoadQuery::new(
            &mut voxels,
            self.position_in_blocks,
            self.lod_index,
        )) {
            Ok(LoadResult::Found) => {
                self.output = Some(BlockDataOutput::loaded(
                    self.position_in_blocks,
                    self.lod_index,
                    voxels,
                    self.max_lod_hint,
                ));
            }
            Ok(LoadResult::NotFound) => {
                if self.generate_cache_data {
                    self.handle_generation_miss(voxels);
                } else {
                    self.output = Some(BlockDataOutput::not_found(
                        self.position_in_blocks,
                        self.lod_index,
                    ));
                }
            }
            Err(error) => {
                // Mirror the C++ apply_result contract: a stream error still
                // delivers a `Loaded` output with no voxels (the C++ side sets
                // `_has_run = true` and emits `TYPE_LOADED` with a null buffer).
                // The terrain treats it as a loadable miss and may re-request.
                self.stream_error = Some(error);
                self.output = Some(BlockDataOutput::loaded_dropped(
                    self.position_in_blocks,
                    self.lod_index,
                ));
            }
        }

        self.has_run = true;
    }

    fn handle_generation_miss(&mut self, voxels: VoxelBuffer) {
        let Some(factory) = self.generation_task_factory.clone() else {
            self.output = Some(BlockDataOutput::needs_generation(
                self.position_in_blocks,
                self.lod_index,
                voxels,
            ));
            return;
        };

        let request = BlockGenerationRequest {
            position_in_blocks: self.position_in_blocks,
            lod_index: self.lod_index,
            block_size: self.block_size,
            voxels,
        };

        match factory.create_task(request) {
            BlockGenerationTaskResult::Scheduled(task) => {
                self.follow_up_tasks.push(task);
            }
            BlockGenerationTaskResult::NotScheduled(request) => {
                let request = *request;
                self.output = Some(BlockDataOutput::needs_generation(
                    request.position_in_blocks,
                    request.lod_index,
                    request.voxels,
                ));
            }
        }
    }
}

impl ThreadedTask for LoadBlockDataTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        self.run_load();
        TaskRunStatus::Complete {
            follow_up_tasks: std::mem::take(&mut self.follow_up_tasks),
        }
    }

    fn priority(&mut self) -> TaskPriority {
        let evaluation = self
            .priority_dependency
            .evaluate(self.lod_index, TASK_PRIORITY_LOAD_BAND2);
        self.too_far = self
            .priority_dependency
            .is_too_far(evaluation.closest_distance_squared);
        evaluation.priority
    }

    fn is_cancelled(&mut self) -> bool {
        if !self.stream_dependency.is_valid() {
            return true;
        }
        if self.cancellation_token.is_valid() && self.cancellation_token.is_cancelled() {
            return true;
        }
        self.too_far
    }

    fn debug_name(&self) -> &'static str {
        "LoadBlockData"
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BlockGenerationRequest, BlockGenerationTaskFactory, BlockGenerationTaskResult,
        LoadBlockDataParams, LoadBlockDataTask,
    };
    use crate::constants::voxel_constants::TASK_PRIORITY_LOAD_BAND2;
    use crate::engine::{PriorityDependency, PriorityViewersData, StreamingDependency};
    use crate::math::{Vector3f, Vector3i};
    use crate::storage::{ChannelDepth, ChannelId, VoxelBuffer, VoxelFormat};
    use crate::streams::{
        BlockDataOutputKind, LoadResult, MemoryStream, StreamResult, VoxelLoadQuery, VoxelStream,
        VoxelStreamError,
    };
    use crate::tasks::{
        ScheduledTask, TaskCancellationToken, TaskLane, TaskPriority, TaskRunStatus, ThreadedTask,
        ThreadedTaskContext,
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    struct ErrorLoadStream;

    impl VoxelStream for ErrorLoadStream {
        fn load_voxel_block(&self, _query: VoxelLoadQuery<'_>) -> StreamResult<LoadResult> {
            Err(VoxelStreamError::Io("load failed".to_string()))
        }
    }

    struct GeneratedMarkerTask {
        position_in_blocks: Vector3i,
        lod_index: u8,
        block_size: u8,
        ran: Arc<AtomicBool>,
    }

    impl ThreadedTask for GeneratedMarkerTask {
        fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
            assert_eq!(self.position_in_blocks, Vector3i::new(3, 4, 5));
            assert_eq!(self.lod_index, 2);
            assert_eq!(self.block_size, 4);
            self.ran.store(true, Ordering::SeqCst);
            TaskRunStatus::Complete {
                follow_up_tasks: Vec::new(),
            }
        }
    }

    struct RecordingGenerationFactory {
        calls: Arc<AtomicUsize>,
        task_ran: Arc<AtomicBool>,
    }

    impl BlockGenerationTaskFactory for RecordingGenerationFactory {
        fn create_task(&self, request: BlockGenerationRequest) -> BlockGenerationTaskResult {
            assert_eq!(request.position_in_blocks, Vector3i::new(3, 4, 5));
            assert_eq!(request.lod_index, 2);
            assert_eq!(request.block_size, 4);
            assert_eq!(request.voxels.size(), Vector3i::new(4, 4, 4));
            assert_eq!(
                request.voxels.channel_depth(ChannelId::Sdf.index()),
                ChannelDepth::Bit32
            );
            self.calls.fetch_add(1, Ordering::SeqCst);
            BlockGenerationTaskResult::Scheduled(ScheduledTask::new(
                Box::new(GeneratedMarkerTask {
                    position_in_blocks: request.position_in_blocks,
                    lod_index: request.lod_index,
                    block_size: request.block_size,
                    ran: self.task_ran.clone(),
                }),
                TaskLane::Parallel,
            ))
        }
    }

    struct NotScheduledFactory;

    impl BlockGenerationTaskFactory for NotScheduledFactory {
        fn create_task(&self, request: BlockGenerationRequest) -> BlockGenerationTaskResult {
            // Returning the request boxed lets the load task recover the voxels.
            BlockGenerationTaskResult::NotScheduled(Box::new(request))
        }
    }

    fn priority_dependency(
        world_position: Vector3f,
        drop_distance_squared: f32,
    ) -> PriorityDependency {
        PriorityDependency::new(
            Arc::new(PriorityViewersData::new(Vec::new())),
            world_position,
            drop_distance_squared,
        )
    }

    fn format_with_sdf32() -> VoxelFormat {
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format
    }

    fn params(
        stream: Arc<dyn VoxelStream>,
        position: Vector3i,
        generate_cache_data: bool,
    ) -> LoadBlockDataParams {
        LoadBlockDataParams {
            position_in_blocks: position,
            lod_index: 2,
            block_size: 4,
            format: format_with_sdf32(),
            generate_cache_data,
            generation_task_factory: None,
            stream_dependency: StreamingDependency::new(stream),
            priority_dependency: priority_dependency(Vector3f::default(), 1_000_000.0),
            cancellation_token: TaskCancellationToken::default(),
        }
    }

    #[test]
    fn load_found_block_outputs_loaded_voxels() {
        let stream = Arc::new(MemoryStream::new());
        let position = Vector3i::new(2, 4, 6);
        let mut stored = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        stored.set_voxel(99, 1, 0, 0, ChannelId::Type.index());
        stream.save_block(position, 2, &stored);
        let mut task = LoadBlockDataTask::new(params(stream, position, true));

        task.run_load();
        let output = task.take_output().unwrap();

        assert_eq!(output.kind, BlockDataOutputKind::Loaded);
        let voxels = output.voxels.as_ref().unwrap();
        assert_eq!(voxels.size(), Vector3i::new(4, 4, 4));
        assert_eq!(voxels.get_voxel(1, 0, 0, ChannelId::Type.index()), 99);
        assert!(task.has_run());
    }

    #[test]
    fn load_not_found_without_generation_outputs_not_found() {
        let stream: Arc<dyn VoxelStream> = Arc::new(MemoryStream::new());
        let position = Vector3i::new(1, 1, 1);
        let mut task = LoadBlockDataTask::new(params(stream, position, false));

        task.run_load();
        let output = task.take_output().unwrap();

        assert_eq!(output.kind, BlockDataOutputKind::NotFound);
        assert!(output.voxels.is_none());
        assert!(task.has_run());
    }

    #[test]
    fn load_not_found_with_generation_outputs_configured_buffer() {
        let stream: Arc<dyn VoxelStream> = Arc::new(MemoryStream::new());
        let position = Vector3i::new(1, 2, 3);
        let mut task = LoadBlockDataTask::new(params(stream, position, true));

        task.run_load();
        let output = task.take_output().unwrap();

        assert_eq!(output.kind, BlockDataOutputKind::NeedsGeneration);
        let voxels = output.voxels.as_ref().unwrap();
        assert_eq!(voxels.size(), Vector3i::new(4, 4, 4));
        assert_eq!(
            voxels.channel_depth(ChannelId::Sdf.index()),
            ChannelDepth::Bit32
        );
    }

    #[test]
    fn load_not_found_with_generation_factory_schedules_follow_up_task() {
        let stream: Arc<dyn VoxelStream> = Arc::new(MemoryStream::new());
        let position = Vector3i::new(3, 4, 5);
        let calls = Arc::new(AtomicUsize::new(0));
        let task_ran = Arc::new(AtomicBool::new(false));
        let mut task_params = params(stream, position, true);
        task_params.generation_task_factory = Some(Arc::new(RecordingGenerationFactory {
            calls: calls.clone(),
            task_ran: task_ran.clone(),
        }));
        let mut task = LoadBlockDataTask::new(task_params);

        let TaskRunStatus::Complete { follow_up_tasks } =
            task.run(ThreadedTaskContext::new(0, TaskPriority::max()))
        else {
            panic!("load task unexpectedly postponed");
        };

        assert!(task.take_output().is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(follow_up_tasks.len(), 1);

        let follow_up_task = follow_up_tasks.into_iter().next().unwrap();
        assert_eq!(follow_up_task.lane(), TaskLane::Parallel);
        let (mut follow_up_task, _) = follow_up_task.into_parts();
        assert!(matches!(
            follow_up_task
                .as_mut()
                .run(ThreadedTaskContext::new(0, TaskPriority::max())),
            TaskRunStatus::Complete { .. }
        ));
        assert!(task_ran.load(Ordering::SeqCst));
    }

    #[test]
    fn load_stream_error_emits_dropped_output_and_exposes_error() {
        let stream: Arc<dyn VoxelStream> = Arc::new(ErrorLoadStream);
        let mut task = LoadBlockDataTask::new(params(stream, Vector3i::default(), true));

        task.run_load();

        let output = task.take_output().unwrap();
        assert_eq!(output.kind, BlockDataOutputKind::Loaded);
        assert!(output.dropped);
        assert!(output.voxels.is_none());
        assert_eq!(output.position_in_blocks, Vector3i::default());
        assert!(task.has_run());
        assert!(matches!(
            task.stream_error(),
            Some(VoxelStreamError::Io(message)) if message == "load failed"
        ));
    }

    #[test]
    fn load_not_found_with_factory_returning_not_scheduled_outputs_needs_generation() {
        let stream: Arc<dyn VoxelStream> = Arc::new(MemoryStream::new());
        let position = Vector3i::new(7, 8, 9);
        let mut task_params = params(stream, position, true);
        task_params.generation_task_factory = Some(Arc::new(NotScheduledFactory));
        let mut task = LoadBlockDataTask::new(task_params);

        task.run_load();
        let output = task.take_output().unwrap();

        assert_eq!(output.kind, BlockDataOutputKind::NeedsGeneration);
        assert_eq!(output.position_in_blocks, position);
        let voxels = output.voxels.as_ref().unwrap();
        assert_eq!(voxels.size(), Vector3i::new(4, 4, 4));
        assert!(task.follow_up_tasks.is_empty());
    }

    #[test]
    fn priority_updates_too_far_cancellation_state() {
        let stream: Arc<dyn VoxelStream> = Arc::new(MemoryStream::new());
        let mut task_params = params(stream, Vector3i::default(), true);
        task_params.priority_dependency = priority_dependency(Vector3f::new(32.0, 0.0, 0.0), 25.0);
        let mut task = LoadBlockDataTask::new(task_params);

        assert!(!task.is_cancelled());
        assert_eq!(task.priority().band2(), TASK_PRIORITY_LOAD_BAND2);
        assert!(task.is_cancelled());
    }

    #[test]
    fn invalid_dependency_or_cancelled_token_cancels_task() {
        let stream: Arc<dyn VoxelStream> = Arc::new(MemoryStream::new());
        let mut dependency_slot = None;
        let dependency = StreamingDependency::reset(&mut dependency_slot, stream.clone());
        let token = TaskCancellationToken::create();
        let mut task_params = params(stream, Vector3i::default(), true);
        task_params.stream_dependency = dependency.clone();
        task_params.cancellation_token = token.clone();
        let mut task = LoadBlockDataTask::new(task_params);

        assert!(!task.is_cancelled());

        token.cancel();
        assert!(task.is_cancelled());

        let mut task = LoadBlockDataTask::new(LoadBlockDataParams {
            cancellation_token: TaskCancellationToken::default(),
            stream_dependency: dependency,
            ..params(Arc::new(MemoryStream::new()), Vector3i::default(), true)
        });
        dependency_slot.as_ref().unwrap().invalidate();

        assert!(task.is_cancelled());
    }
}
