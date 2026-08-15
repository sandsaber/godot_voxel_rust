//! Terminal ownership shared by voxel persistence tasks.

use crate::storage::{BlockLocation, VoxelBuffer};
use crate::streams::StreamResult;
use crate::tasks::TaskPanicPhase;

/// External-I/O progress reached by one physical persistence task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistenceIoPhase {
    BeforeIo,
    CallEntered,
    Acknowledged,
}

/// Exact result returned by the single external operation owned by a task.
#[derive(Debug, PartialEq, Eq)]
pub enum PersistenceAcknowledgement {
    Save(StreamResult<()>),
    Flush(StreamResult<()>),
}

/// Lossless terminal state of one physical block-save attempt.
#[derive(Debug)]
pub struct SaveTaskTerminal {
    pub location: BlockLocation,
    pub block_revision: u64,
    pub save_generation: u64,
    pub payload: VoxelBuffer,
    pub task_panic_phase: Option<TaskPanicPhase>,
    pub phase: PersistenceIoPhase,
    pub acknowledgement: Option<PersistenceAcknowledgement>,
}

/// Lossless terminal state of one physical stream-flush attempt.
#[derive(Debug)]
pub struct FlushTaskTerminal {
    pub checkpoint_generation: u64,
    pub task_panic_phase: Option<TaskPanicPhase>,
    pub phase: PersistenceIoPhase,
    pub acknowledgement: Option<PersistenceAcknowledgement>,
}
