//! Stream block task output shared by load/save task ports.

use crate::math::Vector3i;
use crate::storage::VoxelBuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockDataOutputKind {
    Loaded,
    Saved,
    NeedsGeneration,
    NotFound,
}

#[derive(Debug)]
pub struct BlockDataOutput {
    pub kind: BlockDataOutputKind,
    pub voxels: Option<VoxelBuffer>,
    pub position_in_blocks: Vector3i,
    pub lod_index: u8,
    pub dropped: bool,
    pub max_lod_hint: bool,
    pub initial_load: bool,
    pub had_voxels: bool,
    pub save_generation: u64,
}

impl BlockDataOutput {
    pub fn loaded(
        position_in_blocks: Vector3i,
        lod_index: u8,
        voxels: VoxelBuffer,
        max_lod_hint: bool,
    ) -> Self {
        Self {
            kind: BlockDataOutputKind::Loaded,
            voxels: Some(voxels),
            position_in_blocks,
            lod_index,
            dropped: false,
            max_lod_hint,
            initial_load: false,
            had_voxels: true,
            save_generation: 0,
        }
    }

    pub fn needs_generation(
        position_in_blocks: Vector3i,
        lod_index: u8,
        voxels: VoxelBuffer,
    ) -> Self {
        Self {
            kind: BlockDataOutputKind::NeedsGeneration,
            voxels: Some(voxels),
            position_in_blocks,
            lod_index,
            dropped: false,
            max_lod_hint: false,
            initial_load: false,
            had_voxels: false,
            save_generation: 0,
        }
    }

    pub fn not_found(position_in_blocks: Vector3i, lod_index: u8) -> Self {
        Self {
            kind: BlockDataOutputKind::NotFound,
            voxels: None,
            position_in_blocks,
            lod_index,
            dropped: false,
            max_lod_hint: false,
            initial_load: false,
            had_voxels: false,
            save_generation: 0,
        }
    }

    /// Output produced when a load task encountered a stream error or was
    /// cancelled. Matches the C++ `apply_result` path where `dropped = !_has_run`
    /// still emits a `TYPE_LOADED` output with a null buffer, so the terrain
    /// can re-request or log the failure rather than silently dropping it.
    pub fn loaded_dropped(position_in_blocks: Vector3i, lod_index: u8) -> Self {
        Self {
            kind: BlockDataOutputKind::Loaded,
            voxels: None,
            position_in_blocks,
            lod_index,
            dropped: true,
            max_lod_hint: false,
            initial_load: false,
            had_voxels: false,
            save_generation: 0,
        }
    }

    /// Output produced when a save task was aborted before running (e.g. its
    /// voxels were never set or it was cancelled). Matches the C++ path where
    /// `apply_result` emits `TYPE_SAVED` with `dropped = !_has_run`, so the
    /// caller is informed that the block was not persisted.
    pub fn saved_dropped(
        position_in_blocks: Vector3i,
        lod_index: u8,
        voxels: Option<VoxelBuffer>,
        had_voxels: bool,
        save_generation: u64,
    ) -> Self {
        Self {
            kind: BlockDataOutputKind::Saved,
            voxels,
            position_in_blocks,
            lod_index,
            dropped: true,
            max_lod_hint: false,
            initial_load: false,
            had_voxels,
            save_generation,
        }
    }

    pub fn saved(
        position_in_blocks: Vector3i,
        lod_index: u8,
        had_voxels: bool,
        save_generation: u64,
    ) -> Self {
        Self {
            kind: BlockDataOutputKind::Saved,
            voxels: None,
            position_in_blocks,
            lod_index,
            dropped: false,
            max_lod_hint: false,
            initial_load: false,
            had_voxels,
            save_generation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockDataOutput, BlockDataOutputKind};
    use crate::math::Vector3i;
    use crate::storage::{ChannelId, VoxelBuffer};

    #[test]
    fn loaded_output_preserves_block_identity_and_voxels() {
        let position = Vector3i::new(1, 2, 3);
        let mut voxels = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        voxels.set_voxel(7, 0, 0, 0, ChannelId::Type.index());

        let output = BlockDataOutput::loaded(position, 4, voxels, false);

        assert_eq!(output.kind, BlockDataOutputKind::Loaded);
        assert_eq!(output.position_in_blocks, position);
        assert_eq!(output.lod_index, 4);
        assert!(!output.dropped);
        assert!(!output.max_lod_hint);
        assert!(!output.initial_load);
        assert!(output.had_voxels);
        assert_eq!(
            output
                .voxels
                .as_ref()
                .unwrap()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            7
        );
    }

    #[test]
    fn missing_outputs_have_no_voxels() {
        let position = Vector3i::new(3, 2, 1);

        let not_found = BlockDataOutput::not_found(position, 1);
        let saved = BlockDataOutput::saved(position, 1, false, 0);

        assert_eq!(not_found.kind, BlockDataOutputKind::NotFound);
        assert!(not_found.voxels.is_none());
        assert!(!not_found.had_voxels);

        assert_eq!(saved.kind, BlockDataOutputKind::Saved);
        assert!(saved.voxels.is_none());
        assert!(!saved.had_voxels);
    }

    #[test]
    fn legacy_struct_literal_remains_source_compatible() {
        let output = BlockDataOutput {
            kind: BlockDataOutputKind::Saved,
            voxels: None,
            position_in_blocks: Vector3i::new(4, 5, 6),
            lod_index: 2,
            dropped: false,
            max_lod_hint: false,
            initial_load: false,
            had_voxels: true,
            save_generation: 9,
        };

        assert_eq!(output.save_generation, 9);
    }
}
