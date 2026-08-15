//! Sparse storage block for one chunk of voxel data.

use crate::storage::VoxelBuffer;

/// Reference count for [`VoxelDataBlock::viewers`].
///
/// Matches C++ `RefCount` (a thin `uint32_t` wrapper). Used by
/// [`crate::storage::VoxelData::view_area`] / `unview_area` to keep loaded
/// blocks alive while one or more viewers (e.g. mesh block tasks) reference
/// them. A block reaching zero viewers is eligible for unloading.
#[derive(Debug, Default, Clone, Copy)]
pub struct Viewers(u32);

impl Viewers {
    pub const fn new() -> Self {
        Self(0)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    pub const fn set_exact(&mut self, value: u32) {
        self.0 = value;
    }

    pub fn add(&mut self) {
        self.0 = self.0.saturating_add(1);
    }

    /// Decrements and returns the new value. Saturates at zero so an unpaired
    /// `remove` cannot underflow. The C++ `RefCount` has the same property.
    pub fn remove(&mut self) -> u32 {
        self.0 = self.0.saturating_sub(1);
        self.0
    }
}

#[derive(Debug)]
pub struct VoxelDataBlock {
    voxels: Option<VoxelBuffer>,
    lod_index: u8,
    needs_lodding: bool,
    modified: bool,
    edited: bool,
    /// Optional reference count, exposed publicly to mirror the C++ public
    /// `viewers` field. Owned and mutated by `VoxelData::view_area` /
    /// `unview_area`.
    pub viewers: Viewers,
}

impl VoxelDataBlock {
    pub const fn empty(lod_index: u8) -> Self {
        Self {
            voxels: None,
            lod_index,
            needs_lodding: false,
            modified: false,
            edited: false,
            viewers: Viewers::new(),
        }
    }

    pub const fn with_voxels(voxels: VoxelBuffer, lod_index: u8) -> Self {
        Self {
            voxels: Some(voxels),
            lod_index,
            needs_lodding: false,
            modified: false,
            edited: false,
            viewers: Viewers::new(),
        }
    }

    pub const fn lod_index(&self) -> u8 {
        self.lod_index
    }

    pub const fn has_voxels(&self) -> bool {
        self.voxels.is_some()
    }

    pub fn voxels(&self) -> &VoxelBuffer {
        self.voxels
            .as_ref()
            .expect("voxel data block has no voxels")
    }

    pub fn voxels_mut(&mut self) -> &mut VoxelBuffer {
        self.voxels
            .as_mut()
            .expect("voxel data block has no voxels")
    }

    pub fn set_voxels(&mut self, voxels: VoxelBuffer) {
        self.voxels = Some(voxels);
    }

    pub fn into_voxels(self) -> Option<VoxelBuffer> {
        self.voxels
    }

    pub fn clear_voxels(&mut self) {
        self.voxels = None;
        self.edited = false;
    }

    pub const fn is_modified(&self) -> bool {
        self.modified
    }

    pub const fn set_modified(&mut self, modified: bool) {
        self.modified = modified;
    }

    pub const fn needs_lodding(&self) -> bool {
        self.needs_lodding
    }

    pub const fn set_needs_lodding(&mut self, needs_lodding: bool) {
        self.needs_lodding = needs_lodding;
    }

    pub const fn is_edited(&self) -> bool {
        self.edited
    }

    pub const fn set_edited(&mut self, edited: bool) {
        self.edited = edited;
    }
}

#[cfg(test)]
mod tests {
    use super::{Viewers, VoxelDataBlock};
    use crate::math::Vector3i;
    use crate::storage::{ChannelId, VoxelBuffer};

    #[test]
    fn empty_block_tracks_lod_and_has_no_voxels() {
        let block = VoxelDataBlock::empty(3);

        assert_eq!(block.lod_index(), 3);
        assert!(!block.has_voxels());
        assert!(!block.is_modified());
        assert!(!block.is_edited());
        assert!(!block.needs_lodding());
        assert_eq!(block.viewers.get(), 0);
    }

    #[test]
    fn block_with_voxels_exposes_flags_and_buffer() {
        let mut voxels = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        voxels.set_voxel(7, 1, 0, 0, ChannelId::Type.index());
        let mut block = VoxelDataBlock::with_voxels(voxels, 1);

        assert!(block.has_voxels());
        assert_eq!(
            block.voxels().get_voxel(1, 0, 0, ChannelId::Type.index()),
            7
        );

        block.set_modified(true);
        block.set_edited(true);
        block.set_needs_lodding(true);

        assert!(block.is_modified());
        assert!(block.is_edited());
        assert!(block.needs_lodding());

        block.clear_voxels();
        assert!(!block.has_voxels());
        assert!(!block.is_edited());
        assert!(block.is_modified());
        assert!(block.needs_lodding());
    }

    #[test]
    fn viewers_refcount_round_trips_and_saturates_below_zero() {
        let mut viewers = Viewers::new();
        assert_eq!(viewers.get(), 0);
        viewers.add();
        viewers.add();
        assert_eq!(viewers.get(), 2);
        assert_eq!(viewers.remove(), 1);
        assert_eq!(viewers.remove(), 0);
        // Unpaired remove saturates at zero rather than underflowing.
        assert_eq!(viewers.remove(), 0);
    }

    #[test]
    fn viewers_exact_setter_preserves_the_requested_total() {
        let mut viewers = Viewers::new();
        viewers.set_exact(u32::MAX);
        assert_eq!(viewers.get(), u32::MAX);
        viewers.set_exact(7);
        assert_eq!(viewers.get(), 7);
    }
}
