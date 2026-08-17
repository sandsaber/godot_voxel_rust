//! Sparse map of voxel data blocks for one LOD.

use crate::constants::voxel_constants::{DEFAULT_BLOCK_SIZE_PO2, MAX_LOD};
use crate::math::{Box3i, Vector3i};
use crate::storage::{
    voxel_buffer::{MAX_CHANNELS, SDF_FAR_OUTSIDE},
    ChannelId, VoxelBuffer, VoxelDataBlock, VoxelFormat,
};
use std::collections::HashMap;

#[derive(Debug)]
pub struct VoxelDataMap {
    blocks: HashMap<Vector3i, VoxelDataBlock>,
    key_revisions: HashMap<Vector3i, u64>,
    lod_index: u8,
    format: VoxelFormat,
}

impl VoxelDataMap {
    pub const BLOCK_SIZE_PO2: u8 = DEFAULT_BLOCK_SIZE_PO2;
    pub const BLOCK_SIZE: u32 = 1 << Self::BLOCK_SIZE_PO2;
    pub const BLOCK_SIZE_MASK: u32 = Self::BLOCK_SIZE - 1;

    pub fn new(lod_index: u8) -> Self {
        assert!(
            usize::from(lod_index) < MAX_LOD,
            "LOD index is outside the supported range"
        );
        Self {
            blocks: HashMap::new(),
            key_revisions: HashMap::new(),
            lod_index,
            format: VoxelFormat::new(),
        }
    }

    pub fn create(&mut self, lod_index: u8) {
        assert!(
            usize::from(lod_index) < MAX_LOD,
            "LOD index is outside the supported range"
        );
        self.clear();
        self.lod_index = lod_index;
    }

    pub const fn lod_index(&self) -> u8 {
        self.lod_index
    }

    pub const fn block_size(&self) -> u32 {
        Self::BLOCK_SIZE
    }

    pub const fn block_size_pow2(&self) -> u8 {
        Self::BLOCK_SIZE_PO2
    }

    pub const fn block_size_mask(&self) -> u32 {
        Self::BLOCK_SIZE_MASK
    }

    pub fn set_format(&mut self, format: VoxelFormat) {
        self.format = format;
    }

    pub const fn format(&self) -> &VoxelFormat {
        &self.format
    }

    pub fn voxel_to_block_b(pos: Vector3i, block_size_pow2: u8) -> Vector3i {
        pos >> u32::from(block_size_pow2)
    }

    pub fn voxel_to_block(&self, pos: Vector3i) -> Vector3i {
        Self::voxel_to_block_b(pos, Self::BLOCK_SIZE_PO2)
    }

    pub fn to_local(&self, pos: Vector3i) -> Vector3i {
        pos & Self::BLOCK_SIZE_MASK
    }

    pub fn block_to_voxel(&self, block_pos: Vector3i) -> Vector3i {
        block_pos * Self::BLOCK_SIZE as i32
    }

    #[inline]
    pub fn get_voxel(&self, pos: Vector3i, channel_index: usize) -> u64 {
        let block_pos = self.voxel_to_block(pos);
        let Some(block) = self.get_block(block_pos) else {
            return self.default_raw_value(channel_index);
        };
        if !block.has_voxels() {
            return self.default_raw_value(channel_index);
        }
        let local_pos = self.to_local(pos);
        block
            .voxels()
            .get_voxel(local_pos.x, local_pos.y, local_pos.z, channel_index)
    }

    #[inline]
    pub fn set_voxel(&mut self, value: u64, pos: Vector3i, channel_index: usize) {
        let local_pos = self.to_local(pos);
        let block = self.get_or_create_block_at_voxel_pos(pos);
        block
            .voxels_mut()
            .set_voxel(value, local_pos.x, local_pos.y, local_pos.z, channel_index);
    }

    #[inline]
    pub fn get_voxel_f(&self, pos: Vector3i, channel_index: usize) -> f32 {
        let block_pos = self.voxel_to_block(pos);
        let Some(block) = self.get_block(block_pos) else {
            return SDF_FAR_OUTSIDE;
        };
        if !block.has_voxels() {
            return SDF_FAR_OUTSIDE;
        }
        let local_pos = self.to_local(pos);
        block
            .voxels()
            .get_voxel_f(local_pos.x, local_pos.y, local_pos.z, channel_index)
    }

    #[inline]
    pub fn set_voxel_f(&mut self, value: f32, pos: Vector3i, channel_index: usize) {
        let local_pos = self.to_local(pos);
        let block = self.get_or_create_block_at_voxel_pos(pos);
        block
            .voxels_mut()
            .set_voxel_f(value, local_pos.x, local_pos.y, local_pos.z, channel_index);
    }

    pub fn set_block_buffer(
        &mut self,
        block_pos: Vector3i,
        voxels: VoxelBuffer,
        overwrite: bool,
    ) -> &mut VoxelDataBlock {
        if !self.blocks.contains_key(&block_pos) {
            self.blocks.insert(
                block_pos,
                VoxelDataBlock::with_voxels(voxels, self.lod_index),
            );
        } else if overwrite {
            self.blocks
                .get_mut(&block_pos)
                .expect("block existence was checked")
                .set_voxels(voxels);
        }
        self.blocks
            .get_mut(&block_pos)
            .expect("block exists after set_block_buffer")
    }

    pub fn set_empty_block(&mut self, block_pos: Vector3i, overwrite: bool) -> &mut VoxelDataBlock {
        if !self.blocks.contains_key(&block_pos) {
            self.blocks
                .insert(block_pos, VoxelDataBlock::empty(self.lod_index));
        } else if overwrite {
            self.blocks
                .get_mut(&block_pos)
                .expect("block existence was checked")
                .clear_voxels();
        }
        self.blocks
            .get_mut(&block_pos)
            .expect("block exists after set_empty_block")
    }

    pub fn set_block(
        &mut self,
        block_pos: Vector3i,
        block: VoxelDataBlock,
        overwrite: bool,
    ) -> &mut VoxelDataBlock {
        assert_eq!(
            block.lod_index(),
            self.lod_index,
            "block LOD must match VoxelDataMap LOD"
        );
        if !self.blocks.contains_key(&block_pos) || overwrite {
            self.blocks.insert(block_pos, block);
        }
        self.blocks
            .get_mut(&block_pos)
            .expect("block exists after set_block")
    }

    pub fn remove_block(&mut self, block_pos: Vector3i) -> Option<VoxelDataBlock> {
        self.blocks.remove(&block_pos)
    }

    pub fn get_block(&self, block_pos: Vector3i) -> Option<&VoxelDataBlock> {
        self.blocks.get(&block_pos)
    }

    pub fn get_block_mut(&mut self, block_pos: Vector3i) -> Option<&mut VoxelDataBlock> {
        self.blocks.get_mut(&block_pos)
    }

    pub fn has_block(&self, block_pos: Vector3i) -> bool {
        self.blocks.contains_key(&block_pos)
    }

    pub fn is_block_surrounded(&self, block_pos: Vector3i) -> bool {
        crate::constants::cube_tables::MOORE_NEIGHBORING_3D
            .iter()
            .all(|offset| self.has_block(block_pos + *offset))
    }

    pub fn clear(&mut self) {
        self.blocks.clear();
        self.key_revisions.clear();
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn try_reserve(
        &mut self,
        additional: usize,
    ) -> Result<(), std::collections::TryReserveError> {
        self.blocks.try_reserve(additional)
    }

    /// Public wrapper used by the terrain data view for replication
    /// ordering; returns the raw counter (0 for keys never seen).
    pub fn key_revision_public(&self, block_pos: Vector3i) -> u64 {
        self.key_revision(block_pos)
    }

    pub(crate) fn key_revision(&self, block_pos: Vector3i) -> u64 {
        self.key_revisions.get(&block_pos).copied().unwrap_or(0)
    }

    pub(crate) fn try_reserve_key_revisions(
        &mut self,
        additional: usize,
    ) -> Result<(), std::collections::TryReserveError> {
        self.key_revisions.try_reserve(additional)
    }

    pub(crate) fn commit_key_revision(&mut self, block_pos: Vector3i, revision: u64) {
        debug_assert_eq!(
            self.key_revision(block_pos).checked_add(1),
            Some(revision),
            "key revisions must advance exactly once"
        );
        self.key_revisions.insert(block_pos, revision);
    }

    #[cfg(test)]
    pub(crate) fn set_key_revision_for_test(&mut self, block_pos: Vector3i, revision: u64) {
        self.key_revisions.insert(block_pos, revision);
    }

    pub fn block_positions(&self) -> impl Iterator<Item = Vector3i> + '_ {
        self.blocks.keys().copied()
    }

    pub fn is_area_fully_loaded(&self, voxels_box: Box3i) -> bool {
        let block_box = voxels_box.downscaled(Self::BLOCK_SIZE as i32);
        block_box.all_cells_match(|pos| self.has_block(pos))
    }

    pub fn copy(&self, min_pos: Vector3i, dst_buffer: &mut VoxelBuffer, channels_mask: u32) {
        let channels = channel_indices_from_mask(channels_mask);
        for &channel_index in &channels {
            dst_buffer.set_channel_depth(channel_index, self.format.depths[channel_index]);
        }

        let size = dst_buffer.size();
        if size.x <= 0 || size.y <= 0 || size.z <= 0 {
            return;
        }

        let max_pos = min_pos + size;
        let min_block_pos = self.voxel_to_block(min_pos);
        let max_block_pos = self.voxel_to_block(max_pos - Vector3i::splat(1)) + Vector3i::splat(1);
        let block_size = Vector3i::splat(Self::BLOCK_SIZE as i32);

        for block_pos in Box3i::from_min_max(min_block_pos, max_block_pos).iter_cells_zxy() {
            let src_block_origin = self.block_to_voxel(block_pos);
            if let Some(block) = self.get_block(block_pos).filter(|block| block.has_voxels()) {
                for &channel_index in &channels {
                    dst_buffer.copy_channel_from_area(
                        block.voxels(),
                        min_pos - src_block_origin,
                        block.voxels().size(),
                        Vector3i::zero(),
                        channel_index,
                    );
                }
                continue;
            }

            for &channel_index in &channels {
                dst_buffer.fill_area(
                    self.format
                        .default_raw_value(channel_id_from_index(channel_index)),
                    src_block_origin - min_pos,
                    src_block_origin - min_pos + block_size,
                    channel_index,
                );
            }
        }
    }

    pub fn paste(
        &mut self,
        min_pos: Vector3i,
        src_buffer: &VoxelBuffer,
        channels_mask: u32,
        create_new_blocks: bool,
    ) {
        let channels = channel_indices_from_mask(channels_mask);

        let size = src_buffer.size();
        if size.x <= 0 || size.y <= 0 || size.z <= 0 {
            return;
        }

        let max_pos = min_pos + size;
        let min_block_pos = self.voxel_to_block(min_pos);
        let max_block_pos = self.voxel_to_block(max_pos - Vector3i::splat(1)) + Vector3i::splat(1);

        for block_pos in Box3i::from_min_max(min_block_pos, max_block_pos).iter_cells_zxy() {
            let dst_block_origin = self.block_to_voxel(block_pos);
            let block = if create_new_blocks {
                let block_min_pos = self.block_to_voxel(block_pos);
                Some(self.get_or_create_block_at_voxel_pos(block_min_pos))
            } else {
                self.get_block_mut(block_pos)
            };
            let Some(block) = block else {
                continue;
            };
            if !block.has_voxels() {
                continue;
            }
            for &channel_index in &channels {
                block.voxels_mut().copy_channel_from_area(
                    src_buffer,
                    Vector3i::zero(),
                    size,
                    min_pos - dst_block_origin,
                    channel_index,
                );
            }
        }
    }

    pub fn paste_masked(
        &mut self,
        min_pos: Vector3i,
        src_buffer: &VoxelBuffer,
        channels_mask: u32,
        src_mask_channel: usize,
        src_mask_value: u64,
        create_new_blocks: bool,
    ) {
        let channels = channel_indices_from_mask(channels_mask);
        let src_size = src_buffer.size();
        if src_size.x <= 0 || src_size.y <= 0 || src_size.z <= 0 {
            return;
        }

        // Iterate per destination block (one hashmap lookup per block instead
        // of per source voxel), matching the C++ `paste_masked` strategy.
        let max_pos = min_pos + src_size;
        let min_block_pos = self.voxel_to_block(min_pos);
        let max_block_pos = self.voxel_to_block(max_pos - Vector3i::splat(1)) + Vector3i::splat(1);
        let block_extent = Vector3i::splat(Self::BLOCK_SIZE as i32);

        for block_pos in Box3i::from_min_max(min_block_pos, max_block_pos).iter_cells_zxy() {
            let dst_block_origin = self.block_to_voxel(block_pos);
            let dst_base_pos = min_pos - dst_block_origin;

            let block = if create_new_blocks {
                Some(self.get_or_create_block_at_voxel_pos(dst_block_origin))
            } else {
                self.get_block_mut(block_pos)
            };
            let Some(block) = block else {
                continue;
            };
            if !block.has_voxels() {
                continue;
            }

            // Overlap of this block's interior with the source buffer, in dst
            // local coordinates. VoxelBuffer handles out-of-range reads as
            // no-ops, but clipping lets us skip empty bands entirely.
            let local_min = Vector3i::new(
                dst_base_pos.x.max(0),
                dst_base_pos.y.max(0),
                dst_base_pos.z.max(0),
            );
            let upper = dst_base_pos + src_size;
            let local_max = Vector3i::new(
                upper.x.min(block_extent.x),
                upper.y.min(block_extent.y),
                upper.z.min(block_extent.z),
            );

            let voxels = block.voxels_mut();
            for &channel_index in &channels {
                voxels.read_write_area(local_min, local_max, channel_index, |local_pos, dst_v| {
                    let src_pos = local_pos - dst_base_pos;
                    let mask_value = if channel_index == src_mask_channel {
                        src_buffer.get_voxel(src_pos.x, src_pos.y, src_pos.z, channel_index)
                    } else {
                        src_buffer.get_voxel(src_pos.x, src_pos.y, src_pos.z, src_mask_channel)
                    };
                    if mask_value == src_mask_value {
                        return dst_v;
                    }
                    if channel_index == src_mask_channel {
                        mask_value
                    } else {
                        src_buffer.get_voxel(src_pos.x, src_pos.y, src_pos.z, channel_index)
                    }
                });
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paste_masked_with_destination_mask(
        &mut self,
        min_pos: Vector3i,
        src_buffer: &VoxelBuffer,
        channels_mask: u32,
        src_mask_channel: usize,
        src_mask_value: u64,
        dst_mask_channel: usize,
        dst_writable_values: &[u64],
        create_new_blocks: bool,
    ) {
        let channels = channel_indices_from_mask(channels_mask);
        let src_size = src_buffer.size();
        if src_size.x <= 0 || src_size.y <= 0 || src_size.z <= 0 || dst_writable_values.is_empty() {
            return;
        }

        // Build an O(1) writability lookup. C++ uses a u16-indexed DynamicBitset;
        // we mirror that intent: if every writable value fits in u16 we can use
        // a `Vec<bool>` indexed directly. Otherwise (pathological large values),
        // fall back to a linear scan over the (typically tiny) input slice.
        let writability_lookup = WritabilityLookup::new(dst_writable_values);

        let max_pos = min_pos + src_size;
        let min_block_pos = self.voxel_to_block(min_pos);
        let max_block_pos = self.voxel_to_block(max_pos - Vector3i::splat(1)) + Vector3i::splat(1);
        let block_extent = Vector3i::splat(Self::BLOCK_SIZE as i32);

        for block_pos in Box3i::from_min_max(min_block_pos, max_block_pos).iter_cells_zxy() {
            let dst_block_origin = self.block_to_voxel(block_pos);
            let dst_base_pos = min_pos - dst_block_origin;

            let block = if create_new_blocks {
                Some(self.get_or_create_block_at_voxel_pos(dst_block_origin))
            } else {
                self.get_block_mut(block_pos)
            };
            let Some(block) = block else {
                continue;
            };
            if !block.has_voxels() {
                continue;
            }

            let local_min = Vector3i::new(
                dst_base_pos.x.max(0),
                dst_base_pos.y.max(0),
                dst_base_pos.z.max(0),
            );
            let upper = dst_base_pos + src_size;
            let local_max = Vector3i::new(
                upper.x.min(block_extent.x),
                upper.y.min(block_extent.y),
                upper.z.min(block_extent.z),
            );

            let voxels = block.voxels_mut();
            for &channel_index in channels
                .iter()
                .filter(|&&channel_index| channel_index != dst_mask_channel)
            {
                voxels.read_write_area_with_channel(
                    local_min,
                    local_max,
                    channel_index,
                    dst_mask_channel,
                    |local_pos, dst_v, dst_mask_value| {
                        let src_pos = local_pos - dst_base_pos;
                        if src_buffer.get_voxel(src_pos.x, src_pos.y, src_pos.z, src_mask_channel)
                            == src_mask_value
                        {
                            return dst_v;
                        }
                        if !writability_lookup.is_writable(dst_mask_value) {
                            return dst_v;
                        }
                        src_buffer.get_voxel(src_pos.x, src_pos.y, src_pos.z, channel_index)
                    },
                );
            }

            if channels.contains(&dst_mask_channel) {
                voxels.read_write_area(
                    local_min,
                    local_max,
                    dst_mask_channel,
                    |local_pos, dst_mask_value| {
                        let src_pos = local_pos - dst_base_pos;
                        let src_value = if dst_mask_channel == src_mask_channel {
                            src_buffer.get_voxel(src_pos.x, src_pos.y, src_pos.z, dst_mask_channel)
                        } else {
                            src_buffer.get_voxel(src_pos.x, src_pos.y, src_pos.z, src_mask_channel)
                        };
                        if src_value == src_mask_value {
                            return dst_mask_value;
                        }
                        if !writability_lookup.is_writable(dst_mask_value) {
                            return dst_mask_value;
                        }
                        if dst_mask_channel == src_mask_channel {
                            src_value
                        } else {
                            src_buffer.get_voxel(src_pos.x, src_pos.y, src_pos.z, dst_mask_channel)
                        }
                    },
                );
            }
        }
    }

    fn get_or_create_block_at_voxel_pos(&mut self, pos: Vector3i) -> &mut VoxelDataBlock {
        let block_pos = self.voxel_to_block(pos);
        if !self.blocks.contains_key(&block_pos) {
            let block = self.create_default_block(block_pos);
            return block;
        }
        self.blocks
            .get_mut(&block_pos)
            .expect("block existence was checked")
    }

    fn create_default_block(&mut self, block_pos: Vector3i) -> &mut VoxelDataBlock {
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(Self::BLOCK_SIZE as i32));
        self.format.configure_buffer(&mut voxels);
        self.blocks.insert(
            block_pos,
            VoxelDataBlock::with_voxels(voxels, self.lod_index),
        );
        self.blocks
            .get_mut(&block_pos)
            .expect("block exists after create_default_block")
    }

    fn default_raw_value(&self, channel_index: usize) -> u64 {
        self.format
            .default_raw_value(channel_id_from_index(channel_index))
    }
}

impl Default for VoxelDataMap {
    fn default() -> Self {
        Self::new(0)
    }
}

fn channel_id_from_index(channel_index: usize) -> ChannelId {
    match channel_index {
        0 => ChannelId::Type,
        1 => ChannelId::Sdf,
        2 => ChannelId::Color,
        3 => ChannelId::Indices,
        4 => ChannelId::Weights,
        5 => ChannelId::Data5,
        6 => ChannelId::Data6,
        7 => ChannelId::Data7,
        _ => panic!("channel index is outside the supported range"),
    }
}

fn channel_indices_from_mask(channels_mask: u32) -> Vec<usize> {
    (0..MAX_CHANNELS)
        .filter(|channel_index| (channels_mask & (1u32 << channel_index)) != 0)
        .collect()
}

/// O(1) writability test for masked paste. Mirrors the C++
/// `indices_to_bitarray_u16` fast path: when every writable value fits in a
/// `u16`, build a dense `Vec<bool>` indexed by the destination mask value so
/// the per-voxel probe is a single bounds-checked lookup. When the values are
/// out of `u16` range (pathological for material/typed voxels), fall back to a
/// linear scan over the (typically tiny) input slice — correct, just slower.
enum WritabilityLookup {
    /// Dense bitset indexed by mask value; valid when all values fit in `u16`.
    Indexed(Vec<bool>),
    /// Linear scan over the original slice; used as a fallback for large values
    /// or when the dense table would be unreasonably large.
    Linear(Vec<u64>),
}

impl WritabilityLookup {
    fn new(values: &[u64]) -> Self {
        let max_indexed = values.len().saturating_sub(1);
        // Same ceiling as C++ `indices_to_bitarray_u16` (u16 index space), but
        // we also cap the dense table at a sane size to avoid pathological
        // memory growth if a single large value slips through.
        const INDEXED_MAX_VALUE: u64 = u16::MAX as u64;
        let all_indexable = values
            .iter()
            .all(|value| *value <= INDEXED_MAX_VALUE && (*value as usize) < 65_536);
        if all_indexable && max_indexed < 65_536 {
            let max_value = values.iter().copied().max().unwrap_or(0) as usize;
            let mut table = vec![false; max_value.saturating_add(1).max(1)];
            for value in values {
                table[*value as usize] = true;
            }
            Self::Indexed(table)
        } else {
            Self::Linear(values.to_vec())
        }
    }

    #[inline]
    fn is_writable(&self, value: u64) -> bool {
        match self {
            Self::Indexed(table) => {
                let index = value as usize;
                index < table.len() && table[index]
            }
            Self::Linear(values) => values.contains(&value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::VoxelDataMap;
    use crate::math::{Box3i, Vector3i};
    use crate::storage::{
        voxel_buffer::SDF_FAR_OUTSIDE, ChannelDepth, ChannelId, VoxelBuffer, VoxelFormat,
    };

    #[test]
    fn block_coordinate_conversions_match_cpp_negative_arithmetic_shift() {
        assert_eq!(
            VoxelDataMap::voxel_to_block_b(Vector3i::new(-1, -16, -17), 4),
            Vector3i::new(-1, -1, -2)
        );

        let map = VoxelDataMap::new(0);
        assert_eq!(
            map.voxel_to_block(Vector3i::new(16, 0, -1)),
            Vector3i::new(1, 0, -1)
        );
        assert_eq!(
            map.to_local(Vector3i::new(-1, -16, -17)),
            Vector3i::new(15, 0, 15)
        );
        assert_eq!(
            map.block_to_voxel(Vector3i::new(-2, 0, 3)),
            Vector3i::new(-32, 0, 48)
        );
    }

    #[test]
    fn set_and_get_voxels_create_formatted_blocks() {
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        let mut map = VoxelDataMap::new(2);
        map.set_format(format);

        let pos = Vector3i::new(-1, 0, 17);
        map.set_voxel(42, pos, ChannelId::Type.index());
        map.set_voxel_f(-12.5, pos, ChannelId::Sdf.index());

        assert_eq!(map.lod_index(), 2);
        assert_eq!(map.block_count(), 1);
        assert_eq!(map.get_voxel(pos, ChannelId::Type.index()), 42);
        assert_eq!(map.get_voxel_f(pos, ChannelId::Sdf.index()), -12.5);

        let block = map.get_block(Vector3i::new(-1, 0, 1)).unwrap();
        assert_eq!(
            block.voxels().channel_depth(ChannelId::Sdf.index()),
            ChannelDepth::Bit32
        );
    }

    #[test]
    fn explicit_reservation_preserves_existing_blocks() {
        let mut map = VoxelDataMap::new(0);
        let position = Vector3i::new(-3, 2, 9);
        map.set_empty_block(position, false);

        map.try_reserve(32).unwrap();

        assert!(map.has_block(position));
        map.set_empty_block(Vector3i::new(7, -4, 5), false);
        assert_eq!(map.block_count(), 2);
    }

    #[test]
    fn missing_or_empty_blocks_return_defaults() {
        let mut map = VoxelDataMap::new(0);
        let position = Vector3i::new(4, 5, 6);

        assert_eq!(map.get_voxel(position, ChannelId::Type.index()), 0);
        assert_eq!(
            map.get_voxel(position, ChannelId::Sdf.index()),
            VoxelFormat::new().default_raw_value(ChannelId::Sdf)
        );
        assert_eq!(
            map.get_voxel_f(position, ChannelId::Sdf.index()),
            SDF_FAR_OUTSIDE
        );

        map.set_empty_block(Vector3i::zero(), true);

        assert_eq!(map.get_voxel(position, ChannelId::Type.index()), 0);
        assert_eq!(
            map.get_voxel_f(position, ChannelId::Sdf.index()),
            SDF_FAR_OUTSIDE
        );
    }

    #[test]
    fn block_insert_overwrite_and_removal_match_cpp_contract() {
        let mut map = VoxelDataMap::new(0);
        let block_pos = Vector3i::new(1, 2, 3);
        let mut first = VoxelBuffer::with_size(Vector3i::splat(VoxelDataMap::BLOCK_SIZE as i32));
        first.set_voxel(1, 0, 0, 0, ChannelId::Type.index());
        let mut second = VoxelBuffer::with_size(Vector3i::splat(VoxelDataMap::BLOCK_SIZE as i32));
        second.set_voxel(2, 0, 0, 0, ChannelId::Type.index());

        map.set_block_buffer(block_pos, first, false);
        map.set_block_buffer(block_pos, second, false);

        assert_eq!(map.block_count(), 1);
        assert_eq!(
            map.get_block(block_pos)
                .unwrap()
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            1
        );

        let mut replacement =
            VoxelBuffer::with_size(Vector3i::splat(VoxelDataMap::BLOCK_SIZE as i32));
        replacement.set_voxel(3, 0, 0, 0, ChannelId::Type.index());
        map.set_block_buffer(block_pos, replacement, true);

        assert_eq!(
            map.get_block(block_pos)
                .unwrap()
                .voxels()
                .get_voxel(0, 0, 0, ChannelId::Type.index()),
            3
        );

        map.set_empty_block(block_pos, true);
        assert!(!map.get_block(block_pos).unwrap().has_voxels());

        assert!(map.remove_block(block_pos).is_some());
        assert!(!map.has_block(block_pos));
        assert_eq!(map.block_count(), 0);
    }

    #[test]
    fn area_loaded_requires_every_overlapped_block() {
        let mut map = VoxelDataMap::new(0);
        let area = Box3i::new(Vector3i::new(8, 0, 0), Vector3i::new(24, 16, 16));

        assert!(!map.is_area_fully_loaded(area));

        map.set_empty_block(Vector3i::new(0, 0, 0), true);
        assert!(!map.is_area_fully_loaded(area));

        map.set_empty_block(Vector3i::new(1, 0, 0), true);
        assert!(map.is_area_fully_loaded(area));
    }

    #[test]
    fn paste_fill_writes_across_blocks_and_leaves_neighbors_default() {
        let channel = ChannelId::Type.index();
        let channels_mask = 1u32 << channel;
        let mut source = VoxelBuffer::with_size(Vector3i::new(32, 16, 32));
        source.fill(1, channel);
        let mut map = VoxelDataMap::new(0);
        let area = Box3i::new(Vector3i::new(10, 10, 10), source.size());

        map.paste(area.position, &source, channels_mask, true);

        assert!(area.all_cells_match(|pos| map.get_voxel(pos, channel) == 1));

        let mut outside_is_default = true;
        area.padded(1).for_inner_outline(|pos| {
            if map.get_voxel(pos, channel) != 0 {
                outside_is_default = false;
            }
        });
        assert!(outside_is_default);
    }

    #[test]
    fn paste_without_create_skips_missing_and_empty_blocks() {
        let channel = ChannelId::Type.index();
        let mut source = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        source.fill(5, channel);
        let mut map = VoxelDataMap::new(0);

        map.paste(Vector3i::zero(), &source, 1u32 << channel, false);

        assert_eq!(map.block_count(), 0);
        assert_eq!(map.get_voxel(Vector3i::zero(), channel), 0);

        map.set_empty_block(Vector3i::zero(), true);
        map.paste(Vector3i::zero(), &source, 1u32 << channel, false);

        assert_eq!(map.block_count(), 1);
        assert!(!map.get_block(Vector3i::zero()).unwrap().has_voxels());
        assert_eq!(map.get_voxel(Vector3i::zero(), channel), 0);
    }

    #[test]
    fn copy_round_trips_pasted_voxels_across_blocks() {
        let channel = ChannelId::Type.index();
        let channels_mask = 1u32 << channel;
        let area = Box3i::new(Vector3i::new(10, 10, 10), Vector3i::new(32, 16, 32));
        let mut source = VoxelBuffer::with_size(area.size);
        for pos in Box3i::new(Vector3i::zero(), source.size()).iter_cells_zxy() {
            let value = if pos.x > 0
                && pos.y > 0
                && pos.z > 0
                && pos.x < source.size().x - 1
                && pos.y < source.size().y - 1
                && pos.z < source.size().z - 1
            {
                9
            } else {
                0
            };
            source.set_voxel(value, pos.x, pos.y, pos.z, channel);
        }
        let mut map = VoxelDataMap::new(0);
        map.paste(area.position, &source, channels_mask, true);
        let mut copied = VoxelBuffer::with_size(area.size);

        map.copy(area.position, &mut copied, channels_mask);

        assert!(
            Box3i::new(Vector3i::zero(), area.size).all_cells_match(|pos| {
                copied.get_voxel(pos.x, pos.y, pos.z, channel)
                    == source.get_voxel(pos.x, pos.y, pos.z, channel)
            })
        );
    }

    #[test]
    fn copy_fills_missing_blocks_with_channel_defaults() {
        let channel = ChannelId::Type.index();
        let channels_mask = 1u32 << channel;
        let mut map = VoxelDataMap::new(0);
        map.set_voxel(7, Vector3i::new(1, 1, 1), channel);

        let mut copied = VoxelBuffer::with_size(Vector3i::new(32, 16, 16));
        copied.fill(99, channel);
        map.copy(Vector3i::zero(), &mut copied, channels_mask);

        assert_eq!(copied.get_voxel(1, 1, 1, channel), 7);
        assert_eq!(copied.get_voxel(20, 1, 1, channel), 0);
    }

    #[test]
    fn paste_uniform_source_overwrites_existing_voxels() {
        let channel = ChannelId::Type.index();
        let channels_mask = 1u32 << channel;
        let mut map = VoxelDataMap::new(0);
        map.set_voxel(42, Vector3i::new(1, 1, 1), channel);
        let source = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));

        map.paste(Vector3i::zero(), &source, channels_mask, true);

        assert_eq!(map.get_voxel(Vector3i::new(1, 1, 1), channel), 0);
    }

    #[test]
    fn paste_masked_skips_source_mask_value_and_preserves_existing_voxels() {
        let channel = ChannelId::Type.index();
        let channels_mask = 1u32 << channel;
        let voxel_value = 1;
        let masked_value = 2;
        let mut source = VoxelBuffer::with_size(Vector3i::new(32, 16, 32));
        source.fill(masked_value, channel);
        source.fill_area(
            voxel_value,
            Vector3i::new(1, 1, 1),
            source.size() - Vector3i::new(1, 1, 1),
            channel,
        );
        let mut map = VoxelDataMap::new(0);
        let area = Box3i::new(Vector3i::new(10, 10, 10), source.size());

        map.paste_masked(
            area.position,
            &source,
            channels_mask,
            channel,
            masked_value,
            true,
        );

        assert!(area
            .padded(-1)
            .all_cells_match(|pos| { map.get_voxel(pos, channel) == voxel_value }));

        let mut outline_is_default = true;
        area.for_inner_outline(|pos| {
            if map.get_voxel(pos, channel) != 0 {
                outline_is_default = false;
            }
        });
        assert!(outline_is_default);
    }

    #[test]
    fn paste_masked_with_destination_mask_only_writes_writable_values() {
        let channel = ChannelId::Type.index();
        let channels_mask = 1u32 << channel;
        let box_in_voxels =
            Box3i::from_min_max(Vector3i::new(-10, -5, -10), Vector3i::new(10, 5, 10));
        let mut map = VoxelDataMap::new(0);
        for pos in box_in_voxels.iter_cells() {
            let value = (pos.y - box_in_voxels.position.y) as u64;
            map.set_voxel(value, pos, channel);
        }

        let mut source = VoxelBuffer::with_size(box_in_voxels.size);
        source.fill(100, channel);
        let writable_values = [0, 2, 5, 6];

        map.paste_masked_with_destination_mask(
            box_in_voxels.position,
            &source,
            channels_mask,
            channel,
            999,
            channel,
            &writable_values,
            false,
        );

        assert!(box_in_voxels.all_cells_match(|pos| {
            let original_value = (pos.y - box_in_voxels.position.y) as u64;
            let writable = writable_values.contains(&original_value);
            let expected = if writable { 100 } else { original_value };
            map.get_voxel(pos, channel) == expected
        }));
    }

    #[test]
    fn paste_masked_with_destination_mask_handles_large_writable_value_fallback() {
        // Writable value above u16 forces the linear-scan fallback path; the
        // dense `Vec<bool>` table is not built. The paste must still match.
        let channel = ChannelId::Type.index();
        let channels_mask = 1u32 << channel;
        let mut format = VoxelFormat::new();
        format.depths[channel] = ChannelDepth::Bit32;
        let mut map = VoxelDataMap::new(0);
        map.set_format(format);
        map.set_voxel(70_000, Vector3i::new(1, 1, 1), channel);
        map.set_voxel(70_001, Vector3i::new(2, 2, 2), channel);

        let mut source = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        source.set_channel_depth(channel, ChannelDepth::Bit32);
        source.fill(7, channel);

        map.paste_masked_with_destination_mask(
            Vector3i::zero(),
            &source,
            channels_mask,
            channel,
            // Source mask uses a sentinel channel index that the source never
            // populated, so every source voxel is a candidate for writing.
            999,
            channel,
            &[70_000],
            false,
        );

        assert_eq!(map.get_voxel(Vector3i::new(1, 1, 1), channel), 7);
        assert_eq!(map.get_voxel(Vector3i::new(2, 2, 2), channel), 70_001);
    }

    #[test]
    fn paste_masked_handles_cross_block_paste_with_create_new_blocks() {
        // Verifies the per-block iteration writes across multiple blocks when
        // the paste area spans more than one block, exercising the local-min
        // / local-max clipping in the inner loop.
        let channel = ChannelId::Type.index();
        let channels_mask = 1u32 << channel;
        let masked_value = 9;
        let area = Box3i::from_min_max(Vector3i::new(8, 8, 8), Vector3i::new(40, 40, 40));
        let mut source = VoxelBuffer::with_size(area.size);
        source.fill(masked_value, channel);
        let write_value = 3;
        source.fill_area(
            write_value,
            Vector3i::new(1, 1, 1),
            source.size() - Vector3i::splat(1),
            channel,
        );

        let mut map = VoxelDataMap::new(0);
        map.paste_masked(
            area.position,
            &source,
            channels_mask,
            channel,
            masked_value,
            true,
        );

        assert!(area
            .padded(-1)
            .all_cells_match(|pos| map.get_voxel(pos, channel) == write_value));
        // The single-voxel outline of the area was masked out, so it remains
        // at the block default rather than `write_value`.
        assert_ne!(map.get_voxel(area.position, channel), write_value);
    }

    #[test]
    fn masked_paste_uses_depth_hoisted_destination_writes() {
        let source = include_str!("voxel_data_map.rs");
        let helper_marker = [".read_write", "_area"].concat();
        assert!(
            source.contains(&helper_marker),
            "masked paste should mutate destination blocks through VoxelBuffer's depth-hoisted area helper"
        );

        let old_write_marker = ["voxels", ".set_voxel(value, lx, ly, lz, channel_index)"].concat();
        assert!(
            !source.contains(&old_write_marker),
            "masked paste should not redispatch destination channel depth for every voxel write"
        );
    }
}
