//! Per-block instance residency. One [`InstanceBlock`] holds every scatter
//! instance generated for a single data/mesh block so paging can insert and
//! drop them with the block instead of rebuilding the whole world.

use super::library::InstanceLibrary;
use super::scatter::{BlockInstanceData, InstanceGenerator, RandomScatterGenerator, ScatterConfig};
use crate::math::{Vector3f, Vector3i};
use crate::storage::{ChannelId, VoxelBuffer};
use std::collections::HashMap;

/// Instances generated for one terrain block.
#[derive(Debug, Clone)]
pub struct InstanceBlock {
    pub position_in_blocks: Vector3i,
    pub lod_index: u8,
    pub instances: Vec<BlockInstanceData>,
}

impl InstanceBlock {
    pub fn new(
        position_in_blocks: Vector3i,
        lod_index: u8,
        instances: Vec<BlockInstanceData>,
    ) -> Self {
        Self {
            position_in_blocks,
            lod_index,
            instances,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    pub fn count_for_item(&self, item_index: u32) -> usize {
        self.instances
            .iter()
            .filter(|instance| instance.item_index == item_index)
            .count()
    }
}

/// Resident instance blocks, keyed by `(x, y, z, lod)`.
#[derive(Debug, Clone, Default)]
pub struct InstanceBlockMap {
    blocks: HashMap<(i32, i32, i32, u8), InstanceBlock>,
}

impl InstanceBlockMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn instance_count(&self) -> usize {
        self.blocks
            .values()
            .map(|block| block.instances.len())
            .sum()
    }

    pub fn upsert(&mut self, block: InstanceBlock) {
        let key = block_key(block.position_in_blocks, block.lod_index);
        self.blocks.insert(key, block);
    }

    pub fn remove(&mut self, position: Vector3i, lod_index: u8) -> Option<InstanceBlock> {
        self.blocks.remove(&block_key(position, lod_index))
    }

    pub fn get(&self, position: Vector3i, lod_index: u8) -> Option<&InstanceBlock> {
        self.blocks.get(&block_key(position, lod_index))
    }

    pub fn contains(&self, position: Vector3i, lod_index: u8) -> bool {
        self.blocks.contains_key(&block_key(position, lod_index))
    }

    pub fn keys(&self) -> impl Iterator<Item = (Vector3i, u8)> + '_ {
        self.blocks
            .keys()
            .map(|&(x, y, z, lod)| (Vector3i::new(x, y, z), lod))
    }
}

fn block_key(position: Vector3i, lod_index: u8) -> (i32, i32, i32, u8) {
    (position.x, position.y, position.z, lod_index)
}

/// Surface voxels: solid with air immediately below, in world space.
pub fn extract_surface_points(
    buffer: &VoxelBuffer,
    origin: Vector3f,
    channel: usize,
) -> (Vec<Vector3f>, Vec<Vector3f>) {
    let size = buffer.size();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    if size.y <= 1 {
        return (positions, normals);
    }
    let type_channel = if channel < crate::storage::voxel_buffer::MAX_CHANNELS {
        channel
    } else {
        ChannelId::Type.index()
    };
    for z in 0..size.z {
        for y in 1..size.y {
            for x in 0..size.x {
                let solid = buffer.get_voxel(x, y, z, type_channel);
                let below = buffer.get_voxel(x, y - 1, z, type_channel);
                if solid == 0 || below != 0 {
                    continue;
                }
                positions.push(Vector3f::new(
                    origin.x + x as f32 + 0.5,
                    origin.y + y as f32,
                    origin.z + z as f32 + 0.5,
                ));
                normals.push(Vector3f::new(0.0, 1.0, 0.0));
            }
        }
    }
    (positions, normals)
}

/// Scatter every library item over the given surface and flatten the result.
pub fn scatter_block_instances(
    library: &InstanceLibrary,
    config: &ScatterConfig,
    density_multiplier: f32,
    positions: &[Vector3f],
    normals: &[Vector3f],
) -> Vec<BlockInstanceData> {
    if positions.is_empty() || library.is_empty() || !density_multiplier.is_finite() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (idx, item) in library.items.iter().enumerate() {
        let Ok(item_index) = u32::try_from(idx) else {
            break;
        };
        let gen = RandomScatterGenerator {
            density: item.density * density_multiplier,
            min_scale: item.min_scale,
            max_scale: item.max_scale,
            snap_to_normal: item.snap_to_normal,
        };
        out.extend(gen.generate(positions, normals, item_index, config));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instancing::InstanceLibraryItem;
    use crate::storage::VoxelBuffer;

    #[test]
    fn extract_surface_points_finds_solid_above_air() {
        let mut buffer = VoxelBuffer::with_size(Vector3i::new(2, 3, 1));
        buffer.set_voxel(1, 0, 1, 0, ChannelId::Type.index());
        buffer.set_voxel(1, 1, 1, 0, ChannelId::Type.index());
        let (positions, normals) =
            extract_surface_points(&buffer, Vector3f::new(16.0, 0.0, 0.0), 0);
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].x, 16.5);
        assert_eq!(normals.len(), 2);
    }

    #[test]
    fn instance_block_map_upsert_and_remove() {
        let mut map = InstanceBlockMap::new();
        map.upsert(InstanceBlock::new(
            Vector3i::new(1, 0, 2),
            0,
            vec![BlockInstanceData {
                position: Vector3f::new(1.0, 2.0, 3.0),
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: 1.0,
                item_index: 0,
            }],
        ));
        assert_eq!(map.len(), 1);
        assert_eq!(map.instance_count(), 1);
        assert!(map.contains(Vector3i::new(1, 0, 2), 0));
        assert!(map.remove(Vector3i::new(1, 0, 2), 0).is_some());
        assert!(map.is_empty());
    }

    #[test]
    fn scatter_block_instances_tags_item_index() {
        let mut library = InstanceLibrary::new();
        library.add_item(InstanceLibraryItem {
            density: 1.0,
            min_scale: 1.0,
            max_scale: 1.0,
            ..Default::default()
        });
        let positions = [Vector3f::new(0.0, 1.0, 0.0)];
        let normals = [Vector3f::new(0.0, 1.0, 0.0)];
        let instances = scatter_block_instances(
            &library,
            &ScatterConfig::default(),
            1.0,
            &positions,
            &normals,
        );
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].item_index, 0);
    }

    #[test]
    fn scene_typed_items_scatter_like_multimesh_items() {
        // Scene items carry the same BlockInstanceData payload (position,
        // rotation, scale, item_index); the gdext instancer consumes it to
        // spawn real nodes instead of a MultiMesh.
        let mut library = InstanceLibrary::new();
        library.add_item(InstanceLibraryItem {
            mesh_type: crate::instancing::InstanceMeshType::Scene,
            density: 1.0,
            min_scale: 1.0,
            max_scale: 1.0,
            ..Default::default()
        });
        library.add_item(InstanceLibraryItem {
            density: 1.0,
            min_scale: 1.0,
            max_scale: 1.0,
            ..Default::default()
        });
        let positions = [Vector3f::new(0.0, 1.0, 0.0), Vector3f::new(2.0, 1.0, 0.0)];
        let normals = [Vector3f::new(0.0, 1.0, 0.0); 2];
        let instances = scatter_block_instances(
            &library,
            &ScatterConfig::default(),
            1.0,
            &positions,
            &normals,
        );
        // Each item scatters over every surface point.
        assert_eq!(instances.len(), 4);
        assert_eq!(
            instances
                .iter()
                .filter(|instance| instance.item_index == 0)
                .count(),
            2
        );
        assert_eq!(
            instances
                .iter()
                .filter(|instance| instance.item_index == 1)
                .count(),
            2
        );
        for instance in &instances {
            assert!(instance.rotation[3] != 0.0, "rotation must be a valid quat");
            assert!(instance.scale.is_finite());
        }
    }
}
