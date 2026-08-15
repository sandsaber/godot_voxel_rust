//! Checked coordinate math for finite, nested variable-LOD clipboxes.

use crate::constants::voxel_constants::MAX_LOD;
use crate::math::{Box3i, Vector3i};
use crate::meshers::MeshBlockLocation;
use crate::storage::VoxelData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodMathError {
    InvalidBlockSize,
    InvalidLodCount,
    NegativeDistance,
    DataBlockSizeMismatch { settings: i32, voxel_data: i32 },
    UnsupportedMeshToDataFactor { mesh: i32, data: i32 },
    UnalignedBounds { stride: i32 },
    CoordinateOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LodClipboxSettings {
    pub data_block_size: i32,
    pub mesh_block_size: i32,
    pub lod_count: u8,
    pub lod0_distance_voxels: i32,
    pub secondary_distance_voxels: i32,
    pub unload_hysteresis_blocks: i32,
}

impl LodClipboxSettings {
    pub fn validate_for(&self, data: &VoxelData) -> Result<(), LodMathError> {
        self.validate_block_sizes()?;

        let voxel_data_block_size =
            i32::try_from(data.block_size()).map_err(|_| LodMathError::CoordinateOverflow)?;
        if self.data_block_size != voxel_data_block_size {
            return Err(LodMathError::DataBlockSizeMismatch {
                settings: self.data_block_size,
                voxel_data: voxel_data_block_size,
            });
        }
        if self.mesh_block_size != self.data_block_size {
            return Err(LodMathError::UnsupportedMeshToDataFactor {
                mesh: self.mesh_block_size,
                data: self.data_block_size,
            });
        }

        self.validate_lod_and_distances()?;
        self.validate_bounds_alignment(data.bounds())?;
        Ok(())
    }

    /// Validate settings against an already-constructed volume.
    pub fn validate_for_bounds(
        &self,
        data_block_size: i32,
        bounds: Box3i,
    ) -> Result<(), LodMathError> {
        self.validate_block_sizes()?;
        if self.data_block_size != data_block_size {
            return Err(LodMathError::DataBlockSizeMismatch {
                settings: self.data_block_size,
                voxel_data: data_block_size,
            });
        }
        if self.mesh_block_size != self.data_block_size {
            return Err(LodMathError::UnsupportedMeshToDataFactor {
                mesh: self.mesh_block_size,
                data: self.data_block_size,
            });
        }
        self.validate_lod_and_distances()?;
        self.validate_bounds_alignment(bounds)?;
        Ok(())
    }

    fn validate_for_math(&self) -> Result<(), LodMathError> {
        self.validate_block_sizes()?;
        if self.mesh_block_size != self.data_block_size {
            return Err(LodMathError::UnsupportedMeshToDataFactor {
                mesh: self.mesh_block_size,
                data: self.data_block_size,
            });
        }
        self.validate_lod_and_distances()
    }

    fn validate_block_sizes(&self) -> Result<(), LodMathError> {
        if self.data_block_size <= 0 || self.mesh_block_size <= 0 {
            return Err(LodMathError::InvalidBlockSize);
        }
        Ok(())
    }

    fn validate_lod_and_distances(&self) -> Result<(), LodMathError> {
        if self.lod_count == 0 || usize::from(self.lod_count) >= MAX_LOD {
            return Err(LodMathError::InvalidLodCount);
        }
        if self.lod0_distance_voxels < 0
            || self.secondary_distance_voxels < 0
            || self.unload_hysteresis_blocks < 0
        {
            return Err(LodMathError::NegativeDistance);
        }
        Ok(())
    }

    fn validate_bounds_alignment(&self, bounds_voxels: Box3i) -> Result<(), LodMathError> {
        let coarsest_lod = self
            .lod_count
            .checked_sub(1)
            .ok_or(LodMathError::InvalidLodCount)?;
        let stride = lod_block_stride(self.mesh_block_size, coarsest_lod)?;
        let stride_i64 = i64::from(stride);
        let (min, max) = box_min_max(bounds_voxels)?;
        let validated_bounds = checked_box3i(min, max)?;
        if validated_bounds.is_empty() {
            return Ok(());
        }
        if (0..3).any(|axis| {
            min[axis].rem_euclid(stride_i64) != 0 || max[axis].rem_euclid(stride_i64) != 0
        }) {
            return Err(LodMathError::UnalignedBounds { stride });
        }
        Ok(())
    }

    #[cfg(test)]
    const fn test_three_lods() -> Self {
        Self {
            data_block_size: 16,
            mesh_block_size: 16,
            lod_count: 3,
            lod0_distance_voxels: 96,
            secondary_distance_voxels: 64,
            unload_hysteresis_blocks: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LodClipboxes {
    pub mesh_load: Vec<Box3i>,
    pub mesh_retain: Vec<Box3i>,
    pub data_load: Vec<Box3i>,
    pub data_retain: Vec<Box3i>,
}

impl Eq for LodClipboxes {}

pub fn lod_block_stride(base_block_size: i32, lod_index: u8) -> Result<i32, LodMathError> {
    if base_block_size <= 0 {
        return Err(LodMathError::InvalidBlockSize);
    }
    let multiplier = 1_i64
        .checked_shl(u32::from(lod_index))
        .ok_or(LodMathError::CoordinateOverflow)?;
    let stride = i64::from(base_block_size)
        .checked_mul(multiplier)
        .ok_or(LodMathError::CoordinateOverflow)?;
    i32::try_from(stride).map_err(|_| LodMathError::CoordinateOverflow)
}

pub fn bounds_in_lod_blocks(
    bounds_voxels: Box3i,
    base_block_size: i32,
    lod_index: u8,
) -> Result<Box3i, LodMathError> {
    let stride = i64::from(lod_block_stride(base_block_size, lod_index)?);
    let (voxel_min, voxel_max) = box_min_max(bounds_voxels)?;
    let validated_bounds = checked_box3i(voxel_min, voxel_max)?;
    if validated_bounds.is_empty() {
        return Ok(Box3i::default());
    }

    let mut block_min = [0; 3];
    let mut block_max = [0; 3];
    for axis in 0..3 {
        block_min[axis] = voxel_min[axis].div_euclid(stride);
        block_max[axis] = div_ceil_euclid(voxel_max[axis], stride)?;
    }
    checked_box3i(block_min, block_max)
}

pub fn clipped_meshing_data_box(
    location: MeshBlockLocation,
    block_size: i32,
    padding: i32,
    bounds_voxels: Box3i,
) -> Result<Box3i, LodMathError> {
    if padding < 0 {
        return Err(LodMathError::NegativeDistance);
    }

    let bounds_in_blocks = bounds_in_lod_blocks(bounds_voxels, block_size, location.lod_index)?;
    if !bounds_in_blocks.contains_point(location.position_in_blocks) {
        return Ok(Box3i::default());
    }

    let stride = i64::from(lod_block_stride(block_size, location.lod_index)?);
    let block_position = vector_to_i64(location.position_in_blocks);
    let mut voxel_min = [0; 3];
    let mut voxel_max = [0; 3];
    for axis in 0..3 {
        voxel_min[axis] = block_position[axis]
            .checked_mul(stride)
            .ok_or(LodMathError::CoordinateOverflow)?;
        voxel_max[axis] = voxel_min[axis]
            .checked_add(stride)
            .ok_or(LodMathError::CoordinateOverflow)?;
    }
    checked_box3i(voxel_min, voxel_max)?;

    let padding = i64::from(padding);
    let mut halo_min = [0; 3];
    let mut halo_max = [0; 3];
    for axis in 0..3 {
        halo_min[axis] = block_position[axis]
            .checked_sub(padding)
            .ok_or(LodMathError::CoordinateOverflow)?;
        halo_max[axis] = block_position[axis]
            .checked_add(1)
            .and_then(|value| value.checked_add(padding))
            .ok_or(LodMathError::CoordinateOverflow)?;
    }
    checked_box3i(halo_min, halo_max)?;

    let (bounds_min, bounds_max) = box_min_max(bounds_in_blocks)?;
    for axis in 0..3 {
        halo_min[axis] = halo_min[axis].max(bounds_min[axis]);
        halo_max[axis] = halo_max[axis].min(bounds_max[axis]);
    }
    checked_box3i(halo_min, halo_max)
}

pub fn compute_lod_clipboxes(
    viewer_position_voxels: Vector3i,
    view_distance_voxels: Vector3i,
    bounds_voxels: Box3i,
    settings: LodClipboxSettings,
) -> Result<LodClipboxes, LodMathError> {
    settings.validate_for_math()?;
    settings.validate_bounds_alignment(bounds_voxels)?;
    if view_distance_voxels.x < 0 || view_distance_voxels.y < 0 || view_distance_voxels.z < 0 {
        return Err(LodMathError::NegativeDistance);
    }

    let lod_count = usize::from(settings.lod_count);
    let base_stride = i64::from(settings.mesh_block_size);
    let lod0_chunks = i64::from(settings.lod0_distance_voxels)
        .div_euclid(base_stride)
        .max(1);
    let lodn_chunks = i64::from(settings.secondary_distance_voxels)
        .div_euclid(base_stride)
        .max(1);
    let max_view_distance = vector_to_i64(view_distance_voxels);

    let mut bounds_per_lod = Vec::with_capacity(lod_count);
    let mut mesh_unclipped = Vec::with_capacity(lod_count);
    for lod in 0..settings.lod_count {
        let bounds = bounds_in_lod_blocks(bounds_voxels, settings.mesh_block_size, lod)?;
        bounds_per_lod.push(bounds);

        let stride = i64::from(lod_block_stride(settings.mesh_block_size, lod)?);
        let distance_chunks = relative_lod_distance_chunks(
            lod,
            settings.lod_count,
            lod0_chunks,
            lodn_chunks,
            stride,
            max_view_distance,
        );
        let mut distance_voxels = [0; 3];
        for axis in 0..3 {
            distance_voxels[axis] = distance_chunks[axis]
                .checked_mul(stride)
                .ok_or(LodMathError::CoordinateOverflow)?;
        }

        let make_even = lod + 1 != settings.lod_count;
        let mut mesh_box =
            base_box_in_chunks(viewer_position_voxels, distance_voxels, stride, make_even)?;
        if let Some(child) = mesh_unclipped.last().copied() {
            let child_minimum = minimal_parent_box(child, make_even)?;
            mesh_box = merge_boxes(mesh_box, child_minimum)?;
        }
        mesh_unclipped.push(mesh_box);
    }

    let mut mesh_load = Vec::with_capacity(lod_count);
    for (mesh_box, bounds) in mesh_unclipped
        .into_iter()
        .zip(bounds_per_lod.iter().copied())
    {
        mesh_load.push(checked_box_intersection(mesh_box, bounds)?);
    }

    let mut data_load = Vec::with_capacity(lod_count);
    for (mesh_box, bounds) in mesh_load
        .iter()
        .copied()
        .zip(bounds_per_lod.iter().copied())
    {
        data_load.push(checked_box_intersection(padded_box(mesh_box, 1)?, bounds)?);
    }

    let retain_padding = i64::from(settings.unload_hysteresis_blocks);
    let mesh_retain = build_retain_boxes(&mesh_load, &bounds_per_lod, retain_padding, true)?;
    let data_retain = build_retain_boxes(&data_load, &bounds_per_lod, retain_padding, false)?;

    Ok(LodClipboxes {
        mesh_load,
        mesh_retain,
        data_load,
        data_retain,
    })
}

fn relative_lod_distance_chunks(
    lod: u8,
    lod_count: u8,
    lod0_chunks: i64,
    lodn_chunks: i64,
    lod_stride: i64,
    max_view_distance: [i64; 3],
) -> [i64; 3] {
    let scalar = if lod == 0 {
        lod0_chunks
    } else {
        (lod0_chunks >> u32::from(lod)) + lodn_chunks
    }
    .max(1);
    let mut distance = [scalar; 3];
    if lod + 1 == lod_count {
        for axis in 0..3 {
            distance[axis] =
                distance[axis].max(div_ceil_nonnegative(max_view_distance[axis], lod_stride));
        }
    }
    distance
}

fn base_box_in_chunks(
    viewer_position_voxels: Vector3i,
    distance_voxels: [i64; 3],
    chunk_size: i64,
    make_even: bool,
) -> Result<Box3i, LodMathError> {
    let viewer = vector_to_i64(viewer_position_voxels);
    let mut voxel_min = [0; 3];
    let mut voxel_max = [0; 3];
    for axis in 0..3 {
        voxel_min[axis] = viewer[axis]
            .checked_sub(distance_voxels[axis])
            .ok_or(LodMathError::CoordinateOverflow)?;
        voxel_max[axis] = viewer[axis]
            .checked_add(distance_voxels[axis])
            .and_then(|value| value.checked_add(1))
            .ok_or(LodMathError::CoordinateOverflow)?;
    }
    checked_box3i(voxel_min, voxel_max)?;

    let mut chunk_min = [0; 3];
    let mut chunk_max = [0; 3];
    for axis in 0..3 {
        chunk_min[axis] = voxel_min[axis].div_euclid(chunk_size);
        chunk_max[axis] = div_ceil_euclid(voxel_max[axis], chunk_size)?;
    }
    if make_even {
        round_outward_to_even(&mut chunk_min, &mut chunk_max)?;
    }
    checked_box3i(chunk_min, chunk_max)
}

fn minimal_parent_box(child: Box3i, make_even: bool) -> Result<Box3i, LodMathError> {
    if child.is_empty() {
        return Ok(Box3i::default());
    }
    let (child_min, child_max) = box_min_max(child)?;
    let mut parent_min = [0; 3];
    let mut parent_max = [0; 3];
    for axis in 0..3 {
        parent_min[axis] = child_min[axis]
            .div_euclid(2)
            .checked_sub(1)
            .ok_or(LodMathError::CoordinateOverflow)?;
        parent_max[axis] = div_ceil_euclid(child_max[axis], 2)?
            .checked_add(1)
            .ok_or(LodMathError::CoordinateOverflow)?;
    }
    if make_even {
        round_outward_to_even(&mut parent_min, &mut parent_max)?;
    }
    checked_box3i(parent_min, parent_max)
}

fn build_retain_boxes(
    load_boxes: &[Box3i],
    bounds_per_lod: &[Box3i],
    padding: i64,
    preserve_even_mesh_topology: bool,
) -> Result<Vec<Box3i>, LodMathError> {
    let mut retain_boxes = Vec::with_capacity(load_boxes.len());
    for (lod, (&load_box, &bounds)) in load_boxes.iter().zip(bounds_per_lod.iter()).enumerate() {
        let mut retain = padded_box(load_box, padding)?;
        if let Some(child) = retain_boxes.last().copied() {
            let make_even = preserve_even_mesh_topology && lod + 1 != load_boxes.len();
            retain = merge_boxes(retain, minimal_parent_box(child, make_even)?)?;
        }
        retain_boxes.push(checked_box_intersection(retain, bounds)?);
    }
    Ok(retain_boxes)
}

fn round_outward_to_even(min: &mut [i64; 3], max: &mut [i64; 3]) -> Result<(), LodMathError> {
    for axis in 0..3 {
        min[axis] = min[axis]
            .div_euclid(2)
            .checked_mul(2)
            .ok_or(LodMathError::CoordinateOverflow)?;
        max[axis] = div_ceil_euclid(max[axis], 2)?
            .checked_mul(2)
            .ok_or(LodMathError::CoordinateOverflow)?;
    }
    Ok(())
}

fn padded_box(value: Box3i, padding: i64) -> Result<Box3i, LodMathError> {
    if value.is_empty() {
        return Ok(Box3i::default());
    }
    let (mut min, mut max) = box_min_max(value)?;
    for axis in 0..3 {
        min[axis] = min[axis]
            .checked_sub(padding)
            .ok_or(LodMathError::CoordinateOverflow)?;
        max[axis] = max[axis]
            .checked_add(padding)
            .ok_or(LodMathError::CoordinateOverflow)?;
    }
    checked_box3i(min, max)
}

fn merge_boxes(a: Box3i, b: Box3i) -> Result<Box3i, LodMathError> {
    if a.is_empty() {
        return Ok(b);
    }
    if b.is_empty() {
        return Ok(a);
    }
    let (a_min, a_max) = box_min_max(a)?;
    let (b_min, b_max) = box_min_max(b)?;
    let mut min = [0; 3];
    let mut max = [0; 3];
    for axis in 0..3 {
        min[axis] = a_min[axis].min(b_min[axis]);
        max[axis] = a_max[axis].max(b_max[axis]);
    }
    checked_box3i(min, max)
}

pub(crate) fn checked_box_intersection(a: Box3i, b: Box3i) -> Result<Box3i, LodMathError> {
    if a.is_empty() || b.is_empty() {
        return Ok(Box3i::default());
    }
    let (a_min, a_max) = box_min_max(a)?;
    let (b_min, b_max) = box_min_max(b)?;
    let mut min = [0; 3];
    let mut max = [0; 3];
    for axis in 0..3 {
        min[axis] = a_min[axis].max(b_min[axis]);
        max[axis] = a_max[axis].min(b_max[axis]);
    }
    checked_box3i(min, max)
}

fn box_min_max(value: Box3i) -> Result<([i64; 3], [i64; 3]), LodMathError> {
    let min = vector_to_i64(value.position);
    let size = vector_to_i64(value.size);
    let mut max = [0; 3];
    for axis in 0..3 {
        max[axis] = min[axis]
            .checked_add(size[axis])
            .ok_or(LodMathError::CoordinateOverflow)?;
    }
    Ok((min, max))
}

fn vector_to_i64(value: Vector3i) -> [i64; 3] {
    [i64::from(value.x), i64::from(value.y), i64::from(value.z)]
}

fn checked_box3i(min: [i64; 3], max: [i64; 3]) -> Result<Box3i, LodMathError> {
    if (0..3).any(|axis| max[axis] <= min[axis]) {
        return Ok(Box3i::default());
    }

    let min_x = i32::try_from(min[0]).map_err(|_| LodMathError::CoordinateOverflow)?;
    let min_y = i32::try_from(min[1]).map_err(|_| LodMathError::CoordinateOverflow)?;
    let min_z = i32::try_from(min[2]).map_err(|_| LodMathError::CoordinateOverflow)?;
    i32::try_from(max[0]).map_err(|_| LodMathError::CoordinateOverflow)?;
    i32::try_from(max[1]).map_err(|_| LodMathError::CoordinateOverflow)?;
    i32::try_from(max[2]).map_err(|_| LodMathError::CoordinateOverflow)?;
    let size_x = i32::try_from(
        max[0]
            .checked_sub(min[0])
            .ok_or(LodMathError::CoordinateOverflow)?,
    )
    .map_err(|_| LodMathError::CoordinateOverflow)?;
    let size_y = i32::try_from(
        max[1]
            .checked_sub(min[1])
            .ok_or(LodMathError::CoordinateOverflow)?,
    )
    .map_err(|_| LodMathError::CoordinateOverflow)?;
    let size_z = i32::try_from(
        max[2]
            .checked_sub(min[2])
            .ok_or(LodMathError::CoordinateOverflow)?,
    )
    .map_err(|_| LodMathError::CoordinateOverflow)?;

    Ok(Box3i::new(
        Vector3i::new(min_x, min_y, min_z),
        Vector3i::new(size_x, size_y, size_z),
    ))
}

fn div_ceil_euclid(value: i64, divisor: i64) -> Result<i64, LodMathError> {
    let quotient = value.div_euclid(divisor);
    if value.rem_euclid(divisor) == 0 {
        Ok(quotient)
    } else {
        quotient
            .checked_add(1)
            .ok_or(LodMathError::CoordinateOverflow)
    }
}

fn div_ceil_nonnegative(value: i64, divisor: i64) -> i64 {
    value.div_euclid(divisor) + i64::from(value.rem_euclid(divisor) != 0)
}

#[cfg(test)]
mod tests {
    use super::{
        bounds_in_lod_blocks, clipped_meshing_data_box, compute_lod_clipboxes, lod_block_stride,
        LodClipboxSettings, LodMathError,
    };
    use crate::math::{Box3i, Vector3i};
    use crate::meshers::MeshBlockLocation;
    use crate::storage::VoxelData;
    use crate::terrain::variable_lod_coverage::TransitionFace;

    #[derive(Debug)]
    struct FaceFixture {
        face: TransitionFace,
        location: MeshBlockLocation,
        expected_clipped_halo: Box3i,
        bounds_in_lod_blocks: Box3i,
    }

    fn all_six_face_locations(bounds: Box3i, block_size: i32, lods: [u8; 3]) -> Vec<FaceFixture> {
        assert_eq!(
            bounds,
            Box3i::new(
                Vector3i::new(-256, -192, -128),
                Vector3i::new(512, 384, 256),
            )
        );
        assert_eq!(block_size, 16);

        let expected_bounds = [
            Box3i::new(Vector3i::new(-16, -12, -8), Vector3i::new(32, 24, 16)),
            Box3i::new(Vector3i::new(-8, -6, -4), Vector3i::new(16, 12, 8)),
            Box3i::new(Vector3i::new(-4, -3, -2), Vector3i::new(8, 6, 4)),
        ];
        let mut fixtures = Vec::with_capacity(18);

        for (lod, bounds_in_lod_blocks) in lods.into_iter().zip(expected_bounds) {
            let min = bounds_in_lod_blocks.position;
            let max =
                bounds_in_lod_blocks.position + bounds_in_lod_blocks.size - Vector3i::splat(1);
            let fixtures_for_lod = [
                (
                    TransitionFace::NegativeX,
                    Vector3i::new(min.x, 0, 0),
                    Box3i::new(Vector3i::new(min.x, -1, -1), Vector3i::new(2, 3, 3)),
                ),
                (
                    TransitionFace::PositiveX,
                    Vector3i::new(max.x, 0, 0),
                    Box3i::new(Vector3i::new(max.x - 1, -1, -1), Vector3i::new(2, 3, 3)),
                ),
                (
                    TransitionFace::NegativeY,
                    Vector3i::new(0, min.y, 0),
                    Box3i::new(Vector3i::new(-1, min.y, -1), Vector3i::new(3, 2, 3)),
                ),
                (
                    TransitionFace::PositiveY,
                    Vector3i::new(0, max.y, 0),
                    Box3i::new(Vector3i::new(-1, max.y - 1, -1), Vector3i::new(3, 2, 3)),
                ),
                (
                    TransitionFace::NegativeZ,
                    Vector3i::new(0, 0, min.z),
                    Box3i::new(Vector3i::new(-1, -1, min.z), Vector3i::new(3, 3, 2)),
                ),
                (
                    TransitionFace::PositiveZ,
                    Vector3i::new(0, 0, max.z),
                    Box3i::new(Vector3i::new(-1, -1, max.z - 1), Vector3i::new(3, 3, 2)),
                ),
            ];
            for (face, position, expected_clipped_halo) in fixtures_for_lod {
                fixtures.push(FaceFixture {
                    face,
                    location: MeshBlockLocation::new(position, lod),
                    expected_clipped_halo,
                    bounds_in_lod_blocks,
                });
            }
        }

        fixtures
    }

    fn three_lod_boxes(viewer_position_voxels: Vector3i) -> super::LodClipboxes {
        compute_lod_clipboxes(
            viewer_position_voxels,
            Vector3i::splat(96),
            Box3i::new(Vector3i::splat(-512), Vector3i::splat(1024)),
            LodClipboxSettings::test_three_lods(),
        )
        .unwrap()
    }

    #[test]
    fn lod_bounds_scale_stride_before_clipping() {
        let bounds = Box3i::new(Vector3i::new(-128, -64, -32), Vector3i::new(256, 128, 64));
        assert_eq!(
            bounds_in_lod_blocks(bounds, 16, 2).unwrap(),
            Box3i::new(Vector3i::new(-2, -1, -1), Vector3i::new(4, 2, 2))
        );
    }

    #[test]
    fn negative_crossing_uses_euclidean_clipbox_coordinates() {
        let boxes = compute_lod_clipboxes(
            Vector3i::new(-1, 0, 0),
            Vector3i::splat(96),
            Box3i::new(Vector3i::splat(-512), Vector3i::splat(1024)),
            LodClipboxSettings::test_three_lods(),
        )
        .unwrap();
        assert!(boxes.mesh_load[0].contains_point(Vector3i::new(-1, 0, 0)));
        assert_eq!(boxes.mesh_load[0].position.x, -8);
        assert_eq!(boxes.mesh_load[0].size.x, 14);
    }

    #[test]
    fn empty_and_negative_bounds_stay_empty_when_downscaled() {
        for size in [0, -1] {
            assert_eq!(
                bounds_in_lod_blocks(
                    Box3i::new(Vector3i::splat(31), Vector3i::splat(size)),
                    16,
                    0,
                )
                .unwrap(),
                Box3i::default()
            );
        }
    }

    #[test]
    fn unaligned_bounds_are_rejected_by_compute_and_settings_validation() {
        let bounds = Box3i::new(Vector3i::zero(), Vector3i::splat(48));
        assert_eq!(
            compute_lod_clipboxes(
                Vector3i::zero(),
                Vector3i::splat(96),
                bounds,
                LodClipboxSettings::test_three_lods(),
            ),
            Err(LodMathError::UnalignedBounds { stride: 64 })
        );

        let mut data = VoxelData::new();
        data.set_bounds(bounds);
        assert_eq!(
            LodClipboxSettings::test_three_lods().validate_for(&data),
            Err(LodMathError::UnalignedBounds { stride: 64 })
        );
    }

    #[test]
    fn unaligned_empty_bounds_are_valid_and_produce_empty_clipboxes() {
        for size in [0, -1] {
            let bounds = Box3i::new(Vector3i::splat(1), Vector3i::splat(size));
            let boxes = compute_lod_clipboxes(
                Vector3i::zero(),
                Vector3i::splat(96),
                bounds,
                LodClipboxSettings::test_three_lods(),
            )
            .unwrap();
            assert!(boxes.mesh_load.iter().all(Box3i::is_empty));
            assert!(boxes.mesh_retain.iter().all(Box3i::is_empty));
            assert!(boxes.data_load.iter().all(Box3i::is_empty));
            assert!(boxes.data_retain.iter().all(Box3i::is_empty));

            let mut data = VoxelData::new();
            data.set_bounds(bounds);
            assert_eq!(
                LodClipboxSettings::test_three_lods().validate_for(&data),
                Ok(())
            );
        }
    }

    #[test]
    fn aligned_negative_boundary_preserves_clamped_load_and_retain_nesting() {
        let bounds = Box3i::new(Vector3i::splat(-512), Vector3i::splat(1024));
        let mut settings = LodClipboxSettings::test_three_lods();
        settings.unload_hysteresis_blocks = 1;
        let boxes =
            compute_lod_clipboxes(Vector3i::splat(-512), Vector3i::splat(96), bounds, settings)
                .unwrap();

        for lod in 0..2 {
            assert_eq!(boxes.mesh_load[lod].position.x.rem_euclid(2), 0);
            assert_eq!(boxes.mesh_load[lod].position.y.rem_euclid(2), 0);
            assert_eq!(boxes.mesh_load[lod].position.z.rem_euclid(2), 0);
            assert_eq!(boxes.mesh_load[lod].size.x.rem_euclid(2), 0);
            assert_eq!(boxes.mesh_load[lod].size.y.rem_euclid(2), 0);
            assert_eq!(boxes.mesh_load[lod].size.z.rem_euclid(2), 0);

            let parent_bounds = bounds_in_lod_blocks(bounds, 16, (lod + 1) as u8).unwrap();
            let clamped_load_minimum = boxes.mesh_load[lod]
                .downscaled(2)
                .padded(1)
                .clipped(parent_bounds);
            assert!(boxes.mesh_load[lod + 1].contains_box(clamped_load_minimum));

            let clamped_retain_minimum = boxes.mesh_retain[lod]
                .downscaled(2)
                .padded(1)
                .clipped(parent_bounds);
            assert!(boxes.mesh_retain[lod + 1].contains_box(clamped_retain_minimum));
        }
    }

    #[test]
    fn nonmultiple_distances_match_pinned_upstream_chunk_conversion() {
        let mut settings = LodClipboxSettings::test_three_lods();
        settings.lod0_distance_voxels = 17;
        let boxes = compute_lod_clipboxes(
            Vector3i::zero(),
            Vector3i::splat(96),
            Box3i::new(Vector3i::splat(-512), Vector3i::splat(1024)),
            settings,
        )
        .unwrap();
        assert_eq!(
            boxes.mesh_load[0],
            Box3i::new(Vector3i::splat(-2), Vector3i::splat(4))
        );
    }

    #[test]
    fn every_non_root_mesh_box_is_even_and_parent_contains_children() {
        let boxes = three_lod_boxes(Vector3i::new(37, -19, 65));
        for lod in 0..2 {
            assert_eq!(boxes.mesh_load[lod].position.x.rem_euclid(2), 0);
            assert_eq!(boxes.mesh_load[lod].position.y.rem_euclid(2), 0);
            assert_eq!(boxes.mesh_load[lod].position.z.rem_euclid(2), 0);
            assert_eq!(boxes.mesh_load[lod].size.x.rem_euclid(2), 0);
            assert_eq!(boxes.mesh_load[lod].size.y.rem_euclid(2), 0);
            assert_eq!(boxes.mesh_load[lod].size.z.rem_euclid(2), 0);
            assert!(
                boxes.mesh_load[lod + 1].contains_box(boxes.mesh_load[lod].downscaled(2).padded(1))
            );
        }
    }

    #[test]
    fn retain_boxes_are_larger_without_changing_load_topology() {
        let boxes = three_lod_boxes(Vector3i::zero());
        for lod in 0..3 {
            assert!(boxes.mesh_retain[lod].contains_box(boxes.mesh_load[lod]));
            assert!(boxes.data_retain[lod].contains_box(boxes.data_load[lod]));
        }
    }

    #[test]
    fn unrepresentable_lod_math_returns_error_instead_of_overflowing() {
        assert_eq!(
            lod_block_stride(1 << 30, 2),
            Err(LodMathError::CoordinateOverflow)
        );
        assert!(compute_lod_clipboxes(
            Vector3i::splat(i32::MAX),
            Vector3i::splat(i32::MAX),
            Box3i::new(Vector3i::splat(i32::MIN), Vector3i::splat(i32::MAX)),
            LodClipboxSettings::test_three_lods(),
        )
        .is_err());
    }

    #[test]
    fn unrepresentable_exclusive_box_max_returns_error() {
        assert_eq!(
            bounds_in_lod_blocks(
                Box3i::new(Vector3i::splat(i32::MAX), Vector3i::splat(1)),
                1,
                0,
            ),
            Err(LodMathError::CoordinateOverflow)
        );
    }

    #[test]
    fn validation_accepts_lod_count_before_data_maps_are_resized() {
        assert_eq!(
            LodClipboxSettings::test_three_lods().validate_for(&VoxelData::new()),
            Ok(())
        );
    }

    #[test]
    fn phase2_rejects_data_size_different_from_voxel_data() {
        let mut data = VoxelData::new();
        data.set_lod_count(3).unwrap();
        let mut settings = LodClipboxSettings::test_three_lods();
        settings.data_block_size = 32;
        assert_eq!(
            settings.validate_for(&data),
            Err(LodMathError::DataBlockSizeMismatch {
                settings: 32,
                voxel_data: 16,
            })
        );
    }

    #[test]
    fn phase2_rejects_mesh_to_data_factor_other_than_one() {
        let mut data = VoxelData::new();
        data.set_lod_count(3).unwrap();
        let mut settings = LodClipboxSettings::test_three_lods();
        settings.mesh_block_size = 32;
        assert_eq!(
            settings.validate_for(&data),
            Err(LodMathError::UnsupportedMeshToDataFactor { mesh: 32, data: 16 })
        );
    }

    #[test]
    fn invalid_clipbox_settings_return_typed_errors() {
        let mut settings = LodClipboxSettings::test_three_lods();
        settings.mesh_block_size = 0;
        assert_eq!(
            compute_lod_clipboxes(
                Vector3i::zero(),
                Vector3i::splat(96),
                Box3i::new(Vector3i::splat(-512), Vector3i::splat(1024)),
                settings,
            ),
            Err(LodMathError::InvalidBlockSize)
        );

        let mut settings = LodClipboxSettings::test_three_lods();
        settings.lod_count = 0;
        assert_eq!(
            compute_lod_clipboxes(
                Vector3i::zero(),
                Vector3i::splat(96),
                Box3i::new(Vector3i::splat(-512), Vector3i::splat(1024)),
                settings,
            ),
            Err(LodMathError::InvalidLodCount)
        );

        let mut settings = LodClipboxSettings::test_three_lods();
        settings.unload_hysteresis_blocks = -1;
        assert_eq!(
            compute_lod_clipboxes(
                Vector3i::zero(),
                Vector3i::splat(96),
                Box3i::new(Vector3i::splat(-512), Vector3i::splat(1024)),
                settings,
            ),
            Err(LodMathError::NegativeDistance)
        );
    }

    #[test]
    fn clipped_meshing_data_box_handles_all_six_negative_bounds_faces() {
        let bounds = Box3i::new(
            Vector3i::new(-256, -192, -128),
            Vector3i::new(512, 384, 256),
        );
        for fixture in all_six_face_locations(bounds, 16, [0, 1, 2]) {
            let actual = clipped_meshing_data_box(fixture.location, 16, 1, bounds).unwrap();
            assert_eq!(
                actual, fixture.expected_clipped_halo,
                "lod={} face={:?}",
                fixture.location.lod_index, fixture.face
            );
            assert!(fixture.bounds_in_lod_blocks.contains_box(actual));
        }
    }

    #[test]
    fn clipped_meshing_data_box_is_empty_when_mesh_block_is_outside_bounds() {
        let bounds = Box3i::new(Vector3i::zero(), Vector3i::splat(64));
        assert_eq!(
            clipped_meshing_data_box(
                MeshBlockLocation::new(Vector3i::new(-1, 0, 0), 0),
                16,
                1,
                bounds,
            )
            .unwrap(),
            Box3i::default()
        );
    }
}
