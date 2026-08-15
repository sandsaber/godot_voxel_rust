//! Threaded mesh block task — the engine-agnostic algorithm core.
//!
//! Ported from `meshers/mesh_block_task.{h,cpp}` minus Godot bindings
//! (`Ref<Mesh>`, `ArrayMesh`, GPU/detail-texture paths, `VoxelEngine`
//! callback dispatch). Implements the same pipeline the C++ terrain runs on
//! worker threads:
//!
//! 1. [`gather_voxels_cpu`] — gather the central data block plus its 3×3×3
//!    neighbours into a padded `VoxelBuffer`. Missing neighbours are filled
//!    by the installed generator (the same contract as C++
//!    `copy_block_and_neighbors` with `out_boxes_to_generate = nullptr`).
//! 2. [`MeshBlockTask::run`] — calls the configured [`VoxelMesher`] against
//!    the gathered voxels and stores the [`MesherOutput`].
//!
//! The current port assumes `mesh_block_size_factor == 1` (one mesh block
//! covers exactly one data block), which matches `VoxelTerrain`. The
//! general multi-block case used by `VoxelLodTerrain` is a follow-up.

use crate::engine::MeshingDependency;
use crate::generators::base::{VoxelGenerator, VoxelQueryData};
use crate::math::{Box3i, Vector3i};
use crate::meshers::{MeshArraysPool, MesherInput, MesherOutput, VoxelMesher};
use crate::storage::{SharedVoxelData, VoxelBuffer, VoxelData};
use crate::tasks::{
    RequestCancellation, TaskPriority, TaskRequestTag, TaskRunStatus, ThreadedTask,
    ThreadedTaskContext,
};
use crate::terrain::lod_clipbox::{bounds_in_lod_blocks, lod_block_stride, LodMathError};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeshBuildFeatures {
    pub visuals: bool,
    pub collisions: bool,
    pub variable_lod: bool,
}

impl MeshBuildFeatures {
    pub const fn contains(self, required: Self) -> bool {
        (!required.visuals || self.visuals)
            && (!required.collisions || self.collisions)
            && (!required.variable_lod || self.variable_lod)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadState {
    NotBuilt,
    Empty,
    NonEmpty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshBlockLocation {
    pub position_in_blocks: Vector3i,
    pub lod_index: u8,
}

impl MeshBlockLocation {
    pub const fn new(position_in_blocks: Vector3i, lod_index: u8) -> Self {
        Self {
            position_in_blocks,
            lod_index,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshBlockKey {
    pub location: MeshBlockLocation,
    pub revision: u64,
}

/// Directly submitted mesh output before terrain admission. The payload and
/// its originating pool remain together so a rejected caller-owned output can
/// return pooled Transvoxel arrays without reconstruction.
pub struct BlockMeshOutput {
    key: MeshBlockKey,
    features: MeshBuildFeatures,
    output: Option<Box<MesherOutput>>,
    pool: Arc<MeshArraysPool>,
    dropped: bool,
}

impl BlockMeshOutput {
    pub fn new(
        key: MeshBlockKey,
        features: MeshBuildFeatures,
        output: MesherOutput,
        pool: Arc<MeshArraysPool>,
        dropped: bool,
    ) -> Self {
        Self {
            key,
            features,
            output: Some(Box::new(output)),
            pool,
            dropped,
        }
    }

    pub const fn key(&self) -> MeshBlockKey {
        self.key
    }

    pub const fn features(&self) -> MeshBuildFeatures {
        self.features
    }

    pub const fn dropped(&self) -> bool {
        self.dropped
    }

    pub fn output(&self) -> &MesherOutput {
        self.output
            .as_ref()
            .expect("block mesh output payload exists until normalization")
    }

    pub fn pool(&self) -> &Arc<MeshArraysPool> {
        &self.pool
    }

    pub(crate) fn into_upload(mut self) -> MeshBlockTaskOutput {
        let output = *self
            .output
            .take()
            .expect("block mesh output payload exists until normalization");
        MeshBlockTaskOutput {
            upload: Arc::new(MeshUploadSnapshot::new(
                self.key,
                self.features,
                output,
                self.pool.clone(),
            )),
            dropped: self.dropped,
            request_tag: None,
        }
    }
}

impl std::fmt::Debug for BlockMeshOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockMeshOutput")
            .field("key", &self.key)
            .field("features", &self.features)
            .field("dropped", &self.dropped)
            .finish_non_exhaustive()
    }
}

impl Drop for BlockMeshOutput {
    fn drop(&mut self) {
        if let Some(mut output) = self.output.take() {
            if let Some(arrays) = output.take_first_transvoxel_arrays() {
                self.pool.release(arrays);
            }
        }
    }
}

pub struct MeshUploadSnapshot {
    key: MeshBlockKey,
    features: MeshBuildFeatures,
    visual_state: PayloadState,
    collision_state: PayloadState,
    output: MesherOutput,
    pool: Arc<MeshArraysPool>,
}

impl MeshUploadSnapshot {
    fn new(
        key: MeshBlockKey,
        features: MeshBuildFeatures,
        output: MesherOutput,
        pool: Arc<MeshArraysPool>,
    ) -> Self {
        let visual_state = classify_payload(features.visuals, !output.is_empty());
        let collision_nonempty = !output.collision_surface.positions.is_empty()
            || !output.collision_surface.indices.is_empty()
            || output.collision_surface.submesh_vertex_end > 0
            || output.collision_surface.submesh_index_end > 0;
        let collision_state = classify_payload(features.collisions, collision_nonempty);
        Self {
            key,
            features,
            visual_state,
            collision_state,
            output,
            pool,
        }
    }

    pub const fn key(&self) -> MeshBlockKey {
        self.key
    }

    pub const fn features(&self) -> MeshBuildFeatures {
        self.features
    }

    pub const fn visual_state(&self) -> PayloadState {
        self.visual_state
    }

    pub const fn collision_state(&self) -> PayloadState {
        self.collision_state
    }

    pub const fn output(&self) -> &MesherOutput {
        &self.output
    }

    pub(crate) const fn has_built_payload(&self) -> bool {
        matches!(self.visual_state, PayloadState::NonEmpty)
            || matches!(self.collision_state, PayloadState::NonEmpty)
    }
}

impl std::fmt::Debug for MeshUploadSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshUploadSnapshot")
            .field("key", &self.key)
            .field("features", &self.features)
            .field("visual_state", &self.visual_state)
            .field("collision_state", &self.collision_state)
            .finish_non_exhaustive()
    }
}

impl Drop for MeshUploadSnapshot {
    fn drop(&mut self) {
        if let Some(arrays) = self.output.take_first_transvoxel_arrays() {
            self.pool.release(arrays);
        }
    }
}

const fn classify_payload(built: bool, nonempty: bool) -> PayloadState {
    if !built {
        PayloadState::NotBuilt
    } else if nonempty {
        PayloadState::NonEmpty
    } else {
        PayloadState::Empty
    }
}

pub struct MeshBlockTaskOutput {
    upload: Arc<MeshUploadSnapshot>,
    dropped: bool,
    request_tag: Option<TaskRequestTag>,
}

impl MeshBlockTaskOutput {
    pub fn upload(&self) -> &Arc<MeshUploadSnapshot> {
        &self.upload
    }

    pub const fn dropped(&self) -> bool {
        self.dropped
    }

    /// Exact physical request identity for runner-produced outputs. Direct
    /// compatibility uploads intentionally carry no tag and remain guarded by
    /// their mesh revision at the terrain admission boundary.
    pub const fn request_tag(&self) -> Option<TaskRequestTag> {
        self.request_tag
    }

    pub(crate) fn into_parts(self) -> (Arc<MeshUploadSnapshot>, bool) {
        (self.upload, self.dropped)
    }
}

impl std::fmt::Debug for MeshBlockTaskOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshBlockTaskOutput")
            .field("upload", &self.upload)
            .field("dropped", &self.dropped)
            .field("request_tag", &self.request_tag)
            .finish()
    }
}

/// Configuration for [`MeshBlockTask::new`].
#[derive(Clone)]
pub struct MeshBlockTaskParams {
    pub key: MeshBlockKey,
    pub data: Arc<SharedVoxelData>,
    pub meshing_dependency: Arc<MeshingDependency>,
    /// Hint that collision geometry is wanted (passed to the mesher).
    pub collision_hint: bool,
    /// Hint that the mesh will be used in a variable-LOD context.
    pub lod_hint: bool,
    /// Optional free-list pool for reusable `MeshArrays` buffers (audit §9.6-B3).
    /// When set, meshers acquire a cleared buffer instead of allocating fresh;
    /// the terrain core returns the previous block's arrays on re-mesh/unload.
    pub mesh_arrays_pool: Option<Arc<MeshArraysPool>>,
}

/// Threaded task: gathers voxels and runs a [`VoxelMesher`] against them.
///
/// Ported from C++ `MeshBlockTask`. Synchronous CPU-only path: no GPU
/// generation, no detail-texture baking, no `VoxelEngine` callback dispatch.
/// Callers drain [`MeshBlockTaskOutput`] via [`MeshBlockTask::take_output`].
pub struct MeshBlockTask {
    key: MeshBlockKey,
    data: Arc<SharedVoxelData>,
    meshing_dependency: Arc<MeshingDependency>,
    collision_hint: bool,
    lod_hint: bool,
    mesh_arrays_pool: Option<Arc<MeshArraysPool>>,
    request_tag: Option<TaskRequestTag>,
    request_cancellation: Option<Arc<RequestCancellation>>,
    has_run: bool,
    output: Option<MeshBlockTaskOutput>,
}

impl MeshBlockTask {
    pub fn new(params: MeshBlockTaskParams) -> Self {
        Self {
            key: params.key,
            data: params.data,
            meshing_dependency: params.meshing_dependency,
            collision_hint: params.collision_hint,
            lod_hint: params.lod_hint,
            mesh_arrays_pool: params.mesh_arrays_pool,
            request_tag: None,
            request_cancellation: None,
            has_run: false,
            output: None,
        }
    }

    pub const fn position_in_blocks(&self) -> Vector3i {
        self.key.location.position_in_blocks
    }

    pub const fn lod_index(&self) -> u8 {
        self.key.location.lod_index
    }

    pub const fn key(&self) -> MeshBlockKey {
        self.key
    }

    pub const fn has_run(&self) -> bool {
        self.has_run
    }

    /// Attaches the exact entry-owned physical request identity and shared
    /// cancellation state without changing the compatibility constructor.
    pub fn with_request_control(
        mut self,
        tag: TaskRequestTag,
        cancellation: Arc<RequestCancellation>,
    ) -> Self {
        self.request_tag = Some(tag);
        self.request_cancellation = Some(cancellation);
        self
    }

    pub const fn request_tag(&self) -> Option<TaskRequestTag> {
        self.request_tag
    }

    fn request_is_cancelled(&self) -> bool {
        self.request_cancellation
            .as_deref()
            .is_some_and(RequestCancellation::is_cancelled)
    }

    pub fn take_output(&mut self) -> Option<MeshBlockTaskOutput> {
        self.output.take()
    }

    pub(crate) fn output_ref(&self) -> Option<&MeshBlockTaskOutput> {
        self.output.as_ref()
    }

    /// Run the gather+mesh pipeline synchronously. Equivalent to the C++
    /// `MeshBlockTask::run` CPU branch (gather_voxels_cpu + build_mesh).
    pub fn run_meshing(&mut self) {
        self.output = None;

        if self.request_is_cancelled() || !self.meshing_dependency.is_valid() {
            // The terrain swapped the mesher/generator mid-flight; emit a
            // dropped output so the caller knows to requeue.
            self.complete_dropped();
            return;
        }

        let mesher_handle = self.meshing_dependency.mesher();
        let mesher: &dyn VoxelMesher = mesher_handle.as_ref();
        let Ok(min_padding) = i32::try_from(mesher.minimum_padding()) else {
            self.complete_dropped();
            return;
        };
        let Ok(max_padding) = i32::try_from(mesher.maximum_padding()) else {
            self.complete_dropped();
            return;
        };
        let channels_mask = mesher.used_channels_mask();

        let generator_handle = self.meshing_dependency.generator();

        let Ok(data_block_size) = i32::try_from(self.data.block_size()) else {
            self.complete_dropped();
            return;
        };
        if usize::from(self.lod_index()) >= self.data.lod_count() {
            self.complete_dropped();
            return;
        }
        let Ok(read_box) = checked_clipped_read_box(
            self.position_in_blocks(),
            data_block_size,
            self.lod_index(),
            self.data.bounds(),
        ) else {
            self.complete_dropped();
            return;
        };
        if read_box.is_empty() {
            self.complete_dropped();
            return;
        }

        let mut voxels = VoxelBuffer::with_size(Vector3i::zero());
        let gather_plan = {
            let _read_region = self.data.read_region(self.lod_index() as usize, read_box);
            let gathered = gather_voxels_cpu_shared_snapshot(
                &mut voxels,
                min_padding,
                max_padding,
                channels_mask,
                generator_handle.is_some(),
                &self.data,
                self.lod_index(),
                self.position_in_blocks(),
            );
            match gathered {
                Ok(gathered) => gathered,
                Err(_) => {
                    drop(_read_region);
                    self.complete_dropped();
                    return;
                }
            }
        };
        if self.request_is_cancelled() {
            self.complete_dropped();
            return;
        }
        if let Some(generator) = generator_handle.as_deref() {
            generate_missing_voxel_regions(&mut voxels, generator, &gather_plan, self.lod_index());
        }
        if self.request_is_cancelled() {
            self.complete_dropped();
            return;
        }

        // Build the mesh. The padded buffer is what the mesher sees; the
        // origin reported to the mesher is the world-space corner of the
        // *unpadded* mesh block (matching C++ build_mesh).
        let mesh_block_size = voxels.size() - Vector3i::splat(min_padding + max_padding);
        debug_assert_eq!(mesh_block_size, Vector3i::splat(data_block_size));
        let Ok(block_world_origin) = checked_block_world_origin(
            self.position_in_blocks(),
            data_block_size,
            self.lod_index(),
        ) else {
            self.complete_dropped();
            return;
        };

        let mut surfaces = MesherOutput::default();
        let input = MesherInput {
            voxels: &voxels,
            generator: generator_handle.as_deref(),
            origin_in_voxels: block_world_origin,
            lod_index: self.lod_index(),
            collision_hint: self.collision_hint,
            lod_hint: self.lod_hint,
            mesh_arrays_pool: self.mesh_arrays_pool.as_deref(),
        };
        mesher.build(&mut surfaces, &input);

        // MESH-1 parity: re-check dependency validity AFTER build. If the
        // mesher/generator was swapped mid-flight (between the initial check
        // and now), drop the output so the terrain re-queues with the new
        // dependency.
        let dropped = self.request_is_cancelled() || !self.meshing_dependency.is_valid();

        self.has_run = true;
        self.output = Some(self.snapshot_output(surfaces, dropped));
    }

    fn complete_dropped(&mut self) {
        self.has_run = true;
        self.output = Some(self.snapshot_output(MesherOutput::default(), true));
    }

    fn snapshot_output(&self, output: MesherOutput, dropped: bool) -> MeshBlockTaskOutput {
        let features = MeshBuildFeatures {
            visuals: true,
            collisions: self.collision_hint,
            variable_lod: self.lod_hint,
        };
        let pool = self
            .mesh_arrays_pool
            .clone()
            .unwrap_or_else(|| Arc::new(MeshArraysPool::new()));
        MeshBlockTaskOutput {
            upload: Arc::new(MeshUploadSnapshot::new(self.key, features, output, pool)),
            dropped,
            request_tag: self.request_tag,
        }
    }
}

impl ThreadedTask for MeshBlockTask {
    fn run(&mut self, _ctx: ThreadedTaskContext) -> TaskRunStatus {
        self.run_meshing();
        TaskRunStatus::Complete {
            follow_up_tasks: Vec::new(),
        }
    }

    fn priority(&mut self) -> TaskPriority {
        // TASK-1 parity: use the same band-2 base as C++
        // (TASK_PRIORITY_MESH_BAND2 = 10) instead of the previous minimum
        // (0,0,0,0). This lets load and mesh tasks compete on equal footing
        // in the runner's priority queue rather than mesh being starved.
        TaskPriority::new(
            0,
            0,
            crate::constants::voxel_constants::TASK_PRIORITY_MESH_BAND2,
            0,
        )
    }

    fn is_cancelled(&mut self) -> bool {
        self.request_is_cancelled() || !self.meshing_dependency.is_valid()
    }

    fn debug_name(&self) -> &'static str {
        "MeshBlockTask"
    }
}

/// Gathers a 3×3×3 neighbourhood of data blocks into a padded `dst` buffer.
///
/// Ported from C++ `copy_block_and_neighbors` for `mesh_block_size_factor == 1`.
/// `dst` is configured to `(block_size + min_padding + max_padding)³` with the
/// caller's [`VoxelFormat`], matching C++ which configures the buffer before
/// copying/generating the neighbour regions. The function:
///
/// 1. Copies each neighbour's channel data into the matching sub-region of
///    `dst` (skipping empty/missing blocks).
/// 2. For missing blocks inside the volume bounds, runs `generator` to fill
///    the corresponding region of `dst` directly.
///
/// Compatibility wrapper returning the world-space origin of the *padded*
/// buffer. New checked callers should use [`try_gather_voxels_cpu`]. If legacy
/// input cannot be represented, this wrapper leaves generator work unstarted
/// and returns the zero origin instead of wrapping or panicking.
#[allow(clippy::too_many_arguments)]
pub fn gather_voxels_cpu(
    dst: &mut VoxelBuffer,
    min_padding: i32,
    max_padding: i32,
    channels_mask: u32,
    generator: Option<&dyn VoxelGenerator>,
    voxel_data: &VoxelData,
    lod_index: u8,
    mesh_block_pos: Vector3i,
) -> Vector3i {
    try_gather_voxels_cpu(
        dst,
        min_padding,
        max_padding,
        channels_mask,
        generator,
        voxel_data,
        lod_index,
        mesh_block_pos,
    )
    .unwrap_or_default()
}

/// Checked gather variant. The returned padded-buffer origin is expressed in
/// world voxels, including the `2^lod` scale for both block position and
/// padding.
///
/// Returns [`LodMathError`] before generator work starts when padding, LOD
/// scaling, a neighbour origin, or finite bounds are unrepresentable.
#[allow(clippy::too_many_arguments)]
pub fn try_gather_voxels_cpu(
    dst: &mut VoxelBuffer,
    min_padding: i32,
    max_padding: i32,
    channels_mask: u32,
    generator: Option<&dyn VoxelGenerator>,
    voxel_data: &VoxelData,
    lod_index: u8,
    mesh_block_pos: Vector3i,
) -> Result<Vector3i, LodMathError> {
    let data_block_size =
        i32::try_from(voxel_data.block_size()).map_err(|_| LodMathError::InvalidBlockSize)?;
    let central_origin = checked_block_world_origin(mesh_block_pos, data_block_size, lod_index)?;
    let lod_scale = i64::from(lod_block_stride(1, lod_index)?);
    let negative_padding = min_padding
        .checked_neg()
        .ok_or(LodMathError::CoordinateOverflow)?;
    let padded_origin =
        checked_add_scaled_vector(central_origin, Vector3i::splat(negative_padding), lod_scale)?;
    let gather_plan = gather_voxels_cpu_snapshot(
        dst,
        min_padding,
        max_padding,
        channels_mask,
        generator.is_some(),
        voxel_data,
        lod_index,
        mesh_block_pos,
    )?;
    if let Some(generator) = generator {
        generate_missing_voxel_regions(dst, generator, &gather_plan, lod_index);
    }
    Ok(padded_origin)
}

#[derive(Debug, Clone, Copy)]
struct MissingVoxelRegion {
    dst_offset: Vector3i,
    origin_in_voxels: Vector3i,
}

#[derive(Debug)]
struct GatherVoxelPlan {
    format: crate::storage::VoxelFormat,
    data_block_size: i32,
    channels: Vec<usize>,
    missing_regions: Vec<MissingVoxelRegion>,
}

fn checked_vector3i(values: [i64; 3]) -> Result<Vector3i, LodMathError> {
    Ok(Vector3i::new(
        i32::try_from(values[0]).map_err(|_| LodMathError::CoordinateOverflow)?,
        i32::try_from(values[1]).map_err(|_| LodMathError::CoordinateOverflow)?,
        i32::try_from(values[2]).map_err(|_| LodMathError::CoordinateOverflow)?,
    ))
}

fn checked_scaled_vector(position: Vector3i, scale: i64) -> Result<Vector3i, LodMathError> {
    let values = [
        i64::from(position.x)
            .checked_mul(scale)
            .ok_or(LodMathError::CoordinateOverflow)?,
        i64::from(position.y)
            .checked_mul(scale)
            .ok_or(LodMathError::CoordinateOverflow)?,
        i64::from(position.z)
            .checked_mul(scale)
            .ok_or(LodMathError::CoordinateOverflow)?,
    ];
    checked_vector3i(values)
}

fn checked_add_scaled_vector(
    origin: Vector3i,
    offset: Vector3i,
    scale: i64,
) -> Result<Vector3i, LodMathError> {
    let values = [
        i64::from(origin.x)
            .checked_add(
                i64::from(offset.x)
                    .checked_mul(scale)
                    .ok_or(LodMathError::CoordinateOverflow)?,
            )
            .ok_or(LodMathError::CoordinateOverflow)?,
        i64::from(origin.y)
            .checked_add(
                i64::from(offset.y)
                    .checked_mul(scale)
                    .ok_or(LodMathError::CoordinateOverflow)?,
            )
            .ok_or(LodMathError::CoordinateOverflow)?,
        i64::from(origin.z)
            .checked_add(
                i64::from(offset.z)
                    .checked_mul(scale)
                    .ok_or(LodMathError::CoordinateOverflow)?,
            )
            .ok_or(LodMathError::CoordinateOverflow)?,
    ];
    checked_vector3i(values)
}

fn checked_offset_position(position: Vector3i, offset: Vector3i) -> Option<Vector3i> {
    Some(Vector3i::new(
        position.x.checked_add(offset.x)?,
        position.y.checked_add(offset.y)?,
        position.z.checked_add(offset.z)?,
    ))
}

fn checked_block_world_origin(
    block_position: Vector3i,
    block_size: i32,
    lod_index: u8,
) -> Result<Vector3i, LodMathError> {
    let stride = i64::from(lod_block_stride(block_size, lod_index)?);
    checked_scaled_vector(block_position, stride)
}

fn checked_box_from_i64(min: [i64; 3], max: [i64; 3]) -> Result<Box3i, LodMathError> {
    if max
        .iter()
        .zip(min)
        .any(|(&max_axis, min_axis)| max_axis <= min_axis)
    {
        return Ok(Box3i::default());
    }
    let position = checked_vector3i(min)?;
    let size = checked_vector3i([
        max[0]
            .checked_sub(min[0])
            .ok_or(LodMathError::CoordinateOverflow)?,
        max[1]
            .checked_sub(min[1])
            .ok_or(LodMathError::CoordinateOverflow)?,
        max[2]
            .checked_sub(min[2])
            .ok_or(LodMathError::CoordinateOverflow)?,
    ])?;
    Ok(Box3i::new(position, size))
}

fn checked_clipped_read_box(
    block_position: Vector3i,
    block_size: i32,
    lod_index: u8,
    bounds: Box3i,
) -> Result<Box3i, LodMathError> {
    let stride = i64::from(lod_block_stride(block_size, lod_index)?);
    if !bounds_in_lod_blocks(bounds, block_size, lod_index)?.contains_point(block_position) {
        return Ok(Box3i::default());
    }
    let block = [
        i64::from(block_position.x),
        i64::from(block_position.y),
        i64::from(block_position.z),
    ];
    let bounds_min = [
        i64::from(bounds.position.x),
        i64::from(bounds.position.y),
        i64::from(bounds.position.z),
    ];
    let bounds_size = [
        i64::from(bounds.size.x),
        i64::from(bounds.size.y),
        i64::from(bounds.size.z),
    ];
    let mut min = [0_i64; 3];
    let mut max = [0_i64; 3];
    for axis in 0..3 {
        let bounds_max = bounds_min[axis]
            .checked_add(bounds_size[axis])
            .ok_or(LodMathError::CoordinateOverflow)?;
        i32::try_from(bounds_max).map_err(|_| LodMathError::CoordinateOverflow)?;
        min[axis] = block[axis]
            .checked_sub(1)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(LodMathError::CoordinateOverflow)?
            .max(bounds_min[axis]);
        max[axis] = block[axis]
            .checked_add(2)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(LodMathError::CoordinateOverflow)?
            .min(bounds_max);
    }
    checked_box_from_i64(min, max)
}

#[allow(clippy::too_many_arguments)]
fn gather_voxels_cpu_snapshot(
    dst: &mut VoxelBuffer,
    min_padding: i32,
    max_padding: i32,
    channels_mask: u32,
    queue_missing_regions: bool,
    voxel_data: &VoxelData,
    lod_index: u8,
    mesh_block_pos: Vector3i,
) -> Result<GatherVoxelPlan, LodMathError> {
    if min_padding < 0 || max_padding < 0 {
        return Err(LodMathError::NegativeDistance);
    }
    if usize::from(lod_index) >= voxel_data.lod_count() {
        return Err(LodMathError::InvalidLodCount);
    }
    let data_block_size =
        i32::try_from(voxel_data.block_size()).map_err(|_| LodMathError::InvalidBlockSize)?;
    let mesh_block_size = data_block_size; // factor == 1
    let padded_size = mesh_block_size
        .checked_add(min_padding)
        .and_then(|size| size.checked_add(max_padding))
        .ok_or(LodMathError::CoordinateOverflow)?;
    let format = voxel_data.format();
    let bounds_in_blocks = bounds_in_lod_blocks(voxel_data.bounds(), data_block_size, lod_index)?;

    // (Re)create `dst` at the padded size and configure channels. The C++
    // path calls `dst.create(size, &format)`; our caller already configured
    // the format, so we just resize.
    if dst.size() != Vector3i::splat(padded_size) {
        *dst = VoxelBuffer::with_size(Vector3i::splat(padded_size));
    }
    format.configure_buffer(dst);

    let channels: Vec<usize> = (0..8u32)
        .filter(|ci| (channels_mask & (1u32 << ci)) != 0)
        .map(|ci| ci as usize)
        .collect();
    let mut missing_regions = Vec::new();

    // Padded buffer's world origin (corner of the halo, not the block).
    // Each neighbour occupies a `data_block_size`³ slab of `dst`. Iterate
    // ZXY (matching C++) and compute the source offset into `dst`.
    for dz in -1..=1 {
        for dx in -1..=1 {
            for dy in -1..=1 {
                let Some(neighbour_block_pos) =
                    checked_offset_position(mesh_block_pos, Vector3i::new(dx, dy, dz))
                else {
                    continue;
                };
                if !bounds_in_blocks.contains_point(neighbour_block_pos) {
                    continue;
                }
                let dst_offset =
                    Vector3i::new(dx, dy, dz) * data_block_size + Vector3i::splat(min_padding);

                let neighbour_present = voxel_data
                    .get_block(neighbour_block_pos, lod_index as usize)
                    .is_some_and(|block| block.has_voxels());

                if neighbour_present {
                    let src = voxel_data
                        .get_block(neighbour_block_pos, lod_index as usize)
                        .unwrap()
                        .voxels();
                    for &channel_index in &channels {
                        dst.copy_channel_from_area(
                            src,
                            Vector3i::zero(),
                            src.size(),
                            dst_offset,
                            channel_index,
                        );
                    }
                } else if queue_missing_regions {
                    // Missing neighbour inside bounds: queue the generator
                    // work so callers holding VoxelData's lock can drop it
                    // before running the heavy generator.
                    let neighbour_origin = checked_block_world_origin(
                        neighbour_block_pos,
                        data_block_size,
                        lod_index,
                    )?;
                    missing_regions.push(MissingVoxelRegion {
                        dst_offset,
                        origin_in_voxels: neighbour_origin,
                    });
                }
                // Else: no generator and missing block — `dst` keeps the
                // format default for that region (matches C++ behaviour
                // when no generator is installed).
            }
        }
    }

    Ok(GatherVoxelPlan {
        format,
        data_block_size,
        channels,
        missing_regions,
    })
}

#[allow(clippy::too_many_arguments)]
fn gather_voxels_cpu_shared_snapshot(
    dst: &mut VoxelBuffer,
    min_padding: i32,
    max_padding: i32,
    channels_mask: u32,
    queue_missing_regions: bool,
    voxel_data: &SharedVoxelData,
    lod_index: u8,
    mesh_block_pos: Vector3i,
) -> Result<GatherVoxelPlan, LodMathError> {
    if min_padding < 0 || max_padding < 0 {
        return Err(LodMathError::NegativeDistance);
    }
    if usize::from(lod_index) >= voxel_data.lod_count() {
        return Err(LodMathError::InvalidLodCount);
    }
    let data_block_size =
        i32::try_from(voxel_data.block_size()).map_err(|_| LodMathError::InvalidBlockSize)?;
    let mesh_block_size = data_block_size; // factor == 1
    let padded_size = mesh_block_size
        .checked_add(min_padding)
        .and_then(|size| size.checked_add(max_padding))
        .ok_or(LodMathError::CoordinateOverflow)?;
    let format = voxel_data.format();
    let bounds_in_blocks = bounds_in_lod_blocks(voxel_data.bounds(), data_block_size, lod_index)?;

    if dst.size() != Vector3i::splat(padded_size) {
        *dst = VoxelBuffer::with_size(Vector3i::splat(padded_size));
    }
    format.configure_buffer(dst);

    let channels: Vec<usize> = (0..8u32)
        .filter(|ci| (channels_mask & (1u32 << ci)) != 0)
        .map(|ci| ci as usize)
        .collect();
    let mut missing_regions = Vec::new();
    let mut resident_offsets = Vec::new();

    let central_world_origin =
        checked_block_world_origin(mesh_block_pos, data_block_size, lod_index)?;

    let mut halo_was_clipped = false;
    let mut visit_neighbour = |neighbour_block_pos: Vector3i,
                               dst_offset: Vector3i,
                               src: Option<&VoxelBuffer>|
     -> Result<(), LodMathError> {
        if let Some(src) = src {
            resident_offsets.push(dst_offset);
            for &channel_index in &channels {
                dst.copy_channel_from_area(
                    src,
                    Vector3i::zero(),
                    src.size(),
                    dst_offset,
                    channel_index,
                );
            }
        } else if queue_missing_regions {
            let neighbour_origin =
                checked_block_world_origin(neighbour_block_pos, data_block_size, lod_index)?;
            missing_regions.push(MissingVoxelRegion {
                dst_offset,
                origin_in_voxels: neighbour_origin,
            });
        }
        Ok(())
    };

    voxel_data.with_lod_map(lod_index as usize, |map| -> Result<(), LodMathError> {
        for dz in -1..=1 {
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let Some(neighbour_block_pos) =
                        checked_offset_position(mesh_block_pos, Vector3i::new(dx, dy, dz))
                    else {
                        halo_was_clipped = true;
                        continue;
                    };
                    if !bounds_in_blocks.contains_point(neighbour_block_pos) {
                        halo_was_clipped = true;
                        continue;
                    }
                    let dst_offset =
                        Vector3i::new(dx, dy, dz) * data_block_size + Vector3i::splat(min_padding);

                    let src = map
                        .get_block(neighbour_block_pos)
                        .filter(|block| block.has_voxels())
                        .map(|block| block.voxels());
                    visit_neighbour(neighbour_block_pos, dst_offset, src)?;
                }
            }
        }
        Ok(())
    })?;

    // GATHER-1 parity: clip missing regions to the actual missing area
    // (mesh_data_box minus resident blocks). C++ subtracts resident blocks
    // and clips to bounds, generating only the true remainder (~38.5× fewer
    // samples for a block with all 26 neighbours missing).
    if !halo_was_clipped && !missing_regions.is_empty() && !resident_offsets.is_empty() {
        let padded_box = Box3i::new(
            Vector3i::splat(min_padding),
            Vector3i::splat(mesh_block_size + max_padding),
        );
        let block_vec = Vector3i::splat(data_block_size);
        let mut boxes_to_generate: Vec<Box3i> = vec![padded_box];
        for &offset in &resident_offsets {
            let resident_box = Box3i::new(offset, block_vec).clipped(padded_box);
            if resident_box.is_empty() {
                continue;
            }
            let mut next = Vec::new();
            for b in &boxes_to_generate {
                next.extend(b.difference(resident_box));
            }
            boxes_to_generate = next;
        }
        // Convert clipped boxes back to MissingVoxelRegion entries.
        let lod_stride = i64::from(lod_block_stride(1, lod_index)?);
        let mut clipped_missing_regions = Vec::new();
        for box_to_generate in boxes_to_generate
            .into_iter()
            .filter(|box_to_generate| !box_to_generate.is_empty())
        {
            let offset = box_to_generate.position - Vector3i::splat(min_padding);
            let world_origin = checked_add_scaled_vector(central_world_origin, offset, lod_stride)?;
            clipped_missing_regions.push(MissingVoxelRegion {
                dst_offset: box_to_generate.position,
                origin_in_voxels: world_origin,
            });
        }
        missing_regions = clipped_missing_regions;
    }

    Ok(GatherVoxelPlan {
        format,
        data_block_size,
        channels,
        missing_regions,
    })
}

fn generate_missing_voxel_regions(
    dst: &mut VoxelBuffer,
    generator: &dyn VoxelGenerator,
    gather_plan: &GatherVoxelPlan,
    lod_index: u8,
) {
    if gather_plan.missing_regions.is_empty() {
        return;
    }
    // B3 (audit §9.6-B4): allocate one reusable block-sized scratch buffer
    // outside the 3×3×3 loop instead of up to 27 fresh `VoxelBuffer`s per gather.
    // `create` resets every channel to its uniform default; the generator then
    // writes into it and we copy the requested channels into the padded `dst`.
    let block_size_vec = Vector3i::splat(gather_plan.data_block_size);
    let mut scratch = VoxelBuffer::with_size(block_size_vec);
    gather_plan.format.configure_buffer(&mut scratch);
    for region in &gather_plan.missing_regions {
        // Reset to a clean buffer so leftover data from the previous neighbour
        // does not leak into channels the generator leaves at their default.
        scratch.create(block_size_vec);
        gather_plan.format.configure_buffer(&mut scratch);
        generator.generate_block(VoxelQueryData {
            buffer: &mut scratch,
            origin_in_voxels: region.origin_in_voxels,
            lod: lod_index as u32,
        });
        for &channel_index in &gather_plan.channels {
            dst.copy_channel_from_area(
                &scratch,
                Vector3i::zero(),
                scratch.size(),
                region.dst_offset,
                channel_index,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        try_gather_voxels_cpu, MeshBlockKey, MeshBlockLocation, MeshBlockTask, MeshBlockTaskParams,
    };
    use crate::engine::MeshingDependency;
    use crate::generators::base::{GenResult, VoxelGenerator, VoxelQueryData};
    use crate::math::{Box3i, Vector3i};
    use crate::meshers::{MesherInput, MesherOutput, Surface, SurfaceArrays, VoxelMesher};
    use crate::storage::{ChannelId, SharedVoxelData, VoxelBuffer, VoxelData, VoxelDataBlock};
    use crate::tasks::{RequestCancellation, TaskRequestTag, ThreadedTask};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex, Weak};
    use std::time::{Duration, Instant};

    /// A generator that writes a constant raw value into the SDF channel of
    /// every voxel it sees. Lets us verify the gather step fills missing
    /// neighbours with generator output.
    struct ConstantSdfGenerator {
        value: f32,
    }
    impl VoxelGenerator for ConstantSdfGenerator {
        fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
            input
                .buffer
                .clear_channel_f(ChannelId::Sdf.index(), self.value);
            GenResult::default()
        }
    }

    struct CountingGenerator {
        calls: Arc<AtomicUsize>,
    }

    impl VoxelGenerator for CountingGenerator {
        fn generate_block(&self, _input: VoxelQueryData<'_>) -> GenResult {
            self.calls.fetch_add(1, Ordering::SeqCst);
            GenResult::default()
        }
    }

    struct CancellingMesher {
        cancellation: Arc<RequestCancellation>,
        build_calls: Arc<AtomicUsize>,
    }

    impl VoxelMesher for CancellingMesher {
        fn build(&self, _output: &mut MesherOutput, _input: &MesherInput<'_>) {
            self.build_calls.fetch_add(1, Ordering::SeqCst);
            self.cancellation.cancel();
        }

        fn used_channels_mask(&self) -> u32 {
            0
        }
    }

    struct VoxelDataLockProbeGenerator {
        data: Weak<SharedVoxelData>,
    }

    impl VoxelGenerator for VoxelDataLockProbeGenerator {
        fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
            let data = self.data.upgrade().expect("voxel data still alive");
            let guard = data
                .try_lock()
                .expect("VoxelData lock must be released before generator calls");
            drop(guard);

            input.buffer.clear_channel_f(ChannelId::Sdf.index(), -0.25);
            GenResult::default()
        }
    }

    struct VoxelDataLockProbeMesher {
        data: Weak<SharedVoxelData>,
        build_calls: Arc<Mutex<usize>>,
    }

    impl VoxelMesher for VoxelDataLockProbeMesher {
        fn build(&self, _output: &mut MesherOutput, _input: &MesherInput<'_>) {
            let data = self.data.upgrade().expect("voxel data still alive");
            let guard = data
                .try_lock()
                .expect("VoxelData lock must be released before mesher calls");
            drop(guard);
            *self.build_calls.lock().unwrap() += 1;
        }

        fn used_channels_mask(&self) -> u32 {
            1 << ChannelId::Sdf.index()
        }
    }

    struct OverlapProbeMesher {
        entered: Arc<(Mutex<usize>, Condvar)>,
        inside: Arc<AtomicUsize>,
        max_inside: Arc<AtomicUsize>,
    }

    struct InsideBuildGuard<'a>(&'a AtomicUsize);

    impl Drop for InsideBuildGuard<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl VoxelMesher for OverlapProbeMesher {
        fn build(&self, _output: &mut MesherOutput, _input: &MesherInput<'_>) {
            let current = self.inside.fetch_add(1, Ordering::SeqCst) + 1;
            let _guard = InsideBuildGuard(&self.inside);
            self.max_inside.fetch_max(current, Ordering::SeqCst);

            let (lock, cvar) = &*self.entered;
            let mut entered = lock.lock().unwrap();
            *entered += 1;
            cvar.notify_all();

            let deadline = Instant::now() + Duration::from_secs(2);
            while *entered < 2 {
                let now = Instant::now();
                assert!(
                    now < deadline,
                    "mesh tasks did not overlap inside the shared mesher"
                );
                let timeout = deadline.saturating_duration_since(now);
                let (next, wait) = cvar.wait_timeout(entered, timeout).unwrap();
                entered = next;
                assert!(
                    !wait.timed_out() || *entered >= 2,
                    "mesh tasks did not overlap inside the shared mesher"
                );
            }
        }

        fn used_channels_mask(&self) -> u32 {
            0
        }
    }

    /// A mesher that emits a single transvoxel surface with one dummy
    /// triangle, proving the gather→build pipeline ran end-to-end.
    struct DummyMesher {
        build_calls: Arc<Mutex<usize>>,
    }
    impl VoxelMesher for DummyMesher {
        fn build(&self, output: &mut MesherOutput, _input: &MesherInput<'_>) {
            *self.build_calls.lock().unwrap() += 1;
            let mut arrays = crate::meshers::transvoxel::structures::MeshArrays::default();
            let a = arrays.add_vertex(
                crate::math::Vector3f::zero(),
                crate::math::Vector3f::new(0.0, 1.0, 0.0),
                0,
                0,
                0,
                crate::math::Vector3f::zero(),
            );
            arrays.indices.extend_from_slice(&[a, a, a]);
            output
                .surfaces
                .push(Surface::new(SurfaceArrays::Transvoxel(arrays), 0));
        }

        fn used_channels_mask(&self) -> u32 {
            1 << ChannelId::Sdf.index()
        }
    }

    fn shared_data_with_central_block(block_size: i32) -> Arc<SharedVoxelData> {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(
            Vector3i::splat(-block_size * 4),
            Vector3i::splat(block_size * 8),
        ));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        // Force-residence of the central block by writing one voxel into it.
        data.try_set_voxel(1, Vector3i::new(1, 1, 1), ChannelId::Type.index());
        Arc::new(SharedVoxelData::new(data))
    }

    fn shared_data_for_three_lod_task() -> Arc<SharedVoxelData> {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(4096)));
        data.set_lod_count(3).unwrap();
        Arc::new(SharedVoxelData::new(data))
    }

    #[test]
    fn gather_voxels_cpu_fills_padded_buffer_from_central_block_and_generates_neighbours() {
        let mut data = VoxelData::new();
        let bs = data.block_size() as i32;
        data.set_bounds(Box3i::new(
            Vector3i::splat(-bs * 4),
            Vector3i::splat(bs * 8),
        ));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        // Materialise the central block with a recognisable raw value.
        data.try_set_voxel(7, Vector3i::new(1, 1, 1), ChannelId::Type.index());

        let generator = ConstantSdfGenerator { value: -0.5 };

        let mut dst = VoxelBuffer::with_size(Vector3i::zero());
        try_gather_voxels_cpu(
            &mut dst,
            1,
            1,
            1u32 << ChannelId::Type.index() | 1u32 << ChannelId::Sdf.index(),
            Some(&generator),
            &data,
            0,
            Vector3i::zero(),
        )
        .unwrap();

        // Padded size is bs + 2 (1+1 padding on every axis).
        assert_eq!(dst.size(), Vector3i::splat(bs + 2));
        // The central block wrote Type=7 at local (1,1,1) world; in dst that
        // voxel sits at (min_padding + 1, min_padding + 1, min_padding + 1).
        assert_eq!(dst.get_voxel(2, 2, 2, ChannelId::Type.index()), 7);
        // A voxel in a missing neighbour region (just outside the central
        // block) was filled by the generator with a near -0.5 SDF. The exact
        // value runs through the SDF channel's signed-normalised quantiser
        // (default 16-bit), so check the sign and rough magnitude instead of
        // an exact equality.
        let sdf = dst.get_voxel_f(0, 0, 0, ChannelId::Sdf.index());
        assert!(sdf < 0.0, "expected negative SDF, got {sdf}");
        assert!(
            (sdf - (-0.5)).abs() < 0.05,
            "expected SDF near -0.5, got {sdf}"
        );
    }

    #[test]
    fn checked_gather_origin_scales_block_and_padding_at_lod1() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048)));
        data.set_lod_count(2).unwrap();
        let mut dst = VoxelBuffer::with_size(Vector3i::zero());

        let origin =
            try_gather_voxels_cpu(&mut dst, 1, 1, 0, None, &data, 1, Vector3i::new(2, -3, 4))
                .unwrap();

        assert_eq!(origin, Vector3i::new(62, -98, 126));
    }

    #[test]
    fn mesh_block_task_produces_non_empty_output_for_resident_central_block() {
        let data = shared_data_with_central_block(16);
        let build_calls = Arc::new(Mutex::new(0));
        let mesher: Arc<dyn VoxelMesher> = Arc::new(DummyMesher {
            build_calls: build_calls.clone(),
        });
        let generator: Arc<dyn VoxelGenerator> = Arc::new(ConstantSdfGenerator { value: -1.0 });
        let meshing_dep = MeshingDependency::new(mesher, Some(generator));

        let mut task = MeshBlockTask::new(MeshBlockTaskParams {
            key: MeshBlockKey {
                location: MeshBlockLocation::new(Vector3i::zero(), 0),
                revision: 0,
            },
            data: data.clone(),
            meshing_dependency: meshing_dep,
            collision_hint: false,
            lod_hint: false,
            mesh_arrays_pool: None,
        });

        task.run_meshing();
        let output = task.take_output().expect("task produced output");

        assert!(!output.dropped);
        assert_eq!(
            output.upload().key().location.position_in_blocks,
            Vector3i::zero()
        );
        assert_eq!(output.upload().key().location.lod_index, 0);
        assert!(output.upload().output().total_triangle_count() > 0);
        assert_eq!(*build_calls.lock().unwrap(), 1);
    }

    #[test]
    fn mesh_block_task_clips_i32_min_boundary_without_generating_outside_world() {
        let block_size = 16;
        let bounds = Box3i::new(Vector3i::splat(i32::MIN), Vector3i::splat(64));
        let min_block = Vector3i::splat(i32::MIN.div_euclid(block_size));
        let mut data = VoxelData::new();
        data.set_bounds(bounds);
        for dz in 0..=1 {
            for dx in 0..=1 {
                for dy in 0..=1 {
                    let position = min_block + Vector3i::new(dx, dy, dz);
                    assert!(data.try_set_block(
                        position,
                        VoxelDataBlock::with_voxels(
                            VoxelBuffer::with_size(Vector3i::splat(block_size)),
                            0,
                        ),
                    ));
                }
            }
        }
        let data = Arc::new(SharedVoxelData::new(data));
        let build_calls = Arc::new(Mutex::new(0));
        let generator_calls = Arc::new(AtomicUsize::new(0));
        let mesher: Arc<dyn VoxelMesher> = Arc::new(DummyMesher {
            build_calls: build_calls.clone(),
        });
        let generator: Arc<dyn VoxelGenerator> = Arc::new(CountingGenerator {
            calls: generator_calls.clone(),
        });
        let mut task = MeshBlockTask::new(MeshBlockTaskParams {
            key: MeshBlockKey {
                location: MeshBlockLocation::new(min_block, 0),
                revision: 1,
            },
            data,
            meshing_dependency: MeshingDependency::new(mesher, Some(generator)),
            collision_hint: false,
            lod_hint: false,
            mesh_arrays_pool: None,
        });

        task.run_meshing();

        let output = task.take_output().expect("boundary task produced output");
        assert!(!output.dropped);
        assert_eq!(*build_calls.lock().unwrap(), 1);
        assert_eq!(generator_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn mesh_block_task_releases_data_lock_before_generator_fallback() {
        let data = shared_data_with_central_block(16);
        let build_calls = Arc::new(Mutex::new(0));
        let mesher: Arc<dyn VoxelMesher> = Arc::new(DummyMesher {
            build_calls: build_calls.clone(),
        });
        let generator: Arc<dyn VoxelGenerator> = Arc::new(VoxelDataLockProbeGenerator {
            data: Arc::downgrade(&data),
        });
        let meshing_dep = MeshingDependency::new(mesher, Some(generator));

        let mut task = MeshBlockTask::new(MeshBlockTaskParams {
            key: MeshBlockKey {
                location: MeshBlockLocation::new(Vector3i::zero(), 0),
                revision: 0,
            },
            data,
            meshing_dependency: meshing_dep,
            collision_hint: false,
            lod_hint: false,
            mesh_arrays_pool: None,
        });

        task.run_meshing();
        let output = task.take_output().expect("task produced output");

        assert!(!output.dropped);
        assert_eq!(*build_calls.lock().unwrap(), 1);
    }

    #[test]
    fn mesh_block_task_releases_data_lock_before_mesher_build() {
        let data = shared_data_with_central_block(16);
        let build_calls = Arc::new(Mutex::new(0));
        let mesher: Arc<dyn VoxelMesher> = Arc::new(VoxelDataLockProbeMesher {
            data: Arc::downgrade(&data),
            build_calls: build_calls.clone(),
        });
        let generator: Arc<dyn VoxelGenerator> = Arc::new(ConstantSdfGenerator { value: -1.0 });
        let meshing_dep = MeshingDependency::new(mesher, Some(generator));

        let mut task = MeshBlockTask::new(MeshBlockTaskParams {
            key: MeshBlockKey {
                location: MeshBlockLocation::new(Vector3i::zero(), 0),
                revision: 0,
            },
            data,
            meshing_dependency: meshing_dep,
            collision_hint: false,
            lod_hint: false,
            mesh_arrays_pool: None,
        });

        task.run_meshing();
        let output = task.take_output().expect("task produced output");

        assert!(!output.dropped);
        assert_eq!(*build_calls.lock().unwrap(), 1);
    }

    #[test]
    fn mesh_block_tasks_can_overlap_inside_shared_mesher() {
        let data = shared_data_with_central_block(16);
        let entered = Arc::new((Mutex::new(0), Condvar::new()));
        let inside = Arc::new(AtomicUsize::new(0));
        let max_inside = Arc::new(AtomicUsize::new(0));
        let mesher: Arc<dyn VoxelMesher> = Arc::new(OverlapProbeMesher {
            entered,
            inside,
            max_inside: max_inside.clone(),
        });
        let meshing_dep = MeshingDependency::new(mesher, None);

        let make_task = |position_in_blocks| {
            MeshBlockTask::new(MeshBlockTaskParams {
                key: MeshBlockKey {
                    location: MeshBlockLocation::new(position_in_blocks, 0),
                    revision: 0,
                },
                data: data.clone(),
                meshing_dependency: meshing_dep.clone(),
                collision_hint: false,
                lod_hint: false,
                mesh_arrays_pool: None,
            })
        };
        let mut first = make_task(Vector3i::zero());
        let mut second = make_task(Vector3i::new(1, 0, 0));

        let first = std::thread::spawn(move || {
            first.run_meshing();
            first.take_output().expect("first task produced output")
        });
        let second = std::thread::spawn(move || {
            second.run_meshing();
            second.take_output().expect("second task produced output")
        });

        let first = first.join().expect("first mesh task completed");
        let second = second.join().expect("second mesh task completed");

        assert!(!first.dropped);
        assert!(!second.dropped);
        assert_eq!(max_inside.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn mesh_block_task_emits_dropped_output_when_dependency_invalidated() {
        let data = shared_data_with_central_block(16);
        let build_calls = Arc::new(Mutex::new(0));
        let mesher: Arc<dyn VoxelMesher> = Arc::new(DummyMesher {
            build_calls: build_calls.clone(),
        });
        let meshing_dep = MeshingDependency::new(mesher, None);

        let mut task = MeshBlockTask::new(MeshBlockTaskParams {
            key: MeshBlockKey {
                location: MeshBlockLocation::new(Vector3i::zero(), 0),
                revision: 0,
            },
            data: data.clone(),
            meshing_dependency: meshing_dep.clone(),
            collision_hint: false,
            lod_hint: false,
            mesh_arrays_pool: None,
        });

        // Invalidate the dependency before running; the task must not call
        // the mesher and must emit a dropped output.
        meshing_dep.invalidate();
        task.run_meshing();
        let output = task.take_output().expect("task produced output");

        assert!(output.dropped);
        assert!(output.upload().output().is_empty());
        assert_eq!(*build_calls.lock().unwrap(), 0);
    }

    #[test]
    fn mesh_block_task_request_cancellation_is_checked_before_and_after_work() {
        let pre_cancelled = Arc::new(RequestCancellation::new());
        pre_cancelled.cancel();
        let pre_build_calls = Arc::new(Mutex::new(0));
        let mut pre_cancelled_task = MeshBlockTask::new(MeshBlockTaskParams {
            key: MeshBlockKey {
                location: MeshBlockLocation::new(Vector3i::zero(), 0),
                revision: 3,
            },
            data: shared_data_with_central_block(16),
            meshing_dependency: MeshingDependency::new(
                Arc::new(DummyMesher {
                    build_calls: pre_build_calls.clone(),
                }),
                None,
            ),
            collision_hint: false,
            lod_hint: false,
            mesh_arrays_pool: None,
        })
        .with_request_control(TaskRequestTag::new(5, 3), pre_cancelled);

        assert_eq!(
            pre_cancelled_task.request_tag(),
            Some(TaskRequestTag::new(5, 3))
        );
        assert!(pre_cancelled_task.is_cancelled());
        pre_cancelled_task.run_meshing();
        assert!(pre_cancelled_task.take_output().unwrap().dropped());
        assert_eq!(*pre_build_calls.lock().unwrap(), 0);

        let during_build = Arc::new(RequestCancellation::new());
        let during_build_calls = Arc::new(AtomicUsize::new(0));
        let mut during_build_task = MeshBlockTask::new(MeshBlockTaskParams {
            key: MeshBlockKey {
                location: MeshBlockLocation::new(Vector3i::zero(), 0),
                revision: 4,
            },
            data: shared_data_with_central_block(16),
            meshing_dependency: MeshingDependency::new(
                Arc::new(CancellingMesher {
                    cancellation: during_build.clone(),
                    build_calls: during_build_calls.clone(),
                }),
                None,
            ),
            collision_hint: false,
            lod_hint: false,
            mesh_arrays_pool: None,
        })
        .with_request_control(TaskRequestTag::new(5, 4), during_build);

        during_build_task.run_meshing();
        assert!(during_build_task.take_output().unwrap().dropped());
        assert_eq!(during_build_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn mesh_block_task_implements_threaded_task_contract() {
        let data = shared_data_for_three_lod_task();
        let mesher: Arc<dyn VoxelMesher> = Arc::new(DummyMesher {
            build_calls: Arc::new(Mutex::new(0)),
        });
        let meshing_dep = MeshingDependency::new(mesher, None);

        let mut task = MeshBlockTask::new(MeshBlockTaskParams {
            key: MeshBlockKey {
                location: MeshBlockLocation::new(Vector3i::new(3, 4, 5), 2),
                revision: 0,
            },
            data,
            meshing_dependency: meshing_dep,
            collision_hint: true,
            lod_hint: true,
            mesh_arrays_pool: None,
        });

        use crate::tasks::ThreadedTaskContext;
        assert_eq!(task.position_in_blocks(), Vector3i::new(3, 4, 5));
        assert_eq!(task.lod_index(), 2);
        assert_eq!(task.debug_name(), "MeshBlockTask");
        assert!(!task.is_cancelled());

        // Run via the trait method (the threaded-task entry point).
        let outcome = task.run(ThreadedTaskContext::new(
            0,
            crate::tasks::TaskPriority::max(),
        ));
        assert!(matches!(
            outcome,
            crate::tasks::TaskRunStatus::Complete { .. }
        ));

        // Separate run via run_meshing to assert the output struct shape.
        let mut fresh = MeshBlockTask::new(MeshBlockTaskParams {
            key: MeshBlockKey {
                location: MeshBlockLocation::new(Vector3i::new(3, 4, 5), 2),
                revision: 0,
            },
            data: shared_data_for_three_lod_task(),
            meshing_dependency: MeshingDependency::new(
                Arc::new(DummyMesher {
                    build_calls: Arc::new(Mutex::new(0)),
                }),
                None,
            ),
            collision_hint: true,
            lod_hint: true,
            mesh_arrays_pool: None,
        });
        fresh.run_meshing();
        let worker_upload = fresh.output_ref().unwrap().upload().clone();
        let output = fresh.take_output().unwrap();
        assert!(!output.dropped);
        assert!(Arc::ptr_eq(&worker_upload, output.upload()));
        assert_eq!(
            output.upload().features(),
            super::MeshBuildFeatures {
                visuals: true,
                collisions: true,
                variable_lod: true,
            }
        );
        assert_eq!(
            output.upload().visual_state(),
            super::PayloadState::NonEmpty
        );
        assert_eq!(
            output.upload().collision_state(),
            super::PayloadState::Empty
        );
        assert!(output
            .upload()
            .features()
            .contains(super::MeshBuildFeatures {
                visuals: true,
                collisions: true,
                variable_lod: false,
            }));
        assert!(!super::MeshBuildFeatures {
            visuals: true,
            collisions: false,
            variable_lod: false,
        }
        .contains(super::MeshBuildFeatures {
            visuals: true,
            collisions: true,
            variable_lod: false,
        }));
    }
}
