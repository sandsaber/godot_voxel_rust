//! End-to-end pipeline test: generator → VoxelData → MeshBlockTask → mesh.
//!
//! Exercises the full Phase 4 storage + meshing pipeline headlessly (no
//! Godot, no threads): a procedural generator fills a `VoxelData`, a
//! `MeshBlockTask` gathers voxels with neighbours and runs a real mesher
//! against them, and we assert the output has the geometry we expect.
//!
//! This is the integration test the audit recommended as the validation
//! milestone for Phase 4's algorithmic core.

use std::sync::Arc;
use voxel_core::engine::MeshingDependency;
use voxel_core::generators::base::{VoxelGenerator, VoxelQueryData};
use voxel_core::math::{Box3i, Vector3i};
use voxel_core::meshers::{
    MeshBlockKey, MeshBlockLocation, MeshBlockTask, MeshBlockTaskOutput, MeshBlockTaskParams,
    MesherOutput, Surface, SurfaceArrays, VoxelMesher,
};
use voxel_core::storage::{
    ChannelDepth, ChannelId, SharedVoxelData, VoxelBuffer, VoxelData, VoxelDataBlock, VoxelFormat,
};

/// A generator that produces a sphere SDF centred at the origin. This is the
/// same generator used by `tests/transvoxel_sphere.rs`, reimplemented here to
/// stay self-contained.
struct SphereGenerator {
    radius: f32,
}

impl VoxelGenerator for SphereGenerator {
    fn generate_block(&self, input: VoxelQueryData<'_>) -> voxel_core::generators::base::GenResult {
        let bs = input.buffer.size().x as f32;
        for z in 0..input.buffer.size().z {
            for x in 0..input.buffer.size().x {
                for y in 0..input.buffer.size().y {
                    let wx = input.origin_in_voxels.x as f32 + x as f32 + 0.5;
                    let wy = input.origin_in_voxels.y as f32 + y as f32 + 0.5;
                    let wz = input.origin_in_voxels.z as f32 + z as f32 + 0.5;
                    let d = ((wx * wx + wy * wy + wz * wz).sqrt()) - self.radius;
                    input.buffer.set_voxel_f(d, x, y, z, ChannelId::Sdf.index());
                }
            }
        }
        let _ = bs;
        voxel_core::generators::base::GenResult::default()
    }

    fn used_channels_mask(&self) -> u32 {
        1 << ChannelId::Sdf.index()
    }
}

/// A real mesher (not a stub) that reads the SDF channel and produces a
/// crude "marching cubes"-style triangle per cell where the SDF crosses
/// zero. This isn't the full transvoxel algorithm — the production mesher
/// lives in `meshers::transvoxel` and has its own parity tests — but it
/// exercises the full gather→build→output pipeline with real geometry.
struct ThresholdMesher;

impl VoxelMesher for ThresholdMesher {
    fn build(&self, output: &mut MesherOutput, input: &voxel_core::meshers::MesherInput<'_>) {
        let voxels = input.voxels;
        let channel = ChannelId::Sdf.index();
        let mut arrays = voxel_core::meshers::transvoxel::structures::MeshArrays::default();

        // Walk every cell of the (padded) buffer and emit one triangle when
        // the SDF sign flips between (0,0,0) and (1,1,1) corners. Geometry is
        // intentionally simple — the point is to prove the mesher sees real
        // voxel data the gather step produced.
        let size = voxels.size();
        for z in 0..size.z.saturating_sub(1) {
            for x in 0..size.x.saturating_sub(1) {
                for y in 0..size.y.saturating_sub(1) {
                    let c000 = voxels.get_voxel_f(x, y, z, channel);
                    let c111 = voxels.get_voxel_f(x + 1, y + 1, z + 1, channel);
                    if (c000 < 0.0) != (c111 < 0.0) {
                        let cx = input.origin_in_voxels.x as f32 + x as f32;
                        let cy = input.origin_in_voxels.y as f32 + y as f32;
                        let cz = input.origin_in_voxels.z as f32 + z as f32;
                        let a = arrays.add_vertex(
                            voxel_core::math::Vector3f::new(cx, cy, cz),
                            voxel_core::math::Vector3f::new(0.0, 1.0, 0.0),
                            0,
                            0,
                            0,
                            voxel_core::math::Vector3f::zero(),
                        );
                        let b = arrays.add_vertex(
                            voxel_core::math::Vector3f::new(cx + 1.0, cy, cz),
                            voxel_core::math::Vector3f::new(0.0, 1.0, 0.0),
                            0,
                            0,
                            0,
                            voxel_core::math::Vector3f::zero(),
                        );
                        let c = arrays.add_vertex(
                            voxel_core::math::Vector3f::new(cx, cy, cz + 1.0),
                            voxel_core::math::Vector3f::new(0.0, 1.0, 0.0),
                            0,
                            0,
                            0,
                            voxel_core::math::Vector3f::zero(),
                        );
                        arrays.indices.extend_from_slice(&[a, b, c]);
                    }
                }
            }
        }

        output
            .surfaces
            .push(Surface::new(SurfaceArrays::Transvoxel(arrays), 0));
    }

    fn used_channels_mask(&self) -> u32 {
        1 << ChannelId::Sdf.index()
    }
}

fn sphere_pipeline(radius: f32, mesh_block_pos: Vector3i) -> MeshBlockTaskOutput {
    // 1) Set up VoxelData configured for SDF-only generation. The bounds are
    //    generous so the mesh block and its 3x3x3 neighbours all sit inside.
    let mut data = VoxelData::new();
    let mut format = VoxelFormat::new();
    format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
    data.set_format(format);
    data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048)));
    data.set_streaming_enabled(false);
    data.set_full_load_completed(true);
    // Materialise the central block with the same generator used for missing
    // in-bound neighbours. A single edited negative voxel would create an
    // artificial surface in blocks that are actually far outside the sphere.
    let mut central = VoxelBuffer::with_size(Vector3i::splat(16));
    data.format().configure_buffer(&mut central);
    SphereGenerator { radius }.generate_block(VoxelQueryData {
        buffer: &mut central,
        origin_in_voxels: mesh_block_pos * 16,
        lod: 0,
    });
    assert!(data.try_set_block(mesh_block_pos, VoxelDataBlock::with_voxels(central, 0),));

    let data_handle = Arc::new(SharedVoxelData::new(data));

    // 2) Build the meshing dependency with the sphere generator + threshold
    //    mesher.
    let generator: Arc<dyn VoxelGenerator> = Arc::new(SphereGenerator { radius });
    let mesher: Arc<dyn VoxelMesher> = Arc::new(ThresholdMesher);
    let meshing_dependency = MeshingDependency::new(mesher, Some(generator));

    // 3) Run a MeshBlockTask at the requested position.
    let mut task = MeshBlockTask::new(MeshBlockTaskParams {
        key: MeshBlockKey {
            location: MeshBlockLocation::new(mesh_block_pos, 0),
            revision: 0,
        },
        data: data_handle,
        meshing_dependency,
        collision_hint: false,
        lod_hint: false,
        mesh_arrays_pool: None,
    });
    task.run_meshing();
    task.take_output().expect("task should produce output")
}

#[test]
fn pipeline_produces_non_empty_mesh_for_origin_centred_sphere() {
    let output = sphere_pipeline(8.0, Vector3i::zero());

    assert!(!output.dropped(), "dependency should still be valid");
    assert_eq!(output.upload().key().location.lod_index, 0);
    assert!(
        output.upload().output().total_triangle_count() > 0,
        "expected non-empty mesh for an 8-voxel-radius sphere at the origin"
    );
    assert!(
        output.upload().output().total_vertex_count() > 0,
        "expected vertices in the mesh"
    );
}

#[test]
fn pipeline_empty_mesh_for_block_far_outside_sphere() {
    // A mesh block centred far away from the sphere should produce no
    // geometry (the SDF never crosses zero there).
    let output = sphere_pipeline(8.0, Vector3i::new(50, 0, 0));

    assert!(!output.dropped());
    assert_eq!(
        output.upload().output().total_triangle_count(),
        0,
        "expected empty mesh for a block well outside the sphere"
    );
}

#[test]
fn pipeline_dropped_output_when_dependency_invalidated() {
    // Re-run the setup but invalidate the dependency before the task runs.
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-64), Vector3i::splat(128)));
    data.set_streaming_enabled(false);
    data.set_full_load_completed(true);
    data.try_set_voxel(1, Vector3i::new(1, 1, 1), ChannelId::Type.index());

    let data_handle = Arc::new(SharedVoxelData::new(data));
    let mesher: Arc<dyn VoxelMesher> = Arc::new(ThresholdMesher);
    let meshing_dependency = MeshingDependency::new(mesher, None);

    let mut task = MeshBlockTask::new(MeshBlockTaskParams {
        key: MeshBlockKey {
            location: MeshBlockLocation::new(Vector3i::zero(), 0),
            revision: 0,
        },
        data: data_handle,
        meshing_dependency: meshing_dependency.clone(),
        collision_hint: false,
        lod_hint: false,
        mesh_arrays_pool: None,
    });
    meshing_dependency.invalidate();
    task.run_meshing();

    let output = task.take_output().unwrap();
    assert!(output.dropped());
    assert!(output.upload().output().is_empty());
}

#[test]
fn pipeline_surfaces_are_transvoxel_arrays() {
    // Sanity: the output surface variant matches what the mesher emitted.
    let output = sphere_pipeline(8.0, Vector3i::zero());
    assert!(matches!(
        output.upload().output().surfaces[0].arrays,
        SurfaceArrays::Transvoxel(_)
    ));
}

#[test]
fn pipeline_vertex_positions_are_in_world_space() {
    // Vertices the ThresholdMesher emits carry world-space coordinates
    // derived from `origin_in_voxels`. Verify one of them sits inside the
    // mesh block's voxel range (a basic correctness contract for downstream
    // rendering / collision).
    let output = sphere_pipeline(8.0, Vector3i::zero());
    let arrays = match &output.upload().output().surfaces[0].arrays {
        SurfaceArrays::Transvoxel(a) => a,
        _ => unreachable!(),
    };
    let any_in_block = arrays.vertices.iter().any(|p| {
        p.x >= -1.0 && p.x <= 17.0 && p.y >= -1.0 && p.y <= 17.0 && p.z >= -1.0 && p.z <= 17.0
    });
    assert!(
        any_in_block,
        "expected at least one world-space vertex in the mesh block"
    );
}
