//! End-to-end mesh-block-task benchmarks.
//!
//! Whereas `transvoxel_bench` measures the transvoxel *kernel*
//! (`build_regular_mesh`) in isolation, this benchmark measures the realistic
//! per-chunk cost at the level the terrain pipeline actually drives:
//! [`MeshBlockTask::run_meshing`] = gather a 3×3×3 data neighbourhood +
//! run the mesher. This is the H2 measurement scope called for in
//! `AUDIT.md` §9.4 — the kernel benchmark bypasses the adapter layer where
//! the Wave 3 fixes (B1/B3/B4/B5) land.
//!
//! Two groups:
//! - `block_task/single` — single-threaded gather+mesh of one block.
//! - `block_task/mt` — N worker threads meshing M blocks against a shared
//!   `SharedVoxelData`, to show multi-threaded throughput scaling (the
//!   AUDIT.md Wave-1 DoD: "time ~1/N, >1 core utilized").

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use voxel_core::engine::MeshingDependency;
use voxel_core::generators::base::{GenResult, VoxelGenerator, VoxelQueryData};
use voxel_core::math::Vector3i;
use voxel_core::meshers::{
    MeshArraysPool, MeshBlockKey, MeshBlockLocation, MeshBlockTask, MeshBlockTaskParams,
    TransvoxelMesher, VoxelMesher,
};
use voxel_core::storage::{ChannelId, SharedVoxelData, VoxelData, VoxelDataBlock};

/// A generator that carves a sphere SDF out of an otherwise-solid block.
/// Cheap (no noise), deterministic, and exercises the transvoxel mesher with
/// real surface-crossing cells — the shape the per-block cost is meant for.
struct SphereSdfGenerator {
    block_size: i32,
    radius: f32,
}

impl VoxelGenerator for SphereSdfGenerator {
    fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
        let bs = self.block_size as f32;
        let centre = bs * 0.5;
        let origin = input.origin_in_voxels;
        let channel = ChannelId::Sdf.index();
        let size = input.buffer.size();
        for z in 0..size.z {
            for x in 0..size.x {
                for y in 0..size.y {
                    let wx = (origin.x + x) as f32;
                    let wy = (origin.y + y) as f32;
                    let wz = (origin.z + z) as f32;
                    // Distance to the centre of the *neighbourhood's* central block,
                    // approximated per-block by the block's own centre. Good enough
                    // to produce surface geometry in the central block.
                    let dx = wx - centre;
                    let dy = wy - centre;
                    let dz = wz - centre;
                    let d = (dx * dx + dy * dy + dz * dz).sqrt() - self.radius;
                    input.buffer.set_voxel_f(d, x, y, z, channel);
                }
            }
        }
        GenResult::default()
    }
}

/// Build a `SharedVoxelData` whose central block is resident (so the gather
/// step mostly copies resident data) and a transvoxel meshing dependency.
fn sphere_setup(block_size: i32) -> (Arc<SharedVoxelData>, Arc<MeshingDependency>) {
    let mut data = VoxelData::new();
    data.set_bounds(voxel_core::math::Box3i::new(
        Vector3i::splat(-block_size * 4),
        Vector3i::splat(block_size * 8),
    ));
    data.set_streaming_enabled(false);
    data.set_full_load_completed(true);
    // Force the central block resident by inserting a populated block.
    let mut block_buf = voxel_core::storage::VoxelBuffer::with_size(Vector3i::splat(block_size));
    {
        let mut format = voxel_core::storage::VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = voxel_core::storage::ChannelDepth::Bit16;
        format.configure_buffer(&mut block_buf);
        let centre = block_size as f32 * 0.5;
        let radius = block_size as f32 * 0.35;
        for z in 0..block_size {
            for x in 0..block_size {
                for y in 0..block_size {
                    let d = ((x as f32 - centre).powi(2)
                        + (y as f32 - centre).powi(2)
                        + (z as f32 - centre).powi(2))
                    .sqrt()
                        - radius;
                    block_buf.set_voxel_f(d, x, y, z, ChannelId::Sdf.index());
                }
            }
        }
    }
    let mut block = VoxelDataBlock::with_voxels(block_buf, 0);
    block.set_edited(true);
    data.try_set_block(Vector3i::zero(), block);

    let shared = Arc::new(SharedVoxelData::new(data));
    let mesher: Arc<dyn VoxelMesher> = Arc::new(TransvoxelMesher::new());
    let generator: Arc<dyn VoxelGenerator> = Arc::new(SphereSdfGenerator {
        block_size,
        radius: block_size as f32 * 0.35,
    });
    let meshing_dep = MeshingDependency::new(mesher, Some(generator));
    (shared, meshing_dep)
}

/// One full gather+mesh of the central block. Returns the vertex count so the
/// compiler cannot elide the work. The shared [`MeshArraysPool`] mirrors how
/// the terrain core drives tasks (steady-state buffer reuse).
fn run_one_block(
    data: &Arc<SharedVoxelData>,
    meshing_dep: &Arc<MeshingDependency>,
    pool: &Arc<MeshArraysPool>,
    pos: Vector3i,
) -> usize {
    let mut task = MeshBlockTask::new(MeshBlockTaskParams {
        key: MeshBlockKey {
            location: MeshBlockLocation::new(pos, 0),
            revision: 0,
        },
        data: data.clone(),
        meshing_dependency: meshing_dep.clone(),
        collision_hint: false,
        lod_hint: false,
        mesh_arrays_pool: Some(pool.clone()),
    });
    task.run_meshing();
    let output = task.take_output().expect("task produced output");
    let verts = output.upload().output().total_vertex_count();
    drop(output);
    verts
}

fn bench_block_task_single(c: &mut Criterion) {
    let block_size = 16; // matches the default VoxelData block size
    let (data, meshing_dep) = sphere_setup(block_size);
    let pool = Arc::new(MeshArraysPool::new());
    // Confirm the setup actually produces geometry before timing it.
    let verts = run_one_block(&data, &meshing_dep, &pool, Vector3i::zero());
    assert!(verts > 0, "bench setup produced no geometry");

    let mut group = c.benchmark_group("block_task/single");
    // Throughput unit = voxels in one data block (the work one task does).
    group.throughput(Throughput::Elements((block_size as u64).pow(3)));
    group.bench_function("sphere_16", |b| {
        b.iter(|| {
            let n = run_one_block(&data, &meshing_dep, &pool, Vector3i::zero());
            std::hint::black_box(n);
        });
    });
    group.finish();
}

/// Multi-threaded scaling: spawn `n_threads` scoped threads, each meshing
/// `blocks_per_thread` distinct positions against the shared data. We measure
/// wall-clock time for the whole batch and report throughput as blocks/sec.
/// This is the AUDIT.md Wave-1/§9.4 "does it scale past one core?" check.
fn bench_block_task_mt(c: &mut Criterion) {
    let block_size = 16;
    let (data, meshing_dep) = sphere_setup(block_size);
    let pool = Arc::new(MeshArraysPool::new());

    let mut group = c.benchmark_group("block_task/mt");
    group.sample_size(10);

    for &n_threads in &[1usize, 2, 4, 8] {
        let blocks_per_thread = 16;
        let total_blocks = (n_threads * blocks_per_thread) as u64;
        group.throughput(Throughput::Elements(total_blocks));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n_threads}_threads")),
            &n_threads,
            |b, &n_threads| {
                b.iter(|| {
                    let counter = Arc::new(AtomicUsize::new(0));
                    std::thread::scope(|s| {
                        for t in 0..n_threads {
                            let data = data.clone();
                            let dep = meshing_dep.clone();
                            let pool = pool.clone();
                            let counter = counter.clone();
                            s.spawn(move || {
                                for i in 0..blocks_per_thread {
                                    // Offset each block so threads mostly don't
                                    // collide on the same central block.
                                    let pos = Vector3i::new(
                                        t as i32 * 3 + (i as i32 % 3),
                                        (i as i32 / 3) % 3,
                                        0,
                                    );
                                    let n = run_one_block(&data, &dep, &pool, pos);
                                    counter.fetch_add(n, Ordering::Relaxed);
                                }
                            });
                        }
                    });
                    std::hint::black_box(counter.load(Ordering::Relaxed));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_block_task_single, bench_block_task_mt);
criterion_main!(benches);
