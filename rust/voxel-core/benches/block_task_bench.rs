//! End-to-end mesh-block-task benchmarks.
//!
//! Measures the realistic per-chunk cost at the level the terrain pipeline
//! actually drives: [`MeshBlockTask::run_meshing`] = generator gap-fill +
//! gather of a 3×3×3 data neighbourhood + mesher build, with the shared
//! [`MeshArraysPool`] the terrain core uses for steady-state buffer reuse.
//!
//! Sibling benchmarks (deliberate split, keep in sync):
//! - `transvoxel_bench` — the kernel only (`build_regular_mesh` on a raw
//!   slice): no gather, pool, or task layers.
//! - `mesh_block_bench` — all-resident data dispatched through the real
//!   `ThreadedTaskRunner`: paging throughput including task scheduling.
//! - this bench — raw worker scaling with generator gap-fill in the loop.
//!
//! Two groups:
//! - `block_task/single` — single-threaded gather+mesh of one block.
//! - `block_task/mt` — N worker threads meshing blocks round-robin from the
//!   in-bounds block range (asserting every task does real work), to show
//!   multi-threaded throughput scaling (the retired `rust/pilot` Wave-1 DoD:
//!   "time ~1/N, >1 core utilized").

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use voxel_core::engine::MeshingDependency;
use voxel_core::generators::base::{GenResult, VoxelGenerator, VoxelQueryData};
use voxel_core::math::{Box3i, Vector3i};
use voxel_core::meshers::{
    MeshArraysPool, MeshBlockKey, MeshBlockLocation, MeshBlockTask, MeshBlockTaskParams,
    TransvoxelMesher, VoxelMesher,
};
use voxel_core::storage::{ChannelId, SharedVoxelData, VoxelData, VoxelDataBlock};
use voxel_core::terrain::lod_clipbox::bounds_in_lod_blocks;

/// Volume (in voxels) the bench data covers. Block range is derived from it
/// via the same helper the terrain uses, so bench positions can never drift
/// out of bounds (out-of-bounds tasks are silently dropped before gather and
/// would pollute the scaling numbers with no-op work).
fn volume_bounds(block_size: i32) -> Box3i {
    Box3i::new(
        Vector3i::splat(-block_size * 4),
        Vector3i::splat(block_size * 8),
    )
}

/// A generator that carves a sphere SDF out of an otherwise-solid block.
/// Cheap (no noise), deterministic, and exercises the transvoxel mesher with
/// real surface-crossing cells. The sphere is periodic: one per block-grid
/// cell, confined to each block's interior, so every in-bounds mesh block
/// contains the same surface and does identical meshing work.
struct SphereSdfGenerator {
    block_size: i32,
    radius: f32,
}

impl VoxelGenerator for SphereSdfGenerator {
    fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
        let bs = self.block_size;
        let centre = bs as f32 * 0.5;
        let origin = input.origin_in_voxels;
        let channel = ChannelId::Sdf.index();
        let size = input.buffer.size();
        for z in 0..size.z {
            for x in 0..size.x {
                for y in 0..size.y {
                    // Block-local coordinates: the same sphere for every
                    // block of the grid, centred in each block.
                    let lx = (origin.x + x).rem_euclid(bs) as f32;
                    let ly = (origin.y + y).rem_euclid(bs) as f32;
                    let lz = (origin.z + z).rem_euclid(bs) as f32;
                    let dx = lx - centre;
                    let dy = ly - centre;
                    let dz = lz - centre;
                    let d = (dx * dx + dy * dy + dz * dz).sqrt() - self.radius;
                    input.buffer.set_voxel_f(d, x, y, z, channel);
                }
            }
        }
        GenResult::default()
    }
}

/// Build a `SharedVoxelData` whose central block is resident (so the `single`
/// group's gather mostly copies resident data) and a transvoxel meshing
/// dependency.
fn sphere_setup(block_size: i32) -> (Arc<SharedVoxelData>, Arc<MeshingDependency>) {
    let mut data = VoxelData::new();
    data.set_bounds(volume_bounds(block_size));
    data.set_streaming_enabled(false);
    data.set_full_load_completed(true);
    // Force the central block resident by inserting a populated block. Its
    // contents match the periodic generator's sphere for that block.
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

/// One full gather+mesh of one block. Returns the vertex count so the
/// compiler cannot elide the work. The shared [`MeshArraysPool`] mirrors how
/// the terrain core drives tasks (steady-state buffer reuse). Panics if the
/// task was dropped (out of bounds) or produced no geometry — the bench must
/// never count no-op tasks as meshed blocks.
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
    assert!(!output.dropped(), "bench block {pos:?} was dropped");
    let verts = output.upload().output().total_vertex_count();
    assert!(verts > 0, "bench block {pos:?} produced no geometry");
    drop(output);
    verts
}

fn bench_block_task_single(c: &mut Criterion) {
    let block_size = 16; // matches the default VoxelData block size
    let (data, meshing_dep) = sphere_setup(block_size);
    let pool = Arc::new(MeshArraysPool::new());
    // Confirm the setup actually produces geometry before timing it
    // (also guards against future regressions that would drop the task).
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

/// Multi-threaded scaling: spawn `n_threads` scoped threads meshing
/// `blocks_per_thread` blocks each, round-robin over the in-bounds block
/// positions. We measure wall-clock time for the whole batch and report
/// throughput as blocks/sec — the "does it scale past one core?" check.
///
/// Candidates are inset one block from the volume faces (so no neighbour is
/// clipped by bounds and every gather queues the full 3×3×3) and exclude the
/// 3×3×3 neighbourhood of the resident origin block (so no neighbour is
/// copied from residency): every block generates all 27 neighbours and per-
/// block work is identical across threads — the wall-clock comparison
/// measures parallelism, not workload skew.
fn bench_block_task_mt(c: &mut Criterion) {
    let block_size = 16;
    let (data, meshing_dep) = sphere_setup(block_size);
    let pool = Arc::new(MeshArraysPool::new());

    let blocks_in_bounds =
        bounds_in_lod_blocks(volume_bounds(block_size), block_size, 0).expect("valid bounds");
    let candidates: Vec<Vector3i> = blocks_in_bounds
        .padded(-1)
        .iter_cells_zxy()
        .filter(|p| p.x.abs() > 1 || p.y.abs() > 1 || p.z.abs() > 1)
        .collect();
    const BLOCKS_PER_THREAD: usize = 16;
    assert!(
        candidates.len() >= 8 * BLOCKS_PER_THREAD,
        "not enough in-bounds block positions for 8 threads (got {})",
        candidates.len()
    );

    let mut group = c.benchmark_group("block_task/mt");
    group.sample_size(10);

    for &n_threads in &[1usize, 2, 4, 8] {
        let total_blocks = (n_threads * BLOCKS_PER_THREAD) as u64;
        group.throughput(Throughput::Elements(total_blocks));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n_threads}_threads")),
            &n_threads,
            |b, &n_threads| {
                b.iter(|| {
                    let counter = Arc::new(AtomicUsize::new(0));
                    let total = n_threads * BLOCKS_PER_THREAD;
                    std::thread::scope(|s| {
                        for t in 0..n_threads {
                            let data = data.clone();
                            let dep = meshing_dep.clone();
                            let pool = pool.clone();
                            let counter = counter.clone();
                            let positions = &candidates;
                            s.spawn(move || {
                                // Round-robin over the shared position list so
                                // neighbouring positions (and any locality)
                                // spread evenly across threads.
                                let mut i = t;
                                while i < total {
                                    let pos = positions[i];
                                    let n = run_one_block(&data, &dep, &pool, pos);
                                    counter.fetch_add(n, Ordering::Relaxed);
                                    i += n_threads;
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
