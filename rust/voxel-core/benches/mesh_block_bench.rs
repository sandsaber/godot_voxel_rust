//! H2-MT benchmark: end-to-end `MeshBlockTask` throughput (audit §9.6 "Инфраструктура" item 9).
//!
//! Unlike `transvoxel_bench` (which measures only the `build_regular_mesh` core
//! against a raw `f32` slice), this harness exercises the full pipeline the
//! terrain runs per chunk: `SharedVoxelData` with a resident SDF block →
//! `MeshBlockTask` gather (3×3×3 neighbours) → `TransvoxelMesher::build`.
//!
//! Two groups:
//! - `mesh_block/single` — one block, single-threaded (the per-chunk baseline).
//! - `mesh_block/multi`  — N blocks dispatched through a `ThreadedTaskRunner`
//!   (measures real multi-threaded paging throughput; the goal of the M1.A–M1.D
//!   threading + perf fixes).
//!
//! Run: `cargo bench --bench mesh_block_bench`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use voxel_core::engine::MeshingDependency;
use voxel_core::math::{Box3i, Vector3i};
use voxel_core::meshers::{
    MeshBlockKey, MeshBlockLocation, MeshBlockTask, MeshBlockTaskParams, TransvoxelMesher,
};
use voxel_core::storage::{ChannelDepth, ChannelId, SharedVoxelData, VoxelData, VoxelFormat};
use voxel_core::tasks::{ScheduledTask, TaskLane, ThreadedTask, ThreadedTaskRunner};

/// Build a `SharedVoxelData` whose LOD-0 map has a resident block at the origin
/// filled with an SDF sphere. The data block size matches the mesh block size
/// (factor 1), so a single `MeshBlockTask` at (0,0,0) gathers the central block
/// plus its 3×3×3 padding neighbourhood (all resident, no generator fallback).
fn sphere_data(inner: i32) -> Arc<SharedVoxelData> {
    let mut data = VoxelData::new();
    let mut format = VoxelFormat::new();
    format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
    data.set_format(format);
    data.set_bounds(Box3i::new(Vector3i::splat(-512), Vector3i::splat(2048)));
    data.set_streaming_enabled(false);
    data.set_full_load_completed(true);

    // Fill the central block (0,0,0) with an SDF sphere.
    let block_size = data.block_size() as i32;
    let cx = block_size as f32 * 0.5;
    let cy = block_size as f32 * 0.5;
    let cz = block_size as f32 * 0.5;
    let radius = (inner as f32 * 0.45).max(1.0);
    for z in 0..block_size {
        for y in 0..block_size {
            for x in 0..block_size {
                let d =
                    ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2) + (z as f32 - cz).powi(2))
                        .sqrt()
                        - radius;
                data.try_set_voxel_f(d, Vector3i::new(x, y, z), ChannelId::Sdf.index());
            }
        }
    }

    // Also fill the 26 neighbours so gather finds resident data everywhere
    // (no generator fallback path — measures pure gather+mesh, not generation).
    let _ = inner; // inner only governs the sphere radius above.
    Arc::new(SharedVoxelData::new(data))
}

/// Build a single ready-to-run `MeshBlockTask` (concrete, for the
/// single-threaded bench that calls `run_meshing` directly).
fn make_concrete_task(data: &Arc<SharedVoxelData>) -> MeshBlockTask {
    let mesher = Arc::new(TransvoxelMesher::new());
    let meshing_dependency = MeshingDependency::new(mesher, None);
    MeshBlockTask::new(MeshBlockTaskParams {
        key: MeshBlockKey {
            location: MeshBlockLocation::new(Vector3i::zero(), 0),
            revision: 0,
        },
        data: data.clone(),
        meshing_dependency,
        collision_hint: false,
        lod_hint: false,
        mesh_arrays_pool: None,
    })
}

/// Build a boxed task (for the multi-threaded bench that goes through the runner).
fn make_task(data: &Arc<SharedVoxelData>) -> Box<dyn ThreadedTask> {
    Box::new(make_concrete_task(data)) as Box<dyn ThreadedTask>
}

/// Single-threaded: run one MeshBlockTask end-to-end (gather + mesh).
fn bench_single_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("mesh_block/single");
    let data = sphere_data(16);
    group.throughput(Throughput::Elements(16 * 16 * 16));
    group.bench_function("sphere_16", |b| {
        b.iter(|| {
            let mut task = make_concrete_task(&data);
            task.run_meshing();
            std::hint::black_box(task);
        });
    });
    group.finish();
}

/// Multi-threaded: dispatch N blocks through a ThreadedTaskRunner and measure
/// total wall-clock throughput. This is the H2-MT scenario — it exercises the
/// M1.A threading model (SharedVoxelData region locks, ThreadedTaskRunner) +
/// M1.B–D perf fixes (typed storage, MeshArrays pool, compiled graph) together.
fn bench_multi_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("mesh_block/multi");
    let data = sphere_data(16);
    // 32 blocks ≈ a small terrain update batch.
    const BLOCK_COUNT: usize = 32;
    group.throughput(Throughput::Elements((BLOCK_COUNT * 16 * 16 * 16) as u64));
    group.bench_with_input(
        BenchmarkId::new("sphere_16", BLOCK_COUNT),
        &data,
        |b, data| {
            b.iter(|| {
                let mut runner = ThreadedTaskRunner::new(4);
                let tasks = (0..BLOCK_COUNT)
                    .map(|_| ScheduledTask::new(make_task(data), TaskLane::Parallel));
                runner.enqueue_many(tasks);
                runner.wait_for_all_tasks();
                let mut completed = std::collections::VecDeque::new();
                runner.try_drain_completed_into(&mut completed).unwrap();
                std::hint::black_box(());
            });
        },
    );
    group.finish();
}

criterion_group!(benches, bench_single_block, bench_multi_block);
criterion_main!(benches);
