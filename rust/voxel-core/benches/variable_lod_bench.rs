//! Criterion benchmarks for Variable LOD hot-path operations.
//!
//! Phase E item 7: performance benchmarks for the likely optimization targets
//! identified in the handoff: event-driven upload, packed-array conversion,
//! resource pooling, and avoiding full map scans.
//!
//! Run: `cargo bench --bench variable_lod_bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::sync::Arc;
use voxel_core::generators::simple::Flat;
use voxel_core::math::{Box3i, Vector3i};
use voxel_core::meshers::TransvoxelMesher;
use voxel_core::storage::{ChannelId, VoxelBuffer, VoxelData};
use voxel_core::streams::MemoryStream;
use voxel_core::terrain::clipbox_coordinator::MeshDemand;
use voxel_core::terrain::lod_clipbox::LodClipboxSettings;
use voxel_core::terrain::voxel_terrain_core::{ViewerUpdate, VoxelTerrainCore};

fn make_terrain() -> VoxelTerrainCore {
    let mut data = VoxelData::new();
    data.set_bounds(Box3i::new(Vector3i::splat(-256), Vector3i::splat(512)));
    let generator: Arc<Flat> = Arc::new(Flat::default());
    data.set_generator(Some(generator.clone()));
    let settings = LodClipboxSettings {
        data_block_size: 16,
        mesh_block_size: 16,
        lod_count: 3,
        lod0_distance_voxels: 16,
        secondary_distance_voxels: 16,
        unload_hysteresis_blocks: 2,
    };
    let mesher = Arc::new(TransvoxelMesher::new());
    let dep = voxel_core::engine::MeshingDependency::new(mesher, Some(generator));
    VoxelTerrainCore::new_variable_lod(data, Arc::new(MemoryStream::new()), dep, settings).unwrap()
}

/// Benchmark: a full `try_process` tick with a converged viewer (steady state).
/// Measures the per-tick overhead of the planner + drain + process_meshing
/// path when there is little new work to do.
fn bench_steady_state_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("variable_lod/steady_state_tick");
    group.throughput(Throughput::Elements(1));

    group.bench_function("3-LOD converged", |b| {
        let mut core = make_terrain();
        let viewer = ViewerUpdate {
            id: 1,
            world_position_voxels: Vector3i::zero(),
            horizontal_view_distance_voxels: 48,
            vertical_view_distance_voxels: 48,
            demand: MeshDemand {
                visuals: true,
                collisions: false,
            },
        };
        // Converge first.
        for _ in 0..10 {
            core.try_process(&[viewer]).unwrap();
            core.wait_for_pending_tasks();
        }
        b.iter(|| {
            core.try_process(black_box(&[viewer])).unwrap();
        });
    });

    group.finish();
}

/// Benchmark: a full `try_process` tick with a freshly-entered viewer (paging
/// burst). Measures the planner's scheduling cost when it has a lot of new
/// demand to process.
fn bench_enter_burst_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("variable_lod/enter_burst_tick");
    group.throughput(Throughput::Elements(1));

    group.bench_function("3-LOD first tick", |b| {
        b.iter_batched(
            make_terrain,
            |mut core| {
                let viewer = ViewerUpdate {
                    id: 1,
                    world_position_voxels: Vector3i::zero(),
                    horizontal_view_distance_voxels: 48,
                    vertical_view_distance_voxels: 48,
                    demand: MeshDemand {
                        visuals: true,
                        collisions: false,
                    },
                };
                core.try_process(black_box(&[viewer])).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark: VoxelBuffer packed-array channel operations (get/set/fill).
/// Measures the hot-path voxel access patterns the terrain does per block.
fn bench_packed_array_voxel_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_array/voxel_access");
    let size = Vector3i::splat(16);
    group.throughput(Throughput::Elements((size.x * size.y * size.z) as u64));

    group.bench_function("set_voxel 16³ block", |b| {
        b.iter(|| {
            let mut buf = VoxelBuffer::with_size(size);
            for z in 0..size.z {
                for y in 0..size.y {
                    for x in 0..size.x {
                        buf.set_voxel(black_box(1), x, y, z, ChannelId::Type.index());
                    }
                }
            }
            buf
        });
    });

    group.bench_function("fill 16³ block", |b| {
        b.iter(|| {
            let mut buf = VoxelBuffer::with_size(size);
            buf.fill(black_box(1), ChannelId::Type.index());
            buf
        });
    });

    group.bench_function("get_voxel_f 16³ block", |b| {
        let mut buf = VoxelBuffer::with_size(size);
        buf.fill(0x3F800000, ChannelId::Sdf.index()); // 1.0f as raw
        b.iter(|| {
            let mut sum = 0.0f32;
            for z in 0..size.z {
                for y in 0..size.y {
                    for x in 0..size.x {
                        sum += buf.get_voxel_f(x, y, z, ChannelId::Sdf.index());
                    }
                }
            }
            black_box(sum);
        });
    });

    group.finish();
}

/// Benchmark: VoxelBuffer create + compress + decompress cycle (resource
/// pooling hot path — the terrain creates and discards these per block).
fn bench_buffer_lifecycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_lifecycle");
    let size = Vector3i::splat(16);

    group.bench_function("create + fill + compress + decompress", |b| {
        b.iter(|| {
            let mut buf = VoxelBuffer::with_size(size);
            buf.fill(42, ChannelId::Type.index());
            buf.compress_uniform_channels();
            buf.decompress_channel(ChannelId::Type.index());
            black_box(buf);
        });
    });

    group.finish();
}

criterion_group!(
    variable_lod_benches,
    bench_steady_state_tick,
    bench_enter_burst_tick,
    bench_packed_array_voxel_access,
    bench_buffer_lifecycle,
);
criterion_main!(variable_lod_benches);
