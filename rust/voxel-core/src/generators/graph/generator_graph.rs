//! [`GraphGenerator`] — adapts a [`Graph`] to the [`VoxelGenerator`] trait.
//!
//! Ports the engine-agnostic half of `generators/graph/voxel_generator_graph.cpp`
//! — specifically the `generate_block` loop that walks a `VoxelBuffer` in
//! Y-slices, runs the runtime over each slice, and copies the SDF output back
//! into the SDF channel. Skips the Godot `Resource`/editor/GPU/serialization
//! machinery; that lives in `voxel-gdext`.

use crate::generators::base::{GenResult, VoxelGenerator, VoxelQueryData};
use crate::generators::graph::{
    CompiledGraph, CompiledScratch, Graph, GraphInputs, GraphNodeId, GraphOutput, GraphScratch,
    NodeKind,
};
use crate::math::Vector3i;
use crate::storage::voxel_buffer::ChannelId;
use crate::storage::VoxelBuffer;
use std::sync::{Arc, Mutex};

/// Wraps a [`Graph`] in a [`VoxelGenerator`] that fills a `VoxelBuffer` block
/// by executing the graph one Y-slice at a time. The graph must contain at
/// least one `OutputSdf` node; otherwise `generate_block` is a no-op.
///
/// C1 (audit §9.6-C1): the graph is compiled once (lazily on first
/// `generate_block`) into a [`CompiledGraph`], which caches the topological
/// order, classifies Y-independent nodes (XZ-outer-group prefix), and uses
/// dense scratch buffers. Y-independent subgraphs are evaluated once per block
/// and cached across Y-slices instead of recomputed every slice — up to
/// ~block-height × fewer evaluations for terrain graphs.
pub struct GraphGenerator {
    graph: Graph,
    /// Per-instance scratch for the legacy free-function path.
    scratch: Mutex<GraphScratch>,
    /// Lazily-compiled analysis of `graph` (built on first `generate_block`).
    /// Shared via `Arc` so `generate_block` hands out cheap references
    /// instead of deep-cloning the compiled node vector per block.
    compiled: Mutex<Option<Arc<CompiledGraph>>>,
    /// Dense scratch for the compiled path.
    compiled_scratch: Mutex<CompiledScratch>,
    /// Optional scaling applied to world coordinates before they're fed into
    /// the graph (mirrors C++ `lod` stride handling). `1.0` is the identity.
    coordinate_scale: f32,
    /// Whether the compiled path may reuse the XZ-prefix (outer-group) results
    /// across Y-slices (upstream `use_xz_caching`). When `false`, every slice
    /// re-evaluates the whole graph. Defaults to `true`.
    use_xz_caching: bool,
}

impl GraphGenerator {
    pub fn new(graph: Graph) -> Self {
        Self {
            graph,
            scratch: Mutex::new(GraphScratch::new()),
            compiled: Mutex::new(None),
            compiled_scratch: Mutex::new(CompiledScratch::new()),
            coordinate_scale: 1.0,
            use_xz_caching: true,
        }
    }

    /// Scales input coordinates by `scale` (useful for LOD: `1 << lod`).
    pub fn with_coordinate_scale(mut self, scale: f32) -> Self {
        self.coordinate_scale = scale;
        self
    }

    /// Enables/disables the XZ-prefix cache of the compiled path (upstream
    /// `use_xz_caching`). Disabling forces full re-evaluation every Y-slice.
    pub fn with_xz_caching(mut self, enabled: bool) -> Self {
        self.use_xz_caching = enabled;
        self
    }

    /// Read-only access to the underlying graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Returns the node id of the first `OutputSdf` node, if any. Used by
    /// tests to assert the graph has at least one output before generation.
    pub fn first_sdf_output(&self) -> Option<GraphNodeId> {
        self.graph
            .nodes()
            .iter()
            .find(|n| matches!(n.kind, NodeKind::OutputSdf { .. }))
            .map(|n| n.id)
    }

    /// Ensure the compiled graph is built (once). Returns a cheap `Arc`
    /// handle so the caller doesn't hold the lock across generation and
    /// doesn't pay a deep clone of the compiled nodes per block. On a
    /// topology error (cycle / dangling port) the legacy path is used
    /// instead.
    fn ensure_compiled(&self) -> Option<Arc<CompiledGraph>> {
        let mut guard = self.compiled.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = CompiledGraph::compile(&self.graph).ok().map(Arc::new);
        }
        guard.clone()
    }
}

impl VoxelGenerator for GraphGenerator {
    fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
        // Prefer the compiled path (C1). Fall back to the legacy path if the
        // graph failed to compile (cycle / dangling port).
        if let Some(compiled) = self.ensure_compiled() {
            let mut cscratch = self
                .compiled_scratch
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            generate_block_with_compiled_graph(
                &compiled,
                input,
                &mut cscratch,
                self.coordinate_scale,
                self.use_xz_caching,
            );
        } else {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            generate_block_with_graph(&self.graph, input, &mut scratch, self.coordinate_scale);
        }
        GenResult::default()
    }

    fn used_channels_mask(&self) -> u32 {
        1 << ChannelId::Sdf.index()
    }
}

/// Free-function form of [`GraphGenerator::generate_block`], exposed so a
/// caller can drive a shared `&Graph` (e.g. behind an `Arc<Mutex<>>`) without
/// going through the trait. The C++ side has the same split (the runtime is
/// independent of the `VoxelGenerator` wrapper).
pub fn generate_block_with_graph(
    graph: &Graph,
    input: VoxelQueryData<'_>,
    scratch: &mut GraphScratch,
    coordinate_scale: f32,
) {
    let size = input.buffer.size();
    let sdf_channel = ChannelId::Sdf.index();
    let lod = input.lod;
    let lod_stride = (1u32 << lod) as f32;

    // Pre-allocated scratch buffers for the per-slice X/Z world coordinates.
    // Reused across Y-slices to avoid reallocation. The slice has
    // `size.x * size.z` voxels (ZXY layout — Y innermost).
    let slice_size = (size.x as usize) * (size.z as usize);
    let mut xs: Vec<f32> = vec![0.0; slice_size];
    let mut zs: Vec<f32> = vec![0.0; slice_size];
    let mut outputs: Vec<(GraphOutput, Vec<f32>)> = Vec::new();

    // Build the X and Z coordinate slices once: they are independent of y
    // (ZXY layout — only `world_y` changes per slice).
    for z in 0..size.z {
        for x in 0..size.x {
            let i = (x as usize) + (z as usize) * (size.x as usize);
            xs[i] = (input.origin_in_voxels.x as f32 + x as f32 * lod_stride) * coordinate_scale;
            zs[i] = (input.origin_in_voxels.z as f32 + z as f32 * lod_stride) * coordinate_scale;
        }
    }

    for y in 0..size.y {
        let world_y = (input.origin_in_voxels.y as f32 + y as f32 * lod_stride) * coordinate_scale;
        let inputs = GraphInputs {
            x: &xs,
            y: world_y,
            z: &zs,
        };
        if graph
            .generate(&inputs, slice_size, scratch, &mut outputs)
            .is_err()
        {
            // Topology error: bail out (matches the C++ behaviour of
            // printing an error and leaving the block at its default).
            return;
        }

        // Copy the first SDF output (if any) into the VoxelBuffer's SDF
        // channel for this slice. The C++ runtime supports multiple outputs;
        // the minimal port merges them by writing only the first.
        if let Some((GraphOutput::Sdf, slice)) = outputs.first() {
            write_sdf_slice(input.buffer, sdf_channel, size, y, slice);
        }
    }

    input.buffer.compress_uniform_channels();
}

/// Compiled-path block generation with XZ-outer-group caching (audit §9.6-C1).
///
/// Equivalent to [`generate_block_with_graph`] but drives a [`CompiledGraph`]:
/// the Y-independent prefix is evaluated once on the first Y-slice and cached
/// across subsequent slices; only the Y-dependent tail re-runs per slice. For
/// terrain graphs (mostly XZ-driven) this avoids recomputing nearly the entire
/// graph on each of the block's Y-slices. Pass `use_xz_caching = false` to
/// force full re-evaluation of every slice (upstream `use_xz_caching = off`).
pub fn generate_block_with_compiled_graph(
    compiled: &CompiledGraph,
    input: VoxelQueryData<'_>,
    scratch: &mut CompiledScratch,
    coordinate_scale: f32,
    use_xz_caching: bool,
) {
    let size = input.buffer.size();
    let sdf_channel = ChannelId::Sdf.index();
    let lod_stride = (1u32 << input.lod) as f32;

    // No output node → leave the buffer at its SDF default.
    if !compiled.nodes().iter().any(|n| n.kind.is_output()) {
        return;
    }

    // C3 (audit §9.6-C3): range-analysis fast-path. Propagate the block's
    // world-coordinate intervals through the compiled graph. If the SDF output
    // doesn't straddle zero, the whole block is uniform (air or solid) and can
    // be filled without per-voxel evaluation — the same mechanism the C++ VM
    // uses to skip interior/air blocks.
    let x_range = crate::math::Interval::new(
        (input.origin_in_voxels.x as f32) * coordinate_scale,
        (input.origin_in_voxels.x as f32 + (size.x - 1) as f32 * lod_stride) * coordinate_scale,
    );
    let y_range = crate::math::Interval::new(
        (input.origin_in_voxels.y as f32) * coordinate_scale,
        (input.origin_in_voxels.y as f32 + (size.y - 1) as f32 * lod_stride) * coordinate_scale,
    );
    let z_range = crate::math::Interval::new(
        (input.origin_in_voxels.z as f32) * coordinate_scale,
        (input.origin_in_voxels.z as f32 + (size.z - 1) as f32 * lod_stride) * coordinate_scale,
    );
    let sdf_range = compiled.analyze_range(x_range, y_range, z_range);
    // Only cull when the SDF is provably a single value everywhere (the graph
    // output is constant over the whole block). Sign-only ranges (min>0 or
    // max<0) are left to per-voxel eval because the actual SDF value may carry
    // information the caller needs (e.g. a distance field), and hard nodes
    // (Noise/Cos/Curve) make the range estimate imprecise. This matches the
    // conservative spirit of C++ culling while avoiding false air/solid fills.
    if sdf_range.is_single_value() {
        use crate::storage::voxel_buffer::real_to_raw_voxel;
        let depth = input.buffer.channel_depth(sdf_channel);
        input
            .buffer
            .clear_channel(sdf_channel, real_to_raw_voxel(sdf_range.min, depth));
        input.buffer.compress_uniform_channels();
        return;
    }

    let slice_size = (size.x as usize) * (size.z as usize);
    // XZ coordinates are identical across Y-slices — build once, reuse.
    let mut xs: Vec<f32> = vec![0.0; slice_size];
    let mut zs: Vec<f32> = vec![0.0; slice_size];
    for z in 0..size.z {
        for x in 0..size.x {
            let i = (x as usize) + (z as usize) * (size.x as usize);
            xs[i] = (input.origin_in_voxels.x as f32 + x as f32 * lod_stride) * coordinate_scale;
            zs[i] = (input.origin_in_voxels.z as f32 + z as f32 * lod_stride) * coordinate_scale;
        }
    }

    let mut outputs: Vec<(GraphOutput, Vec<f32>)> = Vec::new();
    for y in 0..size.y {
        let world_y = (input.origin_in_voxels.y as f32 + y as f32 * lod_stride) * coordinate_scale;
        let inputs = GraphInputs {
            x: &xs,
            y: world_y,
            z: &zs,
        };
        // First slice: full eval. Subsequent slices: XZ-prefix cached (unless
        // the caller disabled the cache).
        compiled.generate_slice(
            &inputs,
            slice_size,
            scratch,
            &mut outputs,
            use_xz_caching && y > 0,
        );
        if let Some((GraphOutput::Sdf, slice)) = outputs.first() {
            write_sdf_slice(input.buffer, sdf_channel, size, y, slice);
        }
    }

    input.buffer.compress_uniform_channels();
}

fn write_sdf_slice(
    buffer: &mut VoxelBuffer,
    channel: usize,
    size: Vector3i,
    y: i32,
    slice: &[f32],
) {
    for z in 0..size.z {
        for x in 0..size.x {
            let i = (x as usize) + (z as usize) * (size.x as usize);
            buffer.set_voxel_f(slice[i], x, y, z, channel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::graph::{GraphPort, NodeKind};
    use crate::math::Vector3i;
    use crate::storage::{ChannelDepth, ChannelId, Compression, VoxelBuffer, VoxelFormat};

    /// Build a graph that computes `sin(x) + 1` and writes the result to the
    /// SDF channel. With a constant offset, every voxel of the resulting
    /// buffer is `>= 1.0`, so the block is fully "outside" any iso-surface.
    fn sin_plus_one_graph() -> Graph {
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        let sin = graph.push(NodeKind::Sin {
            a: Some(GraphPort::new(x)),
        });
        let one = graph.push(NodeKind::Constant(1.0));
        let add = graph.push(NodeKind::Add {
            a: Some(GraphPort::new(sin)),
            b: Some(GraphPort::new(one)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(add)),
        });
        graph
    }

    #[test]
    fn generate_block_writes_sin_plus_one_into_sdf_channel() {
        let graph = sin_plus_one_graph();
        let generator = GraphGenerator::new(graph);

        let mut buffer = VoxelBuffer::with_size(Vector3i::new(4, 2, 4));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut buffer);

        let origin = Vector3i::new(10, 0, 0);
        let _ = generator.generate_block(VoxelQueryData {
            buffer: &mut buffer,
            origin_in_voxels: origin,
            lod: 0,
        });

        // SDF value at (x=10, y=0, z=0) should be sin(10) + 1.
        let expected = (10.0f32).sin() + 1.0;
        let actual = buffer.get_voxel_f(0, 0, 0, ChannelId::Sdf.index());
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {expected}, got {actual}"
        );

        // And at (x=12, y=1, z=2).
        let expected2 = (12.0f32).sin() + 1.0;
        let actual2 = buffer.get_voxel_f(2, 1, 2, ChannelId::Sdf.index());
        assert!(
            (actual2 - expected2).abs() < 1e-4,
            "expected {expected2}, got {actual2}"
        );
    }

    #[test]
    fn generate_block_skips_silently_when_the_graph_has_no_output() {
        let mut graph = Graph::new();
        let _ = graph.push(NodeKind::InputX); // no OutputSdf
        let generator = GraphGenerator::new(graph);

        let mut buffer = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut buffer);

        let _ = generator.generate_block(VoxelQueryData {
            buffer: &mut buffer,
            origin_in_voxels: Vector3i::zero(),
            lod: 0,
        });

        // Buffer stays at the SDF default (SDF_FAR_OUTSIDE).
        assert_eq!(
            buffer.get_voxel_f(0, 0, 0, ChannelId::Sdf.index()),
            crate::storage::voxel_buffer::SDF_FAR_OUTSIDE
        );
    }

    #[test]
    fn coordinate_scale_stretches_input_coordinates() {
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(x)),
        });
        let generator = GraphGenerator::new(graph).with_coordinate_scale(2.0);

        let mut buffer = VoxelBuffer::with_size(Vector3i::new(2, 1, 1));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut buffer);

        let _ = generator.generate_block(VoxelQueryData {
            buffer: &mut buffer,
            origin_in_voxels: Vector3i::new(10, 0, 0),
            lod: 0,
        });

        // X is scaled by 2.0: voxel (0,0,0) gets world X 10*2 = 20.
        assert_eq!(buffer.get_voxel_f(0, 0, 0, ChannelId::Sdf.index()), 20.0);
        // Voxel (1,0,0) gets world X 11*2 = 22.
        assert_eq!(buffer.get_voxel_f(1, 0, 0, ChannelId::Sdf.index()), 22.0);
    }

    #[test]
    fn lod_stride_scales_local_coordinates_without_scaling_origin() {
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(x)),
        });
        let generator = GraphGenerator::new(graph);

        let mut buffer = VoxelBuffer::with_size(Vector3i::new(2, 1, 1));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut buffer);

        let _ = generator.generate_block(VoxelQueryData {
            buffer: &mut buffer,
            origin_in_voxels: Vector3i::new(10, 0, 0),
            lod: 1,
        });

        assert_eq!(buffer.get_voxel_f(0, 0, 0, ChannelId::Sdf.index()), 10.0);
        assert_eq!(buffer.get_voxel_f(1, 0, 0, ChannelId::Sdf.index()), 12.0);
    }

    #[test]
    fn generate_block_compresses_uniform_sdf_output() {
        let mut graph = Graph::new();
        let value = graph.push(NodeKind::Constant(2.0));
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(value)),
        });
        let generator = GraphGenerator::new(graph);

        let mut buffer = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut buffer);

        let _ = generator.generate_block(VoxelQueryData {
            buffer: &mut buffer,
            origin_in_voxels: Vector3i::zero(),
            lod: 0,
        });

        assert_eq!(
            buffer.channel_compression(ChannelId::Sdf.index()),
            Compression::Uniform
        );
        assert!(buffer.is_uniform(ChannelId::Sdf.index()));
        assert_eq!(buffer.get_voxel_f(0, 0, 0, ChannelId::Sdf.index()), 2.0);
    }

    #[test]
    fn used_channels_mask_targets_sdf_only() {
        let graph = Graph::new();
        let generator = GraphGenerator::new(graph);
        assert_eq!(
            generator.used_channels_mask(),
            1u32 << ChannelId::Sdf.index()
        );
    }

    #[test]
    fn first_sdf_output_finds_the_output_node() {
        let graph = sin_plus_one_graph();
        let generator = GraphGenerator::new(graph);
        assert!(generator.first_sdf_output().is_some());

        let mut empty = Graph::new();
        empty.push(NodeKind::InputX);
        let empty_gen = GraphGenerator::new(empty);
        assert!(empty_gen.first_sdf_output().is_none());
    }

    /// `Send + Sync` is required by `VoxelGenerator` so the graph generator
    /// can live behind `Arc<dyn VoxelGenerator>`.
    #[test]
    fn graph_generator_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GraphGenerator>();
    }

    #[test]
    fn compiled_path_xz_cache_produces_consistent_output_across_y_slices() {
        // A pure-XZ graph (InputX * InputZ) on a tall block exercises the
        // XZ-prefix cache: the first Y-slice runs the full graph, subsequent
        // slices reuse the cached prefix. The output must be identical on
        // every Y-slice (the graph has no Y-dependence).
        use crate::storage::VoxelFormat;
        let mut g = crate::generators::graph::Graph::new();
        let x = g.push(NodeKind::InputX);
        let z = g.push(NodeKind::InputZ);
        let mul = g.push(NodeKind::Multiply {
            a: Some(crate::generators::graph::GraphPort { node: x, output: 0 }),
            b: Some(crate::generators::graph::GraphPort { node: z, output: 0 }),
        });
        g.push(NodeKind::OutputSdf {
            a: Some(crate::generators::graph::GraphPort {
                node: mul,
                output: 0,
            }),
        });
        let gen = GraphGenerator::new(g);
        let mut buffer = VoxelBuffer::with_size(Vector3i::new(2, 4, 2));
        VoxelFormat::new().configure_buffer(&mut buffer);
        gen.generate_block(VoxelQueryData {
            buffer: &mut buffer,
            origin_in_voxels: Vector3i::new(1, 0, 1),
            lod: 0,
        });
        // Y=0 and Y=3 slices must match (XZ-only graph).
        let y0 = buffer.get_voxel_f(0, 0, 0, ChannelId::Sdf.index());
        let y3 = buffer.get_voxel_f(0, 3, 0, ChannelId::Sdf.index());
        assert!(
            (y0 - y3).abs() < 1e-5,
            "XZ-only graph should produce identical output on every Y-slice: y0={y0}, y3={y3}"
        );
    }

    #[test]
    fn compiled_path_matches_legacy_for_sin_plus_one() {
        // The sin(x)+1 canary graph through the compiled path must produce the
        // same SDF values as the existing golden test asserts. Uses Bit32 SDF
        // depth (matching the golden test) for full float precision.
        use crate::storage::{ChannelDepth, VoxelFormat};
        let gen = GraphGenerator::new(sin_plus_one_graph());
        let mut buffer = VoxelBuffer::with_size(Vector3i::new(4, 2, 4));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut buffer);
        gen.generate_block(VoxelQueryData {
            buffer: &mut buffer,
            origin_in_voxels: Vector3i::new(10, 0, 0),
            lod: 0,
        });
        let v00 = buffer.get_voxel_f(0, 0, 0, ChannelId::Sdf.index());
        let v21 = buffer.get_voxel_f(2, 1, 2, ChannelId::Sdf.index());
        assert!((v00 - (10.0f32.sin() + 1.0)).abs() < 1e-4, "v00={v00}");
        assert!((v21 - (12.0f32.sin() + 1.0)).abs() < 1e-4, "v21={v21}");
    }

    #[test]
    fn xz_caching_toggle_preserves_output() {
        // `with_xz_caching(false)` must produce the same SDF values as the
        // default (cached) path — it only forces full re-evaluation per
        // Y-slice. Uses a Y-mixed graph (sin(x) + y) so the XZ prefix and the
        // Y-dependent tail are both exercised.
        use crate::storage::{ChannelDepth, VoxelFormat};
        fn make_generator(xz_caching: bool) -> GraphGenerator {
            let mut g = Graph::new();
            let x = g.push(NodeKind::InputX);
            let sin = g.push(NodeKind::Sin {
                a: Some(GraphPort::new(x)),
            });
            let y = g.push(NodeKind::InputY);
            let add = g.push(NodeKind::Add {
                a: Some(GraphPort::new(sin)),
                b: Some(GraphPort::new(y)),
            });
            g.push(NodeKind::OutputSdf {
                a: Some(GraphPort::new(add)),
            });
            GraphGenerator::new(g).with_xz_caching(xz_caching)
        }
        let sample = |xz_caching: bool| {
            let mut buffer = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
            let mut format = VoxelFormat::new();
            format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
            format.configure_buffer(&mut buffer);
            make_generator(xz_caching).generate_block(VoxelQueryData {
                buffer: &mut buffer,
                origin_in_voxels: Vector3i::new(10, 20, 30),
                lod: 0,
            });
            (0..4)
                .map(|i| buffer.get_voxel_f(i, i, i, ChannelId::Sdf.index()))
                .collect::<Vec<_>>()
        };
        let cached = sample(true);
        let uncached = sample(false);
        assert_eq!(cached.len(), uncached.len());
        for (a, b) in cached.iter().zip(uncached.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "xz caching toggle changed output: {a} vs {b}"
            );
        }
        // Sanity: values actually vary along the diagonal.
        assert!(cached.iter().any(|v| (*v - cached[0]).abs() > 1e-4));
    }

    #[test]
    fn c3_range_analysis_culls_uniform_sdf_block() {
        // Constant(2.0) → OutputSdf: the SDF is +2 everywhere → fully air.
        // C3 range analysis should detect this and fill uniformly WITHOUT
        // per-voxel eval (the channel ends up Compression::Uniform).
        use crate::generators::graph::{Graph, GraphPort, NodeKind};
        use crate::storage::{ChannelDepth, Compression, VoxelFormat};
        let mut g = Graph::new();
        let c = g.push(NodeKind::Constant(2.0));
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort { node: c, output: 0 }),
        });
        let gen = GraphGenerator::new(g);
        let mut buffer = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut buffer);
        gen.generate_block(VoxelQueryData {
            buffer: &mut buffer,
            origin_in_voxels: Vector3i::new(10, 0, 0),
            lod: 0,
        });
        // The C3 fast-path should have compressed the channel to uniform.
        assert_eq!(
            buffer.channel_compression(ChannelId::Sdf.index()),
            Compression::Uniform,
            "constant-positive SDF should be culled to uniform air"
        );
        // And the value should be the graph's actual SDF (2.0), not a sentinel,
        // since the conservative single-value fast-path fills the real value.
        let v = buffer.get_voxel_f(0, 0, 0, ChannelId::Sdf.index());
        assert!(
            (v - 2.0).abs() < 0.5,
            "culled uniform block should hold the graph's SDF value 2.0, got {v}"
        );
    }

    #[test]
    fn c3_range_analysis_culls_uniform_solid_block() {
        // Constant(-2.0) → OutputSdf: the SDF is -2 everywhere → fully solid.
        // C3 range analysis detects the single-value output and fills it
        // uniformly WITHOUT per-voxel eval (Compression::Uniform).
        use crate::generators::graph::{Graph, GraphPort, NodeKind};
        use crate::storage::{ChannelDepth, Compression, VoxelFormat};
        let mut g = Graph::new();
        let c = g.push(NodeKind::Constant(-2.0));
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort { node: c, output: 0 }),
        });
        let gen = GraphGenerator::new(g);
        let mut buffer = VoxelBuffer::with_size(Vector3i::new(4, 4, 4));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut buffer);
        gen.generate_block(VoxelQueryData {
            buffer: &mut buffer,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });
        assert_eq!(
            buffer.channel_compression(ChannelId::Sdf.index()),
            Compression::Uniform,
            "constant-negative SDF should be culled to uniform solid"
        );
        let v = buffer.get_voxel_f(0, 0, 0, ChannelId::Sdf.index());
        assert!(
            (v - (-2.0)).abs() < 0.5,
            "should hold the graph's SDF value -2.0, got {v}"
        );
    }
}
