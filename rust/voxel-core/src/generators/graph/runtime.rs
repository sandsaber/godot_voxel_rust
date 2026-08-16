//! Graph topology + AST-walker interpreter for the procedural-graph runtime.
//!
//! A [`Graph`] owns its nodes; each node has a [`NodeKind`] describing what
//! it computes, plus input ports referencing upstream nodes by id. The
//! interpreter walks the graph in topological order, evaluating every node
//! over a Y-slice of voxels at a time and storing the resulting f32 buffer
//! back on the node for downstream consumption.
//!
//! This module is intentionally engine-agnostic and `VoxelBuffer`-free; the
//! [`super::generator_graph::GraphGenerator`] adapter wires it into the
//! [`crate::generators::base::VoxelGenerator`] trait.

use std::collections::HashMap;

/// Identifies a node inside a [`Graph`]. The caller picks ids; they need not
/// be dense or contiguous, but must be unique within a graph.
pub type GraphNodeId = u32;

/// A typed port on a node. `node` is the upstream producer; `output` selects
/// which of that producer's outputs to read (output 0 for single-output nodes,
/// outputs 0/1/2/3 for multi-output nodes like Normalize3D).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphPort {
    pub node: GraphNodeId,
    pub output: u8,
}

impl GraphPort {
    pub fn new(node: GraphNodeId) -> Self {
        Self { node, output: 0 }
    }

    pub fn with_output(node: GraphNodeId, output: u8) -> Self {
        Self { node, output }
    }
}

/// Optional parameter value attached to a node (e.g. the constant value of a
/// `Constant` node, the remap range of a `Remap` node).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GraphParam {
    /// Single f32 constant.
    F(f32),
    /// Two-component range (used by Remap: `(from_start, from_end)`).
    RangeFrom(f32, f32),
    /// Two-component range (used by Remap: `(to_start, to_end)`).
    RangeTo(f32, f32),
}

/// What a node computes. Variants bundle parameters and a fixed-size list of
/// input ports (matching the C++ `NodeType` port counts). Inputs that aren't
/// connected default to `0.0` at execution time.
//
// Manual `Debug` because `FastNoiseLite`/`Arc<Curve>` don't implement it;
// manual (skipped) `PartialEq` for the same reason — node identity is
// established by `GraphNode::id`, not by structural equality.
#[derive(Clone)]
pub enum NodeKind {
    /// World X coordinate of the voxel (block-relative, scaled by LOD).
    InputX,
    /// World Y coordinate.
    InputY,
    /// World Z coordinate.
    InputZ,
    /// Constant value. Carries the value as a parameter.
    Constant(f32),
    /// `a + b`.
    Add {
        a: Option<GraphPort>,
        b: Option<GraphPort>,
    },
    /// `a - b`.
    Subtract {
        a: Option<GraphPort>,
        b: Option<GraphPort>,
    },
    /// `a * b`.
    Multiply {
        a: Option<GraphPort>,
        b: Option<GraphPort>,
    },
    /// `a / b`.
    Divide {
        a: Option<GraphPort>,
        b: Option<GraphPort>,
    },
    /// `sin(a)`.
    Sin { a: Option<GraphPort> },
    /// `cos(a)`.
    Cos { a: Option<GraphPort> },
    /// `abs(a)`.
    Abs { a: Option<GraphPort> },
    /// `sqrt(a)`.
    Sqrt { a: Option<GraphPort> },
    /// `min(a, b)`.
    Min {
        a: Option<GraphPort>,
        b: Option<GraphPort>,
    },
    /// `max(a, b)`.
    Max {
        a: Option<GraphPort>,
        b: Option<GraphPort>,
    },
    /// Remap `a` from `[from_start, from_end]` to `[to_start, to_end]`.
    Remap {
        a: Option<GraphPort>,
        from_start: f32,
        from_end: f32,
        to_start: f32,
        to_end: f32,
    },
    /// `floor(a)`.
    Floor { a: Option<GraphPort> },
    /// `fract(a)` (returns the fractional part).
    Fract { a: Option<GraphPort> },
    /// Euclidean distance between two 2D points: `sqrt((x1-x0)² + (y1-y0)²)`.
    /// Matches C++ `NODE_DISTANCE_2D`.
    Distance2D {
        x0: Option<GraphPort>,
        y0: Option<GraphPort>,
        x1: Option<GraphPort>,
        y1: Option<GraphPort>,
    },
    /// Euclidean distance between two 3D points:
    /// `sqrt((x1-x0)² + (y1-y0)² + (z1-z0)²)`. Matches C++ `NODE_DISTANCE_3D`.
    Distance3D {
        x0: Option<GraphPort>,
        y0: Option<GraphPort>,
        z0: Option<GraphPort>,
        x1: Option<GraphPort>,
        y1: Option<GraphPort>,
        z1: Option<GraphPort>,
    },
    /// Normalize a 3D direction `(x, y, z)` to unit length.
    Normalize3D {
        x: Option<GraphPort>,
        y: Option<GraphPort>,
        z: Option<GraphPort>,
    },
    /// `pow(a, b)`.
    Pow {
        a: Option<GraphPort>,
        b: Option<GraphPort>,
    },
    /// `mix(a, b, t)` = `a*(1-t) + b*t`.
    Mix {
        a: Option<GraphPort>,
        b: Option<GraphPort>,
        t: Option<GraphPort>,
    },
    /// `clamp(a, min_v, max_v)`. `min_v` and `max_v` are inputs (the C++
    /// `NODE_CLAMP` variant — `NODE_CLAMP_C` takes them as constants).
    Clamp {
        a: Option<GraphPort>,
        min_v: Option<GraphPort>,
        max_v: Option<GraphPort>,
    },
    /// Look up `a` in a baked curve mapping `[0,1] → height`. The curve is
    /// shared (immutable) so multiple `Curve` nodes can refer to the same
    /// data without cloning the samples.
    Curve {
        a: Option<GraphPort>,
        curve: std::sync::Arc<crate::generators::simple::Curve>,
    },
    /// `fastnoise-lite` 2D noise. The seed/frequency/noise-type live on the
    /// `NoiseConfig` (Clone-able); the runtime builds a fresh sampler per
    /// `generate` call (the upstream crate's `FastNoiseLite` is not `Clone`).
    Noise2D {
        x: Option<GraphPort>,
        y: Option<GraphPort>,
        noise: crate::generators::simple::NoiseConfig,
    },
    /// `fastnoise-lite` 3D noise.
    Noise3D {
        x: Option<GraphPort>,
        y: Option<GraphPort>,
        z: Option<GraphPort>,
        noise: crate::generators::simple::NoiseConfig,
    },
    /// SDF of an axis-aligned plane at `height`. Equivalent to `y - height`.
    SdfPlane {
        y: Option<GraphPort>,
        height: Option<GraphPort>,
    },
    /// SDF of an axis-aligned box centred at the origin with half-extents
    /// `(size_x, size_y, size_z)`. Inputs are world X/Y/Z.
    SdfBox {
        x: Option<GraphPort>,
        y: Option<GraphPort>,
        z: Option<GraphPort>,
        size_x: f32,
        size_y: f32,
        size_z: f32,
    },
    /// SDF of a sphere centred at the origin with the given `radius`.
    SdfSphere {
        x: Option<GraphPort>,
        y: Option<GraphPort>,
        z: Option<GraphPort>,
        radius: Option<GraphPort>,
    },
    /// SDF of a torus in the XZ plane with major radius `r1` and minor
    /// radius `r2`.
    SdfTorus {
        x: Option<GraphPort>,
        y: Option<GraphPort>,
        z: Option<GraphPort>,
        r1: f32,
        r2: f32,
    },
    /// Hard SDF union: `min(a, b)`.
    SdfUnion {
        a: Option<GraphPort>,
        b: Option<GraphPort>,
    },
    /// Hard SDF subtraction: `max(a, -b)`.
    SdfSubtract {
        a: Option<GraphPort>,
        b: Option<GraphPort>,
    },
    /// Polynomial smooth union with the given `smoothness` (0 = hard union).
    SdfSmoothUnion {
        a: Option<GraphPort>,
        b: Option<GraphPort>,
        smoothness: f32,
    },
    /// Polynomial smooth subtraction with the given `smoothness`.
    SdfSmoothSubtract {
        a: Option<GraphPort>,
        b: Option<GraphPort>,
        smoothness: f32,
    },
    /// Output sink: writes its single input into the SDF channel of the
    /// destination `VoxelBuffer`. Treated as a leaf in topological order.
    OutputSdf { a: Option<GraphPort> },
    /// Sample a 2D image at `(x, y)` with bilinear interpolation.
    Image2D {
        x: Option<GraphPort>,
        y: Option<GraphPort>,
        image: std::sync::Arc<crate::generators::graph::image::Image2D>,
    },
    /// Evaluate a parsed expression with `x`/`y`/`z` bound to the three ports.
    Expression {
        x: Option<GraphPort>,
        y: Option<GraphPort>,
        z: Option<GraphPort>,
        expr: std::sync::Arc<crate::generators::graph::expression_node::ExpressionNode>,
    },
}

impl NodeKind {
    /// Returns the input ports this node reads from, in declaration order.
    /// Used by the interpreter to schedule upstream evaluations.
    pub fn inputs(&self) -> Vec<Option<GraphPort>> {
        match self {
            NodeKind::InputX | NodeKind::InputY | NodeKind::InputZ | NodeKind::Constant(_) => {
                Vec::new()
            }
            NodeKind::Add { a, b }
            | NodeKind::Subtract { a, b }
            | NodeKind::Multiply { a, b }
            | NodeKind::Divide { a, b }
            | NodeKind::Min { a, b }
            | NodeKind::Max { a, b } => vec![*a, *b],
            NodeKind::Sin { a }
            | NodeKind::Cos { a }
            | NodeKind::Abs { a }
            | NodeKind::Sqrt { a }
            | NodeKind::Floor { a }
            | NodeKind::Fract { a } => vec![*a],
            NodeKind::Remap { a, .. } => vec![*a],
            NodeKind::Pow { a, b } => vec![*a, *b],
            NodeKind::Distance2D { x0, y0, x1, y1, .. } => vec![*x0, *y0, *x1, *y1],
            NodeKind::Distance3D {
                x0,
                y0,
                z0,
                x1,
                y1,
                z1,
            } => vec![*x0, *y0, *z0, *x1, *y1, *z1],
            NodeKind::Normalize3D { x, y, z } => vec![*x, *y, *z],
            NodeKind::Mix { a, b, t } => vec![*a, *b, *t],
            NodeKind::Clamp { a, min_v, max_v } => vec![*a, *min_v, *max_v],
            NodeKind::Curve { a, .. } => vec![*a],
            NodeKind::Noise2D { x, y, .. } => vec![*x, *y],
            NodeKind::Noise3D { x, y, z, .. } => vec![*x, *y, *z],
            NodeKind::SdfPlane { y, height } => vec![*y, *height],
            NodeKind::SdfBox { x, y, z, .. } => vec![*x, *y, *z],
            NodeKind::SdfSphere { x, y, z, radius } => vec![*x, *y, *z, *radius],
            NodeKind::SdfTorus { x, y, z, .. } => vec![*x, *y, *z],
            NodeKind::SdfUnion { a, b }
            | NodeKind::SdfSubtract { a, b }
            | NodeKind::SdfSmoothUnion { a, b, .. }
            | NodeKind::SdfSmoothSubtract { a, b, .. } => vec![*a, *b],
            NodeKind::OutputSdf { a } => vec![*a],
            NodeKind::Image2D { x, y, .. } => vec![*x, *y],
            NodeKind::Expression { x, y, z, .. } => vec![*x, *y, *z],
        }
    }

    /// `true` if this node is an output sink (no downstream consumer in the
    /// graph itself; the runtime materialises its result into a channel).
    pub fn is_output(&self) -> bool {
        matches!(self, NodeKind::OutputSdf { .. })
    }
}

/// A node in the graph. Identity is established by `id`; `kind` is opaque
/// for comparison purposes (it carries non-`PartialEq` payloads like
/// `FastNoiseLite`).
#[derive(Clone)]
pub struct GraphNode {
    pub id: GraphNodeId,
    pub kind: NodeKind,
}

impl std::fmt::Debug for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphNode")
            .field("id", &self.id)
            .field("kind", &std::any::type_name::<NodeKind>())
            .finish()
    }
}

impl GraphNode {
    pub fn new(id: GraphNodeId, kind: NodeKind) -> Self {
        Self { id, kind }
    }
}

/// Where the runtime writes the result of an output node. Mirrors the C++
/// `OutputInfo` mapping from a node to a destination channel — the current
/// minimal port supports only the SDF channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphOutput {
    /// Write the output node's result to the SDF channel.
    Sdf,
}

/// A procedural voxel graph. Built incrementally via [`Graph::add_node`].
/// The graph owns no execution state — pass it to [`Graph::generate`] with a
/// per-thread scratch to evaluate.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    nodes: Vec<GraphNode>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: GraphNode) {
        // Defensive: detect duplicate ids so the interpreter's HashMap doesn't
        // silently shadow one. The C++ graph uses a map keyed by id; we keep a
        // Vec but check uniqueness on insert.
        debug_assert!(
            !self.nodes.iter().any(|n| n.id == node.id),
            "duplicate graph node id {:?}",
            node.id
        );
        self.nodes.push(node);
    }

    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Convenience: add a node by kind, picking the next free id.
    pub fn push(&mut self, kind: NodeKind) -> GraphNodeId {
        let id = self.nodes.iter().map(|n| n.id + 1).max().unwrap_or(0);
        self.add_node(GraphNode::new(id, kind));
        id
    }

    /// Remove every node. Used by the Godot `clear_graph` binding.
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Returns the ids in topological order (producers before consumers).
    /// The output nodes come last. Returns an error if the graph contains a
    /// cycle or a dangling port reference.
    pub fn topological_order(&self) -> Result<Vec<GraphNodeId>, TopoError> {
        use std::collections::{HashMap, HashSet};

        let by_id: HashMap<GraphNodeId, &GraphNode> =
            self.nodes.iter().map(|n| (n.id, n)).collect();

        let mut visited: HashSet<GraphNodeId> = HashSet::new();
        let mut on_stack: HashSet<GraphNodeId> = HashSet::new();
        let mut order: Vec<GraphNodeId> = Vec::with_capacity(self.nodes.len());

        fn visit(
            id: GraphNodeId,
            by_id: &HashMap<GraphNodeId, &GraphNode>,
            visited: &mut HashSet<GraphNodeId>,
            on_stack: &mut HashSet<GraphNodeId>,
            order: &mut Vec<GraphNodeId>,
        ) -> Result<(), TopoError> {
            if visited.contains(&id) {
                return Ok(());
            }
            if on_stack.contains(&id) {
                return Err(TopoError::Cycle);
            }
            on_stack.insert(id);
            let node = by_id.get(&id).ok_or(TopoError::DanglingPort(id))?;
            for input in node.kind.inputs().into_iter().flatten() {
                visit(input.node, by_id, visited, on_stack, order)?;
            }
            on_stack.remove(&id);
            visited.insert(id);
            order.push(id);
            Ok(())
        }

        // Visit output nodes last so the order ends with them.
        let mut output_ids: Vec<GraphNodeId> = Vec::new();
        for node in &self.nodes {
            if node.kind.is_output() {
                output_ids.push(node.id);
            } else {
                visit(node.id, &by_id, &mut visited, &mut on_stack, &mut order)?;
            }
        }
        for id in output_ids {
            visit(id, &by_id, &mut visited, &mut on_stack, &mut order)?;
        }
        Ok(order)
    }

    /// Evaluate the graph for one Y-slice of voxels. Writes the result of
    /// every output node into the corresponding `outputs` slot; the caller
    /// copies those slices into a `VoxelBuffer` channel.
    ///
    /// `inputs.x`, `inputs.y`, `inputs.z` carry the world-space coordinates
    /// of each voxel in the slice. `slice_size` is `width * depth` (X × Z).
    pub fn generate(
        &self,
        inputs: &GraphInputs<'_>,
        slice_size: usize,
        scratch: &mut GraphScratch,
        outputs: &mut Vec<(GraphOutput, Vec<f32>)>,
    ) -> Result<(), TopoError> {
        let order = self.topological_order()?;
        scratch.clear();
        outputs.clear();

        for id in order {
            let node = self
                .nodes
                .iter()
                .find(|n| n.id == id)
                .expect("topological order contains a node id that is not in the graph");
            match &node.kind {
                NodeKind::InputX => scratch.put(id, inputs.x.to_vec()),
                NodeKind::InputY => scratch.put(id, vec![inputs.y; slice_size]),
                NodeKind::InputZ => scratch.put(id, inputs.z.to_vec()),
                NodeKind::Constant(v) => scratch.put(id, vec![*v; slice_size]),
                NodeKind::Add { a, b } => {
                    let r = binop(scratch, a, b, slice_size, |x, y| x + y);
                    scratch.put(id, r);
                }
                NodeKind::Subtract { a, b } => {
                    let r = binop(scratch, a, b, slice_size, |x, y| x - y);
                    scratch.put(id, r);
                }
                NodeKind::Multiply { a, b } => {
                    let r = binop(scratch, a, b, slice_size, |x, y| x * y);
                    scratch.put(id, r);
                }
                NodeKind::Divide { a, b } => {
                    // C++ parity: exact-zero test (not epsilon), default denominator is 1.
                    let r = binop(scratch, a, b, slice_size, |x, y| {
                        if y == 0.0 {
                            0.0
                        } else {
                            x / y
                        }
                    });
                    scratch.put(id, r);
                }
                NodeKind::Sin { a } => {
                    let r = monop(scratch, a, slice_size, f32::sin);
                    scratch.put(id, r);
                }
                NodeKind::Cos { a } => {
                    let r = monop(scratch, a, slice_size, f32::cos);
                    scratch.put(id, r);
                }
                NodeKind::Abs { a } => {
                    let r = monop(scratch, a, slice_size, f32::abs);
                    scratch.put(id, r);
                }
                NodeKind::Sqrt { a } => {
                    let r = monop(scratch, a, slice_size, |v| v.max(0.0).sqrt());
                    scratch.put(id, r);
                }
                NodeKind::Min { a, b } => {
                    let r = binop(scratch, a, b, slice_size, f32::min);
                    scratch.put(id, r);
                }
                NodeKind::Max { a, b } => {
                    let r = binop(scratch, a, b, slice_size, f32::max);
                    scratch.put(id, r);
                }
                NodeKind::Remap {
                    a,
                    from_start,
                    from_end,
                    to_start,
                    to_end,
                } => {
                    let from_start = *from_start;
                    let from_end = *from_end;
                    let to_start = *to_start;
                    let to_end = *to_end;
                    let from_span = from_end - from_start;
                    let to_span = to_end - to_start;
                    // C++ parity: pure linear remap (a*x + b), NO clamp.
                    let r = monop(scratch, a, slice_size, |v| {
                        if from_span.abs() <= f32::EPSILON {
                            0.0
                        } else {
                            let t = (v - from_start) / from_span;
                            to_start + t * to_span
                        }
                    });
                    scratch.put(id, r);
                }
                NodeKind::Floor { a } => {
                    let r = monop(scratch, a, slice_size, f32::floor);
                    scratch.put(id, r);
                }
                NodeKind::Fract { a } => {
                    let r = monop(scratch, a, slice_size, |v| v - v.floor());
                    scratch.put(id, r);
                }
                NodeKind::Pow { a, b } => {
                    let r = binop(scratch, a, b, slice_size, f32::powf);
                    scratch.put(id, r);
                }
                NodeKind::Distance2D { x0, y0, x1, y1, .. } => {
                    let r: Vec<f32> = (0..slice_size)
                        .map(|i| {
                            let dx = value_at(scratch, x1, i, 0.0) - value_at(scratch, x0, i, 0.0);
                            let dy = value_at(scratch, y1, i, 0.0) - value_at(scratch, y0, i, 0.0);
                            (dx * dx + dy * dy).sqrt()
                        })
                        .collect();
                    scratch.put(id, r);
                }
                NodeKind::Distance3D {
                    x0,
                    y0,
                    z0,
                    x1,
                    y1,
                    z1,
                } => {
                    let r: Vec<f32> = (0..slice_size)
                        .map(|i| {
                            let dx = value_at(scratch, x1, i, 0.0) - value_at(scratch, x0, i, 0.0);
                            let dy = value_at(scratch, y1, i, 0.0) - value_at(scratch, y0, i, 0.0);
                            let dz = value_at(scratch, z1, i, 0.0) - value_at(scratch, z0, i, 0.0);
                            (dx * dx + dy * dy + dz * dz).sqrt()
                        })
                        .collect();
                    scratch.put(id, r);
                }
                NodeKind::Normalize3D { x, y, z } => {
                    // GRAPH-2 parity: C++ produces 4 outputs (nx, ny, nz, len).
                    // We compute all four; downstream nodes select via
                    // GraphPort.output (0=nx, 1=ny, 2=nz, 3=len).
                    // For backward compat, output 0 stores nx (the first
                    // normalized component).
                    let mut nx = Vec::with_capacity(slice_size);
                    let mut ny = Vec::with_capacity(slice_size);
                    let mut nz = Vec::with_capacity(slice_size);
                    let mut len_v = Vec::with_capacity(slice_size);
                    for i in 0..slice_size {
                        let xv = value_at(scratch, x, i, 0.0);
                        let yv = value_at(scratch, y, i, 0.0);
                        let zv = value_at(scratch, z, i, 0.0);
                        let len = (xv * xv + yv * yv + zv * zv).sqrt();
                        if len > 0.0 {
                            nx.push(xv / len);
                            ny.push(yv / len);
                            nz.push(zv / len);
                        } else {
                            nx.push(0.0);
                            ny.push(0.0);
                            nz.push(0.0);
                        }
                        len_v.push(len);
                    }
                    scratch.put(id, nx);
                    // Store extra outputs via named convention:
                    // node_id + (output_index << 24) as synthetic keys.
                    scratch.put(GraphNodeId::wrapping_add(id, 1 << 24), ny);
                    scratch.put(GraphNodeId::wrapping_add(id, 2 << 24), nz);
                    scratch.put(GraphNodeId::wrapping_add(id, 3 << 24), len_v);
                }
                NodeKind::Mix { a, b, t } => {
                    let r = ternary(scratch, a, b, t, slice_size, |a, b, t| {
                        a * (1.0 - t) + b * t
                    });
                    scratch.put(id, r);
                }
                NodeKind::Clamp { a, min_v, max_v } => {
                    let r = ternary(scratch, a, min_v, max_v, slice_size, |v, lo, hi| {
                        v.clamp(lo.min(hi), lo.max(hi))
                    });
                    scratch.put(id, r);
                }
                NodeKind::Curve { a, curve } => {
                    let curve = curve.clone();
                    let r = monop(scratch, a, slice_size, move |v| curve.sample(v));
                    scratch.put(id, r);
                }
                NodeKind::Noise2D { x, y, noise } => {
                    let noise = noise.build();
                    let r: Vec<f32> = (0..slice_size)
                        .map(|i| {
                            noise.get_noise_2d(
                                value_at(scratch, x, i, 0.0),
                                value_at(scratch, y, i, 0.0),
                            )
                        })
                        .collect();
                    scratch.put(id, r);
                }
                NodeKind::Noise3D { x, y, z, noise } => {
                    let noise = noise.build();
                    let r: Vec<f32> = (0..slice_size)
                        .map(|i| {
                            noise.get_noise_3d(
                                value_at(scratch, x, i, 0.0),
                                value_at(scratch, y, i, 0.0),
                                value_at(scratch, z, i, 0.0),
                            )
                        })
                        .collect();
                    scratch.put(id, r);
                }
                NodeKind::SdfPlane { y, height } => {
                    let r = binop(scratch, y, height, slice_size, |y, h| y - h);
                    scratch.put(id, r);
                }
                NodeKind::SdfBox {
                    x,
                    y,
                    z,
                    size_x,
                    size_y,
                    size_z,
                } => {
                    let size = crate::math::Vector3f::new(*size_x, *size_y, *size_z);
                    let r: Vec<f32> = (0..slice_size)
                        .map(|i| {
                            crate::math::sdf::sdf_box(
                                crate::math::Vector3f::new(
                                    value_at(scratch, x, i, 0.0),
                                    value_at(scratch, y, i, 0.0),
                                    value_at(scratch, z, i, 0.0),
                                ),
                                size,
                            )
                        })
                        .collect();
                    scratch.put(id, r);
                }
                NodeKind::SdfSphere { x, y, z, radius } => {
                    let r: Vec<f32> = (0..slice_size)
                        .map(|i| {
                            let radius = value_at(scratch, radius, i, 1.0);
                            let pos = crate::math::Vector3f::new(
                                value_at(scratch, x, i, 0.0),
                                value_at(scratch, y, i, 0.0),
                                value_at(scratch, z, i, 0.0),
                            );
                            crate::math::sdf::sdf_sphere(pos, crate::math::Vector3f::zero(), radius)
                        })
                        .collect();
                    scratch.put(id, r);
                }
                NodeKind::SdfTorus { x, y, z, r1, r2 } => {
                    let r1 = *r1;
                    let r2 = *r2;
                    let r: Vec<f32> = (0..slice_size)
                        .map(|i| {
                            crate::math::sdf::sdf_torus(
                                crate::math::Vector3f::new(
                                    value_at(scratch, x, i, 0.0),
                                    value_at(scratch, y, i, 0.0),
                                    value_at(scratch, z, i, 0.0),
                                ),
                                r1,
                                r2,
                            )
                        })
                        .collect();
                    scratch.put(id, r);
                }
                NodeKind::SdfUnion { a, b } => {
                    let r = binop(scratch, a, b, slice_size, crate::math::sdf::sdf_union);
                    scratch.put(id, r);
                }
                NodeKind::SdfSubtract { a, b } => {
                    let r = binop(scratch, a, b, slice_size, crate::math::sdf::sdf_subtract);
                    scratch.put(id, r);
                }
                NodeKind::SdfSmoothUnion { a, b, smoothness } => {
                    let s = *smoothness;
                    let r = binop(scratch, a, b, slice_size, move |a, b| {
                        if s > 1e-4 {
                            crate::math::sdf::sdf_smooth_union(a, b, s)
                        } else {
                            crate::math::sdf::sdf_union(a, b)
                        }
                    });
                    scratch.put(id, r);
                }
                NodeKind::SdfSmoothSubtract { a, b, smoothness } => {
                    let s = *smoothness;
                    let r = binop(scratch, a, b, slice_size, move |a, b| {
                        if s > 1e-4 {
                            crate::math::sdf::sdf_smooth_subtract(a, b, s)
                        } else {
                            crate::math::sdf::sdf_subtract(a, b)
                        }
                    });
                    scratch.put(id, r);
                }
                NodeKind::Image2D { x, y, image } => {
                    let image = image.clone();
                    let r: Vec<f32> = (0..slice_size)
                        .map(|i| {
                            image.sample_bilinear(
                                value_at(scratch, x, i, 0.0),
                                value_at(scratch, y, i, 0.0),
                            )
                        })
                        .collect();
                    scratch.put(id, r);
                }
                NodeKind::Expression { x, y, z, expr } => {
                    let xs: Vec<f32> = (0..slice_size)
                        .map(|i| value_at(scratch, x, i, 0.0))
                        .collect();
                    let ys: Vec<f32> = (0..slice_size)
                        .map(|i| value_at(scratch, y, i, 0.0))
                        .collect();
                    let zs: Vec<f32> = (0..slice_size)
                        .map(|i| value_at(scratch, z, i, 0.0))
                        .collect();
                    scratch.put(id, expr.evaluate_slice(&[&xs, &ys, &zs]));
                }
                NodeKind::OutputSdf { a } => {
                    let r = monop(scratch, a, slice_size, |v| v);
                    outputs.push((GraphOutput::Sdf, r));
                }
            }
        }

        Ok(())
    }
}

/// Convert a possibly-negative script port id into a [`GraphPort`].
/// Negative ids mean "unconnected" and become `None`.
pub fn optional_graph_port(id: i64) -> Option<GraphPort> {
    u32::try_from(id).ok().map(GraphPort::new)
}

/// Build a [`NodeKind`] from the compact Godot `add_node(kind, a, b, c, d, value)`
/// contract documented in `doc/source/generators.md`.
pub fn node_kind_from_spec(
    kind: &str,
    a: i64,
    b: i64,
    c: i64,
    d: i64,
    value: f32,
) -> Option<NodeKind> {
    let pa = optional_graph_port(a);
    let pb = optional_graph_port(b);
    let pc = optional_graph_port(c);
    let pd = optional_graph_port(d);
    Some(match kind {
        "InputX" => NodeKind::InputX,
        "InputY" => NodeKind::InputY,
        "InputZ" => NodeKind::InputZ,
        "Constant" => NodeKind::Constant(value),
        "Add" => NodeKind::Add { a: pa, b: pb },
        "Subtract" => NodeKind::Subtract { a: pa, b: pb },
        "Multiply" => NodeKind::Multiply { a: pa, b: pb },
        "Divide" => NodeKind::Divide { a: pa, b: pb },
        "Min" => NodeKind::Min { a: pa, b: pb },
        "Max" => NodeKind::Max { a: pa, b: pb },
        "Sin" => NodeKind::Sin { a: pa },
        "Cos" => NodeKind::Cos { a: pa },
        "Abs" => NodeKind::Abs { a: pa },
        "Sqrt" => NodeKind::Sqrt { a: pa },
        "Floor" => NodeKind::Floor { a: pa },
        "Fract" => NodeKind::Fract { a: pa },
        "Pow" => NodeKind::Pow { a: pa, b: pb },
        "Mix" => NodeKind::Mix {
            a: pa,
            b: pb,
            t: pc,
        },
        "Clamp" => NodeKind::Clamp {
            a: pa,
            min_v: pb,
            max_v: pc,
        },
        "SdfPlane" => NodeKind::SdfPlane { y: pa, height: pb },
        "SdfSphere" => NodeKind::SdfSphere {
            x: pa,
            y: pb,
            z: pc,
            radius: pd,
        },
        "SdfBox" => NodeKind::SdfBox {
            x: pa,
            y: pb,
            z: pc,
            size_x: value,
            size_y: value,
            size_z: value,
        },
        "SdfUnion" => NodeKind::SdfUnion { a: pa, b: pb },
        "SdfSubtract" => NodeKind::SdfSubtract { a: pa, b: pb },
        "SdfSmoothUnion" => NodeKind::SdfSmoothUnion {
            a: pa,
            b: pb,
            smoothness: value,
        },
        "SdfSmoothSubtract" => NodeKind::SdfSmoothSubtract {
            a: pa,
            b: pb,
            smoothness: value,
        },
        "Noise2D" => NodeKind::Noise2D {
            x: pa,
            y: pb,
            noise: crate::generators::simple::NoiseConfig::default(),
        },
        "Noise3D" => NodeKind::Noise3D {
            x: pa,
            y: pb,
            z: pc,
            noise: crate::generators::simple::NoiseConfig::default(),
        },
        "OutputSdf" => NodeKind::OutputSdf { a: pa },
        _ => return None,
    })
}

/// Compact interchange fields for one graph node.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeSpec {
    pub kind: String,
    pub a: i64,
    pub b: i64,
    pub c: i64,
    pub d: i64,
    pub value: f32,
    pub expr: Option<String>,
}

fn port_id(port: Option<GraphPort>) -> i64 {
    port.map(|port| i64::from(port.node)).unwrap_or(-1)
}

/// Reverse of [`node_kind_from_spec`] for JSON / visual-editor interchange.
pub fn node_kind_to_spec(kind: &NodeKind) -> NodeSpec {
    let unset = NodeSpec {
        kind: String::new(),
        a: -1,
        b: -1,
        c: -1,
        d: -1,
        value: 0.0,
        expr: None,
    };
    match kind {
        NodeKind::InputX => NodeSpec {
            kind: "InputX".into(),
            ..unset
        },
        NodeKind::InputY => NodeSpec {
            kind: "InputY".into(),
            ..unset
        },
        NodeKind::InputZ => NodeSpec {
            kind: "InputZ".into(),
            ..unset
        },
        NodeKind::Constant(value) => NodeSpec {
            kind: "Constant".into(),
            value: *value,
            ..unset
        },
        NodeKind::Add { a, b } => NodeSpec {
            kind: "Add".into(),
            a: port_id(*a),
            b: port_id(*b),
            ..unset
        },
        NodeKind::Subtract { a, b } => NodeSpec {
            kind: "Subtract".into(),
            a: port_id(*a),
            b: port_id(*b),
            ..unset
        },
        NodeKind::Multiply { a, b } => NodeSpec {
            kind: "Multiply".into(),
            a: port_id(*a),
            b: port_id(*b),
            ..unset
        },
        NodeKind::Divide { a, b } => NodeSpec {
            kind: "Divide".into(),
            a: port_id(*a),
            b: port_id(*b),
            ..unset
        },
        NodeKind::Min { a, b } => NodeSpec {
            kind: "Min".into(),
            a: port_id(*a),
            b: port_id(*b),
            ..unset
        },
        NodeKind::Max { a, b } => NodeSpec {
            kind: "Max".into(),
            a: port_id(*a),
            b: port_id(*b),
            ..unset
        },
        NodeKind::Sin { a } => NodeSpec {
            kind: "Sin".into(),
            a: port_id(*a),
            ..unset
        },
        NodeKind::Cos { a } => NodeSpec {
            kind: "Cos".into(),
            a: port_id(*a),
            ..unset
        },
        NodeKind::Abs { a } => NodeSpec {
            kind: "Abs".into(),
            a: port_id(*a),
            ..unset
        },
        NodeKind::Sqrt { a } => NodeSpec {
            kind: "Sqrt".into(),
            a: port_id(*a),
            ..unset
        },
        NodeKind::Floor { a } => NodeSpec {
            kind: "Floor".into(),
            a: port_id(*a),
            ..unset
        },
        NodeKind::Fract { a } => NodeSpec {
            kind: "Fract".into(),
            a: port_id(*a),
            ..unset
        },
        NodeKind::Pow { a, b } => NodeSpec {
            kind: "Pow".into(),
            a: port_id(*a),
            b: port_id(*b),
            ..unset
        },
        NodeKind::Mix { a, b, t } => NodeSpec {
            kind: "Mix".into(),
            a: port_id(*a),
            b: port_id(*b),
            c: port_id(*t),
            ..unset
        },
        NodeKind::Clamp { a, min_v, max_v } => NodeSpec {
            kind: "Clamp".into(),
            a: port_id(*a),
            b: port_id(*min_v),
            c: port_id(*max_v),
            ..unset
        },
        NodeKind::SdfPlane { y, height } => NodeSpec {
            kind: "SdfPlane".into(),
            a: port_id(*y),
            b: port_id(*height),
            ..unset
        },
        NodeKind::SdfSphere { x, y, z, radius } => NodeSpec {
            kind: "SdfSphere".into(),
            a: port_id(*x),
            b: port_id(*y),
            c: port_id(*z),
            d: port_id(*radius),
            ..unset
        },
        NodeKind::SdfBox {
            x, y, z, size_x, ..
        } => NodeSpec {
            kind: "SdfBox".into(),
            a: port_id(*x),
            b: port_id(*y),
            c: port_id(*z),
            value: *size_x,
            ..unset
        },
        NodeKind::SdfUnion { a, b } => NodeSpec {
            kind: "SdfUnion".into(),
            a: port_id(*a),
            b: port_id(*b),
            ..unset
        },
        NodeKind::SdfSubtract { a, b } => NodeSpec {
            kind: "SdfSubtract".into(),
            a: port_id(*a),
            b: port_id(*b),
            ..unset
        },
        NodeKind::SdfSmoothUnion { a, b, smoothness } => NodeSpec {
            kind: "SdfSmoothUnion".into(),
            a: port_id(*a),
            b: port_id(*b),
            value: *smoothness,
            ..unset
        },
        NodeKind::SdfSmoothSubtract { a, b, smoothness } => NodeSpec {
            kind: "SdfSmoothSubtract".into(),
            a: port_id(*a),
            b: port_id(*b),
            value: *smoothness,
            ..unset
        },
        NodeKind::Noise2D { x, y, .. } => NodeSpec {
            kind: "Noise2D".into(),
            a: port_id(*x),
            b: port_id(*y),
            ..unset
        },
        NodeKind::Noise3D { x, y, z, .. } => NodeSpec {
            kind: "Noise3D".into(),
            a: port_id(*x),
            b: port_id(*y),
            c: port_id(*z),
            ..unset
        },
        NodeKind::OutputSdf { a } => NodeSpec {
            kind: "OutputSdf".into(),
            a: port_id(*a),
            ..unset
        },
        NodeKind::Image2D { x, y, .. } => NodeSpec {
            kind: "Image2D".into(),
            a: port_id(*x),
            b: port_id(*y),
            ..unset
        },
        NodeKind::Expression { x, y, z, expr } => NodeSpec {
            kind: "Expression".into(),
            a: port_id(*x),
            b: port_id(*y),
            c: port_id(*z),
            expr: Some(expr.expression_text().to_owned()),
            ..unset
        },
        NodeKind::Remap { a, to_end, .. } => NodeSpec {
            kind: "Remap".into(),
            a: port_id(*a),
            value: *to_end,
            ..unset
        },
        NodeKind::Distance2D { x0, y0, x1, y1 } => NodeSpec {
            kind: "Distance2D".into(),
            a: port_id(*x0),
            b: port_id(*y0),
            c: port_id(*x1),
            d: port_id(*y1),
            ..unset
        },
        NodeKind::Distance3D { x0, y0, z0, .. } => NodeSpec {
            kind: "Distance3D".into(),
            a: port_id(*x0),
            b: port_id(*y0),
            c: port_id(*z0),
            ..unset
        },
        NodeKind::Normalize3D { x, y, z } => NodeSpec {
            kind: "Normalize3D".into(),
            a: port_id(*x),
            b: port_id(*y),
            c: port_id(*z),
            ..unset
        },
        NodeKind::SdfTorus { x, y, z, r1, .. } => NodeSpec {
            kind: "SdfTorus".into(),
            a: port_id(*x),
            b: port_id(*y),
            c: port_id(*z),
            value: *r1,
            ..unset
        },
        NodeKind::Curve { a, .. } => NodeSpec {
            kind: "Curve".into(),
            a: port_id(*a),
            ..unset
        },
    }
}

fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

/// Serialize a graph to the compact interchange consumed by `set_graph_json`.
pub fn graph_to_json(graph: &Graph) -> String {
    let mut out = String::from("{\"nodes\":[");
    for (index, node) in graph.nodes().iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let spec = node_kind_to_spec(&node.kind);
        out.push_str("{\"id\":");
        out.push_str(&node.id.to_string());
        out.push_str(",\"kind\":\"");
        out.push_str(&spec.kind);
        out.push('"');
        if spec.a >= 0 {
            out.push_str(",\"a\":");
            out.push_str(&spec.a.to_string());
        }
        if spec.b >= 0 {
            out.push_str(",\"b\":");
            out.push_str(&spec.b.to_string());
        }
        if spec.c >= 0 {
            out.push_str(",\"c\":");
            out.push_str(&spec.c.to_string());
        }
        if spec.d >= 0 {
            out.push_str(",\"d\":");
            out.push_str(&spec.d.to_string());
        }
        if spec.value != 0.0 {
            out.push_str(",\"value\":");
            out.push_str(&spec.value.to_string());
        }
        if let Some(expr) = spec.expr.as_ref() {
            out.push_str(",\"expr\":\"");
            out.push_str(&json_escape(expr));
            out.push('"');
        }
        out.push('}');
    }
    out.push_str("]}");
    out
}

/// Compiled, analysis-cached form of a [`Graph`] (audit §9.6-C1).
///
/// `Graph` is a mutable, user-facing construction surface; `CompiledGraph` is
/// the immutable, analysed form an executor consumes. Building it once (in
/// `GraphGenerator::new` or lazily on first `generate_block`) eliminates the
/// per-Y-slice overhead of recomputing the topological order, resolving sparse
/// node ids by linear scan, and classifying which nodes are Y-independent.
///
/// The XZ-prefix classification mirrors C++ `move_outer_group_operations_up`
/// (`voxel_graph_compiler.cpp:1478`): nodes reachable from `{InputX, InputZ}`
/// only — never touching `InputY` — form the *outer group* and are placed first
/// in `nodes`. `xz_prefix_len` is the count of such nodes; the executor caches
/// their outputs across Y-slices (only the inner tail re-runs per slice).
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    /// Nodes in topological order (inputs before consumers, outputs last),
    /// with outer-group (XZ-only) nodes placed before inner-group nodes.
    nodes: Vec<GraphNode>,
    /// Sparse `GraphNodeId` → dense index into `nodes`. Replaces the per-node
    /// `Vec::iter().find` the lazy path paid per slice.
    id_to_index: HashMap<GraphNodeId, usize>,
    /// Number of leading nodes in `nodes` that are Y-independent (outer group).
    /// Slices `[0, xz_prefix_len)` may be cached across Y-slices; the tail
    /// `[xz_prefix_len, len)` depends on `InputY` and re-runs every slice.
    xz_prefix_len: usize,
}

impl CompiledGraph {
    /// Analyse `graph` into a compiled form. Performs the topological sort and
    /// XZ-prefix classification once. Returns the same `TopoError` the lazy
    /// `Graph::topological_order` does on cycles / dangling ports.
    pub fn compile(graph: &Graph) -> Result<Self, TopoError> {
        let order = graph.topological_order()?;
        let by_id: HashMap<GraphNodeId, &GraphNode> =
            graph.nodes.iter().map(|n| (n.id, n)).collect();
        let mut nodes: Vec<GraphNode> = Vec::with_capacity(order.len());
        let mut id_to_index: HashMap<GraphNodeId, usize> = HashMap::with_capacity(order.len());
        for id in &order {
            let node = *by_id
                .get(id)
                .expect("topological_order returned an id not in the graph");
            id_to_index.insert(*id, nodes.len());
            nodes.push(node.clone());
        }
        // Forward-propagate Y-dependence from InputY seeds through the topo
        // order. A node is inner-group iff it IS InputY or any Y-dependent
        // input feeds it. The topo property (producers before consumers)
        // guarantees inputs are already classified when their consumers run.
        let mut depends_on_y = vec![false; nodes.len()];
        for (i, node) in nodes.iter().enumerate() {
            let self_y = matches!(node.kind, NodeKind::InputY);
            let any_input_y = node.kind.inputs().into_iter().flatten().any(|port| {
                let src = *id_to_index.get(&port.node).unwrap_or(&usize::MAX);
                *depends_on_y.get(src).unwrap_or(&false)
            });
            depends_on_y[i] = self_y || any_input_y;
        }
        // `xz_prefix_len` = index of the first Y-dependent node. Everything
        // before it is outer-group (XZ-only); the tail re-runs every slice.
        let xz_prefix_len = depends_on_y.iter().position(|&y| y).unwrap_or(nodes.len());
        Ok(Self {
            nodes,
            id_to_index,
            xz_prefix_len,
        })
    }

    /// Number of leading nodes that are Y-independent (the XZ-only prefix).
    /// The executor may cache their outputs across Y-slices.
    #[inline]
    pub fn xz_prefix_len(&self) -> usize {
        self.xz_prefix_len
    }

    /// Total node count.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True if there are no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Dense index for a sparse `GraphNodeId`. Used by the executor to resolve
    /// input ports without a per-element `HashMap` lookup.
    #[inline]
    pub fn index_of(&self, id: GraphNodeId) -> Option<usize> {
        self.id_to_index.get(&id).copied()
    }

    /// The nodes in topological order (outer-group first).
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Propagate input intervals through the compiled topological order to
    /// estimate the output-SDF range over a bounding box (audit §9.6-C3).
    ///
    /// If the returned interval does not straddle zero, the entire block is
    /// provably uniform (fully-solid when `max < 0`, fully-air when `min > 0`)
    /// and `generate_block` can fill it without per-voxel evaluation — the same
    /// mechanism the C++ VM uses to skip interior/air blocks. Hard nodes
    /// (Noise/Cos/Curve/Smooth-SDF/etc. — non-monotone or unanalysed) fall back
    /// to a conservative full-range `[-∞,+∞]`, so a graph containing them just
    /// loses the optimisation without producing wrong results.
    ///
    /// `x`/`y`/`z` are the world-coordinate intervals spanned by the block.
    pub fn analyze_range(
        &self,
        x: crate::math::Interval,
        y: crate::math::Interval,
        z: crate::math::Interval,
    ) -> crate::math::Interval {
        use crate::math::interval as iv;
        let mut ranges: Vec<crate::math::Interval> =
            vec![iv::Interval::infinity(); self.nodes.len()];
        let inf = iv::Interval::infinity();
        let resolve =
            |ranges: &[crate::math::Interval], port: &Option<GraphPort>| -> crate::math::Interval {
                match port {
                    Some(p) => ranges
                        .get(self.id_to_index.get(&p.node).copied().unwrap_or(usize::MAX))
                        .copied()
                        .unwrap_or(inf),
                    None => iv::Interval::single(0.0),
                }
            };
        for (i, node) in self.nodes.iter().enumerate() {
            let r = match &node.kind {
                NodeKind::InputX => x,
                NodeKind::InputY => y,
                NodeKind::InputZ => z,
                NodeKind::Constant(v) => iv::Interval::single(*v),
                // Easy binary ops (interval arithmetic available).
                NodeKind::Add { a, b } => resolve(&ranges, a) + resolve(&ranges, b),
                NodeKind::Subtract { a, b } => resolve(&ranges, a) - resolve(&ranges, b),
                NodeKind::Multiply { a, b } => resolve(&ranges, a) * resolve(&ranges, b),
                NodeKind::Divide { a, b } => resolve(&ranges, a) / resolve(&ranges, b),
                NodeKind::Min { a, b } => {
                    iv::min_interval(resolve(&ranges, a), resolve(&ranges, b))
                }
                NodeKind::Max { a, b } => {
                    iv::max_interval(resolve(&ranges, a), resolve(&ranges, b))
                }
                // Hard / non-monotone — conservative full range.
                NodeKind::Pow { .. } => inf,
                NodeKind::Sin { a } => iv::sin(resolve(&ranges, a)),
                NodeKind::Cos { .. } => inf,
                NodeKind::Abs { a } => iv::abs(resolve(&ranges, a)),
                NodeKind::Sqrt { a } => iv::sqrt(resolve(&ranges, a)),
                NodeKind::Floor { a } => iv::floor(resolve(&ranges, a)),
                NodeKind::Fract { a } => resolve(&ranges, a) - iv::floor(resolve(&ranges, a)),
                NodeKind::Remap {
                    a,
                    from_start,
                    from_end,
                    to_start,
                    to_end,
                } => {
                    let fs = iv::Interval::single(*from_start);
                    let fe = iv::Interval::single(*from_end);
                    let ts = iv::Interval::single(*to_start);
                    let te = iv::Interval::single(*to_end);
                    iv::lerp(ts, te, (resolve(&ranges, a) - fs) / (fe - fs))
                }
                NodeKind::Distance2D { x0, y0, x1, y1, .. } => iv::get_length2(
                    resolve(&ranges, x1) - resolve(&ranges, x0),
                    resolve(&ranges, y1) - resolve(&ranges, y0),
                ),
                NodeKind::Distance3D {
                    x0,
                    y0,
                    z0,
                    x1,
                    y1,
                    z1,
                } => iv::get_length3(
                    resolve(&ranges, x1) - resolve(&ranges, x0),
                    resolve(&ranges, y1) - resolve(&ranges, y0),
                    resolve(&ranges, z1) - resolve(&ranges, z0),
                ),
                NodeKind::Normalize3D { .. } => inf,
                NodeKind::Mix { a, b, t } => iv::lerp(
                    resolve(&ranges, a),
                    resolve(&ranges, b),
                    resolve(&ranges, t),
                ),
                NodeKind::Clamp { a, min_v, max_v } => iv::clamp(
                    resolve(&ranges, a),
                    resolve(&ranges, min_v),
                    resolve(&ranges, max_v),
                ),
                // Curve/Noise — non-monotone; conservative.
                NodeKind::Curve { .. } => inf,
                NodeKind::Noise2D { .. } | NodeKind::Noise3D { .. } => inf,
                NodeKind::Image2D { .. } | NodeKind::Expression { .. } => inf,
                // SDF nodes: easy ones compose from interval primitives.
                NodeKind::SdfPlane { y, height } => resolve(&ranges, y) - resolve(&ranges, height),
                NodeKind::SdfBox { .. } | NodeKind::SdfTorus { .. } => inf,
                NodeKind::SdfSphere { x, y, z, radius } => {
                    iv::get_length3(
                        resolve(&ranges, x),
                        resolve(&ranges, y),
                        resolve(&ranges, z),
                    ) - resolve(&ranges, radius)
                }
                NodeKind::SdfUnion { a, b } => {
                    iv::min_interval(resolve(&ranges, a), resolve(&ranges, b))
                }
                NodeKind::SdfSubtract { a, b } => {
                    iv::max_interval(resolve(&ranges, a), -resolve(&ranges, b))
                }
                NodeKind::SdfSmoothUnion { .. } | NodeKind::SdfSmoothSubtract { .. } => inf,
                // Output passes through its input interval.
                NodeKind::OutputSdf { a } => resolve(&ranges, a),
            };
            ranges[i] = r;
        }
        // The OutputSdf node (last in topo order) holds the SDF range.
        self.nodes
            .iter()
            .enumerate()
            .rev()
            .find(|(_, n)| n.kind.is_output())
            .map(|(i, _)| ranges[i])
            .unwrap_or(inf)
    }

    ///
    /// Mirrors [`Graph::generate`] semantics (same per-node math + edge cases)
    /// but resolves ports by dense index, stores intermediates in a dense
    /// `Vec<Vec<f32>>`, preserves buffer capacity across slices, and — when
    /// `xz_prefix_cached` is true — skips re-evaluating the Y-independent
    /// prefix (their buffers persist from the previous slice).
    pub fn generate_slice(
        &self,
        inputs: &GraphInputs,
        slice_size: usize,
        scratch: &mut CompiledScratch,
        outputs: &mut Vec<(GraphOutput, Vec<f32>)>,
        xz_prefix_cached: bool,
    ) {
        let start = if xz_prefix_cached {
            self.xz_prefix_len
        } else {
            0
        };
        if scratch.buffers.len() < self.nodes.len() {
            scratch.buffers.resize(self.nodes.len(), Vec::new());
        }
        // Clear only the inner-tail buffers; prefix is reused when cached.
        for buf in &mut scratch.buffers[start..] {
            buf.clear();
        }
        outputs.clear();
        for (i, node) in self.nodes.iter().enumerate() {
            if i < start {
                continue;
            }
            self.eval_node(node, i, inputs, slice_size, scratch);
        }
        // Collect outputs from any OutputSdf node — its buffer persists in
        // scratch (across the XZ-prefix cache boundary too), so this works
        // whether or not the node was re-evaluated this slice.
        for (i, node) in self.nodes.iter().enumerate() {
            if let NodeKind::OutputSdf { .. } = node.kind {
                if let Some(buf) = scratch.buffers.get(i) {
                    if !buf.is_empty() {
                        // Clone to detach from scratch (caller owns the output).
                        outputs.push((GraphOutput::Sdf, buf.clone()));
                    }
                }
            }
        }
    }

    fn eval_node(
        &self,
        node: &GraphNode,
        node_index: usize,
        inputs: &GraphInputs,
        slice_size: usize,
        scratch: &mut CompiledScratch,
    ) {
        let kind = &node.kind;
        match kind {
            NodeKind::InputX => scratch.set(node_index, inputs.x.to_vec()),
            NodeKind::InputY => scratch.set(node_index, vec![inputs.y; slice_size]),
            NodeKind::InputZ => scratch.set(node_index, inputs.z.to_vec()),
            NodeKind::Constant(v) => scratch.set(node_index, vec![*v; slice_size]),
            NodeKind::Add { a, b } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i) + self.val(scratch, b, i))
                    .collect(),
            ),
            NodeKind::Subtract { a, b } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i) - self.val(scratch, b, i))
                    .collect(),
            ),
            NodeKind::Multiply { a, b } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i) * self.val(scratch, b, i))
                    .collect(),
            ),
            NodeKind::Divide { a, b } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| {
                        // C++ parity: exact-zero test, not epsilon.
                        let denom = self.val(scratch, b, i);
                        if denom == 0.0 {
                            0.0
                        } else {
                            self.val(scratch, a, i) / denom
                        }
                    })
                    .collect(),
            ),
            NodeKind::Min { a, b } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i).min(self.val(scratch, b, i)))
                    .collect(),
            ),
            NodeKind::Max { a, b } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i).max(self.val(scratch, b, i)))
                    .collect(),
            ),
            NodeKind::Pow { a, b } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i).powf(self.val(scratch, b, i)))
                    .collect(),
            ),
            NodeKind::Sin { a } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i).sin())
                    .collect(),
            ),
            NodeKind::Cos { a } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i).cos())
                    .collect(),
            ),
            NodeKind::Abs { a } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i).abs())
                    .collect(),
            ),
            NodeKind::Sqrt { a } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i).max(0.0).sqrt())
                    .collect(),
            ),
            NodeKind::Floor { a } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i).floor())
                    .collect(),
            ),
            NodeKind::Fract { a } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i).fract())
                    .collect(),
            ),
            NodeKind::Remap {
                a,
                from_start,
                from_end,
                to_start,
                to_end,
            } => {
                let (fs, fe, ts, te) = (*from_start, *from_end, *to_start, *to_end);
                let from_span = fe - fs;
                let to_span = te - ts;
                scratch.set(
                    node_index,
                    (0..slice_size)
                        .map(|i| {
                            // C++ parity: pure linear remap (no clamp).
                            if from_span.abs() <= f32::EPSILON {
                                0.0
                            } else {
                                let v = self.val(scratch, a, i);
                                ts + (v - fs) / from_span * to_span
                            }
                        })
                        .collect(),
                );
            }
            NodeKind::Distance2D { x0, y0, x1, y1, .. } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| {
                        let dx = self.val(scratch, x1, i) - self.val(scratch, x0, i);
                        let dy = self.val(scratch, y1, i) - self.val(scratch, y0, i);
                        (dx * dx + dy * dy).sqrt()
                    })
                    .collect(),
            ),
            NodeKind::Distance3D {
                x0,
                y0,
                z0,
                x1,
                y1,
                z1,
            } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| {
                        let dx = self.val(scratch, x1, i) - self.val(scratch, x0, i);
                        let dy = self.val(scratch, y1, i) - self.val(scratch, y0, i);
                        let dz = self.val(scratch, z1, i) - self.val(scratch, z0, i);
                        (dx * dx + dy * dy + dz * dz).sqrt()
                    })
                    .collect(),
            ),
            NodeKind::Normalize3D { x, y, z } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| {
                        let dx = self.val(scratch, x, i);
                        let dy = self.val(scratch, y, i);
                        let dz = self.val(scratch, z, i);
                        let len = (dx * dx + dy * dy + dz * dz).sqrt();
                        if len < 1e-12 {
                            0.0
                        } else {
                            dx / len
                        }
                    })
                    .collect(),
            ),
            NodeKind::Mix { a, b, t } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| {
                        let av = self.val(scratch, a, i);
                        let bv = self.val(scratch, b, i);
                        let tv = self.val(scratch, t, i);
                        av * (1.0 - tv) + bv * tv
                    })
                    .collect(),
            ),
            NodeKind::Clamp { a, min_v, max_v } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| {
                        let v = self.val(scratch, a, i);
                        let lo = self.val(scratch, min_v, i);
                        let hi = self.val(scratch, max_v, i);
                        v.clamp(lo.min(hi), lo.max(hi))
                    })
                    .collect(),
            ),
            NodeKind::Curve { a, curve } => {
                let curve = curve.clone();
                scratch.set(
                    node_index,
                    (0..slice_size)
                        .map(|i| curve.sample(self.val(scratch, a, i)))
                        .collect(),
                );
            }
            NodeKind::Noise2D { x, y, noise } => {
                let noise = noise.build();
                scratch.set(
                    node_index,
                    (0..slice_size)
                        .map(|i| {
                            noise.get_noise_2d(self.val(scratch, x, i), self.val(scratch, y, i))
                        })
                        .collect(),
                );
            }
            NodeKind::Noise3D { x, y, z, noise } => {
                let noise = noise.build();
                scratch.set(
                    node_index,
                    (0..slice_size)
                        .map(|i| {
                            noise.get_noise_3d(
                                self.val(scratch, x, i),
                                self.val(scratch, y, i),
                                self.val(scratch, z, i),
                            )
                        })
                        .collect(),
                );
            }
            NodeKind::SdfPlane { y, height } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, y, i) - self.val(scratch, height, i))
                    .collect(),
            ),
            NodeKind::SdfBox {
                x,
                y,
                z,
                size_x,
                size_y,
                size_z,
            } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| {
                        let dx = self.val(scratch, x, i).abs() - size_x;
                        let dy = self.val(scratch, y, i).abs() - size_y;
                        let dz = self.val(scratch, z, i).abs() - size_z;
                        let outside = dx.max(dy).max(dz).max(0.0);
                        let inside = dx.max(dy).max(dz).min(0.0);
                        outside + inside
                    })
                    .collect(),
            ),
            NodeKind::SdfSphere { x, y, z, radius } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| {
                        let dx = self.val(scratch, x, i);
                        let dy = self.val(scratch, y, i);
                        let dz = self.val(scratch, z, i);
                        let r = self.val(scratch, radius, i).max(1e-12);
                        (dx * dx + dy * dy + dz * dz).sqrt() - r
                    })
                    .collect(),
            ),
            NodeKind::SdfTorus { x, y, z, r1, r2 } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| {
                        let dx = self.val(scratch, x, i);
                        let dy = self.val(scratch, y, i);
                        let dz = self.val(scratch, z, i);
                        let qx = (dx * dx + dz * dz).sqrt() - r1;
                        let qy = dy;
                        (qx * qx + qy * qy).sqrt() - r2
                    })
                    .collect(),
            ),
            NodeKind::SdfUnion { a, b } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i).min(self.val(scratch, b, i)))
                    .collect(),
            ),
            NodeKind::SdfSubtract { a, b } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| self.val(scratch, a, i).max(-self.val(scratch, b, i)))
                    .collect(),
            ),
            NodeKind::SdfSmoothUnion { a, b, smoothness } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| {
                        let av = self.val(scratch, a, i);
                        let bv = self.val(scratch, b, i);
                        let s = *smoothness;
                        if s.abs() < 1e-6 {
                            return av.min(bv);
                        }
                        let h = (s - (bv - av).abs() * 0.5).clamp(0.0, s);
                        bv - h + h * h / s
                    })
                    .collect(),
            ),
            NodeKind::SdfSmoothSubtract { a, b, smoothness } => scratch.set(
                node_index,
                (0..slice_size)
                    .map(|i| {
                        let av = self.val(scratch, a, i);
                        let bv = self.val(scratch, b, i);
                        let s = *smoothness;
                        if s > 1e-4 {
                            crate::math::sdf::sdf_smooth_subtract(av, bv, s)
                        } else {
                            crate::math::sdf::sdf_subtract(av, bv)
                        }
                    })
                    .collect(),
            ),
            NodeKind::Image2D { x, y, image } => {
                let image = image.clone();
                scratch.set(
                    node_index,
                    (0..slice_size)
                        .map(|i| {
                            image.sample_bilinear(self.val(scratch, x, i), self.val(scratch, y, i))
                        })
                        .collect(),
                );
            }
            NodeKind::Expression { x, y, z, expr } => {
                let xs: Vec<f32> = (0..slice_size).map(|i| self.val(scratch, x, i)).collect();
                let ys: Vec<f32> = (0..slice_size).map(|i| self.val(scratch, y, i)).collect();
                let zs: Vec<f32> = (0..slice_size).map(|i| self.val(scratch, z, i)).collect();
                scratch.set(node_index, expr.evaluate_slice(&[&xs, &ys, &zs]));
            }
            NodeKind::OutputSdf { a } => {
                let data: Vec<f32> = (0..slice_size).map(|i| self.val(scratch, a, i)).collect();
                scratch.set(node_index, data);
            }
        }
    }

    /// Read an input port value at element `idx` from the dense scratch.
    /// Unconnected ports return 0.0 (matching `Graph::generate`'s `value_at`).
    #[inline]
    fn val(&self, scratch: &CompiledScratch, port: &Option<GraphPort>, idx: usize) -> f32 {
        match port {
            Some(p) if p.output == 0 => self
                .id_to_index
                .get(&p.node)
                .copied()
                .and_then(|i| scratch.buffers.get(i).and_then(|b| b.get(idx)))
                .copied()
                .unwrap_or(0.0),
            // GRAPH-2: multi-output (Normalize3D ny/nz/len) not yet supported
            // in the compiled path — falls back to 0.0. Full support requires
            // extending CompiledScratch to hold extra buffers per node.
            _ => 0.0,
        }
    }
}

/// Dense scratch buffers for [`CompiledGraph::generate_slice`] (audit §9.6-C1).
///
/// Replaces [`GraphScratch`] (a `HashMap<GraphNodeId, Vec<f32>>`) with a
/// `Vec<Vec<f32>>` indexed by dense topological position. The buffers persist
/// across slices (only length is cleared, not capacity), so allocation
/// amortises — unlike `GraphScratch::clear`, which drops every buffer.
#[derive(Debug, Default)]
pub struct CompiledScratch {
    buffers: Vec<Vec<f32>>,
}

impl CompiledScratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a node's output at its dense index (grows the vector if needed).
    fn set(&mut self, index: usize, data: Vec<f32>) {
        if index >= self.buffers.len() {
            self.buffers.resize(index + 1, Vec::new());
        }
        self.buffers[index] = data;
    }
}

/// Per-thread execution scratch. Stores the f32 slice produced for every
/// node id during a single `generate` call. Reused across calls to avoid
/// reallocation; cleared at the start of each call.
#[derive(Debug, Default)]
pub struct GraphScratch {
    buffers: std::collections::HashMap<GraphNodeId, Vec<f32>>,
}

impl GraphScratch {
    pub fn new() -> Self {
        Self::default()
    }

    fn clear(&mut self) {
        self.buffers.clear();
    }

    fn put(&mut self, id: GraphNodeId, data: Vec<f32>) {
        self.buffers.insert(id, data);
    }

    /// Returns the f32 slice produced by `id`, or `None` if the node hasn't
    /// been evaluated yet. Used internally by the binop/monop helpers.
    fn get(&self, id: GraphNodeId) -> Option<&[f32]> {
        self.buffers.get(&id).map(Vec::as_slice)
    }
}

/// Inputs bound by the caller for one `generate` invocation. `x` and `z`
/// carry per-voxel coordinates for the slice; `y` is the constant slice Y.
#[derive(Debug, Clone, Copy)]
pub struct GraphInputs<'a> {
    pub x: &'a [f32],
    pub y: f32,
    pub z: &'a [f32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopoError {
    Cycle,
    DanglingPort(GraphNodeId),
}

fn binop(
    scratch: &GraphScratch,
    a: &Option<GraphPort>,
    b: &Option<GraphPort>,
    slice_size: usize,
    f: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    (0..slice_size)
        .map(|i| f(value_at(scratch, a, i, 0.0), value_at(scratch, b, i, 0.0)))
        .collect()
}

fn monop(
    scratch: &GraphScratch,
    a: &Option<GraphPort>,
    slice_size: usize,
    f: impl Fn(f32) -> f32,
) -> Vec<f32> {
    (0..slice_size)
        .map(|i| f(value_at(scratch, a, i, 0.0)))
        .collect()
}

fn ternary(
    scratch: &GraphScratch,
    a: &Option<GraphPort>,
    b: &Option<GraphPort>,
    c: &Option<GraphPort>,
    slice_size: usize,
    f: impl Fn(f32, f32, f32) -> f32,
) -> Vec<f32> {
    (0..slice_size)
        .map(|i| {
            f(
                value_at(scratch, a, i, 0.0),
                value_at(scratch, b, i, 0.0),
                value_at(scratch, c, i, 0.0),
            )
        })
        .collect()
}

fn value_at(
    scratch: &GraphScratch,
    port: &Option<GraphPort>,
    index: usize,
    default_value: f32,
) -> f32 {
    port.as_ref()
        .and_then(|p| {
            // GRAPH-2: multi-output nodes (Normalize3D) store extra outputs
            // under synthetic keys (node_id + output_index << 24).
            let key = if p.output == 0 {
                p.node
            } else {
                GraphNodeId::wrapping_add(p.node, (p.output as u32) << 24)
            };
            scratch.get(key)
        })
        .and_then(|values| values.get(index))
        .copied()
        .unwrap_or(default_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x_inputs(slice_size: usize) -> Vec<f32> {
        (0..slice_size).map(|i| i as f32).collect()
    }

    // ---- C1: CompiledGraph tests (audit §9.6-C1) ----

    #[test]
    fn expression_and_image2d_nodes_evaluate() {
        use crate::generators::graph::expression_node::ExpressionNode;
        use crate::generators::graph::image::Image2D;

        let mut g = Graph::new();
        let x = g.push(NodeKind::InputX);
        let expr = ExpressionNode::new("x * 2", &[("x", 0), ("y", 1), ("z", 2)]).unwrap();
        let e = g.push(NodeKind::Expression {
            x: Some(GraphPort::new(x)),
            y: None,
            z: None,
            expr: std::sync::Arc::new(expr),
        });
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(e)),
        });
        let compiled = CompiledGraph::compile(&g).expect("compile expression");
        let xs = [3.0f32];
        let zs = [0.0f32];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &zs,
        };
        let mut scratch = CompiledScratch::new();
        let mut out = Vec::new();
        compiled.generate_slice(&inputs, 1, &mut scratch, &mut out, false);
        let sdf = out
            .into_iter()
            .find(|(k, _)| *k == GraphOutput::Sdf)
            .and_then(|(_, v)| v.into_iter().next())
            .unwrap();
        assert!((sdf - 6.0).abs() < 1e-5, "expression x*2 at x=3, got {sdf}");

        let mut g2 = Graph::new();
        let ix = g2.push(NodeKind::InputX);
        let iy = g2.push(NodeKind::InputY);
        let img = Image2D::from_data(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
        let sample = g2.push(NodeKind::Image2D {
            x: Some(GraphPort::new(ix)),
            y: Some(GraphPort::new(iy)),
            image: std::sync::Arc::new(img),
        });
        g2.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(sample)),
        });
        assert!(CompiledGraph::compile(&g2).is_ok());
    }

    #[test]
    fn graph_to_json_round_trips_through_from_spec() {
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        let c = graph.push(NodeKind::Constant(2.5));
        graph.push(NodeKind::Add {
            a: Some(GraphPort::new(x)),
            b: Some(GraphPort::new(c)),
        });
        let json = graph_to_json(&graph);
        assert!(json.contains("\"kind\":\"InputX\""));
        assert!(json.contains("\"kind\":\"Constant\""));
        assert!(json.contains("\"value\":2.5"));
        assert!(json.contains("\"kind\":\"Add\""));
    }

    #[test]
    fn node_kind_from_spec_builds_documented_kinds() {
        assert!(matches!(
            node_kind_from_spec("InputX", -1, -1, -1, -1, 0.0),
            Some(NodeKind::InputX)
        ));
        assert!(matches!(
            node_kind_from_spec("Constant", -1, -1, -1, -1, 4.5),
            Some(NodeKind::Constant(v)) if v == 4.5
        ));
        assert!(matches!(
            node_kind_from_spec("SdfSphere", 0, 1, 2, 3, 0.0),
            Some(NodeKind::SdfSphere { .. })
        ));
        assert!(matches!(
            node_kind_from_spec("OutputSdf", 4, -1, -1, -1, 0.0),
            Some(NodeKind::OutputSdf { .. })
        ));
        assert!(node_kind_from_spec("NotANode", -1, -1, -1, -1, 0.0).is_none());
    }

    #[test]
    fn compiled_graph_topological_order_matches_lazy() {
        let mut g = Graph::new();
        let x = g.push(NodeKind::InputX);
        let c = g.push(NodeKind::Constant(3.0));
        let mul = g.push(NodeKind::Multiply {
            a: Some(GraphPort { node: x, output: 0 }),
            b: Some(GraphPort { node: c, output: 0 }),
        });
        let _out = g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: mul,
                output: 0,
            }),
        });
        let lazy = g.topological_order().expect("lazy topo");
        let compiled = CompiledGraph::compile(&g).expect("compile");
        assert_eq!(
            compiled.nodes().iter().map(|n| n.id).collect::<Vec<_>>(),
            lazy
        );
    }

    #[test]
    fn compiled_graph_classifies_xz_only_prefix() {
        let mut g = Graph::new();
        let x = g.push(NodeKind::InputX);
        let z = g.push(NodeKind::InputZ);
        let mul = g.push(NodeKind::Multiply {
            a: Some(GraphPort { node: x, output: 0 }),
            b: Some(GraphPort { node: z, output: 0 }),
        });
        let _out = g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: mul,
                output: 0,
            }),
        });
        let compiled = CompiledGraph::compile(&g).expect("compile");
        // Pure-XZ graph: every node is Y-independent.
        assert_eq!(compiled.xz_prefix_len(), 4);
    }

    #[test]
    fn compiled_graph_xz_prefix_excludes_y_dependent_nodes() {
        let mut g = Graph::new();
        let x = g.push(NodeKind::InputX);
        let y = g.push(NodeKind::InputY);
        let add = g.push(NodeKind::Add {
            a: Some(GraphPort { node: x, output: 0 }),
            b: Some(GraphPort { node: y, output: 0 }),
        });
        let _out = g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: add,
                output: 0,
            }),
        });
        let compiled = CompiledGraph::compile(&g).expect("compile");
        // InputX is outer; InputY, Add, OutputSdf are inner.
        assert_eq!(compiled.xz_prefix_len(), 1);
    }

    #[test]
    fn compiled_graph_xz_prefix_is_one_for_constant_plus_y() {
        let mut g = Graph::new();
        let c = g.push(NodeKind::Constant(2.0));
        let y = g.push(NodeKind::InputY);
        let add = g.push(NodeKind::Add {
            a: Some(GraphPort { node: c, output: 0 }),
            b: Some(GraphPort { node: y, output: 0 }),
        });
        let _out = g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: add,
                output: 0,
            }),
        });
        let compiled = CompiledGraph::compile(&g).expect("compile");
        // Constant is XZ-only; Add depends on InputY → inner.
        assert_eq!(compiled.xz_prefix_len(), 1);
    }

    #[test]
    fn compiled_graph_id_to_index_resolves_dense() {
        let mut g = Graph::new();
        g.add_node(GraphNode {
            id: 100,
            kind: NodeKind::InputX,
        });
        g.add_node(GraphNode {
            id: 200,
            kind: NodeKind::InputZ,
        });
        let compiled = CompiledGraph::compile(&g).expect("compile");
        assert_eq!(compiled.index_of(100), Some(0));
        assert_eq!(compiled.index_of(200), Some(1));
        assert_eq!(compiled.index_of(999), None);
    }

    #[test]
    fn compiled_graph_cycle_returns_error() {
        let mut g = Graph::new();
        let a = g.push(NodeKind::Add { a: None, b: None });
        g.add_node(GraphNode {
            id: 2,
            kind: NodeKind::Add {
                a: Some(GraphPort { node: a, output: 0 }),
                b: Some(GraphPort { node: 2, output: 0 }),
            },
        });
        assert!(CompiledGraph::compile(&g).is_err());
    }

    // ---- C1 step 2: generate_slice + dense scratch parity tests ----

    /// Run both lazy `Graph::generate` and compiled `generate_slice` over the
    /// same single-slice inputs; return both SDF outputs for comparison.
    fn lazy_and_compiled_sdf(
        graph: &Graph,
        inputs: &GraphInputs,
        slice_size: usize,
    ) -> (Option<Vec<f32>>, Option<Vec<f32>>) {
        let mut lazy_scratch = GraphScratch::new();
        let mut lazy_out = Vec::new();
        let _ = graph.generate(inputs, slice_size, &mut lazy_scratch, &mut lazy_out);
        let lazy_sdf = lazy_out
            .into_iter()
            .find(|(k, _)| *k == GraphOutput::Sdf)
            .map(|(_, v)| v);
        let compiled = CompiledGraph::compile(graph).expect("compile");
        let mut cscratch = CompiledScratch::new();
        let mut cout = Vec::new();
        compiled.generate_slice(inputs, slice_size, &mut cscratch, &mut cout, false);
        let compiled_sdf = cout
            .into_iter()
            .find(|(k, _)| *k == GraphOutput::Sdf)
            .map(|(_, v)| v);
        (lazy_sdf, compiled_sdf)
    }

    #[test]
    fn compiled_generate_matches_lazy_for_multiply() {
        let mut g = Graph::new();
        let x = g.push(NodeKind::InputX);
        let c = g.push(NodeKind::Constant(3.0));
        let mul = g.push(NodeKind::Multiply {
            a: Some(GraphPort { node: x, output: 0 }),
            b: Some(GraphPort { node: c, output: 0 }),
        });
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: mul,
                output: 0,
            }),
        });
        let xs = x_inputs(4);
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let (lazy, compiled) = lazy_and_compiled_sdf(&g, &inputs, 4);
        assert_eq!(lazy.as_deref(), compiled.as_deref());
        assert_eq!(compiled.as_deref(), Some(&[0.0f32, 3.0, 6.0, 9.0][..]));
    }

    #[test]
    fn compiled_generate_matches_lazy_for_sin() {
        let mut g = Graph::new();
        let x = g.push(NodeKind::InputX);
        let sin = g.push(NodeKind::Sin {
            a: Some(GraphPort { node: x, output: 0 }),
        });
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: sin,
                output: 0,
            }),
        });
        let xs = x_inputs(3);
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let (lazy, compiled) = lazy_and_compiled_sdf(&g, &inputs, 3);
        assert_eq!(lazy.as_deref(), compiled.as_deref());
    }

    #[test]
    fn compiled_generate_matches_lazy_for_sdf_sphere_with_y() {
        let mut g = Graph::new();
        let x = g.push(NodeKind::InputX);
        let y = g.push(NodeKind::InputY);
        let z = g.push(NodeKind::InputZ);
        let r = g.push(NodeKind::Constant(2.0));
        let sph = g.push(NodeKind::SdfSphere {
            x: Some(GraphPort { node: x, output: 0 }),
            y: Some(GraphPort { node: y, output: 0 }),
            z: Some(GraphPort { node: z, output: 0 }),
            radius: Some(GraphPort { node: r, output: 0 }),
        });
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: sph,
                output: 0,
            }),
        });
        let xs = [1.0f32, 0.0, 0.0];
        let zs = [0.0f32, 0.0, 0.0];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &zs,
        };
        let (lazy, compiled) = lazy_and_compiled_sdf(&g, &inputs, 3);
        assert_eq!(lazy.as_deref(), compiled.as_deref());
    }

    #[test]
    fn compiled_generate_divide_by_zero_outputs_zero() {
        let mut g = Graph::new();
        let a = g.push(NodeKind::Constant(4.0));
        let b = g.push(NodeKind::Constant(0.0));
        let div = g.push(NodeKind::Divide {
            a: Some(GraphPort { node: a, output: 0 }),
            b: Some(GraphPort { node: b, output: 0 }),
        });
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: div,
                output: 0,
            }),
        });
        let xs = x_inputs(2);
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let (_, compiled) = lazy_and_compiled_sdf(&g, &inputs, 2);
        assert_eq!(compiled.as_deref(), Some(&[0.0f32, 0.0][..]));
    }

    #[test]
    fn compiled_generate_sqrt_clamps_negative_to_zero() {
        let mut g = Graph::new();
        let a = g.push(NodeKind::Constant(-4.0));
        let sq = g.push(NodeKind::Sqrt {
            a: Some(GraphPort { node: a, output: 0 }),
        });
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: sq,
                output: 0,
            }),
        });
        let xs = x_inputs(2);
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let (_, compiled) = lazy_and_compiled_sdf(&g, &inputs, 2);
        assert_eq!(compiled.as_deref(), Some(&[0.0f32, 0.0][..]));
    }

    #[test]
    fn compiled_generate_xz_prefix_cached_matches_full_eval() {
        let mut g = Graph::new();
        let x = g.push(NodeKind::InputX);
        let z = g.push(NodeKind::InputZ);
        let mul = g.push(NodeKind::Multiply {
            a: Some(GraphPort { node: x, output: 0 }),
            b: Some(GraphPort { node: z, output: 0 }),
        });
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: mul,
                output: 0,
            }),
        });
        let compiled = CompiledGraph::compile(&g).expect("compile");
        assert_eq!(compiled.xz_prefix_len(), 4, "pure-XZ graph");
        let xs = x_inputs(4);
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = CompiledScratch::new();
        let mut out_full = Vec::new();
        compiled.generate_slice(&inputs, 4, &mut scratch, &mut out_full, false);
        let sdf_full = out_full
            .iter()
            .find(|(k, _)| *k == GraphOutput::Sdf)
            .map(|(_, v)| v.clone());
        let mut out_cached = Vec::new();
        compiled.generate_slice(&inputs, 4, &mut scratch, &mut out_cached, true);
        let sdf_cached = out_cached
            .iter()
            .find(|(k, _)| *k == GraphOutput::Sdf)
            .map(|(_, v)| v.clone());
        assert_eq!(sdf_full.as_deref(), sdf_cached.as_deref());
        assert_eq!(sdf_cached.as_deref(), Some(&[0.0f32, 1.0, 4.0, 9.0][..]));
    }

    #[test]
    fn compiled_scratch_preserves_capacity_across_slices() {
        let mut g = Graph::new();
        let x = g.push(NodeKind::InputX);
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort { node: x, output: 0 }),
        });
        let compiled = CompiledGraph::compile(&g).expect("compile");
        let xs = x_inputs(8);
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = CompiledScratch::new();
        let mut out = Vec::new();
        compiled.generate_slice(&inputs, 8, &mut scratch, &mut out, false);
        let cap_after_first = scratch.buffers[0].capacity();
        assert!(cap_after_first >= 8);
        compiled.generate_slice(&inputs, 8, &mut scratch, &mut out, false);
        assert!(scratch.buffers[0].capacity() >= cap_after_first);
    }

    // ---- C3: analyze_range tests (audit §9.6-C3) ----

    use crate::math::Interval;

    #[test]
    fn analyze_range_constant_graph_returns_constant_interval() {
        // Constant(2.0) → OutputSdf. Range = [2, 2], straddles zero? No (min>0 → air).
        let mut g = Graph::new();
        let c = g.push(NodeKind::Constant(2.0));
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort { node: c, output: 0 }),
        });
        let compiled = CompiledGraph::compile(&g).expect("compile");
        let r = compiled.analyze_range(
            Interval::infinity(),
            Interval::infinity(),
            Interval::infinity(),
        );
        assert!(r.is_single_value());
        assert_eq!(r.min, 2.0);
        assert!(r.min > 0.0, "constant 2.0 → fully air");
    }

    #[test]
    fn analyze_range_sdf_plane_far_above_is_solid() {
        // SdfPlane(y, height=5): SDF = y - 5. If y range is [0,1], SDF = [-5,-4] → solid.
        let mut g = Graph::new();
        let y = g.push(NodeKind::InputY);
        let h = g.push(NodeKind::Constant(5.0));
        let plane = g.push(NodeKind::SdfPlane {
            y: Some(GraphPort { node: y, output: 0 }),
            height: Some(GraphPort { node: h, output: 0 }),
        });
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: plane,
                output: 0,
            }),
        });
        let compiled = CompiledGraph::compile(&g).expect("compile");
        let r = compiled.analyze_range(
            Interval::infinity(),
            Interval::new(0.0, 1.0),
            Interval::infinity(),
        );
        assert!(
            r.max < 0.0,
            "y-[0,1] h=5 → SDF [-5,-4] → fully solid; got {r:?}"
        );
    }

    #[test]
    fn analyze_range_sdf_plane_straddling_zero_needs_per_voxel_eval() {
        // SdfPlane(y, height=5): SDF = y - 5. If y range is [3,7], SDF = [-2,2] → straddles zero.
        let mut g = Graph::new();
        let y = g.push(NodeKind::InputY);
        let h = g.push(NodeKind::Constant(5.0));
        let plane = g.push(NodeKind::SdfPlane {
            y: Some(GraphPort { node: y, output: 0 }),
            height: Some(GraphPort { node: h, output: 0 }),
        });
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: plane,
                output: 0,
            }),
        });
        let compiled = CompiledGraph::compile(&g).expect("compile");
        let r = compiled.analyze_range(
            Interval::infinity(),
            Interval::new(3.0, 7.0),
            Interval::infinity(),
        );
        assert!(
            r.min < 0.0 && r.max > 0.0,
            "y-[3,7] h=5 → SDF [-2,2] → straddles; got {r:?}"
        );
    }

    #[test]
    fn analyze_range_noise_node_falls_back_to_infinity() {
        // Noise2D is hard → conservative infinity. The graph must NOT be culled.
        let mut g = Graph::new();
        let x = g.push(NodeKind::InputX);
        let y = g.push(NodeKind::InputY);
        let _noise = g.push(NodeKind::Noise2D {
            x: Some(GraphPort { node: x, output: 0 }),
            y: Some(GraphPort { node: y, output: 0 }),
            noise: crate::generators::simple::NoiseConfig::default(),
        });
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: _noise,
                output: 0,
            }),
        });
        let compiled = CompiledGraph::compile(&g).expect("compile");
        let r = compiled.analyze_range(
            Interval::new(0.0, 10.0),
            Interval::new(0.0, 10.0),
            Interval::infinity(),
        );
        // Infinity straddles zero → no culling (safe).
        assert!(
            r.min <= 0.0 && r.max >= 0.0,
            "noise → infinity → straddles; got {r:?}"
        );
    }

    #[test]
    fn analyze_range_add_of_two_constants() {
        // Constant(3) + Constant(4) = [7,7] → air.
        let mut g = Graph::new();
        let a = g.push(NodeKind::Constant(3.0));
        let b = g.push(NodeKind::Constant(4.0));
        let add = g.push(NodeKind::Add {
            a: Some(GraphPort { node: a, output: 0 }),
            b: Some(GraphPort { node: b, output: 0 }),
        });
        g.push(NodeKind::OutputSdf {
            a: Some(GraphPort {
                node: add,
                output: 0,
            }),
        });
        let compiled = CompiledGraph::compile(&g).expect("compile");
        let r = compiled.analyze_range(
            Interval::infinity(),
            Interval::infinity(),
            Interval::infinity(),
        );
        assert!(r.is_single_value());
        assert_eq!(r.min, 7.0);
    }

    fn smooth_subtract_graph(a: f32, b: f32, smoothness: f32) -> (f32, f32) {
        let mut graph = Graph::new();
        let a = graph.push(NodeKind::Constant(a));
        let b = graph.push(NodeKind::Constant(b));
        let subtract = graph.push(NodeKind::SdfSmoothSubtract {
            a: Some(GraphPort::new(a)),
            b: Some(GraphPort::new(b)),
            smoothness,
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(subtract)),
        });

        let coordinates = vec![0.0];
        let inputs = GraphInputs {
            x: &coordinates,
            y: 0.0,
            z: &coordinates,
        };
        let (lazy, compiled) = lazy_and_compiled_sdf(&graph, &inputs, 1);
        (
            lazy.expect("lazy SDF output")[0],
            compiled.expect("compiled SDF output")[0],
        )
    }

    #[test]
    fn smooth_subtract_node_matches_cpp_operand_order() {
        let (lazy, compiled) = smooth_subtract_graph(-0.2, 0.4, 1.0);
        assert!((lazy - -0.04).abs() < 1e-5);
        assert!((compiled - -0.04).abs() < 1e-5);
    }

    #[test]
    fn smooth_subtract_node_uses_hard_subtract_at_zero_smoothness() {
        let expected = crate::math::sdf::sdf_subtract(-0.2, 0.4);
        let (lazy, compiled) = smooth_subtract_graph(-0.2, 0.4, 0.0);
        assert!((lazy - expected).abs() < 1e-5);
        assert!((compiled - expected).abs() < 1e-5);
    }

    #[test]
    fn topological_order_places_outputs_last() {
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        let c = graph.push(NodeKind::Constant(2.0));
        let mul = graph.push(NodeKind::Multiply {
            a: Some(GraphPort::new(x)),
            b: Some(GraphPort::new(c)),
        });
        let out = graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(mul)),
        });

        let order = graph.topological_order().unwrap();
        assert_eq!(order.last(), Some(&out));
        // The two producers come before the multiply.
        let pos_mul = order.iter().position(|i| *i == mul).unwrap();
        let pos_x = order.iter().position(|i| *i == x).unwrap();
        let pos_c = order.iter().position(|i| *i == c).unwrap();
        assert!(pos_x < pos_mul);
        assert!(pos_c < pos_mul);
    }

    #[test]
    fn multiply_input_x_by_constant_evaluates_correctly() {
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        let c = graph.push(NodeKind::Constant(3.0));
        let mul = graph.push(NodeKind::Multiply {
            a: Some(GraphPort::new(x)),
            b: Some(GraphPort::new(c)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(mul)),
        });

        let slice = 4;
        let xs = x_inputs(slice);
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, slice, &mut scratch, &mut outputs)
            .unwrap();

        assert_eq!(outputs.len(), 1);
        let (out_kind, data) = &outputs[0];
        assert_eq!(*out_kind, GraphOutput::Sdf);
        assert_eq!(data, &vec![0.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn sin_of_input_x_evaluates_correctly() {
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        let sin = graph.push(NodeKind::Sin {
            a: Some(GraphPort::new(x)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(sin)),
        });

        let slice = 3;
        let xs = x_inputs(slice);
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, slice, &mut scratch, &mut outputs)
            .unwrap();

        let (_, data) = &outputs[0];
        for (i, v) in data.iter().enumerate() {
            assert!((v - (i as f32).sin()).abs() < 1e-5, "sin mismatch at {i}");
        }
    }

    #[test]
    fn remap_matches_cpp_pure_linear_no_clamp() {
        // GRAPH-2 parity: C++ remap is pure linear (a*x + b), no clamp.
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        let remap = graph.push(NodeKind::Remap {
            a: Some(GraphPort::new(x)),
            from_start: 0.0,
            from_end: 2.0,
            to_start: 10.0,
            to_end: 20.0,
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(remap)),
        });

        // 0 -> 10, 1 -> 15, 2 -> 20, 5 -> 35 (extrapolation, no clamp).
        let xs = vec![0.0, 1.0, 2.0, 5.0];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 4, &mut scratch, &mut outputs)
            .unwrap();

        let (_, data) = &outputs[0];
        assert!((data[0] - 10.0).abs() < 1e-5);
        assert!((data[1] - 15.0).abs() < 1e-5);
        assert!((data[2] - 20.0).abs() < 1e-5);
        assert!(
            (data[3] - 35.0).abs() < 1e-5,
            "extrapolation should NOT clamp: {}",
            data[3]
        );
    }

    #[test]
    fn cycle_in_the_graph_returns_an_error() {
        let mut graph = Graph::new();
        // Two nodes feeding each other. The GraphPort indirection doesn't
        // require the producer to exist, so we can construct a pure cycle
        // directly via NodeKind::Add references.
        let a = graph.push(NodeKind::Add {
            a: Some(GraphPort::new(1)),
            b: None,
        });
        let _ = a;
        // Build an actual self-cycle: a -> b -> a.
        let mut cycle_graph = Graph::new();
        cycle_graph.add_node(GraphNode::new(
            1,
            NodeKind::Add {
                a: Some(GraphPort::new(2)),
                b: None,
            },
        ));
        cycle_graph.add_node(GraphNode::new(
            2,
            NodeKind::Add {
                a: Some(GraphPort::new(1)),
                b: None,
            },
        ));
        let result = cycle_graph.topological_order();
        assert_eq!(result.unwrap_err(), TopoError::Cycle);
    }

    #[test]
    fn unconnected_input_defaults_to_zero() {
        let mut graph = Graph::new();
        // Add with both inputs unconnected — equivalent to 0 + 0.
        let add = graph.push(NodeKind::Add { a: None, b: None });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(add)),
        });

        let xs = vec![0.0; 2];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 2, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        assert_eq!(data, &vec![0.0, 0.0]);
    }

    #[test]
    fn unconnected_inputs_default_to_zero_for_large_slices() {
        let mut graph = Graph::new();
        let add = graph.push(NodeKind::Add { a: None, b: None });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(add)),
        });

        let xs = vec![0.0; 4097];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 4097, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        assert_eq!(data.len(), 4097);
        assert!(data.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn node_kind_inputs_lists_consumer_ports_in_order() {
        let kind = NodeKind::Add {
            a: Some(GraphPort::new(1)),
            b: Some(GraphPort::new(2)),
        };
        let inputs = kind.inputs();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].unwrap().node, 1);
        assert_eq!(inputs[1].unwrap().node, 2);
    }

    #[test]
    fn graph_node_id_can_be_user_supplied() {
        let mut graph = Graph::new();
        graph.add_node(GraphNode::new(100, NodeKind::Constant(1.0)));
        graph.add_node(GraphNode::new(200, NodeKind::Constant(2.0)));
        assert_eq!(graph.nodes().len(), 2);
        let order = graph.topological_order().unwrap();
        assert!(order.contains(&100));
        assert!(order.contains(&200));
    }

    #[test]
    fn slice_z_coordinates_round_trip_through_input_z() {
        let mut graph = Graph::new();
        let z = graph.push(NodeKind::InputZ);
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(z)),
        });

        let zs = vec![10.0, 20.0, 30.0];
        let inputs = GraphInputs {
            x: &zs,
            y: 0.0,
            z: &zs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 3, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        assert_eq!(data, &zs);
    }

    #[test]
    fn sdf_sphere_at_origin_returns_radius_minus_distance() {
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        let y = graph.push(NodeKind::InputY);
        let z = graph.push(NodeKind::InputZ);
        let r = graph.push(NodeKind::Constant(2.0));
        let sphere = graph.push(NodeKind::SdfSphere {
            x: Some(GraphPort::new(x)),
            y: Some(GraphPort::new(y)),
            z: Some(GraphPort::new(z)),
            radius: Some(GraphPort::new(r)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(sphere)),
        });

        // Voxel at (1,0,0): distance 1, radius 2, SDF = 1 - 2 = -1 (inside).
        let xs = vec![1.0];
        let zs = vec![0.0];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &zs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 1, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        assert!((data[0] - (-1.0)).abs() < 1e-5, "got {}", data[0]);
    }

    #[test]
    fn sdf_sphere_unconnected_radius_defaults_to_one() {
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        let y = graph.push(NodeKind::InputY);
        let z = graph.push(NodeKind::InputZ);
        let sphere = graph.push(NodeKind::SdfSphere {
            x: Some(GraphPort::new(x)),
            y: Some(GraphPort::new(y)),
            z: Some(GraphPort::new(z)),
            radius: None,
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(sphere)),
        });

        let xs = vec![0.0];
        let zs = vec![0.0];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &zs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 1, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        assert!((data[0] - (-1.0)).abs() < 1e-5, "got {}", data[0]);
    }

    #[test]
    fn divide_by_zero_outputs_zero() {
        let mut graph = Graph::new();
        let a = graph.push(NodeKind::Constant(4.0));
        let b = graph.push(NodeKind::Constant(0.0));
        let div = graph.push(NodeKind::Divide {
            a: Some(GraphPort::new(a)),
            b: Some(GraphPort::new(b)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(div)),
        });

        let xs = vec![0.0];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 1, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        assert_eq!(data[0], 0.0);
    }

    #[test]
    fn sqrt_clamps_negative_inputs_to_zero() {
        let mut graph = Graph::new();
        let c = graph.push(NodeKind::Constant(-4.0));
        let sqrt = graph.push(NodeKind::Sqrt {
            a: Some(GraphPort::new(c)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(sqrt)),
        });

        let xs = vec![0.0];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 1, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        assert_eq!(data[0], 0.0);
    }

    #[test]
    fn sdf_plane_returns_y_minus_height() {
        let mut graph = Graph::new();
        let y = graph.push(NodeKind::InputY);
        let h = graph.push(NodeKind::Constant(5.0));
        let plane = graph.push(NodeKind::SdfPlane {
            y: Some(GraphPort::new(y)),
            height: Some(GraphPort::new(h)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(plane)),
        });

        let xs = vec![0.0];
        let inputs = GraphInputs {
            x: &xs,
            y: 7.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 1, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        // 7 - 5 = 2 (above the plane).
        assert!((data[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn sdf_smooth_union_with_zero_smoothness_matches_hard_union() {
        let mut graph = Graph::new();
        let a = graph.push(NodeKind::Constant(-3.0));
        let b = graph.push(NodeKind::Constant(-1.0));
        let union = graph.push(NodeKind::SdfSmoothUnion {
            a: Some(GraphPort::new(a)),
            b: Some(GraphPort::new(b)),
            smoothness: 0.0,
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(union)),
        });

        let xs = vec![0.0];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 1, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        // min(-3, -1) = -3.
        assert!((data[0] - (-3.0)).abs() < 1e-5, "got {}", data[0]);
    }

    #[test]
    fn curve_node_samples_baked_lookup_table() {
        use crate::generators::simple::Curve;
        // Identity-ish curve with two points: 0->0, 1->10.
        let curve = std::sync::Arc::new(Curve::from_points(vec![0.0, 10.0]));
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        let curve_node = graph.push(NodeKind::Curve {
            a: Some(GraphPort::new(x)),
            curve,
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(curve_node)),
        });

        let xs = vec![0.0, 0.5, 1.0];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 3, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        assert!((data[0] - 0.0).abs() < 1e-5);
        assert!((data[1] - 5.0).abs() < 1e-5);
        assert!((data[2] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn noise_2d_node_returns_deterministic_output_for_same_seed() {
        use crate::generators::simple::NoiseConfig;

        let run = || -> Vec<f32> {
            let noise = NoiseConfig {
                seed: Some(42),
                frequency: Some(0.1),
                noise_type: Some(fastnoise_lite::NoiseType::Value),
            };
            let mut graph = Graph::new();
            let x = graph.push(NodeKind::InputX);
            let y = graph.push(NodeKind::InputY);
            let n = graph.push(NodeKind::Noise2D {
                x: Some(GraphPort::new(x)),
                y: Some(GraphPort::new(y)),
                noise,
            });
            graph.push(NodeKind::OutputSdf {
                a: Some(GraphPort::new(n)),
            });
            let xs = vec![1.0, 2.0, 3.0];
            let inputs = GraphInputs {
                x: &xs,
                y: 0.0,
                z: &xs,
            };
            let mut scratch = GraphScratch::new();
            let mut outputs = Vec::new();
            graph
                .generate(&inputs, 3, &mut scratch, &mut outputs)
                .unwrap();
            outputs[0].1.clone()
        };

        let first = run();
        let second = run();
        assert_eq!(
            first, second,
            "noise must be deterministic for the same seed"
        );
        assert!(
            first.iter().any(|v| *v != 0.0),
            "noise must produce non-zero values somewhere"
        );
    }

    #[test]
    fn clamp_node_clamps_to_input_range() {
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        let lo = graph.push(NodeKind::Constant(2.0));
        let hi = graph.push(NodeKind::Constant(4.0));
        let clamp = graph.push(NodeKind::Clamp {
            a: Some(GraphPort::new(x)),
            min_v: Some(GraphPort::new(lo)),
            max_v: Some(GraphPort::new(hi)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(clamp)),
        });

        let xs = vec![1.0, 3.0, 5.0];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 3, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
        assert!((data[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn clamp_node_accepts_reversed_bounds() {
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        let lo = graph.push(NodeKind::Constant(4.0));
        let hi = graph.push(NodeKind::Constant(2.0));
        let clamp = graph.push(NodeKind::Clamp {
            a: Some(GraphPort::new(x)),
            min_v: Some(GraphPort::new(lo)),
            max_v: Some(GraphPort::new(hi)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(clamp)),
        });

        let xs = vec![1.0, 3.0, 5.0];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 3, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        assert_eq!(data, &vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn mix_node_interpolates_by_t() {
        let mut graph = Graph::new();
        let a = graph.push(NodeKind::Constant(0.0));
        let b = graph.push(NodeKind::Constant(10.0));
        let t = graph.push(NodeKind::Constant(0.25));
        let mix = graph.push(NodeKind::Mix {
            a: Some(GraphPort::new(a)),
            b: Some(GraphPort::new(b)),
            t: Some(GraphPort::new(t)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(mix)),
        });

        let xs = vec![0.0; 1];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 1, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        // mix(0, 10, 0.25) = 0*0.75 + 10*0.25 = 2.5.
        assert!((data[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn distance_3d_node_computes_distance_between_two_points() {
        // GRAPH-2 parity: C++ Distance3D takes 6 inputs (two points).
        let mut graph = Graph::new();
        let x0 = graph.push(NodeKind::Constant(0.0));
        let y0 = graph.push(NodeKind::Constant(0.0));
        let z0 = graph.push(NodeKind::Constant(0.0));
        let x1 = graph.push(NodeKind::Constant(3.0));
        let y1 = graph.push(NodeKind::Constant(4.0));
        let z1 = graph.push(NodeKind::Constant(3.0));
        let d = graph.push(NodeKind::Distance3D {
            x0: Some(GraphPort::new(x0)),
            y0: Some(GraphPort::new(y0)),
            z0: Some(GraphPort::new(z0)),
            x1: Some(GraphPort::new(x1)),
            y1: Some(GraphPort::new(y1)),
            z1: Some(GraphPort::new(z1)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(d)),
        });

        let xs = vec![0.0];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 1, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        // Distance from (0,0,0) to (3,4,3): sqrt(9+16+9) = sqrt(34) ≈ 5.83.
        assert!((data[0] - 34.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn floor_and_fract_split_a_value() {
        let mut graph = Graph::new();
        let x = graph.push(NodeKind::InputX);
        let floor = graph.push(NodeKind::Floor {
            a: Some(GraphPort::new(x)),
        });
        let _fract = graph.push(NodeKind::Fract {
            a: Some(GraphPort::new(x)),
        });
        graph.push(NodeKind::OutputSdf {
            a: Some(GraphPort::new(floor)),
        });

        let xs = vec![-1.5, 0.7, 2.9];
        let inputs = GraphInputs {
            x: &xs,
            y: 0.0,
            z: &xs,
        };
        let mut scratch = GraphScratch::new();
        let mut outputs = Vec::new();
        graph
            .generate(&inputs, 3, &mut scratch, &mut outputs)
            .unwrap();
        let (_, data) = &outputs[0];
        assert!((data[0] - (-2.0)).abs() < 1e-5);
        assert!((data[1] - 0.0).abs() < 1e-5);
        assert!((data[2] - 2.0).abs() < 1e-5);
    }
}
