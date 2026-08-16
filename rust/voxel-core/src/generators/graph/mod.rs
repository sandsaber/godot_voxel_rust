//! `generators::graph` — runtime for the voxel procedural-graph generator.
//!
//! Engine-agnostic port of `generators/graph/` minus the Godot editor/GUI,
//! GPU shader emission, and (for now) range analysis. The runtime takes a
//! user-built [`Graph`] of typed nodes (inputs, math ops, outputs) and
//! executes it inside a [`VoxelGenerator`] to fill a [`VoxelBuffer`] block.
//!
//! ## Approach
//!
//! The C++ implementation compiles the graph into a flat `Vec<uint16_t>`
//! bytecode VM. This Rust port uses a simpler **AST walker**: each
//! [`GraphNode`] carries an `Op` enum whose variants bundle parameters and
//! references to upstream node ids; execution walks the graph in
//! topological order, writing f32 slices into a per-thread scratch. The
//! walker is fast enough for headless use and tests, and the bytecode VM
//! can be layered in later as an optimisation without changing the
//! user-facing `Graph` API.
//!
//! ## Status
//!
//! Node set includes math/SDF/noise, `Curve`, `Image2D`, and `Expression`.
//! Range analysis is still conservative for hard nodes.

pub mod expression_node;
pub mod generator_graph;
pub mod image;
pub mod runtime;

pub use generator_graph::GraphGenerator;
pub use runtime::{
    node_kind_from_spec, optional_graph_port, CompiledGraph, CompiledScratch, Graph, GraphInputs,
    GraphNode, GraphNodeId, GraphOutput, GraphParam, GraphPort, GraphScratch, NodeKind, TopoError,
};
