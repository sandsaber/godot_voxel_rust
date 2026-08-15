//! The [`VoxelMesher`] trait + shared input/output types.
//!
//! Ported from `meshers/voxel_mesher.h` (the abstract base class). This is
//! the engine-agnostic seam every mesher plugs into so the terrain meshing
//! pipeline can drive transvoxel / cubes / blocky uniformly. The C++ base is
//! a Godot `Resource` carrying material/`Ref<Mesh>`/`Ref<Image>` state —
//! those live in `voxel-gdext` later; only the algorithmic contract is here.
//!
//! ## Output shape
//! [`MesherOutput`] carries a list of [`Surface`]s, each wrapping one of the
//! existing per-mesher array structs ([`MeshArrays`] for transvoxel,
//! [`CubesArrays`] / [`BlockyArrays`] for the blocky family) behind the
//! [`SurfaceArrays`] enum. This keeps the trait object-safe while letting
//! each mesher emit its native attribute layout.

use crate::generators::base::VoxelGenerator;
use crate::math::Vector3i;
use crate::meshers::blocky::mesher::BlockyArrays;
use crate::meshers::cubes::arrays::CubesArrays;
use crate::meshers::transvoxel::structures::MeshArrays;
use crate::storage::VoxelBuffer;

/// Input handed to [`VoxelMesher::build`]. Mirrors `VoxelMesher::Input`.
// Manual `Debug` because `&dyn VoxelGenerator` is not `Debug`.
pub struct MesherInput<'a> {
    /// Voxels to be used as the primary source of data.
    pub voxels: &'a VoxelBuffer,
    /// When using LOD, some meshers can use the generator and edited voxels
    /// to refine results. If `None`, the mesher only uses `voxels`.
    pub generator: Option<&'a dyn VoxelGenerator>,
    /// Origin of the block, required when doing deep sampling.
    pub origin_in_voxels: Vector3i,
    /// LOD index. 0 means highest detail; 1 means half detail, etc.
    pub lod_index: u8,
    /// If `true`, collision information is required. Some meshers return a
    /// separate collision surface; others reuse the render mesh.
    pub collision_hint: bool,
    /// If `true`, the mesh will be used in a variable-LOD context (e.g.
    /// transition meshes may or may not be generated).
    pub lod_hint: bool,
    /// Optional free-list pool of [`MeshArrays`] buffers (audit §9.6-B3). A
    /// mesher that supports reuse (e.g. [`TransvoxelMesher`](crate::meshers::TransvoxelMesher))
    /// acquires a cleared buffer from here instead of allocating fresh, and
    /// the terrain core returns the previous block's arrays when re-meshing
    /// or unloading. `None` falls back to per-build allocation.
    pub mesh_arrays_pool: Option<&'a MeshArraysPool>,
}

impl<'a> std::fmt::Debug for MesherInput<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MesherInput")
            .field("voxels", &self.voxels)
            .field(
                "generator",
                &self.generator.as_ref().map(|_| "<dyn VoxelGenerator>"),
            )
            .field("origin_in_voxels", &self.origin_in_voxels)
            .field("lod_index", &self.lod_index)
            .field("collision_hint", &self.collision_hint)
            .field("lod_hint", &self.lod_hint)
            .field(
                "mesh_arrays_pool",
                &self.mesh_arrays_pool.map(|_| "<MeshArraysPool>"),
            )
            .finish()
    }
}

impl<'a> MesherInput<'a> {
    /// Convenience constructor matching the most common call sites.
    pub fn new(voxels: &'a VoxelBuffer, origin_in_voxels: Vector3i, lod_index: u8) -> Self {
        Self {
            voxels,
            generator: None,
            origin_in_voxels,
            lod_index,
            collision_hint: false,
            lod_hint: false,
            mesh_arrays_pool: None,
        }
    }
}

/// Per-mesher mesh attribute layout. Each variant wraps the existing array
/// struct the mesher already produces, so converting a mesher to the trait
/// is a thin adapter rather than a rewrite.
#[derive(Debug, Default)]
pub enum SurfaceArrays {
    /// Transvoxel output (positions/normals/LOD attribs/indices).
    Transvoxel(MeshArrays),
    /// Cubes greedy/simple output (positions/normals/colors/uvs/indices).
    Cubes(CubesArrays),
    /// Blocky output (positions/normals/uvs/colors/indices/tangents).
    Blocky(BlockyArrays),
    /// Empty default — used by [`MesherOutput::default`] and cleared surfaces.
    #[default]
    Empty,
}

impl SurfaceArrays {
    /// Vertex count regardless of the active variant.
    pub fn vertex_count(&self) -> usize {
        match self {
            Self::Transvoxel(a) => a.vertices.len(),
            Self::Cubes(a) => a.positions.len(),
            Self::Blocky(a) => a.vertex_count(),
            Self::Empty => 0,
        }
    }

    /// Triangle count regardless of the active variant.
    pub fn triangle_count(&self) -> usize {
        let indices = match self {
            Self::Transvoxel(a) => &a.indices,
            Self::Cubes(a) => &a.indices,
            Self::Blocky(a) => &a.indices,
            Self::Empty => return 0,
        };
        indices.len() / 3
    }

    /// Resets to [`SurfaceArrays::Empty`], dropping any allocated buffers.
    pub fn clear(&mut self) {
        *self = Self::Empty;
    }
}

/// One material-grouped surface of a mesher's output. Mirrors
/// `VoxelMesher::Output::Surface` minus the Godot `Array` (we keep the
/// strongly-typed Rust array struct instead).
#[derive(Debug, Default)]
pub struct Surface {
    /// The mesh attribute arrays for this surface.
    pub arrays: SurfaceArrays,
    /// Material slot the mesher picked for this surface (transvoxel always 0;
    /// cubes/blocky split opaque vs transparent).
    pub material_index: u16,
}

impl Surface {
    /// Convenience constructor.
    pub fn new(arrays: SurfaceArrays, material_index: u16) -> Self {
        Self {
            arrays,
            material_index,
        }
    }

    /// `true` when the surface carries no geometry.
    pub fn is_empty(&self) -> bool {
        self.arrays.triangle_count() == 0
    }
}

/// Collision geometry a mesher may produce independently from visual surfaces.
/// Explicit [`Self::positions`] and [`Self::indices`] remain valid when the
/// visual surface list is empty. Alternatively, paired non-negative
/// `submesh_*_end` values select the regular prefix of the first render surface.
/// Mirrors `VoxelMesher::Output::CollisionSurface`.
#[derive(Debug)]
pub struct CollisionSurface {
    pub positions: Vec<crate::math::Vector3f>,
    pub indices: Vec<i32>,
    /// If `>= 0`, the collision surface may actually be a sub-section of the
    /// first render surface (vertex/index range). Defaults to `-1`
    /// (sentinel meaning "no sub-section"), matching the C++ initialiser.
    pub submesh_vertex_end: i32,
    pub submesh_index_end: i32,
}

impl Default for CollisionSurface {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            indices: Vec::new(),
            submesh_vertex_end: -1,
            submesh_index_end: -1,
        }
    }
}

/// Output of [`VoxelMesher::build`]. Mirrors `VoxelMesher::Output` minus
/// Godot-specific fields (`Array shadow_occluder`, `Ref<Image> atlas_image`).
#[derive(Debug, Default)]
pub struct MesherOutput {
    /// Material-grouped render surfaces.
    pub surfaces: Vec<Surface>,
    /// Optional collision surface (only populated when `collision_hint` is
    /// set and the mesher produces one). This payload is independent from
    /// [`Self::surfaces`], so collision-only output is representable.
    pub collision_surface: CollisionSurface,
}

impl MesherOutput {
    /// `true` when no render surface carries any geometry. Collision geometry
    /// is intentionally not considered by this visual-only query.
    pub fn is_empty(&self) -> bool {
        self.surfaces.iter().all(Surface::is_empty)
    }

    /// Total triangle count across every surface.
    pub fn total_triangle_count(&self) -> usize {
        self.surfaces
            .iter()
            .map(|s| s.arrays.triangle_count())
            .sum()
    }

    /// Total vertex count across every surface.
    pub fn total_vertex_count(&self) -> usize {
        self.surfaces.iter().map(|s| s.arrays.vertex_count()).sum()
    }

    /// Reset to a fresh state, reusing allocations.
    pub fn clear(&mut self) {
        for surface in &mut self.surfaces {
            surface.arrays.clear();
        }
        self.surfaces.clear();
        self.collision_surface.positions.clear();
        self.collision_surface.indices.clear();
        self.collision_surface.submesh_vertex_end = -1;
        self.collision_surface.submesh_index_end = -1;
    }

    /// Extract the first transvoxel [`MeshArrays`] from the surfaces, replacing
    /// it with an empty default. Used by the terrain core to return a mesher's
    /// output buffers to the [`MeshArraysPool`] once the block is re-meshed or
    /// unloaded. Non-transvoxel surfaces and extra surfaces are left untouched
    /// (only the first transvoxel slot is pooled, matching the single-surface
    /// contract every builtin mesher emits today).
    pub fn take_first_transvoxel_arrays(&mut self) -> Option<MeshArrays> {
        let surface = self
            .surfaces
            .iter_mut()
            .find(|s| matches!(s.arrays, SurfaceArrays::Transvoxel(_)))?;
        let arrays = if let SurfaceArrays::Transvoxel(a) =
            std::mem::replace(&mut surface.arrays, SurfaceArrays::Empty)
        {
            a
        } else {
            return None;
        };
        Some(arrays)
    }
}

/// Free-list pool of reusable [`MeshArrays`] buffers (audit §9.6-B3).
///
/// Mesher outputs move out of the task into the terrain core's mesh map, so a
/// mesher-level `thread_local` cannot keep the buffers. Instead the pool lives
/// at the terrain level: [`TransvoxelMesher::build`](crate::meshers::TransvoxelMesher)
/// acquires a `MeshArrays` from the pool (clearing it before refilling), and
/// the terrain core returns the previous block's arrays when it re-meshes or
/// unloads a block. Capacity stabilises after the first few dozen blocks.
///
/// Thread-safe via a `Mutex<Vec<_>>`; contention is bounded because each task
/// only holds the lock for the `pop`/`push` of a single buffer.
#[derive(Debug, Default)]
pub struct MeshArraysPool {
    free: std::sync::Mutex<Vec<MeshArrays>>,
}

impl MeshArraysPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a reusable `MeshArrays` from the free-list, or allocate a fresh one
    /// if the pool is empty. The returned buffer is **cleared** (length zero,
    /// capacity preserved) so the caller can `build_regular_mesh` straight into
    /// it without double-filling.
    pub fn acquire(&self) -> MeshArrays {
        let mut arrays = self
            .free
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .unwrap_or_default();
        arrays.clear();
        arrays
    }

    /// Return a `MeshArrays` to the free-list for reuse. The buffer is cleared
    /// first so it does not hold stale geometry while idle.
    pub fn release(&self, mut arrays: MeshArrays) {
        arrays.clear();
        self.free
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(arrays);
    }

    /// Number of idle buffers currently held. Diagnostic only.
    pub fn idle_count(&self) -> usize {
        self.free.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

/// A voxel mesher: converts a [`VoxelBuffer`] into triangle-mesh surfaces.
///
/// Ported from the C++ `VoxelMesher` virtual base. Implementations may be
/// invoked from worker threads, hence the `Send + Sync` bound. `build`
/// appends surfaces into the provided [`MesherOutput`] (the C++ contract is
/// the same — the output is reused across invocations and callers `clear()`
/// it themselves when appropriate).
pub trait VoxelMesher: Send + Sync {
    /// Build mesh surfaces from `input.voxels`, appending to `output`.
    fn build(&self, output: &mut MesherOutput, input: &MesherInput<'_>);

    /// How many neighbor voxels the mesher needs to access toward the
    /// negative axes. If callers don't provide this much padding, the mesher
    /// may produce seams at block edges.
    fn minimum_padding(&self) -> u32 {
        0
    }

    /// How many neighbor voxels the mesher needs to access toward the
    /// positive axes.
    fn maximum_padding(&self) -> u32 {
        0
    }

    /// Bitmask of channels this mesher uses (1 << channel_index). The terrain
    /// uses this to decide which channels to load/materialize before meshing.
    fn used_channels_mask(&self) -> u32 {
        0
    }

    /// `true` if this mesher supports LOD-aware meshing.
    fn supports_lod(&self) -> bool {
        true
    }

    /// `true` if the mesher populates the collision contract in
    /// [`MesherOutput::collision_surface`], either with explicit geometry or a
    /// regular-render-mesh prefix. If `false`, callers may choose to reuse the
    /// complete render mesh as a collider.
    fn is_generating_collision_surface(&self) -> bool {
        false
    }
}

/// Shared mesher storage. Implementations own any scratch storage they need,
/// so worker tasks can call `build` concurrently through one shared handle.
pub type SharedVoxelMesher = std::sync::Arc<dyn VoxelMesher>;

#[cfg(test)]
mod tests {
    use super::{CollisionSurface, MesherInput, MesherOutput, Surface, SurfaceArrays, VoxelMesher};
    use crate::generators::base::{GenResult, VoxelGenerator, VoxelQueryData};
    use crate::math::{Vector3f, Vector3i};
    use crate::meshers::transvoxel::structures::MeshArrays;
    use crate::storage::{ChannelId, VoxelBuffer};

    /// A mesher that emits a fixed single-triangle transvoxel surface, used
    /// to exercise the trait plumbing without depending on real meshing math.
    struct StubMesher;
    impl VoxelMesher for StubMesher {
        fn build(&self, output: &mut MesherOutput, _input: &MesherInput<'_>) {
            let mut arrays = MeshArrays::default();
            let a = arrays.add_vertex(
                Vector3f::new(0.0, 0.0, 0.0),
                Vector3f::new(0.0, 1.0, 0.0),
                0,
                0,
                0,
                Vector3f::zero(),
            );
            let b = arrays.add_vertex(
                Vector3f::new(1.0, 0.0, 0.0),
                Vector3f::new(0.0, 1.0, 0.0),
                0,
                0,
                0,
                Vector3f::zero(),
            );
            let c = arrays.add_vertex(
                Vector3f::new(0.0, 0.0, 1.0),
                Vector3f::new(0.0, 1.0, 0.0),
                0,
                0,
                0,
                Vector3f::zero(),
            );
            arrays.indices.extend_from_slice(&[a, b, c]);
            output
                .surfaces
                .push(Surface::new(SurfaceArrays::Transvoxel(arrays), 0));
        }

        fn used_channels_mask(&self) -> u32 {
            1 << ChannelId::Sdf.index()
        }
    }

    #[test]
    fn build_appends_a_surface_with_geometry() {
        let mesher = StubMesher;
        let voxels = VoxelBuffer::with_size(Vector3i::splat(2));
        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        assert!(!output.is_empty());
        assert_eq!(output.total_vertex_count(), 3);
        assert_eq!(output.total_triangle_count(), 1);
        assert_eq!(output.surfaces.len(), 1);
        assert_eq!(output.surfaces[0].material_index, 0);
    }

    #[test]
    fn clear_resets_output_for_reuse() {
        let mesher = StubMesher;
        let voxels = VoxelBuffer::with_size(Vector3i::splat(2));
        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);
        assert!(!output.is_empty());

        output.clear();
        assert!(output.is_empty());
        assert!(output.surfaces.is_empty());
        // Reusing the same output for another build works.
        mesher.build(&mut output, &input);
        assert_eq!(output.total_triangle_count(), 1);
    }

    #[test]
    fn surface_arrays_variant_counts_agree_with_native_struct() {
        let mut arrays = MeshArrays::default();
        arrays.add_vertex(
            Vector3f::zero(),
            Vector3f::new(0.0, 1.0, 0.0),
            0,
            0,
            0,
            Vector3f::zero(),
        );
        arrays.indices.push(0);
        let wrapped = SurfaceArrays::Transvoxel(arrays);
        assert_eq!(wrapped.vertex_count(), 1);
        // One index is not a full triangle (3), so triangle_count is 0.
        assert_eq!(wrapped.triangle_count(), 0);
    }

    #[test]
    fn collision_surface_defaults_match_cpp_sentinels() {
        let cs = CollisionSurface::default();
        assert!(cs.positions.is_empty());
        assert!(cs.indices.is_empty());
        assert_eq!(cs.submesh_vertex_end, -1);
        assert_eq!(cs.submesh_index_end, -1);
    }

    #[test]
    fn collision_output_is_preserved_when_visual_surface_is_empty() {
        struct CollisionOnlyMesher;
        impl VoxelMesher for CollisionOnlyMesher {
            fn build(&self, output: &mut MesherOutput, _input: &MesherInput<'_>) {
                output.collision_surface.positions.extend([
                    Vector3f::new(0.0, 0.0, 0.0),
                    Vector3f::new(1.0, 0.0, 0.0),
                    Vector3f::new(0.0, 1.0, 0.0),
                ]);
                output.collision_surface.indices.extend([0, 1, 2]);
            }
        }

        let voxels = VoxelBuffer::with_size(Vector3i::splat(2));
        let mut input = MesherInput::new(&voxels, Vector3i::zero(), 0);
        input.collision_hint = true;
        let mut output = MesherOutput::default();

        CollisionOnlyMesher.build(&mut output, &input);

        assert!(output.is_empty());
        assert!(output.surfaces.is_empty());
        assert_eq!(output.collision_surface.positions.len(), 3);
        assert_eq!(output.collision_surface.indices, [0, 1, 2]);
    }

    /// Sanity: the trait object can be boxed and dispatched dynamically,
    /// which is how the mesh block task will hold a `Box<dyn VoxelMesher>`.
    #[test]
    fn boxed_dyn_dispatch_works() {
        let mesher: Box<dyn VoxelMesher> = Box::new(StubMesher);
        let voxels = VoxelBuffer::with_size(Vector3i::splat(2));
        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);
        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);
        assert_eq!(output.total_triangle_count(), 1);
        assert_eq!(mesher.used_channels_mask(), 1 << ChannelId::Sdf.index());
    }

    /// Some meshers consult the generator during build (LOD-affined sampling).
    /// Verify the input's generator slot round-trips through the build call.
    #[test]
    fn input_generator_is_reachable_inside_build() {
        struct ProbingMesher;
        impl VoxelMesher for ProbingMesher {
            fn build(&self, _output: &mut MesherOutput, input: &MesherInput<'_>) {
                assert!(input.generator.is_some());
            }
        }
        struct DummyGen;
        impl VoxelGenerator for DummyGen {
            fn generate_block(&self, _input: VoxelQueryData<'_>) -> GenResult {
                GenResult::default()
            }
        }

        let mesher = ProbingMesher;
        let voxels = VoxelBuffer::with_size(Vector3i::splat(2));
        let gen = DummyGen;
        let input = MesherInput {
            generator: Some(&gen),
            ..MesherInput::new(&voxels, Vector3i::zero(), 0)
        };
        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);
    }
}
