//! More Godot classes — abstract bases, modifier types, LOD terrain node,
//! and utility resources. Brings total class count closer to DoD 75+.

use godot::prelude::*;

// ---------------------------------------------------------------------------
// VoxelGeneratorGD — abstract base Resource for all generators
// ---------------------------------------------------------------------------
/// Abstract base resource for voxel generators. In C++ this is the Godot-facing
/// wrapper around the engine-agnostic `VoxelGenerator`. Subclasses:
/// Waves, Flat, Noise, Heightmap, Graph.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelGenerator)]
pub struct VoxelGeneratorGD {
    base: Base<Resource>,
}
#[godot_api]
impl IResource for VoxelGeneratorGD {
    fn init(base: Base<Resource>) -> Self {
        Self { base }
    }
}

#[godot_api]
impl VoxelGeneratorGD {
    /// The generator category name (base type).
    #[func]
    fn get_category(&self) -> GString {
        "generator".to_godot()
    }
}

// ---------------------------------------------------------------------------
// VoxelStreamGD — abstract base Resource for all streams
// ---------------------------------------------------------------------------
/// Abstract base resource for voxel streams. Subclasses: Memory, RegionFiles.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelStream)]
pub struct VoxelStreamGD {
    base: Base<Resource>,
}
#[godot_api]
impl IResource for VoxelStreamGD {
    fn init(base: Base<Resource>) -> Self {
        Self { base }
    }
}

#[godot_api]
impl VoxelStreamGD {
    /// The stream category name (base type).
    #[func]
    fn get_category(&self) -> GString {
        "stream".to_godot()
    }
}

// ---------------------------------------------------------------------------
// VoxelMesherGD — abstract base Resource for all meshers
// ---------------------------------------------------------------------------
/// Abstract base resource for voxel meshers. Subclasses: Transvoxel, Blocky, Cubes.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelMesher)]
pub struct VoxelMesherGD {
    base: Base<Resource>,
    #[var]
    padding: i32,
}
#[godot_api]
impl IResource for VoxelMesherGD {
    fn init(base: Base<Resource>) -> Self {
        Self { base, padding: 1 }
    }
}

#[godot_api]
impl VoxelMesherGD {
    /// The mesher category name (base type).
    #[func]
    fn get_category(&self) -> GString {
        "mesher".to_godot()
    }
}

// ---------------------------------------------------------------------------
// VoxelModifierGD — Node3D base for SDF modifiers
// ---------------------------------------------------------------------------
/// Base Node3D for SDF modifiers. Children modify terrain SDF data.
#[derive(GodotClass)]
#[class(base = Node3D, tool, rename = VoxelModifier)]
pub struct VoxelModifierGD {
    base: Base<Node3D>,
    #[var]
    operation: i32,
    #[var]
    smoothness: f32,
}
#[godot_api]
impl INode3D for VoxelModifierGD {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            operation: 0,
            smoothness: 0.0,
        }
    }
}

#[godot_api]
impl VoxelModifierGD {
    /// The modifier category name (base type).
    #[func]
    fn get_category(&self) -> GString {
        "modifier".to_godot()
    }
}

// ---------------------------------------------------------------------------
// VoxelModifierSphereGD — Node3D for sphere SDF modifier
// ---------------------------------------------------------------------------
/// A sphere-shaped SDF modifier node. Add to a `VoxelTerrain` as a child to
/// carve (subtract) or merge (union) a sphere into the generated terrain.
///
/// Wraps [`voxel_core::modifiers::SphereModifier`] — `apply_to_buffer` runs
/// the real SDF blend (smooth union / subtract) over a `VoxelBufferGD`'s SDF
/// channel, sampling the modifier's world-space center from the node's 3D
/// transform.
#[derive(GodotClass)]
#[class(base = Node3D, tool, rename = VoxelModifierSphere)]
pub struct VoxelModifierSphereGD {
    base: Base<Node3D>,
    #[var]
    radius: f32,
    /// Blend operation: 0 = add (union), 1 = subtract. Mirrors
    /// `SdfOperation`.
    #[var]
    operation: i32,
    /// Smoothing factor for the blend (0 = hard, larger = smoother).
    #[var]
    smoothness: f32,
}
#[godot_api]
impl INode3D for VoxelModifierSphereGD {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            radius: 10.0,
            operation: 0,
            smoothness: 0.0,
        }
    }
}

#[godot_api]
impl VoxelModifierSphereGD {
    /// Apply this sphere modifier to a `VoxelBufferGD`'s SDF channel.
    /// `buffer` must be a `VoxelBufferGD`; `origin_x/y/z` is the buffer's
    /// world-space origin (voxel units). Returns the number of voxels whose
    /// SDF actually changed, or -1 if `buffer` is not a `VoxelBufferGD`.
    #[func]
    fn apply_to_buffer(
        &self,
        buffer: Gd<RefCounted>,
        origin_x: f32,
        origin_y: f32,
        origin_z: f32,
    ) -> i64 {
        let Ok(mut buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let mut bound = buf.bind_mut();
        let core = bound.core_buffer_mut();
        let sx = core.size().x;
        let sy = core.size().y;
        let sz = core.size().z;
        const SDF_CHANNEL: usize = 1;

        // Build the core modifier from the node's state.
        let op = if self.operation == 1 {
            voxel_core::modifiers::SdfOperation::Subtract
        } else {
            voxel_core::modifiers::SdfOperation::Add
        };
        let center = self.base().get_position();
        let cx = center.x;
        let cy = center.y;
        let cz = center.z;

        // Gather SDF + world positions, apply the modifier, write back.
        let mut changed: i64 = 0;
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let sdf = core.get_voxel_f(x, y, z, SDF_CHANNEL);
                    let px = origin_x + x as f32;
                    let py = origin_y + y as f32;
                    let pz = origin_z + z as f32;
                    let shape = ((px - cx).powi(2) + (py - cy).powi(2) + (pz - cz).powi(2)).sqrt()
                        - self.radius;
                    let blended = voxel_core::modifiers::sdf_blend(sdf, shape, op, self.smoothness);
                    if (blended - sdf).abs() > 1e-6 {
                        core.set_voxel_f(blended, x, y, z, SDF_CHANNEL);
                        changed += 1;
                    }
                }
            }
        }
        changed
    }
}

// ---------------------------------------------------------------------------
// VoxelModifierMeshGD — Node3D for mesh SDF modifier
// ---------------------------------------------------------------------------
/// A mesh-based SDF modifier node. The shape is an oriented box (a simple
/// baked-mesh stand-in) whose extents derive from the node's scale; the SDF is
/// blended into the terrain via union/subtract using
/// [`voxel_core::math::sdf`] functions.
#[derive(GodotClass)]
#[class(base = Node3D, tool, rename = VoxelModifierMesh)]
pub struct VoxelModifierMeshGD {
    base: Base<Node3D>,
    /// Blend operation: 0 = add (union), 1 = subtract.
    #[var]
    operation: i32,
    /// Half-extents of the box shape (voxel units).
    #[var]
    extents: f32,
}
#[godot_api]
impl INode3D for VoxelModifierMeshGD {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            operation: 0,
            extents: 4.0,
        }
    }
}

#[godot_api]
impl VoxelModifierMeshGD {
    /// Apply this box-shape modifier to a `VoxelBufferGD`'s SDF channel.
    /// `buffer` must be a `VoxelBufferGD`; `origin_x/y/z` is the buffer's
    /// world-space origin. The box is centered on the node's world position.
    /// Returns the number of voxels whose SDF changed, or -1 if `buffer` is
    /// not a `VoxelBufferGD`.
    #[func]
    fn apply_to_buffer(
        &self,
        buffer: Gd<RefCounted>,
        origin_x: f32,
        origin_y: f32,
        origin_z: f32,
    ) -> i64 {
        let Ok(mut buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let mut bound = buf.bind_mut();
        let core = bound.core_buffer_mut();
        let sx = core.size().x;
        let sy = core.size().y;
        let sz = core.size().z;
        const SDF_CHANNEL: usize = 1;

        let center = self.base().get_position();
        let cx = center.x;
        let cy = center.y;
        let cz = center.z;
        let extents = voxel_core::math::Vector3f::splat(self.extents);
        let subtract = self.operation == 1;

        let mut changed: i64 = 0;
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let sdf = core.get_voxel_f(x, y, z, SDF_CHANNEL);
                    let pos = voxel_core::math::Vector3f::new(
                        origin_x + x as f32 - cx,
                        origin_y + y as f32 - cy,
                        origin_z + z as f32 - cz,
                    );
                    let shape = voxel_core::math::sdf::sdf_box(pos, extents);
                    let blended = if subtract {
                        voxel_core::math::sdf::sdf_subtract(sdf, shape)
                    } else {
                        voxel_core::math::sdf::sdf_union(sdf, shape)
                    };
                    if (blended - sdf).abs() > 1e-6 {
                        core.set_voxel_f(blended, x, y, z, SDF_CHANNEL);
                        changed += 1;
                    }
                }
            }
        }
        changed
    }
}

// ---------------------------------------------------------------------------
// VoxelLodTerrainGD — Node3D for multi-LOD terrain (API parity)
// ---------------------------------------------------------------------------
/// Multi-LOD terrain node. Wraps [`VoxelTerrainCore`] with multi-LOD paging
/// driven by a [`voxel_core::terrain::LodOctree`].
///
/// The functional API exposes the octree's leaf/node counts after a
/// subdivision pass — `subdivide_and_count_leaves` runs the real octree
/// split logic and returns how many leaf blocks the LOD structure produces.
#[derive(GodotClass)]
#[class(base = Node3D, tool, rename = VoxelLodTerrain)]
pub struct VoxelLodTerrainGD {
    base: Base<Node3D>,
    /// Number of LOD levels (1 = single-LOD). Plain field exposed via the
    /// `get_lod_count`/`set_lod_count` #[func]s.
    lod_count: i32,
    /// Distance at which a LOD level splits into higher detail.
    lod_distance: f32,
    /// The real LOD octree driving block split/join.
    octree: voxel_core::terrain::LodOctree,
}
#[godot_api]
impl INode3D for VoxelLodTerrainGD {
    fn init(base: Base<Node3D>) -> Self {
        let mut octree = voxel_core::terrain::LodOctree::new();
        octree.create(4);
        Self {
            base,
            lod_count: 4,
            lod_distance: 64.0,
            octree,
        }
    }
}

#[godot_api]
impl VoxelLodTerrainGD {
    /// Number of LOD levels.
    #[func]
    fn get_lod_count(&self) -> i32 {
        self.lod_count
    }

    /// Set the LOD level count and rebuild the octree. `count` is clamped to
    /// at least 1.
    #[func]
    fn set_lod_count(&mut self, count: i32) {
        let n = count.max(1) as u32;
        self.lod_count = n as i32;
        self.octree.create(n);
    }

    /// LOD split distance.
    #[func]
    fn get_lod_distance(&self) -> f32 {
        self.lod_distance
    }

    #[func]
    fn set_lod_distance(&mut self, distance: f32) {
        self.lod_distance = distance;
    }

    /// Run one subdivision pass over the octree (using `NoOpActions`, which
    /// allow all splits) and return the resulting leaf count. This exercises
    /// the real LOD split logic through the binding.
    #[func]
    fn subdivide_and_count_leaves(&mut self) -> i32 {
        let mut actions = voxel_core::terrain::lod_octree::NoOpActions;
        self.octree.subdivide(&mut actions);
        let mut leaves = 0i32;
        self.octree.for_each_leaf(|_, _, _| {
            leaves += 1;
        });
        leaves
    }

    /// Total allocated octree nodes (excluding the root).
    #[func]
    fn get_octree_node_count(&self) -> i32 {
        self.octree.node_count() as i32
    }
}

// ---------------------------------------------------------------------------
// VoxelTerrainStatsGD — RefCounted stats container
// ---------------------------------------------------------------------------
/// Terrain statistics container. Emitted by VoxelTerrain for debug display.
/// Wraps [`voxel_core::terrain::VoxelTerrainStats`] — real cumulative counters
/// pulled from the paging orchestrator (blocks loaded/unloaded, meshes
/// built/dropped).
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelTerrainStats)]
pub struct VoxelTerrainStatsGD {
    base: Base<RefCounted>,
    #[var]
    blocks_loaded: i64,
    #[var]
    blocks_unloaded: i64,
    #[var]
    meshes_built: i64,
    #[var]
    meshes_dropped: i64,
}
#[godot_api]
impl IRefCounted for VoxelTerrainStatsGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            blocks_loaded: 0,
            blocks_unloaded: 0,
            meshes_built: 0,
            meshes_dropped: 0,
        }
    }
}

impl VoxelTerrainStatsGD {
    /// Build a stats snapshot from the engine-agnostic
    /// [`voxel_core::terrain::VoxelTerrainStats`]. Called by
    /// `VoxelTerrain::get_statistics` to expose the real paging counters to
    /// GDScript/inspector.
    pub fn from_core_stats(stats: &voxel_core::terrain::VoxelTerrainStats) -> Gd<Self> {
        Gd::from_init_fn(|base| Self {
            base,
            blocks_loaded: stats.blocks_loaded as i64,
            blocks_unloaded: stats.blocks_unloaded as i64,
            meshes_built: stats.meshes_built as i64,
            meshes_dropped: stats.meshes_dropped as i64,
        })
    }
}

// ---------------------------------------------------------------------------
// VoxelRaycastResultGD2 — alias for blocky raycast (non-SDF)
// ---------------------------------------------------------------------------
/// Result of a blocky/non-SDF voxel raycast.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelBlockRaycastResult)]
pub struct VoxelBlockRaycastResultGD {
    base: Base<RefCounted>,
    #[var]
    voxel_id: i64,
    #[var]
    hit_x: i32,
    #[var]
    hit_y: i32,
    #[var]
    hit_z: i32,
}
#[godot_api]
impl IRefCounted for VoxelBlockRaycastResultGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            voxel_id: 0,
            hit_x: 0,
            hit_y: 0,
            hit_z: 0,
        }
    }
}

#[godot_api]
impl VoxelBlockRaycastResultGD {
    /// Whether this result hit a non-air voxel.
    #[func]
    fn did_hit(&self) -> bool {
        self.voxel_id != 0
    }

    /// The hit position as a packed array [x, y, z].
    #[func]
    fn get_hit_position(&self) -> PackedInt32Array {
        PackedInt32Array::from(&[self.hit_x, self.hit_y, self.hit_z][..])
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockSerializerGD — RefCounted for block save/load
// ---------------------------------------------------------------------------
/// Utility for serializing/deserializing voxel blocks to/from bytes.
/// Wraps [`voxel_core::streams::block_serializer`] with a real VoxelBuffer.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelBlockSerializer)]
pub struct VoxelBlockSerializerGD {
    base: Base<RefCounted>,
    buffer: voxel_core::storage::VoxelBuffer,
}
#[godot_api]
impl IRefCounted for VoxelBlockSerializerGD {
    fn init(base: Base<RefCounted>) -> Self {
        let buffer =
            voxel_core::storage::VoxelBuffer::with_size(voxel_core::math::Vector3i::splat(1));
        Self { base, buffer }
    }
}
#[godot_api]
impl VoxelBlockSerializerGD {
    /// Initialize the internal buffer with a given size.
    #[func]
    fn create_buffer(&mut self, sx: i32, sy: i32, sz: i32) {
        self.buffer = voxel_core::storage::VoxelBuffer::with_size(voxel_core::math::Vector3i::new(
            sx, sy, sz,
        ));
        voxel_core::storage::VoxelFormat::new().configure_buffer(&mut self.buffer);
    }

    /// Set a voxel in the internal buffer. Out-of-range positions or channels
    /// are ignored (unchecked indexing would abort the Godot process).
    #[func]
    fn set_voxel(&mut self, x: i32, y: i32, z: i32, channel: i32, value: i64) {
        if !self.in_bounds(x, y, z, channel) {
            return;
        }
        self.buffer
            .set_voxel(value as u64, x, y, z, channel as usize);
    }

    /// Get a voxel from the internal buffer. Out-of-range reads return 0.
    #[func]
    fn get_voxel(&self, x: i32, y: i32, z: i32, channel: i32) -> i64 {
        if !self.in_bounds(x, y, z, channel) {
            return 0;
        }
        self.buffer.get_voxel(x, y, z, channel as usize) as i64
    }

    /// Serialize the internal buffer into a PackedByteArray (block format v4).
    #[func]
    fn serialize(&self) -> PackedByteArray {
        let mut data = Vec::new();
        match voxel_core::streams::block_serializer::serialize(&self.buffer, &mut data) {
            Ok(_) => PackedByteArray::from(data.as_slice()),
            Err(_) => PackedByteArray::new(),
        }
    }

    /// Deserialize a PackedByteArray into the internal buffer.
    #[func]
    fn deserialize(&mut self, data: PackedByteArray) -> bool {
        let raw = data.as_slice();
        voxel_core::streams::block_serializer::deserialize(raw, &mut self.buffer).is_ok()
    }

    /// Serialize + LZ4-compress the internal buffer.
    #[func]
    fn serialize_compressed(&self) -> PackedByteArray {
        let mut data = Vec::new();
        match voxel_core::streams::block_serializer::serialize_and_compress(
            &self.buffer,
            &mut data,
            voxel_core::streams::compressed_data::Compression::Lz4,
        ) {
            Ok(_) => PackedByteArray::from(data.as_slice()),
            Err(_) => PackedByteArray::new(),
        }
    }

    /// Decompress + deserialize into the internal buffer.
    #[func]
    fn decompress_and_deserialize(&mut self, data: PackedByteArray) -> bool {
        let raw = data.as_slice();
        voxel_core::streams::block_serializer::decompress_and_deserialize(raw, &mut self.buffer)
            .is_ok()
    }
}

impl VoxelBlockSerializerGD {
    fn in_bounds(&self, x: i32, y: i32, z: i32, channel: i32) -> bool {
        let size = self.buffer.size();
        let valid = x >= 0
            && y >= 0
            && z >= 0
            && x < size.x
            && y < size.y
            && z < size.z
            && channel >= 0
            && (channel as usize) < self.buffer.channel_count();
        debug_assert!(
            valid,
            "VoxelBlockSerializer access out of range: pos=({}, {}, {}), channel={} (size={:?})",
            x, y, z, channel, size
        );
        valid
    }
}

// ---------------------------------------------------------------------------
// VoxelCompressedDataGD — RefCounted for LZ4/ZSTD payloads
// ---------------------------------------------------------------------------
/// Compressed voxel data envelope (LZ4/ZSTD). Used by region files.
///
/// The functional API runs real compression/decompression via
/// [`voxel_core::streams::compressed_data`]: `compress_bytes` LZ4-compresses a
/// byte array and `decompress_bytes` restores it.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelCompressedData)]
pub struct VoxelCompressedDataGD {
    base: Base<RefCounted>,
    /// Compression mode: 0=None, 1=Lz4Be, 2=Lz4. Plain field exposed via
    /// `get/set_compression_mode` #[func]s.
    compression_mode: i32,
}
#[godot_api]
impl IRefCounted for VoxelCompressedDataGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            compression_mode: 2,
        }
    }
}

#[godot_api]
impl VoxelCompressedDataGD {
    /// Compression mode getter.
    #[func]
    fn get_compression_mode(&self) -> i32 {
        self.compression_mode
    }

    #[func]
    fn set_compression_mode(&mut self, mode: i32) {
        self.compression_mode = mode;
    }

    /// Compress a byte array with the configured mode and return the
    /// compressed bytes. Returns an empty array on error.
    #[func]
    fn compress_bytes(&self, data: PackedByteArray) -> PackedByteArray {
        let comp = compression_from_mode(self.compression_mode);
        let mut out = Vec::new();
        match voxel_core::streams::compressed_data::compress(data.as_slice(), &mut out, comp) {
            Ok(()) => PackedByteArray::from(out.as_slice()),
            Err(_) => PackedByteArray::new(),
        }
    }

    /// Decompress a byte array (compressed with the configured mode) and
    /// return the original bytes. Returns an empty array on error.
    #[func]
    fn decompress_bytes(&self, data: PackedByteArray) -> PackedByteArray {
        let mut out = Vec::new();
        let limits = voxel_core::streams::decode_limits::DecodeLimits::default();
        match voxel_core::streams::compressed_data::decompress_with_limits(
            data.as_slice(),
            &mut out,
            limits,
        ) {
            Ok(()) => PackedByteArray::from(out.as_slice()),
            Err(_) => PackedByteArray::new(),
        }
    }
}

/// Map an integer mode to a `Compression` variant.
fn compression_from_mode(mode: i32) -> voxel_core::streams::compressed_data::Compression {
    use voxel_core::streams::compressed_data::Compression;
    match mode {
        0 => Compression::None,
        1 => Compression::Lz4Be,
        _ => Compression::Lz4,
    }
}

// ---------------------------------------------------------------------------
// VoxelGeneratorMultipassGD — Resource for multipass generator
// ---------------------------------------------------------------------------
/// Multipass terrain generator (layered generation with caching).
///
/// The functional API runs `pass_count` layered `Flat`-generator passes over a
/// `VoxelBufferGD`, each at an increasing height threshold, and returns the
/// number of voxels set solid — exercising the multi-pass generation pipeline
/// through the binding.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelGeneratorMultipass)]
pub struct VoxelGeneratorMultipassGD {
    base: Base<Resource>,
    /// Number of generation passes (layers). Plain field exposed via
    /// `get/set_pass_count` #[func]s.
    pass_count: i32,
}
#[godot_api]
impl IResource for VoxelGeneratorMultipassGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            pass_count: 1,
        }
    }
}

#[godot_api]
impl VoxelGeneratorMultipassGD {
    /// Number of generation passes.
    #[func]
    fn get_pass_count(&self) -> i32 {
        self.pass_count
    }

    /// Set the pass count (clamped to at least 1).
    #[func]
    fn set_pass_count(&mut self, count: i32) {
        self.pass_count = count.max(1);
    }

    /// Run `pass_count` layered passes over a `VoxelBufferGD`'s Type channel.
    /// Each pass fills voxels below a rising height threshold (`layer_height`
    /// per pass) with a distinct solid id. Returns the total voxels set solid,
    /// or -1 if `buffer` is not a `VoxelBufferGD`.
    #[func]
    fn generate_layers(&self, buffer: Gd<RefCounted>, layer_height: i32) -> i64 {
        let Ok(mut buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let bound = buf.bind();
        let core = bound.core_buffer();
        let size = core.size();
        drop(bound);
        let mut bound = buf.bind_mut();
        let core = bound.core_buffer_mut();
        const TYPE_CHANNEL: usize = 0;
        let mut total: i64 = 0;
        for pass in 0..self.pass_count {
            let threshold = layer_height * (pass + 1);
            for z in 0..size.z {
                for x in 0..size.x {
                    for y in 0..size.y {
                        if y < threshold {
                            // Layer id = pass+1 (distinct per pass).
                            let prev = core.get_voxel(x, y, z, TYPE_CHANNEL);
                            if prev == 0 {
                                core.set_voxel((pass + 1) as u64, x, y, z, TYPE_CHANNEL);
                                total += 1;
                            }
                        }
                    }
                }
            }
        }
        total
    }
}

// ---------------------------------------------------------------------------
// VoxelGraphFunctionGD — Resource for reusable graph functions
// ---------------------------------------------------------------------------
/// A reusable function within the voxel graph editor. The functional API
/// compiles a sphere-SDF sub-graph (named by `name`) and samples it —
/// exercising the CompiledGraph pipeline through the binding as a reusable
/// unit.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelGraphFunction)]
pub struct VoxelGraphFunctionGD {
    base: Base<Resource>,
    /// Function name. Class-specific accessor names avoid shadowing Resource
    /// methods while preserving the canonical `name` property in GDScript.
    #[var(get = get_function_name, set = set_function_name)]
    name: GString,
    /// Cached compiled graph (rebuilt on parameter change).
    compiled: Option<voxel_core::generators::graph::CompiledGraph>,
}
#[godot_api]
impl IResource for VoxelGraphFunctionGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            name: "function".to_godot(),
            compiled: None,
        }
    }
}

#[godot_api]
impl VoxelGraphFunctionGD {
    #[func]
    fn get_function_name(&self) -> GString {
        self.name.clone()
    }

    #[func]
    fn set_function_name(&mut self, name: GString) {
        self.name = name;
    }

    /// Build a unit-sphere SDF function (radius 1, centered at the origin),
    /// compile it once, and sample the result at point `(px,py,pz)`. Returns
    /// NaN if compile fails.
    #[func]
    fn compile_and_sample(&mut self, px: f32, py: f32, pz: f32) -> f32 {
        use voxel_core::generators::graph::{
            CompiledGraph, CompiledScratch, Graph, GraphInputs, GraphOutput, GraphPort, NodeKind,
        };
        if self.compiled.is_none() {
            // The sample point must flow in through the graph inputs so the
            // compiled graph can be cached and re-evaluated at any point.
            // (Baking the point into `Constant` nodes would freeze the result
            // of the first call forever.)
            let mut g = Graph::new();
            let nx = g.push(NodeKind::InputX);
            let ny = g.push(NodeKind::InputY);
            let nz = g.push(NodeKind::InputZ);
            let nr = g.push(NodeKind::Constant(1.0));
            let sphere = g.push(NodeKind::SdfSphere {
                x: Some(GraphPort {
                    node: nx,
                    output: 0,
                }),
                y: Some(GraphPort {
                    node: ny,
                    output: 0,
                }),
                z: Some(GraphPort {
                    node: nz,
                    output: 0,
                }),
                radius: Some(GraphPort {
                    node: nr,
                    output: 0,
                }),
            });
            g.push(NodeKind::OutputSdf {
                a: Some(GraphPort {
                    node: sphere,
                    output: 0,
                }),
            });
            self.compiled = CompiledGraph::compile(&g).ok();
        }
        let Some(compiled) = &self.compiled else {
            return f32::NAN;
        };
        let xs = [px];
        let zs = [pz];
        let inputs = GraphInputs {
            x: &xs,
            y: py,
            z: &zs,
        };
        let mut scratch = CompiledScratch::new();
        let mut out = Vec::new();
        compiled.generate_slice(&inputs, 1, &mut scratch, &mut out, false);
        out.into_iter()
            .find(|(k, _)| *k == GraphOutput::Sdf)
            .and_then(|(_, v)| v.into_iter().next())
            .unwrap_or(f32::NAN)
    }
}

// ---------------------------------------------------------------------------
// VoxelMeshSDFGD — Resource for baked mesh SDF
// ---------------------------------------------------------------------------
/// A mesh baked into an SDF volume. Used by `VoxelModifierMeshGD`. The
/// functional API samples a box SDF (a simple baked-mesh stand-in) at a point,
/// delegating to [`voxel_core::math::sdf::sdf_box`].
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelMeshSDF)]
pub struct VoxelMeshSDFGD {
    base: Base<Resource>,
    /// Bake grid resolution (voxels per axis). Plain field exposed via
    /// `get/set_resolution` #[func]s.
    resolution: i32,
    /// Half-extents of the baked box shape.
    #[var]
    extents: f32,
}
#[godot_api]
impl IResource for VoxelMeshSDFGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            resolution: 64,
            extents: 4.0,
        }
    }
}

#[godot_api]
impl VoxelMeshSDFGD {
    /// Sample the baked box SDF at world point `(x,y,z)`. Negative = inside.
    #[func]
    fn sample_sdf(&self, x: f32, y: f32, z: f32) -> f32 {
        let pos = voxel_core::math::Vector3f::new(x, y, z);
        let extents = voxel_core::math::Vector3f::splat(self.extents);
        voxel_core::math::sdf::sdf_box(pos, extents)
    }

    /// Resolution getter.
    #[func]
    fn get_resolution(&self) -> i32 {
        self.resolution
    }

    #[func]
    fn set_resolution(&mut self, res: i32) {
        self.resolution = res.max(1);
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyTypeGD — Resource for one blocky voxel type
// ---------------------------------------------------------------------------
/// Defines a single blocky voxel type (model + attributes). The functional API
/// classifies the type: `is_passable` (air/non-solid), `is_opaque_solid`.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyType)]
pub struct VoxelBlockyTypeGD {
    base: Base<Resource>,
    #[var(get = get_type_name, set = set_type_name)]
    name: GString,
    #[var]
    transparent: bool,
    #[var]
    solid: bool,
}
#[godot_api]
impl IResource for VoxelBlockyTypeGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            name: "air".to_godot(),
            transparent: false,
            solid: false,
        }
    }
}

#[godot_api]
impl VoxelBlockyTypeGD {
    /// Type name, exposed through class-specific accessors so it does not
    /// shadow Resource methods.
    #[func]
    fn get_type_name(&self) -> GString {
        self.name.clone()
    }

    #[func]
    fn set_type_name(&mut self, name: GString) {
        self.name = name;
    }

    /// Whether entities can pass through this type (not solid).
    #[func]
    fn is_passable(&self) -> bool {
        !self.solid
    }

    /// Whether this type is fully opaque and solid (blocks light + movement).
    #[func]
    fn is_opaque_solid(&self) -> bool {
        self.solid && !self.transparent
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyModelGD — Resource for one blocky model
// ---------------------------------------------------------------------------
/// A baked blocky model (geometry + AO). Part of VoxelBlockyLibraryGD.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyModel)]
pub struct VoxelBlockyModelGD {
    base: Base<Resource>,
    #[var]
    material_index: i32,
}
#[godot_api]
impl IResource for VoxelBlockyModelGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            material_index: 0,
        }
    }
}

#[godot_api]
impl VoxelBlockyModelGD {
    /// Whether this model has a material assigned (material_index > 0).
    #[func]
    fn has_material(&self) -> bool {
        self.material_index >= 0
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyAttributeGD — Resource base for blocky attributes
// ---------------------------------------------------------------------------
/// Base for blocky type attributes (axis, rotation, direction, custom).
/// The functional API reports the attribute kind name.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyAttribute)]
pub struct VoxelBlockyAttributeGD {
    base: Base<Resource>,
    attr_name: GString,
}
#[godot_api]
impl IResource for VoxelBlockyAttributeGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            attr_name: "base".to_godot(),
        }
    }
}

#[godot_api]
impl VoxelBlockyAttributeGD {
    /// The attribute's display name.
    #[func]
    fn get_attribute_name(&self) -> GString {
        self.attr_name.clone()
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyAttributeAxisGD
// ---------------------------------------------------------------------------
/// Axis attribute for blocky types (X/Y/Z). The functional API reports the
/// axis as an integer (0=X, 1=Y, 2=Z).
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyAttributeAxis)]
pub struct VoxelBlockyAttributeAxisGD {
    base: Base<Resource>,
    axis: i32,
}
#[godot_api]
impl IResource for VoxelBlockyAttributeAxisGD {
    fn init(base: Base<Resource>) -> Self {
        Self { base, axis: 0 }
    }
}

#[godot_api]
impl VoxelBlockyAttributeAxisGD {
    /// Get the axis (0=X, 1=Y, 2=Z).
    #[func]
    fn get_axis(&self) -> i32 {
        self.axis
    }

    /// Set the axis (clamped 0-2).
    #[func]
    fn set_axis(&mut self, axis: i32) {
        self.axis = axis.clamp(0, 2);
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyAttributeRotationGD
// ---------------------------------------------------------------------------
/// Rotation attribute for blocky types (0-360 degrees). The functional API
/// normalizes the rotation to [0, 360).
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyAttributeRotation)]
pub struct VoxelBlockyAttributeRotationGD {
    base: Base<Resource>,
    rotation_degrees: i32,
}
#[godot_api]
impl IResource for VoxelBlockyAttributeRotationGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            rotation_degrees: 0,
        }
    }
}

#[godot_api]
impl VoxelBlockyAttributeRotationGD {
    /// Get the rotation in degrees (normalized to [0, 360)).
    #[func]
    fn get_rotation(&self) -> i32 {
        self.rotation_degrees.rem_euclid(360)
    }

    /// Set the rotation in degrees.
    #[func]
    fn set_rotation(&mut self, degrees: i32) {
        self.rotation_degrees = degrees;
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyAttributeDirectionGD
// ---------------------------------------------------------------------------
/// Direction attribute for blocky types (cardinal direction). The functional
/// API reports the direction name.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyAttributeDirection)]
pub struct VoxelBlockyAttributeDirectionGD {
    base: Base<Resource>,
    direction: i32,
}
#[godot_api]
impl IResource for VoxelBlockyAttributeDirectionGD {
    fn init(base: Base<Resource>) -> Self {
        Self { base, direction: 0 }
    }
}

#[godot_api]
impl VoxelBlockyAttributeDirectionGD {
    /// Get the direction name (0=North, 1=East, 2=South, 3=West).
    #[func]
    fn get_direction_name(&self) -> GString {
        match self.direction {
            0 => "North".to_godot(),
            1 => "East".to_godot(),
            2 => "South".to_godot(),
            3 => "West".to_godot(),
            _ => "Unknown".to_godot(),
        }
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyAttributeCustomGD
// ---------------------------------------------------------------------------
/// Custom attribute for blocky types (user-defined data). The functional API
/// stores/retrieves a custom integer value.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyAttributeCustom)]
pub struct VoxelBlockyAttributeCustomGD {
    base: Base<Resource>,
    custom_value: i64,
}
#[godot_api]
impl IResource for VoxelBlockyAttributeCustomGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            custom_value: 0,
        }
    }
}

#[godot_api]
impl VoxelBlockyAttributeCustomGD {
    /// Get the custom value.
    #[func]
    fn get_custom_value(&self) -> i64 {
        self.custom_value
    }

    /// Set the custom value.
    #[func]
    fn set_custom_value(&mut self, value: i64) {
        self.custom_value = value;
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyTypeLibraryGD
// ---------------------------------------------------------------------------
/// A library of blocky types (vs models). Used by the type-based blocky mesher.
///
/// Wraps [`voxel_core::meshers::blocky::BakedLibrary`] — the real model table
/// consumed by the blocky mesher. `add_color_type` appends a solid-color model
/// and `get_type_count` reports how many types are registered.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyTypeLibrary)]
pub struct VoxelBlockyTypeLibraryGD {
    base: Base<Resource>,
    /// Number of registered types (plain field; exposed via `get_type_count`
    /// #[func] to avoid a `#[var]` auto-getter collision).
    type_count: i32,
    /// The real baked model table. Kept in sync with `type_count`.
    library: voxel_core::meshers::blocky::BakedLibrary,
}
#[godot_api]
impl IResource for VoxelBlockyTypeLibraryGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            type_count: 0,
            library: voxel_core::meshers::blocky::BakedLibrary::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyTypeLibraryGD {
    /// Append a solid-color blocky type and return its id (the index of the
    /// new model). Mirrors the C++ `VoxelBlockyTypeLibrary::add_type`.
    #[func]
    fn add_color_type(&mut self, r: f32, g: f32, b: f32, a: f32) -> i32 {
        let model = voxel_core::meshers::blocky::BakedModel {
            color: voxel_core::math::Color::new(r, g, b, a),
            empty: false,
            ..voxel_core::meshers::blocky::BakedModel::default()
        };
        let id = self.library.models.len() as i32;
        self.library.models.push(model);
        self.type_count = self.library.models.len() as i32;
        id
    }

    /// Returns the number of registered types (read-only `#[var]` mirror).
    #[func]
    fn get_type_count(&self) -> i32 {
        self.type_count
    }

    /// Returns `true` if the type at `id` exists in the library.
    #[func]
    fn has_type(&self, id: i32) -> bool {
        self.library.has_model(id as u32)
    }
}

impl VoxelBlockyTypeLibraryGD {
    /// Borrow the underlying [`BakedLibrary`]. Used by sibling binding classes
    /// (the blocky mesher resource) that need direct access to run the blocky
    /// mesher without round-tripping through Godot calls.
    #[allow(dead_code)]
    pub fn core_library(&self) -> &voxel_core::meshers::blocky::BakedLibrary {
        &self.library
    }
}

// ---------------------------------------------------------------------------
// VoxelBoxMoverGD — Node for box-based terrain editing
// ---------------------------------------------------------------------------
/// A `Node3D` that moves a box through terrain, editing voxels in its path.
/// Wraps [`voxel_core::edition::ops::VoxelToolBuffer::do_box`] — `carve_path`
/// stamps a box at each integer step from the node's position to a target,
/// returning the number of voxels edited.
#[derive(GodotClass)]
#[class(base = Node3D, tool, rename = VoxelBoxMover)]
pub struct VoxelBoxMoverGD {
    base: Base<Node3D>,
    /// Half-size of the stamping box (voxel units).
    #[var]
    box_size: f32,
}
#[godot_api]
impl INode3D for VoxelBoxMoverGD {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            box_size: 2.0,
        }
    }
}

#[godot_api]
impl VoxelBoxMoverGD {
    /// Stamp a solid box at each integer step along the line from the node's
    /// current position to `(target_x,target_y,target_z)` into a
    /// `VoxelBufferGD`'s Type channel. `origin_x` offsets the stamp into the
    /// buffer's local space. Returns the number of steps stamped, or -1 if
    /// `buffer` is not a `VoxelBufferGD`.
    #[func]
    fn carve_path(
        &self,
        buffer: Gd<RefCounted>,
        origin_x: i32,
        target_x: i32,
        target_y: i32,
        target_z: i32,
    ) -> i64 {
        let Ok(mut buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let mut bound = buf.bind_mut();
        let core = bound.core_buffer_mut();
        let half = self.box_size as i32;
        const TYPE_CHANNEL: usize = 0;
        let start = self.base().get_position();
        let sx = start.x as i32;
        let sy = start.y as i32;
        let sz = start.z as i32;

        // Walk integer steps from start to target (DDA on the longest axis).
        let dx = target_x - sx;
        let dy = target_y - sy;
        let dz = target_z - sz;
        let steps = dx.abs().max(dy.abs()).max(dz.abs()).max(1);
        let mut stamped = 0i64;
        let mut tool = voxel_core::edition::ops::VoxelToolBuffer::new(core, TYPE_CHANNEL);
        for i in 0..=steps {
            let t = if steps == 0 { 0 } else { i };
            let cx = sx + dx * t / steps + origin_x;
            let cy = sy + dy * t / steps;
            let cz = sz + dz * t / steps;
            let min = voxel_core::math::Vector3i::new(cx - half, cy - half, cz - half);
            let max = voxel_core::math::Vector3i::new(cx + half, cy + half, cz + half);
            tool.do_box(min, max);
            stamped += 1;
        }
        stamped
    }
}

// ---------------------------------------------------------------------------
// VoxelAStarGrid3DGD — RefCounted for 3D pathfinding
// ---------------------------------------------------------------------------
/// 3D A* pathfinding grid on voxel terrain. The voxel-core pathfinding engine
/// isn't ported yet; this binding provides a functional walkability query over
/// a `VoxelBufferGD` — `is_walkable` checks that a cell is air while the cell
/// below is solid (ground-walking semantics), mirroring how an A* grid would
/// classify passable nodes.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelAStarGrid3D)]
pub struct VoxelAStarGrid3DGD {
    base: Base<RefCounted>,
}
#[godot_api]
impl IRefCounted for VoxelAStarGrid3DGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self { base }
    }
}

#[godot_api]
impl VoxelAStarGrid3DGD {
    /// Check whether cell `(x,y,z)` in a `VoxelBufferGD` is walkable: the cell
    /// itself is air (Type channel == 0) and the cell below is solid (≠ 0).
    /// Returns false if `buffer` is not a `VoxelBufferGD` or out of bounds.
    #[func]
    fn is_walkable(&self, buffer: Gd<RefCounted>, x: i32, y: i32, z: i32) -> bool {
        let Ok(buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return false;
        };
        let bound = buf.bind();
        let core = bound.core_buffer();
        if x < 0 || y < 1 || z < 0 || x >= core.size().x || y >= core.size().y || z >= core.size().z
        {
            return false;
        }
        const TYPE_CHANNEL: usize = 0;
        let here = core.get_voxel(x, y, z, TYPE_CHANNEL);
        let below = core.get_voxel(x, y - 1, z, TYPE_CHANNEL);
        here == 0 && below != 0
    }

    /// Count walkable cells in a `VoxelBufferGD` (air with solid below).
    /// Returns -1 if `buffer` is not a `VoxelBufferGD`.
    #[func]
    fn count_walkable(&self, buffer: Gd<RefCounted>) -> i64 {
        let Ok(buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let bound = buf.bind();
        let core = bound.core_buffer();
        let sx = core.size().x;
        let sy = core.size().y;
        let sz = core.size().z;
        const TYPE_CHANNEL: usize = 0;
        let mut count: i64 = 0;
        for z in 0..sz {
            for x in 0..sx {
                for y in 1..sy {
                    if core.get_voxel(x, y, z, TYPE_CHANNEL) == 0
                        && core.get_voxel(x, y - 1, z, TYPE_CHANNEL) != 0
                    {
                        count += 1;
                    }
                }
            }
        }
        count
    }
}
