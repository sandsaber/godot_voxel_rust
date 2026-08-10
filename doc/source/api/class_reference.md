# Class reference

This page lists every Godot class exposed by the Rust port's GDExtension (`voxel-gdext`, 79 classes). Class names are the canonical upstream names: where a Rust binding struct carries `rename = X` in its `#[class(...)]` attribute, the Godot-visible name is `X`. The `GD` suffix seen in the Rust source (e.g. `VoxelBufferGD`) is internal to the binding crate only and never appears in GDScript or the editor. Structs without a `rename` (e.g. `VoxelTerrain`) keep their struct name as-is. Properties are `#[var]` fields plus `get_*`/`set_*` `#[func]` pairs; methods are the remaining `#[func]`s. Types use the binding's signatures (`i32`, `i64`, `f32`, `f64`, `bool`, `GString`, `PackedByteArray`, `Gd<...>`, etc.). Classes marked "*Stub: not equivalent to the upstream C++ class.*" are minimal facades with an invented functional API, not full ports.

## Core storage & tools

### VoxelBuffer

Inherits: `RefCounted`

A buffer of voxel data exposing basic per-channel voxel read/write, fill and clear to GDScript.

**Methods**

- `create(size_x: i32, size_y: i32, size_z: i32)`
- `set_voxel(x: i32, y: i32, z: i32, channel: i32, value: i64)`
- `get_voxel(x: i32, y: i32, z: i32, channel: i32) -> i64`
- `get_size_x() -> i32`
- `get_size_y() -> i32`
- `get_size_z() -> i32`
- `fill_channel(channel: i32, value: i64)`
- `clear_channel(channel: i32, value: i64)`

### VoxelFormat

Inherits: `Resource`

Channel depth configuration for a `VoxelBuffer`; maps each of the 8 channels to a bit depth (8/16/32/64).

**Methods**

- `set_channel_depth(index: i32, depth: i32)` — `depth`: 0 = Bit8, 1 = Bit16, 2 = Bit32, 3 = Bit64
- `get_channel_depth(index: i32) -> i32` — returns 0–3, or -1 for an invalid index

### VoxelToolBuffer

Inherits: `RefCounted`

A voxel editing tool over an owned `VoxelBuffer`, providing sphere, box and single-voxel edits on a selected channel.

**Methods**

- `create_buffer(size_x: i32, size_y: i32, size_z: i32)`
- `set_channel(channel: i32)`
- `do_sphere(cx: f64, cy: f64, cz: f64, radius: f64, mode: i32, value: i64)` — `mode`: 0 = Add, 1 = Remove, 2 = Set
- `do_box(min_x: i32, min_y: i32, min_z: i32, max_x: i32, max_y: i32, max_z: i32, mode: i32, value: i64)`
- `set_voxel(x: i32, y: i32, z: i32, value: i64)`
- `get_voxel(x: i32, y: i32, z: i32) -> i64`

### VoxelToolTerrain

Inherits: `RefCounted`

Terrain editing tool that holds a node-path reference to a `VoxelTerrain` for GDScript-callable editing.

*Stub: not equivalent to the upstream C++ class.* Only stores a terrain path; no voxel editing is implemented.

**Methods**

- `set_terrain_path(path: GString)`
- `get_terrain_path() -> GString`

### VoxelBlockSerializer

Inherits: `RefCounted`

Utility for serializing/deserializing voxel blocks to/from bytes, plain (block format v4) and LZ4-compressed.

**Methods**

- `create_buffer(sx: i32, sy: i32, sz: i32)`
- `set_voxel(x: i32, y: i32, z: i32, channel: i32, value: i64)`
- `get_voxel(x: i32, y: i32, z: i32, channel: i32) -> i64`
- `serialize() -> PackedByteArray`
- `deserialize(data: PackedByteArray) -> bool`
- `serialize_compressed() -> PackedByteArray`
- `decompress_and_deserialize(data: PackedByteArray) -> bool`

### VoxelCompressedData

Inherits: `RefCounted`

Compressed voxel data envelope (None / LZ4 big-endian / LZ4), as used by region files.

**Properties**

- `compression_mode: i32` — 0 = None, 1 = Lz4Be, 2 = Lz4

**Methods**

- `compress_bytes(data: PackedByteArray) -> PackedByteArray`
- `decompress_bytes(data: PackedByteArray) -> PackedByteArray`

## Terrain & viewers

### VoxelNode

Inherits: `Node3D`

Base Node3D for voxel volume nodes, holding shared streaming/view-distance properties.

*Stub: not equivalent to the upstream C++ class.* API-parity placeholder; the real terrain node is `VoxelTerrain`.

**Properties**

- `auto_load: bool`
- `max_view_distance: i64`

**Methods**

- `is_streaming() -> bool`
- `get_view_distance_blocks() -> i64`

### VoxelTerrain

Inherits: `Node3D`

The voxel terrain node: wraps the engine-agnostic paging orchestrator that loads data blocks, meshes them with the configured mesher (transvoxel by default), and manages LOD and view/unview based on paired `VoxelViewer` positions.

**Properties**

- `stream: Resource` — must be `VoxelStreamMemory` or `VoxelStreamRegionFiles`
- `generator: Resource` — `VoxelGeneratorWaves`, `VoxelGeneratorFlat`, `VoxelGeneratorNoise`, `VoxelGeneratorHeightmap`, `VoxelGeneratorImage` or `VoxelGeneratorGraph` (getter returns `Variant`)
- `mesher: Resource` — `VoxelMesherTransvoxel` (default), `VoxelMesherCubes` or `VoxelMesherBlocky`; set before `_ready` (getter returns `Variant`)
- `lod_count: i32` — 1 = single-LOD, 2+ = multi-LOD; set before `_ready`
- `material_override: Material`
- `generate_collision: bool`

**Methods**

- `get_mesh_block_count() -> i32`
- `get_version() -> GString`
- `get_statistics() -> Variant` — returns a `VoxelTerrainStats` (or `null` before `_ready`)
- `set_voxel_sdf(world_x: i32, world_y: i32, world_z: i32, value: f32) -> bool`
- `get_voxel_sdf(world_x: i32, world_y: i32, world_z: i32) -> f32`
- `get_bounds() -> PackedInt32Array` — `[min_x, min_y, min_z, size_x, size_y, size_z]`
- `raycast(origin_x: f64, origin_y: f64, origin_z: f64, dir_x: f64, dir_y: f64, dir_z: f64, max_distance: f64) -> PackedFloat32Array` — returns `[x, y, z, hit]` with `hit` = 1.0 on hit

### VoxelLodTerrain

Inherits: `Node3D`

Multi-LOD terrain node; exposes LOD level configuration and the LOD octree's subdivision behavior.

*Stub: not equivalent to the upstream C++ class.* Despite the doc-level claims, the binding holds no terrain core (no stream, generator, meshing or paging); only the LOD octree is functional.

**Properties**

- `lod_count: i32`
- `lod_distance: f32`

**Methods**

- `subdivide_and_count_leaves() -> i32`
- `get_octree_node_count() -> i32`

### VoxelViewer

Inherits: `Node3D`

Marks a viewer position for the terrain paging system; add as a child of (or sibling to) a `VoxelTerrain`.

**Properties**

- `view_distance: i64` — view distance in voxels (horizontal and vertical)

### VoxelTerrainStats

Inherits: `RefCounted`

Cumulative terrain statistics snapshot emitted by `VoxelTerrain` for debug display.

**Properties**

- `blocks_loaded: i64`
- `blocks_unloaded: i64`
- `meshes_built: i64`
- `meshes_dropped: i64`

### VoxelDataBlockEnterInfo

Inherits: `RefCounted`

Information about a data block entering the resident set, as part of terrain lifecycle events.

**Properties**

- `block_x: i32`
- `block_y: i32`
- `block_z: i32`
- `lod: i32`
- `original_position: bool`

**Methods**

- `is_at_origin() -> bool`
- `get_lod_level() -> i32`

### VoxelSaveCompletionTracker

Inherits: `RefCounted`

Tracks completion of save operations so scripts can await terrain persistence.

**Methods**

- `mark_pending()`
- `mark_done()`
- `get_pending_count() -> i32`
- `get_is_done() -> bool`

### VoxelRaycastResult

Inherits: `RefCounted`

Result of an SDF voxel raycast: hit position, previous position, distance along the ray and hit normal.

**Properties**

- `hit_x: i32`
- `hit_y: i32`
- `hit_z: i32`
- `prev_x: i32`
- `prev_y: i32`
- `prev_z: i32`
- `distance: f32`
- `normal_x: i32`
- `normal_y: i32`
- `normal_z: i32`

**Methods**

- `did_hit() -> bool`
- `get_hit_position() -> PackedInt32Array`

### VoxelBlockRaycastResult

Inherits: `RefCounted`

Result of a blocky (non-SDF) voxel raycast.

**Properties**

- `voxel_id: i64`
- `hit_x: i32`
- `hit_y: i32`
- `hit_z: i32`

**Methods**

- `did_hit() -> bool`
- `get_hit_position() -> PackedInt32Array`

### VoxelBoxMover

Inherits: `Node3D`

Moves a box through terrain, stamping box edits into a `VoxelBuffer`'s Type channel along the path.

*Stub: not equivalent to the upstream C++ class.* Invented carving-path API; the upstream class is a box-vs-terrain movement utility.

**Properties**

- `box_size: f32` — half-size of the stamping box in voxel units

**Methods**

- `carve_path(buffer: Gd<RefCounted>, origin_x: i32, target_x: i32, target_y: i32, target_z: i32) -> i64` — returns the number of steps stamped, or -1 if `buffer` is not a `VoxelBuffer`

### VoxelAStarGrid3D

Inherits: `RefCounted`

Walkability queries over voxel terrain data.

*Stub: not equivalent to the upstream C++ class.* The pathfinding engine is not ported; only ground-walking walkability checks over a `VoxelBuffer` are provided.

**Methods**

- `is_walkable(buffer: Gd<RefCounted>, x: i32, y: i32, z: i32) -> bool`
- `count_walkable(buffer: Gd<RefCounted>) -> i64`

## Generators

### VoxelGenerator

Inherits: `Resource`

Abstract base resource for voxel generators (subclasses: Waves, Flat, Noise, Heightmap, Graph).

*Stub: not equivalent to the upstream C++ class.* Marker base with a single category getter; no generation API.

**Methods**

- `get_category() -> GString`

### VoxelGeneratorWaves

Inherits: `Resource`

A simple SDF terrain generator producing rolling waves along the X axis.

**Properties**

- `amplitude: f32`
- `frequency: f32`
- `period: f32`

### VoxelGeneratorFlat

Inherits: `Resource`

A flat terrain generator filling the SDF as a horizontal plane at a given height.

**Properties**

- `height: i64`

### VoxelGeneratorNoise

Inherits: `Resource`

A 3D noise terrain generator producing caves and overhangs via 3D FastNoiseLite.

**Properties**

- `seed: i64`
- `frequency: f32`
- `height_start: f32`
- `height_range: f32`

### VoxelGeneratorHeightmap

Inherits: `Resource`

A heightmap terrain generator driven by 2D noise, producing rolling hills with controllable seed, frequency and height range.

**Properties**

- `seed: i64`
- `frequency: f32`
- `height_range: f32`

**Methods**

- `sample_height(x: f32, z: f32) -> f32`

### VoxelGeneratorImage

Inherits: `Resource`

A heightmap terrain generator driven by an image: pixel luminance becomes terrain height (`height_start + luminance * height_range`). Writes SDF (smooth) or Type (blocky) data depending on `channel`, so it can drive both transvoxel and cubes terrain.

**Properties**

- `height_range: f32`
- `height_start: f32`
- `channel: i32` — `0` = Type (blocky), `1` = Sdf (smooth)
- `repeat: bool` — tile the image instead of clamping at its edges

**Methods**

- `set_image(image: Gd<Image>) -> bool` — loads heights from a Godot `Image` (Rec. 709 luminance)
- `set_heights(data: PackedByteArray, width: i32, height: i32) -> bool` — loads heights from raw bytes (`0..255`), row-major
- `has_image() -> bool`

### VoxelGeneratorGraph

Inherits: `Resource`

A graph-based terrain generator: build a node graph programmatically, compile it, and let it drive terrain (or sample it standalone).

**Methods**

- `clear_graph()` — remove all nodes from the graph under construction
- `add_node(kind: GString, a: i64, b: i64, c: i64, d: i64, value: f32) -> i64` — append a node (`InputX/Y/Z`, `Constant`, arithmetic, SDF ops, `Noise2D/3D`, `OutputSdf`, …); port arguments are node ids, `-1` = unconnected; returns the new node id or `-1` for an unknown kind
- `get_graph_node_count() -> i32`
- `compile_graph() -> bool` — whether the graph compiles (no cycles / dangling ports)
- `get_graph_json() -> GString`
- `set_graph_json(json: GString)` — stored as an interchange string (not parsed back)
- `sample_sphere_sdf(cx: f32, cy: f32, cz: f32, r: f32, px: f32, py: f32, pz: f32) -> f32` — standalone helper; returns the signed distance (negative = inside), or `NaN` if the graph fails to compile
- `get_node_count() -> i32`

### VoxelGeneratorMultipass

Inherits: `Resource`

Multipass terrain generator (layered generation with caching).

**Properties**

- `pass_count: i32`

**Methods**

- `generate_layers(buffer: Gd<RefCounted>, layer_height: i32) -> i64` — returns the total voxels set solid, or -1 if `buffer` is not a `VoxelBuffer`

### VoxelGraphFunction

Inherits: `Resource`

A reusable function within the voxel graph editor; compiles a sub-graph and samples it.

**Properties**

- `name: GString` — accessed explicitly through `get_function_name` / `set_function_name`

**Methods**

- `compile_and_sample(px: f32, py: f32, pz: f32) -> f32` — samples a cached unit-sphere SDF function; `NaN` if compile fails

## Streams

### VoxelStream

Inherits: `Resource`

Abstract base resource for voxel streams (subclasses: Memory, RegionFiles).

*Stub: not equivalent to the upstream C++ class.* Marker base with a single category getter; no load/save API.

**Methods**

- `get_category() -> GString`

### VoxelStreamMemory

Inherits: `Resource`

A stream that keeps voxel blocks in memory (no disk persistence).

**Methods**

- `get_block_count() -> i32`
- `clear()`

### VoxelStreamRegionFiles

Inherits: `Resource`

Saves/loads voxel data to `.vxr` region files on disk; assign to `VoxelTerrain.stream` to enable persistence.

**Properties**

- `directory: GString`

### VoxelStreamSQLite

Inherits: `Resource`

SQLite stream configuration.

*Stub: not equivalent to the upstream C++ class.* Only validates the database path extension; no SQLite I/O is implemented.

**Properties**

- `database_path: GString`

**Methods**

- `has_valid_extension() -> bool`

### VoxelVoxLoader

Inherits: `Resource`

MagicaVoxel `.vox` loader support class.

*Stub: not equivalent to the upstream C++ class.* Only reports extension support; actual `.vox` parsing is exposed by `VoxImporterPlugin` instead.

**Methods**

- `supports_extension(ext: GString) -> bool`

## Meshers & models

### VoxelMesher

Inherits: `Resource`

Abstract base resource for voxel meshers (subclasses: Transvoxel, Blocky, Cubes).

*Stub: not equivalent to the upstream C++ class.* Marker base with a padding property and a category getter; no meshing API.

**Properties**

- `padding: i32`

**Methods**

- `get_category() -> GString`

### VoxelMesherTransvoxel

Inherits: `Resource`

Configuration for the transvoxel smooth terrain mesher.

**Properties**

- `sdf_channel: i32`

**Methods**

- `build_vertex_count(buffer: Gd<RefCounted>, lod_hint: bool) -> i64` — returns -1 if `buffer` is not a `VoxelBuffer`
- `build_triangle_count(buffer: Gd<RefCounted>, lod_hint: bool) -> i64`

### VoxelMesherBlocky

Inherits: `Resource`

Configuration for the blocky (Minecraft-style) terrain mesher.

**Properties**

- `bake_occlusion: bool`
- `occlusion_darkness: f32`
- `type_channel: i32`

**Methods**

- `is_baking_occlusion() -> bool`
- `type_channel_index() -> i32`
- `build_vertex_count(buffer: Gd<RefCounted>) -> i64`

### VoxelMesherCubes

Inherits: `Resource`

Configuration for the cubes (greedy mesh) terrain mesher.

**Properties**

- `greedy: bool`
- `color_channel: i32`

**Methods**

- `is_greedy() -> bool`
- `color_channel_index() -> i32`
- `build_vertex_count(buffer: Gd<RefCounted>) -> i64`

### VoxelColorPalette

Inherits: `Resource`

A 256-entry RGBA color palette used by the cubes mesher (8 bits per channel).

**Methods**

- `set_color(index: i32, r: i32, g: i32, b: i32, a: i32)`
- `get_color(index: i32) -> PackedInt32Array` — `[r, g, b, a]`
- `clear()`

### VoxelBlockyLibrary

Inherits: `Resource`

A library of baked blocky models, maintaining the real model table consumed by the blocky mesher.

**Methods**

- `add_solid_model(r: f32, g: f32, b: f32) -> i32`
- `get_model_count() -> i32`
- `is_empty() -> bool`

### VoxelBlockyTypeLibrary

Inherits: `Resource`

A library of blocky types (vs models), used by the type-based blocky mesher.

**Methods**

- `add_color_type(r: f32, g: f32, b: f32, a: f32) -> i32`
- `get_type_count() -> i32`
- `has_type(id: i32) -> bool`

### VoxelBlockyType

Inherits: `Resource`

Defines a single blocky voxel type (model + attributes).

**Properties**

- `name: GString` — accessed explicitly through `get_type_name` / `set_type_name`
- `transparent: bool`
- `solid: bool`

**Methods**

- `is_passable() -> bool`
- `is_opaque_solid() -> bool`

### VoxelBlockyModel

Inherits: `Resource`

A baked blocky model (geometry + AO), part of a blocky library.

**Properties**

- `material_index: i32`

**Methods**

- `has_material() -> bool`

### VoxelBlockyModelCube

Inherits: `Resource`

A cube-shaped blocky model with a solid color.

**Properties**

- `r: f32`
- `g: f32`
- `b: f32`
- `a: f32`

**Methods**

- `is_solid() -> bool`
- `set_color(r: f32, g: f32, b: f32, a: f32)`

### VoxelBlockyModelEmpty

Inherits: `Resource`

An empty (air) blocky model, the sentinel for passable cells.

**Methods**

- `is_air() -> bool`

### VoxelBlockyModelMesh

Inherits: `Resource`

A mesh-based blocky model with optional transparency.

**Properties**

- `r: f32`
- `g: f32`
- `b: f32`
- `transparent: bool`

**Methods**

- `is_transparent() -> bool`
- `set_color(r: f32, g: f32, b: f32)`

### VoxelBlockyModelFluid

Inherits: `Resource`

A fluid blocky model (water/lava) with a fluid level.

**Properties**

- `fluid_level: i32` — 0–8

**Methods**

- `is_fluid() -> bool`

### VoxelBlockyFluid

Inherits: `Resource`

A fluid type for blocky terrain, reporting flow state.

**Properties**

- `flowing: bool`
- `flow_level: i32` — 0–8, 8 = full block

### VoxelBlockyAttribute

Inherits: `Resource`

Base for blocky type attributes (axis, rotation, direction, custom).

**Methods**

- `get_attribute_name() -> GString`

### VoxelBlockyAttributeAxis

Inherits: `Resource`

Axis attribute for blocky types (X/Y/Z).

**Properties**

- `axis: i32` — 0 = X, 1 = Y, 2 = Z

### VoxelBlockyAttributeRotation

Inherits: `Resource`

Rotation attribute for blocky types (degrees, normalized to [0, 360) on read).

**Properties**

- `rotation: i32`

### VoxelBlockyAttributeDirection

Inherits: `Resource`

Direction attribute for blocky types (cardinal direction).

**Methods**

- `get_direction_name() -> GString` — 0 = North, 1 = East, 2 = South, 3 = West

### VoxelBlockyAttributeCustom

Inherits: `Resource`

Custom attribute for blocky types (user-defined data).

**Properties**

- `custom_value: i64`

### VoxelMeshSDF

Inherits: `Resource`

A mesh baked into an SDF volume, for use by mesh-based modifiers.

*Stub: not equivalent to the upstream C++ class.* No mesh baking exists; `sample_sdf` evaluates an axis-aligned box SDF stand-in derived from `extents`.

**Properties**

- `extents: f32`
- `resolution: i32`

**Methods**

- `sample_sdf(x: f32, y: f32, z: f32) -> f32`

## Instancing

### VoxelInstancer

Inherits: `Node3D`

Scatters instances (trees, rocks, grass) on a parent `VoxelTerrain` using an instance library and a random scatter generator.

**Properties**

- `density_multiplier: f32`

**Methods**

- `add_item(name: GString, density: f64, min_scale: f64, max_scale: f64) -> i32`
- `get_item_count() -> i32`
- `set_seed(seed: i64)`
- `scatter_from_buffer(buffer: Gd<RefCounted>) -> i32` — extracts surface points from a `VoxelBuffer` and returns the total instance count
- `scatter_test(count: i32) -> i32`

### VoxelInstanceLibrary

Inherits: `Resource`

A library of scatter items for instancing.

**Methods**

- `add_item(name: GString, density: f32, min_scale: f32, max_scale: f32, snap_to_normal: bool) -> i32`
- `get_item_count() -> i32`
- `is_empty() -> bool`

### VoxelInstanceLibraryItem

Inherits: `Resource`

One entry in a `VoxelInstanceLibrary`, defining what to scatter and how.

**Properties**

- `name: GString` — accessed explicitly through `get_item_name` / `set_item_name`
- `density: f32`
- `min_scale: f32`
- `max_scale: f32`
- `snap_to_normal: bool`

**Methods**

- `get_average_scale() -> f32`
- `get_scale_range() -> f32`
- `is_disabled() -> bool`

### VoxelInstanceLibraryMultiMeshItem

Inherits: `Resource`

A MultiMesh-based instance library item.

*Stub: not equivalent to the upstream C++ class.* Only tracks an instance count; no multimesh setup.

**Properties**

- `mesh_instance_count: i32`

**Methods**

- `has_instances() -> bool`

### VoxelInstanceLibrarySceneItem

Inherits: `Resource`

A scene-based instance library item (places PackedScenes, not multimesh).

*Stub: not equivalent to the upstream C++ class.* Only records a scene path; no scene instantiation.

**Methods**

- `has_scene() -> bool`

### VoxelInstanceComponent

Inherits: `Resource`

An instance component attached to a node for scatter rendering.

*Stub: not equivalent to the upstream C++ class.* Only a visibility flag; no rendering integration.

**Properties**

- `visible: bool`

## Modifiers

### VoxelModifier

Inherits: `Node3D`

Base Node3D for SDF modifiers; children modify terrain SDF data.

*Stub: not equivalent to the upstream C++ class.* Base marker with shared properties only; no modifier application API.

**Properties**

- `operation: i32`
- `smoothness: f32`

**Methods**

- `get_category() -> GString`

### VoxelModifierSphere

Inherits: `Node3D`

A sphere-shaped SDF modifier node; carve (subtract) or merge (union) a sphere into generated terrain.

**Properties**

- `radius: f32`
- `operation: i32` — 0 = add (union), 1 = subtract
- `smoothness: f32` — 0 = hard blend, larger = smoother

**Methods**

- `apply_to_buffer(buffer: Gd<RefCounted>, origin_x: f32, origin_y: f32, origin_z: f32) -> i64` — returns the number of voxels whose SDF changed, or -1 if `buffer` is not a `VoxelBuffer`

### VoxelModifierMesh

Inherits: `Node3D`

A mesh-based SDF modifier node whose shape is an oriented box centered on the node's position.

**Properties**

- `operation: i32` — 0 = add (union), 1 = subtract
- `extents: f32` — half-extents of the box shape in voxel units

**Methods**

- `apply_to_buffer(buffer: Gd<RefCounted>, origin_x: f32, origin_y: f32, origin_z: f32) -> i64` — returns the number of voxels whose SDF changed, or -1 if `buffer` is not a `VoxelBuffer`

## Noise & curves

### ZN_FastNoiseLite

Inherits: `Resource`

FastNoiseLite noise resource; samples raw 3D noise configured from the resource's seed/frequency/noise type.

**Properties**

- `seed: i32`
- `frequency: f32`
- `noise_type: i32` — 0 = OpenSimplex2, 1 = OpenSimplex2S, 2 = Cellular, 3 = Perlin, 4 = ValueCubic, 5 = Value

**Methods**

- `sample_3d(x: f32, y: f32, z: f32) -> f32`

### FastNoise2

Inherits: `Resource`

FastNoise2-style noise resource; the upstream FastNoise2 C++ library is not ported, so sampling delegates to the fastnoise-lite sampler.

**Properties**

- `seed: i32`
- `frequency: f32`

**Methods**

- `sample_3d(x: f32, y: f32, z: f32) -> f32`
- `sample_2d(x: f32, z: f32) -> f32`

### ZN_SpotNoise

Inherits: `Resource`

Spot noise resource generating discrete spot points via a deterministic noise-threshold acceptance test.

**Properties**

- `density: f32`
- `radius: f32`
- `seed: i32`

**Methods**

- `count_spots(grid_size: i32) -> i32`

### NoisePattern2D

Inherits: `Resource`

A 2D noise pattern resource scaled by a configurable factor.

**Properties**

- `scale: f32`
- `seed: i32`

**Methods**

- `sample_2d(x: f32, z: f32) -> f32`

### ZN_Curve

Inherits: `Resource`

A baked curve resource with linearly-interpolated sampling.

**Methods**

- `sample(t: f32) -> f32`
- `set_identity(count: i32)`
- `get_point_count() -> i32`
- `set_points(values: PackedFloat32Array)`

## Editor plugins

### VoxImporterPlugin

Inherits: `EditorPlugin`

Editor plugin for importing `.vox` (MagicaVoxel) files; parses the binary format and converts the first model into voxel data or an `ArrayMesh`.

**Methods**

- `parse_vox_bytes(bytes: PackedByteArray) -> PackedFloat32Array` — flat `[x, y, z, r, g, b]` per voxel (6 floats each)
- `parse_vox_to_mesh(bytes: PackedByteArray) -> Gd<ArrayMesh>`

### VoxelGraphEditorPlugin

Inherits: `EditorPlugin`

Editor plugin hosting the procedural voxel graph editor as a bottom panel.

**Methods**

- `compile_sample_sphere(radius: f32, px: f32, py: f32, pz: f32) -> f32` — builds and compiles a sphere-SDF graph and samples it; `NaN` if compile fails
- `get_node_count() -> i32` — node count of the default demo graph (constant 6)

### VoxelInstancerEditorPlugin

Inherits: `EditorPlugin`

Editor plugin for the voxel instancer (scatter placement), adding a bottom-panel view.

**Methods**

- `count_surface_points(buffer: Gd<RefCounted>) -> i64` — returns -1 if `buffer` is not a `VoxelBuffer`

### VoxelTerrainEditorPlugin

Inherits: `EditorPlugin`

Editor plugin for editing voxel terrain, adding a bottom-panel view.

**Methods**

- `carve_sphere_into(buffer: Gd<RefCounted>, cx: f32, cy: f32, cz: f32, radius: f32) -> i64` — returns the number of voxels set solid, or -1 if `buffer` is not a `VoxelBuffer`

### VoxelLodTerrainEditorPlugin

Inherits: `EditorPlugin`

Editor plugin for multi-LOD terrain editing, adding a bottom-panel view.

**Methods**

- `carve_box_into(buffer: Gd<RefCounted>, min_x: i32, min_y: i32, min_z: i32, max_x: i32, max_y: i32, max_z: i32) -> i64` — returns the number of voxels set solid, or -1 if `buffer` is not a `VoxelBuffer`

### VoxelGraphNode

Inherits: `Resource`

A graph node descriptor for the graph editor.

*Stub: not equivalent to the upstream C++ class.* Only validates the node type name prefix; no ports or parameters.

**Properties**

- `node_type: GString`

**Methods**

- `is_valid_category() -> bool`

### VoxelGraphConnection

Inherits: `Resource`

A connection between two graph nodes, storing source/target node ids and ports.

*Stub: not equivalent to the upstream C++ class.* Plain endpoint storage with no graph integration.

**Methods**

- `set_connection(src: i32, dst: i32, src_p: i32, dst_p: i32)`
- `is_self_loop() -> bool`

### VoxelGraphPreview

Inherits: `Resource`

Graph preview configuration for the graph editor.

*Stub: not equivalent to the upstream C++ class.* Only validates the resolution; no preview rendering.

**Properties**

- `resolution: i32`

**Methods**

- `is_resolution_valid() -> bool`

### VoxelGraphNodesDocData

Inherits: `Resource`

Documentation data for graph nodes.

*Stub: not equivalent to the upstream C++ class.* A bare doc-entry counter.

**Methods**

- `add_doc() -> i32`
- `get_doc_count() -> i32`

### VoxelGraphEditorWindow

Inherits: `Resource`

The graph editor window state (open/dirty tracking).

*Stub: not equivalent to the upstream C++ class.* Only flag bookkeeping; no window or graph content.

**Methods**

- `open()`
- `close()`
- `get_is_open() -> bool`
- `get_is_dirty() -> bool`
- `mark_dirty()`
- `mark_saved()`

## Utilities/misc

### VoxelEngine

Inherits: `Object`

The voxel engine singleton; wraps a threaded task runner for background task processing.

**Properties**

- `thread_count: i32`

**Methods**

- `start()`
- `stop()`
- `process() -> i32` — drains completed tasks and returns the count drained
- `get_pending_count() -> i32`
- `wait_for_all()`

### VoxelTaskIndicator

Inherits: `RefCounted`

Indicates whether background tasks are pending.

*Stub: not equivalent to the upstream C++ class.* A bare pending-task counter with no task-runner integration.

**Properties**

- `task_count: i32`

**Methods**

- `is_busy() -> bool`
- `add_task()`
- `remove_task()`

### VoxelEditorCameraCache

Inherits: `RefCounted`

Caches the editor camera position so plugins can restore it.

*Stub: not equivalent to the upstream C++ class.* Stores a single position instead of a full camera transform.

**Methods**

- `store(x: f32, y: f32, z: f32)`
- `has_cached() -> bool`
- `get_x() -> f32`
- `get_y() -> f32`
- `get_z() -> f32`

### VoxelAboutWindow

Inherits: `Resource`

The "About" window resource, reporting the voxel-core version string for display.

*Stub: not equivalent to the upstream C++ class.* Only exposes the version string; no window UI.

**Methods**

- `get_version() -> GString`
