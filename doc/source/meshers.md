# Meshers

Meshers turn voxel data into triangle meshes. Three mesher resources exist:

- **`VoxelMesherTransvoxel`** — smooth SDF terrain (transvoxel algorithm).
- **`VoxelMesherBlocky`** — Minecraft-style cube models with baked AO.
- **`VoxelMesherCubes`** — color-cube mesher with optional greedy merging.

They share a common base `VoxelMesher` (exposed for parity: property `padding`,
method `get_category()`).

Assign one to `VoxelTerrain` with `set_mesher()` **before `_ready`** to
choose how the terrain renders (transvoxel is the default when unset):

```gdscript
var terrain := VoxelTerrain.new()
terrain.set_mesher(VoxelMesherCubes.new())          # Minecraft-style
terrain.set_generator(image_gen_with_type_channel)  # must feed the right channel
add_child(terrain)
```

The mesher must match what the generator writes: transvoxel reads the SDF
channel, cubes and blocky read the type/color channels (e.g. a
`VoxelGeneratorImage` with `channel = 0`). The same resources also mesh a
`VoxelBuffer` standalone via the `build_vertex_count` / `build_triangle_count`
methods below.

!!! note "Blocky on terrain"
    `VoxelMesherBlocky` wires through to the terrain, but the binding cannot
    attach a baked model library yet — with an empty library it produces no
    geometry (same as upstream with an unconfigured library).

## VoxelMesherTransvoxel

| Property | Type | Default | Notes |
|---|---|---|---|
| `sdf_channel` | int | `1` | Channel read as SDF. Channel 1 matches the terrain's SDF channel. |

| Method | Notes |
|---|---|
| `build_vertex_count(buffer, lod_hint)` | Runs transvoxel extraction over a `VoxelBuffer` and returns the vertex count. `lod_hint` toggles transition cells on the +X/+Z seam faces. Returns `-1` if `buffer` is not a `VoxelBuffer`. |
| `build_triangle_count(buffer, lod_hint)` | Same build, returns the triangle count. |

```gdscript
var buffer := VoxelBuffer.new()
buffer.create(16, 16, 16)

# The SDF channel (1) starts as empty space. Stamp a sphere into it:
var sphere := VoxelModifierSphere.new()
sphere.radius = 6.0
sphere.position = Vector3(8, 8, 8)
sphere.apply_to_buffer(buffer, 0.0, 0.0, 0.0)
sphere.free()

var mesher := VoxelMesherTransvoxel.new()
print(mesher.build_vertex_count(buffer, false))
print(mesher.build_triangle_count(buffer, false))
```

## VoxelMesherBlocky

!!! note "Status: partially implemented"
    Configuration is stored and the mesher runs, but `build_vertex_count`
    builds against an **empty internal model library**, so it always returns 0
    vertices. There is no way to attach a `VoxelBlockyLibrary` to it yet.

| Property | Type | Default | Notes |
|---|---|---|---|
| `bake_occlusion` | bool | `true` | Whether ambient occlusion is baked. |
| `occlusion_darkness` | float | `0.8` | AO darkness factor (0..1). |
| `type_channel` | int | `0` | Channel read as voxel type ids. |

| Method | Notes |
|---|---|
| `is_baking_occlusion()` | Reports `bake_occlusion`. |
| `type_channel_index()` | Reports `type_channel`. |
| `build_vertex_count(buffer)` | Runs the blocky mesher (see status note). |

## VoxelMesherCubes

When assigned to a terrain, `greedy` and `color_channel` are applied. (The
standalone `build_vertex_count` helper still builds with defaults.)

| Property | Type | Default | Notes |
|---|---|---|---|
| `greedy` | bool | `true` | Greedy rectangle merging. |
| `color_channel` | int | `4` | Channel read as palette colors. |

| Method | Notes |
|---|---|
| `is_greedy()` | Reports `greedy`. |
| `color_channel_index()` | Reports `color_channel`. |
| `build_vertex_count(buffer)` | Runs the cubes mesher over a `VoxelBuffer`. |

## Supporting resources

**`VoxelColorPalette`** — 256-entry RGBA palette used by the cubes mesher:

| Method | Notes |
|---|---|
| `set_color(index, r, g, b, a)` | Set entry `index` (0-255), components 0-255. |
| `get_color(index)` | Returns `[r, g, b, a]`. |
| `clear()` | Reset all entries to transparent black. |

**`VoxelBlockyLibrary`** — library of baked blocky models:

| Method | Notes |
|---|---|
| `add_solid_model(r, g, b)` | Append a solid-color model, returns its index. |
| `add_model(model)` | Append a `VoxelBlockyModel` / `VoxelBlockyModelCube` and keep the resource. |
| `get_model(index)` | The resource previously passed to `add_model` / `add_solid_model`, or `null`. |
| `get_model_count()` | Number of models. |
| `is_empty()` | Whether the library has no models. |

**`VoxelBlockyTypeLibrary`** — library of blocky *types* (model + attributes):

| Method | Notes |
|---|---|
| `add_color_type(r, g, b, a)` | Append a solid-color type, returns its id. |
| `get_type_count()` | Number of registered types. |
| `has_type(id)` | Whether type `id` exists. |
