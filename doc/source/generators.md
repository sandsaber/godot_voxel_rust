# Generators

Generators are `Resource`s that produce voxel data for `VoxelTerrain`. Assign
one with `set_generator()` before the terrain enters the tree
(see [terrain setup](terrain_setup.md)):

```gdscript
var terrain := VoxelTerrain.new()
terrain.set_generator(VoxelGeneratorFlat.new())
add_child(terrain)
```

If nothing is assigned, `VoxelTerrain` falls back to a waves generator with
pattern size 128 and height range 60.

All six generators drive `VoxelTerrain` through the `generator` property.
Flat/Waves/Noise/Heightmap write smooth SDF terrain (pairs with the
transvoxel mesher); Image can also write blocky `Type` data (pairs with the
cubes mesher); Graph produces SDF from a node graph you build in GDScript.
See [Meshers](meshers.md) for choosing the mesher.

## VoxelGeneratorFlat

A horizontal plane at a fixed height.

| Property | Type | Default | Notes |
|---|---|---|---|
| `height` | int | `0` | Height of the flat surface, in voxels. |

## VoxelGeneratorWaves

Rolling wave terrain.

| Property | Type | Default | Notes |
|---|---|---|---|
| `amplitude` | float | `60.0` | Wave height (maps to the generator's height range). |
| `frequency` | float | `0.02` | Exported, but not currently consumed by the terrain pipeline. |
| `period` | float | `128.0` | Wave period; drives the pattern size. |

## VoxelGeneratorNoise

3D noise terrain (caves and overhangs) over a vertical slab.

| Property | Type | Default | Notes |
|---|---|---|---|
| `seed` | int | `0` | Noise seed. |
| `frequency` | float | `0.05` | Higher = more detail. |
| `height_start` | float | `-100.0` | Bottom of the noise slab (world Y). |
| `height_range` | float | `200.0` | Vertical extent of the slab. |

## VoxelGeneratorHeightmap

Rolling hills from 2D noise — the usual choice for outdoor terrain.

| Property | Type | Default | Notes |
|---|---|---|---|
| `seed` | int | `0` | Noise seed. |
| `frequency` | float | `0.02` | Higher = more detail. |
| `height_range` | float | `100.0` | Terrain amplitude in voxels. |

| Method | Notes |
|---|---|
| `sample_height(x, z)` | Terrain height at world `(x, z)`, in `[0, height_range]`. Deterministic for a fixed seed/frequency. |

```gdscript
var gen := VoxelGeneratorHeightmap.new()
gen.seed = 42
gen.frequency = 0.008
gen.height_range = 120.0
print(gen.sample_height(512.0, 512.0))   # ground height there
```

## VoxelGeneratorGraph

Builds terrain from a node graph: `InputX/Y/Z` provide the voxel coordinates,
math/SDF/noise nodes transform them, and `OutputSdf` writes the result into
the SDF channel. Graphs are built programmatically with `add_node()` and can
drive `VoxelTerrain` like any other generator (the graph is compiled once,
then evaluated per block on worker threads).

!!! note "Status: partially implemented"
    There is no visual graph editor yet (the editor plugin only hosts an
    empty panel). Build graphs with `add_node()`. `set_graph_json` parses a
    compact `{"nodes":[...]}` list (same fields as `add_node`). Assigning
    the resource to `VoxelTerrain.generator` uses `GraphGenerator` — it does
    not silently become Waves.

| Method | Notes |
|---|---|
| `clear_graph()` | Remove all nodes. |
| `add_node(kind, a, b, c, d, value)` | Append a node; returns its id (pass ids as ports of later nodes, `-1` = unconnected). Kinds: `InputX`, `InputY`, `InputZ`, `Constant` (uses `value`), `Add`/`Subtract`/`Multiply`/`Divide`/`Min`/`Max`, `Sin`/`Cos`/`Abs`/`Sqrt`/`Floor`/`Fract`, `SdfPlane`, `SdfSphere` (`a`=x, `b`=y, `c`=z, `d`=radius), `SdfBox` (`value` = cube half-extent), `SdfUnion`/`SdfSubtract`, `SdfSmoothUnion`/`SdfSmoothSubtract` (`value` = smoothness), `Noise2D`/`Noise3D`, `OutputSdf`. Returns `-1` for unknown kinds. |
| `add_expression_node(expr, x, y, z)` | Expression node; variables `x`/`y`/`z` bind to the given port ids. |
| `add_image2d_node(width, height, fill, x, y)` | Image2D node filled with `fill`, sampled at ports `x`/`y`. |
| `get_graph_node_count()` | Nodes in the graph under construction. |
| `compile_graph()` | `true` if the graph compiles (no cycles / dangling ports). |
| `get_graph_json()` / `set_graph_json(json)` | Compact interchange. `set_graph_json` parses `{"nodes":[{"kind":...,"a":...,"value":...}]}`. |
| `sample_sphere_sdf(cx, cy, cz, r, px, py, pz)` | Standalone helper: builds a sphere-SDF graph and samples it. Negative = inside, `NaN` on compile failure. |

```gdscript
# Smooth-union of a plane at y=0 and a sphere of radius 10 at the origin.
var graph := VoxelGeneratorGraph.new()
var iy := graph.add_node("InputY", -1, -1, -1, -1, 0.0)
var plane := graph.add_node("SdfPlane", iy, -1, -1, -1, 0.0)   # y - 0
var ix := graph.add_node("InputX", -1, -1, -1, -1, 0.0)
var iz := graph.add_node("InputZ", -1, -1, -1, -1, 0.0)
var ten := graph.add_node("Constant", -1, -1, -1, -1, 10.0)
var sphere := graph.add_node("SdfSphere", ix, iy, iz, ten, 0.0)
var uni := graph.add_node("SdfSmoothUnion", plane, sphere, -1, -1, 4.0)
graph.add_node("OutputSdf", uni, -1, -1, -1, 0.0)
assert(graph.compile_graph())

terrain.set_generator(graph)   # drives VoxelTerrain
```

## VoxelGeneratorImage

Heightmap terrain from an image: pixel luminance becomes terrain height,
`height = height_start + luminance * height_range`. With `channel = 1` (SDF)
it produces smooth terrain for the transvoxel mesher; with `channel = 0`
(Type) it fills blocky voxels for the [cubes mesher](meshers.md) — a
Minecraft-style world from any heightmap image.

| Property | Type | Default | Notes |
|---|---|---|---|
| `height_range` | float | `100.0` | Vertical extent; pixel values `0..1` scale by this. |
| `height_start` | float | `-50.0` | World Y that a black pixel maps to. |
| `channel` | int | `1` | Output channel: `0` = Type (blocky), `1` = SDF (smooth). |
| `repeat` | bool | `false` | Tile the image horizontally instead of clamping. |

| Method | Notes |
|---|---|
| `set_image(image)` | Load heights from a Godot `Image` (Rec. 709 luminance per pixel). Returns `false` for empty images. |
| `set_heights(data, width, height)` | Load heights from raw bytes (`0..255`), row-major. Returns `false` if `data.size() != width * height`. |
| `has_image()` | Whether an image/heightmap is loaded. |

## Base classes

`VoxelGenerator` and `VoxelStream`/`VoxelMesher` base resources exist for API
parity with the original C++ module (each exposes `get_category()`). In this
port the concrete generators are standalone `Resource` subclasses — they do
not inherit from `VoxelGenerator` in the class hierarchy.
