# Editing voxels

!!! note "Status: partially implemented"
    - **Works:** single-voxel SDF edits on `VoxelTerrain`
      (`set_voxel_sdf` / `get_voxel_sdf`) with automatic re-meshing; the
      standalone `VoxelToolBuffer` (sphere/box/set/get on a buffer); and a
      live `VoxelToolTerrain` (obtained via `get_voxel_tool()`) with
      sphere/box/hemisphere/smooth/paste, per-voxel metadata, and tags-aware
      blocky random-tick — all batched per overlapping data block.
    - Modifier nodes (`VoxelModifierSphere`, `VoxelModifierMesh`) work on
      buffers but are **not** automatically applied by `VoxelTerrain`.

## Editing the live terrain

`VoxelTerrain` exposes direct SDF voxel access. SDF semantics: negative =
solid, positive = empty. Writes mark the containing block dirty, and the block
is re-meshed on the next process tick.

```gdscript
# Raise a solid bump around world position (100, 40, 100).
for x in range(-4, 5):
    for z in range(-4, 5):
        $Terrain.set_voxel_sdf(100 + x, 40, 100 + z, -2.0)

# Read back.
print($Terrain.get_voxel_sdf(100, 40, 100))   # ≈ -2.0
```

`set_voxel_sdf()` returns `true` when the write landed, `false` if the terrain
core is not initialised or the write failed. For picking/selection use
`raycast()` — see [physics & collision](physics_collision.md).

## VoxelToolBuffer

A self-contained editing tool over its **own** buffer — it does not edit a
live terrain. Default channel is the SDF channel (1); the internal buffer
starts at 16³.

| Method | Notes |
|---|---|
| `create_buffer(sx, sy, sz)` | (Re)create the internal buffer. |
| `set_channel(channel)` | Channel all operations target. Invalid values keep the current channel. |
| `do_sphere(cx, cy, cz, radius, mode, value)` | Sphere edit. `mode`: `0` = Add, `1` = Remove, `2` = Set. `value` is the voxel value used by Set (blocky-style channels). |
| `do_box(min_x, min_y, min_z, max_x, max_y, max_z, mode, value)` | Box edit, bounds inclusive. Same mode/value semantics. |
| `set_voxel(x, y, z, value)` / `get_voxel(x, y, z)` | Single-voxel access on the current channel. Out-of-range access is ignored / returns `0`. |

```gdscript
var tool := VoxelToolBuffer.new()
tool.create_buffer(32, 32, 32)
tool.set_channel(1)                              # SDF
tool.do_sphere(16.0, 16.0, 16.0, 8.0, 0, 0)      # add a blob
tool.do_sphere(16.0, 20.0, 16.0, 4.0, 1, 0)      # remove part of it
tool.do_box(0, 0, 0, 31, 2, 31, 2, 0)            # set a slab
print(tool.get_voxel(16, 16, 16))
```

Use it for precomputing volumes or tests. To apply a prepared buffer to a
live terrain, use `VoxelToolTerrain.do_paste` (below).

## VoxelToolTerrain

Obtained from a terrain via `get_voxel_tool()`; edits the live terrain with
the same semantics, batched into one storage transaction per overlapping data
block.

| Method | Notes |
|---|---|
| `do_sphere(center, radius, mode)` | Sphere edit. `mode`: `0` = Add, `1` = Remove. |
| `do_box(min, max, mode)` | Axis-aligned box edit (inclusive bounds). |
| `do_hemisphere(center, radius, height_ratio, mode)` | Hemisphere brush. |
| `do_smooth(center, radius, blur_radius)` | Box-blur smoothing pass. |
| `do_paste(position, buffer, channel_mask)` | Paste a `VoxelBuffer` (channels + per-voxel metadata) into the terrain. |
| `set_voxel(position, value)` | Single-voxel write on the tool's channel. |
| `set_voxel_metadata(position, value)` / `get_voxel_metadata(position)` | Per-voxel metadata (nil/int/float/String/PackedByteArray; nil clears). |
| `for_each_voxel_metadata_in_area(aabb, callback)` | Visit metadata in a `[min, max)` box (inverted boxes are empty). |
| `run_blocky_random_tick(aabb, count, callback, batch, tags_mask)` | Tick random-tickable blocky voxels matching `tags_mask`; the scan budget bounds candidates. |

## SDF modifier nodes

Modifiers blend a shape into the SDF channel of a `VoxelBuffer`. They are
`Node3D`s, but `VoxelTerrain` does not scan for them — apply them to buffers
manually.

**`VoxelModifierSphere`**:

| Member | Kind | Notes |
|---|---|---|
| `radius` | property | Sphere radius. Default `10.0`. |
| `operation` | property | `0` = add (union), `1` = subtract. |
| `smoothness` | property | Blend smoothing; `0` = hard edge. |
| `apply_to_buffer(buffer, origin_x, origin_y, origin_z)` | method | Blends the sphere (centered at the node's position) into the buffer's SDF channel. `origin_*` is the buffer's world-space origin. Returns the number of voxels changed, or `-1` if `buffer` is not a `VoxelBuffer`. |

**`VoxelModifierMesh`** — an oriented-box shape (a baked-mesh stand-in):

| Member | Kind | Notes |
|---|---|---|
| `operation` | property | `0` = add, `1` = subtract. |
| `extents` | property | Box half-extents in voxels. Default `4.0`. |
| `apply_to_buffer(buffer, origin_x, origin_y, origin_z)` | method | Same contract as the sphere modifier. |

```gdscript
var buffer := VoxelBuffer.new()
buffer.create(32, 16, 32)

var sphere := VoxelModifierSphere.new()
sphere.radius = 8.0
sphere.position = Vector3(16, 8, 16)
print(sphere.apply_to_buffer(buffer, 0.0, 0.0, 0.0))
sphere.free()
```
