# Instancing

The instancer scatters items (trees, rocks, grass) over voxel surfaces.

!!! warning "Status: MVP — scatter counts only"
    The scatter pipeline computes **how many instances would be placed and
    where**, but there is **no rendering yet**: no `MultiMesh` output, no
    scene instantiation, no attachment to live terrain blocks. Treat the
    returned counts as the functional surface for now.

## VoxelInstancer

A `Node3D` with its own internal item library and scatter configuration.

| Member | Kind | Notes |
|---|---|---|
| `density_multiplier` | property | Global multiplier applied to every item's density. Default `1.0`. |
| `add_item(name, density, min_scale, max_scale)` | method | Register a scatter item, returns its index. |
| `get_item_count()` | method | Number of registered items. |
| `set_seed(seed)` | method | Seed for the scatter random generator. |
| `scatter_from_buffer(buffer)` | method | Extracts surface points from a `VoxelBuffer`'s Type channel (channel 0) and runs every item's scatter generator over them. Returns the total instance count. |
| `scatter_test(count)` | method | Debug helper: scatters over `count` dummy points using item 0. Returns the instance count. |

Surface extraction reads the buffer's Type channel: a voxel is a surface point
when it is solid (`!= 0`) and the voxel directly below it is air (`== 0`).
The buffer must contain blocky-style type data — an SDF-only buffer produces
no points.

```gdscript
var instancer := VoxelInstancer.new()
add_child(instancer)
instancer.add_item("tree", 0.2, 0.8, 1.4)
instancer.set_seed(2026)

# Build a buffer with a floating solid slab: solid voxels with air below.
var buffer := VoxelBuffer.new()
buffer.create(32, 16, 32)
for x in 32:
    for z in 32:
        buffer.set_voxel(x, 5, z, 0, 1)   # channel 0 = Type

print(instancer.scatter_from_buffer(buffer))  # ~20% of 1024 surface points
print(instancer.scatter_test(100))
```

## VoxelInstanceLibrary

A standalone `Resource` library of scatter items.

!!! note "Not consumed by the instancer node yet"
    `VoxelInstancer` maintains its own internal library via `add_item()`.
    There is no method to attach a `VoxelInstanceLibrary` resource to the
    node — this resource is usable standalone only.

| Method | Notes |
|---|---|
| `add_item(name, density, min_scale, max_scale, snap_to_normal)` | Register an item, returns its index. |
| `get_item_count()` | Number of items. |
| `is_empty()` | Whether the library has no items. |

## VoxelInstanceLibraryItem

One scatter item definition.

| Property | Type | Default | Notes |
|---|---|---|---|
| `name` | string | `"Item"` | Item name (`get_item_name` / `set_item_name`). |
| `density` | float | `0.1` | Placement density. |
| `min_scale` | float | `0.8` | Minimum random scale. |
| `max_scale` | float | `1.2` | Maximum random scale. |
| `snap_to_normal` | bool | `true` | Align instances to the surface normal. |

| Method | Notes |
|---|---|
| `get_average_scale()` | Midpoint of the scale range. |
| `get_scale_range()` | `max_scale - min_scale`. |
| `is_disabled()` | `true` when density is `<= 0` (no instances produced). |

## Related stubs

!!! warning "Status: stub"
    These classes exist for API parity but carry no real behavior:

    - `VoxelInstanceLibraryMultiMeshItem` — property `mesh_instance_count`,
      method `has_instances()`. No MultiMesh creation.
    - `VoxelInstanceLibrarySceneItem` — method `has_scene()` (a scene path
      field, no instantiation).
    - `VoxelInstanceComponent` — `is_visible()` / `set_visible(v)` flags only.

The editor plugin `VoxelInstancerEditorPlugin` adds an empty "Voxel Instancer"
bottom panel and exposes `count_surface_points(buffer)`, which counts the same
surface points `scatter_from_buffer` uses.
