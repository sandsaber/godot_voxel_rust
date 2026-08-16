# Streams

Streams persist voxel data. Assign a stream resource to the `stream` property
of `VoxelTerrain` (exported to the inspector) or via `set_stream()`:

```gdscript
var stream := VoxelStreamRegionFiles.new()
stream.directory = "user://voxel_data"
$Terrain.stream = stream
```

`VoxelTerrain` accepts exactly two stream types: **`VoxelStreamMemory`** and
**`VoxelStreamRegionFiles`**. Assigning anything else logs
`VoxelTerrain.stream must be VoxelStreamMemory or VoxelStreamRegionFiles`
and the terrain runs generator-only.

When a stream is assigned, blocks are loaded from the stream first; blocks the
stream does not have (e.g. never saved) fall back to the terrain's generator,
so streamed and generated terrain blend seamlessly. Without a stream, all
terrain comes from the generator and nothing is saved.

## VoxelStreamMemory

An in-memory store. Blocks live only for the lifetime of the resource — no
disk persistence. Useful for tests, and installed automatically inside
`VoxelTerrain` when `lod_count > 1` and no explicit stream is set.

| Method | Notes |
|---|---|
| `get_block_count()` | Number of blocks currently stored. |
| `clear()` | Drop all stored blocks. |

```gdscript
var mem := VoxelStreamMemory.new()
$Terrain.stream = mem
# ... later
print(mem.get_block_count())
```

## VoxelStreamRegionFiles

!!! note "Status: partially implemented"
    Reading and writing `.vxr` region files works. `region_size_po2` and
    `sector_size` are applied when the stream is assigned to terrain, and
    `convert_files` rewrites an existing directory under new region/sector/
    block sizes (into a sibling directory, then atomically swapped in).

Disk persistence using region files, one file per region of
`(1 << region_size_po2)` blocks per axis.

| Property | Type | Default | Notes |
|---|---|---|---|
| `directory` | string | `"user://voxel_data"` | Folder where region files are stored. Created on first save. Use `user://` for runtime saves — `res://` is read-only in exported games. |
| `region_size_po2` | int | `4` (16 blocks/axis) | Power-of-two region size in blocks. |
| `sector_size` | int | `512` | On-disk sector size in bytes. |
| `block_size_po2` | int | `4` (16³ voxels) | Declared block size; actual saved blocks still follow the buffer. |

Implementation details:

- Files are named `lod<N>/r.<x>.<y>.<z>.vxr` using the region coordinates.
- Blocks are saved **LZ4-compressed**.
- A missing file or missing block inside a file is treated as "no saved data"
  (the generator fills it in); corrupt files surface an error.

```gdscript
var stream := VoxelStreamRegionFiles.new()
stream.directory = "user://world1/voxels"

var terrain := VoxelTerrain.new()
terrain.stream = stream
var gen := VoxelGeneratorHeightmap.new()
gen.seed = 7
terrain.set_generator(gen)
add_child(terrain)
```

## VoxelStreamSQLite

!!! warning "Status: stub"
    Only the database path and an extension check exist. There is no schema,
    no reading, no writing, and `VoxelTerrain` does not accept this stream.

| Member | Kind | Notes |
|---|---|---|
| `database_path` | property | Default `"res://data/voxels.db"`. |
| `has_valid_extension()` | method | `true` if `database_path` ends with `.db`. |

## Related: MagicaVoxel `.vox` import

Not a stream, but the supported way to bring external voxel models in:

- `VoxelVoxLoader.supports_extension(ext)` — reports `.vox` support.
- `VoxImporterPlugin` (editor plugin) additionally exposes:
    - `parse_vox_bytes(bytes)` → `PackedFloat32Array` of
      `(x, y, z, r, g, b)` per voxel from the first model.
    - `parse_vox_to_mesh(bytes)` → `ArrayMesh` built from the voxel data.

```gdscript
var plugin := VoxImporterPlugin.new()
var file := FileAccess.open("res://models/house.vox", FileAccess.READ)
var mesh := plugin.parse_vox_to_mesh(file.get_buffer(file.get_length()))
```
