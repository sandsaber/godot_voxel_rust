# Roadmap

Big features after the C++ → Rust migration. Reference items as `R1`…`R8`
in commits/PRs. Statuses: ⬜ not started · 🟡 in progress · ✅ done for the
agreed product slice. Detail and the parity matrix live in
[doc/source/status.md](doc/source/status.md).

GPU / SQLite / multipass / Rapier / full Godot Variant persistence / a
multiplayer transport are **not** next-PR work. They need an explicit go-ahead.

## R1 — Blocky terrain end-to-end ✅

Baked cube/mesh library + `VoxelMesherBlocky` reach `VoxelTerrain`.

- [x] Expose `VoxelBlockyLibrary` baking through the binding (models →
      `BakedLibrary` + `bake_library`)
- [x] Let `VoxelMesherBlocky` carry a library resource into the terrain
      pipeline
- [x] Smoke test: type-channel generator + blocky mesher renders visible
      blocks
- [x] `add_model` keeps Godot resources; cube/mesh/empty/fluid bake into the
      library
- [x] Bake assigned Godot mesh triangles (interior + cube-face split)
- [x] Side-cutout / ortho-rotation on mesh bake (`side_vertex_tolerance`,
      24 ortho bases, `side_cutout_enabled` → `bake_library`)

Leftover (not blocking the slice): per-surface Godot materials beyond the
first baked surface, richer inspector peers.

## R2 — VoxelLodTerrain paging & rendering ✅

Production clipbox planner is live (`VoxelTerrainCore::new_variable_lod` +
`try_process`). The Godot node pages, meshes, splits/joins and uploads
`ArrayMesh`es. A 3-LOD smoke scene covers viewer movement.

- [x] Wire planner decisions to stream load/save in the node
- [x] Render LOD blocks through the shared mesh-lifecycle path
- [x] Viewer-driven split/join in `_process`
- [x] Consume `RenderTopologyChanged` / per-block transition masks in Godot
- [x] Collision surfaces + inspector layer/mask/margin on both terrain nodes
- [x] `block_loaded` / `mesh_block_entered` signals on `VoxelLodTerrain`
- [x] Debug-draw overlay (clipbox leaves, volume bounds, viewer clipboxes,
      edited/metadata boxes). There is no legacy octree; "octree nodes" are
      resident mesh-block leaves.

Leftover: GPU/normalmap inspector fields are stored stubs (deferred by
design with the GPU path).

## R3 — Multiplayer / areas 🟡

The design pass is done: [doc/source/multiplayer.md](doc/source/multiplayer.md)
is the replication-boundary decision record (server-authoritative; edits as
deltas ordered by `block_revision`, edited blocks as v4 serializer bytes;
LOD0 only; generator blocks never replicated; revisions valid per server
process, dropped on rejoin). The interest index is reimplemented as pure
Rust. What remains is the network product itself.

- [x] Write the replication boundary: what is authoritative (edits vs whole
      blocks), how it meets `try_edit_*`, dirty flags, LOD, and stream save
- [x] Port `VoxelAreaFinder` (who cares about which box; what to load/send) —
      `voxel-core::terrain::area_finder` reimplements upstream's linear
      `get_viewers_in_area` query with a spatial hash, deterministic results,
      a hostile-box cell budget, and `box_subtraction` for entered/exited
      interest boxes
- [ ] Only then implement a transport / interest protocol. Not started; needs
      an explicit go-ahead and a peer/RPC owner (gdext or game-side crate).

`VoxelAreaFinder` alone is a medium port. The boundary decision is the large
part. Do not start this "on the side" of serializer work.

## R4 — Terrain editing tools ✅

`VoxelTerrain.get_voxel_tool()` and `VoxelLodTerrain.get_voxel_tool()` return
a live `VoxelToolTerrain`. Sphere/box/hemisphere/smooth/paste run as one
storage transaction per overlapping data block.

- [x] `VoxelToolTerrain` backed by `VoxelTerrainCore` edits (sphere/box)
- [x] Batch sphere/box path in `voxel-core` (`try_edit_sphere` / `try_edit_box`)
- [x] Tool bound to `VoxelLodTerrain`
- [x] Hemisphere brush (`do_hemisphere`) in core and `VoxelToolTerrain`
- [x] Smooth mode (`do_smooth` box-blur) in core and `VoxelToolTerrain`
- [x] Paste (`do_paste`) and tags-aware blocky random-tick
- [x] In-memory per-voxel / block metadata + `for_each_voxel_metadata_in_area`

Persistence of that metadata is R7, not a hole in the tool.

## R5 — Instancing rendering 🟡

- [x] MultiMesh upload from `scatter_from_buffer` / `scatter_test`
- [x] `InstanceBlock` map + stream with terrain mesh-block paging
- [ ] Scene-item instancer (spawn real nodes per instance, not only MultiMesh)

## R6 — Graph editor parity ✅

- [x] `add_node` / `clear_graph` / `compile_graph` programmatic API
- [x] Assigning `VoxelGeneratorGraph` to terrain uses `GraphGenerator` (never silent Waves)
- [x] Compact `set_graph_json` parse for the documented node list
- [x] Wire `ExpressionNode` / `Image2D` into the graph runtime (`add_expression_node`, `add_image2d_node`)
- [x] Visual GraphEdit addon: apply/compile against `add_node` / `compile_and_sample`

Leftover: editor polish, not a missing compile path.

## R7 — Streams & metadata ✅ (narrow slice)

Forest format is locked on disk. `MetadataValue` now persists: the v4 block
serializer writes/reads the metadata section (block entry + sorted per-voxel
entries). `nil`/`int` entries are byte-identical to C++ `VoxelMetadata`
(TYPE_EMPTY/TYPE_U64); `float`/`string`/`bytes` use app-specific tags (≥40).
Foreign C++ custom/Variant entries are skipped without failing the voxel load,
matching upstream.

- [x] `VoxelStreamRegionFiles` region/sector size wired into the stream
- [x] `convert_files` rewrites region/sector size on disk
- [x] `meta.vxrm` locks forest format (block/region/sector size + 8 channel
      depths)
- [x] **Narrow:** persist our `MetadataValue` (`nil`/`int`/`float`/`string`/
      `bytes`) in the v4 metadata section. Byte-compatible with C++ when the
      section is empty; `nil`/`int` round-trip both ways. Our worlds survive
      save/load, including `convert_files` rewrites and memory-stream paging.
- [ ] **Wide (separate project, not the next commit):** full Godot Variant
      codec + custom-metadata factory. That is what unblocks reading arbitrary
      C++ metadata and v2/v3 region migration. It either pulls Variant into
      `voxel-core` or needs a thick gdext-only encoder. Do not start without
      an explicit decision.

## R8 — CI rework ✅

- [x] Automatic Rust CI on PRs (`verify`: fmt + test + clippy). Godot
      smoke and Android are `workflow_dispatch` until the extension load
      path is stable.
- [x] Scheduled TSan, bounded fuzz, and `cargo audit`
- [x] Delete leftover C++ scons workflows
- [x] Make `verify` a required status check on `master`

## Deferred by design (no ETA)

GPU compute path / detail rendering / shaders, SQLite streams, multipass
generator, Rapier physics — intentionally out of scope to keep `voxel-core`
pure-Rust and cross-compilable. Full Variant persistence (wide R7) and the
R3 transport sit next to these until given a go-ahead; the R3 design and
interest index themselves are done.
