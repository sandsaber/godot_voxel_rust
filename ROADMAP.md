# Roadmap

Big features that remain after the C++ → Rust migration. Each item is
independently trackable — reference it in commits/PRs as `R1`, `R2`, …
Statuses: ⬜ not started · 🟡 in progress · ✅ done. Detailed rationale and
the full parity matrix live in [doc/source/status.md](doc/source/status.md).

## R1 — Blocky terrain end-to-end 🟡

Baked cube library + `VoxelMesherBlocky` now reach `VoxelTerrain`. Remaining
work is Godot model-resource peers (meshes/materials), not the paging path.

- [x] Expose `VoxelBlockyLibrary` baking through the binding (models →
      `BakedLibrary` + `bake_library`)
- [x] Let `VoxelMesherBlocky` carry a library resource into the terrain
      pipeline
- [x] Smoke test: type-channel generator + blocky mesher renders visible
      blocks
- [x] `add_model` keeps Godot resources; cube/mesh/empty/fluid bake into the
      library (mesh items currently bake as colored cubes)
- [x] Bake assigned Godot mesh triangles into the interior blocky surface
- [x] Side-cutout / ortho-rotation parity with upstream mesh bake

## R2 — VoxelLodTerrain paging & rendering 🟡

Production clipbox planner is live (`VoxelTerrainCore::new_variable_lod` +
`try_process`). The Godot node pages, meshes, splits/joins and uploads
`ArrayMesh`es; a 3-LOD smoke scene covers viewer movement. Remaining work is
renderer/API parity, not a missing runtime.

- [x] Wire planner decisions to stream load/save in the node
- [x] Render LOD blocks through the shared mesh-lifecycle path
- [x] Viewer-driven split/join in `_process`
- [x] Consume `RenderTopologyChanged` / per-block transition masks in Godot
- [x] Collision surfaces + inspector layer/mask/margin on both terrain nodes
- [x] `block_loaded` / `mesh_block_entered` signals on `VoxelLodTerrain`
- [x] Upstream octree debug-draw / visual parity (clipbox leaves +
      volume/viewer/edit wireframes on `VoxelLodTerrain`)

## R3 — Multiplayer / areas ⬜

- [ ] Port `VoxelAreaFinder` (area sync primitives)
- [ ] Define the replication boundary for voxel edits/block data

## R4 — Terrain editing tools 🟡

`VoxelTerrain.get_voxel_tool()` and `VoxelLodTerrain.get_voxel_tool()` return
a live `VoxelToolTerrain`. Sphere/box edits run as one storage transaction per
overlapping data block (not per voxel).

- [x] `VoxelToolTerrain` backed by `VoxelTerrainCore` edits (sphere/box)
- [x] Batch sphere/box path in `voxel-core` (`try_edit_sphere` / `try_edit_box`)
- [x] Tool bound to `VoxelLodTerrain`
- [x] Hemisphere brush (`do_hemisphere`) in core and `VoxelToolTerrain`
- [x] Smooth mode (`do_smooth` box-blur) in core and `VoxelToolTerrain`
- [x] Paste (`do_paste`) and blocky random-tick on `VoxelToolTerrain`
- [x] Per-voxel metadata store / `for_each_voxel_metadata_in_area` (in-memory;
      serializer Variant codec is still R7)

## R5 — Instancing rendering ⬜

- [x] MultiMesh upload from `scatter_from_buffer` / `scatter_test`
- [x] `InstanceBlock` map + stream with terrain mesh-block paging

## R6 — Graph editor parity 🟡

- [x] `add_node` / `clear_graph` / `compile_graph` programmatic API
- [x] Assigning `VoxelGeneratorGraph` to terrain uses `GraphGenerator` (never silent Waves)
- [x] Compact `set_graph_json` parse for the documented node list
- [x] Wire `ExpressionNode` / `Image2D` into the graph runtime (`add_expression_node`, `add_image2d_node`)
- [x] Visual GraphEdit addon: apply/compile against `add_node` / `compile_and_sample`

## R7 — Streams & metadata ⬜

- [ ] Block metadata section (needs a Variant codec; also unblocks v2/v3
      legacy migration)
- [x] `VoxelStreamRegionFiles` region/sector size wired into the stream
- [x] `convert_files` rewrites region/sector size on disk
- [ ] Channel depths and rotation metadata

## R8 — CI rework 🟡

Rust jobs exist (`rust.yml` on push/PR, scheduled TSan / fuzz / audit).
Leftover C++ scons workflows have been removed from the tree.

- [x] Automatic Rust CI on PRs (`verify`: fmt + test + clippy). Godot
      smoke and Android are `workflow_dispatch` until the extension load
      path is stable.
- [x] Scheduled TSan, bounded fuzz, and `cargo audit`
- [x] Delete leftover C++ scons workflows
- [x] Make `verify` a required status check on `master`

## Deferred by design (no ETA)

GPU compute path / detail rendering / shaders, SQLite streams, multipass
generator, Rapier physics — intentionally out of scope to keep `voxel-core`
pure-Rust and cross-compilable.
