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
- [ ] Real `VoxelBlockyModel` mesh/material peers instead of solid-cube
      placeholders

## R2 — VoxelLodTerrain paging & rendering 🟡

Production clipbox planner is live (`VoxelTerrainCore::new_variable_lod` +
`try_process`). The Godot node pages, meshes, splits/joins and uploads
`ArrayMesh`es; a 3-LOD smoke scene covers viewer movement. Remaining work is
renderer/API parity, not a missing runtime.

- [x] Wire planner decisions to stream load/save in the node
- [x] Render LOD blocks through the shared mesh-lifecycle path
- [x] Viewer-driven split/join in `_process`
- [ ] Consume `RenderTopologyChanged` / per-block transition masks in Godot
- [ ] Collision surfaces on the Variable-LOD node
- [ ] Upstream octree debug-draw / visual parity

## R3 — Multiplayer / areas ⬜

- [ ] Port `VoxelAreaFinder` (area sync primitives)
- [ ] Define the replication boundary for voxel edits/block data

## R4 — Terrain editing tools 🟡

`VoxelTerrain.get_voxel_tool()` returns a live `VoxelToolTerrain` that can
sphere/box/set voxels through `try_edit_voxel`.

- [x] `VoxelToolTerrain` backed by `VoxelTerrainCore` edits (sphere/box)
- [ ] Smooth and paste modes
- [ ] Hemisphere / metadata / random-tick

## R5 — Instancing rendering ⬜

- [ ] `VoxelInstanceBlock` + per-block instance streaming
- [ ] MultiMesh output from scatter results (currently counts only)

## R6 — Graph editor parity ⬜

- [ ] Parse graph JSON back into nodes (`set_graph_json` round-trip)
- [ ] Wire `ExpressionNode` / `Image2D` into the graph runtime
- [ ] Visual editor (GDScript GraphEdit addon or native)

## R7 — Streams & metadata ⬜

- [ ] Block metadata section (needs a Variant codec; also unblocks v2/v3
      legacy migration)
- [ ] `VoxelStreamRegionFiles` settings surface (region/sector size, channel
      depths, rotation, file conversion)

## R8 — CI rework 🟡

Rust jobs exist (`rust.yml` on push/PR, scheduled TSan / fuzz / audit).
Leftover C++ scons workflows have been removed from the tree.

- [x] Automatic Rust CI on push/PR (fmt + test + clippy + smoke + Android)
- [x] Scheduled TSan, bounded fuzz, and `cargo audit`
- [x] Delete leftover C++ scons workflows
- [ ] Make `Rust` a required status check on `master`

## Deferred by design (no ETA)

GPU compute path / detail rendering / shaders, SQLite streams, multipass
generator, Rapier physics — intentionally out of scope to keep `voxel-core`
pure-Rust and cross-compilable.
