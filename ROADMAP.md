# Roadmap

Big features that remain after the C++ → Rust migration. Each item is
independently trackable — reference it in commits/PRs as `R1`, `R2`, …
Statuses: ⬜ not started · 🟡 in progress · ✅ done. Detailed rationale and
the full parity matrix live in [doc/source/status.md](doc/source/status.md).

## R1 — Blocky terrain end-to-end ⬜

Attach a baked model library to `VoxelMesherBlocky` so blocky terrain renders
on `VoxelTerrain`.

- [ ] Expose `VoxelBlockyLibrary` baking through the binding (models →
      `BakedLibrary` + `bake_library`)
- [ ] Let `VoxelMesherBlocky` carry a library resource into the terrain
      pipeline
- [ ] Smoke test: type-channel generator + blocky mesher renders visible
      blocks

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

## R4 — Terrain editing tools ⬜

Upstream `VoxelTool` surface on real terrain.

- [ ] `VoxelToolTerrain` backed by `VoxelTerrainCore` edits (sphere/box)
- [ ] Smooth and paste modes

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

Rust jobs exist (`rust.yml` on push/PR, scheduled TSan / fuzz / audit). The
old scons workflows are still in the tree but disabled in the GitHub UI.

- [x] Automatic Rust CI on push/PR (fmt + test + clippy + smoke + Android)
- [x] Scheduled TSan, bounded fuzz, and `cargo audit`
- [ ] Delete or archive the leftover C++ scons workflows
- [ ] Make `Rust` a required status check (the workflow file was left
      `disabled_manually` after the C++ removal; re-enable it in the Actions UI)

## Deferred by design (no ETA)

GPU compute path / detail rendering / shaders, SQLite streams, multipass
generator, Rapier physics — intentionally out of scope to keep `voxel-core`
pure-Rust and cross-compilable.
