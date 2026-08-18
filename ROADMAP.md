# Roadmap

Big features after the C++ → Rust migration. Reference items as `R1`…`R8`
in commits/PRs. Statuses: ⬜ not started · 🟡 in progress · ✅ done for the
agreed product slice. Detail and the parity matrix live in
[doc/source/status.md](doc/source/status.md).

GPU / SQLite / multipass / Rapier / v2/v3 region migration / the R3
network product (sockets, RPCs, edit deltas) are **not** next-PR work.
They need an explicit go-ahead.

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

Leftover (deliberate skip, revisit with a material-library slice): the
blocky mesher, surface buckets, and the ArrayMesh upload already carry a
per-material index end-to-end (`material_{index}` surface names), but no bake
assigns material_id > 0 today — everything renders under the single terrain
`material_override`, which matches upstream's own VoxelLodTerrain ("No
multi-material supported yet"). MAX_SURFACES = 2 truncates richer meshes
anyway; a real slice needs the upstream material indexer + library resource
surface. Richer inspector peers remain open with it.

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

## R3 — Multiplayer / areas ✅ (protocol slice)

The boundary, interest index, and a transport-agnostic snapshot protocol
are live in `voxel-core::terrain::replication`. The `VoxelTerrainReplicator`
node bridges byte frames to any Godot transport (scripts own sockets/RPCs
by design). See [multiplayer.md](doc/source/multiplayer.md) for the
decision record and frame contract.

- [x] Write the replication boundary: what is authoritative (edits vs whole
      blocks), how it meets `try_edit_*`, dirty flags, LOD, and stream save
- [x] Port `VoxelAreaFinder` (who cares about which box; what to load/send) —
      `voxel-core::terrain::area_finder` reimplements upstream's linear
      `get_viewers_in_area` query with a spatial hash, deterministic results,
      a hostile-box cell budget, and `box_subtraction` for entered/exited
      interest boxes
- [x] Transport-agnostic protocol + reference bridge: length-prefixed
      snapshot frames (kind + version + block + revision + v4 serializer
      payload), per-peer interest via `VoxelAreaFinder`, revision-ordered
      client install, `VoxelTerrainReplicator` node bridging frames to any
      Godot transport. Sockets, RPCs and reliability are game-owned by
      design.
- [ ] Network product: edit *deltas* (edits replicate as whole-block
      snapshots today), rejoin/reconciliation beyond revision drop, LOD>0,
      and actual socket/RPC wiring. Needs an explicit go-ahead.

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

## R5 — Instancing rendering ✅

- [x] MultiMesh upload from `scatter_from_buffer` / `scatter_test`
- [x] `InstanceBlock` map + stream with terrain mesh-block paging
- [x] Scene-item instancer: `VoxelInstancer.set_item_scene` switches an item
      to scene mode; every scattered instance then spawns a real `Node3D`
      (the scene root at the instance transform), streamed per LOD0 mesh
      block and freed on block exit — alongside the MultiMesh path, not
      replacing it. Leftover (deferred with the C++ richness gap):
      per-instance persistence, `VoxelInstanceComponent` bookkeeping,
      mesh-LOD tiers, async generation tasks.

## R6 — Graph editor parity ✅

- [x] `add_node` / `clear_graph` / `compile_graph` programmatic API
- [x] Assigning `VoxelGeneratorGraph` to terrain uses `GraphGenerator` (never silent Waves)
- [x] Compact `set_graph_json` parse for the documented node list
- [x] Wire `ExpressionNode` / `Image2D` into the graph runtime (`add_expression_node`, `add_image2d_node`)
- [x] Visual GraphEdit addon: apply/compile against `add_node` / `compile_and_sample`

Leftover: editor polish, not a missing compile path.

## R7 — Streams & metadata ✅

Forest format is locked on disk. `MetadataValue` persists through the v4
block serializer metadata section — narrow types (nil/int/float/string/
bytes) under app-specific tags; wide Godot Variants (Dictionary/Array/
vectors/colors/packed arrays) under the C++ tag 32 with the exact upstream
wire format. Engine-only Variant types (objects, callables, node paths,
transforms) are skipped non-fatally, matching upstream's
`allow_objects = false`.

- [x] `VoxelStreamRegionFiles` region/sector size wired into the stream
- [x] `convert_files` rewrites region/sector size on disk
- [x] `meta.vxrm` locks forest format (block/region/sector size + 8 channel
      depths)
- [x] **Narrow:** persist our `MetadataValue` (`nil`/`int`/`float`/`string`/
      `bytes`) in the v4 metadata section. Byte-compatible with C++ when the
      section is empty; `nil`/`int` round-trip both ways. Our worlds survive
      save/load, including `convert_files` rewrites and memory-stream paging.
- [x] **Wide codec (landed):** `voxel-core::streams::variant_wire` — a
      pure-Rust encoder/decoder for Godot's Variant binary wire format,
      integrated into the v4 metadata section under tag 32. C++
      custom-metadata entries with supported payloads (scalars, strings,
      vectors/rects/plane/quaternion/AABB/color, Dictionary, Array, packed
      arrays) now round-trip; engine-only payloads are skipped non-fatally.
      `DecodeLimits`-guarded (depth + count budgets).
- [x] **v2/v3 block-payload migration:** old-format block payloads (inside
      v3 region containers) load transparently — v2→v3 remaps the SDF
      channel from legacy unsigned snorm; v3→v4 converts raw Variant
      metadata to tagged VoxelMetadata entries. Both migrations run
      in-memory on deserialization. Leftover: region CONTAINERS older than
      v3 are still rejected (`RegionError::UnsupportedVersion`); only the
      block payloads inside v3 containers migrate.

## R8 — CI rework ✅

- [x] Automatic Rust CI on PRs (`verify`: fmt + test + clippy). Godot
      smoke and Android are `workflow_dispatch` until the extension load
      path is stable.
- [x] Scheduled TSan, bounded fuzz, and `cargo audit`
- [x] Delete leftover C++ scons workflows
- [x] Make `verify` a required status check on `master`

## R9 — Canonical API completion (machine-checked) 🟡

**The standing goal after the Wave 3 port.** The tracker is
[`rust/voxel-gdext/api/port_status.json`](rust/voxel-gdext/api/port_status.json):
raise every non-deferred upstream class from `partial` to `complete`, where
`complete` = the pinned upstream API surface is exposed **and** its runtime
behavior is proven by an executable behavioral test. Kickoff state (after PR
#5): 3/73 `complete`, 56 `partial`, 14 `deferred` (deferred stays deferred —
out of R9 without an explicit go-ahead).

**Process — every stage is one PR that must pass multi-role review before
merge** (the Wave 3 protocol): at least upstream-parity (pinned XML surface
vs binding), runtime correctness, test-quality/evidence, and a verification
runner (fmt/clippy/tests/smoke executed); findings of REQUEST CHANGES block
the merge until resolved. Roles rotate per stage.

Stages are themed cohorts; the exact class list is resolved from
`port_status.json` at stage kickoff (it is the source of truth, this list is
the plan):

- [ ] **Stage 1 — Generators & noise** (the data source of every scene):
      `VoxelGenerator` + Flat / Noise / Heightmap / Image / Graph,
      `VoxelGraphFunction`, `FastNoise2`, `ZN_FastNoiseLite`, `ZN_SpotNoise`
- [ ] **Stage 2 — Meshers & serialization**: `VoxelMesher`, the cubes /
      blocky mesher classes, `VoxelBlockSerializer`, `VoxelColorPalette`,
      `VoxelRaycastResult`
- [ ] **Stage 3 — Streams**: `VoxelStream` and stream implementations still
      `partial`
- [ ] **Stage 4 — Terrain nodes & engine**: `VoxelNode`, the terrain nodes,
      `VoxelEngine`, block/data callbacks
- [ ] **Stage 5 — Blocky library surface**: models, attributes, types,
      the library classes
- [ ] **Stage 6 — Tools**: remaining `VoxelTool*` variants
- [ ] **Stage 7 — Instancing**: `VoxelInstancer`, instance-library items,
      `VoxelMeshSDF`
- [ ] **Stage 8 — Aux**: `VoxelBoxMover`, `VoxelAStarGrid3D`,
      `VoxelVoxLoader`, `VoxelSaveCompletionTracker`, plugin hosts

Done for a stage = every cohort class is either `complete` in
`port_status.json` with cited tests, or `deferred` with a recorded reason.

## Deferred by design (no ETA)

GPU compute path / detail rendering / shaders, SQLite streams, multipass
generator, Rapier physics — intentionally out of scope to keep `voxel-core`
pure-Rust and cross-compilable. The R7-wide Variant *codec* and the R3
transport-agnostic *protocol* have landed; what still sits here is the
v2/v3 region migration and the R3 *network product* (sockets, RPCs,
reliability, edit deltas), until given a go-ahead.
