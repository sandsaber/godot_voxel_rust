# Status & roadmap

Honest, audited state of the C++ → Rust port. ✅ = works end-to-end,
🟡 = partially implemented (details noted), ⬜ = intentionally deferred or not
ported yet.

The upcoming big features (blocky library, Variable-LOD renderer parity,
multiplayer, editing tools, instancing rendering, graph editor, leftover CI)
are tracked individually in
[ROADMAP.md](https://github.com/sandsaber/godot_voxel_rust/blob/master/ROADMAP.md)
at the repository root — reference items as `R1`…`R8` in commits/PRs.

## Engine core (`voxel-core`)

| Area | Status | Notes |
|---|---|---|
| Math / containers / threading | ✅ | Direct ports, heavily tested. |
| Voxel storage (`VoxelBuffer`, channels, compression, memory pool) | ✅ | Typed pool recycling deferred (tracked). |
| Block serializer v4 | ✅ | Metadata section + v2/v3 legacy migration deferred (needs a Variant codec). |
| LZ4 / ZSTD compression | ✅ | LZ4 pure-Rust by default; ZSTD behind an optional feature (it bundles C). |
| Transvoxel mesher | ✅ | Regular + transition cells, texturing modes; verified against C++ goldens (bit-exact indices/masks, 1e-5 floats). |
| Cubes mesher | ✅ | Greedy + simple, palette; atlased mode deferred. |
| Blocky mesher | ✅ | Bake + AO + skirts + shadow occluders; inner-part AO is a TODO. |
| Simple generators (flat/waves/noise/heightmap/image) | ✅ | |
| Graph generator | 🟡 | AST interpreter with Expression/Image2D wired into the runtime; range analysis still limited. Visual editor not ported. |
| Region files (.vxr) | ✅ format | Forest/LRU/`meta.vxrm`/file conversion not ported. |
| Terrain paging (`VoxelTerrainCore`) | ✅ | Fixed-LOD and Variable-LOD clipbox planner, save-on-unload, viewer pairing. |
| Edition (sphere/box/hemisphere/smooth, DDA raycast) | ✅ core | Paste / metadata / random-tick still open. |
| Instancing | 🟡 | Scatter math only (MVP); no instance-block streaming. |
| Modifiers | 🟡 | Sphere modifier real; mesh modifier is a box stand-in; not integrated into streaming. |
| GPU compute / detail rendering / shaders | ⬜ | Deferred by design (keeps the core pure-Rust for Android/WASM). |
| SQLite streams | ⬜ | Deferred by design. |
| Rapier physics | ⬜ | Deferred by design. |

## Godot binding (`voxel-gdext`)

| Feature | Status | Notes |
|---|---|---|
| `VoxelTerrain` paging + rendering | ✅ | Viewer-driven, multi-LOD count, material override, trimesh collision. |
| Mesher selection (`mesher` property) | ✅ | Transvoxel, cubes; blocky wires through but needs a model library to produce geometry. |
| Generators usable by terrain | ✅ | Flat / Waves / Noise / Heightmap / Image / Graph. |
| Streams usable by terrain | ✅ | `VoxelStreamMemory`, `VoxelStreamRegionFiles` (region/sector size from inspector; `convert_files` still a no-op). |
| Editing terrain | 🟡 | Live `VoxelToolTerrain` on both nodes: sphere/box/hemisphere/smooth, batched per data block. Paste/metadata/random-tick still stubbed. |
| Raycast | ✅ | DDA voxel traversal over the SDF channel. |
| `VoxelLodTerrain` | 🟡 | Production Variable-LOD runtime (`new_variable_lod` + `try_process`), 3-LOD smoke scene. Collision layer/mask/margin and lifecycle signals are live. GPU/normalmaps and most debug draws remain stored stubs. |
| Instancer rendering | 🟡 | Scatter uploads MultiMeshes. As a terrain child it streams one instance block per paged LOD0 mesh block. Scene-item instantiation is still open. |
| Editor plugins | 🟡 | `.vox` parsing real. The Voxel Graph bottom panel is a working GraphEdit addon (`add_node` / compile / sample). Instancer plugin is still a stub host. |
| `VoxelBoxMover` / `VoxelAStarGrid3D` | 🟡 | Registered, but semantics differ from upstream (no physics-aware movement / no pathfinding engine yet). |
| `VoxelStreamSQLite`, `VoxelVoxLoader` | 🟡 | Placeholders (path/extension validation only). |

## Missing upstream features without a deferral decision

These upstream areas are absent and were not explicitly listed as deferred:

- Standalone `VoxelGeneratorImage2D`-style extras beyond the implemented
  image generator (blur, 16-bit support).
- Multiplayer / `VoxelAreaFinder` area sync.
- Block metadata round-trip (blocked on the Variant codec, same as the
  serializer metadata section).
- `VoxelStreamRegionFiles` channel depths, rotation, and `convert_files`.
- Real mesh→SDF baking for `VoxelMeshSDF`.

## Test & verification status

- Workspace tests are the source of truth (the inventory grew with the
  Variable-LOD planner; do not treat a copied count in this file as current).
  Coverage: core unit tests, the parity suite, integration/stress, transvoxel
  C++ goldens, TSan (plain concurrency off-Linux), binding unit +
  `port_status` manifest, Godot smoke.
- **C++-golden parity** covers the transvoxel regular mesher + tables: the
  goldens are produced by compiling the *actual upstream C++* in
  `cpp-baseline`, and the comparator enforces bit-exact structural data. The
  rest of the parity suite is Rust re-ports of upstream C++ unit tests,
  hand-computed values, and regression pins.
- Clippy/fmt enforced clean; release uses `panic = "unwind"` (so unit tests
  can `catch_unwind`) but FFI-facing code must still return errors instead of
  panicking — unwind across the Godot C ABI is undefined.
- Fuzz targets exist for `.vox` parsing and block/region payloads, with
  committed seed corpora (`rust/fuzz/seed_corpus/`).
- Godot smoke tests: a runnable 4.7 project verifying class registration and
  real terrain paging.
