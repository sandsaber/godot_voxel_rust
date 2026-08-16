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
| Graph generator | 🟡 | AST interpreter with ~30 node kinds + compiled fast path; `ExpressionNode`/`Image2D` nodes exist but are not wired into the runtime; range analysis limited. |
| Region files (.vxr) | ✅ format | Forest/LRU/`meta.vxrm`/file conversion not ported. |
| Terrain paging (`VoxelTerrainCore`) | ✅ | Fixed-LOD and Variable-LOD clipbox planner, save-on-unload, viewer pairing. |
| Edition (sphere/box/blur, DDA raycast) | ✅ core | Smooth/paste tool modes not ported. |
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
| Streams usable by terrain | ✅ | `VoxelStreamMemory`, `VoxelStreamRegionFiles` (region size hardcoded to 32, no settings surface yet). |
| Editing terrain | 🟡 | `set_voxel_sdf` / `get_voxel_sdf` on `VoxelTerrain`; `VoxelToolTerrain` is a stub (path holder only). |
| Raycast | ✅ | DDA voxel traversal over the SDF channel. |
| `VoxelLodTerrain` | 🟡 | Production Variable-LOD runtime (`new_variable_lod` + `try_process`), 3-LOD smoke scene. Pinned API still partial: transition-mask consumption, collision surfaces, GPU/normalmaps and most debug draws are stored stubs. |
| Instancer rendering | 🟡 | Scatter computes counts; no MultiMesh output yet. |
| Editor plugins | 🟡 | `.vox` parsing real; the other plugins are bottom-panel prototypes. The upstream graph editor is replaced by a small GDScript prototype in the smoke-test project. |
| `VoxelBoxMover` / `VoxelAStarGrid3D` | 🟡 | Registered, but semantics differ from upstream (no physics-aware movement / no pathfinding engine yet). |
| `VoxelStreamSQLite`, `VoxelVoxLoader` | 🟡 | Placeholders (path/extension validation only). |

## Missing upstream features without a deferral decision

These upstream areas are absent and were not explicitly listed as deferred:

- Standalone `VoxelGeneratorImage2D`-style extras beyond the implemented
  image generator (blur, 16-bit support).
- Multiplayer / `VoxelAreaFinder` area sync.
- Block metadata round-trip (blocked on the Variant codec, same as the
  serializer metadata section).
- `VoxelStreamRegionFiles` full configuration surface (sector size, channel
  depths, rotation, file conversion).
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
- Clippy/fmt enforced clean; `panic = "abort"` in release, so FFI-facing code
  is written to return errors instead of panicking.
- Fuzz targets exist for `.vox` parsing and block/region payloads, with
  committed seed corpora (`rust/fuzz/seed_corpus/`).
- Godot smoke tests: a runnable 4.7 project verifying class registration and
  real terrain paging.
