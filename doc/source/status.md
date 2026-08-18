# Status & roadmap

Honest, audited state of the C++ → Rust port. ✅ = works end-to-end,
🟡 = partially implemented (details noted), ⬜ = intentionally deferred or not
ported yet.

Track upcoming work as `R1`…`R8` in
[ROADMAP.md](https://github.com/sandsaber/godot_voxel_rust/blob/master/ROADMAP.md).
That file is the product queue; this page is the parity matrix.

## Engine core (`voxel-core`)

| Area | Status | Notes |
|---|---|---|
| Math / containers / threading | ✅ | Direct ports, heavily tested. |
| Voxel storage (`VoxelBuffer`, channels, compression, memory pool) | ✅ | In-memory block + per-voxel `MetadataValue` (int/float/string/bytes). Typed pool recycling deferred. |
| Block serializer v4 | ✅ | Voxel channels, `meta.vxrm` forest format, and the v4 metadata section persist: narrow `MetadataValue` (nil/int/float/string/bytes) + wide Godot Variants (Dictionary/Array/vectors/colors/packed arrays) via `streams::variant_wire` (R7 wide). Engine-only Variant types still skipped non-fatally. v2/v3 block payloads migrate in-memory on load (SDF remap + Variant conversion). |
| LZ4 / ZSTD compression | ✅ | LZ4 pure-Rust by default; ZSTD behind an optional feature (it bundles C). |
| Transvoxel mesher | ✅ | Regular + transition cells, texturing modes; verified against C++ goldens (bit-exact indices/masks, 1e-5 floats). Typed zero-copy SDF input (Wave 3 B1): monomorphized `build_regular_mesh` per channel depth, dyn-dispatch adapter only as fallback and for transition passes. |
| Cubes mesher | ✅ | Greedy + simple, palette; atlased mode deferred. Zero-copy typed-channel slice (`Cow<[u32]>` via `channel_typed_slice`) with per-voxel fallback (Wave 3 B5). |
| Blocky mesher | ✅ | Bake + AO + skirts + shadow occluders + mesh-face split + ortho-rotation + cutout sides. Inner-part AO is still a TODO. |
| Simple generators (flat/waves/noise/heightmap/image) | ✅ | |
| Graph generator | ✅ runtime | AST interpreter with Expression/Image2D. Range analysis still limited. |
| Region files (`.vxr` + `meta.vxrm`) | ✅ | Channel depths locked on first save. `convert_files` rewrites region/sector/block size. LRU eviction and cross-process file locking remain open. |
| Terrain paging (`VoxelTerrainCore`) | ✅ | Fixed-LOD and Variable-LOD clipbox planner, save-on-unload, viewer pairing, debug snapshot. |
| Multiplayer replication (R3) | 🟡 | Boundary + interest index + transport-agnostic snapshot protocol: `terrain::replication` (revision-ordered frames of edited LOD0 blocks, per-peer interest, `ReplicationServer`/`Client`) + `VoxelTerrainReplicator` reference bridge. No sockets — game-owned. Edit deltas, rejoin reconciliation and socket/RPC wiring remain open. |
| Edition (sphere/box/hemisphere/smooth/paste, DDA raycast, random-tick) | ✅ core | Tags-aware random-tick when a baked library is present. `MetadataValue` persists through save/load (narrow R7). |
| Instancing | ✅ core | Scatter math + MultiMesh + per-mesh-block streaming + scene items (`BlockInstanceData` drives MultiMesh *or* `PackedScene` roots). C++-only extras (per-instance persistence, component bookkeeping, async tasks) remain out. |
| Modifiers | 🟡 | Sphere modifier real; mesh modifier is a box stand-in; not integrated into streaming. |
| GPU compute / detail rendering / shaders | ⬜ | Deferred by design (keeps the core pure-Rust for Android/WASM). |
| SQLite streams | ⬜ | Deferred by design. |
| Rapier physics | ⬜ | Deferred by design. |

## Godot binding (`voxel-gdext`)

| Feature | Status | Notes |
|---|---|---|
| `VoxelTerrain` paging + rendering | ✅ | Viewer-driven, material override, trimesh collision, inspector layer/mask/margin. |
| Mesher selection (`mesher` property) | ✅ | Transvoxel, cubes, blocky (library required for visible geometry). |
| Generators usable by terrain | ✅ | Flat / Waves / Noise / Heightmap / Image / Graph (Graph is never silent Waves). |
| Streams usable by terrain | ✅ | `VoxelStreamMemory`, `VoxelStreamRegionFiles` (inspector region/sector/`block_size_po2`, live `convert_files`, `meta.vxrm`). |
| Editing terrain | ✅ tool | Live `VoxelToolTerrain` on both nodes: sphere/box/hemisphere/smooth/paste, batched per data block, tags-aware random-tick. Metadata persists through save/load — including C++ Dictionary/Array payloads via the R7 wide codec; Object payloads stay unreadable by design (upstream `allow_objects = false`). |
| Raycast | ✅ | DDA voxel traversal over the SDF channel. |
| `VoxelLodTerrain` | ✅ runtime | Production clipbox planner, 3-LOD smoke scene, collision settings, lifecycle signals, wireframe debug-draw. GPU/normalmap inspector fields remain stored stubs. |
| Instancer rendering | ✅ | Scatter uploads MultiMeshes; as a terrain child it streams one instance block per paged LOD0 mesh block. Scene items (`set_item_scene`) spawn real `Node3D`s per instance, streamed and freed with paging. Streaming extracts from the Type channel only (SDF-only terrain yields zero instances, warned once). |
| Editor plugins | 🟡 | `.vox` parsing real. Voxel Graph bottom panel is a working GraphEdit addon. Instancer plugin is still a stub host. |
| `VoxelBoxMover` / `VoxelAStarGrid3D` | 🟡 | Registered, but semantics differ from upstream (no physics-aware movement / no pathfinding engine yet). |
| `VoxelStreamSQLite`, `VoxelVoxLoader` | 🟡 | Placeholders (path/extension validation only). SQLite is deferred by design. |

## What is left (and how big it is)

Tracked in ROADMAP; this is the honest size, not a new queue.

| Item | Size | Notes |
|---|---|---|

| R3 network product — edit deltas, rejoin reconciliation, sockets/RPCs | Several stages | Transport-agnostic protocol + reference bridge done; the game owns the transport. |
| Graph editor polish, extra Image2D extras, `VoxelMeshSDF` bake | Small–medium each | Not blocking generate→mesh→page→save. |

Intentionally **not** next: GPU, SQLite, multipass, Rapier, v2/v3 region migration, the R3 network product (sockets/RPCs/edit deltas).

## Performance & scaling

Wave 3 perf work (originally on the `rust/pilot` branch) lives here after an
explicit port; the pilot branch is retired.

- **`block_task_bench`** measures the pipeline the terrain actually drives
  (`MeshBlockTask::run_meshing` = generator gap-fill + 3×3×3 gather + mesher
  build, shared `MeshArraysPool`): a 16³ SDF sphere block meshes in **~58 µs**
  (~70 Melem/s) single-threaded on macOS arm64 (central block resident,
  26 neighbours gap-filled). The MT group meshes round-robin over positions
  inset from the volume faces (every gather queues the full 3×3×3) with no
  residency — 27 neighbours generated per block, ~280 µs per block
  single-threaded — and asserts every task is real work (never dropped,
  non-empty output): throughput scales **2.0× / 4.0× / 4.9×** on 2/4/8 scoped
  threads (16 blocks per thread, spawn included) — near-linear through 4
  threads, sublinear at 8 (shared pool, allocator, memory bandwidth).
  Sibling benches: `mesh_block_bench` (all-resident data through the real
  `ThreadedTaskRunner`), `transvoxel_bench` (kernel only).
- **`transvoxel_bench`** (kernel only): sphere_16 ~25 µs (~162 Melem/s),
  sphere_32 ~147 µs, sphere_64 ~909 µs on the same machine — within noise of
  the pre-port values. Since the Wave 3 genericization of
  `build_regular_mesh`, this bench measures the monomorphized typed path too;
  no measurable kernel-level win or regression from it.
- Port decisions: **B1** typed SDF input kept (hybrid: typed fast path for
  the regular mesh, adapter fallback + transition meshes); **B3** TLS
  free-list dropped as superseded — master's `MeshArraysPool` recycles via
  `Drop` on both `BlockMeshOutput` and `MeshUploadSnapshot`, which a
  thread-local cannot do once buffers cross threads; **B4** scratch-hoist
  was already on master, its regression test ported; **B5** kept for the
  cubes path (blocky half superseded by master's all-depths
  `ChannelData` dispatch).

## Test & verification status

- Workspace tests are the source of truth (do not treat a copied count in
  this file as current). Coverage: core unit tests, the parity suite,
  integration/stress, transvoxel C++ goldens, TSan (plain concurrency
  off-Linux), binding unit + `port_status` manifest, Godot smoke.
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
- Godot smoke tests: a runnable 4.7 project with seven checks — class
  registration, runtime paging, scene loading, runtime correctness
  (remesh/unload/safety/persistence), 3-LOD `VoxelLodTerrain`, blocky
  terrain, and instancer streaming.
