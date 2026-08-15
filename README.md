Voxel Tools for Godot
=========================

A voxel terrain engine for Godot Engine 4 whose engine-agnostic core has been
ported from C++ to Rust. The canonical Godot-facing API is still being ported.

This fork is a **pure Rust GDExtension** — no C++ module code remains. The
engine core (`voxel-core`) is engine-agnostic and fully unit-testable. The
Godot binding (`voxel-gdext`) provides a functional fixed-LOD runtime, a production Variable-LOD
`VoxelLodTerrain`, and a partial canonical API. It loads in Godot 4.7+.
See [Status](doc/source/status.md) and [Roadmap](ROADMAP.md).

> The checked-in verification commands cover workspace tests, Clippy,
> formatting, parity data and Godot smoke tests. Canonical API progress is
> tracked separately in
> [`rust/voxel-gdext/api/port_status.json`](rust/voxel-gdext/api/port_status.json).

![Blocky screenshot](doc/source/images/blocky_screenshot.webp)
![Smooth screenshot](doc/source/images/smooth_screenshot.webp)

Features
---------------------------

- Realtime 3D terrain editable in-game (overhangs, tunnels, creation/destruction)
- Polygon-based: voxels are transformed into chunked meshes via the Transvoxel algorithm
- Fixed-LOD Godot terrain paging, editing, remeshing and collision generation
- Variable-LOD `VoxelLodTerrain` paging, clipboxes, coverage and transition
  topology
- Voxel data streaming (memory, region files, generators)
- Minecraft-style blocky terrain with baked ambient occlusion
- Smooth terrain with level of detail (Transvoxel + SINGLE_S4 texturing)
- Procedural graph generator (24+ node types, expression nodes, image lookups)
- Core voxel instancing algorithms (the canonical Godot runtime is partial)
- **Pure Rust core** — cross-compiles without Godot or C dependencies

Building
---------------

This is a **native GDExtension** — you compile it to a `.so`/`.dylib`/`.dll`,
then point a `.gdextension` file at it. Quick start:

```bash
cd rust
cargo build -p voxel-gdext --release    # → rust/target/release/libvoxel_gdext.so
```

For the full integration walkthrough (every platform: Linux/Windows/macOS,
Android, iOS; the `.gdextension` setup; debug vs release; verifying it loads),
see **[Integration guide](rust/docs/INTEGRATION.md)**.

Testing
---------------

```bash
cd rust
cargo test --workspace
cargo clippy --workspace --all-targets      # warning-clean
cargo fmt --check                           # clean
```

Project structure
---------------

```
rust/
├── voxel-core/          # Engine-agnostic Rust core (all logic)
│   ├── src/             # Core modules and unit tests
│   └── tests/           # C++ golden parity + integration/stress suites
├── voxel-gdext/         # Godot GDExtension binding (canonical API partial)
│   ├── src/             # Godot classes and voxel-core adapters
│   └── smoke_test/      # Godot 4.7 project + VoxelGeneratorGraph addon
├── cpp-baseline/        # C++ parity harness (reference data generation)
├── tsan/                # ThreadSanitizer tests
└── fuzz/                # cargo-fuzz targets
```

Status
---------------

The original C++ runtime module has been removed and `voxel-core` is the Rust
source of truth. The fixed-LOD and Variable-LOD GDExtension runtimes are
verified in Godot 4.7.1; the complete canonical upstream API is still partial:

- Workspace tests, C++ golden parity suites and focused GDExtension tests cover
  the implemented paths; Clippy and formatting are enforced.
- Canonical API progress is tracked per class in
  [`rust/voxel-gdext/api/port_status.json`](rust/voxel-gdext/api/port_status.json).
- Full paging + generation + meshing runs end-to-end in the headless Godot
  smoke suite, including a 3-LOD Variable-LOD scene.

Remaining big features (blocky model library on terrain, multiplayer areas,
full terrain tools, instancing rendering, graph editor) are tracked in
**[ROADMAP.md](ROADMAP.md)**.

Class names follow upstream godot_voxel (`#[class(rename=…)]`); see
[`AGENTS.md`](AGENTS.md) for the naming scheme.

Documentation
---------------

- [AGENTS.md](AGENTS.md) — repo guide for AI agents and contributors (architecture,
  crate layout, build/test/smoke commands, conventions).
- [MkDocs site](doc/source/index.md) — setup, architecture, class reference.
- [Status](doc/source/status.md) / [Roadmap](ROADMAP.md)
- [Integration guide](rust/docs/INTEGRATION.md) — how to build the extension and
  load it in a Godot project (Linux/Windows/macOS/Android/iOS).
- [Rust gdext binding](rust/voxel-gdext/README.md) — build, load, and verify in Godot.
- [Original upstream docs](https://voxel-tools.readthedocs.io/en/latest/)

Credits
---------------

Originally developed by [Zylann](https://github.com/Zylann/godot_voxel).
Rust port by the community. See the supporter list in the original project.
