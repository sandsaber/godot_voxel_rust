Voxel Tools for Godot
=========================

A Rust-first voxel terrain engine for Godot Engine 4, ported from
[Zylann/godot_voxel](https://github.com/Zylann/godot_voxel).

The runtime is a **pure Rust GDExtension** — no C++ module code remains. The
engine core (`voxel-core`) is engine-agnostic and fully unit-testable. The thin
Godot binding (`voxel-gdext`) registers 79 classes under their canonical
upstream names. Feature completeness varies by subsystem; see
[Status](doc/source/status.md) and [Roadmap](ROADMAP.md).

> **Verified locally on 2026-08-10:** 1494 default-feature tests and 1496
> all-feature tests pass (0 failed), strict Clippy and rustfmt are clean, the
> Godot 4.7.1 smoke suite passes, and `voxel-core` cross-compiles to Android
> `aarch64`.

![Blocky screenshot](doc/source/images/blocky_screenshot.webp)
![Smooth screenshot](doc/source/images/smooth_screenshot.webp)

Working end-to-end in Godot
---------------------------

- Viewer-driven terrain paging, generation and mesh upload through `VoxelTerrain`
- Smooth SDF terrain with Transvoxel, transition cells and SINGLE_S4 texturing
- Cubes meshing, memory/region streams and trimesh collision generation
- Flat, waves, noise, heightmap, image and graph generators
- Direct SDF voxel editing and DDA raycasting
- Pure-Rust core cross-compilation to Android, iOS and macOS

Implemented in core, but not yet complete in the Godot product surface:

- Blocky meshing has baked models and ambient occlusion, but terrain cannot yet
  receive a baked block library.
- `VoxelTerrain` supports multiple LOD maps; the separate `VoxelLodTerrain`
  class is still an octree facade without its own paging/rendering.
- Instancing currently provides scatter math and counts, not MultiMesh output.
- Graph `ExpressionNode`/`Image2D` runtime wiring and the full visual editor are
  still pending.

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
cargo test --workspace                      # 1494 tests (0 failed)
cargo test --workspace --all-features       # 1496 tests (0 failed)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Project structure
---------------

```
rust/
├── voxel-core/          # Engine-agnostic Rust core (all logic)
│   ├── src/             # 800+ unit tests
│   └── tests/           # 674 parity tests + integration + transvoxel parity
├── voxel-gdext/         # Godot GDExtension binding (79 classes)
│   ├── src/             # #[func] methods delegating to voxel-core
│   └── smoke_test/      # Godot 4.7 project + VoxelGeneratorGraph addon
├── cpp-baseline/        # C++ parity harness (reference data generation)
├── tsan/                # ThreadSanitizer tests
└── fuzz/                # cargo-fuzz targets
```

Status
---------------

The runtime migration from C++ to Rust is complete: Rust is the source of truth
and the old C++ module has been removed. Full upstream feature parity is not
complete yet:

- **1494 tests pass** (800 unit + 674 parity + 5 integration + 5 transvoxel
  parity + 1 stress + 5 TSan + 3 gdext unit + 1 doc-test), clippy/fmt clean.
- **79 Godot classes** are registered under canonical upstream names
  (`VoxelBuffer`, `VoxelMesherBlocky`, `VoxelTerrain`, …); some are deliberately
  partial facades or placeholders documented in the status matrix.
- Full paging + generation + meshing pipeline runs end-to-end (verified headless:
  210 mesh blocks generated).

Remaining big features (blocky model library on terrain, `VoxelLodTerrain`
paging/rendering, multiplayer areas, full terrain tools, instancing
rendering, graph editor) are tracked in **[ROADMAP.md](ROADMAP.md)**.

Class names follow upstream godot_voxel (`#[class(rename=…)]`); see
[`AGENTS.md`](AGENTS.md) for the naming scheme.

Documentation
---------------

- [AGENTS.md](AGENTS.md) — repo guide for AI agents and contributors (architecture,
  crate layout, build/test/smoke commands, conventions).
- [Integration guide](rust/docs/INTEGRATION.md) — how to build the extension and
  load it in a Godot project (Linux/Windows/macOS/Android/iOS).
- [Rust gdext binding](rust/voxel-gdext/README.md) — build, load, and verify in Godot.
- [Original upstream docs](https://voxel-tools.readthedocs.io/en/latest/)

Credits
---------------

Originally developed by [Zylann](https://github.com/Zylann/godot_voxel).
Rust port by the community. See the supporter list in the original project.
