# `voxel-gdext` — Godot 4 GDExtension bindings

The thin binding layer: the only crate in the workspace that depends on the
[`godot`](https://godot-rust.github.io) crate (gdext) and exposes Rust symbols
to GDScript via `#[func]`/`#[base]`/`#[signal]`. All engine-agnostic logic lives
in [`voxel-core`](../voxel-core); this crate wraps it into Godot classes.

## Status

The Rust binding foundation is complete: 79 Godot classes are registered and
the `VoxelTerrain` paging/generation/meshing pipeline runs end-to-end on Godot
4.7+. Individual classes have different parity levels. In particular, blocky
library binding, standalone `VoxelLodTerrain` rendering, full terrain tools and
instancer rendering remain partial; see the repository
[status matrix](../../doc/source/status.md) and [roadmap](../../ROADMAP.md).

### Class names

Classes are exposed under **canonical names** matching the upstream godot_voxel
C++ module (`VoxelBuffer`, `VoxelMesherBlocky`, `VoxelTerrain`, …), via the
`#[class(rename = ...)]` attribute — the Rust structs keep a `GD` suffix
(`VoxelBufferGD`) internally, but ClassDB and GDScript see `VoxelBuffer`.

Exceptions (to avoid clashing with Godot builtins, matching upstream's `ZN_`
prefix): `ZN_FastNoiseLite`, `ZN_SpotNoise`, `ZN_Curve` (Godot already ships
`Curve`, `FastNoiseLite`).

## Build

```sh
cd rust
cargo build -p voxel-gdext              # debug .so/.dylib/.dll
cargo build -p voxel-gdext --release    # optimized
```

This is a `cdylib`; the artifact is `target/<profile>/libvoxel_gdext.so` (Linux),
`libvoxel_gdext.dylib` (macOS), or `voxel_gdext.dll` (Windows).

## Load in Godot 4.7

1. Copy `voxel_gdext.gdextension.in` → `voxel_gdext.gdextension` somewhere under
   `res://` (e.g. the crate dir). The template defaults to debug desktop
   artifacts (`cargo build -p voxel-gdext`) and release Android artifacts
   (`./scripts/android-build.sh`); switch the paths if you load a release desktop
   build.
2. (Re)open the Godot project — the editor scans for `.gdextension` files and
   loads the library on startup.
3. The classes are now available in GDScript:

```gdscript
var terrain = VoxelTerrain.new()
terrain.set_generator(VoxelGeneratorWaves.new())
add_child(terrain)
var viewer = VoxelViewer.new()
terrain.add_child(viewer)   # viewer must be a child of the terrain
```

### Verified

Tested headless against Godot 4.7.1.stable on Linux x86_64 and macOS arm64,
most recently on 2026-08-10. The `smoke_test/` Godot project ships three
runnable checks plus a driver script.

**Reproducing on a clean checkout** — the compiled library is a git-ignored build
artifact, so build it first. The driver does everything:

```sh
cd rust
./voxel-gdext/smoke_test/run_smoke_test.sh          # debug build + all 3 checks
./voxel-gdext/smoke_test/run_smoke_test.sh --release
```

It (1) builds `voxel-gdext`, (2) copies the host `.so`/`.dylib`/`.dll` next to
the `.gdextension`, (3) creates Godot's generated extension list, then runs:

- **`api_test.gd`** (`godot --headless --script api_test.gd`) — class
  registration, `VoxelTerrain` instantiate, `set_generator`, property round-trip,
  `VoxelBuffer` voxel read/write. The SDF edit is asserted **honestly**: before
  `_ready()` runs it reports the not-ready state (set=false, sdf=0.0) rather than
  a false positive.
- **`runtime_scene.tscn`** — builds a `VoxelTerrain` + `VoxelGeneratorWaves` +
  `VoxelViewer` in the tree and pumps real frames: paging generates **210 mesh
  blocks** by frame 10, and the live edition path is verified (`set_voxel_sdf`
  returns true, `get_voxel_sdf` reads back -1.0).
- **`smoke_test.tscn`** — loads a scene containing canonical `VoxelTerrain` and
  `VoxelViewer` nodes and verifies the basic runtime surface.

```
Initialize godot-rust (API v4.7.stable.official, runtime v4.7.1.stable.official, safeguards strict)
voxel-gdext: Scene stage initialized (voxel-core v0.1.0)
[runtime] PASS set_voxel_sdf/get_voxel_sdf (set=true sdf=-1.000000)
[runtime] frame 10 — mesh_block_count=210
[runtime] DONE after 40 frames — mesh_block_count=210, paging ran without crash
```

## Android

`voxel-gdext` cross-compiles for Android arm64 (device) and x86_64 (emulator)
with the NDK. The build script works around the rustc↔NDK LLVM skew (NDK r29's
`lld` can't parse rustc/LLVM-22 objects; the script forces rust's `lld` via
`-fuse-ld=lld`) and exports `CC`/`CXX` so the `godot` crate builds `godot-cpp`
for the same target.

```sh
cd rust
./scripts/android-build.sh                                  # aarch64 .so (device)
./scripts/android-build.sh --target x86_64-linux-android    # x86_64 .so (emulator)
./scripts/android-build.sh --strip                          # strip debug symbols
```

Verified with NDK r29 (14206865), `ANDROID_API=21`, Godot `api-4-7`: produces
`target/<triple>/release/libvoxel_gdext.so` exporting `gdext_rust_init`
(~3.2 MB unstripped). The `.gdextension.in` carries matching `android.arm64`
and `android.x86_64` entries.

Loading the `.so` inside a Godot Android export template still requires a
custom template compiled with `platform=android` (the stock template does not
load GDExtensions at runtime on device) — that packaging step is tracked as
the remaining Phase 2 mobile-half item (needs an SDK + device/emulator).
